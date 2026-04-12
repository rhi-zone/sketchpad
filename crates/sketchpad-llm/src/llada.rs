//! LLaDA: Large Language Diffusion with Masking
//!
//! A bidirectional diffusion language model that generates text by iterative
//! denoising. Unlike autoregressive LLMs, LLaDA starts with all output positions
//! masked and progressively unmasks them over T diffusion steps.
//!
//! # Architecture
//!
//! - Standard bidirectional transformer (BERT-style) — **no causal mask**
//! - Pre-norm with RMSNorm
//! - SwiGLU FFN
//! - Grouped-query attention (GQA)
//! - RoPE positional embeddings
//! - Special [MASK] token
//!
//! # Generation Algorithm (masked diffusion)
//!
//! 1. Append `output_len` MASK tokens to the prompt
//! 2. For t = T down to 1:
//!    a. Forward pass (full bidirectional attention) → logits for all positions
//!    b. Sample a token for each masked position from predicted distribution
//!    c. Compute mask ratio r_t = t / T
//!    d. Re-mask the lowest-confidence (1 - r_t) fraction of newly-unmasked tokens
//! 3. Return the fully unmasked sequence

use burn::nn::{Embedding, EmbeddingConfig};
use burn::prelude::*;

use sketchpad_core::glu::SwiGluFfn;
use sketchpad_core::rmsnorm::RmsNorm;
use sketchpad_core::rope::RotaryEmbedding;
use sketchpad_core::transformer::{MultiHeadAttention, MultiHeadAttentionConfig};

/// LLaDA model configuration
#[derive(Debug, Clone)]
pub struct LLaDaConfig {
    /// Vocabulary size (must include MASK token)
    pub vocab_size: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Hidden dimension (d_model)
    pub d_model: usize,
    /// Number of query attention heads
    pub num_heads: usize,
    /// Number of KV heads (for GQA)
    pub num_kv_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// FFN intermediate size
    pub intermediate_size: usize,
    /// Special [MASK] token ID used during diffusion
    pub mask_token_id: u32,
    /// Default number of diffusion steps
    pub num_diffusion_steps: usize,
    /// RMSNorm epsilon
    pub norm_eps: f64,
    /// RoPE base frequency
    pub rope_base: f32,
    /// Maximum sequence length (for RoPE precomputation)
    pub max_seq_len: usize,
}

impl LLaDaConfig {
    /// Tiny config for testing
    pub fn tiny() -> Self {
        Self {
            vocab_size: 1000,
            num_layers: 2,
            d_model: 128,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 32,
            intermediate_size: 256,
            mask_token_id: 999,
            num_diffusion_steps: 10,
            norm_eps: 1e-5,
            rope_base: 10000.0,
            max_seq_len: 512,
        }
    }

    /// Initialize the model and runtime
    pub fn init<B: Backend>(&self, device: &B::Device) -> (LLaDa<B>, LLaDaRuntime<B>) {
        let layers: Vec<LLaDaLayer<B>> = (0..self.num_layers)
            .map(|_| {
                let attention =
                    MultiHeadAttentionConfig::gqa(self.d_model, self.num_heads, self.num_kv_heads)
                        .init(device);

                let ffn =
                    sketchpad_core::glu::SwiGluFfnConfig::new(self.d_model, self.intermediate_size)
                        .init(device);

                LLaDaLayer {
                    input_norm: RmsNorm::with_eps(self.d_model, self.norm_eps, device),
                    attention,
                    post_attention_norm: RmsNorm::with_eps(self.d_model, self.norm_eps, device),
                    ffn,
                }
            })
            .collect();

        let model = LLaDa {
            embed_tokens: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            layers,
            norm: RmsNorm::with_eps(self.d_model, self.norm_eps, device),
            lm_head: burn::nn::LinearConfig::new(self.d_model, self.vocab_size)
                .with_bias(false)
                .init(device),
        };

        let runtime = LLaDaRuntime {
            rope: RotaryEmbedding::with_base(
                self.head_dim,
                self.max_seq_len,
                self.rope_base,
                device,
            ),
            config: self.clone(),
        };

        (model, runtime)
    }
}

/// A single LLaDA transformer layer
///
/// Pre-norm architecture: norm → attention → residual → norm → FFN → residual
#[derive(Module, Debug)]
pub struct LLaDaLayer<B: Backend> {
    /// Pre-attention layer norm
    pub input_norm: RmsNorm<B>,
    /// Bidirectional self-attention (no causal mask)
    pub attention: MultiHeadAttention<B>,
    /// Pre-FFN layer norm
    pub post_attention_norm: RmsNorm<B>,
    /// SwiGLU feed-forward network
    pub ffn: SwiGluFfn<B>,
}

impl<B: Backend> LLaDaLayer<B> {
    /// Forward pass through a single layer
    ///
    /// # Arguments
    ///
    /// * `x` - Hidden states [batch, seq_len, d_model]
    /// * `rope` - Rotary position embeddings
    ///
    /// # Returns
    ///
    /// Updated hidden states [batch, seq_len, d_model]
    pub fn forward(&self, x: Tensor<B, 3>, rope: &RotaryEmbedding<B>) -> Tensor<B, 3> {
        // Attention sub-layer (bidirectional: mask = None)
        let residual = x.clone();
        let normed = self.input_norm.forward(x);
        let attn_out = self.attention.forward(normed, Some(rope), 0, None);
        let x = residual + attn_out;

        // FFN sub-layer
        let residual = x.clone();
        let normed = self.post_attention_norm.forward(x);
        let ffn_out = self.ffn.forward(normed);
        residual + ffn_out
    }
}

/// LLaDA bidirectional diffusion language model
#[derive(Module, Debug)]
pub struct LLaDa<B: Backend> {
    /// Token embeddings
    pub embed_tokens: Embedding<B>,
    /// Transformer layers
    pub layers: Vec<LLaDaLayer<B>>,
    /// Final layer norm
    pub norm: RmsNorm<B>,
    /// Language model head
    pub lm_head: burn::nn::Linear<B>,
}

/// Runtime state for LLaDA (non-module state)
pub struct LLaDaRuntime<B: Backend> {
    /// Rotary position embeddings
    pub rope: RotaryEmbedding<B>,
    /// Model configuration
    pub config: LLaDaConfig,
}

/// Output from the LLaDA forward pass
pub struct LLaDaOutput<B: Backend> {
    /// Logits over vocabulary: [batch, seq_len, vocab_size]
    pub logits: Tensor<B, 3>,
    /// Hidden states from the final layer: [batch, seq_len, d_model]
    pub hidden_states: Tensor<B, 3>,
}

impl<B: Backend> LLaDa<B> {
    /// Forward pass through the full model
    ///
    /// Unlike autoregressive models, no causal mask is applied — all tokens can
    /// attend to all other tokens, enabling bidirectional context.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Token IDs [batch, seq_len] (may include MASK tokens)
    /// * `runtime` - Model runtime containing RoPE and config
    ///
    /// # Returns
    ///
    /// Logits over the full vocabulary for every position
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        runtime: &LLaDaRuntime<B>,
    ) -> LLaDaOutput<B> {
        let mut hidden_states = self.embed_tokens.forward(input_ids);

        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states, &runtime.rope);
        }

        hidden_states = self.norm.forward(hidden_states);
        let logits = self.lm_head.forward(hidden_states.clone());

        LLaDaOutput {
            logits,
            hidden_states,
        }
    }

    /// Generate text using the masked diffusion process
    ///
    /// Implements iterative unmasking: at each step, samples tokens for all
    /// masked positions, then re-masks the lowest-confidence fraction to
    /// allow correction in subsequent steps.
    ///
    /// # Arguments
    ///
    /// * `prompt_ids` - Conditioning prefix [batch, prompt_len]
    /// * `output_len` - Number of new tokens to generate
    /// * `num_steps` - Number of diffusion steps (higher = better quality, slower)
    /// * `sampler` - Sampling configuration (temperature, top-k, etc.)
    ///
    /// # Returns
    ///
    /// Full sequence [batch, prompt_len + output_len] with MASK tokens replaced
    pub fn generate(
        &self,
        prompt_ids: Tensor<B, 2, Int>,
        output_len: usize,
        num_steps: usize,
        runtime: &LLaDaRuntime<B>,
        sampler: &crate::sampling::SamplerConfig,
    ) -> Tensor<B, 2, Int> {
        let [batch, prompt_len] = prompt_ids.dims();
        let device = prompt_ids.device();
        let mask_id = runtime.config.mask_token_id as i32;

        // Step 1: construct initial sequence — prompt + output_len MASK tokens
        let mask_fill: Vec<i32> = vec![mask_id; batch * output_len];
        let mask_data = burn::tensor::TensorData::new(mask_fill, [batch, output_len]);
        let mask_tokens = Tensor::<B, 2, Int>::from_data(mask_data, &device);

        let mut sequence = Tensor::cat(vec![prompt_ids, mask_tokens], 1);
        let total_len = prompt_len + output_len;

        // Track which output positions are still masked — starts all masked
        let mut is_masked: Vec<Vec<bool>> = vec![vec![true; output_len]; batch];

        // Step 2: iterative denoising loop
        for step in (1..=num_steps).rev() {
            // Forward pass — full bidirectional attention
            let output = self.forward(sequence.clone(), runtime);

            // Extract logits over the output portion only: [batch, output_len, vocab_size]
            let vocab_size = runtime.config.vocab_size;
            let output_logits =
                output
                    .logits
                    .slice([0..batch, prompt_len..total_len, 0..vocab_size]);

            // Convert logits to host for per-token sampling
            let logits_data: Vec<f32> = output_logits
                .reshape([batch * output_len * vocab_size])
                .to_data()
                .to_vec()
                .unwrap();

            // Retrieve current sequence ids to update
            let seq_data: Vec<i64> = sequence.clone().to_data().to_vec().unwrap();

            // Mask ratio for this step: fraction that remains masked
            let mask_ratio = step as f32 / num_steps as f32;

            // For each batch and each masked output position: sample a token,
            // then decide which sampled tokens to re-mask based on confidence.
            let mut new_seq_data: Vec<i32> = seq_data.iter().map(|&v| v as i32).collect();

            for (b, batch_mask) in is_masked.iter_mut().enumerate() {
                // Gather currently masked positions for this batch item
                let masked_positions: Vec<usize> =
                    (0..output_len).filter(|&i| batch_mask[i]).collect();

                if masked_positions.is_empty() {
                    continue;
                }

                // Build context tokens for sampler (full sequence so far)
                let ctx_start = b * total_len;
                let context_tokens: Vec<u32> = seq_data[ctx_start..ctx_start + total_len]
                    .iter()
                    .map(|&v| v as u32)
                    .collect();

                // Sample a candidate token for each masked position
                let mut candidates: Vec<(usize, u32, f32)> = Vec::new(); // (pos, token, confidence)
                for &pos in &masked_positions {
                    let logit_start = (b * output_len + pos) * vocab_size;
                    let logits_slice = &logits_data[logit_start..logit_start + vocab_size];

                    let token_id =
                        crate::sampling::sample_token(logits_slice, &context_tokens, sampler);

                    // Confidence = max softmax probability (greedy confidence)
                    let max_logit = logits_slice
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let exp_sum: f32 = logits_slice.iter().map(|&l| (l - max_logit).exp()).sum();
                    let confidence = (logits_slice[token_id as usize] - max_logit).exp() / exp_sum;

                    candidates.push((pos, token_id, confidence));
                }

                // How many to keep unmasked this step: ceil((1 - mask_ratio) * masked_count)
                let keep_count = {
                    let n = masked_positions.len() as f32;
                    let k = ((1.0 - mask_ratio) * n).ceil() as usize;
                    k.max(1).min(masked_positions.len())
                };

                // Sort by confidence descending — keep the most confident
                candidates.sort_unstable_by(|a, b| {
                    b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)
                });

                for (rank, &(pos, token_id, _)) in candidates.iter().enumerate() {
                    let seq_idx = b * total_len + prompt_len + pos;
                    if rank < keep_count {
                        // Unmask: write the sampled token
                        new_seq_data[seq_idx] = token_id as i32;
                        batch_mask[pos] = false;
                    }
                    // Tokens beyond keep_count stay masked — they were already masked,
                    // so new_seq_data already holds mask_id for those positions.
                }
            }

            // Rebuild sequence tensor from updated data
            let new_data = burn::tensor::TensorData::new(new_seq_data, [batch, total_len]);
            sequence = Tensor::<B, 2, Int>::from_data(new_data, &device);
        }

        sequence
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_llada_forward() {
        let device = Default::default();
        let config = LLaDaConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        // Input with a MASK token in the middle
        let input_ids = Tensor::<TestBackend, 2, Int>::from_ints([[1, 999, 3, 999]], &device);
        let output = model.forward(input_ids, &runtime);

        assert_eq!(output.logits.dims(), [1, 4, 1000]);
        assert_eq!(output.hidden_states.dims(), [1, 4, 128]);
    }

    #[test]
    fn test_llada_generate() {
        let device = Default::default();
        let config = LLaDaConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        let prompt = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device);
        let generated = model.generate(
            prompt,
            4, // output_len
            5, // num_steps
            &runtime,
            &crate::sampling::SamplerConfig::greedy(),
        );

        assert_eq!(generated.dims(), [1, 6]); // prompt (2) + output (4)
    }

    #[test]
    fn test_llada_generate_no_mask_in_output() {
        let device = Default::default();
        let config = LLaDaConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        let prompt = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device);
        let generated = model.generate(
            prompt,
            3,
            10,
            &runtime,
            &crate::sampling::SamplerConfig::greedy(),
        );

        // After full diffusion, no output tokens should remain as MASK
        let ids: Vec<i64> = generated.to_data().to_vec().unwrap();
        let mask_id = config.mask_token_id as i64;
        let output_tokens = &ids[2..]; // skip prompt
        for &id in output_tokens {
            assert_ne!(
                id, mask_id,
                "output token {id} should not be MASK after full diffusion"
            );
        }
    }

    #[test]
    fn test_llada_config_tiny() {
        let config = LLaDaConfig::tiny();
        assert_eq!(config.num_diffusion_steps, 10);
        assert_eq!(config.mask_token_id, 999);
    }
}
