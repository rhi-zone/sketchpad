//! Mistral Model Implementation
//!
//! Mistral is a decoder-only transformer similar to LLaMA but with
//! sliding window attention for efficient long-context processing.
//!
//! # Architecture
//!
//! - Same as LLaMA: pre-norm, RoPE, SwiGLU, GQA
//! - Sliding window attention (4096 tokens for Mistral 7B)
//! - 32k context length support
//!
//! # Supported Variants
//!
//! - Mistral 7B v0.1, v0.2, v0.3
//! - Mistral Nemo 12B

use burn::nn::{Embedding, EmbeddingConfig};
use burn::prelude::*;

use sketchpad_core::kv_cache::{AttentionCache, ModelKvCache};
use sketchpad_core::rmsnorm::RmsNorm;
use sketchpad_core::rope::RotaryEmbedding;
use sketchpad_core::transformer::{
    TransformerBlock, TransformerBlockConfig, causal_mask, sliding_window_mask,
};

/// Mistral Model Configuration
#[derive(Debug, Clone)]
pub struct MistralConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Intermediate (FFN) dimension
    pub intermediate_size: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of KV heads (for GQA)
    pub num_kv_heads: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Sliding window size (None for full attention)
    pub sliding_window: Option<usize>,
    /// RMSNorm epsilon
    pub norm_eps: f64,
    /// RoPE base frequency
    pub rope_base: f32,
}

impl MistralConfig {
    /// Mistral 7B v0.1 configuration
    pub fn mistral_7b_v01() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8, // GQA
            max_seq_len: 32768,
            sliding_window: Some(4096),
            norm_eps: 1e-5,
            rope_base: 10000.0,
        }
    }

    /// Mistral 7B v0.3 configuration (larger vocab, no sliding window in some versions)
    pub fn mistral_7b_v03() -> Self {
        Self {
            vocab_size: 32768,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            max_seq_len: 32768,
            sliding_window: None, // v0.3 removed sliding window
            norm_eps: 1e-5,
            rope_base: 1000000.0, // Higher base for longer context
        }
    }

    /// Mistral Nemo 12B configuration
    pub fn mistral_nemo_12b() -> Self {
        Self {
            vocab_size: 131072, // Large vocab with tekken tokenizer
            hidden_size: 5120,
            intermediate_size: 14336,
            num_layers: 40,
            num_heads: 32,
            num_kv_heads: 8,
            max_seq_len: 128000,
            sliding_window: None,
            norm_eps: 1e-5,
            rope_base: 1000000.0,
        }
    }

    /// Creates a tiny model for testing
    pub fn tiny() -> Self {
        Self {
            vocab_size: 1000,
            hidden_size: 128,
            intermediate_size: 256,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            max_seq_len: 512,
            sliding_window: Some(64),
            norm_eps: 1e-5,
            rope_base: 10000.0,
        }
    }

    /// Initialize the model and runtime
    pub fn init<B: Backend>(&self, device: &B::Device) -> (Mistral<B>, MistralRuntime<B>) {
        let head_dim = self.hidden_size / self.num_heads;

        let layers: Vec<TransformerBlock<B>> = (0..self.num_layers)
            .map(|_| {
                TransformerBlockConfig::with_gqa(
                    self.hidden_size,
                    self.intermediate_size,
                    self.num_heads,
                    self.num_kv_heads,
                )
                .init(device)
            })
            .collect();

        let model = Mistral {
            embed_tokens: EmbeddingConfig::new(self.vocab_size, self.hidden_size).init(device),
            layers,
            norm: RmsNorm::with_eps(self.hidden_size, self.norm_eps, device),
            lm_head: burn::nn::LinearConfig::new(self.hidden_size, self.vocab_size)
                .with_bias(false)
                .init(device),
        };

        let runtime = MistralRuntime {
            rope: RotaryEmbedding::with_base(head_dim, self.max_seq_len, self.rope_base, device),
            config: self.clone(),
        };

        (model, runtime)
    }
}

/// Mistral Model
#[derive(Module, Debug)]
pub struct Mistral<B: Backend> {
    /// Token embeddings
    pub embed_tokens: Embedding<B>,
    /// Transformer layers
    pub layers: Vec<TransformerBlock<B>>,
    /// Final layer norm
    pub norm: RmsNorm<B>,
    /// Language model head
    pub lm_head: burn::nn::Linear<B>,
}

/// Runtime state for Mistral
pub struct MistralRuntime<B: Backend> {
    /// Rotary position embeddings
    pub rope: RotaryEmbedding<B>,
    /// Model configuration
    pub config: MistralConfig,
}

/// Output from the Mistral model
pub struct MistralOutput<B: Backend> {
    /// Logits over vocabulary: [batch, seq_len, vocab_size]
    pub logits: Tensor<B, 3>,
    /// Hidden states from final layer
    pub hidden_states: Tensor<B, 3>,
}

impl<B: Backend> Mistral<B> {
    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Input token IDs [batch, seq_len]
    /// * `runtime` - Model runtime containing RoPE and config
    /// * `cache` - Optional KV cache for incremental generation
    ///
    /// # Returns
    ///
    /// Model output containing logits and hidden states
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        runtime: &MistralRuntime<B>,
        cache: Option<&mut dyn AttentionCache<B>>,
    ) -> MistralOutput<B> {
        let [_batch, seq_len] = input_ids.dims();
        let device = input_ids.device();

        let mut hidden_states = self.embed_tokens.forward(input_ids);

        match cache {
            Some(cache) => {
                let start_pos = cache.seq_len();

                // For single-token decode, no mask needed — sliding window doesn't
                // apply when attending to all cached tokens one step at a time.
                // For multi-token prefill with cache, use a prefill causal mask.
                let mask = if seq_len > 1 {
                    let total_len = start_pos + seq_len;
                    Some(prefill_causal_mask::<B>(seq_len, total_len, &device))
                } else {
                    None
                };

                for (layer_idx, layer) in self.layers.iter().enumerate() {
                    hidden_states = layer.forward_cached(
                        hidden_states,
                        Some(&runtime.rope),
                        start_pos,
                        mask.clone(),
                        cache,
                        layer_idx,
                    );
                }
            }
            None => {
                // Use sliding window mask if configured, otherwise causal
                let mask = if seq_len > 1 {
                    Some(match runtime.config.sliding_window {
                        Some(window) => sliding_window_mask::<B>(seq_len, window, &device),
                        None => causal_mask::<B>(seq_len, &device),
                    })
                } else {
                    None
                };

                for layer in &self.layers {
                    hidden_states =
                        layer.forward(hidden_states, Some(&runtime.rope), 0, mask.clone());
                }
            }
        }

        hidden_states = self.norm.forward(hidden_states);
        let logits = self.lm_head.forward(hidden_states.clone());

        MistralOutput {
            logits,
            hidden_states,
        }
    }

    /// Generate text autoregressively with KV caching
    ///
    /// Uses incremental generation: the prompt is processed in one pass (prefill),
    /// then each subsequent token is generated by only computing the new token
    /// against cached K/V from previous steps.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Initial prompt token IDs [batch, prompt_len]
    /// * `runtime` - Model runtime containing RoPE and config
    /// * `max_new_tokens` - Maximum tokens to generate
    /// * `sampler` - Sampling configuration
    ///
    /// # Returns
    ///
    /// Generated token IDs including the prompt
    pub fn generate(
        &self,
        input_ids: Tensor<B, 2, Int>,
        runtime: &MistralRuntime<B>,
        max_new_tokens: usize,
        sampler: &crate::sampling::SamplerConfig,
    ) -> Tensor<B, 2, Int> {
        let [batch, _prompt_len] = input_ids.dims();
        let mut cache =
            ModelKvCache::<B>::new(runtime.config.num_layers, runtime.config.max_seq_len);

        // Track generated token IDs for repetition/DRY penalties
        let input_data: Vec<i64> = input_ids.to_data().to_vec().unwrap();
        let mut context_tokens: Vec<u32> = input_data.iter().map(|&id| id as u32).collect();

        // Prefill: process the entire prompt at once
        let output = self.forward(input_ids.clone(), runtime, Some(&mut cache));

        let seq_len = input_ids.dims()[1];
        let last_logits = output.logits.slice([
            0..batch,
            (seq_len - 1)..seq_len,
            0..runtime.config.vocab_size,
        ]);
        let last_logits = last_logits.reshape([batch, runtime.config.vocab_size]);
        let token_id = crate::sampling::sample_from_logits(last_logits, &context_tokens, sampler);
        context_tokens.push(token_id);

        let device = input_ids.device();
        let mut next_token = Tensor::<B, 2, Int>::from_ints([[token_id as i32]], &device);
        let mut all_tokens = Tensor::cat(vec![input_ids, next_token.clone()], 1);

        // Decode: generate one token at a time using the cache
        for _ in 1..max_new_tokens {
            let output = self.forward(next_token, runtime, Some(&mut cache));

            let last_logits = output
                .logits
                .slice([0..batch, 0..1, 0..runtime.config.vocab_size]);
            let last_logits = last_logits.reshape([batch, runtime.config.vocab_size]);
            let token_id =
                crate::sampling::sample_from_logits(last_logits, &context_tokens, sampler);
            context_tokens.push(token_id);

            next_token = Tensor::<B, 2, Int>::from_ints([[token_id as i32]], &device);
            all_tokens = Tensor::cat(vec![all_tokens, next_token.clone()], 1);
        }

        all_tokens
    }
}

/// Creates a causal mask for prefill with cached positions
///
/// When there are already `start_pos` cached tokens, new tokens at positions
/// `[start_pos, start_pos + new_len)` should be able to attend to all positions
/// `[0, start_pos + new_len)` but not to future positions within the new chunk.
///
/// Returns a mask of shape [new_len, total_len] where total_len = start_pos + new_len.
fn prefill_causal_mask<B: Backend>(
    new_len: usize,
    total_len: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let start_pos = total_len - new_len;
    let mut mask_data = vec![0.0f32; new_len * total_len];
    for i in 0..new_len {
        let global_pos = start_pos + i;
        for j in (global_pos + 1)..total_len {
            mask_data[i * total_len + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::<B, 1>::from_floats(mask_data.as_slice(), device).reshape([new_len, total_len])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_mistral_tiny_forward() {
        let device = Default::default();
        let config = MistralConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        let input_ids = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device);
        let output = model.forward(input_ids, &runtime, None);

        assert_eq!(output.logits.dims(), [1, 4, 1000]);
        assert_eq!(output.hidden_states.dims(), [1, 4, 128]);
    }

    #[test]
    fn test_mistral_generate() {
        let device = Default::default();
        let config = MistralConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        let prompt = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device);
        let generated = model.generate(
            prompt,
            &runtime,
            3,
            &crate::sampling::SamplerConfig::greedy(),
        );

        assert_eq!(generated.dims(), [1, 5]);
    }

    #[test]
    fn test_mistral_configs() {
        let _ = MistralConfig::mistral_7b_v01();
        let _ = MistralConfig::mistral_7b_v03();
        let _ = MistralConfig::mistral_nemo_12b();
    }

    #[test]
    fn test_mistral_sliding_window() {
        let device = Default::default();
        let mut config = MistralConfig::tiny();
        config.sliding_window = Some(2); // Very small window for testing
        let (model, runtime) = config.init::<TestBackend>(&device);

        // Test with sequence longer than window
        let input_ids = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4, 5, 6]], &device);
        let output = model.forward(input_ids, &runtime, None);

        assert_eq!(output.logits.dims(), [1, 6, 1000]);
    }

    #[test]
    fn test_mistral_no_sliding_window() {
        let device = Default::default();
        let mut config = MistralConfig::tiny();
        config.sliding_window = None; // Full attention
        let (model, runtime) = config.init::<TestBackend>(&device);

        let input_ids = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device);
        let output = model.forward(input_ids, &runtime, None);

        assert_eq!(output.logits.dims(), [1, 4, 1000]);
    }
}
