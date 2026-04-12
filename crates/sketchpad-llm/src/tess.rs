//! TESS-2: Text Encoder with Simplex Diffusion
//!
//! A simplex diffusion language model where generation happens by diffusing in
//! the probability simplex over vocabulary rather than discrete token space.
//! Unlike LLaDA (which masks tokens), TESS-2 operates continuously: start from
//! a uniform distribution over vocabulary, apply a bidirectional transformer to
//! get predicted distributions, then take a diffusion step in simplex space toward
//! the target.
//!
//! # Architecture
//!
//! - Standard bidirectional transformer (same backbone as LLaDA — no causal mask)
//! - Pre-norm with RMSNorm
//! - SwiGLU FFN
//! - Grouped-query attention (GQA)
//! - RoPE positional embeddings
//!
//! # Simplex Diffusion Process
//!
//! Forward process (noising toward uniform):
//! ```text
//! q(x_t | x_0) = (1 - t) * x_0 + t * uniform
//! ```
//!
//! Reverse step (denoising toward predicted distribution):
//! ```text
//! x_{t-1} = (1 - s) * x_t + s * p_θ(x_0 | x_t)
//! where s = dt / (T - t + dt)
//! ```
//!
//! Soft embeddings at each step:
//! ```text
//! emb_t = E @ x_t   (weighted sum of token embeddings)
//! ```
//!
//! # Generation Algorithm
//!
//! 1. Embed prompt tokens as hard embeddings
//! 2. Initialize output positions as uniform over vocabulary
//! 3. For t = 1.0 down to 0.0 in `num_steps` steps:
//!    a. Compute soft embeddings for output positions
//!    b. Concatenate with prompt embeddings
//!    c. Forward pass → predicted logits for output positions
//!    d. Convert logits → probability distribution p_0 via softmax
//!    e. Interpolate: x_{t-1} = (1 - s) * x_t + s * p_0
//! 4. Argmax of final x_0 to get hard token IDs

use burn::nn::{Embedding, EmbeddingConfig};
use burn::prelude::*;

use sketchpad_core::glu::SwiGluFfn;
use sketchpad_core::rmsnorm::RmsNorm;
use sketchpad_core::rope::RotaryEmbedding;
use sketchpad_core::transformer::{MultiHeadAttention, MultiHeadAttentionConfig};

/// TESS-2 model configuration
#[derive(Debug, Clone)]
pub struct TessConfig {
    /// Vocabulary size
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
    /// Default number of diffusion steps (TESS uses more than LLaDA)
    pub num_diffusion_steps: usize,
    /// RMSNorm epsilon
    pub norm_eps: f64,
    /// RoPE base frequency
    pub rope_base: f32,
    /// Maximum sequence length (for RoPE precomputation)
    pub max_seq_len: usize,
}

impl TessConfig {
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
            num_diffusion_steps: 50,
            norm_eps: 1e-5,
            rope_base: 10000.0,
            max_seq_len: 512,
        }
    }

    /// Initialize the model and runtime
    pub fn init<B: Backend>(&self, device: &B::Device) -> (Tess<B>, TessRuntime<B>) {
        let layers: Vec<TessLayer<B>> = (0..self.num_layers)
            .map(|_| {
                let attention =
                    MultiHeadAttentionConfig::gqa(self.d_model, self.num_heads, self.num_kv_heads)
                        .init(device);

                let ffn =
                    sketchpad_core::glu::SwiGluFfnConfig::new(self.d_model, self.intermediate_size)
                        .init(device);

                TessLayer {
                    input_norm: RmsNorm::with_eps(self.d_model, self.norm_eps, device),
                    attention,
                    post_attention_norm: RmsNorm::with_eps(self.d_model, self.norm_eps, device),
                    ffn,
                }
            })
            .collect();

        let model = Tess {
            embed_tokens: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            layers,
            norm: RmsNorm::with_eps(self.d_model, self.norm_eps, device),
            lm_head: burn::nn::LinearConfig::new(self.d_model, self.vocab_size)
                .with_bias(false)
                .init(device),
        };

        let runtime = TessRuntime {
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

/// A single TESS-2 transformer layer
///
/// Pre-norm architecture: norm → attention → residual → norm → FFN → residual
/// Identical to LLaDaLayer — bidirectional, no causal mask.
#[derive(Module, Debug)]
pub struct TessLayer<B: Backend> {
    /// Pre-attention layer norm
    pub input_norm: RmsNorm<B>,
    /// Bidirectional self-attention (no causal mask)
    pub attention: MultiHeadAttention<B>,
    /// Pre-FFN layer norm
    pub post_attention_norm: RmsNorm<B>,
    /// SwiGLU feed-forward network
    pub ffn: SwiGluFfn<B>,
}

impl<B: Backend> TessLayer<B> {
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

/// TESS-2 bidirectional simplex diffusion language model
#[derive(Module, Debug)]
pub struct Tess<B: Backend> {
    /// Token embeddings
    pub embed_tokens: Embedding<B>,
    /// Transformer layers
    pub layers: Vec<TessLayer<B>>,
    /// Final layer norm
    pub norm: RmsNorm<B>,
    /// Language model head
    pub lm_head: burn::nn::Linear<B>,
}

/// Runtime state for TESS-2 (non-module state)
pub struct TessRuntime<B: Backend> {
    /// Rotary position embeddings
    pub rope: RotaryEmbedding<B>,
    /// Model configuration
    pub config: TessConfig,
}

impl<B: Backend> Tess<B> {
    /// Forward pass using soft (continuous) embeddings as input
    ///
    /// Takes pre-computed soft embeddings rather than discrete token IDs. This
    /// is the core of simplex diffusion: at each step, the input embeddings are
    /// weighted sums of all token embeddings according to current probability
    /// distributions.
    ///
    /// # Arguments
    ///
    /// * `soft_embeddings` - Soft input embeddings [batch, seq_len, d_model]
    /// * `runtime` - Model runtime containing RoPE and config
    ///
    /// # Returns
    ///
    /// Logits over vocabulary for every position [batch, seq_len, vocab_size]
    pub fn forward_soft(
        &self,
        soft_embeddings: Tensor<B, 3>,
        runtime: &TessRuntime<B>,
    ) -> Tensor<B, 3> {
        let mut hidden_states = soft_embeddings;

        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states, &runtime.rope);
        }

        hidden_states = self.norm.forward(hidden_states);
        self.lm_head.forward(hidden_states)
    }

    /// Forward pass using discrete token IDs as input
    ///
    /// Embeds tokens to hard embeddings then runs the bidirectional transformer.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Token IDs [batch, seq_len]
    /// * `runtime` - Model runtime containing RoPE and config
    ///
    /// # Returns
    ///
    /// Logits over vocabulary for every position [batch, seq_len, vocab_size]
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>, runtime: &TessRuntime<B>) -> Tensor<B, 3> {
        let embeddings = self.embed_tokens.forward(input_ids);
        self.forward_soft(embeddings, runtime)
    }

    /// Generate text using simplex diffusion
    ///
    /// Implements iterative denoising in the probability simplex: starts from
    /// uniform distributions over vocabulary and progressively refines them
    /// toward peaked (argmax-able) distributions via transformer predictions.
    ///
    /// # Arguments
    ///
    /// * `prompt_ids` - Conditioning prefix [batch, prompt_len]
    /// * `output_len` - Number of new tokens to generate
    /// * `num_steps` - Number of diffusion steps (higher = better quality, slower)
    /// * `runtime` - Model runtime containing RoPE and config
    /// * `sampler` - Sampling configuration (used for argmax / temperature at final step)
    ///
    /// # Returns
    ///
    /// Generated token IDs [batch, output_len]
    pub fn generate(
        &self,
        prompt_ids: Tensor<B, 2, Int>,
        output_len: usize,
        num_steps: usize,
        runtime: &TessRuntime<B>,
        _sampler: &crate::sampling::SamplerConfig,
    ) -> Tensor<B, 2, Int> {
        let [batch, _prompt_len] = prompt_ids.dims();
        let vocab_size = runtime.config.vocab_size;
        let device = prompt_ids.device();

        // Step 1: embed prompt as hard token embeddings [batch, prompt_len, d_model]
        let prompt_embeddings = self.embed_tokens.forward(prompt_ids);

        // Step 2: initialize output positions as uniform distribution over vocabulary
        // x_t shape: [batch, output_len, vocab_size], values = 1/vocab_size
        let uniform_val = 1.0_f32 / vocab_size as f32;
        let x_t_flat: Vec<f32> = vec![uniform_val; batch * output_len * vocab_size];
        let x_t_data = burn::tensor::TensorData::new(x_t_flat, [batch, output_len, vocab_size]);
        let mut x_t = Tensor::<B, 3>::from_data(x_t_data, &device);

        // Get the embedding matrix weight [vocab_size, d_model] to compute soft embeddings
        // soft_emb = x_t @ embed_weight  →  [batch, output_len, d_model]
        let embed_weight = self.embed_tokens.weight.val(); // [vocab_size, d_model]

        let dt = 1.0_f32 / num_steps as f32;

        // Step 3: iterative reverse diffusion  t: 1.0 → 0.0
        for step in 0..num_steps {
            // Current t value (from 1.0 down to dt)
            let t = 1.0_f32 - step as f32 * dt;

            // Compute soft output embeddings: [batch, output_len, d_model]
            // x_t is [batch, output_len, vocab_size], embed_weight is [vocab_size, d_model]
            // Result: batch matrix multiply → [batch, output_len, d_model]
            let soft_out_emb = x_t
                .clone()
                .matmul(embed_weight.clone().unsqueeze::<3>().expand([
                    batch,
                    vocab_size,
                    runtime.config.d_model,
                ]));

            // Concatenate prompt embeddings with soft output embeddings
            let full_embeddings = Tensor::cat(vec![prompt_embeddings.clone(), soft_out_emb], 1);

            // Forward pass → logits [batch, prompt_len + output_len, vocab_size]
            let logits = self.forward_soft(full_embeddings, runtime);

            // Extract logits for output positions only [batch, output_len, vocab_size]
            let [_, total_len, _] = logits.dims();
            let prompt_len = total_len - output_len;
            let output_logits = logits.slice([0..batch, prompt_len..total_len, 0..vocab_size]);

            // Convert logits to probability distribution via softmax
            let p_0 = burn::tensor::activation::softmax(output_logits, 2);

            // Compute interpolation weight s = dt / (t - dt + dt) = dt / t
            // Clamp to [0, 1] to handle numerical edge cases near t=0
            let s = if t > 1e-6 { (dt / t).min(1.0) } else { 1.0 };

            // Simplex interpolation step: x_{t-1} = (1 - s) * x_t + s * p_0
            x_t = x_t.mul_scalar(1.0_f32 - s) + p_0.mul_scalar(s);
        }

        // Step 4: convert final probability distributions to hard token IDs via argmax
        // x_t is now x_0: [batch, output_len, vocab_size]
        // argmax along vocab dimension → [batch, output_len, 1], then squeeze dim 2
        x_t.argmax(2).squeeze_dim::<2>(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_tess_forward_soft() {
        let device = Default::default();
        let config = TessConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        // Soft embeddings: [1, 4, d_model]
        let soft_emb = Tensor::<TestBackend, 3>::zeros([1, 4, config.d_model], &device);
        let logits = model.forward_soft(soft_emb, &runtime);

        assert_eq!(logits.dims(), [1, 4, 1000]);
    }

    #[test]
    fn test_tess_forward() {
        let device = Default::default();
        let config = TessConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        let input_ids = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device);
        let logits = model.forward(input_ids, &runtime);

        assert_eq!(logits.dims(), [1, 4, 1000]);
    }

    #[test]
    fn test_tess_generate() {
        let device = Default::default();
        let config = TessConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        let prompt = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device);
        let generated = model.generate(
            prompt,
            4, // output_len
            5, // num_steps
            &runtime,
            &crate::sampling::SamplerConfig::greedy(),
        );

        // Output should be [batch, output_len] — just the generated portion
        assert_eq!(generated.dims(), [1, 4]);
    }

    #[test]
    fn test_tess_generate_valid_token_ids() {
        let device = Default::default();
        let config = TessConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        let prompt = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device);
        let generated = model.generate(
            prompt,
            3,
            10,
            &runtime,
            &crate::sampling::SamplerConfig::greedy(),
        );

        // All generated token IDs should be in [0, vocab_size)
        let ids: Vec<i64> = generated.to_data().to_vec().unwrap();
        let vocab_size = config.vocab_size as i64;
        for &id in &ids {
            assert!(id >= 0 && id < vocab_size, "token id {id} out of range");
        }
    }

    #[test]
    fn test_tess_config_tiny() {
        let config = TessConfig::tiny();
        assert_eq!(config.num_diffusion_steps, 50);
        assert_eq!(config.vocab_size, 1000);
    }
}
