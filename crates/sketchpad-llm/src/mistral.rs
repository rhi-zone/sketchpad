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

use sketchpad_core::kv_cache::ModelKvCache;
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
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        runtime: &MistralRuntime<B>,
        mut cache: Option<&mut ModelKvCache<B>>,
    ) -> MistralOutput<B> {
        let [_batch, seq_len] = input_ids.dims();
        let device = input_ids.device();

        let start_pos = cache.as_ref().map(|c| c.seq_len()).unwrap_or(0);

        let mut hidden_states = self.embed_tokens.forward(input_ids);

        // Use sliding window mask if configured, otherwise causal
        let mask = if seq_len > 1 {
            Some(match runtime.config.sliding_window {
                Some(window) => sliding_window_mask::<B>(seq_len, window, &device),
                None => causal_mask::<B>(seq_len, &device),
            })
        } else {
            None
        };

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let _ = cache.as_mut().map(|c| c.layer(layer_idx));
            hidden_states =
                layer.forward(hidden_states, Some(&runtime.rope), start_pos, mask.clone());
        }

        hidden_states = self.norm.forward(hidden_states);
        let logits = self.lm_head.forward(hidden_states.clone());

        MistralOutput {
            logits,
            hidden_states,
        }
    }

    /// Generate text autoregressively
    pub fn generate(
        &self,
        input_ids: Tensor<B, 2, Int>,
        runtime: &MistralRuntime<B>,
        max_new_tokens: usize,
        temperature: f32,
    ) -> Tensor<B, 2, Int> {
        let [batch, _prompt_len] = input_ids.dims();
        let mut all_tokens = input_ids;

        for _ in 0..max_new_tokens {
            let output = self.forward(all_tokens.clone(), runtime, None);

            let seq_len = all_tokens.dims()[1];
            let last_logits = output.logits.slice([
                0..batch,
                (seq_len - 1)..seq_len,
                0..runtime.config.vocab_size,
            ]);
            let last_logits = last_logits.reshape([batch, runtime.config.vocab_size]);

            let scaled_logits = if (temperature - 1.0).abs() > 1e-6 {
                last_logits / temperature
            } else {
                last_logits
            };

            let next_token = scaled_logits.argmax(1).reshape([batch, 1]);
            all_tokens = Tensor::cat(vec![all_tokens, next_token], 1);
        }

        all_tokens
    }
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
        let generated = model.generate(prompt, &runtime, 3, 1.0);

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
