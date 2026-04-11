//! Mixtral Model Implementation
//!
//! Mixtral is a Mixture of Experts model based on the LLaMA architecture.
//! It replaces the dense FFN with a sparse MoE layer, where each token
//! is routed to a subset of expert FFNs.
//!
//! # Architecture
//!
//! - Same as LLaMA: pre-norm, RoPE, GQA
//! - FFN replaced with Sparse MoE (8 experts, top-2 routing)
//! - Each expert is a SwiGLU FFN
//!
//! # Supported Variants
//!
//! - Mixtral 8x7B (8 experts, ~12B active params)
//! - Mixtral 8x22B (8 experts, ~39B active params)

use burn::nn::{Embedding, EmbeddingConfig};
use burn::prelude::*;

use sketchpad_core::kv_cache::{AttentionCache, ModelKvCache};
use sketchpad_core::moe::{SparseMoeFfn, SparseMoeFfnConfig};
use sketchpad_core::rmsnorm::RmsNorm;
use sketchpad_core::rope::RotaryEmbedding;
use sketchpad_core::transformer::{MultiHeadAttention, MultiHeadAttentionConfig, causal_mask};

/// Mixtral Model Configuration
#[derive(Debug, Clone)]
pub struct MixtralConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Intermediate (FFN) dimension per expert
    pub intermediate_size: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of KV heads (for GQA)
    pub num_kv_heads: usize,
    /// Number of experts
    pub num_experts: usize,
    /// Number of experts activated per token
    pub num_experts_per_tok: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// RMSNorm epsilon
    pub norm_eps: f64,
    /// RoPE base frequency
    pub rope_base: f32,
}

impl MixtralConfig {
    /// Mixtral 8x7B configuration
    pub fn mixtral_8x7b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8, // GQA
            num_experts: 8,
            num_experts_per_tok: 2,
            max_seq_len: 32768,
            norm_eps: 1e-5,
            rope_base: 1000000.0,
        }
    }

    /// Mixtral 8x22B configuration
    pub fn mixtral_8x22b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 6144,
            intermediate_size: 16384,
            num_layers: 56,
            num_heads: 48,
            num_kv_heads: 8, // GQA
            num_experts: 8,
            num_experts_per_tok: 2,
            max_seq_len: 65536,
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
            num_experts: 4,
            num_experts_per_tok: 2,
            max_seq_len: 512,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        }
    }

    /// Initialize the model and runtime
    pub fn init<B: Backend>(&self, device: &B::Device) -> (Mixtral<B>, MixtralRuntime<B>) {
        let head_dim = self.hidden_size / self.num_heads;

        let layers: Vec<MixtralLayer<B>> = (0..self.num_layers)
            .map(|_| MixtralLayer {
                attention: MultiHeadAttentionConfig::gqa(
                    self.hidden_size,
                    self.num_heads,
                    self.num_kv_heads,
                )
                .init(device),
                moe: SparseMoeFfnConfig::new(
                    self.hidden_size,
                    self.intermediate_size,
                    self.num_experts,
                    self.num_experts_per_tok,
                )
                .init(device),
                input_norm: RmsNorm::with_eps(self.hidden_size, self.norm_eps, device),
                post_attention_norm: RmsNorm::with_eps(self.hidden_size, self.norm_eps, device),
            })
            .collect();

        let model = Mixtral {
            embed_tokens: EmbeddingConfig::new(self.vocab_size, self.hidden_size).init(device),
            layers,
            norm: RmsNorm::with_eps(self.hidden_size, self.norm_eps, device),
            lm_head: burn::nn::LinearConfig::new(self.hidden_size, self.vocab_size)
                .with_bias(false)
                .init(device),
        };

        let runtime = MixtralRuntime {
            rope: RotaryEmbedding::with_base(head_dim, self.max_seq_len, self.rope_base, device),
            config: self.clone(),
        };

        (model, runtime)
    }
}

/// Single Mixtral transformer layer with MoE
#[derive(Module, Debug)]
pub struct MixtralLayer<B: Backend> {
    /// Multi-head attention
    pub attention: MultiHeadAttention<B>,
    /// Sparse MoE FFN
    pub moe: SparseMoeFfn<B>,
    /// Input layer norm
    pub input_norm: RmsNorm<B>,
    /// Post-attention layer norm
    pub post_attention_norm: RmsNorm<B>,
}

impl<B: Backend> MixtralLayer<B> {
    /// Forward pass through the layer
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        start_pos: usize,
        mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        // Pre-norm attention with residual
        let h = x.clone()
            + self
                .attention
                .forward(self.input_norm.forward(x), Some(rope), start_pos, mask);

        // Pre-norm MoE FFN with residual
        h.clone() + self.moe.forward(self.post_attention_norm.forward(h))
    }

    /// Forward pass with KV cache support
    pub fn forward_cached(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        start_pos: usize,
        mask: Option<Tensor<B, 2>>,
        cache: &mut dyn AttentionCache<B>,
        layer_idx: usize,
    ) -> Tensor<B, 3> {
        // Pre-norm attention with residual (cached)
        let h = x.clone()
            + self.attention.forward_cached(
                self.input_norm.forward(x),
                Some(rope),
                start_pos,
                mask,
                cache,
                layer_idx,
            );

        // Pre-norm MoE FFN with residual
        h.clone() + self.moe.forward(self.post_attention_norm.forward(h))
    }
}

/// Mixtral Model
#[derive(Module, Debug)]
pub struct Mixtral<B: Backend> {
    /// Token embeddings
    pub embed_tokens: Embedding<B>,
    /// Transformer layers with MoE
    pub layers: Vec<MixtralLayer<B>>,
    /// Final layer norm
    pub norm: RmsNorm<B>,
    /// Language model head
    pub lm_head: burn::nn::Linear<B>,
}

/// Runtime state for Mixtral
pub struct MixtralRuntime<B: Backend> {
    /// Rotary position embeddings
    pub rope: RotaryEmbedding<B>,
    /// Model configuration
    pub config: MixtralConfig,
}

/// Output from the Mixtral model
pub struct MixtralOutput<B: Backend> {
    /// Logits over vocabulary: [batch, seq_len, vocab_size]
    pub logits: Tensor<B, 3>,
    /// Hidden states from final layer
    pub hidden_states: Tensor<B, 3>,
}

impl<B: Backend> Mixtral<B> {
    /// Forward pass through the model
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        runtime: &MixtralRuntime<B>,
        cache: Option<&mut dyn AttentionCache<B>>,
    ) -> MixtralOutput<B> {
        let [_batch, seq_len] = input_ids.dims();
        let device = input_ids.device();

        let mut hidden_states = self.embed_tokens.forward(input_ids);

        match cache {
            Some(cache) => {
                let start_pos = cache.seq_len();

                let mask = if seq_len > 1 {
                    let total_len = start_pos + seq_len;
                    Some(prefill_causal_mask::<B>(seq_len, total_len, &device))
                } else {
                    None
                };

                for (layer_idx, layer) in self.layers.iter().enumerate() {
                    hidden_states = layer.forward_cached(
                        hidden_states,
                        &runtime.rope,
                        start_pos,
                        mask.clone(),
                        cache,
                        layer_idx,
                    );
                }
            }
            None => {
                let mask = if seq_len > 1 {
                    Some(causal_mask::<B>(seq_len, &device))
                } else {
                    None
                };

                for layer in &self.layers {
                    hidden_states = layer.forward(hidden_states, &runtime.rope, 0, mask.clone());
                }
            }
        }

        hidden_states = self.norm.forward(hidden_states);
        let logits = self.lm_head.forward(hidden_states.clone());

        MixtralOutput {
            logits,
            hidden_states,
        }
    }

    /// Generate text autoregressively with KV caching
    ///
    /// Uses incremental generation: the prompt is processed in one pass (prefill),
    /// then each subsequent token is generated by only computing the new token
    /// against cached K/V from previous steps.
    pub fn generate(
        &self,
        input_ids: Tensor<B, 2, Int>,
        runtime: &MixtralRuntime<B>,
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
    fn test_mixtral_tiny_forward() {
        let device = Default::default();
        let config = MixtralConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        let input_ids = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device);
        let output = model.forward(input_ids, &runtime, None);

        assert_eq!(output.logits.dims(), [1, 4, 1000]);
        assert_eq!(output.hidden_states.dims(), [1, 4, 128]);
    }

    #[test]
    fn test_mixtral_generate() {
        let device = Default::default();
        let config = MixtralConfig::tiny();
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
    fn test_mixtral_configs() {
        let _ = MixtralConfig::mixtral_8x7b();
        let _ = MixtralConfig::mixtral_8x22b();
    }

    #[test]
    fn test_mixtral_batch() {
        let device = Default::default();
        let config = MixtralConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        let input_ids = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3], [4, 5, 6]], &device);
        let output = model.forward(input_ids, &runtime, None);

        assert_eq!(output.logits.dims(), [2, 3, 1000]);
    }
}
