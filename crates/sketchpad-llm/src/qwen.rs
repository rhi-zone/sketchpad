//! Qwen 2.5 Model Implementation
//!
//! Qwen 2.5 is Alibaba's multilingual LLM with strong performance across
//! many languages and tasks.
//!
//! # Architecture
//!
//! Similar to LLaMA:
//! - Pre-norm with RMSNorm
//! - RoPE positional embeddings
//! - SwiGLU activation
//! - GQA (Grouped Query Attention)
//!
//! Key differences:
//! - Bias in QKV projections
//! - Large vocabulary (151,936 tokens)
//! - Tied embeddings (lm_head shares weights with embed_tokens)
//!
//! # Supported Variants
//!
//! - Qwen 2.5 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B
//! - Qwen 2.5 Coder variants
//! - Qwen 2.5 Math variants

use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::DType;

use sketchpad_core::glu::{SwiGluFfn, SwiGluFfnConfig};
use sketchpad_core::kv_cache::{
    AttentionCache, CacheUpdate, CompressedKvCache, KvCacheConfig, ModelKvCache,
};
use sketchpad_core::rmsnorm::RmsNorm;
use sketchpad_core::rope::RotaryEmbedding;
use sketchpad_core::transformer::causal_mask;

use crate::sampling::{SamplerConfig, sample_from_logits};

/// Qwen 2.5 Model Configuration
#[derive(Debug, Clone)]
pub struct QwenConfig {
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
    /// RMSNorm epsilon
    pub norm_eps: f64,
    /// RoPE base frequency
    pub rope_base: f32,
    /// Whether to tie embeddings (lm_head = embed_tokens)
    pub tie_embeddings: bool,
}

impl QwenConfig {
    /// Qwen 2.5 0.5B configuration
    pub fn qwen25_0_5b() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 1024,
            intermediate_size: 2816,
            num_layers: 24,
            num_heads: 16,
            num_kv_heads: 2,
            max_seq_len: 32768,
            norm_eps: 1e-6,
            rope_base: 1000000.0,
            tie_embeddings: true,
        }
    }

    /// Qwen 2.5 1.5B configuration
    pub fn qwen25_1_5b() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 1536,
            intermediate_size: 8960,
            num_layers: 28,
            num_heads: 12,
            num_kv_heads: 2,
            max_seq_len: 32768,
            norm_eps: 1e-6,
            rope_base: 1000000.0,
            tie_embeddings: true,
        }
    }

    /// Qwen 2.5 7B configuration
    pub fn qwen25_7b() -> Self {
        Self {
            vocab_size: 152064,
            hidden_size: 3584,
            intermediate_size: 18944,
            num_layers: 28,
            num_heads: 28,
            num_kv_heads: 4,
            max_seq_len: 131072,
            norm_eps: 1e-6,
            rope_base: 1000000.0,
            tie_embeddings: false,
        }
    }

    /// Qwen 2.5 14B configuration
    pub fn qwen25_14b() -> Self {
        Self {
            vocab_size: 152064,
            hidden_size: 5120,
            intermediate_size: 13824,
            num_layers: 48,
            num_heads: 40,
            num_kv_heads: 8,
            max_seq_len: 131072,
            norm_eps: 1e-5,
            rope_base: 1000000.0,
            tie_embeddings: false,
        }
    }

    /// Qwen 2.5 72B configuration
    pub fn qwen25_72b() -> Self {
        Self {
            vocab_size: 152064,
            hidden_size: 8192,
            intermediate_size: 29568,
            num_layers: 80,
            num_heads: 64,
            num_kv_heads: 8,
            max_seq_len: 131072,
            norm_eps: 1e-5,
            rope_base: 1000000.0,
            tie_embeddings: false,
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
            norm_eps: 1e-6,
            rope_base: 10000.0,
            tie_embeddings: true,
        }
    }

    /// Initialize the model and runtime
    pub fn init<B: Backend>(&self, device: &B::Device) -> (Qwen<B>, QwenRuntime<B>) {
        let head_dim = self.hidden_size / self.num_heads;

        let layers: Vec<QwenLayer<B>> = (0..self.num_layers)
            .map(|_| {
                QwenLayerConfig {
                    hidden_size: self.hidden_size,
                    intermediate_size: self.intermediate_size,
                    num_heads: self.num_heads,
                    num_kv_heads: self.num_kv_heads,
                    norm_eps: self.norm_eps,
                }
                .init(device)
            })
            .collect();

        let embed_tokens = EmbeddingConfig::new(self.vocab_size, self.hidden_size).init(device);

        // For tied embeddings, lm_head is None and we use embed_tokens.weight
        let lm_head = if self.tie_embeddings {
            None
        } else {
            Some(
                LinearConfig::new(self.hidden_size, self.vocab_size)
                    .with_bias(false)
                    .init(device),
            )
        };

        let model = Qwen {
            embed_tokens,
            layers,
            norm: RmsNorm::with_eps(self.hidden_size, self.norm_eps, device),
            lm_head,
        };

        let runtime = QwenRuntime {
            rope: RotaryEmbedding::with_base(head_dim, self.max_seq_len, self.rope_base, device),
            config: self.clone(),
        };

        (model, runtime)
    }
}

/// Qwen layer configuration
struct QwenLayerConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    norm_eps: f64,
}

impl QwenLayerConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> QwenLayer<B> {
        let head_dim = self.hidden_size / self.num_heads;
        let kv_dim = head_dim * self.num_kv_heads;

        QwenLayer {
            attention: QwenAttention {
                q_proj: LinearConfig::new(self.hidden_size, self.hidden_size)
                    .with_bias(true)
                    .init(device),
                k_proj: LinearConfig::new(self.hidden_size, kv_dim)
                    .with_bias(true)
                    .init(device),
                v_proj: LinearConfig::new(self.hidden_size, kv_dim)
                    .with_bias(true)
                    .init(device),
                o_proj: LinearConfig::new(self.hidden_size, self.hidden_size)
                    .with_bias(false)
                    .init(device),
                num_heads: self.num_heads,
                num_kv_heads: self.num_kv_heads,
                head_dim,
            },
            ffn: SwiGluFfnConfig::new(self.hidden_size, self.intermediate_size).init(device),
            input_norm: RmsNorm::with_eps(self.hidden_size, self.norm_eps, device),
            post_attention_norm: RmsNorm::with_eps(self.hidden_size, self.norm_eps, device),
        }
    }
}

/// Qwen attention with bias in QKV
#[derive(Module, Debug)]
pub struct QwenAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> QwenAttention<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        start_pos: usize,
        mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden] = x.dims();

        let dtype = x.dtype();
        let (q, k, v) = if dtype == DType::F16 {
            let x32 = x.cast(DType::F32);
            let q = self.q_proj.forward(x32.clone()).cast(dtype);
            let k = self.k_proj.forward(x32.clone()).cast(dtype);
            let v = self.v_proj.forward(x32).cast(dtype);
            (q, k, v)
        } else {
            let q = self.q_proj.forward(x.clone());
            let k = self.k_proj.forward(x.clone());
            let v = self.v_proj.forward(x);
            (q, k, v)
        };

        let q = q
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        let (q, k) = rope.forward(q, k, start_pos);

        // Repeat KV for GQA
        let k = self.repeat_kv(k);
        let v = self.repeat_kv(v);

        let scale = (self.head_dim as f64).powf(-0.5);
        let attn = q.matmul(k.transpose()) * scale;

        let attn = match mask {
            Some(m) => attn + m.unsqueeze::<3>().unsqueeze(),
            None => attn,
        };

        let attn = burn::tensor::activation::softmax(attn, 3);
        let out = attn.matmul(v);

        let out = out
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * self.head_dim]);
        if dtype == DType::F16 {
            self.o_proj.forward(out.cast(DType::F32)).cast(dtype)
        } else {
            self.o_proj.forward(out)
        }
    }

    pub fn forward_cached(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        start_pos: usize,
        mask: Option<Tensor<B, 2>>,
        cache: &mut dyn AttentionCache<B>,
        layer_idx: usize,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden] = x.dims();

        let dtype = x.dtype();
        let (q, k, v) = if dtype == DType::F16 {
            let x32 = x.cast(DType::F32);
            let q = self.q_proj.forward(x32.clone()).cast(dtype);
            let k = self.k_proj.forward(x32.clone()).cast(dtype);
            let v = self.v_proj.forward(x32).cast(dtype);
            (q, k, v)
        } else {
            let q = self.q_proj.forward(x.clone());
            let k = self.k_proj.forward(x.clone());
            let v = self.v_proj.forward(x);
            (q, k, v)
        };

        let q = q
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        let (q, k) = rope.forward(q, k, start_pos);

        // Update cache and get full K/V sequence
        let CacheUpdate { k, v } = cache.update_layer(layer_idx, k, v);

        // Repeat KV for GQA
        let k = self.repeat_kv(k);
        let v = self.repeat_kv(v);

        let scale = (self.head_dim as f64).powf(-0.5);
        let attn = q.matmul(k.transpose()) * scale;

        let attn = match mask {
            Some(m) => attn + m.unsqueeze::<3>().unsqueeze(),
            None => attn,
        };

        let attn = burn::tensor::activation::softmax(attn, 3);
        let out = attn.matmul(v);

        let out = out
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * self.head_dim]);
        if dtype == DType::F16 {
            self.o_proj.forward(out.cast(DType::F32)).cast(dtype)
        } else {
            self.o_proj.forward(out)
        }
    }

    fn repeat_kv(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        if self.num_kv_heads == self.num_heads {
            return x;
        }

        let [batch, kv_heads, seq_len, head_dim] = x.dims();
        let n_rep = self.num_heads / self.num_kv_heads;

        x.unsqueeze_dim::<5>(2).repeat_dim(2, n_rep).reshape([
            batch,
            kv_heads * n_rep,
            seq_len,
            head_dim,
        ])
    }
}

/// Qwen transformer layer
#[derive(Module, Debug)]
pub struct QwenLayer<B: Backend> {
    pub attention: QwenAttention<B>,
    pub ffn: SwiGluFfn<B>,
    pub input_norm: RmsNorm<B>,
    pub post_attention_norm: RmsNorm<B>,
}

impl<B: Backend> QwenLayer<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        start_pos: usize,
        mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        let h = x.clone()
            + self
                .attention
                .forward(self.input_norm.forward(x), rope, start_pos, mask);
        h.clone() + self.ffn.forward(self.post_attention_norm.forward(h))
    }

    pub fn forward_cached(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        start_pos: usize,
        mask: Option<Tensor<B, 2>>,
        cache: &mut dyn AttentionCache<B>,
        layer_idx: usize,
    ) -> Tensor<B, 3> {
        let h = x.clone()
            + self.attention.forward_cached(
                self.input_norm.forward(x),
                rope,
                start_pos,
                mask,
                cache,
                layer_idx,
            );
        h.clone() + self.ffn.forward(self.post_attention_norm.forward(h))
    }
}

/// Qwen Model
#[derive(Module, Debug)]
pub struct Qwen<B: Backend> {
    pub embed_tokens: Embedding<B>,
    pub layers: Vec<QwenLayer<B>>,
    pub norm: RmsNorm<B>,
    /// LM head (None if tied embeddings)
    pub lm_head: Option<Linear<B>>,
}

/// Runtime state for Qwen
pub struct QwenRuntime<B: Backend> {
    pub rope: RotaryEmbedding<B>,
    pub config: QwenConfig,
}

impl<B: Backend> QwenRuntime<B> {
    /// Create a KV cache according to the given config
    pub fn create_kv_cache(&self, config: &KvCacheConfig) -> Box<dyn AttentionCache<B>> {
        let head_dim = self.config.hidden_size / self.config.num_heads;
        match config {
            KvCacheConfig::Standard => Box::new(ModelKvCache::<B>::new(
                self.config.num_layers,
                self.config.max_seq_len,
            )),
            KvCacheConfig::Compressed { method } => Box::new(CompressedKvCache::<B>::new(
                self.config.num_layers,
                self.config.num_kv_heads,
                head_dim,
                method.clone(),
            )),
        }
    }
}

/// Output from the Qwen model
pub struct QwenOutput<B: Backend> {
    pub logits: Tensor<B, 3>,
    pub hidden_states: Tensor<B, 3>,
}

impl<B: Backend> Qwen<B> {
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        runtime: &QwenRuntime<B>,
        cache: Option<&mut dyn AttentionCache<B>>,
    ) -> QwenOutput<B> {
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

        // Compute logits - use tied embeddings or lm_head
        let logits = match &self.lm_head {
            Some(head) => head.forward(hidden_states.clone()),
            None => {
                // Tied embeddings: logits = hidden_states @ embed_tokens.weight.T
                // hidden_states: [batch, seq_len, hidden_size]
                // weight: [vocab_size, hidden_size] -> transpose to [hidden_size, vocab_size]
                let [batch, seq_len, hidden_size] = hidden_states.dims();
                let weight = self.embed_tokens.weight.val().transpose();
                // Reshape for batch matmul: [batch * seq_len, hidden_size] @ [hidden_size, vocab_size]
                let flat_hidden = hidden_states
                    .clone()
                    .reshape([batch * seq_len, hidden_size]);
                let flat_logits = flat_hidden.matmul(weight);
                flat_logits.reshape([batch, seq_len, runtime.config.vocab_size])
            }
        };

        QwenOutput {
            logits,
            hidden_states,
        }
    }

    pub fn generate(
        &self,
        input_ids: Tensor<B, 2, Int>,
        runtime: &QwenRuntime<B>,
        max_new_tokens: usize,
        cache: &mut dyn AttentionCache<B>,
        sampler: &SamplerConfig,
    ) -> Tensor<B, 2, Int> {
        let [batch, _prompt_len] = input_ids.dims();

        // Track generated token IDs for repetition/DRY penalties
        let input_data: Vec<i64> = input_ids.to_data().to_vec().unwrap();
        let mut context_tokens: Vec<u32> = input_data.iter().map(|&id| id as u32).collect();

        // Prefill: process the entire prompt at once
        let output = self.forward(input_ids.clone(), runtime, Some(cache));

        let seq_len = input_ids.dims()[1];
        let last_logits = output.logits.slice([
            0..batch,
            (seq_len - 1)..seq_len,
            0..runtime.config.vocab_size,
        ]);
        let last_logits = last_logits.reshape([batch, runtime.config.vocab_size]);
        let token_id = sample_from_logits(last_logits, &context_tokens, sampler);
        context_tokens.push(token_id);

        let device = input_ids.device();
        let mut next_token = Tensor::<B, 2, Int>::from_ints([[token_id as i32]], &device);
        let mut all_tokens = Tensor::cat(vec![input_ids, next_token.clone()], 1);

        // Decode: generate one token at a time using the cache
        for _ in 1..max_new_tokens {
            let output = self.forward(next_token, runtime, Some(cache));

            let last_logits = output
                .logits
                .slice([0..batch, 0..1, 0..runtime.config.vocab_size]);
            let last_logits = last_logits.reshape([batch, runtime.config.vocab_size]);
            let token_id = sample_from_logits(last_logits, &context_tokens, sampler);
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
    fn test_qwen_tiny_forward() {
        let device = Default::default();
        let config = QwenConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        let input_ids = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device);
        let output = model.forward(input_ids, &runtime, None);

        assert_eq!(output.logits.dims(), [1, 4, 1000]);
        assert_eq!(output.hidden_states.dims(), [1, 4, 128]);
    }

    #[test]
    fn test_qwen_generate() {
        use sketchpad_core::kv_cache::KvCacheConfig;

        let device = Default::default();
        let config = QwenConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        let prompt = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device);
        let mut cache = runtime.create_kv_cache(&KvCacheConfig::Standard);
        let generated = model.generate(
            prompt,
            &runtime,
            3,
            cache.as_mut(),
            &crate::sampling::SamplerConfig::greedy(),
        );

        assert_eq!(generated.dims(), [1, 5]);
    }

    #[test]
    fn test_qwen_configs() {
        let _ = QwenConfig::qwen25_0_5b();
        let _ = QwenConfig::qwen25_1_5b();
        let _ = QwenConfig::qwen25_7b();
        let _ = QwenConfig::qwen25_14b();
        let _ = QwenConfig::qwen25_72b();
    }

    #[test]
    fn test_qwen_tied_embeddings() {
        let device = Default::default();
        let mut config = QwenConfig::tiny();
        config.tie_embeddings = true;
        let (model, _runtime) = config.init::<TestBackend>(&device);

        assert!(model.lm_head.is_none());
    }

    #[test]
    fn test_qwen_untied_embeddings() {
        let device = Default::default();
        let mut config = QwenConfig::tiny();
        config.tie_embeddings = false;
        let (model, _runtime) = config.init::<TestBackend>(&device);

        assert!(model.lm_head.is_some());
    }
}
