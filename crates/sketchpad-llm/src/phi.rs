//! Phi Model Implementation
//!
//! Phi is Microsoft's family of small language models.
//!
//! # Architecture
//!
//! Phi-3 architecture:
//! - RMSNorm for layer normalization
//! - RoPE positional embeddings (with optional scaled/LongRoPE)
//! - SwiGLU activation in FFN (gate * silu(up))
//! - Grouped Query Attention (GQA) in larger variants
//! - QKV has bias, output projection does not
//!
//! # Supported Variants
//!
//! - Phi-3 Mini (3.8B) - 32 heads, full attention
//! - Phi-3 Small (7B) - 32 heads, 8 KV heads (GQA)
//! - Phi-3 Medium (14B) - 40 heads, 10 KV heads (GQA)
//! - Phi-3.5 Mini (3.8B) - Similar to Phi-3 Mini

use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;

use sketchpad_core::kv_cache::ModelKvCache;
use sketchpad_core::rmsnorm::RmsNorm;
use sketchpad_core::rope::RotaryEmbedding;
use sketchpad_core::transformer::causal_mask;

/// Phi Model Configuration
#[derive(Debug, Clone)]
pub struct PhiConfig {
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
    /// RoPE scaling factor (for long context)
    pub rope_scaling: Option<f32>,
}

impl PhiConfig {
    /// Phi-3 Mini (3.8B) configuration
    pub fn phi3_mini() -> Self {
        Self {
            vocab_size: 32064,
            hidden_size: 3072,
            intermediate_size: 8192,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 32, // Full attention
            max_seq_len: 4096,
            norm_eps: 1e-5,
            rope_base: 10000.0,
            rope_scaling: None,
        }
    }

    /// Phi-3 Mini 128K (long context)
    pub fn phi3_mini_128k() -> Self {
        Self {
            vocab_size: 32064,
            hidden_size: 3072,
            intermediate_size: 8192,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 32,
            max_seq_len: 131072,
            norm_eps: 1e-5,
            rope_base: 10000.0,
            rope_scaling: Some(10.0), // LongRoPE scaling
        }
    }

    /// Phi-3 Small (7B) configuration
    pub fn phi3_small() -> Self {
        Self {
            vocab_size: 100352,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8, // GQA
            max_seq_len: 8192,
            norm_eps: 1e-5,
            rope_base: 10000.0,
            rope_scaling: None,
        }
    }

    /// Phi-3 Medium (14B) configuration
    pub fn phi3_medium() -> Self {
        Self {
            vocab_size: 32064,
            hidden_size: 5120,
            intermediate_size: 17920,
            num_layers: 40,
            num_heads: 40,
            num_kv_heads: 10, // GQA
            max_seq_len: 4096,
            norm_eps: 1e-5,
            rope_base: 10000.0,
            rope_scaling: None,
        }
    }

    /// Phi-3.5 Mini (3.8B) configuration
    pub fn phi3_5_mini() -> Self {
        Self {
            vocab_size: 32064,
            hidden_size: 3072,
            intermediate_size: 8192,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 32,
            max_seq_len: 131072,
            norm_eps: 1e-5,
            rope_base: 10000.0,
            rope_scaling: Some(10.0),
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
            norm_eps: 1e-5,
            rope_base: 10000.0,
            rope_scaling: None,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> (Phi<B>, PhiRuntime<B>) {
        let head_dim = self.hidden_size / self.num_heads;

        let layers: Vec<PhiLayer<B>> = (0..self.num_layers)
            .map(|_| {
                PhiLayerConfig {
                    hidden_size: self.hidden_size,
                    intermediate_size: self.intermediate_size,
                    num_heads: self.num_heads,
                    num_kv_heads: self.num_kv_heads,
                    norm_eps: self.norm_eps,
                }
                .init(device)
            })
            .collect();

        let model = Phi {
            embed_tokens: EmbeddingConfig::new(self.vocab_size, self.hidden_size).init(device),
            layers,
            norm: RmsNorm::with_eps(self.hidden_size, self.norm_eps, device),
            lm_head: LinearConfig::new(self.hidden_size, self.vocab_size)
                .with_bias(false)
                .init(device),
        };

        let runtime = PhiRuntime {
            rope: RotaryEmbedding::with_base(head_dim, self.max_seq_len, self.rope_base, device),
            config: self.clone(),
        };

        (model, runtime)
    }
}

struct PhiLayerConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    norm_eps: f64,
}

impl PhiLayerConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> PhiLayer<B> {
        let head_dim = self.hidden_size / self.num_heads;
        let kv_dim = head_dim * self.num_kv_heads;

        PhiLayer {
            attention: PhiAttention {
                qkv_proj: LinearConfig::new(self.hidden_size, self.hidden_size + 2 * kv_dim)
                    .with_bias(true)
                    .init(device),
                o_proj: LinearConfig::new(self.hidden_size, self.hidden_size)
                    .with_bias(false)
                    .init(device),
                num_heads: self.num_heads,
                num_kv_heads: self.num_kv_heads,
                head_dim,
            },
            ffn: PhiFfn {
                gate_up_proj: LinearConfig::new(self.hidden_size, 2 * self.intermediate_size)
                    .with_bias(false)
                    .init(device),
                down_proj: LinearConfig::new(self.intermediate_size, self.hidden_size)
                    .with_bias(false)
                    .init(device),
                intermediate_size: self.intermediate_size,
            },
            input_norm: RmsNorm::with_eps(self.hidden_size, self.norm_eps, device),
            post_attention_norm: RmsNorm::with_eps(self.hidden_size, self.norm_eps, device),
        }
    }
}

/// Phi attention with fused QKV projection
#[derive(Module, Debug)]
pub struct PhiAttention<B: Backend> {
    /// Fused QKV projection (output: [Q, K, V] concatenated)
    pub qkv_proj: Linear<B>,
    /// Output projection
    pub o_proj: Linear<B>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> PhiAttention<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        start_pos: usize,
        mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden] = x.dims();
        let kv_dim = self.head_dim * self.num_kv_heads;

        // Fused QKV projection
        let qkv = self.qkv_proj.forward(x);

        // Split into Q, K, V
        let q = qkv
            .clone()
            .slice([0..batch, 0..seq_len, 0..self.num_heads * self.head_dim]);
        let k = qkv.clone().slice([
            0..batch,
            0..seq_len,
            self.num_heads * self.head_dim..self.num_heads * self.head_dim + kv_dim,
        ]);
        let v = qkv.slice([
            0..batch,
            0..seq_len,
            self.num_heads * self.head_dim + kv_dim..self.num_heads * self.head_dim + 2 * kv_dim,
        ]);

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
        self.o_proj.forward(out)
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

/// Phi FFN with fused gate/up projection and SiLU activation
#[derive(Module, Debug)]
pub struct PhiFfn<B: Backend> {
    /// Fused gate + up projection
    pub gate_up_proj: Linear<B>,
    /// Down projection
    pub down_proj: Linear<B>,
    /// Intermediate size (stored to avoid weight dimension lookup)
    #[module(skip)]
    pub intermediate_size: usize,
}

impl<B: Backend> PhiFfn<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden] = x.dims();

        let gate_up = self.gate_up_proj.forward(x);

        // Split into gate and up
        let gate = gate_up
            .clone()
            .slice([0..batch, 0..seq_len, 0..self.intermediate_size]);
        let up = gate_up.slice([
            0..batch,
            0..seq_len,
            self.intermediate_size..2 * self.intermediate_size,
        ]);

        // SwiGLU: silu(gate) * up
        let gate = burn::tensor::activation::silu(gate);
        self.down_proj.forward(gate * up)
    }
}

/// Phi transformer layer
#[derive(Module, Debug)]
pub struct PhiLayer<B: Backend> {
    pub attention: PhiAttention<B>,
    pub ffn: PhiFfn<B>,
    pub input_norm: RmsNorm<B>,
    pub post_attention_norm: RmsNorm<B>,
}

impl<B: Backend> PhiLayer<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        start_pos: usize,
        mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        // Pre-norm attention
        let normed = self.input_norm.forward(x.clone());
        let h = x + self.attention.forward(normed, rope, start_pos, mask);

        // Pre-FFN norm
        let normed = self.post_attention_norm.forward(h.clone());
        h + self.ffn.forward(normed)
    }
}

/// Phi Model
#[derive(Module, Debug)]
pub struct Phi<B: Backend> {
    pub embed_tokens: Embedding<B>,
    pub layers: Vec<PhiLayer<B>>,
    pub norm: RmsNorm<B>,
    pub lm_head: Linear<B>,
}

/// Runtime state for Phi
pub struct PhiRuntime<B: Backend> {
    pub rope: RotaryEmbedding<B>,
    pub config: PhiConfig,
}

/// Output from the Phi model
pub struct PhiOutput<B: Backend> {
    pub logits: Tensor<B, 3>,
    pub hidden_states: Tensor<B, 3>,
}

impl<B: Backend> Phi<B> {
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        runtime: &PhiRuntime<B>,
        mut cache: Option<&mut ModelKvCache<B>>,
    ) -> PhiOutput<B> {
        let [_batch, seq_len] = input_ids.dims();
        let device = input_ids.device();

        let start_pos = cache.as_ref().map(|c| c.seq_len()).unwrap_or(0);

        let mut hidden_states = self.embed_tokens.forward(input_ids);

        let mask = if seq_len > 1 {
            Some(causal_mask::<B>(seq_len, &device))
        } else {
            None
        };

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let _ = cache.as_mut().map(|c| c.layer(layer_idx));
            hidden_states = layer.forward(hidden_states, &runtime.rope, start_pos, mask.clone());
        }

        hidden_states = self.norm.forward(hidden_states);

        let logits = self.lm_head.forward(hidden_states.clone());

        PhiOutput {
            logits,
            hidden_states,
        }
    }

    pub fn generate(
        &self,
        input_ids: Tensor<B, 2, Int>,
        runtime: &PhiRuntime<B>,
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
    fn test_phi_tiny_forward() {
        let device = Default::default();
        let config = PhiConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        let input_ids = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device);
        let output = model.forward(input_ids, &runtime, None);

        assert_eq!(output.logits.dims(), [1, 4, 1000]);
        assert_eq!(output.hidden_states.dims(), [1, 4, 128]);
    }

    #[test]
    fn test_phi_generate() {
        let device = Default::default();
        let config = PhiConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        let prompt = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device);
        let generated = model.generate(prompt, &runtime, 3, 1.0);

        assert_eq!(generated.dims(), [1, 5]);
    }

    #[test]
    fn test_phi_configs() {
        let _ = PhiConfig::phi3_mini();
        let _ = PhiConfig::phi3_mini_128k();
        let _ = PhiConfig::phi3_small();
        let _ = PhiConfig::phi3_medium();
        let _ = PhiConfig::phi3_5_mini();
    }

    #[test]
    fn test_phi_gqa() {
        let device = Default::default();
        // Test with GQA config (num_kv_heads < num_heads)
        let mut config = PhiConfig::tiny();
        config.num_heads = 4;
        config.num_kv_heads = 2; // GQA with 2:1 ratio
        let (model, runtime) = config.init::<TestBackend>(&device);

        let input_ids = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3]], &device);
        let output = model.forward(input_ids, &runtime, None);

        assert_eq!(output.logits.dims(), [1, 3, 1000]);
    }

    #[test]
    fn test_fused_qkv() {
        let device = Default::default();
        let config = PhiConfig::tiny();
        let (model, _runtime) = config.init::<TestBackend>(&device);

        // Verify fused QKV has correct output dimension
        // Burn's Linear weight is [in_features, out_features]
        let hidden_size = config.hidden_size;
        let head_dim = hidden_size / config.num_heads;
        let kv_dim = head_dim * config.num_kv_heads;
        let expected_qkv_dim = hidden_size + 2 * kv_dim;

        let qkv_dims = model.layers[0].attention.qkv_proj.weight.val().dims();
        assert_eq!(qkv_dims[0], hidden_size); // in_features
        assert_eq!(qkv_dims[1], expected_qkv_dim); // out_features
    }
}
