//! Gemma 2 Model Implementation
//!
//! Gemma 2 is Google's open-weight LLM with unique architectural features.
//!
//! # Architecture
//!
//! Key differences from LLaMA:
//! - Interleaved sliding window and global attention (alternating layers)
//! - Logit soft-capping in attention
//! - GeGLU activation (GELU instead of SiLU)
//! - Pre and post normalization around both attention and FFN
//! - RMSNorm with learned scaling
//!
//! # Supported Variants
//!
//! - Gemma 2 2B
//! - Gemma 2 9B
//! - Gemma 2 27B

use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;

use sketchpad_core::kv_cache::ModelKvCache;
use sketchpad_core::rmsnorm::RmsNorm;
use sketchpad_core::rope::RotaryEmbedding;
use sketchpad_core::transformer::{causal_mask, sliding_window_mask};

/// Gemma 2 Model Configuration
#[derive(Debug, Clone)]
pub struct GemmaConfig {
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
    /// Sliding window size for local attention layers
    pub sliding_window: usize,
    /// Attention logit soft-cap value
    pub attn_logit_softcap: f32,
    /// Final logit soft-cap value
    pub final_logit_softcap: f32,
    /// RMSNorm epsilon
    pub norm_eps: f64,
    /// RoPE base frequency
    pub rope_base: f32,
}

impl GemmaConfig {
    /// Gemma 2 2B configuration
    pub fn gemma2_2b() -> Self {
        Self {
            vocab_size: 256000,
            hidden_size: 2304,
            intermediate_size: 9216,
            num_layers: 26,
            num_heads: 8,
            num_kv_heads: 4,
            max_seq_len: 8192,
            sliding_window: 4096,
            attn_logit_softcap: 50.0,
            final_logit_softcap: 30.0,
            norm_eps: 1e-6,
            rope_base: 10000.0,
        }
    }

    /// Gemma 2 9B configuration
    pub fn gemma2_9b() -> Self {
        Self {
            vocab_size: 256000,
            hidden_size: 3584,
            intermediate_size: 14336,
            num_layers: 42,
            num_heads: 16,
            num_kv_heads: 8,
            max_seq_len: 8192,
            sliding_window: 4096,
            attn_logit_softcap: 50.0,
            final_logit_softcap: 30.0,
            norm_eps: 1e-6,
            rope_base: 10000.0,
        }
    }

    /// Gemma 2 27B configuration
    pub fn gemma2_27b() -> Self {
        Self {
            vocab_size: 256000,
            hidden_size: 4608,
            intermediate_size: 36864,
            num_layers: 46,
            num_heads: 32,
            num_kv_heads: 16,
            max_seq_len: 8192,
            sliding_window: 4096,
            attn_logit_softcap: 50.0,
            final_logit_softcap: 30.0,
            norm_eps: 1e-6,
            rope_base: 10000.0,
        }
    }

    /// Creates a tiny model for testing
    pub fn tiny() -> Self {
        Self {
            vocab_size: 1000,
            hidden_size: 128,
            intermediate_size: 256,
            num_layers: 4, // Need at least 4 to test interleaving
            num_heads: 4,
            num_kv_heads: 2,
            max_seq_len: 512,
            sliding_window: 64,
            attn_logit_softcap: 50.0,
            final_logit_softcap: 30.0,
            norm_eps: 1e-6,
            rope_base: 10000.0,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> (Gemma<B>, GemmaRuntime<B>) {
        let head_dim = self.hidden_size / self.num_heads;

        let layers: Vec<GemmaLayer<B>> = (0..self.num_layers)
            .map(|i| {
                GemmaLayerConfig {
                    hidden_size: self.hidden_size,
                    intermediate_size: self.intermediate_size,
                    num_heads: self.num_heads,
                    num_kv_heads: self.num_kv_heads,
                    norm_eps: self.norm_eps,
                    // Even layers use sliding window, odd layers use global
                    use_sliding_window: i % 2 == 0,
                }
                .init(device)
            })
            .collect();

        let model = Gemma {
            embed_tokens: EmbeddingConfig::new(self.vocab_size, self.hidden_size).init(device),
            layers,
            norm: RmsNorm::with_eps(self.hidden_size, self.norm_eps, device),
        };

        let runtime = GemmaRuntime {
            rope: RotaryEmbedding::with_base(head_dim, self.max_seq_len, self.rope_base, device),
            config: self.clone(),
        };

        (model, runtime)
    }
}

struct GemmaLayerConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    norm_eps: f64,
    use_sliding_window: bool,
}

impl GemmaLayerConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> GemmaLayer<B> {
        let head_dim = self.hidden_size / self.num_heads;
        let kv_dim = head_dim * self.num_kv_heads;

        GemmaLayer {
            attention: GemmaAttention {
                q_proj: LinearConfig::new(self.hidden_size, self.hidden_size)
                    .with_bias(false)
                    .init(device),
                k_proj: LinearConfig::new(self.hidden_size, kv_dim)
                    .with_bias(false)
                    .init(device),
                v_proj: LinearConfig::new(self.hidden_size, kv_dim)
                    .with_bias(false)
                    .init(device),
                o_proj: LinearConfig::new(self.hidden_size, self.hidden_size)
                    .with_bias(false)
                    .init(device),
                num_heads: self.num_heads,
                num_kv_heads: self.num_kv_heads,
                head_dim,
            },
            ffn: GemmaFfn {
                gate_proj: LinearConfig::new(self.hidden_size, self.intermediate_size)
                    .with_bias(false)
                    .init(device),
                up_proj: LinearConfig::new(self.hidden_size, self.intermediate_size)
                    .with_bias(false)
                    .init(device),
                down_proj: LinearConfig::new(self.intermediate_size, self.hidden_size)
                    .with_bias(false)
                    .init(device),
            },
            input_norm: RmsNorm::with_eps(self.hidden_size, self.norm_eps, device),
            post_attention_norm: RmsNorm::with_eps(self.hidden_size, self.norm_eps, device),
            pre_ffn_norm: RmsNorm::with_eps(self.hidden_size, self.norm_eps, device),
            post_ffn_norm: RmsNorm::with_eps(self.hidden_size, self.norm_eps, device),
            use_sliding_window: self.use_sliding_window,
        }
    }
}

/// Gemma attention
#[derive(Module, Debug)]
pub struct GemmaAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> GemmaAttention<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        start_pos: usize,
        mask: Option<Tensor<B, 2>>,
        softcap: f32,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden] = x.dims();

        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

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
        let mut attn = q.matmul(k.transpose()) * scale;

        // Apply logit soft-capping: softcap * tanh(logits / softcap)
        if softcap > 0.0 {
            attn = (attn / softcap).tanh() * softcap;
        }

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

/// Gemma FFN with GeGLU (GELU gating)
#[derive(Module, Debug)]
pub struct GemmaFfn<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B: Backend> GemmaFfn<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // GeGLU: GELU(gate) * up
        let gate = burn::tensor::activation::gelu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}

/// Gemma transformer layer with pre/post norms
#[derive(Module, Debug)]
pub struct GemmaLayer<B: Backend> {
    pub attention: GemmaAttention<B>,
    pub ffn: GemmaFfn<B>,
    pub input_norm: RmsNorm<B>,
    pub post_attention_norm: RmsNorm<B>,
    pub pre_ffn_norm: RmsNorm<B>,
    pub post_ffn_norm: RmsNorm<B>,
    pub use_sliding_window: bool,
}

impl<B: Backend> GemmaLayer<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        start_pos: usize,
        sliding_mask: Option<Tensor<B, 2>>,
        global_mask: Option<Tensor<B, 2>>,
        softcap: f32,
    ) -> Tensor<B, 3> {
        let mask = if self.use_sliding_window {
            sliding_mask
        } else {
            global_mask
        };

        // Pre-norm attention
        let normed = self.input_norm.forward(x.clone());
        let attn_out = self
            .attention
            .forward(normed, rope, start_pos, mask, softcap);
        // Post-attention norm + residual
        let h = x + self.post_attention_norm.forward(attn_out);

        // Pre-FFN norm
        let normed = self.pre_ffn_norm.forward(h.clone());
        let ffn_out = self.ffn.forward(normed);
        // Post-FFN norm + residual
        h + self.post_ffn_norm.forward(ffn_out)
    }
}

/// Gemma Model
#[derive(Module, Debug)]
pub struct Gemma<B: Backend> {
    pub embed_tokens: Embedding<B>,
    pub layers: Vec<GemmaLayer<B>>,
    pub norm: RmsNorm<B>,
}

/// Runtime state for Gemma
pub struct GemmaRuntime<B: Backend> {
    pub rope: RotaryEmbedding<B>,
    pub config: GemmaConfig,
}

/// Output from the Gemma model
pub struct GemmaOutput<B: Backend> {
    pub logits: Tensor<B, 3>,
    pub hidden_states: Tensor<B, 3>,
}

impl<B: Backend> Gemma<B> {
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        runtime: &GemmaRuntime<B>,
        mut cache: Option<&mut ModelKvCache<B>>,
    ) -> GemmaOutput<B> {
        let [batch, seq_len] = input_ids.dims();
        let device = input_ids.device();

        let start_pos = cache.as_ref().map(|c| c.seq_len()).unwrap_or(0);

        // Embedding with scaling
        let mut hidden_states = self.embed_tokens.forward(input_ids);
        // Gemma scales embeddings by sqrt(hidden_size)
        let scale = (runtime.config.hidden_size as f32).sqrt();
        hidden_states = hidden_states * scale;

        // Prepare both mask types
        let (sliding_mask, global_mask) = if seq_len > 1 {
            (
                Some(sliding_window_mask::<B>(
                    seq_len,
                    runtime.config.sliding_window,
                    &device,
                )),
                Some(causal_mask::<B>(seq_len, &device)),
            )
        } else {
            (None, None)
        };

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let _ = cache.as_mut().map(|c| c.layer(layer_idx));
            hidden_states = layer.forward(
                hidden_states,
                &runtime.rope,
                start_pos,
                sliding_mask.clone(),
                global_mask.clone(),
                runtime.config.attn_logit_softcap,
            );
        }

        hidden_states = self.norm.forward(hidden_states);

        // Tied embeddings for logits
        let [_b, _s, hidden_size] = hidden_states.dims();
        let weight = self.embed_tokens.weight.val().transpose();
        let flat_hidden = hidden_states
            .clone()
            .reshape([batch * seq_len, hidden_size]);
        let mut logits = flat_hidden.matmul(weight);

        // Apply final logit soft-capping
        if runtime.config.final_logit_softcap > 0.0 {
            let cap = runtime.config.final_logit_softcap;
            logits = (logits / cap).tanh() * cap;
        }

        let logits = logits.reshape([batch, seq_len, runtime.config.vocab_size]);

        GemmaOutput {
            logits,
            hidden_states,
        }
    }

    pub fn generate(
        &self,
        input_ids: Tensor<B, 2, Int>,
        runtime: &GemmaRuntime<B>,
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
    fn test_gemma_tiny_forward() {
        let device = Default::default();
        let config = GemmaConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        let input_ids = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device);
        let output = model.forward(input_ids, &runtime, None);

        assert_eq!(output.logits.dims(), [1, 4, 1000]);
        assert_eq!(output.hidden_states.dims(), [1, 4, 128]);
    }

    #[test]
    fn test_gemma_generate() {
        let device = Default::default();
        let config = GemmaConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        let prompt = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device);
        let generated = model.generate(prompt, &runtime, 3, 1.0);

        assert_eq!(generated.dims(), [1, 5]);
    }

    #[test]
    fn test_gemma_configs() {
        let _ = GemmaConfig::gemma2_2b();
        let _ = GemmaConfig::gemma2_9b();
        let _ = GemmaConfig::gemma2_27b();
    }

    #[test]
    fn test_geglu() {
        let device = Default::default();

        let ffn = GemmaFfn {
            gate_proj: LinearConfig::new(64, 128).with_bias(false).init(&device),
            up_proj: LinearConfig::new(64, 128).with_bias(false).init(&device),
            down_proj: LinearConfig::new(128, 64).with_bias(false).init(&device),
        };

        let x = Tensor::<TestBackend, 3>::zeros([1, 4, 64], &device);
        let out = ffn.forward(x);

        assert_eq!(out.dims(), [1, 4, 64]);
    }

    #[test]
    fn test_interleaved_attention() {
        let device = Default::default();
        let config = GemmaConfig::tiny();
        let (model, _runtime) = config.init::<TestBackend>(&device);

        // Check that layers alternate between sliding window and global
        assert!(model.layers[0].use_sliding_window); // Layer 0: sliding
        assert!(!model.layers[1].use_sliding_window); // Layer 1: global
        assert!(model.layers[2].use_sliding_window); // Layer 2: sliding
        assert!(!model.layers[3].use_sliding_window); // Layer 3: global
    }
}
