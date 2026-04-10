//! Gemma 4 Model Implementation
//!
//! Gemma 4 is Google's Mixture-of-Experts LLM, building on Gemma 2's architecture
//! with sparse MoE layers replacing dense FFN on selected layers.
//!
//! # Architecture
//!
//! Key features:
//! - Interleaved sliding window and global attention (like Gemma 2)
//! - Logit soft-capping in attention and on final logits
//! - GeGLU activation (GELU instead of SiLU)
//! - Pre and post normalization around both attention and FFN
//! - Sparse MoE with 128 experts (top-8) + 1 shared expert on MoE layers
//! - Dense FFN on non-MoE layers
//! - Tied embeddings (no separate lm_head)
//!
//! # Supported Variants
//!
//! - Gemma 4 26B-A4B (128 experts, 8 active, 1 shared)

use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;

use sketchpad_core::kv_cache::ModelKvCache;
use sketchpad_core::rmsnorm::RmsNorm;
use sketchpad_core::rope::RotaryEmbedding;
use sketchpad_core::transformer::{causal_mask, sliding_window_mask};

/// Gemma 4 Model Configuration
#[derive(Debug, Clone)]
pub struct Gemma4Config {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Intermediate (FFN) dimension per expert / dense FFN
    pub intermediate_size: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of KV heads (for GQA)
    pub num_kv_heads: usize,
    /// Head dimension
    pub head_dim: usize,
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
    /// Total number of experts
    pub num_experts: usize,
    /// Number of shared experts (always active)
    pub num_shared_experts: usize,
    /// Number of experts activated per token (top-k)
    pub num_experts_per_tok: usize,
    /// Which layers use MoE (others use dense FFN)
    pub moe_layers: Vec<usize>,
}

impl Gemma4Config {
    /// Gemma 4 26B-A4B configuration
    pub fn gemma4_27b() -> Self {
        // MoE on every other layer starting from 1 (odd layers)
        let moe_layers: Vec<usize> = (0..30).filter(|i| i % 2 == 1).collect();

        Self {
            vocab_size: 262144,
            hidden_size: 3840,
            intermediate_size: 24576,
            num_layers: 30,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            max_seq_len: 8192,
            sliding_window: 1024,
            attn_logit_softcap: 50.0,
            final_logit_softcap: 30.0,
            norm_eps: 1e-6,
            rope_base: 1000000.0,
            num_experts: 128,
            num_shared_experts: 1,
            num_experts_per_tok: 8,
            moe_layers,
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
            head_dim: 32,
            max_seq_len: 512,
            sliding_window: 64,
            attn_logit_softcap: 50.0,
            final_logit_softcap: 30.0,
            norm_eps: 1e-6,
            rope_base: 10000.0,
            num_experts: 4,
            num_shared_experts: 1,
            num_experts_per_tok: 2,
            moe_layers: vec![1, 3], // Odd layers are MoE
        }
    }

    /// Returns true if the given layer index uses MoE
    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        self.moe_layers.contains(&layer_idx)
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> (Gemma4<B>, Gemma4Runtime<B>) {
        let layers: Vec<Gemma4Layer<B>> = (0..self.num_layers)
            .map(|i| {
                let is_moe = self.is_moe_layer(i);
                Gemma4LayerConfig {
                    hidden_size: self.hidden_size,
                    intermediate_size: self.intermediate_size,
                    num_heads: self.num_heads,
                    num_kv_heads: self.num_kv_heads,
                    head_dim: self.head_dim,
                    norm_eps: self.norm_eps,
                    use_sliding_window: i % 2 == 0,
                    is_moe,
                    num_experts: self.num_experts,
                    num_shared_experts: self.num_shared_experts,
                    num_experts_per_tok: self.num_experts_per_tok,
                }
                .init(device)
            })
            .collect();

        let model = Gemma4 {
            embed_tokens: EmbeddingConfig::new(self.vocab_size, self.hidden_size).init(device),
            layers,
            norm: RmsNorm::with_eps(self.hidden_size, self.norm_eps, device),
        };

        let runtime = Gemma4Runtime {
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

struct Gemma4LayerConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    norm_eps: f64,
    use_sliding_window: bool,
    is_moe: bool,
    num_experts: usize,
    num_shared_experts: usize,
    num_experts_per_tok: usize,
}

impl Gemma4LayerConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Gemma4Layer<B> {
        let kv_dim = self.head_dim * self.num_kv_heads;
        let q_dim = self.head_dim * self.num_heads;

        let attention = Gemma4Attention {
            q_proj: LinearConfig::new(self.hidden_size, q_dim)
                .with_bias(false)
                .init(device),
            k_proj: LinearConfig::new(self.hidden_size, kv_dim)
                .with_bias(false)
                .init(device),
            v_proj: LinearConfig::new(self.hidden_size, kv_dim)
                .with_bias(false)
                .init(device),
            o_proj: LinearConfig::new(q_dim, self.hidden_size)
                .with_bias(false)
                .init(device),
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
        };

        let ffn = if self.is_moe {
            Gemma4Ffn::Moe(Gemma4MoE::new(
                self.hidden_size,
                self.intermediate_size,
                self.num_experts,
                self.num_shared_experts,
                self.num_experts_per_tok,
                device,
            ))
        } else {
            Gemma4Ffn::Dense(Gemma4DenseFfn {
                gate_proj: LinearConfig::new(self.hidden_size, self.intermediate_size)
                    .with_bias(false)
                    .init(device),
                up_proj: LinearConfig::new(self.hidden_size, self.intermediate_size)
                    .with_bias(false)
                    .init(device),
                down_proj: LinearConfig::new(self.intermediate_size, self.hidden_size)
                    .with_bias(false)
                    .init(device),
            })
        };

        Gemma4Layer {
            attention,
            ffn,
            input_norm: RmsNorm::with_eps(self.hidden_size, self.norm_eps, device),
            post_attention_norm: RmsNorm::with_eps(self.hidden_size, self.norm_eps, device),
            pre_ffn_norm: RmsNorm::with_eps(self.hidden_size, self.norm_eps, device),
            post_ffn_norm: RmsNorm::with_eps(self.hidden_size, self.norm_eps, device),
            use_sliding_window: self.use_sliding_window,
        }
    }
}

/// Gemma 4 attention (same structure as Gemma 2)
#[derive(Module, Debug)]
pub struct Gemma4Attention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> Gemma4Attention<B> {
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

/// Dense FFN with GeGLU (GELU gating) for non-MoE layers
#[derive(Module, Debug)]
pub struct Gemma4DenseFfn<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B: Backend> Gemma4DenseFfn<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // GeGLU: GELU(gate) * up
        let gate = burn::tensor::activation::gelu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}

/// Single expert FFN (GeGLU) for MoE layers
#[derive(Module, Debug)]
pub struct Gemma4ExpertFfn<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B: Backend> Gemma4ExpertFfn<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // GeGLU: GELU(gate) * up
        let gate = burn::tensor::activation::gelu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}

/// Gemma 4 Mixture of Experts layer
///
/// Contains a router, a set of sparse experts, and shared experts
/// that are always active. The output combines routed expert output
/// with shared expert output.
#[derive(Module, Debug)]
pub struct Gemma4MoE<B: Backend> {
    /// Router: linear projection from hidden_size to num_experts
    pub router: Linear<B>,
    /// Sparse experts (selected by router)
    pub experts: Vec<Gemma4ExpertFfn<B>>,
    /// Shared experts (always active, output added to routed output)
    pub shared_experts: Vec<Gemma4ExpertFfn<B>>,
    /// Number of experts to activate per token
    pub top_k: usize,
    /// Total number of routed experts
    pub num_experts: usize,
}

impl<B: Backend> Gemma4MoE<B> {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        num_experts: usize,
        num_shared_experts: usize,
        top_k: usize,
        device: &B::Device,
    ) -> Self {
        let router = LinearConfig::new(hidden_size, num_experts)
            .with_bias(false)
            .init(device);

        let experts = (0..num_experts)
            .map(|_| Gemma4ExpertFfn {
                gate_proj: LinearConfig::new(hidden_size, intermediate_size)
                    .with_bias(false)
                    .init(device),
                up_proj: LinearConfig::new(hidden_size, intermediate_size)
                    .with_bias(false)
                    .init(device),
                down_proj: LinearConfig::new(intermediate_size, hidden_size)
                    .with_bias(false)
                    .init(device),
            })
            .collect();

        let shared_experts = (0..num_shared_experts)
            .map(|_| Gemma4ExpertFfn {
                gate_proj: LinearConfig::new(hidden_size, intermediate_size)
                    .with_bias(false)
                    .init(device),
                up_proj: LinearConfig::new(hidden_size, intermediate_size)
                    .with_bias(false)
                    .init(device),
                down_proj: LinearConfig::new(intermediate_size, hidden_size)
                    .with_bias(false)
                    .init(device),
            })
            .collect();

        Self {
            router,
            experts,
            shared_experts,
            top_k,
            num_experts,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, hidden_size] = x.dims();
        let num_tokens = batch * seq_len;
        let device = x.device();

        // Flatten to [num_tokens, hidden_size]
        let x_flat = x.clone().reshape([num_tokens, hidden_size]);

        // Router logits: [num_tokens, num_experts]
        let logits: Tensor<B, 2> = self.router.forward(x_flat.clone());

        // Softmax over experts
        let probs = burn::tensor::activation::softmax(logits, 1);

        // Get top-k experts and weights
        let (top_weights, top_indices) = probs.topk_with_indices(self.top_k, 1);

        // Normalize weights
        let weight_sum = top_weights.clone().sum_dim(1);
        let routing_weights = top_weights / weight_sum;

        // Compute routed expert outputs
        let mut output = Tensor::zeros([num_tokens, hidden_size], &device);

        for expert_idx in 0..self.num_experts {
            let expert_mask =
                self.compute_expert_mask(&top_indices, expert_idx, num_tokens, &device);

            if !self.has_assigned_tokens(&expert_mask) {
                continue;
            }

            let expert_weights = self.get_expert_weights(
                &top_indices,
                &routing_weights,
                expert_idx,
                num_tokens,
                &device,
            );

            let expert_input = x_flat.clone().reshape([num_tokens, 1, hidden_size]);
            let expert_out = self.experts[expert_idx].forward(expert_input);
            let expert_out = expert_out.reshape([num_tokens, hidden_size]);

            let weighted_out = expert_out * expert_weights.unsqueeze_dim(1);
            output = output + weighted_out;
        }

        // Add shared expert output
        for shared_expert in &self.shared_experts {
            let shared_input = x_flat.clone().reshape([num_tokens, 1, hidden_size]);
            let shared_out = shared_expert.forward(shared_input);
            let shared_out = shared_out.reshape([num_tokens, hidden_size]);
            output = output + shared_out;
        }

        output.reshape([batch, seq_len, hidden_size])
    }

    fn compute_expert_mask(
        &self,
        expert_indices: &Tensor<B, 2, Int>,
        expert_idx: usize,
        num_tokens: usize,
        device: &B::Device,
    ) -> Tensor<B, 1, Bool> {
        let expert_tensor = Tensor::<B, 2, Int>::from_ints([[expert_idx as i32; 1]; 1], device)
            .repeat_dim(0, num_tokens)
            .repeat_dim(1, self.top_k);

        let matches = expert_indices.clone().equal(expert_tensor);
        matches.any_dim(1).squeeze_dim::<1>(1)
    }

    fn has_assigned_tokens(&self, mask: &Tensor<B, 1, Bool>) -> bool {
        let count: i32 = mask.clone().int().sum().into_scalar().elem();
        count > 0
    }

    fn get_expert_weights(
        &self,
        expert_indices: &Tensor<B, 2, Int>,
        routing_weights: &Tensor<B, 2>,
        expert_idx: usize,
        num_tokens: usize,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let expert_tensor = Tensor::<B, 2, Int>::from_ints([[expert_idx as i32; 1]; 1], device)
            .repeat_dim(0, num_tokens)
            .repeat_dim(1, self.top_k);

        let matches = expert_indices.clone().equal(expert_tensor);
        let matches_float = matches.float();

        (routing_weights.clone() * matches_float)
            .sum_dim(1)
            .squeeze_dim::<1>(1)
    }
}

/// FFN variant: either dense or MoE
#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum Gemma4Ffn<B: Backend> {
    Dense(Gemma4DenseFfn<B>),
    Moe(Gemma4MoE<B>),
}

impl<B: Backend> Gemma4Ffn<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::Dense(ffn) => ffn.forward(x),
            Self::Moe(moe) => moe.forward(x),
        }
    }
}

/// Gemma 4 transformer layer with pre/post norms and MoE/dense FFN
#[derive(Module, Debug)]
pub struct Gemma4Layer<B: Backend> {
    pub attention: Gemma4Attention<B>,
    pub ffn: Gemma4Ffn<B>,
    pub input_norm: RmsNorm<B>,
    pub post_attention_norm: RmsNorm<B>,
    pub pre_ffn_norm: RmsNorm<B>,
    pub post_ffn_norm: RmsNorm<B>,
    pub use_sliding_window: bool,
}

impl<B: Backend> Gemma4Layer<B> {
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

/// Gemma 4 Model
#[derive(Module, Debug)]
pub struct Gemma4<B: Backend> {
    pub embed_tokens: Embedding<B>,
    pub layers: Vec<Gemma4Layer<B>>,
    pub norm: RmsNorm<B>,
}

/// Runtime state for Gemma 4
pub struct Gemma4Runtime<B: Backend> {
    pub rope: RotaryEmbedding<B>,
    pub config: Gemma4Config,
}

/// Output from the Gemma 4 model
pub struct Gemma4Output<B: Backend> {
    pub logits: Tensor<B, 3>,
    pub hidden_states: Tensor<B, 3>,
}

impl<B: Backend> Gemma4<B> {
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        runtime: &Gemma4Runtime<B>,
        mut cache: Option<&mut ModelKvCache<B>>,
    ) -> Gemma4Output<B> {
        let [batch, seq_len] = input_ids.dims();
        let device = input_ids.device();

        let start_pos = cache.as_ref().map(|c| c.seq_len()).unwrap_or(0);

        // Embedding with scaling (same as Gemma 2)
        let mut hidden_states = self.embed_tokens.forward(input_ids);
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

        Gemma4Output {
            logits,
            hidden_states,
        }
    }

    pub fn generate(
        &self,
        input_ids: Tensor<B, 2, Int>,
        runtime: &Gemma4Runtime<B>,
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
    fn test_gemma4_tiny_forward() {
        let device = Default::default();
        let config = Gemma4Config::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        let input_ids = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device);
        let output = model.forward(input_ids, &runtime, None);

        assert_eq!(output.logits.dims(), [1, 4, 1000]);
        assert_eq!(output.hidden_states.dims(), [1, 4, 128]);
    }

    #[test]
    fn test_gemma4_generate() {
        let device = Default::default();
        let config = Gemma4Config::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        let prompt = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device);
        let generated = model.generate(prompt, &runtime, 3, 1.0);

        assert_eq!(generated.dims(), [1, 5]);
    }

    #[test]
    fn test_gemma4_configs() {
        let _ = Gemma4Config::gemma4_27b();
    }

    #[test]
    fn test_gemma4_layer_types() {
        let device = Default::default();
        let config = Gemma4Config::tiny();
        let (model, _runtime) = config.init::<TestBackend>(&device);

        // Layer 0: dense, sliding window
        assert!(model.layers[0].use_sliding_window);
        assert!(matches!(model.layers[0].ffn, Gemma4Ffn::Dense(_)));

        // Layer 1: MoE, global attention
        assert!(!model.layers[1].use_sliding_window);
        assert!(matches!(model.layers[1].ffn, Gemma4Ffn::Moe(_)));

        // Layer 2: dense, sliding window
        assert!(model.layers[2].use_sliding_window);
        assert!(matches!(model.layers[2].ffn, Gemma4Ffn::Dense(_)));

        // Layer 3: MoE, global attention
        assert!(!model.layers[3].use_sliding_window);
        assert!(matches!(model.layers[3].ffn, Gemma4Ffn::Moe(_)));
    }

    #[test]
    fn test_gemma4_dense_ffn() {
        let device = Default::default();

        let ffn = Gemma4DenseFfn {
            gate_proj: LinearConfig::new(64, 128).with_bias(false).init(&device),
            up_proj: LinearConfig::new(64, 128).with_bias(false).init(&device),
            down_proj: LinearConfig::new(128, 64).with_bias(false).init(&device),
        };

        let x = Tensor::<TestBackend, 3>::zeros([1, 4, 64], &device);
        let out = ffn.forward(x);

        assert_eq!(out.dims(), [1, 4, 64]);
    }

    #[test]
    fn test_gemma4_moe_ffn() {
        let device = Default::default();

        let moe = Gemma4MoE::new(64, 128, 4, 1, 2, &device);

        let x = Tensor::<TestBackend, 3>::zeros([1, 4, 64], &device);
        let out = moe.forward(x);

        assert_eq!(out.dims(), [1, 4, 64]);
    }

    #[test]
    fn test_gemma4_is_moe_layer() {
        let config = Gemma4Config::tiny();
        assert!(!config.is_moe_layer(0));
        assert!(config.is_moe_layer(1));
        assert!(!config.is_moe_layer(2));
        assert!(config.is_moe_layer(3));
    }
}
