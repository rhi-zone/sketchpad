//! Zamba / Zamba2: Hybrid Mamba-Backbone with Shared Attention
//!
//! Zyphra's hybrid architecture: a Mamba SSM backbone with shared transformer
//! attention layers interspersed at a regular period.
//!
//! Key distinction from Jamba: in Zamba the attention layers share a *single*
//! set of weights across all attention blocks (weight tying), rather than
//! having separate weights per layer.
//!
//! Zamba2 extends this with per-layer LoRA adapters on the shared attention,
//! enabling per-layer specialization despite shared base weights.
//!
//! Architecture:
//! - Embedding → N blocks → RMSNorm → LM head
//! - Most blocks: RMSNorm → Mamba → RMSNorm → MLP
//! - Attention blocks (layer_idx >= attn_layer_offset and
//!   (layer_idx - attn_layer_offset) % attn_layer_period == 0):
//!   RMSNorm → SharedAttention (+ optional LoRA) → add residual
//!   then RMSNorm → MLP → add residual
//!
//! References:
//! - Zamba: https://arxiv.org/abs/2405.16712
//! - Zamba2: https://arxiv.org/abs/2411.15242

use burn::module::{Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::{Int, activation};
use sketchpad_core::rmsnorm::RmsNorm;

/// Zamba/Zamba2 configuration
#[derive(Clone, Debug)]
pub struct ZambaConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Model dimension
    pub d_model: usize,
    /// Mamba SSM state dimension (default 16)
    pub d_state: usize,
    /// Mamba convolution kernel width (default 4)
    pub d_conv: usize,
    /// Mamba expansion factor (default 2)
    pub expand: usize,
    /// Number of query heads in shared attention
    pub num_heads: usize,
    /// Number of KV heads in shared attention (GQA)
    pub num_kv_heads: usize,
    /// Per-head dimension
    pub head_dim: usize,
    /// First layer (0-indexed) that gets an attention block
    pub attn_layer_offset: usize,
    /// Period between attention layers
    pub attn_layer_period: usize,
    /// MLP hidden dimension
    pub intermediate_size: usize,
    /// If true: Zamba2 variant with LoRA adapters on shared attention
    pub is_zamba2: bool,
    /// LoRA rank (Zamba2 only, default 64)
    pub lora_rank: usize,
    /// RMSNorm epsilon
    pub norm_eps: f64,
}

impl ZambaConfig {
    /// Zamba 7B configuration
    pub fn zamba_7b() -> Self {
        Self {
            vocab_size: 32000,
            num_layers: 76,
            d_model: 3712,
            d_state: 16,
            d_conv: 4,
            expand: 2,
            num_heads: 56,
            num_kv_heads: 56,
            head_dim: 64,
            attn_layer_offset: 6,
            attn_layer_period: 6,
            intermediate_size: 14848,
            is_zamba2: false,
            lora_rank: 64,
            norm_eps: 1e-5,
        }
    }

    /// Zamba2-2.7B configuration
    pub fn zamba2_2_7b() -> Self {
        Self {
            vocab_size: 32000,
            num_layers: 54,
            d_model: 2560,
            d_state: 64,
            d_conv: 4,
            expand: 2,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 80,
            attn_layer_offset: 3,
            attn_layer_period: 6,
            intermediate_size: 10240,
            is_zamba2: true,
            lora_rank: 64,
            norm_eps: 1e-5,
        }
    }

    /// Tiny configuration for testing
    pub fn tiny() -> Self {
        Self {
            vocab_size: 1000,
            num_layers: 8,
            d_model: 64,
            d_state: 8,
            d_conv: 4,
            expand: 2,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            attn_layer_offset: 1,
            attn_layer_period: 4,
            intermediate_size: 128,
            is_zamba2: false,
            lora_rank: 8,
            norm_eps: 1e-5,
        }
    }

    /// Inner Mamba dimension (d_model * expand)
    pub fn d_inner(&self) -> usize {
        self.d_model * self.expand
    }

    /// Mamba dt_rank — low-rank projection dimension for timestep
    pub fn dt_rank(&self) -> usize {
        self.d_model.div_ceil(16)
    }

    /// Returns true if layer_idx is an attention layer
    pub fn is_attention_layer(&self, layer_idx: usize) -> bool {
        layer_idx >= self.attn_layer_offset
            && (layer_idx - self.attn_layer_offset) % self.attn_layer_period == 0
    }

    /// Count how many attention layers precede layer_idx (for LoRA adapter indexing)
    pub fn attn_layer_position(&self, layer_idx: usize) -> usize {
        (self.attn_layer_offset..=layer_idx)
            .filter(|&i| self.is_attention_layer(i))
            .count()
            .saturating_sub(1)
    }

    /// Initialize model and runtime
    pub fn init<B: Backend>(&self, device: &B::Device) -> (Zamba<B>, ZambaRuntime<B>) {
        // Build shared attention (stored in runtime, not in model)
        let shared_attn = ZambaSharedAttention::new(self, device);

        // Build per-layer LoRA adapters (one per attention layer, for Zamba2)
        let num_attn_layers = (0..self.num_layers)
            .filter(|&i| self.is_attention_layer(i))
            .count();
        let lora_adapters: Vec<Option<ZambaLoraAdapter<B>>> = if self.is_zamba2 {
            (0..num_attn_layers)
                .map(|_| Some(ZambaLoraAdapter::new(self, device)))
                .collect()
        } else {
            (0..num_attn_layers).map(|_| None).collect()
        };

        // Build blocks
        let layers: Vec<ZambaBlock<B>> = (0..self.num_layers)
            .map(|i| ZambaBlock::new(self, i, device))
            .collect();

        let model = Zamba {
            embed_tokens: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            layers,
            ln_f: RmsNorm::with_eps(self.d_model, self.norm_eps, device),
            lm_head: LinearConfig::new(self.d_model, self.vocab_size)
                .with_bias(false)
                .init(device),
        };

        let runtime = ZambaRuntime {
            config: self.clone(),
            shared_attn,
            lora_adapters,
        };

        (model, runtime)
    }
}

/// Runtime: holds config, shared attention weights, and LoRA adapters.
///
/// These are stored here rather than in the model because Burn's Module derive
/// does not support Arc-shared weights — the shared attention must be stored once
/// and referenced explicitly during forward.
pub struct ZambaRuntime<B: Backend> {
    pub config: ZambaConfig,
    /// Single shared attention used by all attention blocks
    pub shared_attn: ZambaSharedAttention<B>,
    /// Per-attention-layer LoRA adapters (empty `None` slots for non-Zamba2)
    pub lora_adapters: Vec<Option<ZambaLoraAdapter<B>>>,
}

/// Mutable recurrent state for one Zamba inference pass
#[derive(Clone, Debug)]
pub struct ZambaState<B: Backend> {
    /// Per-Mamba-layer SSM hidden state [batch, d_inner, d_state]
    pub ssm_states: Vec<Tensor<B, 3>>,
    /// Per-Mamba-layer convolution state [batch, d_inner, d_conv-1]
    pub conv_states: Vec<Tensor<B, 3>>,
    /// KV cache per attention layer [batch, n_kv_heads, cached_len, head_dim]
    pub k_caches: Vec<Option<Tensor<B, 4>>>,
    pub v_caches: Vec<Option<Tensor<B, 4>>>,
}

impl<B: Backend> ZambaState<B> {
    /// Create fresh zero state
    pub fn new(config: &ZambaConfig, batch: usize, device: &B::Device) -> Self {
        let d_inner = config.d_inner();
        let num_mamba = (0..config.num_layers)
            .filter(|&i| !config.is_attention_layer(i))
            .count();
        let num_attn = config.num_layers - num_mamba;

        Self {
            ssm_states: (0..num_mamba)
                .map(|_| Tensor::zeros([batch, d_inner, config.d_state], device))
                .collect(),
            conv_states: (0..num_mamba)
                .map(|_| Tensor::zeros([batch, d_inner, config.d_conv - 1], device))
                .collect(),
            k_caches: (0..num_attn).map(|_| None).collect(),
            v_caches: (0..num_attn).map(|_| None).collect(),
        }
    }
}

/// Model output
pub struct ZambaOutput<B: Backend> {
    /// Logits over vocabulary [batch, seq, vocab_size]
    pub logits: Tensor<B, 3>,
    /// Hidden states [batch, seq, d_model]
    pub hidden_states: Tensor<B, 3>,
}

/// Full Zamba model
#[derive(Module, Debug)]
pub struct Zamba<B: Backend> {
    pub embed_tokens: Embedding<B>,
    pub layers: Vec<ZambaBlock<B>>,
    pub ln_f: RmsNorm<B>,
    pub lm_head: Linear<B>,
}

impl<B: Backend> Zamba<B> {
    /// Forward pass.
    ///
    /// The shared attention and LoRA adapters live in `runtime` and are passed
    /// explicitly to each block that needs them.
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        runtime: &ZambaRuntime<B>,
        mut state: Option<&mut ZambaState<B>>,
    ) -> ZambaOutput<B> {
        let mut hidden = self.embed_tokens.forward(input_ids);

        // Track independent mamba/attn layer indices for state slicing
        let mut mamba_idx = 0usize;
        let mut attn_idx = 0usize;

        for (layer_idx, block) in self.layers.iter().enumerate() {
            if runtime.config.is_attention_layer(layer_idx) {
                let (k_cache, v_cache) = match state.as_deref_mut() {
                    Some(s) => (
                        Some(&mut s.k_caches[attn_idx]),
                        Some(&mut s.v_caches[attn_idx]),
                    ),
                    None => (None, None),
                };
                let lora = runtime.lora_adapters[attn_idx].as_ref();
                hidden = block.forward_attn(hidden, &runtime.shared_attn, lora, k_cache, v_cache);
                attn_idx += 1;
            } else {
                let (ssm_state, conv_state) = match state.as_deref_mut() {
                    Some(s) => (
                        Some(&mut s.ssm_states[mamba_idx]),
                        Some(&mut s.conv_states[mamba_idx]),
                    ),
                    None => (None, None),
                };
                hidden = block.forward_mamba(hidden, ssm_state, conv_state);
                mamba_idx += 1;
            }
        }

        hidden = self.ln_f.forward(hidden);
        let logits = self.lm_head.forward(hidden.clone());

        ZambaOutput {
            logits,
            hidden_states: hidden,
        }
    }

    /// Initialize fresh recurrent state
    pub fn init_state(
        &self,
        runtime: &ZambaRuntime<B>,
        batch: usize,
        device: &B::Device,
    ) -> ZambaState<B> {
        ZambaState::new(&runtime.config, batch, device)
    }

    /// Autoregressive generation: prefill prompt, then decode token-by-token
    pub fn generate(
        &self,
        input_ids: Tensor<B, 2, Int>,
        runtime: &ZambaRuntime<B>,
        max_new_tokens: usize,
        sampler: &crate::sampling::SamplerConfig,
    ) -> Tensor<B, 2, Int> {
        let [batch, _] = input_ids.dims();
        let device = input_ids.device();

        let input_data: Vec<i64> = input_ids.to_data().to_vec().unwrap();
        let mut context_tokens: Vec<u32> = input_data.iter().map(|&id| id as u32).collect();

        let mut state = self.init_state(runtime, batch, &device);

        // Prefill
        let output = self.forward(input_ids.clone(), runtime, Some(&mut state));
        let [_, seq_len, vocab_size] = output.logits.dims();
        let mut last_logits = output
            .logits
            .slice([0..batch, seq_len - 1..seq_len, 0..vocab_size])
            .squeeze_dim::<2>(1);

        let mut all_tokens = input_ids;

        for _ in 0..max_new_tokens {
            let token_id =
                crate::sampling::sample_from_logits(last_logits, &context_tokens, sampler);
            context_tokens.push(token_id);

            let next_token = Tensor::<B, 2, Int>::from_ints([[token_id as i32]], &device);
            all_tokens = Tensor::cat(vec![all_tokens, next_token.clone()], 1);

            let output = self.forward(next_token, runtime, Some(&mut state));
            last_logits = output.logits.squeeze_dim::<2>(1);
        }

        all_tokens
    }
}

/// A single Zamba layer block.
///
/// All blocks own a Mamba mixer + MLP.  The `is_attn` flag indicates whether
/// this position in the stack is an attention layer (the actual attention weights
/// live in `ZambaRuntime::shared_attn`; this struct only stores the norms).
#[derive(Module, Debug)]
pub struct ZambaBlock<B: Backend> {
    /// Pre-norm for Mamba (or pre-norm for shared attention on attn layers)
    pub pre_norm: RmsNorm<B>,
    /// Mamba mixer — always present (used on non-attention layers; on attention
    /// layers the mixer output is added as an auxiliary residual per Zamba paper)
    pub mamba: ZambaMamba<B>,
    /// Pre-norm for MLP
    pub mlp_norm: RmsNorm<B>,
    /// SwiGLU MLP
    pub mlp: ZambaMlp<B>,
    /// Whether this is an attention block position
    #[module(skip)]
    pub is_attn: bool,
}

impl<B: Backend> ZambaBlock<B> {
    pub fn new(config: &ZambaConfig, layer_idx: usize, device: &B::Device) -> Self {
        Self {
            pre_norm: RmsNorm::with_eps(config.d_model, config.norm_eps, device),
            mamba: ZambaMamba::new(config, device),
            mlp_norm: RmsNorm::with_eps(config.d_model, config.norm_eps, device),
            mlp: ZambaMlp::new(config, device),
            is_attn: config.is_attention_layer(layer_idx),
        }
    }

    /// Forward for a standard Mamba block
    pub fn forward_mamba(
        &self,
        x: Tensor<B, 3>,
        ssm_state: Option<&mut Tensor<B, 3>>,
        conv_state: Option<&mut Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        // Mamba sub-layer with residual
        let residual = x.clone();
        let normed = self.pre_norm.forward(x);
        let mamba_out = self.mamba.forward(normed, ssm_state, conv_state);
        let x = residual + mamba_out;

        // MLP sub-layer with residual
        let residual = x.clone();
        let normed = self.mlp_norm.forward(x);
        let mlp_out = self.mlp.forward(normed);
        residual + mlp_out
    }

    /// Forward for an attention block
    ///
    /// In Zamba the attention block also runs the Mamba mixer and adds its output
    /// as an extra residual, giving the model both local (SSM) and global (attn)
    /// context at these layers.
    pub fn forward_attn(
        &self,
        x: Tensor<B, 3>,
        shared_attn: &ZambaSharedAttention<B>,
        lora: Option<&ZambaLoraAdapter<B>>,
        k_cache: Option<&mut Option<Tensor<B, 4>>>,
        v_cache: Option<&mut Option<Tensor<B, 4>>>,
    ) -> Tensor<B, 3> {
        let residual = x.clone();

        // Shared attention sub-layer
        let normed = self.pre_norm.forward(x.clone());
        let mut attn_out = shared_attn.forward_cached(normed.clone(), k_cache, v_cache);

        // LoRA correction (Zamba2)
        if let Some(adapter) = lora {
            attn_out = attn_out + adapter.forward(normed);
        }

        let x = residual + attn_out;

        // MLP sub-layer with residual
        let residual = x.clone();
        let normed = self.mlp_norm.forward(x);
        let mlp_out = self.mlp.forward(normed);
        residual + mlp_out
    }
}

/// Mamba selective SSM cell (replicates MambaMixer logic, adapted for ZambaConfig)
#[derive(Module, Debug)]
pub struct ZambaMamba<B: Backend> {
    pub in_proj: Linear<B>,
    pub conv1d: Conv1d<B>,
    pub x_proj: Linear<B>,
    pub dt_proj: Linear<B>,
    pub a_log: Param<Tensor<B, 2>>,
    pub d: Param<Tensor<B, 1>>,
    pub out_proj: Linear<B>,
    #[module(skip)]
    pub d_inner: usize,
    #[module(skip)]
    pub d_state: usize,
    #[module(skip)]
    pub d_conv: usize,
    #[module(skip)]
    pub dt_rank: usize,
}

impl<B: Backend> ZambaMamba<B> {
    pub fn new(config: &ZambaConfig, device: &B::Device) -> Self {
        let d_inner = config.d_inner();
        let d_state = config.d_state;
        let dt_rank = config.dt_rank();

        let a_log_data: Vec<f32> = (0..d_inner)
            .flat_map(|_| (1..=d_state).map(|i| (i as f32).ln()))
            .collect();
        let a_log: Tensor<B, 2> =
            Tensor::<B, 1>::from_floats(&a_log_data[..], device).reshape([d_inner, d_state]);

        Self {
            in_proj: LinearConfig::new(config.d_model, d_inner * 2)
                .with_bias(false)
                .init(device),
            conv1d: Conv1dConfig::new(d_inner, d_inner, config.d_conv)
                .with_groups(d_inner)
                .with_padding(burn::nn::PaddingConfig1d::Explicit(config.d_conv - 1))
                .with_bias(true)
                .init(device),
            x_proj: LinearConfig::new(d_inner, dt_rank + d_state * 2)
                .with_bias(false)
                .init(device),
            dt_proj: LinearConfig::new(dt_rank, d_inner)
                .with_bias(true)
                .init(device),
            a_log: Param::from_tensor(a_log),
            d: Param::from_tensor(Tensor::ones([d_inner], device)),
            out_proj: LinearConfig::new(d_inner, config.d_model)
                .with_bias(false)
                .init(device),
            d_inner,
            d_state,
            d_conv: config.d_conv,
            dt_rank,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        ssm_state: Option<&mut Tensor<B, 3>>,
        conv_state: Option<&mut Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _] = x.dims();

        // Split into SSM branch (x) and gating branch (z)
        let xz = self.in_proj.forward(x);
        let x_branch = xz.clone().slice([0..batch, 0..seq_len, 0..self.d_inner]);
        let z = xz.slice([0..batch, 0..seq_len, self.d_inner..self.d_inner * 2]);

        // Depthwise conv along sequence
        let x_t = x_branch.swap_dims(1, 2); // [batch, d_inner, seq]
        let x_t = if seq_len == 1 {
            if let Some(cs) = conv_state {
                let conv_in = Tensor::cat(vec![cs.clone(), x_t.clone()], 2);
                let total = conv_in.dims()[2];
                let start = total - (self.d_conv - 1);
                *cs = conv_in
                    .clone()
                    .slice([0..batch, 0..self.d_inner, start..total]);
                let out = self.conv1d.forward(conv_in);
                let out_len = out.dims()[2];
                out.slice([0..batch, 0..self.d_inner, out_len - 1..out_len])
            } else {
                self.conv1d.forward(x_t)
            }
        } else {
            let out = self.conv1d.forward(x_t);
            out.slice([0..batch, 0..self.d_inner, 0..seq_len])
        };

        let x_t = activation::silu(x_t).swap_dims(1, 2); // back to [batch, seq, d_inner]

        // SSM parameter projections
        let x_proj = self.x_proj.forward(x_t.clone());
        let dt_low = x_proj
            .clone()
            .slice([0..batch, 0..seq_len, 0..self.dt_rank]);
        let b = x_proj.clone().slice([
            0..batch,
            0..seq_len,
            self.dt_rank..self.dt_rank + self.d_state,
        ]);
        let c = x_proj.slice([
            0..batch,
            0..seq_len,
            self.dt_rank + self.d_state..self.dt_rank + self.d_state * 2,
        ]);

        let dt = activation::softplus(self.dt_proj.forward(dt_low), 1.0);
        let a = -self.a_log.val().exp();

        let y = self.ssm(x_t.clone(), dt, a, b, c, ssm_state);

        // Gate and project
        let z = activation::silu(z);
        self.out_proj.forward(y * z)
    }

    fn ssm(
        &self,
        x: Tensor<B, 3>,
        dt: Tensor<B, 3>,
        a: Tensor<B, 2>,
        b: Tensor<B, 3>,
        c: Tensor<B, 3>,
        ssm_state: Option<&mut Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _] = x.dims();
        let device = x.device();
        let d = self.d.val();

        let mut h = match &ssm_state {
            Some(s) => (*s).clone(),
            None => Tensor::zeros([batch, self.d_inner, self.d_state], &device),
        };

        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let x_t = x
                .clone()
                .slice([0..batch, t..t + 1, 0..self.d_inner])
                .squeeze_dim::<2>(1);
            let dt_t = dt
                .clone()
                .slice([0..batch, t..t + 1, 0..self.d_inner])
                .squeeze_dim::<2>(1);
            let b_t = b
                .clone()
                .slice([0..batch, t..t + 1, 0..self.d_state])
                .squeeze_dim::<2>(1);
            let c_t = c
                .clone()
                .slice([0..batch, t..t + 1, 0..self.d_state])
                .squeeze_dim::<2>(1);

            let dt_exp =
                dt_t.clone()
                    .unsqueeze_dim::<3>(2)
                    .expand([batch, self.d_inner, self.d_state]);
            let a_exp = a
                .clone()
                .unsqueeze_dim::<3>(0)
                .expand([batch, self.d_inner, self.d_state]);
            let d_a = (dt_exp.clone() * a_exp).exp();

            let b_exp = b_t
                .unsqueeze_dim::<3>(1)
                .expand([batch, self.d_inner, self.d_state]);
            let d_b = dt_exp * b_exp;

            let x_exp =
                x_t.clone()
                    .unsqueeze_dim::<3>(2)
                    .expand([batch, self.d_inner, self.d_state]);
            h = d_a * h + d_b * x_exp;

            let c_exp = c_t
                .unsqueeze_dim::<3>(1)
                .expand([batch, self.d_inner, self.d_state]);
            let y_t = (h.clone() * c_exp).sum_dim(2).squeeze_dim::<2>(2);
            let d_exp = d
                .clone()
                .unsqueeze_dim::<2>(0)
                .expand([batch, self.d_inner]);
            outputs.push((y_t + d_exp * x_t).unsqueeze_dim::<3>(1));
        }

        if let Some(s) = ssm_state {
            *s = h;
        }

        Tensor::cat(outputs, 1)
    }
}

/// SwiGLU feed-forward network
#[derive(Module, Debug)]
pub struct ZambaMlp<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B: Backend> ZambaMlp<B> {
    pub fn new(config: &ZambaConfig, device: &B::Device) -> Self {
        Self {
            gate_proj: LinearConfig::new(config.d_model, config.intermediate_size)
                .with_bias(false)
                .init(device),
            up_proj: LinearConfig::new(config.d_model, config.intermediate_size)
                .with_bias(false)
                .init(device),
            down_proj: LinearConfig::new(config.intermediate_size, config.d_model)
                .with_bias(false)
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = activation::silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}

/// Shared GQA attention — single instance used by all attention blocks
#[derive(Module, Debug)]
pub struct ZambaSharedAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    #[module(skip)]
    pub num_heads: usize,
    #[module(skip)]
    pub num_kv_heads: usize,
    #[module(skip)]
    pub head_dim: usize,
}

impl<B: Backend> ZambaSharedAttention<B> {
    pub fn new(config: &ZambaConfig, device: &B::Device) -> Self {
        Self {
            q_proj: LinearConfig::new(config.d_model, config.num_heads * config.head_dim)
                .with_bias(false)
                .init(device),
            k_proj: LinearConfig::new(config.d_model, config.num_kv_heads * config.head_dim)
                .with_bias(false)
                .init(device),
            v_proj: LinearConfig::new(config.d_model, config.num_kv_heads * config.head_dim)
                .with_bias(false)
                .init(device),
            o_proj: LinearConfig::new(config.num_heads * config.head_dim, config.d_model)
                .with_bias(false)
                .init(device),
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
        }
    }

    /// Full-sequence attention (no cache)
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _] = x.dims();

        let q = self
            .q_proj
            .forward(x.clone())
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = self
            .k_proj
            .forward(x.clone())
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = self
            .v_proj
            .forward(x)
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        let (k, v) = self.expand_kv(k, v, batch, seq_len);

        let scale = (self.head_dim as f32).sqrt();
        let attn_weights = q.matmul(k.swap_dims(2, 3)) / scale;
        let mask = Self::causal_mask::<B>(seq_len, &attn_weights.device());
        let attn_weights = activation::softmax(attn_weights + mask, 3);
        let out = attn_weights.matmul(v).swap_dims(1, 2).reshape([
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ]);
        self.o_proj.forward(out)
    }

    /// Attention with KV cache accumulation for autoregressive decoding
    pub fn forward_cached(
        &self,
        x: Tensor<B, 3>,
        k_cache: Option<&mut Option<Tensor<B, 4>>>,
        v_cache: Option<&mut Option<Tensor<B, 4>>>,
    ) -> Tensor<B, 3> {
        let [batch, new_seq, _] = x.dims();

        let q = self
            .q_proj
            .forward(x.clone())
            .reshape([batch, new_seq, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k_new = self
            .k_proj
            .forward(x.clone())
            .reshape([batch, new_seq, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v_new = self
            .v_proj
            .forward(x)
            .reshape([batch, new_seq, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        // Append to cache if provided
        let (k_full, v_full) = match (k_cache, v_cache) {
            (Some(kc), Some(vc)) => {
                let k_full = match kc.take() {
                    Some(cached) => Tensor::cat(vec![cached, k_new], 2),
                    None => k_new,
                };
                let v_full = match vc.take() {
                    Some(cached) => Tensor::cat(vec![cached, v_new], 2),
                    None => v_new,
                };
                *kc = Some(k_full.clone());
                *vc = Some(v_full.clone());
                (k_full, v_full)
            }
            _ => (k_new, v_new),
        };

        let total_seq = k_full.dims()[2];
        let (k, v) = self.expand_kv(k_full, v_full, batch, total_seq);

        let scale = (self.head_dim as f32).sqrt();
        let attn_weights = q.matmul(k.swap_dims(2, 3)) / scale;
        let mask = Self::prefill_causal_mask::<B>(new_seq, total_seq, &attn_weights.device());
        let attn_weights = activation::softmax(attn_weights + mask, 3);
        let out = attn_weights.matmul(v).swap_dims(1, 2).reshape([
            batch,
            new_seq,
            self.num_heads * self.head_dim,
        ]);
        self.o_proj.forward(out)
    }

    /// Expand KV heads for GQA
    fn expand_kv(
        &self,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        batch: usize,
        seq_len: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let n_rep = self.num_heads / self.num_kv_heads;
        if n_rep <= 1 {
            return (k, v);
        }
        let k = k
            .unsqueeze_dim::<5>(2)
            .expand([batch, self.num_kv_heads, n_rep, seq_len, self.head_dim])
            .reshape([batch, self.num_heads, seq_len, self.head_dim]);
        let v = v
            .unsqueeze_dim::<5>(2)
            .expand([batch, self.num_kv_heads, n_rep, seq_len, self.head_dim])
            .reshape([batch, self.num_heads, seq_len, self.head_dim]);
        (k, v)
    }

    fn causal_mask<B2: Backend>(seq_len: usize, device: &B2::Device) -> Tensor<B2, 4> {
        let data: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..seq_len).map(move |j| if j <= i { 0.0f32 } else { f32::NEG_INFINITY })
            })
            .collect();
        Tensor::<B2, 1>::from_floats(&data[..], device).reshape([1, 1, seq_len, seq_len])
    }

    fn prefill_causal_mask<B2: Backend>(
        new_seq: usize,
        total_seq: usize,
        device: &B2::Device,
    ) -> Tensor<B2, 4> {
        let cached = total_seq - new_seq;
        let data: Vec<f32> = (0..new_seq)
            .flat_map(|i| {
                (0..total_seq).map(move |j| {
                    if j <= cached + i {
                        0.0f32
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();
        Tensor::<B2, 1>::from_floats(&data[..], device).reshape([1, 1, new_seq, total_seq])
    }
}

/// Per-layer LoRA adapter on shared attention (Zamba2 only)
///
/// Applied as: `attn_out += lora_b(lora_a(x)) * scale`
#[derive(Module, Debug)]
pub struct ZambaLoraAdapter<B: Backend> {
    pub lora_a: Linear<B>,
    pub lora_b: Linear<B>,
    #[module(skip)]
    pub scale: f32,
}

impl<B: Backend> ZambaLoraAdapter<B> {
    pub fn new(config: &ZambaConfig, device: &B::Device) -> Self {
        let rank = config.lora_rank;
        // Scale following the standard LoRA convention: alpha / rank
        let scale = 1.0f32 / rank as f32;
        Self {
            lora_a: LinearConfig::new(config.d_model, rank)
                .with_bias(false)
                .init(device),
            lora_b: LinearConfig::new(rank, config.d_model)
                .with_bias(false)
                .init(device),
            scale,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.lora_b.forward(self.lora_a.forward(x)) * self.scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_zamba_config_presets() {
        let _ = ZambaConfig::zamba_7b();
        let _ = ZambaConfig::zamba2_2_7b();
    }

    #[test]
    fn test_zamba_tiny_forward() {
        let device = Default::default();
        let config = ZambaConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        let input_ids = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device);
        let output = model.forward(input_ids, &runtime, None);

        assert_eq!(output.logits.dims(), [1, 4, 1000]);
        assert_eq!(output.hidden_states.dims(), [1, 4, 64]);
    }

    #[test]
    fn test_zamba_with_state() {
        let device = Default::default();
        let config = ZambaConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        let mut state = model.init_state(&runtime, 1, &device);

        let input1 = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device);
        let _ = model.forward(input1, &runtime, Some(&mut state));

        let input2 = Tensor::<TestBackend, 2, Int>::from_ints([[3]], &device);
        let output = model.forward(input2, &runtime, Some(&mut state));

        assert_eq!(output.logits.dims(), [1, 1, 1000]);
    }

    #[test]
    fn test_zamba2_tiny_forward() {
        let device = Default::default();
        let mut config = ZambaConfig::tiny();
        config.is_zamba2 = true;

        let (model, runtime) = config.init::<TestBackend>(&device);
        let input_ids = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3]], &device);
        let output = model.forward(input_ids, &runtime, None);

        assert_eq!(output.logits.dims(), [1, 3, 1000]);
    }

    #[test]
    fn test_zamba_generate() {
        let device = Default::default();
        let config = ZambaConfig::tiny();
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
    fn test_attention_layer_detection() {
        let config = ZambaConfig::tiny(); // offset=1, period=4
        // layer 0: 0 < 1 → false
        assert!(!config.is_attention_layer(0));
        // layer 1: (1-1)%4 == 0 → true
        assert!(config.is_attention_layer(1));
        // layer 5: (5-1)%4 == 0 → true
        assert!(config.is_attention_layer(5));
        // layer 2: (2-1)%4 == 1 → false
        assert!(!config.is_attention_layer(2));
    }

    #[test]
    fn test_lora_adapter_forward() {
        let device = Default::default();
        let config = ZambaConfig::tiny();
        let adapter = ZambaLoraAdapter::<TestBackend>::new(&config, &device);

        let x = Tensor::<TestBackend, 3>::zeros([1, 4, 64], &device);
        let out = adapter.forward(x);
        assert_eq!(out.dims(), [1, 4, 64]);
    }
}
