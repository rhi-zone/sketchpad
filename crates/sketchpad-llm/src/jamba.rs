//! Jamba: Hybrid Transformer-Mamba-MoE Architecture
//!
//! Jamba is AI21's hybrid architecture combining:
//! - Mamba (selective SSM) layers for efficient sequence modeling
//! - Transformer attention layers for global context
//! - Mixture of Experts (MoE) for parameter efficiency
//!
//! Architecture:
//! - Interleaves Transformer and Mamba at 1:7 ratio
//! - MoE layers every 2 blocks
//! - 16 experts with top-k=4 routing
//!
//! Reference: "Jamba: A Hybrid Transformer-Mamba Language Model"
//! https://arxiv.org/abs/2403.19887

use burn::module::{Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::{Int, activation};

/// Jamba model configuration
#[derive(Clone, Debug)]
pub struct JambaConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Model dimension
    pub d_model: usize,
    /// Number of layers
    pub n_layer: usize,
    /// Attention head dimension
    pub head_dim: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of KV heads for GQA
    pub n_kv_heads: usize,
    /// SSM state dimension
    pub d_state: usize,
    /// Convolution kernel size
    pub d_conv: usize,
    /// SSM expansion factor
    pub expand: usize,
    /// Delta rank for SSM
    pub dt_rank: usize,
    /// FFN intermediate size
    pub intermediate_size: usize,
    /// Number of experts
    pub n_experts: usize,
    /// Number of experts to route to
    pub n_experts_per_tok: usize,
    /// Attention layer frequency (1 attention per N layers)
    pub attn_layer_period: usize,
    /// MoE layer frequency
    pub moe_layer_period: usize,
    /// Layer norm epsilon
    pub layer_norm_eps: f64,
}

impl JambaConfig {
    /// Jamba 1.0 (52B total, 12B active)
    pub fn jamba_1_0() -> Self {
        Self {
            vocab_size: 65536,
            d_model: 4096,
            n_layer: 32,
            head_dim: 128,
            n_heads: 32,
            n_kv_heads: 8,
            d_state: 16,
            d_conv: 4,
            expand: 2,
            dt_rank: 256,
            intermediate_size: 14336,
            n_experts: 16,
            n_experts_per_tok: 4,
            attn_layer_period: 8, // 1 attention per 8 layers
            moe_layer_period: 2,  // MoE every 2 layers
            layer_norm_eps: 1e-5,
        }
    }

    /// Jamba Mini (12B active params)
    pub fn jamba_mini() -> Self {
        Self {
            vocab_size: 65536,
            d_model: 2560,
            n_layer: 24,
            head_dim: 128,
            n_heads: 20,
            n_kv_heads: 4,
            d_state: 16,
            d_conv: 4,
            expand: 2,
            dt_rank: 160,
            intermediate_size: 8960,
            n_experts: 16,
            n_experts_per_tok: 4,
            attn_layer_period: 8,
            moe_layer_period: 2,
            layer_norm_eps: 1e-5,
        }
    }

    /// Tiny configuration for testing
    pub fn tiny() -> Self {
        Self {
            vocab_size: 1000,
            d_model: 64,
            n_layer: 8,
            head_dim: 16,
            n_heads: 4,
            n_kv_heads: 2,
            d_state: 8,
            d_conv: 4,
            expand: 2,
            dt_rank: 4,
            intermediate_size: 128,
            n_experts: 4,
            n_experts_per_tok: 2,
            attn_layer_period: 4, // 1 attention per 4 layers
            moe_layer_period: 2,
            layer_norm_eps: 1e-5,
        }
    }

    /// Inner dimension for Mamba (d_model * expand)
    pub fn d_inner(&self) -> usize {
        self.d_model * self.expand
    }

    /// Initialize the model and runtime
    pub fn init<B: Backend>(&self, device: &B::Device) -> (Jamba<B>, JambaRuntime<B>) {
        let layers: Vec<JambaBlock<B>> = (0..self.n_layer)
            .map(|i| JambaBlock::new(self, i, device))
            .collect();

        let model = Jamba {
            embed_tokens: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            layers,
            ln_f: LayerNormConfig::new(self.d_model)
                .with_epsilon(self.layer_norm_eps)
                .init(device),
            lm_head: LinearConfig::new(self.d_model, self.vocab_size)
                .with_bias(false)
                .init(device),
        };

        let runtime = JambaRuntime {
            config: self.clone(),
            _marker: std::marker::PhantomData,
        };

        (model, runtime)
    }
}

/// Runtime configuration (non-Module data)
pub struct JambaRuntime<B: Backend> {
    pub config: JambaConfig,
    pub _marker: std::marker::PhantomData<B>,
}

/// State for one Jamba layer during inference
#[derive(Clone, Debug)]
pub struct JambaState<B: Backend> {
    /// SSM hidden state (for Mamba layers) [batch, d_inner, d_state]
    pub ssm_state: Option<Tensor<B, 3>>,
    /// Convolution state [batch, d_inner, d_conv-1]
    pub conv_state: Option<Tensor<B, 3>>,
    /// KV cache for attention layers
    pub k_cache: Option<Tensor<B, 4>>,
    pub v_cache: Option<Tensor<B, 4>>,
}

impl<B: Backend> JambaState<B> {
    /// Create new zero state for a specific layer type
    pub fn new_mamba(config: &JambaConfig, batch: usize, device: &B::Device) -> Self {
        let d_inner = config.d_inner();
        Self {
            ssm_state: Some(Tensor::zeros([batch, d_inner, config.d_state], device)),
            conv_state: Some(Tensor::zeros([batch, d_inner, config.d_conv - 1], device)),
            k_cache: None,
            v_cache: None,
        }
    }

    /// Create state for attention layer
    pub fn new_attention() -> Self {
        Self {
            ssm_state: None,
            conv_state: None,
            k_cache: None,
            v_cache: None,
        }
    }

    /// Create appropriate state for a layer
    pub fn for_layer(
        config: &JambaConfig,
        layer_idx: usize,
        batch: usize,
        device: &B::Device,
    ) -> Self {
        if (layer_idx + 1) % config.attn_layer_period == 0 {
            Self::new_attention()
        } else {
            Self::new_mamba(config, batch, device)
        }
    }
}

/// Jamba model output
pub struct JambaOutput<B: Backend> {
    /// Logits over vocabulary [batch, seq, vocab_size]
    pub logits: Tensor<B, 3>,
    /// Hidden states [batch, seq, d_model]
    pub hidden_states: Tensor<B, 3>,
}

/// Jamba model
#[derive(Module, Debug)]
pub struct Jamba<B: Backend> {
    /// Token embeddings
    pub embed_tokens: Embedding<B>,
    /// Jamba layers (mix of Mamba, attention, and MoE)
    pub layers: Vec<JambaBlock<B>>,
    /// Final layer norm
    pub ln_f: LayerNorm<B>,
    /// Language model head
    pub lm_head: Linear<B>,
}

impl<B: Backend> Jamba<B> {
    /// Forward pass
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        mut states: Option<&mut Vec<JambaState<B>>>,
    ) -> JambaOutput<B> {
        // Embed tokens
        let mut hidden_states = self.embed_tokens.forward(input_ids);

        // Process through layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let state = states.as_mut().map(|s| &mut s[layer_idx]);
            hidden_states = layer.forward(hidden_states, state);
        }

        // Final norm
        hidden_states = self.ln_f.forward(hidden_states);

        // Project to vocabulary
        let logits = self.lm_head.forward(hidden_states.clone());

        JambaOutput {
            logits,
            hidden_states,
        }
    }

    /// Initialize fresh states for recurrent inference
    pub fn init_states(
        &self,
        runtime: &JambaRuntime<B>,
        batch: usize,
        device: &B::Device,
    ) -> Vec<JambaState<B>> {
        (0..runtime.config.n_layer)
            .map(|i| JambaState::for_layer(&runtime.config, i, batch, device))
            .collect()
    }

    /// Generate text autoregressively
    pub fn generate(
        &self,
        input_ids: Tensor<B, 2, Int>,
        runtime: &JambaRuntime<B>,
        max_new_tokens: usize,
        temperature: f32,
    ) -> Tensor<B, 2, Int> {
        let [batch, _] = input_ids.dims();
        let device = input_ids.device();

        // Initialize states
        let mut states = self.init_states(runtime, batch, &device);

        // Process prompt
        let output = self.forward(input_ids.clone(), Some(&mut states));

        // Get last token prediction
        let [_, seq_len, vocab_size] = output.logits.dims();
        let mut last_logits = output
            .logits
            .slice([0..batch, seq_len - 1..seq_len, 0..vocab_size])
            .squeeze_dim::<2>(1);

        let mut all_tokens = input_ids;

        // Generate new tokens
        for _ in 0..max_new_tokens {
            let scaled_logits = if (temperature - 1.0).abs() > 1e-6 {
                last_logits.clone() / temperature
            } else {
                last_logits.clone()
            };

            let next_token = scaled_logits.argmax(1).reshape([batch, 1]);
            all_tokens = Tensor::cat(vec![all_tokens, next_token.clone()], 1);

            // Forward single token with state
            let output = self.forward(next_token, Some(&mut states));
            last_logits = output.logits.squeeze_dim::<2>(1);
        }

        all_tokens
    }
}

/// Jamba block - can be Mamba, Attention, or MoE
#[derive(Module, Debug)]
pub struct JambaBlock<B: Backend> {
    /// Pre-norm
    pub ln: LayerNorm<B>,
    /// Core layer (Mamba mixer or Attention)
    pub core: JambaCore<B>,
    /// FFN (dense or MoE)
    pub ffn: JambaFFN<B>,
}

/// Core layer type
#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum JambaCore<B: Backend> {
    Mamba(JambaMambaMixer<B>),
    Attention(JambaAttention<B>),
}

/// FFN type
#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum JambaFFN<B: Backend> {
    Dense(JambaDenseFFN<B>),
    MoE(JambaMoEFFN<B>),
}

impl<B: Backend> JambaBlock<B> {
    pub fn new(config: &JambaConfig, layer_idx: usize, device: &B::Device) -> Self {
        let is_attention = (layer_idx + 1) % config.attn_layer_period == 0;
        let is_moe = (layer_idx + 1) % config.moe_layer_period == 0;

        let core = if is_attention {
            JambaCore::Attention(JambaAttention::new(config, device))
        } else {
            JambaCore::Mamba(JambaMambaMixer::new(config, device))
        };

        let ffn = if is_moe {
            JambaFFN::MoE(JambaMoEFFN::new(config, device))
        } else {
            JambaFFN::Dense(JambaDenseFFN::new(config, device))
        };

        Self {
            ln: LayerNormConfig::new(config.d_model)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
            core,
            ffn,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, state: Option<&mut JambaState<B>>) -> Tensor<B, 3> {
        // Pre-norm for core
        let residual = x.clone();
        let x = self.ln.forward(x);

        // Core layer (Mamba or Attention)
        let x = match &self.core {
            JambaCore::Mamba(mamba) => {
                let (ssm_state, conv_state) = if let Some(s) = state {
                    (s.ssm_state.as_mut(), s.conv_state.as_mut())
                } else {
                    (None, None)
                };
                mamba.forward(x, ssm_state, conv_state)
            }
            JambaCore::Attention(attn) => {
                // For simplicity, not using KV cache in this implementation
                attn.forward(x)
            }
        };

        let x = x + residual;

        // FFN with residual
        let residual = x.clone();
        let x = match &self.ffn {
            JambaFFN::Dense(ffn) => ffn.forward(x),
            JambaFFN::MoE(moe) => moe.forward(x),
        };

        x + residual
    }
}

/// Mamba mixer for Jamba (similar to standalone Mamba)
#[derive(Module, Debug)]
pub struct JambaMambaMixer<B: Backend> {
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

impl<B: Backend> JambaMambaMixer<B> {
    pub fn new(config: &JambaConfig, device: &B::Device) -> Self {
        let d_inner = config.d_inner();
        let d_state = config.d_state;
        let dt_rank = config.dt_rank;

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

        let xz = self.in_proj.forward(x);
        let x = xz.clone().slice([0..batch, 0..seq_len, 0..self.d_inner]);
        let z = xz.slice([0..batch, 0..seq_len, self.d_inner..self.d_inner * 2]);

        let x = x.swap_dims(1, 2);

        let x = if seq_len == 1 {
            if let Some(cs) = conv_state {
                let conv_in = Tensor::cat(vec![cs.clone(), x.clone()], 2);
                let total_len = conv_in.dims()[2];
                let state_start = total_len - (self.d_conv - 1);
                *cs = conv_in
                    .clone()
                    .slice([0..batch, 0..self.d_inner, state_start..total_len]);
                let x = self.conv1d.forward(conv_in);
                let seq_out = x.dims()[2];
                x.slice([0..batch, 0..self.d_inner, seq_out - 1..seq_out])
            } else {
                self.conv1d.forward(x)
            }
        } else {
            let x = self.conv1d.forward(x);
            x.slice([0..batch, 0..self.d_inner, 0..seq_len])
        };

        let x = activation::silu(x);
        let x = x.swap_dims(1, 2);

        let x_proj = self.x_proj.forward(x.clone());
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

        let dt = self.dt_proj.forward(dt_low);
        let dt = burn::tensor::activation::softplus(dt, 1.0);

        let a = -self.a_log.val().exp();

        let y = self.ssm_forward(x.clone(), dt, a, b, c, ssm_state);

        let z = activation::silu(z);
        let y = y * z;

        self.out_proj.forward(y)
    }

    fn ssm_forward(
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

            let dt_expanded =
                dt_t.clone()
                    .unsqueeze_dim::<3>(2)
                    .expand([batch, self.d_inner, self.d_state]);
            let a_expanded =
                a.clone()
                    .unsqueeze_dim::<3>(0)
                    .expand([batch, self.d_inner, self.d_state]);
            let d_a = (dt_expanded.clone() * a_expanded).exp();

            let b_expanded = b_t
                .unsqueeze_dim::<3>(1)
                .expand([batch, self.d_inner, self.d_state]);
            let d_b = dt_expanded * b_expanded;

            let x_expanded =
                x_t.clone()
                    .unsqueeze_dim::<3>(2)
                    .expand([batch, self.d_inner, self.d_state]);
            h = d_a * h + d_b * x_expanded;

            let c_expanded = c_t
                .unsqueeze_dim::<3>(1)
                .expand([batch, self.d_inner, self.d_state]);
            let y_t = (h.clone() * c_expanded).sum_dim(2).squeeze_dim::<2>(2);
            let d_expanded = d
                .clone()
                .unsqueeze_dim::<2>(0)
                .expand([batch, self.d_inner]);
            let y_t = y_t + d_expanded * x_t;

            outputs.push(y_t.unsqueeze_dim::<3>(1));
        }

        if let Some(s) = ssm_state {
            *s = h;
        }

        Tensor::cat(outputs, 1)
    }
}

/// Attention layer for Jamba (with GQA)
#[derive(Module, Debug)]
pub struct JambaAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    #[module(skip)]
    pub n_heads: usize,
    #[module(skip)]
    pub n_kv_heads: usize,
    #[module(skip)]
    pub head_dim: usize,
}

impl<B: Backend> JambaAttention<B> {
    pub fn new(config: &JambaConfig, device: &B::Device) -> Self {
        Self {
            q_proj: LinearConfig::new(config.d_model, config.n_heads * config.head_dim)
                .with_bias(false)
                .init(device),
            k_proj: LinearConfig::new(config.d_model, config.n_kv_heads * config.head_dim)
                .with_bias(false)
                .init(device),
            v_proj: LinearConfig::new(config.d_model, config.n_kv_heads * config.head_dim)
                .with_bias(false)
                .init(device),
            o_proj: LinearConfig::new(config.n_heads * config.head_dim, config.d_model)
                .with_bias(false)
                .init(device),
            n_heads: config.n_heads,
            n_kv_heads: config.n_kv_heads,
            head_dim: config.head_dim,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _] = x.dims();

        let q = self
            .q_proj
            .forward(x.clone())
            .reshape([batch, seq_len, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = self
            .k_proj
            .forward(x.clone())
            .reshape([batch, seq_len, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = self
            .v_proj
            .forward(x)
            .reshape([batch, seq_len, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        // Expand KV for GQA
        let n_rep = self.n_heads / self.n_kv_heads;
        let k = if n_rep > 1 {
            k.unsqueeze_dim::<5>(2)
                .expand([batch, self.n_kv_heads, n_rep, seq_len, self.head_dim])
                .reshape([batch, self.n_heads, seq_len, self.head_dim])
        } else {
            k
        };
        let v = if n_rep > 1 {
            v.unsqueeze_dim::<5>(2)
                .expand([batch, self.n_kv_heads, n_rep, seq_len, self.head_dim])
                .reshape([batch, self.n_heads, seq_len, self.head_dim])
        } else {
            v
        };

        // Scaled dot-product attention
        let scale = (self.head_dim as f32).sqrt();
        let attn = q.matmul(k.swap_dims(2, 3)) / scale;

        // Causal mask
        let mask = Self::causal_mask::<B>(seq_len, &attn.device());
        let attn = attn + mask;

        let attn = activation::softmax(attn, 3);
        let out = attn.matmul(v);

        // Reshape back
        let out = out
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.n_heads * self.head_dim]);
        self.o_proj.forward(out)
    }

    fn causal_mask<B2: Backend>(seq_len: usize, device: &B2::Device) -> Tensor<B2, 4> {
        let mask_data: Vec<f32> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY }))
            .collect();
        Tensor::<B2, 1>::from_floats(&mask_data[..], device).reshape([1, 1, seq_len, seq_len])
    }
}

/// Dense FFN
#[derive(Module, Debug)]
pub struct JambaDenseFFN<B: Backend> {
    pub ln: LayerNorm<B>,
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B: Backend> JambaDenseFFN<B> {
    pub fn new(config: &JambaConfig, device: &B::Device) -> Self {
        Self {
            ln: LayerNormConfig::new(config.d_model)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
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
        let x = self.ln.forward(x);
        let gate = activation::silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}

/// MoE FFN with top-k routing
#[derive(Module, Debug)]
pub struct JambaMoEFFN<B: Backend> {
    pub ln: LayerNorm<B>,
    pub router: Linear<B>,
    pub experts: Vec<JambaExpert<B>>,
    #[module(skip)]
    pub n_experts_per_tok: usize,
}

impl<B: Backend> JambaMoEFFN<B> {
    pub fn new(config: &JambaConfig, device: &B::Device) -> Self {
        let experts = (0..config.n_experts)
            .map(|_| JambaExpert::new(config, device))
            .collect();

        Self {
            ln: LayerNormConfig::new(config.d_model)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
            router: LinearConfig::new(config.d_model, config.n_experts)
                .with_bias(false)
                .init(device),
            experts,
            n_experts_per_tok: config.n_experts_per_tok,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, d_model] = x.dims();
        let device = x.device();

        let x = self.ln.forward(x);

        // Compute router logits
        let router_logits = self.router.forward(x.clone());
        let router_probs = activation::softmax(router_logits, 2);

        // For simplicity, use a weighted average of all experts
        // In a full implementation, you'd use top-k selection
        let mut output = Tensor::zeros([batch, seq_len, d_model], &device);

        for (expert_idx, expert) in self.experts.iter().enumerate() {
            let expert_out = expert.forward(x.clone());
            let weight =
                router_probs
                    .clone()
                    .slice([0..batch, 0..seq_len, expert_idx..expert_idx + 1]);
            output = output + expert_out * weight.expand([batch, seq_len, d_model]);
        }

        output
    }
}

/// Single expert FFN
#[derive(Module, Debug)]
pub struct JambaExpert<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B: Backend> JambaExpert<B> {
    pub fn new(config: &JambaConfig, device: &B::Device) -> Self {
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_jamba_config() {
        let _ = JambaConfig::jamba_1_0();
        let _ = JambaConfig::jamba_mini();
    }

    #[test]
    fn test_jamba_tiny_forward() {
        let device = Default::default();
        let config = JambaConfig::tiny();
        let (model, _runtime) = config.init::<TestBackend>(&device);

        let input_ids = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device);
        let output = model.forward(input_ids, None);

        assert_eq!(output.logits.dims(), [1, 4, 1000]);
        assert_eq!(output.hidden_states.dims(), [1, 4, 64]);
    }

    #[test]
    fn test_jamba_with_state() {
        let device = Default::default();
        let config = JambaConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        let mut states = model.init_states(&runtime, 1, &device);

        // First forward pass
        let input1 = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device);
        let _ = model.forward(input1, Some(&mut states));

        // Second forward pass (incremental)
        let input2 = Tensor::<TestBackend, 2, Int>::from_ints([[3]], &device);
        let output = model.forward(input2, Some(&mut states));

        assert_eq!(output.logits.dims(), [1, 1, 1000]);
    }

    #[test]
    fn test_jamba_generate() {
        let device = Default::default();
        let config = JambaConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        let prompt = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device);
        let generated = model.generate(prompt, &runtime, 3, 1.0);

        assert_eq!(generated.dims(), [1, 5]);
    }

    #[test]
    fn test_jamba_mamba_mixer() {
        let device = Default::default();
        let config = JambaConfig::tiny();
        let mixer = JambaMambaMixer::new(&config, &device);

        let x = Tensor::<TestBackend, 3>::zeros([1, 4, 64], &device);
        let output = mixer.forward(x, None, None);

        assert_eq!(output.dims(), [1, 4, 64]);
    }

    #[test]
    fn test_jamba_attention() {
        let device = Default::default();
        let config = JambaConfig::tiny();
        let attn = JambaAttention::new(&config, &device);

        let x = Tensor::<TestBackend, 3>::zeros([1, 4, 64], &device);
        let output = attn.forward(x);

        assert_eq!(output.dims(), [1, 4, 64]);
    }

    #[test]
    fn test_jamba_moe() {
        let device = Default::default();
        let config = JambaConfig::tiny();
        let moe = JambaMoEFFN::new(&config, &device);

        let x = Tensor::<TestBackend, 3>::zeros([1, 4, 64], &device);
        let output = moe.forward(x);

        assert_eq!(output.dims(), [1, 4, 64]);
    }
}
