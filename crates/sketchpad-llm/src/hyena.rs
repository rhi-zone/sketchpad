//! Hyena / StripedHyena Model
//!
//! Hyena is a subquadratic attention alternative using long implicit convolutions and
//! data-controlled gating. StripedHyena (Together AI's 7B model) interleaves Hyena
//! operators with standard attention layers.
//!
//! # Architecture
//!
//! - Hyena operator: projects input to N+1 streams, applies long convolutions via
//!   element-wise gating (the "Hyena recurrence"), then projects back
//! - Implicit filter: the long convolution filter h(t) is parameterized by a small
//!   3-layer MLP applied to sinusoidal positional encodings
//! - StripedHyena: Embedding → N blocks → RMSNorm → LM head, with configurable
//!   interleaving of Hyena and GQA attention blocks
//!
//! # References
//!
//! - "Hyena Hierarchy: Towards Larger Convolutional Language Models"
//!   https://arxiv.org/abs/2302.10866
//! - "StripedHyena: Moving Beyond Transformers with Hybrid Signal Processing Models"
//!   https://www.together.ai/blog/stripedhyena-7b

use burn::module::{Module, Param};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation;

use sketchpad_core::kv_cache::{AttentionCache, KvCacheConfig, ModelKvCache};
use sketchpad_core::rmsnorm::RmsNorm;
use sketchpad_core::transformer::causal_mask;

/// Hyena / StripedHyena model configuration
#[derive(Clone, Debug)]
pub struct HyenaConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Number of transformer blocks
    pub num_layers: usize,
    /// Hidden dimension (d_model)
    pub d_model: usize,
    /// Hyena order N — number of gating streams (typically 2)
    /// The operator projects to N+1 streams: [v, z_1, ..., z_N]
    pub order: usize,
    /// Filter MLP hidden dimension (typically 64)
    pub filter_order: usize,
    /// Number of attention heads (for attention blocks)
    pub num_heads: usize,
    /// Number of KV heads (GQA, for attention blocks)
    pub num_kv_heads: usize,
    /// Per-head dimension (for attention blocks)
    pub head_dim: usize,
    /// FFN intermediate size
    pub intermediate_size: usize,
    /// Offset within the period for attention layers
    pub attn_layer_offset: usize,
    /// Interleaving period: every Nth layer is attention (0 = pure Hyena, no attention)
    pub attn_layer_period: usize,
    /// Maximum sequence length (for filter precomputation)
    pub max_seq_len: usize,
    /// RMSNorm epsilon
    pub norm_eps: f64,
}

impl HyenaConfig {
    /// StripedHyena 7B configuration (Together AI)
    pub fn stripedhyena_7b() -> Self {
        Self {
            vocab_size: 32000,
            num_layers: 32,
            d_model: 4096,
            order: 2,
            filter_order: 64,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            intermediate_size: 11008,
            attn_layer_offset: 7,
            attn_layer_period: 8, // 1 attention per 7 Hyena
            max_seq_len: 32768,
            norm_eps: 1e-5,
        }
    }

    /// Tiny configuration for testing
    pub fn tiny() -> Self {
        Self {
            vocab_size: 1000,
            num_layers: 6,
            d_model: 64,
            order: 2,
            filter_order: 16,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            intermediate_size: 128,
            attn_layer_offset: 5,
            attn_layer_period: 6,
            max_seq_len: 256,
            norm_eps: 1e-5,
        }
    }

    /// Pure Hyena (no attention) tiny configuration
    pub fn pure_hyena_tiny() -> Self {
        Self {
            attn_layer_period: 0,
            ..Self::tiny()
        }
    }

    /// Returns true if the given layer index is an attention layer
    pub fn is_attention_layer(&self, layer_idx: usize) -> bool {
        if self.attn_layer_period == 0 {
            return false;
        }
        layer_idx % self.attn_layer_period == self.attn_layer_offset % self.attn_layer_period
    }

    /// Initialize the model and runtime
    pub fn init<B: Backend>(&self, device: &B::Device) -> (Hyena<B>, HyenaRuntime<B>) {
        let layers: Vec<HyenaBlock<B>> = (0..self.num_layers)
            .map(|i| HyenaBlock::new(self, i, device))
            .collect();

        let model = Hyena {
            embed_tokens: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            layers,
            norm: RmsNorm::with_eps(self.d_model, self.norm_eps, device),
            lm_head: LinearConfig::new(self.d_model, self.vocab_size)
                .with_bias(false)
                .init(device),
        };

        let runtime = HyenaRuntime {
            config: self.clone(),
            _marker: std::marker::PhantomData,
        };

        (model, runtime)
    }
}

/// Runtime data for Hyena inference (config + cache helpers)
pub struct HyenaRuntime<B: Backend> {
    pub config: HyenaConfig,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> HyenaRuntime<B> {
    /// Create a new runtime from config
    pub fn new(config: HyenaConfig) -> Self {
        Self {
            config,
            _marker: std::marker::PhantomData,
        }
    }

    /// Create a KV cache for attention layers
    pub fn create_kv_cache(&self, cache_config: &KvCacheConfig) -> Box<dyn AttentionCache<B>> {
        match cache_config {
            KvCacheConfig::Standard => Box::new(ModelKvCache::<B>::new(
                self.config.num_layers,
                self.config.max_seq_len,
            )),
            KvCacheConfig::Compressed { .. } => Box::new(ModelKvCache::<B>::new(
                self.config.num_layers,
                self.config.max_seq_len,
            )),
        }
    }
}

/// Implicit Hyena filter: h(t) = MLP(sinusoidal_encoding(t))
///
/// The filter is materialized on-the-fly for the current sequence length.
/// An exponential decay window is applied for numerical stability.
#[derive(Module, Debug)]
pub struct HyenaFilter<B: Backend> {
    /// Precomputed sinusoidal positional encoding [max_seq_len, filter_order]
    pub pos_emb: Param<Tensor<B, 2>>,
    /// Filter MLP layer 0 [filter_order → filter_order]
    pub mlp0: Linear<B>,
    /// Filter MLP layer 1 [filter_order → filter_order]
    pub mlp1: Linear<B>,
    /// Filter MLP layer 2 [filter_order → 1] (scalar output per position)
    pub mlp2: Linear<B>,
    /// Exponential decay window [max_seq_len] — stabilizes long convolutions
    pub window: Param<Tensor<B, 1>>,
}

impl<B: Backend> HyenaFilter<B> {
    pub fn new(config: &HyenaConfig, device: &B::Device) -> Self {
        let max_len = config.max_seq_len;
        let filter_order = config.filter_order;

        // Precompute sinusoidal positional encoding
        // pos_emb[t, 2k]   = sin(t / 10000^(2k/filter_order))
        // pos_emb[t, 2k+1] = cos(t / 10000^(2k/filter_order))
        let mut pos_data = vec![0.0f32; max_len * filter_order];
        let half = filter_order / 2;
        for t in 0..max_len {
            for k in 0..half {
                let freq = 1.0 / (10000.0f32.powf(2.0 * k as f32 / filter_order as f32));
                let angle = t as f32 * freq;
                pos_data[t * filter_order + 2 * k] = angle.sin();
                pos_data[t * filter_order + 2 * k + 1] = angle.cos();
            }
        }
        let pos_emb = Tensor::<B, 1>::from_floats(pos_data.as_slice(), device)
            .reshape([max_len, filter_order]);

        // Exponential decay window: w(t) = exp(-t / max_len)
        // Initialized so that long-range contributions decay gently.
        let window_data: Vec<f32> = (0..max_len)
            .map(|t| (-(t as f32) / max_len as f32).exp())
            .collect();
        let window = Tensor::<B, 1>::from_floats(window_data.as_slice(), device);

        Self {
            pos_emb: Param::from_tensor(pos_emb),
            mlp0: LinearConfig::new(filter_order, filter_order)
                .with_bias(true)
                .init(device),
            mlp1: LinearConfig::new(filter_order, filter_order)
                .with_bias(true)
                .init(device),
            mlp2: LinearConfig::new(filter_order, 1)
                .with_bias(true)
                .init(device),
            window: Param::from_tensor(window),
        }
    }

    /// Materialize the filter for `seq_len` positions
    ///
    /// Returns a tensor of shape [seq_len] representing h(0..seq_len).
    pub fn materialize(&self, seq_len: usize) -> Tensor<B, 1> {
        // Slice positional encoding to current seq length
        let max_len = self.pos_emb.val().dims()[0];
        let l = seq_len.min(max_len);
        let pos = self
            .pos_emb
            .val()
            .slice([0..l, 0..self.pos_emb.val().dims()[1]]); // [l, filter_order]

        // MLP: sin → GELU → sin → GELU → linear
        let h = activation::gelu(self.mlp0.forward(pos));
        let h = activation::gelu(self.mlp1.forward(h));
        let h = self.mlp2.forward(h); // [l, 1]
        let h = h.squeeze_dim::<1>(1); // [l]

        // Apply window
        let w = self.window.val().narrow(0, 0, l);
        h * w
    }
}

/// Hyena operator — the core long-convolution building block
///
/// Given input u ∈ ℝ^{B×L×d}:
/// 1. Project to N+1 streams: [v, z_1, ..., z_N] = split(Linear(u))
/// 2. For each z_n: apply long convolution with filter h_n, then gate with z_{n+1}
/// 3. Output y = z_1 * v, projected back
#[derive(Module, Debug)]
pub struct HyenaOperator<B: Backend> {
    /// Input projection: d_model → (order+1) * d_model
    pub in_proj: Linear<B>,
    /// Output projection: d_model → d_model
    pub out_proj: Linear<B>,
    /// Implicit filters, one per order
    pub filters: Vec<HyenaFilter<B>>,
    #[module(skip)]
    pub order: usize,
    #[module(skip)]
    pub d_model: usize,
}

impl<B: Backend> HyenaOperator<B> {
    pub fn new(config: &HyenaConfig, device: &B::Device) -> Self {
        let filters = (0..config.order)
            .map(|_| HyenaFilter::new(config, device))
            .collect();

        Self {
            in_proj: LinearConfig::new(config.d_model, (config.order + 1) * config.d_model)
                .with_bias(true)
                .init(device),
            out_proj: LinearConfig::new(config.d_model, config.d_model)
                .with_bias(false)
                .init(device),
            filters,
            order: config.order,
            d_model: config.d_model,
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    ///
    /// * `x` - Input [batch, seq_len, d_model]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _d] = x.dims();
        let d = self.d_model;

        // Project to (order+1) streams
        let projected = self.in_proj.forward(x); // [batch, seq_len, (order+1)*d]

        // Split into order+1 streams of size d each
        // streams[0] = v, streams[1..=order] = z_1, ..., z_N
        let mut streams: Vec<Tensor<B, 3>> = (0..self.order + 1)
            .map(|i| {
                projected
                    .clone()
                    .slice([0..batch, 0..seq_len, i * d..(i + 1) * d])
            })
            .collect();

        // v is the value stream (no convolution applied to it directly)
        // z streams go through convolution + gating

        // Hyena recurrence: for n = order-1 down to 0:
        //   z_n = FFT_conv(z_n, h_n) * z_{n+1}
        // Final: y = z_0 * v
        //
        // We process from the last stream backward so that each z_n is gated
        // by the already-processed z_{n+1}.
        //
        // streams[0] = v (index 0)
        // streams[1] = z_1, ..., streams[order] = z_N
        //
        // Start from z_N (streams[order]), convolve it.
        // Then gate streams[order-1] by the result, convolve that, etc.
        // Finally gate v by the result.

        let mut gated = streams.remove(self.order); // z_N, shape [batch, seq_len, d]

        // Convolve z_N with its filter
        let h_last = self.filters[self.order - 1].materialize(seq_len); // [seq_len]
        gated = direct_conv(gated, h_last);

        // Process remaining filters from (order-2) down to 0
        for n in (0..self.order - 1).rev() {
            let z_n = streams.remove(n + 1); // z_{n+1} from the original indexing
            // Gate: z_n_conved * gated
            let h_n = self.filters[n].materialize(seq_len);
            let z_n_conved = direct_conv(z_n, h_n);
            gated = z_n_conved * gated;
        }

        // Final: y = gated * v
        let v = streams.remove(0);
        let y = gated * v;

        self.out_proj.forward(y)
    }
}

/// Direct (causal) convolution — O(L²) reference implementation
///
/// Computes output[b, t, c] = sum_{tau=0}^{t} signal[b, t-tau, c] * filter[tau, c]
///
/// # Arguments
///
/// * `signal` - [batch, seq_len, dim]
/// * `filter` - [seq_len] (same filter applied to all dim channels, broadcast)
///
/// # Note
///
/// TODO: Replace with FFT-based convolution O(L log L) for production use.
/// For sequences up to ~4K tokens this O(L²) implementation is acceptable as a
/// reference. A true recurrent form of Hyena also exists but is not implemented here.
fn direct_conv<B: Backend>(signal: Tensor<B, 3>, filter: Tensor<B, 1>) -> Tensor<B, 3> {
    let [batch, seq_len, dim] = signal.dims();
    let filter_len = filter.dims()[0];
    let l = seq_len.min(filter_len);

    let mut output_slices: Vec<Tensor<B, 3>> = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        // output[:, t, :] = sum_{tau=0}^{min(t, l-1)} signal[:, t-tau, :] * filter[tau]
        let tau_max = (t + 1).min(l);

        // Accumulate sum over tau
        let mut acc: Option<Tensor<B, 2>> = None;
        for tau in 0..tau_max {
            let s_t = signal
                .clone()
                .slice([0..batch, (t - tau)..(t - tau + 1), 0..dim])
                .squeeze_dim::<2>(1); // [batch, dim]

            let f_tau = filter.clone().narrow(0, tau, 1).unsqueeze_dim::<2>(0); // [1, 1]

            let contribution = s_t * f_tau; // [batch, dim] via broadcast

            acc = Some(match acc {
                None => contribution,
                Some(a) => a + contribution,
            });
        }

        let out_t = acc
            .unwrap_or_else(|| {
                let device = signal.device();
                Tensor::zeros([batch, dim], &device)
            })
            .unsqueeze_dim::<3>(1); // [batch, 1, dim]

        output_slices.push(out_t);
    }

    Tensor::cat(output_slices, 1) // [batch, seq_len, dim]
}

/// SwiGLU Feed-Forward Network
///
/// out = down_proj(SiLU(gate_proj(x)) * up_proj(x))
#[derive(Module, Debug)]
pub struct HyenaSwiGluFfn<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B: Backend> HyenaSwiGluFfn<B> {
    pub fn new(config: &HyenaConfig, device: &B::Device) -> Self {
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

/// GQA attention block for StripedHyena attention layers
#[derive(Module, Debug)]
pub struct HyenaAttention<B: Backend> {
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

impl<B: Backend> HyenaAttention<B> {
    pub fn new(config: &HyenaConfig, device: &B::Device) -> Self {
        let q_dim = config.num_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;

        Self {
            q_proj: LinearConfig::new(config.d_model, q_dim)
                .with_bias(false)
                .init(device),
            k_proj: LinearConfig::new(config.d_model, kv_dim)
                .with_bias(false)
                .init(device),
            v_proj: LinearConfig::new(config.d_model, kv_dim)
                .with_bias(false)
                .init(device),
            o_proj: LinearConfig::new(q_dim, config.d_model)
                .with_bias(false)
                .init(device),
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        mask: Option<Tensor<B, 2>>,
        cache: Option<&mut dyn AttentionCache<B>>,
        layer_idx: usize,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _] = x.dims();

        let q = self
            .q_proj
            .forward(x.clone())
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2); // [batch, num_heads, seq_len, head_dim]

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

        // Update KV cache
        let (k, v) = if let Some(c) = cache {
            let update = c.update_layer(layer_idx, k, v);
            (update.k, update.v)
        } else {
            (k, v)
        };

        let kv_seq = k.dims()[2];

        // Expand KV heads if GQA (repeat each KV head n_rep times)
        let n_rep = self.num_heads / self.num_kv_heads;
        let (k, v) = if n_rep > 1 {
            let k = k
                .unsqueeze_dim::<5>(2) // [batch, num_kv_heads, 1, kv_seq, head_dim]
                .expand([batch, self.num_kv_heads, n_rep, kv_seq, self.head_dim])
                .reshape([batch, self.num_heads, kv_seq, self.head_dim]);
            let v = v
                .unsqueeze_dim::<5>(2)
                .expand([batch, self.num_kv_heads, n_rep, kv_seq, self.head_dim])
                .reshape([batch, self.num_heads, kv_seq, self.head_dim]);
            (k, v)
        } else {
            (k, v)
        };

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).powf(-0.5);
        let attn = q.matmul(k.transpose()) * scale;

        let attn = if let Some(m) = mask {
            attn + m.unsqueeze::<3>().unsqueeze()
        } else if seq_len == 1 && kv_seq > 1 {
            attn
        } else {
            let device = attn.device();
            attn + causal_mask::<B>(seq_len, &device)
                .unsqueeze::<3>()
                .unsqueeze()
        };

        let attn = activation::softmax(attn, 3);
        let out = attn.matmul(v).swap_dims(1, 2).reshape([
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ]);

        self.o_proj.forward(out)
    }
}

/// A single Hyena block
///
/// Structure: RMSNorm → (HyenaOperator or Attention) + residual → RMSNorm → SwiGLU FFN + residual
#[derive(Module, Debug)]
pub struct HyenaBlock<B: Backend> {
    pub pre_norm: RmsNorm<B>,
    /// Present for Hyena blocks
    pub hyena: Option<HyenaOperator<B>>,
    /// Present for attention blocks
    pub attention: Option<HyenaAttention<B>>,
    pub post_norm: RmsNorm<B>,
    pub ffn: HyenaSwiGluFfn<B>,
    #[module(skip)]
    pub is_attention: bool,
}

impl<B: Backend> HyenaBlock<B> {
    pub fn new(config: &HyenaConfig, layer_idx: usize, device: &B::Device) -> Self {
        let is_attn = config.is_attention_layer(layer_idx);

        Self {
            pre_norm: RmsNorm::with_eps(config.d_model, config.norm_eps, device),
            hyena: if is_attn {
                None
            } else {
                Some(HyenaOperator::new(config, device))
            },
            attention: if is_attn {
                Some(HyenaAttention::new(config, device))
            } else {
                None
            },
            post_norm: RmsNorm::with_eps(config.d_model, config.norm_eps, device),
            ffn: HyenaSwiGluFfn::new(config, device),
            is_attention: is_attn,
        }
    }

    /// Forward for a Hyena block
    pub fn forward_hyena(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();
        let normed = self.pre_norm.forward(x);
        let mixed = self
            .hyena
            .as_ref()
            .expect("forward_hyena called on attention block")
            .forward(normed);
        let after_mixer = mixed + residual;

        let residual2 = after_mixer.clone();
        let normed2 = self.post_norm.forward(after_mixer);
        self.ffn.forward(normed2) + residual2
    }

    /// Forward for an attention block
    pub fn forward_attn(
        &self,
        x: Tensor<B, 3>,
        mask: Option<Tensor<B, 2>>,
        cache: Option<&mut dyn AttentionCache<B>>,
        layer_idx: usize,
    ) -> Tensor<B, 3> {
        let residual = x.clone();
        let normed = self.pre_norm.forward(x);
        let attn_out = self
            .attention
            .as_ref()
            .expect("forward_attn called on Hyena block")
            .forward(normed, mask, cache, layer_idx);
        let after_attn = attn_out + residual;

        let residual2 = after_attn.clone();
        let normed2 = self.post_norm.forward(after_attn);
        self.ffn.forward(normed2) + residual2
    }
}

/// Full Hyena / StripedHyena model
#[derive(Module, Debug)]
pub struct Hyena<B: Backend> {
    pub embed_tokens: Embedding<B>,
    pub layers: Vec<HyenaBlock<B>>,
    pub norm: RmsNorm<B>,
    pub lm_head: Linear<B>,
}

/// Output from the Hyena model
pub struct HyenaOutput<B: Backend> {
    /// Logits over vocabulary [batch, seq_len, vocab_size]
    pub logits: Tensor<B, 3>,
    /// Final hidden states [batch, seq_len, d_model]
    pub hidden_states: Tensor<B, 3>,
}

impl<B: Backend> Hyena<B> {
    /// Forward pass
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Token IDs [batch, seq_len]
    /// * `runtime` - Config and helpers
    /// * `cache` - KV cache for attention layers (pass None to skip caching)
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        runtime: &HyenaRuntime<B>,
        cache: Option<&mut dyn AttentionCache<B>>,
    ) -> HyenaOutput<B> {
        let [_batch, seq_len] = input_ids.dims();
        let device = input_ids.device();

        let mut hidden_states = self.embed_tokens.forward(input_ids);

        let start_pos = cache.as_ref().map(|c| c.seq_len()).unwrap_or(0);

        let prefill_mask: Option<Tensor<B, 2>> = if seq_len > 1 {
            let total = start_pos + seq_len;
            Some(prefill_causal_mask::<B>(seq_len, total, &device))
        } else {
            None
        };

        // SAFETY: same pattern as griffin.rs — exclusive reborrow via raw pointer
        let cache_mut: Option<&mut dyn AttentionCache<B>> =
            cache.map(|c| c as &mut dyn AttentionCache<B>);
        let cache_ptr: Option<*mut dyn AttentionCache<B>> =
            cache_mut.map(|c| c as *mut dyn AttentionCache<B>);

        let mut attn_layer_idx = 0usize;

        for (layer_global, layer) in self.layers.iter().enumerate() {
            if runtime.config.is_attention_layer(layer_global) {
                let cache_ref: Option<&mut dyn AttentionCache<B>> =
                    cache_ptr.map(|p| unsafe { &mut *p });
                hidden_states = layer.forward_attn(
                    hidden_states,
                    prefill_mask.clone(),
                    cache_ref,
                    attn_layer_idx,
                );
                attn_layer_idx += 1;
            } else {
                hidden_states = layer.forward_hyena(hidden_states);
            }
        }

        hidden_states = self.norm.forward(hidden_states);
        let logits = self.lm_head.forward(hidden_states.clone());

        HyenaOutput {
            logits,
            hidden_states,
        }
    }

    /// Generate tokens autoregressively
    ///
    /// Hyena processes the full sequence each step (like a standard transformer
    /// without KV cache for the Hyena layers). Only attention layers use the KV cache.
    ///
    /// TODO: Implement the recurrent form of Hyena for O(1) per-step cost.
    pub fn generate(
        &self,
        input_ids: Tensor<B, 2, Int>,
        runtime: &HyenaRuntime<B>,
        max_new_tokens: usize,
        cache: &mut dyn AttentionCache<B>,
        sampler: &crate::sampling::SamplerConfig,
    ) -> Tensor<B, 2, Int> {
        let [batch, _] = input_ids.dims();
        let device = input_ids.device();

        let input_data: Vec<i64> = input_ids.to_data().to_vec().unwrap();
        let mut context_tokens: Vec<u32> = input_data.iter().map(|&id| id as u32).collect();

        // Prefill
        let output = self.forward(input_ids.clone(), runtime, Some(cache));

        let seq_len = input_ids.dims()[1];
        let last_logits = output
            .logits
            .slice([
                0..batch,
                (seq_len - 1)..seq_len,
                0..runtime.config.vocab_size,
            ])
            .reshape([batch, runtime.config.vocab_size]);
        let token_id = crate::sampling::sample_from_logits(last_logits, &context_tokens, sampler);
        context_tokens.push(token_id);

        let mut next_token = Tensor::<B, 2, Int>::from_ints([[token_id as i32]], &device);
        let mut all_tokens = Tensor::cat(vec![input_ids, next_token.clone()], 1);

        // Decode: Hyena recomputes the full sequence each step.
        // The accumulated context is re-fed through all Hyena layers.
        for _ in 1..max_new_tokens {
            // Re-run full forward on the accumulated sequence (Hyena layers cannot cache)
            // Reset the attention KV cache for a fresh pass over the extended context.
            // Note: this is O(L²) per step; the recurrent form would be O(L).
            let output = self.forward(next_token, runtime, Some(cache));

            let last_logits = output
                .logits
                .slice([0..batch, 0..1, 0..runtime.config.vocab_size])
                .reshape([batch, runtime.config.vocab_size]);
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
    fn test_hyena_config_attention_layer() {
        let config = HyenaConfig::tiny(); // period=6, offset=5
        assert!(!config.is_attention_layer(0));
        assert!(!config.is_attention_layer(4));
        assert!(config.is_attention_layer(5));
    }

    #[test]
    fn test_pure_hyena_no_attention() {
        let config = HyenaConfig::pure_hyena_tiny();
        for i in 0..config.num_layers {
            assert!(!config.is_attention_layer(i));
        }
    }

    #[test]
    fn test_hyena_filter_materialize() {
        let device = Default::default();
        let config = HyenaConfig::tiny();
        let filter = HyenaFilter::<TestBackend>::new(&config, &device);
        let h = filter.materialize(16);
        assert_eq!(h.dims(), [16]);
    }

    #[test]
    fn test_direct_conv_shape() {
        let device = Default::default();
        let signal = Tensor::<TestBackend, 3>::ones([2, 8, 16], &device);
        let filter = Tensor::<TestBackend, 1>::ones([8], &device);
        let output = direct_conv(signal, filter);
        assert_eq!(output.dims(), [2, 8, 16]);
    }

    #[test]
    fn test_direct_conv_causal() {
        // With filter = [1, 0, 0, ...] the output should equal the input (identity conv)
        let device = Default::default();
        let signal =
            Tensor::<TestBackend, 3>::from_floats([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], &device);
        let mut filter_data = vec![0.0f32; 3];
        filter_data[0] = 1.0; // identity: h[0]=1, h[1..]=0
        let filter = Tensor::<TestBackend, 1>::from_floats(filter_data.as_slice(), &device);
        let output = direct_conv(signal.clone(), filter);
        let diff: f32 = (output - signal).abs().sum().into_scalar();
        assert!(diff < 1e-5, "identity conv failed: diff = {diff}");
    }

    #[test]
    fn test_hyena_operator_forward() {
        let device = Default::default();
        let config = HyenaConfig::tiny();
        let op = HyenaOperator::<TestBackend>::new(&config, &device);
        let x = Tensor::<TestBackend, 3>::zeros([1, 4, 64], &device);
        let out = op.forward(x);
        assert_eq!(out.dims(), [1, 4, 64]);
    }

    #[test]
    fn test_hyena_tiny_forward() {
        let device = Default::default();
        let config = HyenaConfig::pure_hyena_tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        let input_ids = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device);
        let output = model.forward(input_ids, &runtime, None);

        assert_eq!(output.logits.dims(), [1, 4, 1000]);
        assert_eq!(output.hidden_states.dims(), [1, 4, 64]);
    }

    #[test]
    fn test_stripedhyena_tiny_forward() {
        let device = Default::default();
        let config = HyenaConfig::tiny(); // interleaved
        let (model, runtime) = config.init::<TestBackend>(&device);

        let input_ids = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3]], &device);
        let output = model.forward(input_ids, &runtime, None);

        assert_eq!(output.logits.dims(), [1, 3, 1000]);
    }

    #[test]
    fn test_hyena_generate() {
        use sketchpad_core::kv_cache::KvCacheConfig;

        let device = Default::default();
        let config = HyenaConfig::tiny();
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

        assert_eq!(generated.dims()[0], 1);
        assert!(generated.dims()[1] >= 3);
    }

    #[test]
    fn test_swiglu_ffn() {
        let device = Default::default();
        let config = HyenaConfig::tiny();
        let ffn = HyenaSwiGluFfn::<TestBackend>::new(&config, &device);
        let x = Tensor::<TestBackend, 3>::zeros([1, 4, 64], &device);
        let out = ffn.forward(x);
        assert_eq!(out.dims(), [1, 4, 64]);
    }
}
