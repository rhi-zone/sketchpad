//! Griffin / Hawk (RecurrentGemma) Model
//!
//! Griffin is Google DeepMind's hybrid architecture combining:
//! - RG-LRU (Real-Gated Linear Recurrent Unit) layers for O(1) recurrent inference
//! - Local sliding-window attention layers for global context (optional)
//! - GEGLU FFN in each block
//!
//! Hawk is the recurrence-only variant — Griffin with no attention layers.
//!
//! Key properties:
//! - Recurrent state is [batch, d_model] (not sequence-indexed)
//! - Attention layers use a local sliding window (default 2048 tokens)
//! - Interleaving: configurable, typically 1 attention per 6 recurrent layers
//!
//! Reference: "Griffin: Mixing Gated Linear Recurrences with Local Attention"
//! https://arxiv.org/abs/2402.19427

use burn::module::{Module, Param};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::{Int, activation};

use sketchpad_core::kv_cache::{AttentionCache, KvCacheConfig, ModelKvCache};
use sketchpad_core::rmsnorm::RmsNorm;
use sketchpad_core::transformer::sliding_window_mask;

/// Griffin / Hawk model configuration
#[derive(Clone, Debug)]
pub struct GriffinConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Number of transformer/recurrent blocks
    pub num_layers: usize,
    /// Hidden dimension (d_model)
    pub d_model: usize,
    /// RG-LRU expansion dimension (typically d_model * 3/2 or equal to d_model)
    pub d_conv: usize,
    /// Number of attention heads (for attention layers)
    pub num_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Local attention window size
    pub window_size: usize,
    /// FFN intermediate size
    pub intermediate_size: usize,
    /// Offset: first layer index that is an attention layer (within the period)
    pub attn_layer_offset: usize,
    /// Period: every Nth layer is an attention layer (0 = Hawk, no attention)
    pub attn_layer_period: usize,
    /// RMSNorm epsilon
    pub norm_eps: f64,
}

impl GriffinConfig {
    /// RecurrentGemma 2B configuration (Griffin)
    pub fn recurrent_gemma_2b() -> Self {
        Self {
            vocab_size: 256000,
            num_layers: 26,
            d_model: 2560,
            d_conv: 2560,
            num_heads: 10,
            head_dim: 256,
            window_size: 2048,
            intermediate_size: 7680,
            attn_layer_offset: 5,
            attn_layer_period: 6, // every 6th layer is attention (layers 5, 11, 17, 23)
            norm_eps: 1e-6,
        }
    }

    /// RecurrentGemma 9B configuration (Griffin)
    pub fn recurrent_gemma_9b() -> Self {
        Self {
            vocab_size: 256000,
            num_layers: 46,
            d_model: 4096,
            d_conv: 4096,
            num_heads: 16,
            head_dim: 256,
            window_size: 2048,
            intermediate_size: 16384,
            attn_layer_offset: 5,
            attn_layer_period: 6,
            norm_eps: 1e-6,
        }
    }

    /// Hawk 2B (recurrence only, no attention)
    pub fn hawk_2b() -> Self {
        Self {
            vocab_size: 256000,
            num_layers: 26,
            d_model: 2560,
            d_conv: 2560,
            num_heads: 10,
            head_dim: 256,
            window_size: 2048,
            intermediate_size: 7680,
            attn_layer_offset: 0,
            attn_layer_period: 0, // 0 = Hawk (no attention layers)
            norm_eps: 1e-6,
        }
    }

    /// Tiny configuration for testing
    pub fn tiny() -> Self {
        Self {
            vocab_size: 1000,
            num_layers: 6,
            d_model: 64,
            d_conv: 64,
            num_heads: 4,
            head_dim: 16,
            window_size: 8,
            intermediate_size: 128,
            attn_layer_offset: 5,
            attn_layer_period: 6, // only layer 5 is attention
            norm_eps: 1e-6,
        }
    }

    /// Returns true if the given layer index is an attention layer
    pub fn is_attention_layer(&self, layer_idx: usize) -> bool {
        if self.attn_layer_period == 0 {
            return false; // Hawk: no attention
        }
        layer_idx % self.attn_layer_period == self.attn_layer_offset % self.attn_layer_period
    }

    /// Initialize the model and runtime
    pub fn init<B: Backend>(&self, device: &B::Device) -> (Griffin<B>, GriffinRuntime<B>) {
        let layers: Vec<GriffinBlock<B>> = (0..self.num_layers)
            .map(|i| GriffinBlock::new(self, i, device))
            .collect();

        let model = Griffin {
            embed_tokens: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            layers,
            norm: RmsNorm::with_eps(self.d_model, self.norm_eps, device),
            lm_head: LinearConfig::new(self.d_model, self.vocab_size)
                .with_bias(false)
                .init(device),
        };

        let runtime = GriffinRuntime {
            config: self.clone(),
            _marker: std::marker::PhantomData,
        };

        (model, runtime)
    }
}

/// Runtime data for Griffin (non-Module, lives outside the Module tree)
pub struct GriffinRuntime<B: Backend> {
    pub config: GriffinConfig,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> GriffinRuntime<B> {
    /// Create a new runtime from a config
    pub fn new(config: GriffinConfig) -> Self {
        Self {
            config,
            _marker: std::marker::PhantomData,
        }
    }

    /// Count the number of attention layers
    pub fn num_attention_layers(&self) -> usize {
        (0..self.config.num_layers)
            .filter(|&i| self.config.is_attention_layer(i))
            .count()
    }

    /// Create a KV cache for the attention layers
    pub fn create_kv_cache(&self, cache_config: &KvCacheConfig) -> Box<dyn AttentionCache<B>> {
        match cache_config {
            KvCacheConfig::Standard => Box::new(ModelKvCache::<B>::new(
                self.config.num_layers,
                self.config.window_size * 2, // conservative upper bound
            )),
            KvCacheConfig::Compressed { .. } => {
                // Fall back to standard for Griffin (compression less meaningful with windows)
                Box::new(ModelKvCache::<B>::new(
                    self.config.num_layers,
                    self.config.window_size * 2,
                ))
            }
        }
    }
}

/// Mutable recurrent state for Griffin inference
pub struct GriffinState<B: Backend> {
    /// Per-recurrent-layer RG-LRU hidden state [batch, d_model]
    pub rg_lru_states: Vec<Tensor<B, 2>>,
}

impl<B: Backend> GriffinState<B> {
    /// Create zero state for all recurrent layers
    pub fn new(config: &GriffinConfig, batch: usize, device: &B::Device) -> Self {
        let rg_lru_states = (0..config.num_layers)
            .filter(|&i| !config.is_attention_layer(i))
            .map(|_| Tensor::zeros([batch, config.d_model], device))
            .collect();
        Self { rg_lru_states }
    }
}

/// Griffin model output
pub struct GriffinOutput<B: Backend> {
    /// Logits over vocabulary [batch, seq_len, vocab_size]
    pub logits: Tensor<B, 3>,
    /// Final hidden states [batch, seq_len, d_model]
    pub hidden_states: Tensor<B, 3>,
}

/// Full Griffin / Hawk model
#[derive(Module, Debug)]
pub struct Griffin<B: Backend> {
    /// Token embeddings
    pub embed_tokens: Embedding<B>,
    /// Blocks (RG-LRU or local attention, each with GEGLU FFN)
    pub layers: Vec<GriffinBlock<B>>,
    /// Final RMSNorm
    pub norm: RmsNorm<B>,
    /// LM head
    pub lm_head: Linear<B>,
}

impl<B: Backend> Griffin<B> {
    /// Forward pass
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Token IDs [batch, seq_len]
    /// * `runtime` - Config + helpers
    /// * `rg_lru_states` - Mutable recurrent states indexed by recurrent-layer count
    /// * `cache` - KV cache for attention layers (required if any attention layers exist)
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        runtime: &GriffinRuntime<B>,
        mut rg_lru_states: Option<&mut Vec<Tensor<B, 2>>>,
        cache: Option<&mut dyn AttentionCache<B>>,
    ) -> GriffinOutput<B> {
        let [_batch, seq_len] = input_ids.dims();
        let device = input_ids.device();

        let mut hidden_states = self.embed_tokens.forward(input_ids);

        // Compute start position before we take a mutable borrow on the cache
        let start_pos = cache.as_ref().map(|c| c.seq_len()).unwrap_or(0);

        // Prefill mask for attention layers (causal over [start_pos .. start_pos+seq_len])
        let prefill_mask: Option<Tensor<B, 2>> = if seq_len > 1 {
            let total = start_pos + seq_len;
            Some(prefill_causal_mask::<B>(seq_len, total, &device))
        } else {
            None
        };

        // Take the mutable borrow once so we can use it in the loop
        let cache_mut: Option<&mut dyn AttentionCache<B>> =
            cache.map(|c| c as &mut dyn AttentionCache<B>);
        // SAFETY: we reborrow via raw pointer to avoid the borrow checker rejecting
        // the repeated mutable access inside the loop.  We ensure exclusivity by
        // only using `cache_ptr` inside this function and never aliasing.
        let cache_ptr: Option<*mut dyn AttentionCache<B>> =
            cache_mut.map(|c| c as *mut dyn AttentionCache<B>);

        let mut rg_lru_idx = 0usize;
        let mut attn_layer_idx = 0usize;

        for (layer_global, layer) in self.layers.iter().enumerate() {
            if runtime.config.is_attention_layer(layer_global) {
                // SAFETY: we hold the only reference via cache_ptr and no other code
                // aliases it during this loop body.
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
                let state = rg_lru_states.as_deref_mut().map(|s| &mut s[rg_lru_idx]);
                hidden_states = layer.forward_recurrent(hidden_states, state);
                rg_lru_idx += 1;
            }
        }

        hidden_states = self.norm.forward(hidden_states);
        let logits = self.lm_head.forward(hidden_states.clone());

        GriffinOutput {
            logits,
            hidden_states,
        }
    }

    /// Generate tokens autoregressively
    ///
    /// Uses KV cache for attention layers, recurrent state for RG-LRU layers.
    pub fn generate(
        &self,
        input_ids: Tensor<B, 2, Int>,
        runtime: &GriffinRuntime<B>,
        max_new_tokens: usize,
        cache: &mut dyn AttentionCache<B>,
        sampler: &crate::sampling::SamplerConfig,
    ) -> Tensor<B, 2, Int> {
        let [batch, _] = input_ids.dims();
        let device = input_ids.device();

        let input_data: Vec<i64> = input_ids.to_data().to_vec().unwrap();
        let mut context_tokens: Vec<u32> = input_data.iter().map(|&id| id as u32).collect();

        // Initialize recurrent states
        let num_recurrent = runtime.config.num_layers
            - (0..runtime.config.num_layers)
                .filter(|&i| runtime.config.is_attention_layer(i))
                .count();
        let mut rg_lru_states: Vec<Tensor<B, 2>> = (0..num_recurrent)
            .map(|_| Tensor::zeros([batch, runtime.config.d_model], &device))
            .collect();

        // Prefill
        let output = self.forward(
            input_ids.clone(),
            runtime,
            Some(&mut rg_lru_states),
            Some(cache),
        );

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

        // Decode
        for _ in 1..max_new_tokens {
            let output = self.forward(next_token, runtime, Some(&mut rg_lru_states), Some(cache));

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

/// A single Griffin block
///
/// Two variants via dispatch (not an enum to keep Module derive simple):
/// - Recurrent: RMSNorm → RG-LRU → RMSNorm → GEGLU FFN
/// - Attention:  RMSNorm → LocalAttn → RMSNorm → GEGLU FFN
#[derive(Module, Debug)]
pub struct GriffinBlock<B: Backend> {
    /// Pre-norm before the mixer
    pub pre_norm: RmsNorm<B>,
    /// RG-LRU mixer (present for recurrent blocks, stub for attention)
    pub rg_lru: Option<GriffinRgLru<B>>,
    /// Local attention (present for attention blocks, stub for recurrent)
    pub attention: Option<GriffinLocalAttn<B>>,
    /// Pre-norm before FFN
    pub post_norm: RmsNorm<B>,
    /// GEGLU feed-forward network
    pub ffn: GriffinGeGluFfn<B>,
    /// Whether this is an attention block (for dispatch)
    #[module(skip)]
    pub is_attention: bool,
}

impl<B: Backend> GriffinBlock<B> {
    pub fn new(config: &GriffinConfig, layer_idx: usize, device: &B::Device) -> Self {
        let is_attn = config.is_attention_layer(layer_idx);

        let rg_lru = if is_attn {
            None
        } else {
            Some(GriffinRgLru::new(config, device))
        };

        let attention = if is_attn {
            Some(GriffinLocalAttn::new(config, device))
        } else {
            None
        };

        Self {
            pre_norm: RmsNorm::with_eps(config.d_model, config.norm_eps, device),
            rg_lru,
            attention,
            post_norm: RmsNorm::with_eps(config.d_model, config.norm_eps, device),
            ffn: GriffinGeGluFfn::new(config, device),
            is_attention: is_attn,
        }
    }

    /// Forward for recurrent block
    pub fn forward_recurrent(
        &self,
        x: Tensor<B, 3>,
        state: Option<&mut Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        let residual = x.clone();
        let normed = self.pre_norm.forward(x);
        let mixed = self
            .rg_lru
            .as_ref()
            .expect("forward_recurrent called on attention block")
            .forward(normed, state);
        let after_mixer = mixed + residual;

        let residual2 = after_mixer.clone();
        let normed2 = self.post_norm.forward(after_mixer);
        self.ffn.forward(normed2) + residual2
    }

    /// Forward for attention block
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
            .expect("forward_attn called on recurrent block")
            .forward(normed, mask, cache, layer_idx);
        let after_attn = attn_out + residual;

        let residual2 = after_attn.clone();
        let normed2 = self.post_norm.forward(after_attn);
        self.ffn.forward(normed2) + residual2
    }
}

/// Real-Gated Linear Recurrent Unit (RG-LRU)
///
/// Recurrence equations:
/// ```text
/// input_gate  = sigmoid(W_i * x)
/// recur_gate  = sigmoid(W_r * x)
/// log_a       = -8 * softplus(recur_gate)   → log decay in [-8, 0]
/// a           = exp(log_a)                  → decay in (0, 1)
/// h           = a * h_prev + sqrt(1 - a²) * (input_gate * x)
/// output      = h * sigmoid(W_out * x)
/// ```
#[derive(Module, Debug)]
pub struct GriffinRgLru<B: Backend> {
    /// Projects x → input gate logits [d_model → d_model]
    pub input_gate_proj: Linear<B>,
    /// Projects x → recurrence gate logits [d_model → d_model]
    pub recurrence_gate_proj: Linear<B>,
    /// Projects x → output gate logits [d_model → d_model]
    pub output_gate_proj: Linear<B>,
    /// Scalar A parameter (log form), one per channel [d_model]
    pub a_log: Param<Tensor<B, 1>>,
}

impl<B: Backend> GriffinRgLru<B> {
    pub fn new(config: &GriffinConfig, device: &B::Device) -> Self {
        let d = config.d_model;

        // Initialize a_log to small negative values so initial decay ≈ 0.9
        let a_log_init = Tensor::<B, 1>::zeros([d], device) - 0.5f32;

        Self {
            input_gate_proj: LinearConfig::new(d, d).with_bias(true).init(device),
            recurrence_gate_proj: LinearConfig::new(d, d).with_bias(true).init(device),
            output_gate_proj: LinearConfig::new(d, d).with_bias(true).init(device),
            a_log: Param::from_tensor(a_log_init),
        }
    }

    /// Forward pass over a sequence, updating hidden state in place
    ///
    /// # Arguments
    ///
    /// * `x` - Input [batch, seq_len, d_model]
    /// * `h_state` - Optional mutable hidden state [batch, d_model]; updated in-place
    pub fn forward(&self, x: Tensor<B, 3>, mut h_state: Option<&mut Tensor<B, 2>>) -> Tensor<B, 3> {
        let [batch, seq_len, d_model] = x.dims();
        let device = x.device();

        // Gate projections
        let input_gate = activation::sigmoid(self.input_gate_proj.forward(x.clone()));
        let recur_gate = activation::sigmoid(self.recurrence_gate_proj.forward(x.clone()));
        let out_gate_logits = self.output_gate_proj.forward(x.clone());

        // Compute per-channel log decay from recurrence gate
        // log_a ∈ [-8, 0], a ∈ (exp(-8), 1)
        let log_a = activation::softplus(recur_gate, 1.0) * (-8.0f32);
        let a = log_a.exp(); // [batch, seq_len, d_model]

        // Variance-preserving normalization factor: sqrt(1 - a^2)
        let one: Tensor<B, 3> = Tensor::ones([batch, seq_len, d_model], &device);
        let norm_factor = (one - a.clone() * a.clone()).sqrt();

        // Gated input
        let gated_x = input_gate * x; // [batch, seq_len, d_model]

        // Initialize hidden state
        let mut h: Tensor<B, 2> = match &h_state {
            Some(s) => (*s).clone(),
            None => Tensor::zeros([batch, d_model], &device),
        };

        // Recurrent loop over sequence
        let mut outputs = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let a_t = a
                .clone()
                .slice([0..batch, t..t + 1, 0..d_model])
                .squeeze_dim::<2>(1); // [batch, d_model]
            let norm_t = norm_factor
                .clone()
                .slice([0..batch, t..t + 1, 0..d_model])
                .squeeze_dim::<2>(1);
            let gx_t = gated_x
                .clone()
                .slice([0..batch, t..t + 1, 0..d_model])
                .squeeze_dim::<2>(1);

            h = a_t * h + norm_t * gx_t;
            outputs.push(h.clone().unsqueeze_dim::<3>(1));
        }

        // Update caller's hidden state
        if let Some(s) = h_state.as_mut() {
            **s = h;
        }

        let h_seq = Tensor::cat(outputs, 1); // [batch, seq_len, d_model]

        // Output gate
        let out_gate = activation::sigmoid(out_gate_logits);
        h_seq * out_gate
    }
}

/// Local (sliding-window) self-attention for Griffin
#[derive(Module, Debug)]
pub struct GriffinLocalAttn<B: Backend> {
    /// Query projection
    pub q_proj: Linear<B>,
    /// Key projection
    pub k_proj: Linear<B>,
    /// Value projection
    pub v_proj: Linear<B>,
    /// Output projection
    pub o_proj: Linear<B>,
    #[module(skip)]
    pub num_heads: usize,
    #[module(skip)]
    pub head_dim: usize,
    #[module(skip)]
    pub window_size: usize,
}

impl<B: Backend> GriffinLocalAttn<B> {
    pub fn new(config: &GriffinConfig, device: &B::Device) -> Self {
        let hidden = config.d_model;
        let kv_dim = config.num_heads * config.head_dim;

        Self {
            q_proj: LinearConfig::new(hidden, kv_dim)
                .with_bias(false)
                .init(device),
            k_proj: LinearConfig::new(hidden, kv_dim)
                .with_bias(false)
                .init(device),
            v_proj: LinearConfig::new(hidden, kv_dim)
                .with_bias(false)
                .init(device),
            o_proj: LinearConfig::new(kv_dim, hidden)
                .with_bias(false)
                .init(device),
            num_heads: config.num_heads,
            head_dim: config.head_dim,
            window_size: config.window_size,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        mask: Option<Tensor<B, 2>>,
        cache: Option<&mut dyn AttentionCache<B>>,
        layer_idx: usize,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden] = x.dims();
        let device = x.device();

        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        // Reshape: [batch, seq, num_heads * head_dim] → [batch, num_heads, seq, head_dim]
        let q = q
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);

        // Optionally update KV cache
        let (k, v) = if let Some(c) = cache {
            let update = c.update_layer(layer_idx, k, v);
            (update.k, update.v)
        } else {
            (k, v)
        };

        let kv_seq = k.dims()[2];

        // Attention: scale dot-product
        let scale = (self.head_dim as f64).powf(-0.5);
        let attn = q.matmul(k.transpose()) * scale; // [batch, heads, q_seq, k_seq]

        // Apply mask
        let attn = if let Some(m) = mask {
            // m is [q_seq, kv_seq] — add head/batch dims
            attn + m.unsqueeze::<3>().unsqueeze()
        } else if seq_len == 1 && kv_seq > 1 {
            // Single-token decode with cache: no masking needed
            attn
        } else {
            // Apply sliding window mask for prefill without cache
            let sw_mask = sliding_window_mask::<B>(seq_len, self.window_size, &device);
            attn + sw_mask.unsqueeze::<3>().unsqueeze()
        };

        let attn = burn::tensor::activation::softmax(attn, 3);
        let out = attn.matmul(v); // [batch, heads, seq, head_dim]

        // Reshape back: [batch, seq, num_heads * head_dim]
        let out = out
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * self.head_dim]);

        self.o_proj.forward(out)
    }
}

/// GEGLU Feed-Forward Network
///
/// out = down_proj(GELU(gate_proj(x)) * up_proj(x))
#[derive(Module, Debug)]
pub struct GriffinGeGluFfn<B: Backend> {
    /// Gate projection [d_model → intermediate_size]
    pub gate_proj: Linear<B>,
    /// Up projection [d_model → intermediate_size]
    pub up_proj: Linear<B>,
    /// Down projection [intermediate_size → d_model]
    pub down_proj: Linear<B>,
}

impl<B: Backend> GriffinGeGluFfn<B> {
    pub fn new(config: &GriffinConfig, device: &B::Device) -> Self {
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
        let gate = activation::gelu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
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
    fn test_griffin_config() {
        let _ = GriffinConfig::recurrent_gemma_2b();
        let _ = GriffinConfig::recurrent_gemma_9b();
        let _ = GriffinConfig::hawk_2b();
    }

    #[test]
    fn test_is_attention_layer() {
        let config = GriffinConfig::tiny(); // period=6, offset=5
        assert!(!config.is_attention_layer(0));
        assert!(!config.is_attention_layer(4));
        assert!(config.is_attention_layer(5));
        assert!(!config.is_attention_layer(6)); // would be next period but only 6 layers

        let hawk = GriffinConfig::hawk_2b();
        for i in 0..hawk.num_layers {
            assert!(!hawk.is_attention_layer(i));
        }
    }

    #[test]
    fn test_griffin_tiny_forward() {
        let device = Default::default();
        let config = GriffinConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        let input_ids = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device);
        let output = model.forward(input_ids, &runtime, None, None);

        assert_eq!(output.logits.dims(), [1, 4, 1000]);
        assert_eq!(output.hidden_states.dims(), [1, 4, 64]);
    }

    #[test]
    fn test_griffin_with_state() {
        let device = Default::default();
        let config = GriffinConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        let batch = 1;
        let num_recurrent = config.num_layers
            - (0..config.num_layers)
                .filter(|&i| config.is_attention_layer(i))
                .count();
        let mut states: Vec<Tensor<TestBackend, 2>> = (0..num_recurrent)
            .map(|_| Tensor::zeros([batch, config.d_model], &device))
            .collect();

        let input1 = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device);
        let _ = model.forward(input1, &runtime, Some(&mut states), None);

        let input2 = Tensor::<TestBackend, 2, Int>::from_ints([[3]], &device);
        let output = model.forward(input2, &runtime, Some(&mut states), None);

        assert_eq!(output.logits.dims(), [1, 1, 1000]);
    }

    #[test]
    fn test_rg_lru() {
        let device = Default::default();
        let config = GriffinConfig::tiny();
        let lru = GriffinRgLru::new(&config, &device);

        let x = Tensor::<TestBackend, 3>::zeros([1, 4, 64], &device);
        let output = lru.forward(x, None);
        assert_eq!(output.dims(), [1, 4, 64]);
    }

    #[test]
    fn test_rg_lru_state_update() {
        let device = Default::default();
        let config = GriffinConfig::tiny();
        let lru = GriffinRgLru::new(&config, &device);

        let mut h = Tensor::<TestBackend, 2>::zeros([1, 64], &device);
        let x = Tensor::<TestBackend, 3>::ones([1, 2, 64], &device);
        let _ = lru.forward(x, Some(&mut h));
        // State should have been updated (non-zero after processing ones)
        let h_sum: f32 = h.sum().into_scalar();
        assert!(h_sum.abs() > 0.0);
    }

    #[test]
    fn test_geglu_ffn() {
        let device = Default::default();
        let config = GriffinConfig::tiny();
        let ffn = GriffinGeGluFfn::new(&config, &device);

        let x = Tensor::<TestBackend, 3>::zeros([1, 4, 64], &device);
        let output = ffn.forward(x);
        assert_eq!(output.dims(), [1, 4, 64]);
    }

    #[test]
    fn test_hawk_tiny_forward() {
        let device = Default::default();
        let mut config = GriffinConfig::tiny();
        config.attn_layer_period = 0; // Hawk: no attention
        let (model, runtime) = config.init::<TestBackend>(&device);

        let input_ids = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3]], &device);
        let output = model.forward(input_ids, &runtime, None, None);
        assert_eq!(output.logits.dims(), [1, 3, 1000]);
    }

    #[test]
    fn test_griffin_generate() {
        use sketchpad_core::kv_cache::KvCacheConfig;

        let device = Default::default();
        let config = GriffinConfig::tiny();
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
}
