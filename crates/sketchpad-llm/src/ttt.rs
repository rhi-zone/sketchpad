//! TTT: Test-Time Training layers
//!
//! "Learning to (Learn at Test Time)" — each "attention" layer is actually a tiny
//! neural network (the "inner model") whose weights are updated via gradient descent
//! on each new token during inference. The hidden state IS the weights of this inner
//! model, making it a recurrent architecture with learned update rules.
//!
//! TTT-Linear: inner model is a single linear layer W (the hidden state).
//! For each input token:
//! 1. Create a self-supervised task: predict v from k (k,v come from linear projections of x)
//! 2. Compute gradient of loss ||Wk - v||² w.r.t. W
//! 3. Update: W = W - lr * gradient
//! 4. Output: W @ q (apply updated weights to query)
//!
//! The state is O(d²) per layer — the weight matrix W_inner: [head_dim, head_dim].
//!
//! Reference: "Learning to (Learn at Test Time): RNNs with Expressive Hidden States"
//! https://arxiv.org/abs/2407.04620

use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::{Int, activation};

/// TTT model configuration
#[derive(Clone, Debug)]
pub struct TttConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Number of transformer-style layers
    pub num_layers: usize,
    /// Model hidden dimension
    pub d_model: usize,
    /// Number of attention heads (each head has its own W_inner)
    pub num_heads: usize,
    /// Dimension per head (d_model / num_heads)
    pub head_dim: usize,
    /// FFN intermediate size
    pub intermediate_size: usize,
    /// Inner model learning rate (gradient step size, default 1e-3)
    pub ttt_lr: f32,
    /// Use TTT-MLP (2-layer inner model) vs TTT-Linear (default false)
    pub use_mlp: bool,
    /// Inner MLP hidden dim (only used when use_mlp = true, default d_model/4)
    pub mlp_hidden: usize,
    /// Layer norm epsilon
    pub layer_norm_eps: f64,
}

impl TttConfig {
    /// Small TTT model for testing
    pub fn tiny() -> Self {
        Self {
            vocab_size: 1000,
            num_layers: 2,
            d_model: 64,
            num_heads: 4,
            head_dim: 16,
            intermediate_size: 128,
            ttt_lr: 1e-3,
            use_mlp: false,
            mlp_hidden: 16,
            layer_norm_eps: 1e-5,
        }
    }

    /// Initialize the model and runtime
    pub fn init<B: Backend>(&self, device: &B::Device) -> (Ttt<B>, TttRuntime<B>) {
        let layers: Vec<TttBlock<B>> = (0..self.num_layers)
            .map(|_| TttBlock::new(self, device))
            .collect();

        let model = Ttt {
            embed_tokens: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            layers,
            ln_f: LayerNormConfig::new(self.d_model)
                .with_epsilon(self.layer_norm_eps)
                .init(device),
            lm_head: LinearConfig::new(self.d_model, self.vocab_size)
                .with_bias(false)
                .init(device),
        };

        let runtime = TttRuntime {
            config: self.clone(),
            _marker: std::marker::PhantomData,
        };

        (model, runtime)
    }
}

/// Runtime configuration (non-Module data)
pub struct TttRuntime<B: Backend> {
    pub config: TttConfig,
    pub _marker: std::marker::PhantomData<B>,
}

/// Per-layer state: the W_inner matrices for each head
///
/// Shape: [batch, num_heads, head_dim, head_dim]
#[derive(Clone, Debug)]
pub struct TttLayerState<B: Backend> {
    /// Linear inner model weights: [batch, num_heads, head_dim, head_dim]
    pub w_inner: Tensor<B, 4>,
    /// (TTT-MLP only) Second layer weights: [batch, num_heads, head_dim, head_dim]
    pub w_inner2: Option<Tensor<B, 4>>,
}

impl<B: Backend> TttLayerState<B> {
    /// Create a new zero state for TTT-Linear
    pub fn new_linear(batch: usize, num_heads: usize, head_dim: usize, device: &B::Device) -> Self {
        Self {
            w_inner: Tensor::zeros([batch, num_heads, head_dim, head_dim], device),
            w_inner2: None,
        }
    }

    /// Create a new zero state for TTT-MLP
    pub fn new_mlp(
        batch: usize,
        num_heads: usize,
        head_dim: usize,
        mlp_hidden: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            // W1: head_dim -> mlp_hidden
            w_inner: Tensor::zeros([batch, num_heads, mlp_hidden, head_dim], device),
            // W2: mlp_hidden -> head_dim
            w_inner2: Some(Tensor::zeros(
                [batch, num_heads, head_dim, mlp_hidden],
                device,
            )),
        }
    }
}

/// Full TTT inference state — one TttLayerState per layer
pub struct TttState<B: Backend> {
    pub layers: Vec<TttLayerState<B>>,
}

impl<B: Backend> TttState<B> {
    pub fn new(config: &TttConfig, batch: usize, device: &B::Device) -> Self {
        let layers = (0..config.num_layers)
            .map(|_| {
                if config.use_mlp {
                    TttLayerState::new_mlp(
                        batch,
                        config.num_heads,
                        config.head_dim,
                        config.mlp_hidden,
                        device,
                    )
                } else {
                    TttLayerState::new_linear(batch, config.num_heads, config.head_dim, device)
                }
            })
            .collect();
        Self { layers }
    }
}

/// TTT model output
pub struct TttOutput<B: Backend> {
    /// Logits over vocabulary [batch, seq, vocab_size]
    pub logits: Tensor<B, 3>,
    /// Hidden states [batch, seq, d_model]
    pub hidden_states: Tensor<B, 3>,
}

/// Full TTT model
#[derive(Module, Debug)]
pub struct Ttt<B: Backend> {
    /// Token embeddings
    pub embed_tokens: Embedding<B>,
    /// TTT blocks
    pub layers: Vec<TttBlock<B>>,
    /// Final layer norm
    pub ln_f: LayerNorm<B>,
    /// Language model head
    pub lm_head: Linear<B>,
}

impl<B: Backend> Ttt<B> {
    /// Forward pass over a full sequence
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        states: Option<&mut TttState<B>>,
        config: &TttConfig,
    ) -> TttOutput<B> {
        let mut hidden_states = self.embed_tokens.forward(input_ids);

        match states {
            Some(s) => {
                for (layer_idx, layer) in self.layers.iter().enumerate() {
                    hidden_states =
                        layer.forward(hidden_states, Some(&mut s.layers[layer_idx]), config);
                }
            }
            None => {
                for layer in &self.layers {
                    hidden_states = layer.forward(hidden_states, None, config);
                }
            }
        }

        hidden_states = self.ln_f.forward(hidden_states);
        let logits = self.lm_head.forward(hidden_states.clone());

        TttOutput {
            logits,
            hidden_states,
        }
    }

    /// Initialize fresh per-layer states for recurrent inference
    pub fn init_states(
        &self,
        runtime: &TttRuntime<B>,
        batch: usize,
        device: &B::Device,
    ) -> TttState<B> {
        TttState::new(&runtime.config, batch, device)
    }

    /// Generate text autoregressively (purely recurrent — no KV cache)
    pub fn generate(
        &self,
        input_ids: Tensor<B, 2, Int>,
        runtime: &TttRuntime<B>,
        max_new_tokens: usize,
        sampler: &crate::sampling::SamplerConfig,
    ) -> Tensor<B, 2, Int> {
        let [batch, _] = input_ids.dims();
        let device = input_ids.device();

        let input_data: Vec<i64> = input_ids.to_data().to_vec().unwrap();
        let mut context_tokens: Vec<u32> = input_data.iter().map(|&id| id as u32).collect();

        let mut states = self.init_states(runtime, batch, &device);

        // Process prompt
        let output = self.forward(input_ids.clone(), Some(&mut states), &runtime.config);

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

            let output = self.forward(next_token, Some(&mut states), &runtime.config);
            last_logits = output.logits.squeeze_dim::<2>(1);
        }

        all_tokens
    }
}

/// A single TTT transformer block
///
/// Structure: LayerNorm → TTT mechanism → residual; LayerNorm → SwiGLU FFN → residual
#[derive(Module, Debug)]
pub struct TttBlock<B: Backend> {
    /// Pre-norm before TTT mechanism
    pub ln1: LayerNorm<B>,
    /// TTT layer (replaces attention)
    pub ttt: TttLinearLayer<B>,
    /// Pre-norm before FFN
    pub ln2: LayerNorm<B>,
    /// SwiGLU FFN
    pub ffn: TttSwiGluFfn<B>,
}

impl<B: Backend> TttBlock<B> {
    pub fn new(config: &TttConfig, device: &B::Device) -> Self {
        Self {
            ln1: LayerNormConfig::new(config.d_model)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
            ttt: TttLinearLayer::new(config, device),
            ln2: LayerNormConfig::new(config.d_model)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
            ffn: TttSwiGluFfn::new(config, device),
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        state: Option<&mut TttLayerState<B>>,
        config: &TttConfig,
    ) -> Tensor<B, 3> {
        // TTT sub-layer with residual
        let residual = x.clone();
        let normed = self.ln1.forward(x);
        let ttt_out = self.ttt.forward(normed, state, config);
        let x = ttt_out + residual;

        // FFN sub-layer with residual
        let residual = x.clone();
        let normed = self.ln2.forward(x);
        let ffn_out = self.ffn.forward(normed);
        ffn_out + residual
    }
}

/// SwiGLU feed-forward network
#[derive(Module, Debug)]
pub struct TttSwiGluFfn<B: Backend> {
    /// Gate projection: d_model -> intermediate_size
    pub gate_proj: Linear<B>,
    /// Up projection: d_model -> intermediate_size
    pub up_proj: Linear<B>,
    /// Down projection: intermediate_size -> d_model
    pub down_proj: Linear<B>,
}

impl<B: Backend> TttSwiGluFfn<B> {
    pub fn new(config: &TttConfig, device: &B::Device) -> Self {
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

/// Bundled sequence dimensions passed to recurrent scan helpers
struct TttScanDims {
    batch: usize,
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
}

/// TTT-Linear layer: replaces attention with a gradient-updated linear inner model
///
/// For each token, per head:
///   residual = W_inner @ k - v        (forward pass error)
///   grad     = residual ⊗ k           (outer product — gradient of ||W_inner @ k - v||²)
///   W_inner  = W_inner - lr * grad    (gradient step)
///   output   = W_inner @ q            (apply updated weights to query)
///
/// No autograd required — the gradient formula is closed-form.
#[derive(Module, Debug)]
pub struct TttLinearLayer<B: Backend> {
    /// Query projection: d_model -> d_model (num_heads * head_dim)
    pub w_q: Linear<B>,
    /// Key projection: d_model -> d_model
    pub w_k: Linear<B>,
    /// Value projection: d_model -> d_model
    pub w_v: Linear<B>,
    /// Output projection: d_model -> d_model
    pub w_o: Linear<B>,
    /// Learned per-head TTT learning rate scale
    pub lr_scale: burn::module::Param<Tensor<B, 1>>,
    /// Config values (non-module)
    #[module(skip)]
    pub num_heads: usize,
    #[module(skip)]
    pub head_dim: usize,
    #[module(skip)]
    pub use_mlp: bool,
    #[module(skip)]
    pub mlp_hidden: usize,
}

impl<B: Backend> TttLinearLayer<B> {
    pub fn new(config: &TttConfig, device: &B::Device) -> Self {
        let d_model = config.d_model;
        Self {
            w_q: LinearConfig::new(d_model, d_model)
                .with_bias(false)
                .init(device),
            w_k: LinearConfig::new(d_model, d_model)
                .with_bias(false)
                .init(device),
            w_v: LinearConfig::new(d_model, d_model)
                .with_bias(false)
                .init(device),
            w_o: LinearConfig::new(d_model, d_model)
                .with_bias(false)
                .init(device),
            lr_scale: burn::module::Param::from_tensor(Tensor::ones([config.num_heads], device)),
            num_heads: config.num_heads,
            head_dim: config.head_dim,
            use_mlp: config.use_mlp,
            mlp_hidden: config.mlp_hidden,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        state: Option<&mut TttLayerState<B>>,
        config: &TttConfig,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _d_model] = x.dims();
        let num_heads = self.num_heads;
        let head_dim = self.head_dim;
        let ttt_lr = config.ttt_lr;

        // Project to Q, K, V  [batch, seq, d_model]
        let q = self.w_q.forward(x.clone());
        let k = self.w_k.forward(x.clone());
        let v = self.w_v.forward(x);

        // Reshape to [batch, seq, num_heads, head_dim]
        let q = q.reshape([batch, seq_len, num_heads, head_dim]);
        let k = k.reshape([batch, seq_len, num_heads, head_dim]);
        let v = v.reshape([batch, seq_len, num_heads, head_dim]);

        // Transpose to [batch, num_heads, seq, head_dim] for per-head processing
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        // Per-head TTT update
        let dims = TttScanDims {
            batch,
            num_heads,
            head_dim,
            seq_len,
        };
        let output = if let Some(s) = state {
            if self.use_mlp {
                self.ttt_mlp_recurrent(q, k, v, s, ttt_lr, dims)
            } else {
                self.ttt_linear_recurrent(q, k, v, s, ttt_lr, dims)
            }
        } else {
            // Stateless — create a temporary state for this sequence
            let device = q.device();
            if self.use_mlp {
                let mut s =
                    TttLayerState::new_mlp(batch, num_heads, head_dim, self.mlp_hidden, &device);
                self.ttt_mlp_recurrent(q, k, v, &mut s, ttt_lr, dims)
            } else {
                let mut s = TttLayerState::new_linear(batch, num_heads, head_dim, &device);
                self.ttt_linear_recurrent(q, k, v, &mut s, ttt_lr, dims)
            }
        };

        // output: [batch, num_heads, seq, head_dim]
        // Transpose back: [batch, seq, num_heads, head_dim]
        let output = output.swap_dims(1, 2);
        // Reshape: [batch, seq, d_model]
        let output = output.reshape([batch, seq_len, num_heads * head_dim]);
        // Output projection
        self.w_o.forward(output)
    }

    /// TTT-Linear recurrent scan over the sequence
    ///
    /// For each time step t:
    ///   residual = W_inner @ k_t - v_t
    ///   grad     = residual.unsqueeze(-1) @ k_t.unsqueeze(-2)
    ///   W_inner  = W_inner - lr * grad
    ///   out_t    = W_inner @ q_t
    ///
    /// All operations are on the batched, multi-head tensors:
    ///   W_inner: [batch, num_heads, head_dim, head_dim]
    ///   k_t, v_t, q_t: [batch, num_heads, head_dim]
    fn ttt_linear_recurrent(
        &self,
        q: Tensor<B, 4>, // [batch, num_heads, seq, head_dim]
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        state: &mut TttLayerState<B>,
        ttt_lr: f32,
        dims: TttScanDims,
    ) -> Tensor<B, 4> {
        let TttScanDims {
            batch,
            num_heads,
            head_dim,
            seq_len,
        } = dims;
        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            // [batch, num_heads, head_dim]
            let k_t = k
                .clone()
                .slice([0..batch, 0..num_heads, t..t + 1, 0..head_dim])
                .squeeze_dim::<3>(2);
            let v_t = v
                .clone()
                .slice([0..batch, 0..num_heads, t..t + 1, 0..head_dim])
                .squeeze_dim::<3>(2);
            let q_t = q
                .clone()
                .slice([0..batch, 0..num_heads, t..t + 1, 0..head_dim])
                .squeeze_dim::<3>(2);

            // residual = W_inner @ k_t - v_t
            // W_inner: [batch, num_heads, head_dim, head_dim]
            // k_t unsqueezed: [batch, num_heads, head_dim, 1]
            let k_t_col = k_t.clone().unsqueeze_dim::<4>(3);
            // matmul: [batch, num_heads, head_dim, 1]
            let pred = state.w_inner.clone().matmul(k_t_col).squeeze_dim::<3>(3);
            // residual: [batch, num_heads, head_dim]
            let residual = pred - v_t;

            // grad = residual ⊗ k_t (outer product)
            // residual: [batch, num_heads, head_dim, 1]
            // k_t: [batch, num_heads, 1, head_dim]
            let res_col = residual.unsqueeze_dim::<4>(3);
            let k_t_row = k_t.unsqueeze_dim::<4>(2);
            // grad: [batch, num_heads, head_dim, head_dim]
            let grad = res_col.matmul(k_t_row);

            // W_inner = W_inner - lr * grad
            state.w_inner = state.w_inner.clone() - grad * ttt_lr;

            // output_t = W_inner @ q_t
            let q_t_col = q_t.unsqueeze_dim::<4>(3);
            let out_t = state.w_inner.clone().matmul(q_t_col).squeeze_dim::<3>(3);
            // out_t: [batch, num_heads, head_dim] -> unsqueeze seq dim
            outputs.push(out_t.unsqueeze_dim::<4>(2));
        }

        // [batch, num_heads, seq, head_dim]
        Tensor::cat(outputs, 2)
    }

    /// TTT-MLP recurrent scan: inner model is W2(gelu(W1 @ k)) → predict v
    fn ttt_mlp_recurrent(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        state: &mut TttLayerState<B>,
        ttt_lr: f32,
        dims: TttScanDims,
    ) -> Tensor<B, 4> {
        let TttScanDims {
            batch,
            num_heads,
            head_dim,
            seq_len,
        } = dims;
        // state.w_inner:  [batch, num_heads, mlp_hidden, head_dim]  (W1)
        // state.w_inner2: [batch, num_heads, head_dim,   mlp_hidden] (W2)
        let mut w2 = state.w_inner2.as_ref().unwrap().clone();

        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let k_t = k
                .clone()
                .slice([0..batch, 0..num_heads, t..t + 1, 0..head_dim])
                .squeeze_dim::<3>(2);
            let v_t = v
                .clone()
                .slice([0..batch, 0..num_heads, t..t + 1, 0..head_dim])
                .squeeze_dim::<3>(2);
            let q_t = q
                .clone()
                .slice([0..batch, 0..num_heads, t..t + 1, 0..head_dim])
                .squeeze_dim::<3>(2);

            // Forward: h = gelu(W1 @ k_t), pred = W2 @ h
            let k_col = k_t.clone().unsqueeze_dim::<4>(3); // [batch, num_heads, head_dim, 1]
            // W1 @ k_t: [batch, num_heads, mlp_hidden, 1]
            let h = state.w_inner.clone().matmul(k_col);
            let h = activation::gelu(h);
            // W2 @ h: [batch, num_heads, head_dim, 1]
            let pred = w2.clone().matmul(h.clone()).squeeze_dim::<3>(3);
            // residual: [batch, num_heads, head_dim]
            let residual = pred - v_t;

            // Gradient for W2: d/dW2 ||W2 @ h - v||² = residual ⊗ h
            let res_col = residual.clone().unsqueeze_dim::<4>(3); // [batch, num_heads, head_dim, 1]
            let h_squeezed = h.squeeze_dim::<3>(3); // [batch, num_heads, mlp_hidden]
            let h_row = h_squeezed.clone().unsqueeze_dim::<4>(2); // [batch, num_heads, 1, mlp_hidden]
            let grad_w2 = res_col.clone().matmul(h_row); // [batch, num_heads, head_dim, mlp_hidden]
            w2 = w2 - grad_w2 * ttt_lr;

            // Gradient for W1 (backprop through gelu): simplified — use gelu(x) ≈ x, grad ≈ W2^T @ residual ⊗ k
            // Full gradient: d/dW1 = (W2^T @ residual * gelu'(W1@k)) ⊗ k
            // Approximate with gelu'(x) ≈ 1 for stability:
            let w2_t = w2.clone().swap_dims(2, 3); // [batch, num_heads, mlp_hidden, head_dim]
            let delta_h = w2_t.matmul(res_col); // [batch, num_heads, mlp_hidden, 1]
            let k_row = k_t.clone().unsqueeze_dim::<4>(2); // [batch, num_heads, 1, head_dim]
            let grad_w1 = delta_h.matmul(k_row); // [batch, num_heads, mlp_hidden, head_dim]
            state.w_inner = state.w_inner.clone() - grad_w1 * ttt_lr;

            // Output: W2 @ gelu(W1 @ q_t)
            let q_col = q_t.unsqueeze_dim::<4>(3);
            let h_q = activation::gelu(state.w_inner.clone().matmul(q_col));
            let out_t = w2.clone().matmul(h_q).squeeze_dim::<3>(3);
            outputs.push(out_t.unsqueeze_dim::<4>(2));
        }

        // Write w2 back to state
        state.w_inner2 = Some(w2);

        Tensor::cat(outputs, 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_ttt_config() {
        let config = TttConfig::tiny();
        assert_eq!(config.num_heads * config.head_dim, config.d_model);
    }

    #[test]
    fn test_ttt_tiny_forward() {
        let device = Default::default();
        let config = TttConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        let input_ids = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device);
        let output = model.forward(input_ids, None, &runtime.config);

        assert_eq!(output.logits.dims(), [1, 4, 1000]);
        assert_eq!(output.hidden_states.dims(), [1, 4, 64]);
    }

    #[test]
    fn test_ttt_with_state() {
        let device = Default::default();
        let config = TttConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        let mut states = model.init_states(&runtime, 1, &device);

        // First forward pass
        let input1 = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device);
        let _ = model.forward(input1, Some(&mut states), &runtime.config);

        // Second forward pass (incremental)
        let input2 = Tensor::<TestBackend, 2, Int>::from_ints([[3]], &device);
        let output = model.forward(input2, Some(&mut states), &runtime.config);

        assert_eq!(output.logits.dims(), [1, 1, 1000]);
    }

    #[test]
    fn test_ttt_generate() {
        let device = Default::default();
        let config = TttConfig::tiny();
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
    fn test_ttt_block() {
        let device = Default::default();
        let config = TttConfig::tiny();
        let block = TttBlock::new(&config, &device);

        let x = Tensor::<TestBackend, 3>::zeros([1, 4, 64], &device);
        let output = block.forward(x, None, &config);

        assert_eq!(output.dims(), [1, 4, 64]);
    }

    #[test]
    fn test_ttt_swiglu_ffn() {
        let device = Default::default();
        let config = TttConfig::tiny();
        let ffn = TttSwiGluFfn::new(&config, &device);

        let x = Tensor::<TestBackend, 3>::zeros([1, 4, 64], &device);
        let output = ffn.forward(x);
        assert_eq!(output.dims(), [1, 4, 64]);
    }

    #[test]
    fn test_ttt_mlp_forward() {
        let device = Default::default();
        let mut config = TttConfig::tiny();
        config.use_mlp = true;
        let (model, runtime) = config.init::<TestBackend>(&device);

        let input_ids = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3]], &device);
        let output = model.forward(input_ids, None, &runtime.config);

        assert_eq!(output.logits.dims(), [1, 3, 1000]);
    }

    #[test]
    fn test_ttt_state_shape() {
        let device = Default::default();
        let config = TttConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);
        let states = model.init_states(&runtime, 2, &device);

        assert_eq!(states.layers.len(), config.num_layers);
        // Each layer state: [batch=2, num_heads=4, head_dim=16, head_dim=16]
        assert_eq!(states.layers[0].w_inner.dims(), [2, 4, 16, 16]);
        assert!(states.layers[0].w_inner2.is_none());
    }
}
