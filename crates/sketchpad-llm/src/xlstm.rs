//! xLSTM: Extended Long Short-Term Memory
//!
//! Two new LSTM cell variants:
//! - **sLSTM** (scalar LSTM): exponential input/forget gates with normalizer state
//! - **mLSTM** (matrix LSTM): covariance-style matrix memory, much higher capacity
//!
//! Reference: "xLSTM: Extended Long Short-Term Memory" (Beck et al., 2024)
//! https://arxiv.org/abs/2405.04517

use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::{Int, activation};

/// xLSTM model configuration
#[derive(Clone, Debug)]
pub struct XLstmConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Model dimension
    pub d_model: usize,
    /// Expansion dimension (typically 4 * d_model for mLSTM)
    pub d_inner: usize,
    /// Number of heads for mLSTM
    pub num_heads: usize,
    /// Per-head key/query dimension
    pub qk_dim: usize,
    /// Per-head value dimension
    pub v_dim: usize,
    /// Which layer indices use mLSTM; all others use sLSTM. Empty = all sLSTM.
    pub mlstm_at_layers: Vec<usize>,
    /// Layer norm epsilon
    pub layer_norm_eps: f64,
}

impl XLstmConfig {
    /// Small configuration for testing
    pub fn tiny() -> Self {
        Self {
            vocab_size: 1000,
            num_layers: 2,
            d_model: 64,
            d_inner: 128,
            num_heads: 4,
            qk_dim: 16,
            v_dim: 16,
            mlstm_at_layers: vec![1],
            layer_norm_eps: 1e-5,
        }
    }

    /// Initialize model and runtime
    pub fn init<B: Backend>(&self, device: &B::Device) -> (XLstm<B>, XLstmRuntime<B>) {
        let layers: Vec<XLstmBlock<B>> = (0..self.num_layers)
            .map(|layer_idx| XLstmBlock::new(self, layer_idx, device))
            .collect();

        let model = XLstm {
            embed_tokens: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            layers,
            ln_f: LayerNormConfig::new(self.d_model)
                .with_epsilon(self.layer_norm_eps)
                .init(device),
            lm_head: LinearConfig::new(self.d_model, self.vocab_size)
                .with_bias(false)
                .init(device),
        };

        let runtime = XLstmRuntime {
            config: self.clone(),
            _marker: std::marker::PhantomData,
        };

        (model, runtime)
    }
}

/// Runtime configuration (non-Module data)
pub struct XLstmRuntime<B: Backend> {
    pub config: XLstmConfig,
    pub _marker: std::marker::PhantomData<B>,
}

/// Per-layer state variant for recurrent inference
#[derive(Clone, Debug)]
pub enum XLstmLayerState<B: Backend> {
    /// sLSTM state: cell, hidden, normalizer, stabilizer — all [batch, d_model]
    SLstm {
        c: Tensor<B, 2>,
        h: Tensor<B, 2>,
        n: Tensor<B, 2>,
        m: Tensor<B, 2>,
    },
    /// mLSTM state: matrix memory [batch, d_v_total, d_k_total],
    /// normalizer [batch, d_k_total], stabilizer scalar [batch, 1]
    MLstm {
        big_c: Tensor<B, 3>,
        n: Tensor<B, 2>,
        m: Tensor<B, 2>,
    },
}

impl<B: Backend> XLstmLayerState<B> {
    fn new_slstm(d_model: usize, batch: usize, device: &B::Device) -> Self {
        Self::SLstm {
            c: Tensor::zeros([batch, d_model], device),
            h: Tensor::zeros([batch, d_model], device),
            n: Tensor::zeros([batch, d_model], device),
            m: Tensor::full([batch, d_model], f32::NEG_INFINITY, device),
        }
    }

    fn new_mlstm(d_v_total: usize, d_k_total: usize, batch: usize, device: &B::Device) -> Self {
        Self::MLstm {
            big_c: Tensor::zeros([batch, d_v_total, d_k_total], device),
            n: Tensor::zeros([batch, d_k_total], device),
            m: Tensor::full([batch, 1], f32::NEG_INFINITY, device),
        }
    }
}

// ---------------------------------------------------------------------------
// sLSTM cell
// ---------------------------------------------------------------------------

/// Scalar LSTM cell with exponential gates and per-cell normalizer state.
#[derive(Module, Debug)]
pub struct SLstmCell<B: Backend> {
    /// Input projection for cell input z: d_model -> d_model
    pub w_z: Linear<B>,
    /// Recurrent projection for cell input z: d_model -> d_model
    pub r_z: Linear<B>,
    /// Input projection for input gate i: d_model -> d_model
    pub w_i: Linear<B>,
    /// Recurrent projection for input gate i: d_model -> d_model
    pub r_i: Linear<B>,
    /// Input projection for forget gate f: d_model -> d_model
    pub w_f: Linear<B>,
    /// Recurrent projection for forget gate f: d_model -> d_model
    pub r_f: Linear<B>,
    /// Input projection for output gate o: d_model -> d_model
    pub w_o: Linear<B>,
    /// Recurrent projection for output gate o: d_model -> d_model
    pub r_o: Linear<B>,
    #[module(skip)]
    pub d_model: usize,
}

impl<B: Backend> SLstmCell<B> {
    pub fn new(d_model: usize, device: &B::Device) -> Self {
        let make_linear = |in_dim: usize, out_dim: usize| {
            LinearConfig::new(in_dim, out_dim)
                .with_bias(true)
                .init(device)
        };
        Self {
            w_z: make_linear(d_model, d_model),
            r_z: make_linear(d_model, d_model),
            w_i: make_linear(d_model, d_model),
            r_i: make_linear(d_model, d_model),
            w_f: make_linear(d_model, d_model),
            r_f: make_linear(d_model, d_model),
            w_o: make_linear(d_model, d_model),
            r_o: make_linear(d_model, d_model),
            d_model,
        }
    }

    /// Single-step forward for one token.
    ///
    /// `x`  : [batch, d_model]  — current input
    /// state : mutable sLSTM state
    pub fn step(&self, x: Tensor<B, 2>, state: &mut XLstmLayerState<B>) -> Tensor<B, 2> {
        let (c_prev, h_prev, n_prev, m_prev) = match state {
            XLstmLayerState::SLstm { c, h, n, m } => (c.clone(), h.clone(), n.clone(), m.clone()),
            _ => panic!("SLstmCell::step called with non-sLSTM state"),
        };

        // Gate pre-activations
        let z_pre = self.w_z.forward(x.clone()) + self.r_z.forward(h_prev.clone());
        let i_pre = self.w_i.forward(x.clone()) + self.r_i.forward(h_prev.clone());
        let f_pre = self.w_f.forward(x.clone()) + self.r_f.forward(h_prev.clone());
        let o_pre = self.w_o.forward(x.clone()) + self.r_o.forward(h_prev);

        // Cell input
        let z = z_pre.tanh();
        // Output gate (sigmoid)
        let o = activation::sigmoid(o_pre);

        // Stabilizer: m = max(log(f) + m_prev, log(i))
        // log(f) = f_pre  (forget gate is exp(f_pre), so log = f_pre)
        // log(i) = i_pre
        let log_f = f_pre; // already log-space
        let log_i = i_pre;

        let m_new = (log_f.clone() + m_prev.clone()).max_pair(log_i.clone());

        // Stabilized gates: f_hat = exp(log_f + m_prev - m_new), i_hat = exp(log_i - m_new)
        let f_hat = (log_f + m_prev - m_new.clone()).exp();
        let i_hat = (log_i - m_new.clone()).exp();

        // Cell and normalizer update
        let c_new = f_hat.clone() * c_prev + i_hat.clone() * z;
        let n_new = f_hat * n_prev + i_hat;

        // Hidden state: h = o * tanh(c / max(|n|, 1))
        let n_abs = n_new.clone().abs();
        let denom = n_abs.max_pair(Tensor::ones_like(&n_new));
        let h_new = o * (c_new.clone() / denom).tanh();

        *state = XLstmLayerState::SLstm {
            c: c_new,
            h: h_new.clone(),
            n: n_new,
            m: m_new,
        };

        h_new
    }

    /// Sequence forward (processes each token step-by-step).
    ///
    /// `x`  : [batch, seq, d_model]
    pub fn forward(&self, x: Tensor<B, 3>, state: Option<&mut XLstmLayerState<B>>) -> Tensor<B, 3> {
        let [batch, seq_len, _] = x.dims();
        let device = x.device();

        let mut owned_state;
        let state_ref: &mut XLstmLayerState<B> = match state {
            Some(s) => s,
            None => {
                owned_state = XLstmLayerState::new_slstm(self.d_model, batch, &device);
                &mut owned_state
            }
        };

        let mut outputs = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let x_t = x
                .clone()
                .slice([0..batch, t..t + 1, 0..self.d_model])
                .squeeze_dim::<2>(1);
            let h_t = self.step(x_t, state_ref);
            outputs.push(h_t.unsqueeze_dim::<3>(1));
        }

        Tensor::cat(outputs, 1)
    }
}

// ---------------------------------------------------------------------------
// mLSTM cell
// ---------------------------------------------------------------------------

/// Matrix LSTM cell with covariance-style memory and multi-head attention-style
/// key/query/value projections.
#[derive(Module, Debug)]
pub struct MLstmCell<B: Backend> {
    /// Query projection: d_model -> num_heads * qk_dim
    pub w_q: Linear<B>,
    /// Key projection: d_model -> num_heads * qk_dim
    pub w_k: Linear<B>,
    /// Value projection: d_model -> num_heads * v_dim
    pub w_v: Linear<B>,
    /// Input gate: d_model -> 1 (scalar per head is broadcast)
    pub w_i: Linear<B>,
    /// Forget gate: d_model -> 1
    pub w_f: Linear<B>,
    /// Output projection: num_heads * v_dim -> d_model
    pub out_proj: Linear<B>,
    #[module(skip)]
    pub num_heads: usize,
    #[module(skip)]
    pub qk_dim: usize,
    #[module(skip)]
    pub v_dim: usize,
    #[module(skip)]
    pub d_model: usize,
}

impl<B: Backend> MLstmCell<B> {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        qk_dim: usize,
        v_dim: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            w_q: LinearConfig::new(d_model, num_heads * qk_dim)
                .with_bias(false)
                .init(device),
            w_k: LinearConfig::new(d_model, num_heads * qk_dim)
                .with_bias(false)
                .init(device),
            w_v: LinearConfig::new(d_model, num_heads * v_dim)
                .with_bias(false)
                .init(device),
            w_i: LinearConfig::new(d_model, 1).with_bias(true).init(device),
            w_f: LinearConfig::new(d_model, 1).with_bias(true).init(device),
            out_proj: LinearConfig::new(num_heads * v_dim, d_model)
                .with_bias(false)
                .init(device),
            num_heads,
            qk_dim,
            v_dim,
            d_model,
        }
    }

    /// Single-step forward for one token.
    ///
    /// `x`  : [batch, d_model]
    pub fn step(&self, x: Tensor<B, 2>, state: &mut XLstmLayerState<B>) -> Tensor<B, 2> {
        let (big_c_prev, n_prev, m_prev) = match state {
            XLstmLayerState::MLstm { big_c, n, m } => (big_c.clone(), n.clone(), m.clone()),
            _ => panic!("MLstmCell::step called with non-mLSTM state"),
        };

        let [batch] = [x.dims()[0]];
        let d_k = self.num_heads * self.qk_dim;
        let d_v = self.num_heads * self.v_dim;
        let scale = (self.qk_dim as f32).sqrt();

        // Projections
        let q = self.w_q.forward(x.clone()); // [batch, d_k]
        let k = self.w_k.forward(x.clone()) / scale; // [batch, d_k], scaled
        let v = self.w_v.forward(x.clone()); // [batch, d_v]

        // Gate pre-activations (scalar, log-space)
        let log_i = self.w_i.forward(x.clone()); // [batch, 1]
        let log_f = self.w_f.forward(x); // [batch, 1]

        // Stabilizer
        let m_new = (log_f.clone() + m_prev.clone()).max_pair(log_i.clone()); // [batch, 1]

        let f_hat = (log_f + m_prev - m_new.clone()).exp(); // [batch, 1]
        let i_hat = (log_i - m_new.clone()).exp(); // [batch, 1]

        // Matrix state update: C = f_hat * C_prev + i_hat * (v ⊗ k)
        // v ⊗ k = outer product: [batch, d_v, 1] * [batch, 1, d_k] -> [batch, d_v, d_k]
        let v_outer = v.clone().unsqueeze_dim::<3>(2); // [batch, d_v, 1]
        let k_outer = k.clone().unsqueeze_dim::<3>(1); // [batch, 1, d_k]
        let vk = v_outer.matmul(k_outer); // [batch, d_v, d_k]

        let f_hat_3d = f_hat
            .clone()
            .unsqueeze_dim::<3>(2)
            .expand([batch, d_v, d_k]); // broadcast
        let i_hat_3d = i_hat
            .clone()
            .unsqueeze_dim::<3>(2)
            .expand([batch, d_v, d_k]);
        let big_c_new = f_hat_3d * big_c_prev + i_hat_3d * vk;

        // Normalizer vector update: n = f_hat * n_prev + i_hat * k
        let f_hat_2d = f_hat.expand([batch, d_k]);
        let i_hat_2d = i_hat.expand([batch, d_k]);
        let n_new = f_hat_2d * n_prev + i_hat_2d * k;

        // Output: h = C @ q / max(|n · q|, 1)
        // C @ q: [batch, d_v, d_k] x [batch, d_k, 1] -> [batch, d_v, 1] -> [batch, d_v]
        let q_col = q.clone().unsqueeze_dim::<3>(2); // [batch, d_k, 1]
        let h_unnorm = big_c_new.clone().matmul(q_col).squeeze_dim::<2>(2); // [batch, d_v]

        // Denominator: |n · q| = |sum_k n_k * q_k| for each batch
        let n_dot_q = (n_new.clone() * q).sum_dim(1).squeeze_dim::<2>(1); // [batch]
        let denom = n_dot_q.clone().abs().max_pair(Tensor::ones_like(&n_dot_q)); // [batch]
        let denom_2d = denom.unsqueeze_dim::<2>(1).expand([batch, d_v]);
        let h = h_unnorm / denom_2d; // [batch, d_v]

        // Output projection
        let out = self.out_proj.forward(h);

        *state = XLstmLayerState::MLstm {
            big_c: big_c_new,
            n: n_new,
            m: m_new,
        };

        out
    }

    /// Sequence forward.
    ///
    /// `x`  : [batch, seq, d_model]
    pub fn forward(&self, x: Tensor<B, 3>, state: Option<&mut XLstmLayerState<B>>) -> Tensor<B, 3> {
        let [batch, seq_len, _] = x.dims();
        let device = x.device();
        let d_k = self.num_heads * self.qk_dim;
        let d_v = self.num_heads * self.v_dim;

        let mut owned_state;
        let state_ref: &mut XLstmLayerState<B> = match state {
            Some(s) => s,
            None => {
                owned_state = XLstmLayerState::new_mlstm(d_v, d_k, batch, &device);
                &mut owned_state
            }
        };

        let mut outputs = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let x_t = x
                .clone()
                .slice([0..batch, t..t + 1, 0..self.d_model])
                .squeeze_dim::<2>(1);
            let h_t = self.step(x_t, state_ref);
            outputs.push(h_t.unsqueeze_dim::<3>(1));
        }

        Tensor::cat(outputs, 1)
    }
}

// ---------------------------------------------------------------------------
// Cell enum for dynamic dispatch inside XLstmBlock
// ---------------------------------------------------------------------------

/// Either an sLSTM or an mLSTM cell, selected per layer.
#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
pub enum XLstmCellKind<B: Backend> {
    SLstm(SLstmCell<B>),
    MLstm(MLstmCell<B>),
}

// ---------------------------------------------------------------------------
// XLstm block (pre-norm + cell + output projection)
// ---------------------------------------------------------------------------

/// One xLSTM block: LayerNorm → (sLSTM | mLSTM) → output projection.
#[derive(Module, Debug)]
pub struct XLstmBlock<B: Backend> {
    /// Pre-norm
    pub ln: LayerNorm<B>,
    /// LSTM cell (sLSTM or mLSTM)
    pub cell: XLstmCellKind<B>,
    /// Output projection: d_inner -> d_model (used for mLSTM expansion)
    pub out_proj: Option<Linear<B>>,
    #[module(skip)]
    pub is_mlstm: bool,
}

impl<B: Backend> XLstmBlock<B> {
    pub fn new(config: &XLstmConfig, layer_idx: usize, device: &B::Device) -> Self {
        let is_mlstm = config.mlstm_at_layers.contains(&layer_idx);
        let ln = LayerNormConfig::new(config.d_model)
            .with_epsilon(config.layer_norm_eps)
            .init(device);

        let (cell, out_proj) = if is_mlstm {
            let cell = MLstmCell::new(
                config.d_model,
                config.num_heads,
                config.qk_dim,
                config.v_dim,
                device,
            );
            // mLSTM out_proj is already inside MLstmCell (d_v -> d_model), so no extra proj
            (XLstmCellKind::MLstm(cell), None)
        } else {
            let cell = SLstmCell::new(config.d_model, device);
            (XLstmCellKind::SLstm(cell), None)
        };

        Self {
            ln,
            cell,
            out_proj,
            is_mlstm,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, state: Option<&mut XLstmLayerState<B>>) -> Tensor<B, 3> {
        let residual = x.clone();
        let normed = self.ln.forward(x);

        let cell_out = match &self.cell {
            XLstmCellKind::SLstm(cell) => cell.forward(normed, state),
            XLstmCellKind::MLstm(cell) => cell.forward(normed, state),
        };

        let projected = match &self.out_proj {
            Some(proj) => proj.forward(cell_out),
            None => cell_out,
        };

        projected + residual
    }
}

// ---------------------------------------------------------------------------
// XLstm model
// ---------------------------------------------------------------------------

/// Full xLSTM language model.
#[derive(Module, Debug)]
pub struct XLstm<B: Backend> {
    pub embed_tokens: Embedding<B>,
    pub layers: Vec<XLstmBlock<B>>,
    pub ln_f: LayerNorm<B>,
    pub lm_head: Linear<B>,
}

/// Output from the xLSTM model
pub struct XLstmOutput<B: Backend> {
    /// Logits over vocabulary [batch, seq_len, vocab_size]
    pub logits: Tensor<B, 3>,
    /// Final hidden states [batch, seq_len, d_model]
    pub hidden_states: Tensor<B, 3>,
}

impl<B: Backend> XLstm<B> {
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        mut states: Option<&mut Vec<XLstmLayerState<B>>>,
    ) -> XLstmOutput<B> {
        let mut hidden = self.embed_tokens.forward(input_ids);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let state = states.as_mut().map(|s| &mut s[layer_idx]);
            hidden = layer.forward(hidden, state);
        }

        hidden = self.ln_f.forward(hidden);
        let logits = self.lm_head.forward(hidden.clone());

        XLstmOutput {
            logits,
            hidden_states: hidden,
        }
    }

    /// Initialize fresh states for all layers.
    pub fn init_states(
        &self,
        runtime: &XLstmRuntime<B>,
        batch: usize,
        device: &B::Device,
    ) -> Vec<XLstmLayerState<B>> {
        let cfg = &runtime.config;
        (0..cfg.num_layers)
            .map(|layer_idx| {
                if cfg.mlstm_at_layers.contains(&layer_idx) {
                    let d_v = cfg.num_heads * cfg.v_dim;
                    let d_k = cfg.num_heads * cfg.qk_dim;
                    XLstmLayerState::new_mlstm(d_v, d_k, batch, device)
                } else {
                    XLstmLayerState::new_slstm(cfg.d_model, batch, device)
                }
            })
            .collect()
    }

    /// Autoregressive generation.
    pub fn generate(
        &self,
        input_ids: Tensor<B, 2, Int>,
        runtime: &XLstmRuntime<B>,
        max_new_tokens: usize,
        sampler: &crate::sampling::SamplerConfig,
    ) -> Tensor<B, 2, Int> {
        let [batch, _] = input_ids.dims();
        let device = input_ids.device();

        let input_data: Vec<i64> = input_ids.to_data().to_vec().unwrap();
        let mut context_tokens: Vec<u32> = input_data.iter().map(|&id| id as u32).collect();

        let mut states = self.init_states(runtime, batch, &device);

        // Process prompt
        let output = self.forward(input_ids.clone(), Some(&mut states));

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

            let output = self.forward(next_token, Some(&mut states));
            last_logits = output.logits.squeeze_dim::<2>(1);
        }

        all_tokens
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_xlstm_config_tiny() {
        let _ = XLstmConfig::tiny();
    }

    #[test]
    fn test_slstm_cell_forward() {
        let device = Default::default();
        let cell = SLstmCell::<TestBackend>::new(64, &device);
        let x = Tensor::zeros([1, 4, 64], &device);
        let out = cell.forward(x, None);
        assert_eq!(out.dims(), [1, 4, 64]);
    }

    #[test]
    fn test_mlstm_cell_forward() {
        let device = Default::default();
        let cell = MLstmCell::<TestBackend>::new(64, 4, 16, 16, &device);
        let x = Tensor::zeros([1, 4, 64], &device);
        let out = cell.forward(x, None);
        assert_eq!(out.dims(), [1, 4, 64]);
    }

    #[test]
    fn test_xlstm_tiny_forward() {
        let device = Default::default();
        let config = XLstmConfig::tiny();
        let (model, _runtime) = config.init::<TestBackend>(&device);
        let input_ids = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device);
        let output = model.forward(input_ids, None);
        assert_eq!(output.logits.dims(), [1, 4, 1000]);
        assert_eq!(output.hidden_states.dims(), [1, 4, 64]);
    }

    #[test]
    fn test_xlstm_with_state() {
        let device = Default::default();
        let config = XLstmConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);
        let mut states = model.init_states(&runtime, 1, &device);

        let input1 = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device);
        let _ = model.forward(input1, Some(&mut states));

        let input2 = Tensor::<TestBackend, 2, Int>::from_ints([[3]], &device);
        let output = model.forward(input2, Some(&mut states));
        assert_eq!(output.logits.dims(), [1, 1, 1000]);
    }

    #[test]
    fn test_xlstm_generate() {
        let device = Default::default();
        let config = XLstmConfig::tiny();
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
}
