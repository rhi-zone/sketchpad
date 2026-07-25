//! Mamba: Selective State Space Model
//!
//! Mamba is a state space model (SSM) architecture that achieves linear scaling
//! with sequence length while maintaining the ability to selectively focus on
//! relevant parts of the input.
//!
//! Key innovations:
//! - Selective state spaces: B, C, Δ parameters are input-dependent
//! - Hardware-aware parallel scan for efficient GPU computation
//! - Linear O(n) time complexity vs O(n²) for transformers
//! - Constant memory state (no KV cache needed)
//!
//! Reference: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
//! https://arxiv.org/abs/2312.00752

use burn::module::{Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::{Int, activation};

/// Mamba model configuration
#[derive(Clone, Debug)]
pub struct MambaConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Model dimension (d_model)
    pub d_model: usize,
    /// Number of layers
    pub n_layer: usize,
    /// SSM state dimension (typically 16)
    pub d_state: usize,
    /// Convolution kernel size (typically 4)
    pub d_conv: usize,
    /// Expansion factor for inner dimension (typically 2)
    pub expand: usize,
    /// Delta/timestep rank for low-rank projection (typically d_model/16)
    pub dt_rank: usize,
    /// Layer norm epsilon
    pub layer_norm_eps: f64,
}

impl MambaConfig {
    /// Mamba-130M configuration
    pub fn mamba_130m() -> Self {
        Self {
            vocab_size: 50280,
            d_model: 768,
            n_layer: 24,
            d_state: 16,
            d_conv: 4,
            expand: 2,
            dt_rank: 48, // 768 / 16
            layer_norm_eps: 1e-5,
        }
    }

    /// Mamba-370M configuration
    pub fn mamba_370m() -> Self {
        Self {
            vocab_size: 50280,
            d_model: 1024,
            n_layer: 48,
            d_state: 16,
            d_conv: 4,
            expand: 2,
            dt_rank: 64, // 1024 / 16
            layer_norm_eps: 1e-5,
        }
    }

    /// Mamba-790M configuration
    pub fn mamba_790m() -> Self {
        Self {
            vocab_size: 50280,
            d_model: 1536,
            n_layer: 48,
            d_state: 16,
            d_conv: 4,
            expand: 2,
            dt_rank: 96, // 1536 / 16
            layer_norm_eps: 1e-5,
        }
    }

    /// Mamba-1.4B configuration
    pub fn mamba_1_4b() -> Self {
        Self {
            vocab_size: 50280,
            d_model: 2048,
            n_layer: 48,
            d_state: 16,
            d_conv: 4,
            expand: 2,
            dt_rank: 128, // 2048 / 16
            layer_norm_eps: 1e-5,
        }
    }

    /// Mamba-2.8B configuration
    pub fn mamba_2_8b() -> Self {
        Self {
            vocab_size: 50280,
            d_model: 2560,
            n_layer: 64,
            d_state: 16,
            d_conv: 4,
            expand: 2,
            dt_rank: 160, // 2560 / 16
            layer_norm_eps: 1e-5,
        }
    }

    /// Tiny configuration for testing
    pub fn tiny() -> Self {
        Self {
            vocab_size: 1000,
            d_model: 64,
            n_layer: 2,
            d_state: 8,
            d_conv: 4,
            expand: 2,
            dt_rank: 4, // 64 / 16
            layer_norm_eps: 1e-5,
        }
    }

    /// Inner dimension (d_model * expand)
    pub fn d_inner(&self) -> usize {
        self.d_model * self.expand
    }

    /// Initialize the model and runtime
    pub fn init<B: Backend>(&self, device: &B::Device) -> (Mamba<B>, MambaRuntime<B>) {
        let layers: Vec<MambaBlock<B>> = (0..self.n_layer)
            .map(|_| MambaBlock::new(self, device))
            .collect();

        let model = Mamba {
            embed_tokens: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            layers,
            ln_f: LayerNormConfig::new(self.d_model)
                .with_epsilon(self.layer_norm_eps)
                .init(device),
            lm_head: LinearConfig::new(self.d_model, self.vocab_size)
                .with_bias(false)
                .init(device),
        };

        let runtime = MambaRuntime {
            config: self.clone(),
            _marker: std::marker::PhantomData,
        };

        (model, runtime)
    }
}

/// Runtime configuration (non-Module data)
pub struct MambaRuntime<B: Backend> {
    pub config: MambaConfig,
    pub _marker: std::marker::PhantomData<B>,
}

/// State for one Mamba layer during inference
#[derive(Clone, Debug)]
pub struct MambaState<B: Backend> {
    /// SSM hidden state [batch, d_inner, d_state]
    pub ssm_state: Tensor<B, 3>,
    /// Convolution state [batch, d_inner, d_conv-1]
    pub conv_state: Tensor<B, 3>,
}

impl<B: Backend> MambaState<B> {
    /// Create new zero state
    pub fn new(config: &MambaConfig, batch: usize, device: &B::Device) -> Self {
        let d_inner = config.d_inner();
        Self {
            ssm_state: Tensor::zeros([batch, d_inner, config.d_state], device),
            conv_state: Tensor::zeros([batch, d_inner, config.d_conv - 1], device),
        }
    }
}

/// Mamba model output
pub struct MambaOutput<B: Backend> {
    /// Logits over vocabulary [batch, seq, vocab_size]
    pub logits: Tensor<B, 3>,
    /// Hidden states [batch, seq, d_model]
    pub hidden_states: Tensor<B, 3>,
}

/// Mamba model
#[derive(Module, Debug)]
pub struct Mamba<B: Backend> {
    /// Token embeddings
    pub embed_tokens: Embedding<B>,
    /// Mamba layers
    pub layers: Vec<MambaBlock<B>>,
    /// Final layer norm
    pub ln_f: LayerNorm<B>,
    /// Language model head
    pub lm_head: Linear<B>,
}

impl<B: Backend> Mamba<B> {
    /// Forward pass
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        mut states: Option<&mut Vec<MambaState<B>>>,
    ) -> MambaOutput<B> {
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

        MambaOutput {
            logits,
            hidden_states,
        }
    }

    /// Initialize fresh states for recurrent inference
    pub fn init_states(
        &self,
        runtime: &MambaRuntime<B>,
        batch: usize,
        device: &B::Device,
    ) -> Vec<MambaState<B>> {
        (0..runtime.config.n_layer)
            .map(|_| MambaState::new(&runtime.config, batch, device))
            .collect()
    }

    /// Generate text autoregressively
    pub fn generate(
        &self,
        input_ids: Tensor<B, 2, Int>,
        runtime: &MambaRuntime<B>,
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

/// Single Mamba block
#[derive(Module, Debug)]
pub struct MambaBlock<B: Backend> {
    /// Pre-norm
    pub ln: LayerNorm<B>,
    /// Mamba mixer
    pub mixer: MambaMixer<B>,
}

impl<B: Backend> MambaBlock<B> {
    pub fn new(config: &MambaConfig, device: &B::Device) -> Self {
        Self {
            ln: LayerNormConfig::new(config.d_model)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
            mixer: MambaMixer::new(config, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, state: Option<&mut MambaState<B>>) -> Tensor<B, 3> {
        let residual = x.clone();
        let x = self.ln.forward(x);
        let x = self.mixer.forward(x, state);
        x + residual
    }
}

/// Mamba mixer (core selective SSM)
#[derive(Module, Debug)]
pub struct MambaMixer<B: Backend> {
    /// Input projection: d_model -> d_inner * 2 (for x and z branches)
    pub in_proj: Linear<B>,
    /// Depthwise convolution
    pub conv1d: Conv1d<B>,
    /// SSM parameter projections: d_inner -> dt_rank + 2*d_state
    pub x_proj: Linear<B>,
    /// Time step projection: dt_rank -> d_inner
    pub dt_proj: Linear<B>,
    /// SSM A parameter (log form) [d_inner, d_state]
    pub a_log: Param<Tensor<B, 2>>,
    /// SSM D parameter [d_inner]
    pub d: Param<Tensor<B, 1>>,
    /// Output projection
    pub out_proj: Linear<B>,
    /// Config values
    #[module(skip)]
    pub d_inner: usize,
    #[module(skip)]
    pub d_state: usize,
    #[module(skip)]
    pub d_conv: usize,
    #[module(skip)]
    pub dt_rank: usize,
}

impl<B: Backend> MambaMixer<B> {
    pub fn new(config: &MambaConfig, device: &B::Device) -> Self {
        let d_inner = config.d_inner();
        let d_state = config.d_state;
        let dt_rank = config.dt_rank;

        // A is initialized as log of a range [1, d_state]
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
                .with_groups(d_inner) // Depthwise
                .with_padding(burn::nn::PaddingConfig1d::Explicit(config.d_conv - 1))
                .with_bias(true)
                .init(device),
            x_proj: LinearConfig::new(d_inner, dt_rank + d_state * 2) // dt_rank + B + C
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

    pub fn forward(&self, x: Tensor<B, 3>, mut state: Option<&mut MambaState<B>>) -> Tensor<B, 3> {
        let [batch, seq_len, _] = x.dims();

        // Project to inner dimension (x and gate branches)
        let xz = self.in_proj.forward(x);
        let x = xz.clone().slice([0..batch, 0..seq_len, 0..self.d_inner]);
        let z = xz.slice([0..batch, 0..seq_len, self.d_inner..self.d_inner * 2]);

        // Transpose for conv: [batch, seq, d_inner] -> [batch, d_inner, seq]
        let x = x.swap_dims(1, 2);

        // Causal convolution
        let x = if seq_len == 1 {
            // Single token - use conv state
            if let Some(ref mut s) = state {
                // Append new input to state
                let conv_in = Tensor::cat(vec![s.conv_state.clone(), x.clone()], 2);
                // Update state with last d_conv-1 elements
                let total_len = conv_in.dims()[2];
                let state_start = total_len - (self.d_conv - 1);
                s.conv_state =
                    conv_in
                        .clone()
                        .slice([0..batch, 0..self.d_inner, state_start..total_len]);
                // Apply conv1d
                let x = self.conv1d.forward(conv_in);
                // Take only the last position
                let seq_out = x.dims()[2];
                x.slice([0..batch, 0..self.d_inner, seq_out - 1..seq_out])
            } else {
                // No state - just do regular conv with padding
                self.conv1d.forward(x)
            }
        } else {
            // Multi-token - regular conv
            let x = self.conv1d.forward(x);
            // Remove extra padding at the end
            x.slice([0..batch, 0..self.d_inner, 0..seq_len])
        };

        // Apply SiLU activation
        let x = activation::silu(x);

        // Transpose back: [batch, d_inner, seq] -> [batch, seq, d_inner]
        let x = x.swap_dims(1, 2);

        // Compute SSM parameters from x
        let x_proj = self.x_proj.forward(x.clone());

        // Split into dt, B, C
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

        // Project dt from low-rank to full d_inner and apply softplus
        let dt = self.dt_proj.forward(dt_low);
        let dt = burn::tensor::activation::softplus(dt, 1.0);

        // Get A (negative exponentiated from a_log)
        let a = -self.a_log.val().exp();

        // Run SSM
        let y = self.ssm_forward(x.clone(), dt, a, b, c, &mut state);

        // Gating with z branch (SiLU)
        let z = activation::silu(z);
        let y = y * z;

        // Output projection
        self.out_proj.forward(y)
    }

    /// Run the selective state space model
    fn ssm_forward(
        &self,
        x: Tensor<B, 3>,  // [batch, seq, d_inner]
        dt: Tensor<B, 3>, // [batch, seq, d_inner]
        a: Tensor<B, 2>,  // [d_inner, d_state]
        b: Tensor<B, 3>,  // [batch, seq, d_state]
        c: Tensor<B, 3>,  // [batch, seq, d_state]
        state: &mut Option<&mut MambaState<B>>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _] = x.dims();
        let device = x.device();

        // Get D parameter
        let d = self.d.val();

        // Initialize or get hidden state
        let mut h = match state {
            Some(s) => s.ssm_state.clone(),
            None => Tensor::zeros([batch, self.d_inner, self.d_state], &device),
        };

        // Process sequence
        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            // Extract time step values
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

            // Discretize: dA = exp(dt * A), dB = dt * B
            // A is [d_inner, d_state], dt_t is [batch, d_inner]
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

            // Update hidden state: h = dA * h + dB * x
            let x_expanded =
                x_t.clone()
                    .unsqueeze_dim::<3>(2)
                    .expand([batch, self.d_inner, self.d_state]);
            h = d_a * h + d_b * x_expanded;

            // Compute output: y = (C @ h) + D * x
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

        // Update state
        if let Some(s) = state {
            s.ssm_state = h;
        }

        Tensor::cat(outputs, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_mamba_config() {
        let _ = MambaConfig::mamba_130m();
        let _ = MambaConfig::mamba_370m();
        let _ = MambaConfig::mamba_790m();
        let _ = MambaConfig::mamba_1_4b();
        let _ = MambaConfig::mamba_2_8b();
    }

    #[test]
    fn test_mamba_tiny_forward() {
        let device = Default::default();
        let config = MambaConfig::tiny();
        let (model, _runtime) = config.init::<TestBackend>(&device);

        let input_ids = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device);
        let output = model.forward(input_ids, None);

        assert_eq!(output.logits.dims(), [1, 4, 1000]);
        assert_eq!(output.hidden_states.dims(), [1, 4, 64]);
    }

    #[test]
    fn test_mamba_with_state() {
        let device = Default::default();
        let config = MambaConfig::tiny();
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
    fn test_mamba_generate() {
        let device = Default::default();
        let config = MambaConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        let prompt = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device);
        let generated = model.generate(prompt, &runtime, 3, 1.0);

        assert_eq!(generated.dims(), [1, 5]);
    }

    #[test]
    fn test_mamba_mixer() {
        let device = Default::default();
        let config = MambaConfig::tiny();
        let mixer = MambaMixer::new(&config, &device);

        let x = Tensor::<TestBackend, 3>::zeros([1, 4, 64], &device);
        let output = mixer.forward(x, None);

        assert_eq!(output.dims(), [1, 4, 64]);
    }

    #[test]
    fn test_mamba_block() {
        let device = Default::default();
        let config = MambaConfig::tiny();
        let block = MambaBlock::new(&config, &device);

        let x = Tensor::<TestBackend, 3>::zeros([1, 4, 64], &device);
        let output = block.forward(x, None);

        assert_eq!(output.dims(), [1, 4, 64]);
    }
}
