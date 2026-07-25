//! Gated Linear Unit variants
//!
//! Provides SwiGLU, GeGLU, and related gated activation functions used in
//! modern transformer FFN layers. These activations multiply one part of
//! the input by a gated transformation of another part.

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

/// SwiGLU Feed-Forward Network
///
/// Implements the SwiGLU FFN used in LLaMA and similar models.
/// Uses SiLU (Swish) as the gating activation.
///
/// # Architecture
///
/// ```text
/// output = down_proj(gate_proj(x) * SiLU(up_proj(x)))
/// ```
///
/// # References
///
/// - [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
/// - Used in LLaMA, PaLM, Qwen, Mistral
#[derive(Module, Debug)]
pub struct SwiGluFfn<B: Backend> {
    /// Gate projection (for gating activation)
    pub gate_proj: Linear<B>,
    /// Up projection
    pub up_proj: Linear<B>,
    /// Down projection
    pub down_proj: Linear<B>,
}

/// Configuration for SwiGluFfn
pub struct SwiGluFfnConfig {
    /// Input/output dimension
    pub hidden_size: usize,
    /// Intermediate (expanded) dimension
    pub intermediate_size: usize,
    /// Whether to use bias in linear layers
    pub bias: bool,
}

impl SwiGluFfnConfig {
    /// Creates a new config (LLaMA-style, no bias)
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            bias: false,
        }
    }

    /// Creates a config with bias enabled
    pub fn with_bias(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            bias: true,
        }
    }

    /// Initializes the SwiGluFfn module
    pub fn init<B: Backend>(&self, device: &B::Device) -> SwiGluFfn<B> {
        let gate_config =
            LinearConfig::new(self.hidden_size, self.intermediate_size).with_bias(self.bias);
        let up_config =
            LinearConfig::new(self.hidden_size, self.intermediate_size).with_bias(self.bias);
        let down_config =
            LinearConfig::new(self.intermediate_size, self.hidden_size).with_bias(self.bias);

        SwiGluFfn {
            gate_proj: gate_config.init(device),
            up_proj: up_config.init(device),
            down_proj: down_config.init(device),
        }
    }
}

impl<B: Backend> SwiGluFfn<B> {
    /// Applies the SwiGLU feed-forward network
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape [batch, seq_len, hidden_size]
    ///
    /// # Returns
    ///
    /// Output tensor of shape [batch, seq_len, hidden_size]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = self.gate_proj.forward(x.clone());
        let up = self.up_proj.forward(x);
        let gate_activated = burn::tensor::activation::silu(gate);
        self.down_proj.forward(gate_activated * up)
    }
}

/// GeGLU Feed-Forward Network
///
/// Implements GeGLU FFN using GELU as the gating activation.
///
/// # Architecture
///
/// ```text
/// output = down_proj(gate_proj(x) * GELU(up_proj(x)))
/// ```
#[derive(Module, Debug)]
pub struct GeGluFfn<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
}

/// Configuration for GeGluFfn
pub struct GeGluFfnConfig {
    /// Input/output dimension
    pub hidden_size: usize,
    /// Intermediate (expanded) dimension
    pub intermediate_size: usize,
    /// Whether to use bias in linear layers
    pub bias: bool,
}

impl GeGluFfnConfig {
    /// Creates a new config
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            bias: false,
        }
    }

    /// Initializes the GeGluFfn module
    pub fn init<B: Backend>(&self, device: &B::Device) -> GeGluFfn<B> {
        let gate_config =
            LinearConfig::new(self.hidden_size, self.intermediate_size).with_bias(self.bias);
        let up_config =
            LinearConfig::new(self.hidden_size, self.intermediate_size).with_bias(self.bias);
        let down_config =
            LinearConfig::new(self.intermediate_size, self.hidden_size).with_bias(self.bias);

        GeGluFfn {
            gate_proj: gate_config.init(device),
            up_proj: up_config.init(device),
            down_proj: down_config.init(device),
        }
    }
}

impl<B: Backend> GeGluFfn<B> {
    /// Applies the GeGLU feed-forward network
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = self.gate_proj.forward(x.clone());
        let up = self.up_proj.forward(x);
        let gate_activated = burn::tensor::activation::gelu(gate);
        self.down_proj.forward(gate_activated * up)
    }
}

/// Standalone SwiGLU activation function
///
/// Applies SwiGLU to a tensor that has already been projected to 2x intermediate size.
/// The input is split in half, with one half gated by SiLU of the other.
///
/// # Arguments
///
/// * `x` - Input tensor where last dimension is 2 * intermediate_size
///
/// # Returns
///
/// Tensor with last dimension halved
pub fn swiglu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let dims = x.dims();
    let last_dim = dims[D - 1];
    assert!(last_dim % 2 == 0, "Last dimension must be even for SwiGLU");

    let half = last_dim / 2;
    let (x1, x2) = split_last_dim(x, half);

    burn::tensor::activation::silu(x1) * x2
}

/// Standalone GeGLU activation function
///
/// Applies GeGLU to a tensor that has already been projected to 2x intermediate size.
pub fn geglu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let dims = x.dims();
    let last_dim = dims[D - 1];
    assert!(last_dim % 2 == 0, "Last dimension must be even for GeGLU");

    let half = last_dim / 2;
    let (x1, x2) = split_last_dim(x, half);

    burn::tensor::activation::gelu(x1) * x2
}

/// Helper to split a tensor along the last dimension
fn split_last_dim<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    split_at: usize,
) -> (Tensor<B, D>, Tensor<B, D>) {
    let dims = x.dims();
    let last_dim = dims[D - 1];

    let x1 = x.clone().narrow(D - 1, 0, split_at);
    let x2 = x.narrow(D - 1, split_at, last_dim - split_at);

    (x1, x2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_swiglu_ffn_shape() {
        let device = Default::default();
        let config = SwiGluFfnConfig::new(256, 512);
        let ffn = config.init::<TestBackend>(&device);

        let x = Tensor::zeros([2, 16, 256], &device);
        let y = ffn.forward(x);

        assert_eq!(y.dims(), [2, 16, 256]);
    }

    #[test]
    fn test_geglu_ffn_shape() {
        let device = Default::default();
        let config = GeGluFfnConfig::new(128, 256);
        let ffn = config.init::<TestBackend>(&device);

        let x = Tensor::zeros([1, 4, 128], &device);
        let y = ffn.forward(x);

        assert_eq!(y.dims(), [1, 4, 128]);
    }

    #[test]
    fn test_swiglu_standalone() {
        let device = Default::default();
        let x: Tensor<TestBackend, 3> = Tensor::ones([2, 8, 64], &device);
        let y = swiglu(x);

        assert_eq!(y.dims(), [2, 8, 32]);
    }

    #[test]
    fn test_geglu_standalone() {
        let device = Default::default();
        let x: Tensor<TestBackend, 3> = Tensor::ones([2, 8, 64], &device);
        let y = geglu(x);

        assert_eq!(y.dims(), [2, 8, 32]);
    }
}
