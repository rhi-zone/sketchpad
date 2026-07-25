//! Layer normalization implementation
//!
//! Provides layer normalization as used in transformer architectures.
//! Normalizes across the last dimension of the input tensor.

use burn::prelude::*;

/// Layer normalization module
///
/// Normalizes inputs across the last dimension, then applies a learned
/// affine transformation (scale and shift). This is the standard layer
/// normalization used in transformer models like CLIP.
///
/// # Formula
///
/// For input x with last dimension of size D:
/// ```text
/// y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
/// ```
#[derive(Module, Debug)]
pub struct LayerNorm<B: Backend> {
    /// Scale parameter
    pub weight: Tensor<B, 1>,
    /// Shift parameter
    pub bias: Tensor<B, 1>,
    /// Epsilon for numerical stability
    pub eps: f64,
}

impl<B: Backend> LayerNorm<B> {
    /// Creates a new layer normalization module
    ///
    /// # Arguments
    ///
    /// * `size` - Size of the normalized dimension (last dimension)
    /// * `device` - Device to create tensors on
    pub fn new(size: usize, device: &B::Device) -> Self {
        Self {
            weight: Tensor::ones([size], device),
            bias: Tensor::zeros([size], device),
            eps: 1e-5,
        }
    }

    /// Creates layer norm from pre-loaded weight and bias
    pub fn from_weight_bias(weight: Tensor<B, 1>, bias: Tensor<B, 1>) -> Self {
        Self {
            weight,
            bias,
            eps: 1e-5,
        }
    }

    /// Applies layer normalization to the input tensor
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of any dimensionality
    ///
    /// # Returns
    ///
    /// Normalized tensor with same shape as input
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let last_dim = D - 1;
        let mean = x.clone().mean_dim(last_dim);
        let var = x.clone().var(last_dim);

        let x_norm = (x - mean) / (var + self.eps).sqrt();

        // Apply affine transformation
        // Weight and bias broadcasting handled by Burn
        x_norm * self.weight.clone().unsqueeze() + self.bias.clone().unsqueeze()
    }
}
