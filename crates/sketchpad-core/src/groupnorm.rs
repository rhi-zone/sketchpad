//! Group normalization implementation
//!
//! Provides group normalization as used in UNet and VAE architectures.
//! Divides channels into groups and normalizes within each group.

use burn::prelude::*;

/// Group normalization module
///
/// Divides channels into groups and normalizes each group independently.
/// This is commonly used in diffusion models (UNet, VAE) as an alternative
/// to batch normalization that works well with small batch sizes.
///
/// # Formula
///
/// For input with C channels divided into G groups:
/// ```text
/// y = (x - mean(x_group)) / sqrt(var(x_group) + eps) * weight + bias
/// ```
///
/// # Reference
///
/// "Group Normalization" - Wu & He, 2018
#[derive(Module, Debug)]
pub struct GroupNorm<B: Backend> {
    /// Number of groups to divide channels into
    pub num_groups: usize,
    /// Scale parameter (gamma), shape [num_channels]
    pub weight: Tensor<B, 1>,
    /// Bias parameter (beta), shape [num_channels]
    pub bias: Tensor<B, 1>,
    /// Epsilon for numerical stability
    pub eps: f64,
}

impl<B: Backend> GroupNorm<B> {
    /// Creates a new group normalization module
    ///
    /// # Arguments
    ///
    /// * `num_groups` - Number of groups to divide channels into (typically 32)
    /// * `num_channels` - Total number of input channels (must be divisible by num_groups)
    /// * `device` - Device to create tensors on
    pub fn new(num_groups: usize, num_channels: usize, device: &B::Device) -> Self {
        Self {
            num_groups,
            weight: Tensor::ones([num_channels], device),
            bias: Tensor::zeros([num_channels], device),
            eps: 1e-5,
        }
    }

    /// Applies group normalization to a 4D tensor
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape `[batch, channels, height, width]`
    ///
    /// # Returns
    ///
    /// Normalized tensor with same shape as input
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, channels, height, width] = x.dims();
        let group_size = channels / self.num_groups;

        // Clamp input to prevent Inf from propagating
        // This handles cases where preceding operations produced Inf
        let x = x.clamp(-65000.0, 65000.0);

        // Reshape to [batch, num_groups, group_size * height * width]
        let x = x.reshape([batch, self.num_groups, group_size * height * width]);

        // For f16/bf16, mean and variance computation can overflow (f16 max ~65504).
        // Compute the entire normalization pass in f32, then cast back.
        use burn::tensor::DType;
        let original_dtype = x.dtype();
        let x_f32 = x.cast(DType::F32);
        let mean_f32 = x_f32.clone().mean_dim(2);
        let mean_expanded_f32 = mean_f32.unsqueeze::<3>();
        let diff_f32 = x_f32 - mean_expanded_f32;
        let var_f32 = (diff_f32.clone() * diff_f32.clone()).mean_dim(2);

        // Expand var for broadcasting
        let var_f32 = var_f32.unsqueeze::<3>(); // [batch, num_groups, 1]

        // Normalize entirely in f32, then cast back
        let x_f32 = diff_f32 / (var_f32 + self.eps).sqrt();
        let x = x_f32.cast(original_dtype);

        // Reshape back to [batch, channels, height, width]
        let x = x.reshape([batch, channels, height, width]);

        // Apply weight and bias
        let weight = self.weight.clone().reshape([1, channels, 1, 1]);
        let bias = self.bias.clone().reshape([1, channels, 1, 1]);

        let x = x * weight + bias;

        // Clamp to prevent f16 overflow (max ~65504)
        // This is a common practice in mixed-precision inference
        x.clamp(-65000.0, 65000.0)
    }
}
