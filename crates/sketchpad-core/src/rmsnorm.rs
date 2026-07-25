//! Root Mean Square Layer Normalization
//!
//! Provides RMSNorm as used in modern transformer architectures like LLaMA,
//! Qwen, and DiT-based models. Simpler and faster than LayerNorm.

use burn::prelude::*;

/// Root Mean Square Layer Normalization
///
/// Unlike LayerNorm, RMSNorm does not subtract the mean or apply a bias.
/// This makes it computationally cheaper while maintaining similar performance
/// in practice.
///
/// # Formula
///
/// For input x with last dimension of size D:
/// ```text
/// y = x / sqrt(mean(x^2) + eps) * weight
/// ```
///
/// # References
///
/// - [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
/// - Used in LLaMA, Qwen, Mistral, and many modern transformers
#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    /// Learnable scale weight
    pub weight: Tensor<B, 1>,
    /// Epsilon for numerical stability
    pub eps: f64,
}

impl<B: Backend> RmsNorm<B> {
    /// Creates a new RMSNorm module
    ///
    /// # Arguments
    ///
    /// * `size` - Size of the normalized dimension (last dimension)
    /// * `device` - Device to create tensors on
    pub fn new(size: usize, device: &B::Device) -> Self {
        Self {
            weight: Tensor::ones([size], device),
            eps: 1e-6,
        }
    }

    /// Creates RMSNorm with a custom epsilon value
    ///
    /// # Arguments
    ///
    /// * `size` - Size of the normalized dimension
    /// * `eps` - Small constant for numerical stability (default: 1e-6)
    /// * `device` - Device to create tensors on
    pub fn with_eps(size: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            weight: Tensor::ones([size], device),
            eps,
        }
    }

    /// Creates RMSNorm from a loaded weight tensor
    ///
    /// # Arguments
    ///
    /// * `weight` - Pre-loaded weight tensor
    /// * `eps` - Epsilon for numerical stability
    pub fn from_weight(weight: Tensor<B, 1>, eps: f64) -> Self {
        Self { weight, eps }
    }

    /// Applies RMS normalization to the input tensor
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
        // Compute mean of squared values
        let mean_sq = x.clone().powf_scalar(2.0).mean_dim(last_dim);
        // RMS normalization
        let x_norm = x / (mean_sq + self.eps).sqrt();
        // Apply learned scale
        x_norm * self.weight.clone().unsqueeze()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_rmsnorm_shape_preserved() {
        let device = Default::default();
        let norm = RmsNorm::<TestBackend>::new(64, &device);

        let x = Tensor::zeros([2, 10, 64], &device);
        let y = norm.forward(x);

        assert_eq!(y.dims(), [2, 10, 64]);
    }

    #[test]
    fn test_rmsnorm_normalized_output() {
        let device = Default::default();
        let norm = RmsNorm::<TestBackend>::new(4, &device);

        // Input with known values
        let x: Tensor<TestBackend, 2> = Tensor::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
        let y = norm.forward(x);

        // After RMSNorm, values should be scaled relative to RMS
        // RMS = sqrt((1 + 4 + 9 + 16) / 4) = sqrt(7.5) ≈ 2.739
        let y_data: Vec<f32> = y.into_data().to_vec().unwrap();
        assert!(y_data.iter().all(|&v| v.abs() < 2.0));
    }
}
