//! Precision configuration for inference
//!
//! Supports running models in different precision modes:
//! - fp32: Full precision (default)
//! - fp16: Half precision (faster, less memory)
//! - bf16: Brain floating point (good balance)
//!
//! # Note on Burn's precision model
//!
//! In Burn, tensor precision is determined at **compile time** by the backend type,
//! not at runtime. For example:
//! - `NdArray<f32>` uses fp32
//! - `NdArray<half::f16>` uses fp16
//! - `Wgpu<f16, i32>` uses fp16 on GPU
//!
//! The types in this module are for **configuration** purposes - to specify what
//! precision you want when loading models or selecting backends. They do NOT
//! perform runtime precision conversion (which would require backend support).
//!
//! # Runtime upcasting
//!
//! [`Upcasted`] wraps a `Linear<B>` layer to upcast inputs to f32 before the
//! matmul, then downcast the output back. This prevents f16 overflow in large
//! matmul reductions where intermediate sums can exceed the f16 range (~65504).

use burn::nn::Linear;
use burn::prelude::*;
use burn::tensor::DType;

/// Wraps a [`Linear`] layer to upcast inputs to f32 before the matmul, then
/// downcast the output back to the original dtype.
///
/// This prevents f16 overflow in large matmul reductions, where partial sums
/// can exceed the f16 range (~65504). The pattern is:
///
/// ```text
/// input (f16) → cast(f32) → Linear → cast(f16) → output
/// ```
///
/// Use this in place of bare `Linear<B>` at sites where f16 overflow has been
/// observed (e.g. attention projections, large feed-forward layers).
pub struct Upcasted<B: Backend> {
    inner: Linear<B>,
}

impl<B: Backend> Upcasted<B> {
    /// Wraps an existing `Linear` layer.
    pub fn new(inner: Linear<B>) -> Self {
        Self { inner }
    }

    /// Applies the linear layer with f32 upcasting.
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let original_dtype = x.dtype();
        let out = self.inner.forward(x.cast(DType::F32));
        out.cast(original_dtype)
    }
}

/// Precision mode for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrecisionMode {
    /// Full 32-bit precision (default)
    #[default]
    Fp32,
    /// 16-bit half precision
    Fp16,
    /// 16-bit brain floating point
    Bf16,
}

impl PrecisionMode {
    /// Get a human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            PrecisionMode::Fp32 => "fp32",
            PrecisionMode::Fp16 => "fp16",
            PrecisionMode::Bf16 => "bf16",
        }
    }

    /// Memory savings compared to fp32 (approximate)
    pub fn memory_savings(&self) -> f32 {
        match self {
            PrecisionMode::Fp32 => 1.0,
            PrecisionMode::Fp16 => 0.5,
            PrecisionMode::Bf16 => 0.5,
        }
    }
}

/// Configuration for mixed precision inference
#[derive(Debug, Clone, Default)]
pub struct PrecisionConfig {
    /// Precision for UNet weights
    pub unet_precision: PrecisionMode,
    /// Precision for VAE weights
    pub vae_precision: PrecisionMode,
    /// Precision for text encoder weights
    pub text_encoder_precision: PrecisionMode,
    /// Whether to use fp16 for intermediate computations
    pub compute_precision: PrecisionMode,
}

impl PrecisionConfig {
    /// Full fp32 precision (default, most accurate)
    pub fn fp32() -> Self {
        Self::default()
    }

    /// Full fp16 precision (fastest, may have quality loss)
    pub fn fp16() -> Self {
        Self {
            unet_precision: PrecisionMode::Fp16,
            vae_precision: PrecisionMode::Fp16,
            text_encoder_precision: PrecisionMode::Fp16,
            compute_precision: PrecisionMode::Fp16,
        }
    }

    /// Full bf16 precision (good balance)
    pub fn bf16() -> Self {
        Self {
            unet_precision: PrecisionMode::Bf16,
            vae_precision: PrecisionMode::Bf16,
            text_encoder_precision: PrecisionMode::Bf16,
            compute_precision: PrecisionMode::Bf16,
        }
    }

    /// Mixed precision: fp16 for UNet, fp32 for VAE
    /// (recommended - VAE benefits from higher precision)
    pub fn mixed() -> Self {
        Self {
            unet_precision: PrecisionMode::Fp16,
            vae_precision: PrecisionMode::Fp32,
            text_encoder_precision: PrecisionMode::Fp16,
            compute_precision: PrecisionMode::Fp16,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_mode_default() {
        assert_eq!(PrecisionMode::default(), PrecisionMode::Fp32);
    }

    #[test]
    fn test_precision_config_presets() {
        let fp32 = PrecisionConfig::fp32();
        assert_eq!(fp32.unet_precision, PrecisionMode::Fp32);

        let fp16 = PrecisionConfig::fp16();
        assert_eq!(fp16.unet_precision, PrecisionMode::Fp16);

        let mixed = PrecisionConfig::mixed();
        assert_eq!(mixed.unet_precision, PrecisionMode::Fp16);
        assert_eq!(mixed.vae_precision, PrecisionMode::Fp32);
    }

    #[test]
    fn test_memory_savings() {
        assert_eq!(PrecisionMode::Fp32.memory_savings(), 1.0);
        assert_eq!(PrecisionMode::Fp16.memory_savings(), 0.5);
    }
}
