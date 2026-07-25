//! Classifier-Free Guidance utilities
//!
//! Provides shared guidance functions used across samplers.

use burn::prelude::*;

/// Configuration for CFG++ guidance
#[derive(Debug, Clone)]
pub struct CfgPlusPlusConfig {
    /// Guidance rescale factor (0.0 = standard, 0.7 = recommended)
    pub guidance_rescale: f32,
}

impl Default for CfgPlusPlusConfig {
    fn default() -> Self {
        Self {
            guidance_rescale: 0.7,
        }
    }
}

/// Compute standard deviation of a tensor
///
/// Used for guidance rescaling to prevent over-saturation.
pub fn compute_tensor_std<B: Backend>(tensor: &Tensor<B, 4>) -> f32 {
    let flattened = tensor.clone().flatten::<1>(0, 3);
    let var = flattened.var(0);
    let std_tensor = var.sqrt();
    let std_data = std_tensor.into_data();
    std_data.to_vec::<f32>().unwrap()[0]
}

/// Apply CFG++ (Classifier-Free Guidance Plus Plus)
///
/// CFG++ applies guidance in the denoised prediction space rather than
/// the noise prediction space. This reduces artifacts and improves
/// image quality at high guidance scales.
///
/// # Arguments
/// * `noise_pred_uncond` - Unconditional noise prediction
/// * `noise_pred_cond` - Conditional noise prediction
/// * `sample` - Current latent sample
/// * `sigma` - Current noise level
/// * `guidance_scale` - CFG scale (typically 7.0-15.0)
/// * `guidance_rescale` - Rescale factor to prevent saturation (0.0-1.0, 0.7 recommended)
///
/// # Returns
/// The guided denoised prediction (x0 space)
pub fn apply_cfg_plus_plus<B: Backend>(
    noise_pred_uncond: Tensor<B, 4>,
    noise_pred_cond: Tensor<B, 4>,
    sample: Tensor<B, 4>,
    sigma: f32,
    guidance_scale: f32,
    guidance_rescale: f32,
) -> Tensor<B, 4> {
    // Convert noise predictions to x0 predictions
    let x0_uncond = sample.clone() - noise_pred_uncond * sigma;
    let x0_cond = sample - noise_pred_cond * sigma;

    // Apply guidance in x0 space
    let x0_guided = x0_uncond.clone() + (x0_cond.clone() - x0_uncond) * guidance_scale;

    // Optionally rescale to prevent over-saturation
    if guidance_rescale > 0.0 {
        let std_cond = compute_tensor_std(&x0_cond);
        let std_guided = compute_tensor_std(&x0_guided);

        if std_guided > 1e-6 {
            let rescale_factor =
                std_cond / std_guided * guidance_rescale + (1.0 - guidance_rescale);
            x0_guided * rescale_factor
        } else {
            x0_guided
        }
    } else {
        x0_guided
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cfg_plus_plus_config_default() {
        let config = CfgPlusPlusConfig::default();
        assert_eq!(config.guidance_rescale, 0.7);
    }
}
