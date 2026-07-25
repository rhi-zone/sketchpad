//! DDIM (Denoising Diffusion Implicit Models) Sampler
//!
//! Implements deterministic sampling for faster inference.

use burn::prelude::*;

use crate::scheduler::NoiseSchedule;

/// DDIM sampler configuration
#[derive(Debug, Clone)]
pub struct DdimConfig {
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// Eta parameter (0.0 = deterministic DDIM, 1.0 = DDPM)
    pub eta: f64,
}

impl Default for DdimConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 50,
            eta: 0.0,
        }
    }
}

/// DDIM Sampler
///
/// Denoising Diffusion Implicit Models enable deterministic sampling
/// with fewer steps than DDPM.
pub struct DdimSampler<B: Backend> {
    /// Noise schedule
    schedule: NoiseSchedule<B>,
    /// Sampler configuration
    config: DdimConfig,
    /// Timestep indices for sampling
    timesteps: Vec<usize>,
}

impl<B: Backend> DdimSampler<B> {
    /// Create a new DDIM sampler
    pub fn new(schedule: NoiseSchedule<B>, config: DdimConfig) -> Self {
        let step_ratio = schedule.num_train_steps / config.num_inference_steps;
        let timesteps: Vec<usize> = (0..config.num_inference_steps)
            .rev()
            .map(|i| i * step_ratio)
            .collect();

        Self {
            schedule,
            config,
            timesteps,
        }
    }

    /// Get the timesteps for inference
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Get the number of inference steps
    pub fn num_steps(&self) -> usize {
        self.config.num_inference_steps
    }

    /// Perform one DDIM step
    ///
    /// Given the current noisy latent and predicted noise, compute the next latent.
    ///
    /// # Arguments
    /// * `latent` - Current noisy latent [batch, channels, height, width]
    /// * `noise_pred` - Predicted noise from UNet [batch, channels, height, width]
    /// * `step_index` - Current step index (0 = highest noise)
    pub fn step(
        &self,
        latent: Tensor<B, 4>,
        noise_pred: Tensor<B, 4>,
        step_index: usize,
    ) -> Tensor<B, 4> {
        let t = self.timesteps[step_index];

        // Get alpha values
        let alpha_cumprod_t = self.schedule.alpha_cumprod_at(t);
        let alpha_cumprod_t_prev = if step_index + 1 < self.timesteps.len() {
            self.schedule
                .alpha_cumprod_at(self.timesteps[step_index + 1])
        } else {
            // Final step: alpha = 1.0
            Tensor::ones([1], &latent.device())
        };

        // Compute predicted x0
        let sqrt_alpha_t = alpha_cumprod_t.clone().sqrt();
        let sqrt_one_minus_alpha_t = (alpha_cumprod_t.clone().neg() + 1.0).sqrt();

        // pred_x0 = (latent - sqrt(1-alpha_t) * noise_pred) / sqrt(alpha_t)
        let pred_x0 = (latent.clone() - noise_pred.clone() * sqrt_one_minus_alpha_t.unsqueeze())
            / sqrt_alpha_t.unsqueeze();

        // Compute direction pointing to x_t
        let sqrt_one_minus_alpha_prev = (alpha_cumprod_t_prev.clone().neg() + 1.0).sqrt();

        // For eta = 0 (deterministic DDIM), sigma = 0
        let sigma = if self.config.eta > 0.0 {
            let one_minus_alpha_prev = alpha_cumprod_t_prev.clone().neg() + 1.0;
            let one_minus_alpha_t = alpha_cumprod_t.clone().neg() + 1.0;
            let alpha_diff = alpha_cumprod_t.neg() + alpha_cumprod_t_prev.clone();
            let variance = one_minus_alpha_prev / one_minus_alpha_t * alpha_diff;
            variance.sqrt() * self.config.eta
        } else {
            Tensor::zeros([1], &latent.device())
        };

        // Direction pointing to x_t
        let dir_xt = (sqrt_one_minus_alpha_prev.powi_scalar(2) - sigma.clone().powi_scalar(2))
            .clamp_min(0.0)
            .sqrt();

        // Compute previous latent
        let sqrt_alpha_prev = alpha_cumprod_t_prev.sqrt();
        let prev_latent = pred_x0 * sqrt_alpha_prev.unsqueeze() + dir_xt.unsqueeze() * noise_pred;

        // Add noise if eta > 0
        if self.config.eta > 0.0 {
            let noise = Tensor::random(
                latent.shape(),
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &latent.device(),
            );
            prev_latent + noise * sigma.unsqueeze()
        } else {
            prev_latent
        }
    }

    /// Initialize latent with random noise
    pub fn init_latent(
        &self,
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        Tensor::random(
            [batch_size, channels, height, width],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        )
    }
}

/// Apply classifier-free guidance
///
/// Combines conditional and unconditional predictions:
/// `output = uncond + guidance_scale * (cond - uncond)`
pub fn apply_guidance<B: Backend>(
    noise_pred_uncond: Tensor<B, 4>,
    noise_pred_cond: Tensor<B, 4>,
    guidance_scale: f64,
) -> Tensor<B, 4> {
    noise_pred_uncond.clone() + (noise_pred_cond - noise_pred_uncond) * guidance_scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ddim_config_default() {
        let config = DdimConfig::default();
        assert_eq!(config.num_inference_steps, 50);
        assert_eq!(config.eta, 0.0);
    }
}
