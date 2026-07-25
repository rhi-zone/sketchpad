//! Euler CFG++ samplers
//!
//! CFG++ (Classifier-Free Guidance Plus Plus) is an improved guidance method
//! that applies guidance in the denoised space rather than noise space,
//! leading to better image quality and fewer artifacts.

use burn::prelude::*;

use crate::guidance::apply_cfg_plus_plus;
use crate::scheduler::{
    NoiseSchedule, get_ancestral_step, sampler_timesteps, sigmas_from_timesteps,
};

/// Configuration for Euler CFG++ sampler
#[derive(Debug, Clone)]
pub struct EulerCfgPlusPlusConfig {
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// Guidance rescale factor (0.0 = standard CFG, 0.7 = recommended)
    pub guidance_rescale: f32,
}

impl Default for EulerCfgPlusPlusConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 30,
            guidance_rescale: 0.7,
        }
    }
}

/// Euler CFG++ Sampler
///
/// Uses CFG++ guidance which applies classifier-free guidance in the
/// denoised prediction space rather than the noise prediction space.
/// This reduces artifacts and improves image quality at high guidance scales.
pub struct EulerCfgPlusPlusSampler<B: Backend> {
    config: EulerCfgPlusPlusConfig,
    timesteps: Vec<usize>,
    sigmas: Vec<f32>,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> EulerCfgPlusPlusSampler<B> {
    /// Create a new Euler CFG++ sampler
    pub fn new(config: EulerCfgPlusPlusConfig, schedule: &NoiseSchedule<B>) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let mut sigmas = sigmas_from_timesteps(schedule, &timesteps);
        sigmas.push(0.0);

        Self {
            config,
            timesteps,
            sigmas,
            _marker: std::marker::PhantomData,
        }
    }

    /// Get the timesteps
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Apply CFG++ guidance
    pub fn apply_cfg_plus_plus_guidance(
        &self,
        noise_pred_uncond: Tensor<B, 4>,
        noise_pred_cond: Tensor<B, 4>,
        sample: Tensor<B, 4>,
        sigma: f32,
        guidance_scale: f32,
    ) -> Tensor<B, 4> {
        apply_cfg_plus_plus(
            noise_pred_uncond,
            noise_pred_cond,
            sample,
            sigma,
            guidance_scale,
            self.config.guidance_rescale,
        )
    }

    /// Perform one Euler CFG++ step
    pub fn step(
        &self,
        x0_guided: Tensor<B, 4>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = self.sigmas[timestep_idx + 1];

        // Euler step
        let dt = sigma_next - sigma;
        let denoised = x0_guided;
        let derivative = (sample.clone() - denoised) / sigma;

        sample + derivative * dt
    }
}

/// Euler Ancestral CFG++ Sampler
///
/// Combines CFG++ guidance with ancestral sampling (noise injection).
pub struct EulerAncestralCfgPlusPlusSampler<B: Backend> {
    config: EulerCfgPlusPlusConfig,
    timesteps: Vec<usize>,
    sigmas: Vec<f32>,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> EulerAncestralCfgPlusPlusSampler<B> {
    /// Create a new Euler Ancestral CFG++ sampler
    pub fn new(config: EulerCfgPlusPlusConfig, schedule: &NoiseSchedule<B>) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let mut sigmas = sigmas_from_timesteps(schedule, &timesteps);
        sigmas.push(0.0);

        Self {
            config,
            timesteps,
            sigmas,
            _marker: std::marker::PhantomData,
        }
    }

    /// Get the timesteps
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Apply CFG++ guidance
    pub fn apply_cfg_plus_plus_guidance(
        &self,
        noise_pred_uncond: Tensor<B, 4>,
        noise_pred_cond: Tensor<B, 4>,
        sample: Tensor<B, 4>,
        sigma: f32,
        guidance_scale: f32,
    ) -> Tensor<B, 4> {
        apply_cfg_plus_plus(
            noise_pred_uncond,
            noise_pred_cond,
            sample,
            sigma,
            guidance_scale,
            self.config.guidance_rescale,
        )
    }

    /// Perform one Euler Ancestral CFG++ step
    pub fn step(
        &self,
        x0_guided: Tensor<B, 4>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = self.sigmas[timestep_idx + 1];

        if sigma_next == 0.0 {
            return x0_guided;
        }

        // Compute sigma_up and sigma_down for ancestral sampling (eta=1.0)
        let (sigma_down, sigma_up) = get_ancestral_step(sigma, sigma_next, 1.0);

        // Euler step to sigma_down
        let derivative = (sample.clone() - x0_guided.clone()) / sigma;
        let sample_down = sample + derivative * (sigma_down - sigma);

        // Add noise
        let device = sample_down.device();
        let shape = sample_down.dims();
        let noise: Tensor<B, 4> =
            Tensor::random(shape, burn::tensor::Distribution::Normal(0.0, 1.0), &device);

        sample_down + noise * sigma_up
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euler_cfg_plus_plus_config_default() {
        let config = EulerCfgPlusPlusConfig::default();
        assert_eq!(config.num_inference_steps, 30);
        assert_eq!(config.guidance_rescale, 0.7);
    }
}
