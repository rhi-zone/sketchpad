//! Euler sampler for diffusion models
//!
//! Implements the simple Euler method for ODE-based sampling.
//! Fast and produces good results with ~20-30 steps.
//!
//! Uses k-diffusion formulation for ComfyUI/A1111 compatibility.

use burn::prelude::*;

use crate::scheduler::{
    NoiseSchedule, SigmaSchedule, compute_sigmas, init_noise_latent, sampler_timesteps,
};

/// Euler sampler configuration
#[derive(Debug, Clone)]
pub struct EulerConfig {
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// Sigma schedule type
    pub sigma_schedule: SigmaSchedule,
    /// Eta for ancestral sampling (0 = deterministic, 1 = full noise)
    pub eta: f32,
    /// Noise scale multiplier
    pub s_noise: f32,
}

impl Default for EulerConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 30,
            sigma_schedule: SigmaSchedule::Karras,
            eta: 1.0,
            s_noise: 1.0,
        }
    }
}

/// Euler Sampler (ancestral variant)
///
/// Uses the Euler method to solve the diffusion ODE.
/// This is faster than DDIM and often produces good results.
pub struct EulerSampler<B: Backend> {
    /// Timestep indices for sampling
    timesteps: Vec<usize>,
    /// Sigma values at each timestep
    sigmas: Vec<f32>,
    /// Number of inference steps
    num_inference_steps: usize,
    /// Phantom for backend type
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> EulerSampler<B> {
    /// Create a new Euler sampler
    pub fn new(schedule: NoiseSchedule<B>, config: EulerConfig, _device: &B::Device) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let sigmas = compute_sigmas(&schedule, &timesteps, config.sigma_schedule);

        Self {
            timesteps,
            sigmas,
            num_inference_steps: config.num_inference_steps,
            _marker: std::marker::PhantomData,
        }
    }

    /// Get the timesteps for inference
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Get the number of inference steps
    pub fn num_steps(&self) -> usize {
        self.num_inference_steps
    }

    /// Perform one Euler step
    pub fn step(
        &self,
        latent: Tensor<B, 4>,
        noise_pred: Tensor<B, 4>,
        step_index: usize,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[step_index];
        let sigma_next = self.sigmas[step_index + 1];

        // Compute dt
        let dt = sigma_next - sigma;

        // Euler step: x_next = x + dt * dx/dt
        // For diffusion: dx/dt = (x - denoised) / sigma
        // Where denoised = x - sigma * noise_pred

        // Compute denoised estimate
        let denoised = latent.clone() - noise_pred.clone() * sigma;

        // Derivative
        let derivative = (latent.clone() - denoised) / sigma;

        // Euler step
        latent + derivative * dt
    }

    /// Initialize latent with random noise scaled appropriately
    pub fn init_latent(
        &self,
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let sigma_max = self.sigmas[0];
        let noise: Tensor<B, 4> = Tensor::random(
            [batch_size, channels, height, width],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );
        noise * sigma_max
    }
}

/// Euler Ancestral sampler
///
/// Adds noise during sampling for more stochastic results.
/// Uses k-diffusion formulation with configurable eta.
/// Often produces more creative outputs.
pub struct EulerAncestralSampler<B: Backend> {
    /// Timestep indices for sampling
    timesteps: Vec<usize>,
    /// Sigma values at each timestep
    sigmas: Vec<f32>,
    /// Number of inference steps
    num_inference_steps: usize,
    /// Eta for noise injection (0 = ODE, 1 = full ancestral)
    eta: f32,
    /// Noise scale multiplier
    s_noise: f32,
    /// Phantom for backend type
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> EulerAncestralSampler<B> {
    /// Create a new Euler Ancestral sampler
    pub fn new(schedule: NoiseSchedule<B>, config: EulerConfig, _device: &B::Device) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let sigmas = compute_sigmas(&schedule, &timesteps, config.sigma_schedule);

        Self {
            timesteps,
            sigmas,
            num_inference_steps: config.num_inference_steps,
            eta: config.eta,
            s_noise: config.s_noise,
            _marker: std::marker::PhantomData,
        }
    }

    /// Returns the timestep indices used for sampling
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Returns the number of inference steps
    pub fn num_steps(&self) -> usize {
        self.num_inference_steps
    }

    /// Get the sigma values
    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    /// Performs one Euler Ancestral step with stochastic noise injection
    ///
    /// Uses k-diffusion formulation with configurable eta for noise injection.
    pub fn step(
        &self,
        latent: Tensor<B, 4>,
        noise_pred: Tensor<B, 4>,
        step_index: usize,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[step_index];
        let sigma_next = self.sigmas[step_index + 1];

        // Compute denoised
        let denoised = latent.clone() - noise_pred.clone() * sigma;

        if sigma_next == 0.0 {
            // Last step: just return denoised
            return denoised;
        }

        // Compute sigma_up and sigma_down using k-diffusion formula
        // sigma_up = min(sigma_next, eta * sqrt(sigma_next^2 * (sigma^2 - sigma_next^2) / sigma^2))
        let sigma_up_unscaled =
            (sigma_next.powi(2) * (sigma.powi(2) - sigma_next.powi(2)) / sigma.powi(2)).sqrt();
        let sigma_up = (self.eta * sigma_up_unscaled).min(sigma_next) * self.s_noise;
        let sigma_down = (sigma_next.powi(2) - sigma_up.powi(2)).sqrt();

        // Derivative d = (x - denoised) / sigma
        let derivative = (latent.clone() - denoised.clone()) / sigma;

        // Euler step to sigma_down
        let dt = sigma_down - sigma;
        let mut result = latent + derivative * dt;

        // Add noise
        if sigma_up > 0.0 {
            let noise: Tensor<B, 4> = Tensor::random(
                result.shape(),
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &result.device(),
            );
            result = result + noise * sigma_up;
        }

        result
    }

    /// Initializes a random noise latent scaled for the first sigma
    pub fn init_latent(
        &self,
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        init_noise_latent(batch_size, channels, height, width, self.sigmas[0], device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euler_config_default() {
        let config = EulerConfig::default();
        assert_eq!(config.num_inference_steps, 30);
    }
}
