//! DPM2 (Diffusion Probabilistic Models 2nd order) samplers
//!
//! Second-order solvers for the diffusion ODE/SDE.

use burn::prelude::*;

use crate::scheduler::{
    NoiseSchedule, compute_sigmas_karras, sampler_timesteps, sigmas_from_timesteps,
};

/// Configuration for DPM2 sampler
#[derive(Debug, Clone)]
pub struct Dpm2Config {
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// Solver order
    pub solver_order: usize,
    /// Use Karras sigmas
    pub use_karras_sigmas: bool,
}

impl Default for Dpm2Config {
    fn default() -> Self {
        Self {
            num_inference_steps: 30,
            solver_order: 2,
            use_karras_sigmas: false,
        }
    }
}

/// DPM2 Sampler (second-order DPM solver using midpoint method)
///
/// True second-order DPM2 requires two model evaluations per step:
/// 1. Evaluate at current sigma to get derivative
/// 2. Take half step to sigma_mid
/// 3. Evaluate at sigma_mid to get midpoint derivative
/// 4. Use midpoint derivative for the full step
///
/// Use `midpoint_sample` to get the intermediate sample, then call
/// `step` with both model outputs for true second-order accuracy.
pub struct Dpm2Sampler<B: Backend> {
    timesteps: Vec<usize>,
    sigmas: Vec<f32>,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> Dpm2Sampler<B> {
    /// Create a new DPM2 sampler
    pub fn new(config: Dpm2Config, schedule: &NoiseSchedule<B>) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let sigmas = compute_sigmas_karras(schedule, &timesteps, config.use_karras_sigmas);

        Self {
            timesteps,
            sigmas,
            _marker: std::marker::PhantomData,
        }
    }

    /// Get timesteps
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Get the sigma value for a given step index
    pub fn sigma(&self, timestep_idx: usize) -> f32 {
        self.sigmas[timestep_idx]
    }

    /// Get the midpoint sigma between current and next step
    pub fn sigma_mid(&self, timestep_idx: usize) -> f32 {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = self.sigmas[timestep_idx + 1];
        (sigma * sigma_next).sqrt()
    }

    /// Compute the intermediate sample at sigma_mid for second model evaluation
    ///
    /// Returns (midpoint_sample, sigma_mid) for use with the model.
    pub fn midpoint_sample(
        &self,
        model_output: Tensor<B, 4>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, f32) {
        let sigma = self.sigmas[timestep_idx];
        let sigma_mid = self.sigma_mid(timestep_idx);

        // Compute derivative at current sigma
        let denoised = sample.clone() - model_output * sigma;
        let d = (sample.clone() - denoised) / sigma;

        // Take half step to sigma_mid
        let sample_mid = sample + d * (sigma_mid - sigma);

        (sample_mid, sigma_mid)
    }

    /// Perform one DPM2 step
    ///
    /// For true second-order accuracy, provide `model_output_mid` from evaluating
    /// the model at the sample returned by `midpoint_sample`. If not provided,
    /// falls back to first-order (Euler) accuracy.
    pub fn step(
        &self,
        model_output: Tensor<B, 4>,
        model_output_mid: Option<Tensor<B, 4>>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = self.sigmas[timestep_idx + 1];

        if sigma_next == 0.0 {
            // Final step: just denoise
            return sample.clone() - model_output * sigma;
        }

        let sigma_mid = self.sigma_mid(timestep_idx);

        // Compute derivative at current sigma
        let denoised = sample.clone() - model_output.clone() * sigma;
        let d = (sample.clone() - denoised) / sigma;

        let d_mid = if let Some(output_mid) = model_output_mid {
            // True second-order: use derivative at midpoint
            let sample_mid = sample.clone() + d.clone() * (sigma_mid - sigma);
            let denoised_mid = sample_mid.clone() - output_mid * sigma_mid;
            (sample_mid - denoised_mid) / sigma_mid
        } else {
            // Fallback to first-order: reuse current derivative
            d
        };

        // Full step using midpoint derivative
        sample + d_mid * (sigma_next - sigma)
    }
}

/// DPM2 Ancestral sampler (with noise injection)
pub struct Dpm2AncestralSampler<B: Backend> {
    timesteps: Vec<usize>,
    sigmas: Vec<f32>,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> Dpm2AncestralSampler<B> {
    /// Create a new DPM2 Ancestral sampler
    pub fn new(config: Dpm2Config, schedule: &NoiseSchedule<B>) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let mut sigmas = sigmas_from_timesteps(schedule, &timesteps);
        sigmas.push(0.0);

        Self {
            timesteps,
            sigmas,
            _marker: std::marker::PhantomData,
        }
    }

    /// Get timesteps
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Perform one DPM2 Ancestral step
    pub fn step(
        &self,
        model_output: Tensor<B, 4>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = self.sigmas[timestep_idx + 1];

        if sigma_next == 0.0 {
            return sample.clone() - model_output * sigma;
        }

        // Ancestral sampling adds noise
        let sigma_up =
            (sigma_next.powi(2) * (sigma.powi(2) - sigma_next.powi(2)) / sigma.powi(2)).sqrt();
        let sigma_down = (sigma_next.powi(2) - sigma_up.powi(2)).sqrt();

        // Denoising step
        let denoised = sample.clone() - model_output * sigma;
        let d = (sample.clone() - denoised.clone()) / sigma;

        let sample_next = sample + d * (sigma_down - sigma);

        // Add noise
        if sigma_up > 0.0 {
            let device = sample_next.device();
            let shape = sample_next.dims();
            let noise: Tensor<B, 4> =
                Tensor::random(shape, burn::tensor::Distribution::Normal(0.0, 1.0), &device);
            sample_next + noise * sigma_up
        } else {
            sample_next
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dpm2_config_default() {
        let config = Dpm2Config::default();
        assert_eq!(config.solver_order, 2);
        assert!(!config.use_karras_sigmas);
    }
}
