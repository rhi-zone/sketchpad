//! Res Multistep (Restart Multistep) sampler
//!
//! A restart-based multistep sampler that can improve quality by
//! periodically restarting from a higher noise level.

use burn::prelude::*;
use std::collections::VecDeque;

use crate::scheduler::{
    NoiseSchedule, compute_sigmas_karras, get_ancestral_step, sampler_timesteps,
};

/// Configuration for Res Multistep sampler
#[derive(Debug, Clone)]
pub struct ResMultistepConfig {
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// Solver order (1-3)
    pub solver_order: usize,
    /// Number of restart iterations
    pub restart_iterations: usize,
    /// Sigma at which to restart (relative to current)
    pub restart_sigma_ratio: f32,
    /// Use Karras sigmas
    pub use_karras_sigmas: bool,
}

impl Default for ResMultistepConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 25,
            solver_order: 2,
            restart_iterations: 1,
            restart_sigma_ratio: 2.0,
            use_karras_sigmas: true,
        }
    }
}

/// Res Multistep Sampler
///
/// Restart-based multistep sampler that periodically adds noise
/// and restarts the sampling process for improved quality.
pub struct ResMultistepSampler<B: Backend> {
    config: ResMultistepConfig,
    timesteps: Vec<usize>,
    sigmas: Vec<f32>,
    /// History of model outputs for multistep
    model_outputs: VecDeque<Tensor<B, 4>>,
    /// Current restart iteration
    current_restart: usize,
}

impl<B: Backend> ResMultistepSampler<B> {
    /// Create a new Res Multistep sampler
    pub fn new(config: ResMultistepConfig, schedule: &NoiseSchedule<B>) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let sigmas = compute_sigmas_karras(schedule, &timesteps, config.use_karras_sigmas);

        Self {
            config,
            timesteps,
            sigmas,
            model_outputs: VecDeque::new(),
            current_restart: 0,
        }
    }

    /// Get the timesteps
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Reset the sampler state
    pub fn reset(&mut self) {
        self.model_outputs.clear();
        self.current_restart = 0;
    }

    /// Check if a restart should occur at this step
    pub fn should_restart(&self, timestep_idx: usize) -> bool {
        if self.current_restart >= self.config.restart_iterations {
            return false;
        }

        // Restart at regular intervals through the sampling process
        let restart_interval = self.timesteps.len() / (self.config.restart_iterations + 1);
        timestep_idx > 0 && timestep_idx % restart_interval == 0
    }

    /// Add noise for restart
    pub fn add_restart_noise(
        &mut self,
        sample: Tensor<B, 4>,
        current_sigma: f32,
    ) -> (Tensor<B, 4>, f32) {
        let target_sigma = current_sigma * self.config.restart_sigma_ratio;

        // Add noise to reach target sigma
        let noise_level = (target_sigma.powi(2) - current_sigma.powi(2)).sqrt();

        let device = sample.device();
        let shape = sample.dims();
        let noise: Tensor<B, 4> =
            Tensor::random(shape, burn::tensor::Distribution::Normal(0.0, 1.0), &device);

        let noised_sample = sample + noise * noise_level;

        // Clear history on restart
        self.model_outputs.clear();
        self.current_restart += 1;

        (noised_sample, target_sigma)
    }

    /// Get multistep coefficients
    fn get_multistep_coefficients(&self, order: usize, step_idx: usize) -> Vec<f32> {
        let mut coeffs = vec![1.0];

        if order >= 2 && step_idx >= 1 {
            let h = self.sigmas[step_idx + 1] - self.sigmas[step_idx];
            let h_prev = self.sigmas[step_idx] - self.sigmas[step_idx - 1];
            let r = h / h_prev;

            coeffs = vec![1.0 + r / 2.0, -r / 2.0];
        }

        if order >= 3 && step_idx >= 2 {
            let h = self.sigmas[step_idx + 1] - self.sigmas[step_idx];
            let h_1 = self.sigmas[step_idx] - self.sigmas[step_idx - 1];
            let h_2 = self.sigmas[step_idx - 1] - self.sigmas[step_idx - 2];
            let r_1 = h / h_1;
            let r_2 = h_1 / h_2;

            coeffs = vec![
                1.0 + r_1 * (1.0 + r_1) / 2.0 / (1.0 + r_2),
                -r_1 * (1.0 + r_1 + r_2) / 2.0 / (1.0 + r_2),
                r_1 * r_1 * r_2 / 2.0 / (1.0 + r_2),
            ];
        }

        coeffs
    }

    /// Perform one Res Multistep step
    pub fn step(
        &mut self,
        model_output: Tensor<B, 4>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = self.sigmas[timestep_idx + 1];

        // Store output for multistep
        self.model_outputs.push_front(model_output.clone());
        if self.model_outputs.len() > self.config.solver_order {
            self.model_outputs.pop_back();
        }

        // Denoised prediction
        let denoised = sample.clone() - model_output.clone() * sigma;

        if sigma_next == 0.0 {
            return denoised;
        }

        // Determine effective order
        let order = self.model_outputs.len().min(self.config.solver_order);
        let coeffs = self.get_multistep_coefficients(order, timestep_idx);

        // Compute weighted combination
        let mut weighted_denoised = Tensor::zeros(denoised.dims(), &denoised.device());

        for (i, coeff) in coeffs.iter().enumerate().take(self.model_outputs.len()) {
            if let Some(output) = self.model_outputs.get(i) {
                let d = sample.clone()
                    - output.clone() * self.sigmas[timestep_idx - i.min(timestep_idx)];
                weighted_denoised = weighted_denoised + d * *coeff;
            }
        }

        // Apply update
        let sigma_ratio = sigma_next / sigma;
        sample * sigma_ratio + weighted_denoised * (1.0 - sigma_ratio)
    }
}

/// Res Multistep with SDE (stochastic variant)
pub struct ResMultistepSdeSampler<B: Backend> {
    config: ResMultistepConfig,
    timesteps: Vec<usize>,
    sigmas: Vec<f32>,
    model_outputs: VecDeque<Tensor<B, 4>>,
    current_restart: usize,
    /// Eta for noise injection
    pub eta: f32,
}

impl<B: Backend> ResMultistepSdeSampler<B> {
    /// Create a new Res Multistep SDE sampler
    pub fn new(config: ResMultistepConfig, schedule: &NoiseSchedule<B>, eta: f32) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let sigmas = compute_sigmas_karras(schedule, &timesteps, config.use_karras_sigmas);

        Self {
            config,
            timesteps,
            sigmas,
            model_outputs: VecDeque::new(),
            current_restart: 0,
            eta,
        }
    }

    /// Get the timesteps
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Reset the sampler state
    pub fn reset(&mut self) {
        self.model_outputs.clear();
        self.current_restart = 0;
    }

    /// Perform one step with SDE noise
    pub fn step(
        &mut self,
        model_output: Tensor<B, 4>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = self.sigmas[timestep_idx + 1];

        self.model_outputs.push_front(model_output.clone());
        if self.model_outputs.len() > self.config.solver_order {
            self.model_outputs.pop_back();
        }

        let denoised = sample.clone() - model_output.clone() * sigma;

        if sigma_next == 0.0 {
            return denoised;
        }

        // Compute sigma_down and sigma_up for SDE
        let (sigma_down, sigma_up) = get_ancestral_step(sigma, sigma_next, self.eta);

        // Deterministic step
        let sigma_ratio = sigma_down / sigma;
        let result = sample.clone() * sigma_ratio + denoised * (1.0 - sigma_ratio);

        // Add noise
        if sigma_up > 0.0 {
            let device = result.device();
            let shape = result.dims();
            let noise: Tensor<B, 4> =
                Tensor::random(shape, burn::tensor::Distribution::Normal(0.0, 1.0), &device);
            result + noise * sigma_up
        } else {
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_res_multistep_config_default() {
        let config = ResMultistepConfig::default();
        assert_eq!(config.num_inference_steps, 25);
        assert_eq!(config.solver_order, 2);
        assert_eq!(config.restart_iterations, 1);
        assert!(config.use_karras_sigmas);
    }
}
