//! Heun sampler implementations
//!
//! Heun's method is a second-order Runge-Kutta method that provides
//! better accuracy than Euler at the cost of two function evaluations per step.

use burn::prelude::*;

use crate::scheduler::{NoiseSchedule, sampler_timesteps, sigmas_from_timesteps};

/// Configuration for Heun sampler
#[derive(Debug, Clone)]
pub struct HeunConfig {
    /// Number of inference steps
    pub num_inference_steps: usize,
}

impl Default for HeunConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 30,
        }
    }
}

/// Heun sampler (second-order)
///
/// Uses Heun's method (improved Euler / modified trapezoidal) for
/// solving the probability flow ODE.
pub struct HeunSampler<B: Backend> {
    timesteps: Vec<usize>,
    sigmas: Vec<f32>,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> HeunSampler<B> {
    /// Create a new Heun sampler
    pub fn new(config: HeunConfig, schedule: &NoiseSchedule<B>) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let sigmas = sigmas_from_timesteps(schedule, &timesteps);

        Self {
            timesteps,
            sigmas,
            _marker: std::marker::PhantomData,
        }
    }

    /// Get the timesteps for this sampler
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Perform one Heun step
    ///
    /// Heun's method:
    /// 1. Compute initial derivative d1 = model(x, t)
    /// 2. Compute Euler step: x_euler = x + dt * d1
    /// 3. Compute derivative at new point: d2 = model(x_euler, t+dt)
    /// 4. Combine: x_next = x + dt * (d1 + d2) / 2
    pub fn step(
        &self,
        model_output: Tensor<B, 4>,
        model_output_2: Option<Tensor<B, 4>>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = if timestep_idx + 1 < self.sigmas.len() {
            self.sigmas[timestep_idx + 1]
        } else {
            0.0
        };

        let dt = sigma_next - sigma;

        // Convert model output to derivative
        let d1 = (sample.clone() - model_output.clone()) / sigma;

        if let Some(model_output_2) = model_output_2 {
            // Full Heun step with second evaluation
            let d2 = (sample.clone() + d1.clone() * dt - model_output_2) / sigma_next.max(1e-8);
            sample + (d1 + d2) * (dt / 2.0)
        } else {
            // Fall back to Euler if no second evaluation
            sample + d1 * dt
        }
    }

    /// Euler step (first part of Heun)
    pub fn euler_step(
        &self,
        model_output: Tensor<B, 4>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = if timestep_idx + 1 < self.sigmas.len() {
            self.sigmas[timestep_idx + 1]
        } else {
            0.0
        };

        let dt = sigma_next - sigma;
        let d = (sample.clone() - model_output) / sigma;

        sample + d * dt
    }
}

/// HeunPP2 sampler (Heun with predictor-corrector)
///
/// A variant of Heun that uses a predictor-corrector approach
/// for improved stability.
pub struct HeunPP2Sampler<B: Backend> {
    timesteps: Vec<usize>,
    sigmas: Vec<f32>,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> HeunPP2Sampler<B> {
    /// Create a new HeunPP2 sampler
    pub fn new(config: HeunConfig, schedule: &NoiseSchedule<B>) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let sigmas = sigmas_from_timesteps(schedule, &timesteps);

        Self {
            timesteps,
            sigmas,
            _marker: std::marker::PhantomData,
        }
    }

    /// Get the timesteps
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Perform one HeunPP2 step
    pub fn step(
        &self,
        model_output: Tensor<B, 4>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
        prev_sample: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = if timestep_idx + 1 < self.sigmas.len() {
            self.sigmas[timestep_idx + 1]
        } else {
            0.0
        };

        // Denoised prediction
        let denoised = sample.clone() - model_output.clone() * sigma;

        // Compute derivative
        let d = (sample.clone() - denoised.clone()) / sigma;

        let dt = sigma_next - sigma;

        if let Some(prev) = prev_sample {
            // Use previous sample for better estimate
            let d_prev = (prev - denoised.clone()) / sigma;
            let d_avg = (d + d_prev) * 0.5;
            denoised + d_avg * sigma_next
        } else {
            // Standard Euler step
            sample + d * dt
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heun_config_default() {
        let config = HeunConfig::default();
        assert_eq!(config.num_inference_steps, 30);
    }
}
