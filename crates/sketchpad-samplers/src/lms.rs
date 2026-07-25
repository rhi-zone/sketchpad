//! LMS (Linear Multi-Step) sampler
//!
//! Uses past derivatives to predict future values with higher accuracy.
//! Higher order = more past samples used = better accuracy but more memory.

use burn::prelude::*;
use std::collections::VecDeque;

use crate::scheduler::{NoiseSchedule, sampler_timesteps, sigmas_from_timesteps};

/// Configuration for LMS sampler
#[derive(Debug, Clone)]
pub struct LmsConfig {
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// Order of the LMS method (1-4)
    pub order: usize,
}

impl Default for LmsConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 30,
            order: 4,
        }
    }
}

/// LMS sampler
///
/// Linear Multi-Step method uses a linear combination of previous
/// derivatives to estimate the next sample.
pub struct LmsSampler<B: Backend> {
    /// Sampler configuration
    config: LmsConfig,
    /// Timestep indices for sampling
    timesteps: Vec<usize>,
    /// Sigma values at each timestep
    sigmas: Vec<f32>,
    /// History of derivatives for multi-step
    derivatives: VecDeque<Tensor<B, 4>>,
}

impl<B: Backend> LmsSampler<B> {
    /// Create a new LMS sampler
    pub fn new(config: LmsConfig, schedule: &NoiseSchedule<B>) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let mut sigmas = sigmas_from_timesteps(schedule, &timesteps);
        sigmas.push(0.0);

        Self {
            config,
            timesteps,
            sigmas,
            derivatives: VecDeque::new(),
        }
    }

    /// Get the timesteps
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Reset the derivative history
    pub fn reset(&mut self) {
        self.derivatives.clear();
    }

    /// Compute LMS coefficients for given order
    fn get_lms_coefficients(
        order: usize,
        _t: f32,
        t_prev: f32,
        sigmas: &[f32],
        step_idx: usize,
    ) -> Vec<f32> {
        let mut coeffs = vec![0.0; order];

        for i in 0..order {
            let mut coeff = 1.0;
            for j in 0..order {
                if i != j {
                    let sigma_i = if step_idx >= i {
                        sigmas[step_idx - i]
                    } else {
                        sigmas[0]
                    };
                    let sigma_j = if step_idx >= j {
                        sigmas[step_idx - j]
                    } else {
                        sigmas[0]
                    };
                    coeff *= (t_prev - sigma_j) / (sigma_i - sigma_j + 1e-8);
                }
            }
            coeffs[i] = coeff;
        }

        coeffs
    }

    /// Perform one LMS step
    pub fn step(
        &mut self,
        model_output: Tensor<B, 4>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = self.sigmas[timestep_idx + 1];

        // Compute derivative
        let derivative = (sample.clone() - model_output) / sigma;

        // Add to history
        self.derivatives.push_front(derivative.clone());
        if self.derivatives.len() > self.config.order {
            self.derivatives.pop_back();
        }

        // Determine effective order (limited by history size)
        let order = self.derivatives.len().min(self.config.order);

        if order == 1 {
            // First step: use Euler
            sample + derivative * (sigma_next - sigma)
        } else {
            // LMS with multiple past derivatives
            let coeffs =
                Self::get_lms_coefficients(order, sigma, sigma_next, &self.sigmas, timestep_idx);

            let mut result = sample;
            for (i, coeff) in coeffs.iter().enumerate().take(order) {
                if let Some(d) = self.derivatives.get(i) {
                    result = result + d.clone() * (*coeff * (sigma_next - sigma));
                }
            }
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lms_config_default() {
        let config = LmsConfig::default();
        assert_eq!(config.order, 4);
        assert_eq!(config.num_inference_steps, 30);
    }
}
