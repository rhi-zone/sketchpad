//! DEIS (Diffusion Exponential Integrator Sampler)
//!
//! A fast sampler based on exponential integrators for diffusion ODEs.
//! Uses polynomial extrapolation for improved accuracy.

use burn::prelude::*;
use std::collections::VecDeque;

use crate::scheduler::{NoiseSchedule, sampler_timesteps, sigmas_from_timesteps};

/// Configuration for DEIS sampler
#[derive(Debug, Clone)]
pub struct DeisConfig {
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// Order of the method (1-4)
    pub order: usize,
    /// Solver type
    pub solver_type: DeisSolverType,
}

/// DEIS solver type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeisSolverType {
    /// Logarithmic rho schedule (default, better for most cases)
    #[default]
    LogRho,
    /// Polynomial interpolation
    Polynomial,
}

impl Default for DeisConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 20,
            order: 3,
            solver_type: DeisSolverType::LogRho,
        }
    }
}

/// DEIS sampler
///
/// Diffusion Exponential Integrator Sampler uses exponential integrators
/// combined with polynomial extrapolation for fast, high-quality sampling.
pub struct DeisSampler<B: Backend> {
    /// Sampler configuration
    config: DeisConfig,
    /// Timestep indices for sampling
    timesteps: Vec<usize>,
    /// Sigma values at each timestep
    sigmas: Vec<f32>,
    /// Log sigma values for numerical stability
    log_sigmas: Vec<f32>,
    /// History of model outputs for multi-step methods
    model_outputs: VecDeque<Tensor<B, 4>>,
}

impl<B: Backend> DeisSampler<B> {
    /// Create a new DEIS sampler
    pub fn new(config: DeisConfig, schedule: &NoiseSchedule<B>) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let mut sigmas = sigmas_from_timesteps(schedule, &timesteps);
        sigmas.push(0.0);
        let log_sigmas: Vec<f32> = sigmas.iter().map(|s| (s + 1e-10).ln()).collect();

        Self {
            config,
            timesteps,
            sigmas,
            log_sigmas,
            model_outputs: VecDeque::new(),
        }
    }

    /// Get the timesteps
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Reset the model output history
    pub fn reset(&mut self) {
        self.model_outputs.clear();
    }

    /// Compute DEIS coefficients for polynomial interpolation
    fn get_deis_coefficients(&self, order: usize, step_idx: usize) -> Vec<f32> {
        let mut coeffs = vec![0.0; order];
        let h = self.log_sigmas[step_idx + 1] - self.log_sigmas[step_idx];

        match order {
            1 => {
                // First order (Euler)
                coeffs[0] = 1.0;
            }
            2 => {
                // Second order
                if step_idx >= 1 {
                    let h_0 = self.log_sigmas[step_idx] - self.log_sigmas[step_idx - 1];
                    let r = h / h_0;
                    coeffs[0] = 1.0 + 0.5 * r;
                    coeffs[1] = -0.5 * r;
                } else {
                    coeffs[0] = 1.0;
                }
            }
            3 => {
                // Third order
                if step_idx >= 2 {
                    let h_0 = self.log_sigmas[step_idx] - self.log_sigmas[step_idx - 1];
                    let h_1 = self.log_sigmas[step_idx - 1] - self.log_sigmas[step_idx - 2];
                    let r_0 = h / h_0;
                    let r_1 = h_0 / h_1;

                    coeffs[0] = 1.0 + r_0 * (1.0 + r_0) / (1.0 + r_1) / 2.0;
                    coeffs[1] = -r_0 * (1.0 + r_0 + r_1) / (1.0 + r_1) / 2.0;
                    coeffs[2] = r_0 * r_0 * r_1 / (1.0 + r_1) / 2.0;
                } else if step_idx >= 1 {
                    let h_0 = self.log_sigmas[step_idx] - self.log_sigmas[step_idx - 1];
                    let r = h / h_0;
                    coeffs[0] = 1.0 + 0.5 * r;
                    coeffs[1] = -0.5 * r;
                } else {
                    coeffs[0] = 1.0;
                }
            }
            4 => {
                // Fourth order
                if step_idx >= 3 {
                    let h_0 = self.log_sigmas[step_idx] - self.log_sigmas[step_idx - 1];
                    let h_1 = self.log_sigmas[step_idx - 1] - self.log_sigmas[step_idx - 2];
                    let h_2 = self.log_sigmas[step_idx - 2] - self.log_sigmas[step_idx - 3];
                    let r_0 = h / h_0;
                    let r_1 = h_0 / h_1;
                    let r_2 = h_1 / h_2;

                    // Lagrange interpolation coefficients for 4 points
                    let d0 = 1.0 + r_1 + r_1 * r_2;
                    let d1 = 1.0 + r_0;
                    let d2 = 1.0 + r_0 + r_0 * r_1;

                    coeffs[0] = 1.0
                        + r_0 / 2.0 * d1 / d0
                        + r_0 * r_0 / 6.0 * d2 / (d0 * (1.0 + r_2))
                        + r_0 * r_0 * r_0 / 24.0 * (1.0 + r_0 + r_0 * r_1 + r_0 * r_1 * r_2)
                            / (d0 * (1.0 + r_2));
                    coeffs[1] = -r_0 / 2.0 * (1.0 + r_0 + r_1 + r_1 * r_2) / d0
                        - r_0 * r_0 / 6.0 * (1.0 + r_0 + r_0 * r_1) / (d0 * r_1)
                        - r_0 * r_0 * r_0 / 24.0 * (1.0 + r_0) / (d0 * r_1 * r_2);
                    coeffs[2] = r_0 * r_0 / 6.0 * r_1 * (1.0 + r_1 + r_1 * r_2)
                        / (d0 * (1.0 + r_2))
                        + r_0 * r_0 * r_0 / 24.0 * r_1 / (d0 * r_2);
                    coeffs[3] = -r_0 * r_0 * r_0 / 24.0 * r_1 * r_2 / d0;
                } else {
                    // Fall back to lower order if not enough history
                    return self.get_deis_coefficients((step_idx + 1).min(3), step_idx);
                }
            }
            _ => {
                // Order > 4: fall back to fourth order
                return self.get_deis_coefficients(4.min(step_idx + 1), step_idx);
            }
        }

        coeffs
    }

    /// Perform one DEIS step
    pub fn step(
        &mut self,
        model_output: Tensor<B, 4>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = self.sigmas[timestep_idx + 1];

        // Store model output
        self.model_outputs.push_front(model_output.clone());
        if self.model_outputs.len() > self.config.order {
            self.model_outputs.pop_back();
        }

        // Compute denoised prediction
        let denoised = sample.clone() - model_output.clone() * sigma;

        if sigma_next == 0.0 {
            return denoised;
        }

        // Determine effective order
        let order = self.model_outputs.len().min(self.config.order);
        let coeffs = self.get_deis_coefficients(order, timestep_idx);

        // Compute weighted combination of derivatives
        let mut derivative = Tensor::zeros(sample.dims(), &sample.device());
        for (i, coeff) in coeffs.iter().enumerate().take(order) {
            if let Some(output) = self.model_outputs.get(i) {
                derivative = derivative + output.clone() * *coeff;
            }
        }

        // Apply the step
        let sigma_ratio = sigma_next / sigma;
        sample.clone() * sigma_ratio + derivative * (sigma_next - sigma * sigma_ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deis_config_default() {
        let config = DeisConfig::default();
        assert_eq!(config.order, 3);
        assert_eq!(config.num_inference_steps, 20);
    }
}
