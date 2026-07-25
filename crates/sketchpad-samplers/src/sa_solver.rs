//! SA-Solver (Stochastic Adams Solver)
//!
//! An advanced multi-step solver that combines Adams-Bashforth predictor
//! with Adams-Moulton corrector for diffusion models.

use burn::prelude::*;
use std::collections::VecDeque;

use crate::scheduler::{
    NoiseSchedule, adams_bashforth_coefficients, compute_sigmas_karras, sampler_timesteps,
};

/// Configuration for SA-Solver
#[derive(Debug, Clone)]
pub struct SaSolverConfig {
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// Predictor order (1-4)
    pub predictor_order: usize,
    /// Corrector order (0-4, 0 = no correction)
    pub corrector_order: usize,
    /// Tau function type for schedule
    pub tau_type: TauType,
    /// Use Karras sigmas
    pub use_karras_sigmas: bool,
}

/// Tau function type for SA-Solver
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TauType {
    /// Linear tau schedule
    #[default]
    Linear,
    /// Polynomial tau schedule
    Polynomial,
    /// Cosine tau schedule
    Cosine,
}

impl Default for SaSolverConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 25,
            predictor_order: 3,
            corrector_order: 2,
            tau_type: TauType::Linear,
            use_karras_sigmas: true,
        }
    }
}

/// SA-Solver sampler
///
/// Stochastic Adams solver uses Adams-Bashforth for prediction and
/// Adams-Moulton for correction, providing high-order accuracy.
pub struct SaSolver<B: Backend> {
    /// Solver configuration
    config: SaSolverConfig,
    /// Timestep indices for sampling
    timesteps: Vec<usize>,
    /// Sigma values at each timestep
    sigmas: Vec<f32>,
    /// History of model outputs (epsilon predictions)
    model_outputs: VecDeque<Tensor<B, 4>>,
    /// History of timestep indices
    timestep_history: VecDeque<usize>,
}

impl<B: Backend> SaSolver<B> {
    /// Create a new SA-Solver
    pub fn new(config: SaSolverConfig, schedule: &NoiseSchedule<B>) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let sigmas = compute_sigmas_karras(schedule, &timesteps, config.use_karras_sigmas);

        Self {
            config,
            timesteps,
            sigmas,
            model_outputs: VecDeque::new(),
            timestep_history: VecDeque::new(),
        }
    }

    /// Get the timesteps
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Reset the history
    pub fn reset(&mut self) {
        self.model_outputs.clear();
        self.timestep_history.clear();
    }

    /// Returns Adams-Bashforth coefficients for given order
    fn get_ab_coefficients(order: usize) -> Vec<f32> {
        adams_bashforth_coefficients(order)
    }

    /// Returns Adams-Moulton coefficients for given order
    fn get_am_coefficients(order: usize) -> Vec<f32> {
        match order {
            0 => vec![],
            1 => vec![1.0],
            2 => vec![0.5, 0.5],
            3 => vec![5.0 / 12.0, 8.0 / 12.0, -1.0 / 12.0],
            4 => vec![9.0 / 24.0, 19.0 / 24.0, -5.0 / 24.0, 1.0 / 24.0],
            _ => vec![1.0],
        }
    }

    /// Performs predictor step using Adams-Bashforth method
    fn predict(&self, sample: &Tensor<B, 4>, sigma: f32, sigma_next: f32) -> Tensor<B, 4> {
        let order = self.model_outputs.len().min(self.config.predictor_order);
        let coeffs = Self::get_ab_coefficients(order);

        // Compute weighted combination of past derivatives
        let h = sigma_next - sigma;
        let mut derivative = Tensor::zeros(sample.dims(), &sample.device());

        for (i, coeff) in coeffs.iter().enumerate().take(order) {
            if let Some(output) = self.model_outputs.get(i) {
                let sigma_i = if i < self.sigmas.len() {
                    self.sigmas[self.timestep_history.get(i).copied().unwrap_or(0)]
                } else {
                    sigma
                };
                let d = (sample.clone() - output.clone() * sigma_i) / sigma_i.max(1e-8);
                derivative = derivative + d * *coeff;
            }
        }

        sample.clone() + derivative * h
    }

    /// Performs corrector step using Adams-Moulton method
    fn correct(
        &self,
        predicted: &Tensor<B, 4>,
        model_output_new: &Tensor<B, 4>,
        sample: &Tensor<B, 4>,
        sigma: f32,
        sigma_next: f32,
    ) -> Tensor<B, 4> {
        if self.config.corrector_order == 0 {
            return predicted.clone();
        }

        let order = (self.model_outputs.len() + 1).min(self.config.corrector_order);
        let coeffs = Self::get_am_coefficients(order);

        if coeffs.is_empty() {
            return predicted.clone();
        }

        let h = sigma_next - sigma;

        // New derivative from predicted point
        let d_new =
            (predicted.clone() - model_output_new.clone() * sigma_next) / sigma_next.max(1e-8);

        let mut derivative = d_new * coeffs[0];

        // Add past derivatives
        for (i, coeff) in coeffs.iter().enumerate().skip(1).take(order - 1) {
            if let Some(output) = self.model_outputs.get(i - 1) {
                let sigma_i = if i - 1 < self.sigmas.len() {
                    self.sigmas[self.timestep_history.get(i - 1).copied().unwrap_or(0)]
                } else {
                    sigma
                };
                let d = (sample.clone() - output.clone() * sigma_i) / sigma_i.max(1e-8);
                derivative = derivative + d * *coeff;
            }
        }

        sample.clone() + derivative * h
    }

    /// Perform one SA-Solver step (predictor only, for use without corrector model call)
    pub fn step(
        &mut self,
        model_output: Tensor<B, 4>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = self.sigmas[timestep_idx + 1];

        // Store for multi-step
        self.model_outputs.push_front(model_output.clone());
        self.timestep_history.push_front(timestep_idx);
        if self.model_outputs.len() > self.config.predictor_order {
            self.model_outputs.pop_back();
            self.timestep_history.pop_back();
        }

        if sigma_next == 0.0 {
            // Final step: return denoised
            return sample.clone() - model_output * sigma;
        }

        // Predictor step
        self.predict(&sample, sigma, sigma_next)
    }

    /// Perform SA-Solver step with correction (requires two model calls)
    pub fn step_with_correction(
        &mut self,
        model_output: Tensor<B, 4>,
        model_output_corrector: Tensor<B, 4>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = self.sigmas[timestep_idx + 1];

        // Store for multi-step
        self.model_outputs.push_front(model_output.clone());
        self.timestep_history.push_front(timestep_idx);
        if self.model_outputs.len() > self.config.predictor_order.max(self.config.corrector_order) {
            self.model_outputs.pop_back();
            self.timestep_history.pop_back();
        }

        if sigma_next == 0.0 {
            return sample.clone() - model_output * sigma;
        }

        // Predictor step
        let predicted = self.predict(&sample, sigma, sigma_next);

        // Corrector step
        self.correct(
            &predicted,
            &model_output_corrector,
            &sample,
            sigma,
            sigma_next,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sa_solver_config_default() {
        let config = SaSolverConfig::default();
        assert_eq!(config.predictor_order, 3);
        assert_eq!(config.corrector_order, 2);
        assert!(config.use_karras_sigmas);
    }

    #[test]
    fn test_ab_coefficients() {
        fn get_ab_coefficients(order: usize) -> Vec<f32> {
            match order {
                1 => vec![1.0],
                2 => vec![1.5, -0.5],
                3 => vec![23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0],
                4 => vec![55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -9.0 / 24.0],
                _ => vec![1.0],
            }
        }

        let coeffs_1 = get_ab_coefficients(1);
        assert_eq!(coeffs_1, vec![1.0]);

        let coeffs_2 = get_ab_coefficients(2);
        assert_eq!(coeffs_2, vec![1.5, -0.5]);
    }

    #[test]
    fn test_am_coefficients() {
        fn get_am_coefficients(order: usize) -> Vec<f32> {
            match order {
                0 => vec![],
                1 => vec![1.0],
                2 => vec![0.5, 0.5],
                3 => vec![5.0 / 12.0, 8.0 / 12.0, -1.0 / 12.0],
                4 => vec![9.0 / 24.0, 19.0 / 24.0, -5.0 / 24.0, 1.0 / 24.0],
                _ => vec![1.0],
            }
        }

        let coeffs_2 = get_am_coefficients(2);
        assert_eq!(coeffs_2, vec![0.5, 0.5]);
    }
}
