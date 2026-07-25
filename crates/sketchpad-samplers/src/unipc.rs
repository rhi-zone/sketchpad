//! UniPC (Unified Predictor-Corrector) sampler
//!
//! A unified framework that combines predictor-corrector methods
//! with multi-step approaches for fast, high-quality sampling.

use burn::prelude::*;
use std::collections::VecDeque;

use crate::scheduler::{NoiseSchedule, compute_sigmas_karras, sampler_timesteps};

/// Configuration for UniPC sampler
#[derive(Debug, Clone)]
pub struct UniPcConfig {
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// Solver order (1-3)
    pub solver_order: usize,
    /// Predictor order (can differ from solver order)
    pub predictor_order: usize,
    /// Corrector order (0 = no correction)
    pub corrector_order: usize,
    /// Use Karras sigmas
    pub use_karras_sigmas: bool,
    /// Lower order final steps
    pub lower_order_final: bool,
}

impl Default for UniPcConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 20,
            solver_order: 3,
            predictor_order: 3,
            corrector_order: 1,
            use_karras_sigmas: true,
            lower_order_final: true,
        }
    }
}

/// UniPC sampler
///
/// UniPC uses a predictor-corrector framework with configurable
/// multi-step methods for both prediction and correction phases.
pub struct UniPcSampler<B: Backend> {
    config: UniPcConfig,
    timesteps: Vec<usize>,
    sigmas: Vec<f32>,
    /// History of model outputs
    model_outputs: VecDeque<Tensor<B, 4>>,
    /// History of timestep lambdas
    lambda_history: VecDeque<f32>,
}

impl<B: Backend> UniPcSampler<B> {
    /// Create a new UniPC sampler
    pub fn new(config: UniPcConfig, schedule: &NoiseSchedule<B>) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let sigmas = compute_sigmas_karras(schedule, &timesteps, config.use_karras_sigmas);

        Self {
            config,
            timesteps,
            sigmas,
            model_outputs: VecDeque::new(),
            lambda_history: VecDeque::new(),
        }
    }

    /// Get the timesteps
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Reset the history
    pub fn reset(&mut self) {
        self.model_outputs.clear();
        self.lambda_history.clear();
    }

    /// Multi-step predictor update
    fn multistep_uni_p_bh_update(
        &self,
        model_output: &Tensor<B, 4>,
        sample: &Tensor<B, 4>,
        sigma: f32,
        sigma_next: f32,
        order: usize,
    ) -> Tensor<B, 4> {
        // Convert to x0 prediction
        let x0_pred = sample.clone() - model_output.clone() * sigma;

        if sigma_next == 0.0 {
            return x0_pred;
        }

        let lambda_t = if sigma > 0.0 {
            let alpha = 1.0 / (1.0 + sigma * sigma).sqrt();
            (alpha / sigma).ln()
        } else {
            0.0
        };

        let lambda_s = if sigma_next > 0.0 {
            let alpha_next = 1.0 / (1.0 + sigma_next * sigma_next).sqrt();
            (alpha_next / sigma_next).ln()
        } else {
            0.0
        };

        let h = lambda_s - lambda_t;
        let alpha_s = 1.0 / (1.0 + sigma_next * sigma_next).sqrt();
        let alpha_t = 1.0 / (1.0 + sigma * sigma).sqrt();

        // First order
        let mut result =
            (sample.clone() / alpha_t) * alpha_s + x0_pred.clone() * ((-h).exp() - 1.0);

        // Higher order corrections
        if order >= 2 && !self.model_outputs.is_empty() {
            if let Some(prev_output) = self.model_outputs.front() {
                let x0_prev = prev_output.clone();
                let d1 = (x0_pred.clone() - x0_prev) / h;
                result = result + d1 * ((-h).exp() * (h + 1.0) - 1.0) / h;
            }
        }

        if order >= 3 && self.model_outputs.len() >= 2 {
            if let (Some(prev1), Some(prev2)) =
                (self.model_outputs.front(), self.model_outputs.get(1))
            {
                let x0_prev1 = prev1.clone();
                let x0_prev2 = prev2.clone();
                let d1 = (x0_pred.clone() - x0_prev1.clone()) / h;
                let d2 = (x0_prev1 - x0_prev2) / h;
                let d2_diff = (d1 - d2) / h;
                result =
                    result + d2_diff * ((-h).exp() * (h * h / 2.0 + h + 1.0) - h - 1.0) / (h * h);
            }
        }

        result
    }

    /// Corrector step
    fn multistep_uni_c_bh_update(
        &self,
        model_output: &Tensor<B, 4>,
        sample: &Tensor<B, 4>,
        predicted: &Tensor<B, 4>,
        sigma: f32,
        sigma_next: f32,
    ) -> Tensor<B, 4> {
        if self.config.corrector_order == 0 || sigma_next == 0.0 {
            return predicted.clone();
        }

        // Simple first-order corrector
        let x0_pred = sample.clone() - model_output.clone() * sigma;
        let x0_pred_corrected = predicted.clone() - model_output.clone() * sigma_next;

        // Blend prediction and correction
        let alpha = 0.5;
        x0_pred.clone() * (1.0 - alpha) + x0_pred_corrected * alpha
    }

    /// Perform one UniPC step
    pub fn step(
        &mut self,
        model_output: Tensor<B, 4>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = self.sigmas[timestep_idx + 1];

        // Determine effective order
        let order = if self.config.lower_order_final && timestep_idx + 1 >= self.timesteps.len() - 1
        {
            1
        } else {
            (self.model_outputs.len() + 1).min(self.config.solver_order)
        };

        // Predictor step
        let predicted =
            self.multistep_uni_p_bh_update(&model_output, &sample, sigma, sigma_next, order);

        // Corrector step (if enabled)
        let result =
            self.multistep_uni_c_bh_update(&model_output, &sample, &predicted, sigma, sigma_next);

        // Store x0 prediction for multi-step
        let x0_pred = sample.clone() - model_output * sigma;
        self.model_outputs.push_front(x0_pred);
        if self.model_outputs.len() > self.config.solver_order {
            self.model_outputs.pop_back();
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unipc_config_default() {
        let config = UniPcConfig::default();
        assert_eq!(config.solver_order, 3);
        assert_eq!(config.corrector_order, 1);
        assert!(config.use_karras_sigmas);
    }
}
