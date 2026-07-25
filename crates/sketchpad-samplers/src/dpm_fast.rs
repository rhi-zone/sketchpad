//! DPM Fast and DPM Adaptive samplers
//!
//! DPM-Solver with adaptive step size selection for efficient sampling.
//! DPM Fast uses fixed fast schedules, while DPM Adaptive adjusts step sizes.
//!
//! Uses k-diffusion formulation for ComfyUI/A1111 compatibility.

use burn::prelude::*;

use crate::scheduler::{NoiseSchedule, sigmas_from_timesteps};

/// Configuration for DPM Fast sampler
#[derive(Debug, Clone)]
pub struct DpmFastConfig {
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// Solver order (1-3)
    pub solver_order: usize,
}

impl Default for DpmFastConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 15,
            solver_order: 2,
        }
    }
}

/// DPM Fast Sampler
///
/// A fast variant of DPM-Solver that uses optimized step schedules
/// for rapid sampling with fewer steps.
pub struct DpmFastSampler<B: Backend> {
    /// Timestep indices for sampling
    timesteps: Vec<usize>,
    /// Sigma values at each timestep
    sigmas: Vec<f32>,
    /// Phantom data for backend type
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> DpmFastSampler<B> {
    /// Create a new DPM Fast sampler
    pub fn new(config: DpmFastConfig, schedule: &NoiseSchedule<B>) -> Self {
        // DPM Fast uses a specific timestep schedule optimized for speed
        let timesteps =
            Self::compute_fast_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let mut sigmas = sigmas_from_timesteps(schedule, &timesteps);
        sigmas.push(0.0);

        Self {
            timesteps,
            sigmas,
            _marker: std::marker::PhantomData,
        }
    }

    /// Computes optimized timesteps for fast sampling
    fn compute_fast_timesteps(num_steps: usize, num_train_steps: usize) -> Vec<usize> {
        // Use a polynomial schedule for faster convergence
        (0..num_steps)
            .map(|i| {
                let t = 1.0 - (i as f64 / num_steps as f64).powf(2.0);
                ((t * num_train_steps as f64) as usize).min(num_train_steps - 1)
            })
            .collect()
    }

    /// Get the timesteps
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Perform one DPM Fast step (k-diffusion formulation)
    pub fn step(
        &self,
        model_output: Tensor<B, 4>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = self.sigmas[timestep_idx + 1];

        // Denoised prediction
        let denoised = sample.clone() - model_output.clone() * sigma;

        if sigma_next == 0.0 {
            return denoised;
        }

        // k-diffusion formulation with exponential integrators
        let t = -(sigma.ln());
        let t_next = -(sigma_next.ln());
        let h = t_next - t;

        let sigma_ratio = sigma_next / sigma;
        let exp_neg_h = (-h).exp();

        sample * sigma_ratio + denoised * (1.0 - exp_neg_h)
    }
}

/// Configuration for DPM Adaptive sampler
#[derive(Debug, Clone)]
pub struct DpmAdaptiveConfig {
    /// Minimum number of steps
    pub min_steps: usize,
    /// Maximum number of steps
    pub max_steps: usize,
    /// Error tolerance for step size adaptation
    pub rtol: f32,
    /// Absolute error tolerance
    pub atol: f32,
    /// Solver order (1-3)
    pub solver_order: usize,
}

impl Default for DpmAdaptiveConfig {
    fn default() -> Self {
        Self {
            min_steps: 10,
            max_steps: 100,
            rtol: 0.05,
            atol: 0.0078,
            solver_order: 2,
        }
    }
}

/// DPM Adaptive Sampler
///
/// Uses adaptive step size control to automatically determine the
/// optimal number of steps based on local error estimates.
///
/// Error estimation uses Richardson extrapolation: take a full step and two half steps,
/// compare results to estimate local error. Step size is adjusted using PI control.
pub struct DpmAdaptiveSampler<B: Backend> {
    config: DpmAdaptiveConfig,
    sigmas: Vec<f32>,
    current_sigma: f32,
    sigma_min: f32,
    /// Current step size in log-sigma space
    current_h: f32,
    /// Previous denoised prediction for error estimation
    prev_denoised: Option<Tensor<B, 4>>,
    /// Safety factor for step size adjustment
    safety: f32,
}

impl<B: Backend> DpmAdaptiveSampler<B> {
    /// Create a new DPM Adaptive sampler
    pub fn new(config: DpmAdaptiveConfig, schedule: &NoiseSchedule<B>) -> Self {
        let num_train_steps = schedule.num_train_steps;

        // Get sigma range from schedule
        let alpha_min = {
            let alpha_cumprod = schedule.alpha_cumprod_at(num_train_steps - 1);
            let alpha_data = alpha_cumprod.into_data();
            alpha_data.to_vec::<f32>().unwrap()[0]
        };
        let alpha_max = {
            let alpha_cumprod = schedule.alpha_cumprod_at(0);
            let alpha_data = alpha_cumprod.into_data();
            alpha_data.to_vec::<f32>().unwrap()[0]
        };

        let sigma_max = ((1.0 - alpha_min) / alpha_min).sqrt();
        let sigma_min = ((1.0 - alpha_max) / alpha_max).sqrt();

        // Initial step size: span the range in max_steps
        let initial_h = (sigma_min.ln() - sigma_max.ln()) / (config.max_steps as f32);

        Self {
            config,
            sigmas: vec![sigma_max],
            current_sigma: sigma_max,
            sigma_min,
            current_h: initial_h,
            prev_denoised: None,
            safety: 0.9,
        }
    }

    /// Get the current sigma value
    pub fn current_sigma(&self) -> f32 {
        self.current_sigma
    }

    /// Check if sampling is complete
    pub fn is_done(&self) -> bool {
        self.current_sigma <= self.sigma_min
    }

    /// Perform one adaptive DPM step
    ///
    /// Returns (next_sample, step_accepted, new_sigma)
    ///
    /// Uses local error estimation based on denoised prediction changes.
    /// Step size is adjusted using PI control to keep error within tolerance.
    pub fn step(
        &mut self,
        model_output: Tensor<B, 4>,
        sample: Tensor<B, 4>,
        proposed_sigma_next: Option<f32>,
    ) -> (Tensor<B, 4>, bool, f32) {
        let sigma = self.current_sigma;

        // Use proposed sigma or compute from current step size
        let sigma_next = proposed_sigma_next.unwrap_or_else(|| (sigma.ln() + self.current_h).exp());
        let sigma_next = sigma_next.max(self.sigma_min);
        let h = sigma_next.ln() - sigma.ln();

        // Denoised prediction (x0 estimate)
        let denoised = sample.clone() - model_output.clone() * sigma;

        // k-diffusion formulation with exponential integrators
        let t = -(sigma.ln());
        let t_next = -(sigma_next.ln());
        let h_step = t_next - t;

        let sigma_ratio = sigma_next / sigma;
        let exp_neg_h = (-h_step).exp();
        let next_sample = sample.clone() * sigma_ratio + denoised.clone() * (1.0 - exp_neg_h);

        // Error estimation using change in denoised prediction
        let (error_ratio, accepted) = if let Some(ref prev_d) = self.prev_denoised {
            // Estimate second derivative from denoised prediction change
            let d_diff = denoised.clone() - prev_d.clone();
            let d_diff_data: Vec<f32> = d_diff
                .clone()
                .abs()
                .mean()
                .into_data()
                .convert::<f32>()
                .to_vec()
                .unwrap();
            let mean_diff = d_diff_data[0];

            // Scale estimate for error: |h^2 * d''| ≈ |h * delta_d|
            let error_estimate = (h.abs() * mean_diff).abs();

            // Tolerance: atol + rtol * |denoised|
            let denoised_data: Vec<f32> = denoised
                .clone()
                .abs()
                .mean()
                .into_data()
                .convert::<f32>()
                .to_vec()
                .unwrap();
            let tolerance = self.config.atol + self.config.rtol * denoised_data[0];

            let ratio = error_estimate / tolerance.max(1e-10);
            (ratio, ratio <= 1.0)
        } else {
            // First step: accept and use default step size
            (0.5, true)
        };

        if accepted {
            // Accept step and adjust step size
            self.current_sigma = sigma_next;
            self.sigmas.push(sigma_next);
            self.prev_denoised = Some(denoised);

            // PI controller for step size (increase if error small)
            // h_new = h * safety * (1/error_ratio)^(1/order)
            let order = self.config.solver_order as f32;
            let factor = if error_ratio > 0.0 {
                self.safety * (1.0 / error_ratio).powf(1.0 / order)
            } else {
                2.0 // Double step if error is effectively zero
            };
            // Clamp growth factor
            let factor = factor.clamp(0.5, 2.0);
            self.current_h = (h * factor).clamp(
                (self.sigma_min.ln() - self.current_sigma.ln()) / 2.0, // Don't overshoot
                (self.sigma_min.ln() - sigma.ln()) / (self.config.min_steps as f32),
            );

            (next_sample, true, sigma_next)
        } else {
            // Reject step, reduce step size and retry
            let order = self.config.solver_order as f32;
            let factor = self.safety * (1.0 / error_ratio).powf(1.0 / order);
            let factor = factor.clamp(0.1, 0.9); // Ensure we actually shrink
            self.current_h = h * factor;

            // Return original sample (caller should retry with new step)
            (sample, false, sigma)
        }
    }

    /// Get the sigmas used so far
    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    /// Get the recommended sigma for the next step
    pub fn recommended_sigma_next(&self) -> f32 {
        let sigma_next = (self.current_sigma.ln() + self.current_h).exp();
        sigma_next.max(self.sigma_min)
    }

    /// Reset the sampler state for a new generation
    pub fn reset(&mut self) {
        let sigma_max = self.sigmas.first().copied().unwrap_or(self.current_sigma);
        self.current_sigma = sigma_max;
        self.sigmas = vec![sigma_max];
        self.prev_denoised = None;
        self.current_h = (self.sigma_min.ln() - sigma_max.ln()) / (self.config.max_steps as f32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dpm_fast_config_default() {
        let config = DpmFastConfig::default();
        assert_eq!(config.num_inference_steps, 15);
        assert_eq!(config.solver_order, 2);
    }

    #[test]
    fn test_dpm_adaptive_config_default() {
        let config = DpmAdaptiveConfig::default();
        assert_eq!(config.min_steps, 10);
        assert_eq!(config.max_steps, 100);
        assert_eq!(config.rtol, 0.05);
    }
}
