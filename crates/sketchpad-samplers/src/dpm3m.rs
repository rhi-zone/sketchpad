//! DPM++ 3M SDE sampler
//!
//! Third-order DPM-Solver++ with SDE (stochastic differential equation) formulation.
//! Uses k-diffusion formulation for ComfyUI/A1111 compatibility.
//!
//! Provides higher quality than 2M variants by using polynomial extrapolation
//! with two previous denoised predictions.

use burn::prelude::*;

use crate::scheduler::{NoiseSchedule, SigmaSchedule, compute_sigmas, sampler_timesteps};

/// Configuration for DPM++ 3M SDE sampler
#[derive(Debug, Clone)]
pub struct Dpm3mSdeConfig {
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// Sigma schedule type
    pub sigma_schedule: SigmaSchedule,
    /// Eta for noise injection (0 = ODE, 1 = full SDE)
    pub eta: f32,
    /// S_noise parameter for noise scaling
    pub s_noise: f32,
}

impl Default for Dpm3mSdeConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 20,
            sigma_schedule: SigmaSchedule::Karras,
            eta: 1.0,
            s_noise: 1.0,
        }
    }
}

/// DPM++ 3M SDE sampler
///
/// Third-order DPM-Solver++ with stochastic noise injection.
/// Uses k-diffusion formulation with exponential integrators.
///
/// Tracks two previous denoised predictions for third-order polynomial
/// extrapolation.
pub struct Dpm3mSdeSampler<B: Backend> {
    /// Sampler configuration
    config: Dpm3mSdeConfig,
    /// Timestep indices for sampling
    timesteps: Vec<usize>,
    /// Sigma values at each timestep
    sigmas: Vec<f32>,
    /// Previous denoised (most recent)
    denoised_1: Option<Tensor<B, 4>>,
    /// Previous denoised (second most recent)
    denoised_2: Option<Tensor<B, 4>>,
    /// Previous step size h_1 (most recent)
    h_1: Option<f32>,
    /// Previous step size h_2 (second most recent)
    h_2: Option<f32>,
}

impl<B: Backend> Dpm3mSdeSampler<B> {
    /// Create a new DPM++ 3M SDE sampler
    pub fn new(config: Dpm3mSdeConfig, schedule: &NoiseSchedule<B>) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let sigmas = compute_sigmas(schedule, &timesteps, config.sigma_schedule);

        Self {
            config,
            timesteps,
            sigmas,
            denoised_1: None,
            denoised_2: None,
            h_1: None,
            h_2: None,
        }
    }

    /// Get the timesteps
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Get the sigma values
    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    /// Number of inference steps
    pub fn num_steps(&self) -> usize {
        self.config.num_inference_steps
    }

    /// Reset the sampler state
    pub fn reset(&mut self) {
        self.denoised_1 = None;
        self.denoised_2 = None;
        self.h_1 = None;
        self.h_2 = None;
    }

    /// Perform one DPM++ 3M SDE step
    ///
    /// Uses k-diffusion formulation with exponential integrators and
    /// third-order polynomial extrapolation.
    pub fn step(
        &mut self,
        model_output: Tensor<B, 4>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = self.sigmas[timestep_idx + 1];

        // Convert to denoised
        let denoised = sample.clone() - model_output.clone() * sigma;

        if sigma_next == 0.0 {
            return denoised;
        }

        // Log-SNR space
        let t = -(sigma.ln());
        let t_next = -(sigma_next.ln());
        let h = t_next - t;

        // Noise injection parameters (k-diffusion SDE formulation)
        let eta = self.config.eta;
        let exp_neg_2_eta_h = (-2.0 * eta * h).exp();
        let sigma_up = sigma_next * (1.0 - exp_neg_2_eta_h).max(0.0).sqrt() * self.config.s_noise;
        let exp_neg_eta_h = (-eta * h).exp();
        let sigma_down = sigma_next * exp_neg_eta_h;

        // Exponential integrator coefficient
        let exp_neg_h = (-h).exp();

        // Compute corrected denoised based on history
        let denoised_d = if let (Some(d1), Some(h_1)) = (&self.denoised_1, self.h_1) {
            if let (Some(d2), Some(h_2)) = (&self.denoised_2, self.h_2) {
                // Third order (3M)
                let r0 = h_1 / h;
                let r1 = h_2 / h;

                // Finite differences
                let d1_0 = denoised.clone() - d1.clone();
                let d1_1 = d1.clone() - d2.clone();
                let d2_0 = d1_0.clone() - d1_1.clone() * (r0 / r1);

                // Third-order polynomial extrapolation coefficients
                // phi_2 = 1/(2r0), phi_3 = 1/(6*r0*(r0+r1))
                let phi_2 = 1.0 / (2.0 * r0);
                let phi_3 = 1.0 / (6.0 * r0 * (r0 + r1));

                denoised.clone() + d1_0 * phi_2 + d2_0 * phi_3
            } else {
                // Second order (2M style)
                let r = h_1 / h;
                let coeff = 1.0 / (2.0 * r);
                denoised.clone() * (1.0 + coeff) - d1.clone() * coeff
            }
        } else {
            // First order
            denoised.clone()
        };

        // Update history
        self.h_2 = self.h_1;
        self.denoised_2 = self.denoised_1.take();
        self.h_1 = Some(h);
        self.denoised_1 = Some(denoised);

        // DPM++ update with exponential integrator
        let sigma_ratio = sigma_down / sigma;
        let mut result = sample * sigma_ratio + denoised_d * (1.0 - exp_neg_h);

        // Add noise for SDE
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
        let sigma_max = self.sigmas[0];
        let noise: Tensor<B, 4> = Tensor::random(
            [batch_size, channels, height, width],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );
        noise * sigma_max
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dpm3m_sde_config_default() {
        let config = Dpm3mSdeConfig::default();
        assert_eq!(config.num_inference_steps, 20);
        assert_eq!(config.sigma_schedule, SigmaSchedule::Karras);
        assert_eq!(config.eta, 1.0);
    }
}
