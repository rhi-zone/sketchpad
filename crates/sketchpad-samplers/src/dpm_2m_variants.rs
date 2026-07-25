//! DPM++ 2M variant samplers
//!
//! Includes DPM++ 2M CFG++ and DPM++ 2M SDE Heun variants.
//! Uses k-diffusion formulation for ComfyUI/A1111 compatibility.

use burn::prelude::*;

use crate::guidance::apply_cfg_plus_plus;
use crate::scheduler::{NoiseSchedule, SigmaSchedule, compute_sigmas, sampler_timesteps};

/// Configuration for DPM++ 2M CFG++ sampler
#[derive(Debug, Clone)]
pub struct Dpm2mCfgPlusPlusConfig {
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// Guidance rescale factor (0.0 = standard, 0.7 = recommended)
    pub guidance_rescale: f32,
    /// Sigma schedule type
    pub sigma_schedule: SigmaSchedule,
}

impl Default for Dpm2mCfgPlusPlusConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 20,
            guidance_rescale: 0.7,
            sigma_schedule: SigmaSchedule::Karras,
        }
    }
}

/// DPM++ 2M CFG++ Sampler
///
/// Second-order multistep DPM-Solver++ with CFG++ guidance.
/// CFG++ applies guidance in the denoised space for better quality.
/// Uses k-diffusion formulation with exponential integrators.
pub struct Dpm2mCfgPlusPlusSampler<B: Backend> {
    /// Sampler configuration
    config: Dpm2mCfgPlusPlusConfig,
    /// Timestep indices for sampling
    timesteps: Vec<usize>,
    /// Sigma values at each timestep
    sigmas: Vec<f32>,
    /// Previous denoised prediction for multistep
    prev_denoised: Option<Tensor<B, 4>>,
    /// Previous sigma
    prev_sigma: Option<f32>,
}

impl<B: Backend> Dpm2mCfgPlusPlusSampler<B> {
    /// Create a new DPM++ 2M CFG++ sampler
    pub fn new(config: Dpm2mCfgPlusPlusConfig, schedule: &NoiseSchedule<B>) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let sigmas = compute_sigmas(schedule, &timesteps, config.sigma_schedule);

        Self {
            config,
            timesteps,
            sigmas,
            prev_denoised: None,
            prev_sigma: None,
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

    /// Reset the sampler state
    pub fn reset(&mut self) {
        self.prev_denoised = None;
        self.prev_sigma = None;
    }

    /// Apply CFG++ guidance
    pub fn apply_cfg_plus_plus_guidance(
        &self,
        noise_pred_uncond: Tensor<B, 4>,
        noise_pred_cond: Tensor<B, 4>,
        sample: Tensor<B, 4>,
        sigma: f32,
        guidance_scale: f32,
    ) -> Tensor<B, 4> {
        apply_cfg_plus_plus(
            noise_pred_uncond,
            noise_pred_cond,
            sample,
            sigma,
            guidance_scale,
            self.config.guidance_rescale,
        )
    }

    /// Perform one DPM++ 2M CFG++ step (k-diffusion formulation)
    pub fn step(
        &mut self,
        denoised: Tensor<B, 4>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = self.sigmas[timestep_idx + 1];

        if sigma_next == 0.0 {
            self.prev_denoised = None;
            self.prev_sigma = None;
            return denoised;
        }

        // k-diffusion formulation
        let t = -(sigma.ln());
        let t_next = -(sigma_next.ln());
        let h = t_next - t;

        let sigma_ratio = sigma_next / sigma;
        let exp_neg_h = (-h).exp();

        let result = if let (Some(prev_d), Some(prev_s)) = (&self.prev_denoised, self.prev_sigma) {
            // Second order (multistep) - k-diffusion formula
            let t_prev = -(prev_s.ln());
            let h_prev = t - t_prev;
            let r = h_prev / h;

            // k-diffusion: (1 + 1/(2r)) * d - (1/(2r)) * d_prev
            let coeff = 1.0 / (2.0 * r);
            let denoised_d = denoised.clone() * (1.0 + coeff) - prev_d.clone() * coeff;

            sample.clone() * sigma_ratio + denoised_d * (1.0 - exp_neg_h)
        } else {
            // First order
            sample.clone() * sigma_ratio + denoised.clone() * (1.0 - exp_neg_h)
        };

        self.prev_denoised = Some(denoised);
        self.prev_sigma = Some(sigma);
        result
    }
}

/// Configuration for DPM++ 2M SDE Heun sampler
#[derive(Debug, Clone)]
pub struct Dpm2mSdeHeunConfig {
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// Sigma schedule type
    pub sigma_schedule: SigmaSchedule,
    /// Eta for noise injection
    pub eta: f32,
    /// S_noise parameter
    pub s_noise: f32,
}

impl Default for Dpm2mSdeHeunConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 20,
            sigma_schedule: SigmaSchedule::Karras,
            eta: 1.0,
            s_noise: 1.0,
        }
    }
}

/// DPM++ 2M SDE Heun Sampler
///
/// Combines DPM++ 2M multistep with SDE noise injection and
/// Heun's method for the SDE correction step.
/// Uses k-diffusion formulation with exponential integrators.
pub struct Dpm2mSdeHeunSampler<B: Backend> {
    config: Dpm2mSdeHeunConfig,
    timesteps: Vec<usize>,
    sigmas: Vec<f32>,
    /// Previous denoised prediction
    prev_denoised: Option<Tensor<B, 4>>,
    /// Previous sigma
    prev_sigma: Option<f32>,
}

impl<B: Backend> Dpm2mSdeHeunSampler<B> {
    /// Create a new DPM++ 2M SDE Heun sampler
    pub fn new(config: Dpm2mSdeHeunConfig, schedule: &NoiseSchedule<B>) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let sigmas = compute_sigmas(schedule, &timesteps, config.sigma_schedule);

        Self {
            config,
            timesteps,
            sigmas,
            prev_denoised: None,
            prev_sigma: None,
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

    /// Reset the sampler state
    pub fn reset(&mut self) {
        self.prev_denoised = None;
        self.prev_sigma = None;
    }

    /// Perform one DPM++ 2M SDE Heun step (k-diffusion formulation)
    ///
    /// This combines the DPM++ 2M multistep with SDE noise injection
    /// and uses Heun's method for improved accuracy on the first step.
    pub fn step(
        &mut self,
        model_output: Tensor<B, 4>,
        model_output_heun: Option<Tensor<B, 4>>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = self.sigmas[timestep_idx + 1];

        // Denoised prediction
        let denoised = sample.clone() - model_output.clone() * sigma;

        if sigma_next == 0.0 {
            self.prev_denoised = None;
            self.prev_sigma = None;
            return denoised;
        }

        // SDE noise parameters (k-diffusion formulation)
        let t = -(sigma.ln());
        let t_next = -(sigma_next.ln());
        let h = t_next - t;

        let exp_neg_2_eta_h = (-2.0 * self.config.eta * h).exp();
        let sigma_up = sigma_next * (1.0 - exp_neg_2_eta_h).max(0.0).sqrt() * self.config.s_noise;
        let exp_neg_eta_h = (-self.config.eta * h).exp();
        let sigma_down = sigma_next * exp_neg_eta_h;

        let exp_neg_h = (-h).exp();

        // Compute the update
        let result = if let (Some(prev_d), Some(prev_s)) = (&self.prev_denoised, self.prev_sigma) {
            // Second order with multistep correction (k-diffusion formula)
            let t_prev = -(prev_s.ln());
            let h_prev = t - t_prev;
            let r = h_prev / h;

            // k-diffusion: (1 + 1/(2r)) * d - (1/(2r)) * d_prev
            let coeff = 1.0 / (2.0 * r);
            let denoised_d = denoised.clone() * (1.0 + coeff) - prev_d.clone() * coeff;

            let sigma_ratio = sigma_down / sigma;
            sample.clone() * sigma_ratio + denoised_d * (1.0 - exp_neg_h)
        } else if let Some(output_heun) = model_output_heun {
            // Heun's method for first step
            // First, Euler predictor using k-diffusion formulation
            let sigma_ratio_down = sigma_down / sigma;
            let sample_euler =
                sample.clone() * sigma_ratio_down + denoised.clone() * (1.0 - exp_neg_h);

            // Heun corrector
            let denoised_2 = sample_euler.clone() - output_heun * sigma_down;

            // Average denoised predictions
            let denoised_avg = (denoised.clone() + denoised_2) * 0.5;

            sample.clone() * sigma_ratio_down + denoised_avg * (1.0 - exp_neg_h)
        } else {
            // First order (k-diffusion Euler)
            let sigma_ratio = sigma_down / sigma;
            sample.clone() * sigma_ratio + denoised.clone() * (1.0 - exp_neg_h)
        };

        self.prev_denoised = Some(denoised);
        self.prev_sigma = Some(sigma);

        // Add noise for SDE
        if sigma_up > 0.0 {
            let noise: Tensor<B, 4> = Tensor::random(
                result.shape(),
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &result.device(),
            );
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
    fn test_dpm2m_cfg_plus_plus_config_default() {
        let config = Dpm2mCfgPlusPlusConfig::default();
        assert_eq!(config.num_inference_steps, 20);
        assert_eq!(config.guidance_rescale, 0.7);
        assert_eq!(config.sigma_schedule, SigmaSchedule::Karras);
    }

    #[test]
    fn test_dpm2m_sde_heun_config_default() {
        let config = Dpm2mSdeHeunConfig::default();
        assert_eq!(config.num_inference_steps, 20);
        assert_eq!(config.eta, 1.0);
        assert_eq!(config.sigma_schedule, SigmaSchedule::Karras);
    }
}
