//! DPM++ 2S Ancestral samplers
//!
//! Second-order singlestep DPM-Solver++ with ancestral sampling.
//! The "2S" indicates second-order singlestep (vs multistep).
//!
//! Uses k-diffusion formulation for ComfyUI/A1111 compatibility.

use burn::prelude::*;

use crate::guidance::apply_cfg_plus_plus;
use crate::scheduler::{
    NoiseSchedule, SigmaSchedule, compute_sigmas, get_ancestral_step, sampler_timesteps,
};

/// Configuration for DPM++ 2S Ancestral sampler
#[derive(Debug, Clone)]
pub struct Dpm2sAncestralConfig {
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// Sigma schedule type
    pub sigma_schedule: SigmaSchedule,
    /// S_noise parameter for noise scaling
    pub s_noise: f32,
    /// Eta for ancestral sampling (0 = deterministic, 1 = full noise)
    pub eta: f32,
}

impl Default for Dpm2sAncestralConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 25,
            sigma_schedule: SigmaSchedule::Karras,
            s_noise: 1.0,
            eta: 1.0,
        }
    }
}

/// DPM++ 2S Ancestral Sampler
///
/// Second-order singlestep DPM-Solver++ with noise injection.
/// Uses k-diffusion formulation with exponential integrators.
pub struct Dpm2sAncestralSampler<B: Backend> {
    config: Dpm2sAncestralConfig,
    timesteps: Vec<usize>,
    sigmas: Vec<f32>,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> Dpm2sAncestralSampler<B> {
    /// Create a new DPM++ 2S Ancestral sampler
    pub fn new(config: Dpm2sAncestralConfig, schedule: &NoiseSchedule<B>) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let sigmas = compute_sigmas(schedule, &timesteps, config.sigma_schedule);

        Self {
            config,
            timesteps,
            sigmas,
            _marker: std::marker::PhantomData,
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

    /// Perform one DPM++ 2S Ancestral step (k-diffusion formulation)
    ///
    /// This is a singlestep method, requiring two model evaluations per step.
    /// The first evaluation is at the current point, the second at the midpoint.
    pub fn step(
        &self,
        model_output: Tensor<B, 4>,
        model_output_2: Option<Tensor<B, 4>>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = self.sigmas[timestep_idx + 1];

        // Denoised prediction (x0)
        let denoised = sample.clone() - model_output.clone() * sigma;

        if sigma_next == 0.0 {
            return denoised;
        }

        let (sigma_down, sigma_up) = get_ancestral_step(sigma, sigma_next, self.config.eta);

        // k-diffusion formulation using exponential integrators
        let t = -(sigma.ln());
        let t_next = -(sigma_down.ln());
        let h = t_next - t;
        let r = 0.5; // Midpoint

        let result = if let Some(model_output_mid) = model_output_2 {
            // Full second-order step with midpoint evaluation
            // First compute midpoint sample
            let sigma_s = (-(t + r * h)).exp();
            let exp_neg_rh = (-(r * h)).exp();

            // x_2 = (sigma_s / sigma) * x - (1 - exp(-r*h)) * denoised
            let x_mid = sample.clone() * (sigma_s / sigma) + denoised.clone() * (1.0 - exp_neg_rh);

            // Midpoint denoised
            let denoised_mid = x_mid - model_output_mid * sigma_s;

            // Full step with correction
            let exp_neg_h = (-h).exp();
            let sigma_ratio = sigma_down / sigma;

            // x = (sigma_down / sigma) * x - (1 - exp(-h)) * denoised
            //   + ((1 - exp(-h)) - 2*(1 - exp(-r*h))) * (denoised_mid - denoised)
            let base = sample.clone() * sigma_ratio + denoised.clone() * (1.0 - exp_neg_h);
            let correction_coeff = (1.0 - exp_neg_h) - 2.0 * (1.0 - exp_neg_rh);
            base + (denoised_mid - denoised) * correction_coeff
        } else {
            // First-order fallback (Euler with exponential integrator)
            let exp_neg_h = (-h).exp();
            let sigma_ratio = sigma_down / sigma;
            sample.clone() * sigma_ratio + denoised * (1.0 - exp_neg_h)
        };

        // Add ancestral noise
        if sigma_up > 0.0 {
            let noise: Tensor<B, 4> = Tensor::random(
                result.shape(),
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &result.device(),
            );
            result + noise * (sigma_up * self.config.s_noise)
        } else {
            result
        }
    }

    /// Get the midpoint sigma for the second model evaluation
    pub fn get_midpoint_sigma(&self, timestep_idx: usize) -> f32 {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = self.sigmas[timestep_idx + 1];
        let (sigma_down, _) = get_ancestral_step(sigma, sigma_next, self.config.eta);

        // Midpoint in log-sigma space
        let t = -(sigma.ln());
        let t_next = -(sigma_down.ln());
        let h = t_next - t;
        let t_mid = t + 0.5 * h;
        (-t_mid).exp()
    }
}

/// DPM++ 2S Ancestral CFG++ Sampler
///
/// Combines DPM++ 2S Ancestral with CFG++ guidance.
/// Uses k-diffusion formulation.
pub struct Dpm2sAncestralCfgPlusPlusSampler<B: Backend> {
    config: Dpm2sAncestralConfig,
    timesteps: Vec<usize>,
    sigmas: Vec<f32>,
    /// Guidance rescale factor
    pub guidance_rescale: f32,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> Dpm2sAncestralCfgPlusPlusSampler<B> {
    /// Create a new DPM++ 2S Ancestral CFG++ sampler
    pub fn new(
        config: Dpm2sAncestralConfig,
        schedule: &NoiseSchedule<B>,
        guidance_rescale: f32,
    ) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let sigmas = compute_sigmas(schedule, &timesteps, config.sigma_schedule);

        Self {
            config,
            timesteps,
            sigmas,
            guidance_rescale,
            _marker: std::marker::PhantomData,
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

    /// Apply CFG++ guidance in denoised space
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
            self.guidance_rescale,
        )
    }

    /// Perform one step with pre-computed guided x0 (k-diffusion formulation)
    pub fn step(
        &self,
        x0_guided: Tensor<B, 4>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = self.sigmas[timestep_idx + 1];

        if sigma_next == 0.0 {
            return x0_guided;
        }

        // Ancestral step
        let (sigma_down, sigma_up) = get_ancestral_step(sigma, sigma_next, self.config.eta);

        // k-diffusion formulation using exponential integrators
        let t = -(sigma.ln());
        let t_next = -(sigma_down.ln());
        let h = t_next - t;
        let exp_neg_h = (-h).exp();
        let sigma_ratio = sigma_down / sigma;

        let result = sample * sigma_ratio + x0_guided * (1.0 - exp_neg_h);

        // Add noise
        if sigma_up > 0.0 {
            let noise: Tensor<B, 4> = Tensor::random(
                result.shape(),
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &result.device(),
            );
            result + noise * (sigma_up * self.config.s_noise)
        } else {
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dpm2s_ancestral_config_default() {
        let config = Dpm2sAncestralConfig::default();
        assert_eq!(config.num_inference_steps, 25);
        assert_eq!(config.sigma_schedule, SigmaSchedule::Karras);
        assert_eq!(config.s_noise, 1.0);
        assert_eq!(config.eta, 1.0);
    }
}
