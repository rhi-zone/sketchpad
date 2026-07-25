//! DPM++ (Diffusion Probabilistic Model) Samplers
//!
//! Implements DPM-Solver++ for high-quality, fast sampling.
//! These samplers typically produce excellent results in 15-25 steps.
//!
//! Uses the k-diffusion formulation (ComfyUI/A1111 compatible) for numerical
//! stability and ecosystem compatibility.
//!
//! For algorithm details comparing the paper vs k-diffusion formulations,
//! see `docs/samplers.md` in the repository.

use burn::prelude::*;

use crate::scheduler::{
    NoiseSchedule, SigmaSchedule, compute_sigmas, init_noise_latent, sampler_timesteps,
};

/// DPM++ configuration
#[derive(Debug, Clone)]
pub struct DpmConfig {
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// Solver order (1 = Euler-like, 2 = second-order)
    pub solver_order: usize,
    /// Sigma schedule type
    pub sigma_schedule: SigmaSchedule,
    /// Enable debug logging (sigma values, step details)
    pub debug: bool,
}

impl Default for DpmConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 25,
            solver_order: 2,
            sigma_schedule: SigmaSchedule::Karras,
            debug: false,
        }
    }
}

/// DPM++ 2M Sampler (second-order multistep)
///
/// High-quality sampler that produces excellent results in ~20 steps.
/// Uses a second-order multistep method for better accuracy.
pub struct DpmPlusPlusSampler<B: Backend> {
    /// Timestep indices for sampling
    timesteps: Vec<usize>,
    /// Sigma values at each timestep
    sigmas: Vec<f32>,
    /// Number of inference steps
    num_inference_steps: usize,
    /// Previous model output for multistep
    prev_sample: Option<Tensor<B, 4>>,
    /// Previous sigma
    prev_sigma: Option<f32>,
    /// Debug logging enabled
    debug: bool,
}

impl<B: Backend> DpmPlusPlusSampler<B> {
    /// Create a new DPM++ 2M sampler
    pub fn new(schedule: NoiseSchedule<B>, config: DpmConfig, _device: &B::Device) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let sigmas = compute_sigmas(&schedule, &timesteps, config.sigma_schedule);

        Self {
            timesteps,
            sigmas,
            num_inference_steps: config.num_inference_steps,
            prev_sample: None,
            prev_sigma: None,
            debug: config.debug,
        }
    }

    /// Returns the timestep indices used for sampling
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Returns the number of inference steps
    pub fn num_steps(&self) -> usize {
        self.num_inference_steps
    }

    /// Resets internal state for a new generation
    pub fn reset(&mut self) {
        self.prev_sample = None;
        self.prev_sigma = None;
    }

    /// Get the sigma values
    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    /// Perform one DPM++ 2M step
    ///
    /// Uses the k-diffusion formulation for compatibility with ComfyUI/A1111.
    pub fn step(
        &mut self,
        latent: Tensor<B, 4>,
        noise_pred: Tensor<B, 4>,
        step_index: usize,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[step_index];
        let sigma_next = self.sigmas[step_index + 1];

        if self.debug {
            eprintln!(
                "[dpm++] step {} sigma={:.6} sigma_next={:.6}",
                step_index, sigma, sigma_next
            );
        }

        // t = -log(sigma) in k-diffusion notation
        let t = -(sigma.ln());
        let t_next = if sigma_next > 0.0 {
            -(sigma_next.ln())
        } else {
            f32::INFINITY
        };
        let h = t_next - t;

        // Compute denoised estimate (x0 prediction)
        // For sigma-parameterized models: x0 = x_t - sigma * epsilon
        let denoised = latent.clone() - noise_pred.clone() * sigma;

        // sigma ratio for the update
        let sigma_ratio = sigma_next / sigma;

        // Use second-order when r is reasonable, fall back to first-order for extreme cases
        if self.prev_sample.is_none() || sigma_next == 0.0 {
            // First-order step or final step
            // x = (sigma_next / sigma) * x - expm1(-h) * denoised
            // expm1(-h) = exp(-h) - 1, so -expm1(-h) = 1 - exp(-h)
            let exp_neg_h = (-h).exp();
            let result = latent.clone() * sigma_ratio + denoised.clone() * (1.0 - exp_neg_h);

            self.prev_sample = Some(denoised);
            self.prev_sigma = Some(sigma);

            result
        } else {
            // Second-order multistep (2M)
            let prev_denoised = self.prev_sample.take().unwrap();
            let prev_sigma = self.prev_sigma.take().unwrap();

            let t_prev = -(prev_sigma.ln());
            let h_prev = t - t_prev;
            let r = h_prev / h;

            if self.debug {
                eprintln!("[dpm++] h={:.6} h_prev={:.6} r={:.6}", h, h_prev, r);
            }

            // Second-order correction using k-diffusion formula:
            // denoised_d = (1 + 1/(2r)) * denoised - (1/(2r)) * old_denoised
            let coeff = 1.0 / (2.0 * r);
            let denoised_d = denoised.clone() * (1.0 + coeff) - prev_denoised * coeff;

            // x = (sigma_next / sigma) * x - expm1(-h) * denoised_d
            let exp_neg_h = (-h).exp();
            let result = latent.clone() * sigma_ratio + denoised_d * (1.0 - exp_neg_h);

            self.prev_sample = Some(denoised);
            self.prev_sigma = Some(sigma);

            result
        }
    }

    /// Perform one DPM++ 2M step with pre-computed denoised value
    ///
    /// This variant is used when denoised is computed externally using a scaled input
    /// (c_in * x), while the step formula uses unscaled x for the sigma_ratio term.
    /// This matches ComfyUI's k-diffusion formulation exactly.
    pub fn step_with_denoised(
        &mut self,
        latent: Tensor<B, 4>,
        denoised: Tensor<B, 4>,
        step_index: usize,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[step_index];
        let sigma_next = self.sigmas[step_index + 1];

        if self.debug {
            eprintln!(
                "[dpm++] step {} sigma={:.6} sigma_next={:.6}",
                step_index, sigma, sigma_next
            );
        }

        // t = -log(sigma) in k-diffusion notation
        let t = -(sigma.ln());
        let t_next = if sigma_next > 0.0 {
            -(sigma_next.ln())
        } else {
            f32::INFINITY
        };
        let h = t_next - t;

        // sigma ratio for the update
        let sigma_ratio = sigma_next / sigma;

        if self.prev_sample.is_none() || sigma_next == 0.0 {
            // First-order step or final step
            // x = (sigma_next / sigma) * x - expm1(-h) * denoised
            // expm1(-h) = exp(-h) - 1, so -expm1(-h) = 1 - exp(-h)
            let exp_neg_h = (-h).exp();
            let result = latent * sigma_ratio + denoised.clone() * (1.0 - exp_neg_h);

            self.prev_sample = Some(denoised);
            self.prev_sigma = Some(sigma);

            result
        } else {
            // Second-order multistep (2M)
            let prev_denoised = self.prev_sample.take().unwrap();
            let prev_sigma = self.prev_sigma.take().unwrap();

            let t_prev = -(prev_sigma.ln());
            let h_prev = t - t_prev;
            let r = h_prev / h;

            if self.debug {
                eprintln!("[dpm++] h={:.6} h_prev={:.6} r={:.6}", h, h_prev, r);
            }

            // Second-order correction using k-diffusion formula:
            // denoised_d = (1 + 1/(2r)) * denoised - (1/(2r)) * old_denoised
            let coeff = 1.0 / (2.0 * r);
            let denoised_d = denoised.clone() * (1.0 + coeff) - prev_denoised * coeff;

            // x = (sigma_next / sigma) * x - expm1(-h) * denoised_d
            let exp_neg_h = (-h).exp();
            let result = latent * sigma_ratio + denoised_d * (1.0 - exp_neg_h);

            self.prev_sample = Some(denoised);
            self.prev_sigma = Some(sigma);

            result
        }
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
        init_noise_latent(batch_size, channels, height, width, self.sigmas[0], device)
    }
}

/// DPM++ 2M SDE Sampler (second-order multistep with stochastic noise)
///
/// Combines the second-order multistep method of DPM++ 2M with stochastic
/// noise injection. Uses k-diffusion formulation for ComfyUI/A1111 compatibility.
///
/// Good for creative generation with ~20-30 steps.
pub struct DpmPlusPlusSdeSampler<B: Backend> {
    /// Timestep indices for sampling
    timesteps: Vec<usize>,
    /// Sigma values at each timestep
    sigmas: Vec<f32>,
    /// Number of inference steps
    num_inference_steps: usize,
    /// Noise multiplier (0.0 = deterministic, 1.0 = full SDE)
    eta: f32,
    /// Noise scale multiplier
    s_noise: f32,
    /// Previous denoised for second-order correction
    prev_denoised: Option<Tensor<B, 4>>,
    /// Previous sigma
    prev_sigma: Option<f32>,
}

impl<B: Backend> DpmPlusPlusSdeSampler<B> {
    /// Create a new DPM++ 2M SDE sampler
    pub fn new(
        schedule: NoiseSchedule<B>,
        config: DpmConfig,
        eta: f32,
        _device: &B::Device,
    ) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let sigmas = compute_sigmas(&schedule, &timesteps, config.sigma_schedule);

        Self {
            timesteps,
            sigmas,
            num_inference_steps: config.num_inference_steps,
            eta,
            s_noise: 1.0,
            prev_denoised: None,
            prev_sigma: None,
        }
    }

    /// Create with custom noise scale
    pub fn with_s_noise(mut self, s_noise: f32) -> Self {
        self.s_noise = s_noise;
        self
    }

    /// Returns the timestep indices used for sampling
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Returns the number of inference steps
    pub fn num_steps(&self) -> usize {
        self.num_inference_steps
    }

    /// Get the sigma values
    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    /// Reset state for new generation
    pub fn reset(&mut self) {
        self.prev_denoised = None;
        self.prev_sigma = None;
    }

    /// Performs one DPM++ 2M SDE step
    ///
    /// Uses k-diffusion formulation with exponential integrators and proper
    /// noise injection for SDE sampling.
    pub fn step(
        &mut self,
        latent: Tensor<B, 4>,
        noise_pred: Tensor<B, 4>,
        step_index: usize,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[step_index];
        let sigma_next = self.sigmas[step_index + 1];

        // Compute denoised estimate
        let denoised = latent.clone() - noise_pred.clone() * sigma;

        if sigma_next == 0.0 {
            // Last step: just return denoised
            return denoised;
        }

        // Convert to log-SNR space (t = -log(sigma))
        let t = -(sigma.ln());
        let t_next = -(sigma_next.ln());
        let h = t_next - t;

        // Compute noise injection parameters
        // sigma_up = sigma_next * sqrt(1 - exp(-2*eta*h))
        let exp_neg_2_eta_h = (-2.0 * self.eta * h).exp();
        let sigma_up = sigma_next * (1.0 - exp_neg_2_eta_h).max(0.0).sqrt() * self.s_noise;

        // Effective sigma to step to (accounting for noise we'll add)
        // sigma_down = sigma_next * exp(-eta*h)
        let exp_neg_eta_h = (-self.eta * h).exp();
        let sigma_down = sigma_next * exp_neg_eta_h;

        // Second-order correction using previous denoised
        let denoised_d =
            if let (Some(prev_d), Some(prev_s)) = (&self.prev_denoised, self.prev_sigma) {
                let t_prev = -(prev_s.ln());
                let h_prev = t - t_prev;
                let r = h_prev / h;

                // k-diffusion formula: (1 + 1/(2r)) * d - (1/(2r)) * d_prev
                let coeff = 1.0 / (2.0 * r);
                denoised.clone() * (1.0 + coeff) - prev_d.clone() * coeff
            } else {
                denoised.clone()
            };

        // Store for next step
        self.prev_denoised = Some(denoised);
        self.prev_sigma = Some(sigma);

        // DPM++ update with exponential integrator
        // x = (sigma_down / sigma) * x + (1 - exp(-h)) * denoised_d
        let sigma_ratio = sigma_down / sigma;
        let exp_neg_h = (-h).exp();
        let mut result = latent * sigma_ratio + denoised_d * (1.0 - exp_neg_h);

        // Add noise
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
        init_noise_latent(batch_size, channels, height, width, self.sigmas[0], device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dpm_config_default() {
        let config = DpmConfig::default();
        assert_eq!(config.num_inference_steps, 25);
        assert_eq!(config.solver_order, 2);
    }
}
