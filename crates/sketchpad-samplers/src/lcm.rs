//! LCM (Latent Consistency Model) sampler
//!
//! A distillation-based sampler that enables high-quality generation
//! in very few steps (1-8 steps typically).

use burn::prelude::*;

use crate::scheduler::NoiseSchedule;

/// Configuration for LCM sampler
#[derive(Debug, Clone)]
pub struct LcmConfig {
    /// Number of inference steps (typically 1-8)
    pub num_inference_steps: usize,
    /// Original number of training steps
    pub original_inference_steps: usize,
    /// Guidance scale (usually lower than other samplers, 1.0-2.0)
    pub guidance_scale: f32,
}

impl Default for LcmConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 4,
            original_inference_steps: 50,
            guidance_scale: 1.5,
        }
    }
}

/// LCM Sampler
///
/// Latent Consistency Models use a consistency distillation approach
/// to generate high-quality samples in very few steps.
pub struct LcmSampler<B: Backend> {
    /// Timestep indices for sampling
    timesteps: Vec<usize>,
    /// Cumulative product of alphas
    alphas_cumprod: Vec<f32>,
    /// Phantom data for backend type
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> LcmSampler<B> {
    /// Create a new LCM sampler
    pub fn new(config: LcmConfig, schedule: &NoiseSchedule<B>) -> Self {
        let num_train_timesteps = schedule.num_train_steps;
        let timesteps = Self::compute_timesteps(&config, num_train_timesteps);

        // Compute alphas_cumprod from schedule
        let mut alphas_cumprod = Vec::with_capacity(num_train_timesteps);
        for t in 0..num_train_timesteps {
            let alpha_cumprod = schedule.alpha_cumprod_at(t);
            let alpha_data = alpha_cumprod.into_data();
            let alpha: f32 = alpha_data.convert::<f32>().to_vec().unwrap()[0];
            alphas_cumprod.push(alpha);
        }

        Self {
            timesteps,
            alphas_cumprod,
            _marker: std::marker::PhantomData,
        }
    }

    /// Computes timestep indices for LCM sampling
    fn compute_timesteps(config: &LcmConfig, num_train_timesteps: usize) -> Vec<usize> {
        // LCM uses specific timestep spacing based on original inference steps
        let c = num_train_timesteps / config.original_inference_steps;
        let lcm_origin_steps = config.original_inference_steps;

        // Generate timesteps for LCM
        let step_ratio = lcm_origin_steps / config.num_inference_steps;

        (0..config.num_inference_steps)
            .map(|i| {
                // Reverse order, skipping by step_ratio
                let idx = (config.num_inference_steps - 1 - i) * step_ratio;
                (idx * c + c - 1).min(num_train_timesteps - 1)
            })
            .collect()
    }

    /// Get the timesteps
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Gets predicted original sample (x0) from model output using epsilon prediction
    fn get_predicted_original(
        &self,
        model_output: &Tensor<B, 4>,
        sample: &Tensor<B, 4>,
        timestep: usize,
    ) -> Tensor<B, 4> {
        let alpha_prod_t = self.alphas_cumprod[timestep];
        let beta_prod_t = 1.0 - alpha_prod_t;

        // Assuming epsilon prediction
        // x_0 = (x_t - sqrt(1 - alpha_cumprod_t) * epsilon) / sqrt(alpha_cumprod_t)
        (sample.clone() - model_output.clone() * beta_prod_t.sqrt()) / alpha_prod_t.sqrt()
    }

    /// Perform one LCM step
    pub fn step(
        &self,
        model_output: Tensor<B, 4>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let timestep = self.timesteps[timestep_idx];

        // Get predicted x0
        let pred_original = self.get_predicted_original(&model_output, &sample, timestep);

        // If this is the last step, return the prediction
        if timestep_idx + 1 >= self.timesteps.len() {
            return pred_original;
        }

        let next_timestep = self.timesteps[timestep_idx + 1];

        let alpha_prod_t = self.alphas_cumprod[timestep];
        let alpha_prod_t_next = self.alphas_cumprod[next_timestep];
        let beta_prod_t_next = 1.0 - alpha_prod_t_next;

        // Compute the next sample using the consistency function
        // x_{t-1} = sqrt(alpha_cumprod_{t-1}) * x_0 + sqrt(1 - alpha_cumprod_{t-1}) * noise_pred

        // For LCM, we directly jump to the predicted state
        // The noise component is derived from the current sample
        let noise_pred = (sample.clone() - pred_original.clone() * alpha_prod_t.sqrt())
            / (1.0 - alpha_prod_t).sqrt();

        pred_original * alpha_prod_t_next.sqrt() + noise_pred * beta_prod_t_next.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lcm_config_default() {
        let config = LcmConfig::default();
        assert_eq!(config.num_inference_steps, 4);
        assert_eq!(config.guidance_scale, 1.5);
    }
}
