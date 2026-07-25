//! DDPM (Denoising Diffusion Probabilistic Models) sampler
//!
//! The original diffusion model sampler that uses stochastic sampling
//! with learned variance or fixed variance schedules.

use burn::prelude::*;

use crate::scheduler::NoiseSchedule;

/// Variance type for DDPM
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VarianceType {
    /// Fixed small variance: β_t
    #[default]
    FixedSmall,
    /// Fixed large variance: β̃_t
    FixedLarge,
    /// Learned variance (model predicts variance)
    Learned,
    /// Learned range (interpolation between small and large)
    LearnedRange,
}

/// Configuration for DDPM sampler
#[derive(Debug, Clone)]
pub struct DdpmConfig {
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// Total training timesteps
    pub num_train_timesteps: usize,
    /// Variance type
    pub variance_type: VarianceType,
    /// Clip sample to [-clip_range, clip_range]
    pub clip_sample: bool,
    /// Clip sample range
    pub clip_sample_range: f32,
    /// Prediction type (epsilon, v_prediction, sample)
    pub prediction_type: String,
    /// Beta start (for computing betas)
    pub beta_start: f64,
    /// Beta end (for computing betas)
    pub beta_end: f64,
}

impl Default for DdpmConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 1000,
            num_train_timesteps: 1000,
            variance_type: VarianceType::FixedSmall,
            clip_sample: true,
            clip_sample_range: 1.0,
            prediction_type: "epsilon".to_string(),
            beta_start: 0.00085,
            beta_end: 0.012,
        }
    }
}

/// DDPM sampler
///
/// Implements the original DDPM sampling with optional variance learning.
pub struct DdpmSampler<B: Backend> {
    /// Sampler configuration
    config: DdpmConfig,
    /// Timestep indices for sampling
    timesteps: Vec<usize>,
    /// Alpha values (1 - beta)
    alphas: Vec<f32>,
    /// Cumulative product of alphas
    alphas_cumprod: Vec<f32>,
    /// Beta values at each timestep
    betas: Vec<f32>,
    /// Phantom data for backend type
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> DdpmSampler<B> {
    /// Create a new DDPM sampler
    pub fn new(config: DdpmConfig, schedule: &NoiseSchedule<B>) -> Self {
        let timesteps = Self::compute_timesteps(&config);

        // Compute betas from config
        let num_steps = config.num_train_timesteps;
        let betas: Vec<f32> = (0..num_steps)
            .map(|i| {
                let t = i as f64 / (num_steps - 1) as f64;
                (config.beta_start + t * (config.beta_end - config.beta_start)) as f32
            })
            .collect();

        let alphas: Vec<f32> = betas.iter().map(|b| 1.0 - b).collect();

        // Compute alphas_cumprod from schedule
        let mut alphas_cumprod = Vec::with_capacity(num_steps);
        for t in 0..num_steps {
            let alpha_cumprod = schedule.alpha_cumprod_at(t);
            let alpha_data = alpha_cumprod.into_data();
            let alpha: f32 = alpha_data.convert::<f32>().to_vec().unwrap()[0];
            alphas_cumprod.push(alpha);
        }

        Self {
            config,
            timesteps,
            alphas,
            alphas_cumprod,
            betas,
            _marker: std::marker::PhantomData,
        }
    }

    /// Computes the timestep indices from config
    fn compute_timesteps(config: &DdpmConfig) -> Vec<usize> {
        let step_ratio = config.num_train_timesteps / config.num_inference_steps;
        (0..config.num_inference_steps)
            .rev()
            .map(|i| i * step_ratio)
            .collect()
    }

    /// Get the timesteps
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Computes variance for a timestep based on variance type
    fn get_variance(&self, t: usize) -> f32 {
        let alpha_prod_t = self.alphas_cumprod[t];
        let alpha_prod_t_prev = if t > 0 {
            self.alphas_cumprod[t - 1]
        } else {
            1.0
        };

        // β̃_t = (1 - α_{t-1}) / (1 - α_t) * β_t
        let variance = (1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t) * self.betas[t];

        match self.config.variance_type {
            VarianceType::FixedSmall => variance.max(1e-20),
            VarianceType::FixedLarge => self.betas[t],
            VarianceType::Learned | VarianceType::LearnedRange => variance.max(1e-20),
        }
    }

    /// Perform one DDPM step
    pub fn step(
        &self,
        model_output: Tensor<B, 4>,
        timestep: usize,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let t = timestep;

        let alpha_prod_t = self.alphas_cumprod[t];
        let alpha_prod_t_prev = if t > 0 {
            self.alphas_cumprod[t - 1]
        } else {
            1.0
        };
        let beta_prod_t = 1.0 - alpha_prod_t;
        let beta_prod_t_prev = 1.0 - alpha_prod_t_prev;

        // Predict x_0
        let pred_original = if self.config.prediction_type == "epsilon" {
            (sample.clone() - model_output.clone() * beta_prod_t.sqrt()) / alpha_prod_t.sqrt()
        } else if self.config.prediction_type == "sample" {
            model_output.clone()
        } else {
            // v_prediction
            sample.clone() * alpha_prod_t.sqrt() - model_output.clone() * beta_prod_t.sqrt()
        };

        let pred_original = if self.config.clip_sample {
            pred_original.clamp(
                -self.config.clip_sample_range,
                self.config.clip_sample_range,
            )
        } else {
            pred_original
        };

        // Coefficients for x_{t-1}
        let coeff_orig = (alpha_prod_t_prev.sqrt() * self.betas[t]) / beta_prod_t;
        let coeff_curr = (self.alphas[t].sqrt() * beta_prod_t_prev) / beta_prod_t;

        let pred_prev = pred_original * coeff_orig + sample * coeff_curr;

        // Add noise for t > 0
        if t > 0 {
            let variance = self.get_variance(t);
            let device = pred_prev.device();
            let shape = pred_prev.dims();
            let noise: Tensor<B, 4> =
                Tensor::random(shape, burn::tensor::Distribution::Normal(0.0, 1.0), &device);
            pred_prev + noise * variance.sqrt()
        } else {
            pred_prev
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ddpm_config_default() {
        let config = DdpmConfig::default();
        assert_eq!(config.num_inference_steps, 1000);
        assert_eq!(config.variance_type, VarianceType::FixedSmall);
    }
}
