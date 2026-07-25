//! iPNDM (improved Pseudo Numerical Diffusion Model) samplers
//!
//! Fourth-order pseudo-linear multi-step methods for diffusion models.
//! Uses past function evaluations for higher-order accuracy.

use burn::prelude::*;
use std::collections::VecDeque;

use crate::scheduler::{NoiseSchedule, adams_bashforth_coefficients, sampler_timesteps};

/// Configuration for iPNDM sampler
#[derive(Debug, Clone)]
pub struct IpndmConfig {
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// Order of the method (1-4)
    pub order: usize,
}

impl Default for IpndmConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 25,
            order: 4,
        }
    }
}

/// iPNDM sampler
///
/// Improved Pseudo Numerical Diffusion Model uses a fourth-order
/// linear multi-step method adapted for the diffusion ODE.
pub struct IpndmSampler<B: Backend> {
    /// Sampler configuration
    config: IpndmConfig,
    /// Timestep indices for sampling
    timesteps: Vec<usize>,
    /// Cumulative product of alphas
    alphas_cumprod: Vec<f32>,
    /// History of epsilon predictions
    ets: VecDeque<Tensor<B, 4>>,
}

/// Extracts all alpha_cumprod values from a noise schedule
fn extract_alphas_cumprod<B: Backend>(schedule: &NoiseSchedule<B>) -> Vec<f32> {
    let mut alphas_cumprod = Vec::with_capacity(schedule.num_train_steps);
    for t in 0..schedule.num_train_steps {
        let alpha_cumprod = schedule.alpha_cumprod_at(t);
        let alpha_data = alpha_cumprod.into_data();
        let alpha: f32 = alpha_data.convert::<f32>().to_vec().unwrap()[0];
        alphas_cumprod.push(alpha);
    }
    alphas_cumprod
}

impl<B: Backend> IpndmSampler<B> {
    /// Create a new iPNDM sampler
    pub fn new(config: IpndmConfig, schedule: &NoiseSchedule<B>) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let alphas_cumprod = extract_alphas_cumprod(schedule);

        Self {
            config,
            timesteps,
            alphas_cumprod,
            ets: VecDeque::new(),
        }
    }

    /// Get the timesteps
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Reset the history
    pub fn reset(&mut self) {
        self.ets.clear();
    }

    /// Get coefficients for the linear multi-step method
    fn get_linear_multistep_coeff(order: usize) -> Vec<f32> {
        adams_bashforth_coefficients(order)
    }

    /// Perform one iPNDM step
    pub fn step(
        &mut self,
        model_output: Tensor<B, 4>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let t = self.timesteps[timestep_idx];
        let t_prev = if timestep_idx + 1 < self.timesteps.len() {
            self.timesteps[timestep_idx + 1]
        } else {
            0
        };

        let alpha_prod_t = self.alphas_cumprod[t];
        let alpha_prod_t_prev = self.alphas_cumprod[t_prev];
        let beta_prod_t = 1.0 - alpha_prod_t;
        let beta_prod_t_prev = 1.0 - alpha_prod_t_prev;

        // Store current prediction
        self.ets.push_front(model_output.clone());
        if self.ets.len() > self.config.order {
            self.ets.pop_back();
        }

        // Determine effective order
        let order = self.ets.len().min(self.config.order);
        let coeffs = Self::get_linear_multistep_coeff(order);

        // Compute weighted combination of past predictions
        let mut e_t = Tensor::zeros(model_output.dims(), &model_output.device());
        for (i, coeff) in coeffs.iter().enumerate().take(order) {
            if let Some(et) = self.ets.get(i) {
                e_t = e_t + et.clone() * *coeff;
            }
        }

        // iPNDM update formula
        // x_{t-1} = sqrt(α_{t-1}) * (x_t - sqrt(β_t) * ε_θ) / sqrt(α_t)
        //         + sqrt(β_{t-1}) * ε_θ

        let pred_original =
            (sample.clone() - e_t.clone() * beta_prod_t.sqrt()) / alpha_prod_t.sqrt();

        pred_original.clone() * alpha_prod_t_prev.sqrt() + e_t * beta_prod_t_prev.sqrt()
    }
}

/// iPNDM-v sampler (velocity prediction variant)
///
/// Variant of iPNDM that uses v-prediction instead of epsilon prediction.
pub struct IpndmVSampler<B: Backend> {
    config: IpndmConfig,
    timesteps: Vec<usize>,
    alphas_cumprod: Vec<f32>,
    /// History of v predictions
    vs: VecDeque<Tensor<B, 4>>,
}

impl<B: Backend> IpndmVSampler<B> {
    /// Create a new iPNDM-v sampler
    pub fn new(config: IpndmConfig, schedule: &NoiseSchedule<B>) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let alphas_cumprod = extract_alphas_cumprod(schedule);

        Self {
            config,
            timesteps,
            alphas_cumprod,
            vs: VecDeque::new(),
        }
    }

    /// Get the timesteps
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Reset the history
    pub fn reset(&mut self) {
        self.vs.clear();
    }

    /// Get coefficients for the linear multi-step method
    fn get_linear_multistep_coeff(order: usize) -> Vec<f32> {
        IpndmSampler::<B>::get_linear_multistep_coeff(order)
    }

    /// Perform one iPNDM-v step
    pub fn step(
        &mut self,
        model_output: Tensor<B, 4>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let t = self.timesteps[timestep_idx];
        let t_prev = if timestep_idx + 1 < self.timesteps.len() {
            self.timesteps[timestep_idx + 1]
        } else {
            0
        };

        let alpha_prod_t = self.alphas_cumprod[t];
        let alpha_prod_t_prev = self.alphas_cumprod[t_prev];
        let beta_prod_t = 1.0 - alpha_prod_t;
        let beta_prod_t_prev = 1.0 - alpha_prod_t_prev;

        // Store current v prediction
        self.vs.push_front(model_output.clone());
        if self.vs.len() > self.config.order {
            self.vs.pop_back();
        }

        // Determine effective order
        let order = self.vs.len().min(self.config.order);
        let coeffs = Self::get_linear_multistep_coeff(order);

        // Compute weighted combination of past v predictions
        let mut v_t = Tensor::zeros(model_output.dims(), &model_output.device());
        for (i, coeff) in coeffs.iter().enumerate().take(order) {
            if let Some(v) = self.vs.get(i) {
                v_t = v_t + v.clone() * *coeff;
            }
        }

        // Convert v-prediction to x0 and epsilon
        // v = sqrt(α) * ε - sqrt(β) * x
        // x0 = sqrt(α) * x - sqrt(β) * v
        // ε = sqrt(α) * v + sqrt(β) * x
        let pred_original = sample.clone() * alpha_prod_t.sqrt() - v_t.clone() * beta_prod_t.sqrt();
        let pred_epsilon = v_t.clone() * alpha_prod_t.sqrt() + sample.clone() * beta_prod_t.sqrt();

        // Standard DDPM update with predicted values
        pred_original * alpha_prod_t_prev.sqrt() + pred_epsilon * beta_prod_t_prev.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ipndm_config_default() {
        let config = IpndmConfig::default();
        assert_eq!(config.order, 4);
        assert_eq!(config.num_inference_steps, 25);
    }

    #[test]
    fn test_linear_multistep_coeffs() {
        fn get_linear_multistep_coeff(order: usize) -> Vec<f32> {
            match order {
                1 => vec![1.0],
                2 => vec![1.5, -0.5],
                3 => vec![23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0],
                4 => vec![55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -9.0 / 24.0],
                _ => vec![1.0],
            }
        }

        let coeffs_1 = get_linear_multistep_coeff(1);
        assert_eq!(coeffs_1, vec![1.0]);

        let coeffs_2 = get_linear_multistep_coeff(2);
        assert_eq!(coeffs_2, vec![1.5, -0.5]);
    }
}
