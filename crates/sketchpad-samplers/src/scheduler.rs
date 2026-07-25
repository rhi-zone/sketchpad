//! Noise schedules for diffusion models
//!
//! This module provides noise schedule utilities shared across all samplers.

use burn::prelude::*;

// ============================================================================
// Prediction Type (epsilon vs v-prediction)
// ============================================================================

/// Model prediction type
///
/// Different diffusion models are trained to predict different quantities:
/// - Epsilon (noise): SD 1.x, SDXL
/// - V-prediction (velocity): SD 2.x
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PredictionType {
    /// Model predicts the noise (epsilon) added to the sample
    #[default]
    Epsilon,
    /// Model predicts the velocity v = alpha_t * epsilon - sqrt(1-alpha_t) * x0
    VPrediction,
    /// Model predicts the original sample x0
    Sample,
}

/// Convert v-prediction to epsilon prediction
///
/// v = alpha_t * epsilon - sigma_t * x
/// epsilon = (v + sigma_t * x) / alpha_t
pub fn v_to_epsilon<B: Backend>(
    v: Tensor<B, 4>,
    sample: Tensor<B, 4>,
    alpha_t: Tensor<B, 1>,
    sigma_t: Tensor<B, 1>,
) -> Tensor<B, 4> {
    (v + sample * sigma_t.unsqueeze()) / alpha_t.unsqueeze()
}

/// Convert epsilon prediction to v-prediction
///
/// v = alpha_t * epsilon - sigma_t * x
pub fn epsilon_to_v<B: Backend>(
    epsilon: Tensor<B, 4>,
    sample: Tensor<B, 4>,
    alpha_t: Tensor<B, 1>,
    sigma_t: Tensor<B, 1>,
) -> Tensor<B, 4> {
    epsilon * alpha_t.unsqueeze() - sample * sigma_t.unsqueeze()
}

/// Convert v-prediction to predicted x0
///
/// From v = alpha_t * epsilon - sigma_t * x0, we can derive:
/// x0 = alpha_t * x - sigma_t * v
pub fn v_to_sample<B: Backend>(
    v: Tensor<B, 4>,
    sample: Tensor<B, 4>,
    alpha_t: Tensor<B, 1>,
    sigma_t: Tensor<B, 1>,
) -> Tensor<B, 4> {
    sample * alpha_t.unsqueeze() - v * sigma_t.unsqueeze()
}

/// Convert epsilon prediction to predicted x0
///
/// x0 = (x - sigma_t * epsilon) / alpha_t
pub fn epsilon_to_sample<B: Backend>(
    epsilon: Tensor<B, 4>,
    sample: Tensor<B, 4>,
    alpha_t: Tensor<B, 1>,
    sigma_t: Tensor<B, 1>,
) -> Tensor<B, 4> {
    (sample - epsilon * sigma_t.unsqueeze()) / alpha_t.unsqueeze()
}

/// Convert any prediction type to epsilon
pub fn to_epsilon<B: Backend>(
    model_output: Tensor<B, 4>,
    sample: Tensor<B, 4>,
    alpha_t: Tensor<B, 1>,
    sigma_t: Tensor<B, 1>,
    prediction_type: PredictionType,
) -> Tensor<B, 4> {
    match prediction_type {
        PredictionType::Epsilon => model_output,
        PredictionType::VPrediction => v_to_epsilon(model_output, sample, alpha_t, sigma_t),
        PredictionType::Sample => {
            // epsilon = (x - alpha_t * x0) / sigma_t
            (sample - model_output * alpha_t.unsqueeze()) / sigma_t.unsqueeze()
        }
    }
}

/// Convert any prediction type to predicted x0
pub fn to_sample<B: Backend>(
    model_output: Tensor<B, 4>,
    sample: Tensor<B, 4>,
    alpha_t: Tensor<B, 1>,
    sigma_t: Tensor<B, 1>,
    prediction_type: PredictionType,
) -> Tensor<B, 4> {
    match prediction_type {
        PredictionType::Epsilon => epsilon_to_sample(model_output, sample, alpha_t, sigma_t),
        PredictionType::VPrediction => v_to_sample(model_output, sample, alpha_t, sigma_t),
        PredictionType::Sample => model_output,
    }
}

// ============================================================================
// Schedule Configuration
// ============================================================================

/// Noise schedule configuration
#[derive(Debug, Clone)]
pub struct ScheduleConfig {
    /// Number of training timesteps
    pub num_train_steps: usize,
    /// Minimum signal rate for cosine schedule
    pub min_signal_rate: f64,
    /// Maximum signal rate for cosine schedule
    pub max_signal_rate: f64,
}

impl Default for ScheduleConfig {
    fn default() -> Self {
        Self {
            num_train_steps: 1000,
            min_signal_rate: 0.02,
            max_signal_rate: 0.95,
        }
    }
}

/// Precomputed noise schedule values
pub struct NoiseSchedule<B: Backend> {
    /// Cumulative product of alphas: ᾱₜ
    pub alphas_cumprod: Tensor<B, 1>,
    /// Number of training steps
    pub num_train_steps: usize,
}

impl<B: Backend> NoiseSchedule<B> {
    /// Create a linear beta schedule (used by SD 1.x)
    pub fn linear(num_steps: usize, beta_start: f64, beta_end: f64, device: &B::Device) -> Self {
        let betas: Vec<f32> = (0..num_steps)
            .map(|i| {
                let t = i as f64 / (num_steps - 1) as f64;
                (beta_start + t * (beta_end - beta_start)) as f32
            })
            .collect();

        let alphas: Vec<f32> = betas.iter().map(|b| 1.0 - b).collect();

        // Cumulative product
        let mut alphas_cumprod = Vec::with_capacity(num_steps);
        let mut cumprod = 1.0f32;
        for alpha in alphas {
            cumprod *= alpha;
            alphas_cumprod.push(cumprod);
        }

        let data = TensorData::new(alphas_cumprod, [num_steps]);
        Self {
            alphas_cumprod: Tensor::from_data(data, device),
            num_train_steps: num_steps,
        }
    }

    /// Create a scaled linear beta schedule (used by SDXL, SD 1.x with scaling)
    ///
    /// Unlike linear which interpolates betas directly, scaled_linear interpolates
    /// sqrt(beta) and then squares the result:
    /// `betas = linspace(sqrt(beta_start), sqrt(beta_end), num_steps) ** 2`
    pub fn scaled_linear(
        num_steps: usize,
        beta_start: f64,
        beta_end: f64,
        device: &B::Device,
    ) -> Self {
        let sqrt_beta_start = beta_start.sqrt();
        let sqrt_beta_end = beta_end.sqrt();

        let betas: Vec<f32> = (0..num_steps)
            .map(|i| {
                let t = i as f64 / (num_steps - 1) as f64;
                let sqrt_beta = sqrt_beta_start + t * (sqrt_beta_end - sqrt_beta_start);
                (sqrt_beta * sqrt_beta) as f32
            })
            .collect();

        let alphas: Vec<f32> = betas.iter().map(|b| 1.0 - b).collect();

        // Cumulative product
        let mut alphas_cumprod = Vec::with_capacity(num_steps);
        let mut cumprod = 1.0f32;
        for alpha in alphas {
            cumprod *= alpha;
            alphas_cumprod.push(cumprod);
        }

        let data = TensorData::new(alphas_cumprod, [num_steps]);
        Self {
            alphas_cumprod: Tensor::from_data(data, device),
            num_train_steps: num_steps,
        }
    }

    /// Create an offset cosine schedule (used by some models like Imagen, DeepFloyd IF)
    ///
    /// Note: SDXL does NOT use cosine schedule - it uses scaled_linear.
    pub fn cosine(config: &ScheduleConfig, device: &B::Device) -> Self {
        let n = config.num_train_steps;
        let min_rate = config.min_signal_rate;
        let max_rate = config.max_signal_rate;

        let alphas_cumprod: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                let signal_rate = (1.0 - t) * max_rate + t * min_rate;
                let angle = signal_rate * std::f64::consts::FRAC_PI_2;
                (angle.cos().powi(2)) as f32
            })
            .collect();

        let data = TensorData::new(alphas_cumprod, [n]);
        Self {
            alphas_cumprod: Tensor::from_data(data, device),
            num_train_steps: n,
        }
    }

    /// Create the default SD 1.x schedule
    pub fn sd1x(device: &B::Device) -> Self {
        Self::linear(1000, 0.00085, 0.012, device)
    }

    /// Create the default SD 2.x schedule (same as SD 1.x but uses v-prediction)
    ///
    /// SD 2.x uses the same linear beta schedule as SD 1.x but the model
    /// is trained with v-prediction instead of epsilon prediction.
    pub fn sd2x(device: &B::Device) -> Self {
        Self::linear(1000, 0.00085, 0.012, device)
    }

    /// Create the default SDXL schedule
    ///
    /// SDXL uses a scaled_linear beta schedule (same parameters as SD 1.x):
    /// - beta_start: 0.00085
    /// - beta_end: 0.012
    /// - num_train_steps: 1000
    /// - prediction_type: epsilon
    ///
    /// See: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/scheduler/scheduler_config.json
    pub fn sdxl(device: &B::Device) -> Self {
        Self::scaled_linear(1000, 0.00085, 0.012, device)
    }

    /// Get alpha_cumprod at a specific timestep
    pub fn alpha_cumprod_at(&self, t: usize) -> Tensor<B, 1> {
        self.alphas_cumprod.clone().slice(t..t + 1)
    }

    /// Get sqrt(alpha_cumprod) at timestep
    pub fn sqrt_alpha_cumprod_at(&self, t: usize) -> Tensor<B, 1> {
        self.alpha_cumprod_at(t).sqrt()
    }

    /// Get sqrt(1 - alpha_cumprod) at timestep
    pub fn sqrt_one_minus_alpha_cumprod_at(&self, t: usize) -> Tensor<B, 1> {
        (self.alpha_cumprod_at(t).neg() + 1.0).sqrt()
    }

    /// Convert sigma to the closest training timestep
    ///
    /// Given sigma = sqrt((1 - alpha_cumprod) / alpha_cumprod), we can compute
    /// alpha_cumprod = 1 / (1 + sigma^2), then find the timestep with the closest
    /// alpha_cumprod value.
    ///
    /// This is needed when using non-uniform sigma schedules (e.g., Karras) where
    /// the inference sigmas don't correspond to evenly-spaced training timesteps.
    pub fn sigma_to_timestep(&self, sigma: f32) -> usize {
        // alpha_cumprod = 1 / (1 + sigma^2)
        let target_alpha = 1.0 / (1.0 + sigma * sigma);

        // Get all alphas_cumprod as f32 for comparison
        let alphas_data = self.alphas_cumprod.clone().into_data();
        let alphas: Vec<f32> = alphas_data.convert::<f32>().to_vec().unwrap();

        // Find closest timestep
        let mut best_t = 0;
        let mut best_dist = f32::INFINITY;
        for (t, &alpha) in alphas.iter().enumerate() {
            let dist = (alpha - target_alpha).abs();
            if dist < best_dist {
                best_dist = dist;
                best_t = t;
            }
        }

        best_t
    }

    /// Convert multiple sigmas to training timesteps
    pub fn sigmas_to_timesteps(&self, sigmas: &[f32]) -> Vec<usize> {
        sigmas.iter().map(|&s| self.sigma_to_timestep(s)).collect()
    }

    /// Get sigma at a specific timestep
    pub fn sigma_at(&self, t: usize) -> f32 {
        let alpha_cumprod = self.alpha_cumprod_at(t);
        let alpha_data = alpha_cumprod.into_data();
        let alpha: f32 = alpha_data.convert::<f32>().to_vec().unwrap()[0];
        ((1.0 - alpha) / alpha).sqrt()
    }

    /// Get the maximum sigma (at t = num_train_steps - 1)
    pub fn sigma_max(&self) -> f32 {
        self.sigma_at(self.num_train_steps - 1)
    }

    /// Get the minimum sigma (at t = 0)
    pub fn sigma_min(&self) -> f32 {
        self.sigma_at(0)
    }
}

/// Generate timestep sequence for inference
pub fn inference_timesteps(num_inference_steps: usize, num_train_steps: usize) -> Vec<usize> {
    let step_ratio = num_train_steps / num_inference_steps;
    (0..num_inference_steps)
        .rev()
        .map(|i| i * step_ratio)
        .collect()
}

// ============================================================================
// Sigma Utilities (shared across samplers)
// ============================================================================

/// Compute sigmas from timesteps using the noise schedule
///
/// Converts alpha_cumprod values to sigma = sqrt((1 - alpha) / alpha)
pub fn sigmas_from_timesteps<B: Backend>(
    schedule: &NoiseSchedule<B>,
    timesteps: &[usize],
) -> Vec<f32> {
    timesteps
        .iter()
        .map(|&t| {
            let alpha_cumprod = schedule.alpha_cumprod_at(t);
            let alpha_data = alpha_cumprod.into_data();
            let alpha: f32 = alpha_data.convert::<f32>().to_vec().unwrap()[0];
            ((1.0 - alpha) / alpha).sqrt()
        })
        .collect()
}

// ============================================================================
// Sigma Schedule Types
// ============================================================================

/// Sigma schedule type for sampler timestep spacing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SigmaSchedule {
    /// Simple/Normal - uniform spacing in timestep space (default for most samplers)
    #[default]
    Normal,
    /// Karras - uses Karras et al. schedule with rho=7.0 for improved quality
    Karras,
    /// Exponential - exponential spacing between sigma_max and sigma_min
    Exponential,
    /// SGM Uniform - uniform spacing in sigma space (Score-based Generative Models)
    SgmUniform,
    /// Beta - beta distribution spacing (more steps at high noise)
    Beta,
    /// Linear Quadratic - blend of linear and quadratic spacing
    LinearQuadratic,
    /// Custom sigmas provided by user
    Custom,
}

/// Apply Karras noise schedule transformation
///
/// Transforms sigmas using the Karras et al. schedule for improved sampling.
/// Uses rho=7.0 as recommended in the paper.
pub fn apply_karras_schedule(sigmas: &[f32], rho: f32) -> Vec<f32> {
    if sigmas.is_empty() {
        return Vec::new();
    }

    let sigma_min = *sigmas.last().unwrap_or(&0.0);
    let sigma_max = *sigmas.first().unwrap_or(&1.0);
    let n = sigmas.len();

    (0..n)
        .map(|i| {
            let t = i as f32 / (n - 1).max(1) as f32;
            let sigma_min_inv_rho = sigma_min.powf(1.0 / rho);
            let sigma_max_inv_rho = sigma_max.powf(1.0 / rho);
            (sigma_max_inv_rho + t * (sigma_min_inv_rho - sigma_max_inv_rho)).powf(rho)
        })
        .collect()
}

/// Apply exponential schedule transformation
///
/// Creates exponentially spaced sigmas between sigma_max and sigma_min.
pub fn apply_exponential_schedule(sigmas: &[f32]) -> Vec<f32> {
    if sigmas.is_empty() {
        return Vec::new();
    }

    let sigma_min = *sigmas.last().unwrap_or(&0.0);
    let sigma_max = *sigmas.first().unwrap_or(&1.0);
    let n = sigmas.len();

    // Avoid log(0)
    let log_min = (sigma_min.max(1e-10)).ln();
    let log_max = sigma_max.ln();

    (0..n)
        .map(|i| {
            let t = i as f32 / (n - 1).max(1) as f32;
            (log_max + t * (log_min - log_max)).exp()
        })
        .collect()
}

/// Apply SGM uniform schedule
///
/// Uniform spacing in sigma space (used by Score-based Generative Models).
pub fn apply_sgm_uniform_schedule(sigmas: &[f32]) -> Vec<f32> {
    if sigmas.is_empty() {
        return Vec::new();
    }

    let sigma_min = *sigmas.last().unwrap_or(&0.0);
    let sigma_max = *sigmas.first().unwrap_or(&1.0);
    let n = sigmas.len();

    (0..n)
        .map(|i| {
            let t = i as f32 / (n - 1).max(1) as f32;
            sigma_max + t * (sigma_min - sigma_max)
        })
        .collect()
}

/// Apply beta distribution schedule
///
/// Uses beta distribution CDF for spacing (concentrates steps at high noise).
pub fn apply_beta_schedule(sigmas: &[f32], alpha: f32, beta_param: f32) -> Vec<f32> {
    if sigmas.is_empty() {
        return Vec::new();
    }

    let sigma_min = *sigmas.last().unwrap_or(&0.0);
    let sigma_max = *sigmas.first().unwrap_or(&1.0);
    let n = sigmas.len();

    // Simplified beta-like spacing using power function
    // Full beta distribution would require the incomplete beta function
    (0..n)
        .map(|i| {
            let t = i as f32 / (n - 1).max(1) as f32;
            // Use power function as approximation: t^alpha / (t^alpha + (1-t)^beta)
            let t_powered = t.powf(alpha);
            let one_minus_t_powered = (1.0 - t).powf(beta_param);
            let beta_t = if t_powered + one_minus_t_powered > 0.0 {
                t_powered / (t_powered + one_minus_t_powered)
            } else {
                t
            };
            sigma_max + beta_t * (sigma_min - sigma_max)
        })
        .collect()
}

/// Apply linear-quadratic blend schedule
///
/// Blends linear and quadratic spacing for a balance of quality and speed.
pub fn apply_linear_quadratic_schedule(sigmas: &[f32], blend: f32) -> Vec<f32> {
    if sigmas.is_empty() {
        return Vec::new();
    }

    let sigma_min = *sigmas.last().unwrap_or(&0.0);
    let sigma_max = *sigmas.first().unwrap_or(&1.0);
    let n = sigmas.len();

    (0..n)
        .map(|i| {
            let t = i as f32 / (n - 1).max(1) as f32;
            let linear = t;
            let quadratic = t * t;
            let blended = linear * (1.0 - blend) + quadratic * blend;
            sigma_max + blended * (sigma_min - sigma_max)
        })
        .collect()
}

/// Compute sigmas with specified schedule type
///
/// This is the main entry point for computing sigmas in samplers.
/// Appends sigma=0.0 at the end for the final denoising step.
pub fn compute_sigmas<B: Backend>(
    schedule: &NoiseSchedule<B>,
    timesteps: &[usize],
    sigma_schedule: SigmaSchedule,
) -> Vec<f32> {
    // Get base sigmas at the sampler timesteps
    let base_sigmas = sigmas_from_timesteps(schedule, timesteps);

    let mut sigmas = match sigma_schedule {
        SigmaSchedule::Normal => base_sigmas,
        SigmaSchedule::Karras => apply_karras_schedule(&base_sigmas, 7.0),
        SigmaSchedule::Exponential => apply_exponential_schedule(&base_sigmas),
        SigmaSchedule::SgmUniform => apply_sgm_uniform_schedule(&base_sigmas),
        SigmaSchedule::Beta => apply_beta_schedule(&base_sigmas, 0.6, 0.6),
        SigmaSchedule::LinearQuadratic => apply_linear_quadratic_schedule(&base_sigmas, 0.5),
        SigmaSchedule::Custom => base_sigmas,
    };

    sigmas.push(0.0);
    sigmas
}

/// Compute sigmas with optional Karras schedule (legacy API)
///
/// For backwards compatibility. Prefer `compute_sigmas` with `SigmaSchedule`.
pub fn compute_sigmas_karras<B: Backend>(
    schedule: &NoiseSchedule<B>,
    timesteps: &[usize],
    use_karras: bool,
) -> Vec<f32> {
    compute_sigmas(
        schedule,
        timesteps,
        if use_karras {
            SigmaSchedule::Karras
        } else {
            SigmaSchedule::Normal
        },
    )
}

/// Create custom sigmas from user-provided values
///
/// Allows users to specify exact sigma values for sampling.
/// If `normalize` is true, scales sigmas so the sum equals the original sum.
pub fn custom_sigmas(sigmas: Vec<f32>, normalize: bool) -> Vec<f32> {
    if sigmas.is_empty() {
        return vec![0.0];
    }

    let mut result = if normalize {
        let sum: f32 = sigmas.iter().sum();
        if sum > 0.0 {
            sigmas
                .iter()
                .map(|s| s / sum * sigmas.len() as f32)
                .collect()
        } else {
            sigmas
        }
    } else {
        sigmas
    };

    // Ensure we end with 0.0
    if result.last() != Some(&0.0) {
        result.push(0.0);
    }

    result
}

/// Compute ancestral sampling step parameters
///
/// For stochastic samplers (ancestral, SDE), computes:
/// - sigma_down: the deterministic step target
/// - sigma_up: the noise injection level
///
/// The eta parameter controls stochasticity (0 = ODE, 1 = full SDE)
pub fn get_ancestral_step(sigma: f32, sigma_next: f32, eta: f32) -> (f32, f32) {
    if sigma_next == 0.0 {
        return (0.0, 0.0);
    }

    let sigma_up = (sigma_next.powi(2) * (sigma.powi(2) - sigma_next.powi(2)) / sigma.powi(2))
        .sqrt()
        .min(sigma_next)
        * eta;
    let sigma_down = (sigma_next.powi(2) - sigma_up.powi(2)).sqrt();

    (sigma_down, sigma_up)
}

/// Generate timesteps for a sampler
///
/// Creates evenly spaced timesteps from high noise to low noise.
pub fn sampler_timesteps(num_inference_steps: usize, num_train_steps: usize) -> Vec<usize> {
    let step_ratio = num_train_steps / num_inference_steps;
    (0..num_inference_steps)
        .rev()
        .map(|i| (i * step_ratio).min(num_train_steps - 1))
        .collect()
}

/// Initialize a random latent tensor scaled by sigma
///
/// Creates a random normal tensor and scales it by the given sigma value.
/// This is typically used to initialize the latent at the start of sampling,
/// where sigma_max is the initial noise level.
pub fn init_noise_latent<B: Backend>(
    batch_size: usize,
    channels: usize,
    height: usize,
    width: usize,
    sigma: f32,
    device: &B::Device,
) -> Tensor<B, 4> {
    let noise: Tensor<B, 4> = Tensor::random(
        [batch_size, channels, height, width],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );
    noise * sigma
}

/// Adams-Bashforth coefficients for linear multi-step methods
///
/// These coefficients are used by iPNDM and SA-Solver samplers
/// for multi-step ODE integration.
pub fn adams_bashforth_coefficients(order: usize) -> Vec<f32> {
    match order {
        1 => vec![1.0],
        2 => vec![1.5, -0.5],
        3 => vec![23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0],
        4 => vec![55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -9.0 / 24.0],
        _ => vec![1.0], // Fallback to first order
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_timesteps() {
        let steps = inference_timesteps(50, 1000);
        assert_eq!(steps.len(), 50);
        assert_eq!(steps[0], 980); // First step (highest noise)
        assert_eq!(steps[49], 0); // Last step (lowest noise)
    }
}
