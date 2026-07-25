//! Diffusion Samplers for Image Generation
//!
//! This crate provides a comprehensive collection of diffusion samplers
//! for denoising latents in Stable Diffusion and related models.
//!
//! # Sampler Categories
//!
//! ## Basic Samplers
//! - [`DdimSampler`] - Deterministic, fast (10-50 steps)
//! - [`DdpmSampler`] - Original DDPM, stochastic
//! - [`EulerSampler`] - Euler method, balanced speed/quality
//! - [`EulerAncestralSampler`] - Euler with ancestral sampling
//!
//! ## DPM++ Family (Recommended)
//! - [`DpmPlusPlusSampler`] - DPM++ 2M, excellent quality
//! - [`DpmPlusPlusSdeSampler`] - DPM++ 2M SDE, more detail
//! - [`Dpm2sAncestralSampler`] - DPM++ 2S ancestral
//! - [`Dpm3mSdeSampler`] - DPM++ 3M SDE, highest quality
//!
//! ## Advanced Samplers
//! - [`HeunSampler`] - Heun's method, high quality
//! - [`LmsSampler`] - Linear multistep
//! - [`UniPcSampler`] - UniPC predictor-corrector
//! - [`DeisSampler`] - DEIS exponential integrator
//! - [`SaSolver`] - SA-Solver with stochastic churn
//!
//! ## Fast Samplers
//! - [`LcmSampler`] - Latent Consistency Model (4-8 steps)
//! - [`DpmFastSampler`] - Fast DPM sampling
//! - [`DpmAdaptiveSampler`] - Adaptive step sizing
//!
//! ## CFG++ Variants
//! - [`EulerCfgPlusPlusSampler`] - Improved guidance
//! - [`EulerAncestralCfgPlusPlusSampler`]
//! - [`Dpm2mCfgPlusPlusSampler`]
//!
//! # Noise Schedules
//!
//! Use [`ScheduleConfig`] to configure noise schedules:
//! - Linear (SD 1.x default)
//! - Scaled linear (SDXL)
//! - Cosine
//! - Karras sigmas
//!
//! # Example
//!
//! ```ignore
//! use sketchpad_samplers::{DpmPlusPlusSampler, DpmConfig, ScheduleConfig};
//!
//! let schedule = ScheduleConfig::scaled_linear(1000);
//! let sampler = DpmPlusPlusSampler::new(DpmConfig {
//!     schedule,
//!     steps: 30,
//!     ..Default::default()
//! });
//!
//! // Sampling loop
//! for t in sampler.timesteps() {
//!     let noise_pred = unet.forward(latents, t, cond);
//!     latents = sampler.step(latents, noise_pred, t);
//! }
//! ```

pub mod ddim;
pub mod ddpm;
pub mod deis;
pub mod dpm;
pub mod dpm2;
pub mod dpm3m;
pub mod dpm_2m_variants;
pub mod dpm_2s;
pub mod dpm_fast;
pub mod euler;
pub mod euler_cfg;
pub mod guidance;
pub mod heun;
pub mod ipndm;
pub mod lcm;
pub mod lms;
pub mod res_multistep;
pub mod sa_solver;
pub mod scheduler;
pub mod unipc;

pub use ddim::{DdimConfig, DdimSampler, apply_guidance};
pub use ddpm::{DdpmConfig, DdpmSampler, VarianceType};
pub use deis::{DeisConfig, DeisSampler, DeisSolverType};
pub use dpm::{DpmConfig, DpmPlusPlusSampler, DpmPlusPlusSdeSampler};
pub use dpm_2m_variants::{
    Dpm2mCfgPlusPlusConfig, Dpm2mCfgPlusPlusSampler, Dpm2mSdeHeunConfig, Dpm2mSdeHeunSampler,
};
pub use dpm_2s::{Dpm2sAncestralCfgPlusPlusSampler, Dpm2sAncestralConfig, Dpm2sAncestralSampler};
pub use dpm_fast::{DpmAdaptiveConfig, DpmAdaptiveSampler, DpmFastConfig, DpmFastSampler};
pub use dpm2::{Dpm2AncestralSampler, Dpm2Config, Dpm2Sampler};
pub use dpm3m::{Dpm3mSdeConfig, Dpm3mSdeSampler};
pub use euler::{EulerAncestralSampler, EulerConfig, EulerSampler};
pub use euler_cfg::{
    EulerAncestralCfgPlusPlusSampler, EulerCfgPlusPlusConfig, EulerCfgPlusPlusSampler,
};
pub use guidance::{CfgPlusPlusConfig, apply_cfg_plus_plus, compute_tensor_std};
pub use heun::{HeunConfig, HeunPP2Sampler, HeunSampler};
pub use ipndm::{IpndmConfig, IpndmSampler, IpndmVSampler};
pub use lcm::{LcmConfig, LcmSampler};
pub use lms::{LmsConfig, LmsSampler};
pub use res_multistep::{ResMultistepConfig, ResMultistepSampler, ResMultistepSdeSampler};
pub use sa_solver::{SaSolver, SaSolverConfig, TauType};
pub use scheduler::{
    NoiseSchedule, PredictionType, ScheduleConfig, SigmaSchedule, apply_beta_schedule,
    apply_exponential_schedule, apply_karras_schedule, apply_linear_quadratic_schedule,
    apply_sgm_uniform_schedule, compute_sigmas, compute_sigmas_karras, custom_sigmas,
    epsilon_to_sample, epsilon_to_v, get_ancestral_step, sampler_timesteps, sigmas_from_timesteps,
    to_epsilon, to_sample, v_to_epsilon, v_to_sample,
};
pub use unipc::{UniPcConfig, UniPcSampler};
