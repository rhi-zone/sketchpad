//! Diffusion pipeline trait and implementations

// Many sampling loops use step_idx to index timestep_tensors AND pass to sampler.step()
#![allow(clippy::needless_range_loop)]

mod sd1x;
mod sdxl;

pub use sd1x::{
    Img2ImgConfig, InpaintConfig, Sd1xConditioning, StableDiffusion1x, StableDiffusion1xImg2Img,
    StableDiffusion1xInpaint,
};
pub use sdxl::{
    BaseRefinerConfig, RefinerConfig, SdxlConditioning, SdxlImg2ImgConfig, SdxlInpaintConfig,
    SdxlSampleConfig, StableDiffusionXL, StableDiffusionXLImg2Img, StableDiffusionXLInpaint,
    StableDiffusionXLRefiner, StableDiffusionXLWithRefiner,
};

use burn::prelude::*;

/// Sampler algorithm selection
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SamplerType {
    /// DDIM - Deterministic, fast
    Ddim,
    /// DDPM - Original stochastic sampler
    Ddpm,
    /// Euler - Balanced speed/quality
    Euler,
    /// Euler Ancestral - Euler with noise injection
    EulerA,
    /// DPM++ 2M - Excellent quality (default, recommended)
    #[default]
    DpmPlusPlus,
    /// DPM++ 2M SDE - More detail, stochastic
    DpmPlusPlusSde,
    /// DPM++ 2S Ancestral
    Dpm2sA,
    /// DPM++ 3M SDE - Highest quality
    Dpm3mSde,
    /// Heun - High quality, 2x NFE
    Heun,
    /// LMS - Linear multi-step
    Lms,
    /// UniPC - Predictor-corrector
    UniPc,
    /// DEIS - Exponential integrator
    Deis,
    /// LCM - Fast (4-8 steps)
    Lcm,
}

/// Noise schedule type (sigma spacing)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ScheduleType {
    /// Simple/Normal - uniform spacing in timestep space
    Normal,
    /// Karras sigma schedule (default, better quality)
    #[default]
    Karras,
    /// Exponential spacing between sigma_max and sigma_min
    Exponential,
    /// SGM Uniform - uniform spacing in sigma space
    SgmUniform,
    /// Beta distribution spacing (more steps at high noise)
    Beta,
    /// Linear-Quadratic blend
    LinearQuadratic,
}

impl ScheduleType {
    /// Convert to sampler's SigmaSchedule
    pub fn to_sigma_schedule(self) -> sketchpad_samplers::SigmaSchedule {
        match self {
            ScheduleType::Normal => sketchpad_samplers::SigmaSchedule::Normal,
            ScheduleType::Karras => sketchpad_samplers::SigmaSchedule::Karras,
            ScheduleType::Exponential => sketchpad_samplers::SigmaSchedule::Exponential,
            ScheduleType::SgmUniform => sketchpad_samplers::SigmaSchedule::SgmUniform,
            ScheduleType::Beta => sketchpad_samplers::SigmaSchedule::Beta,
            ScheduleType::LinearQuadratic => sketchpad_samplers::SigmaSchedule::LinearQuadratic,
        }
    }
}

/// Compute sinusoidal size embedding for SDXL
///
/// Used for encoding image dimensions (width, height, crop coordinates)
/// into the model's conditioning. Returns a 256-dim embedding.
pub(crate) fn compute_size_embedding<B: Backend>(value: usize, device: &B::Device) -> Tensor<B, 1> {
    let half_dim = 128;
    let value = value as f32;
    let mut emb = vec![0.0f32; 256];
    for i in 0..half_dim {
        let freq = (-((i as f32) / half_dim as f32) * (10000.0f32).ln()).exp();
        emb[i] = (value * freq).sin();
        emb[i + half_dim] = (value * freq).cos();
    }
    Tensor::from_data(TensorData::new(emb, [256]), device)
}

/// Debug flags for pipeline and sampler diagnostics
#[derive(Debug, Clone, Copy, Default)]
pub struct DebugConfig {
    /// Print sampler/pipeline debug info (sigmas, latent stats, etc.)
    pub sampler: bool,
    /// Panic on NaN/Inf values in tensors
    pub nan: bool,
}

/// Configuration for sampling
#[derive(Debug, Clone)]
pub struct SampleConfig {
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub guidance_scale: f64,
    pub seed: Option<u64>,
    /// Sampler algorithm (default: DPM++ 2M)
    pub sampler: SamplerType,
    /// Noise schedule (default: Karras)
    pub schedule: ScheduleType,
    /// Denoising strength (1.0 = full generation, <1.0 = partial)
    pub denoise: f64,
    /// Debug output flags
    pub debug: DebugConfig,
}

impl Default for SampleConfig {
    fn default() -> Self {
        Self {
            width: 512,
            height: 512,
            steps: 50,
            guidance_scale: 7.5,
            seed: None,
            sampler: SamplerType::default(),
            schedule: ScheduleType::default(),
            denoise: 1.0,
            debug: DebugConfig::default(),
        }
    }
}

// ============================================================================
// Step Callback Types
// ============================================================================

/// What to output at each sampling step
#[derive(Debug, Clone, Copy, Default)]
pub enum StepOutput {
    /// No output, minimal overhead
    #[default]
    None,
    /// Raw latent tensor (~0.1ms GPU->CPU copy)
    Latent,
    /// Latent visualized as RGB without VAE decode (~0.1ms)
    LatentPreview,
    /// Full VAE decode to image (~50-100ms, expensive!)
    Decoded,
}

/// Information passed to step callback
pub struct StepInfo<B: Backend> {
    /// Current step (0-indexed)
    pub step: usize,
    /// Total number of steps
    pub total_steps: usize,
    /// Current timestep value
    pub timestep: usize,
    /// Output based on StepOutput setting
    pub output: Option<Tensor<B, 4>>,
}

/// Latent format for preview coefficients
///
/// Coefficients from ComfyUI's latent_formats.py:
/// https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/latent_formats.py
/// Licensed under GPL-3.0
#[derive(Debug, Clone, Copy, Default)]
pub enum LatentFormat {
    // 4-channel models
    /// Stable Diffusion 1.x
    #[default]
    SD15,
    /// Stable Diffusion XL
    SDXL,
    /// SDXL Playground 2.5
    SDXLPlayground25,
    /// SD 4x Upscaler
    SDX4,
    /// Stable Cascade stage B
    SCB,

    // 12-channel models
    /// Mochi video
    Mochi,

    // 16-channel models
    /// Stable Diffusion 3
    SD3,
    /// Flux.1
    Flux,
    /// Stable Cascade Prior
    SCPrior,
    /// HunyuanVideo
    HunyuanVideo,
    /// Wan 2.1
    Wan21,

    // 32-channel models
    /// HunyuanVideo 1.5
    HunyuanVideo15,
    /// Flux.2 (Kontext)
    Flux2,

    // 48-channel models
    /// Wan 2.2
    Wan22,

    // 128-channel models
    /// LTXV video
    LTXV,
}

impl LatentFormat {
    /// Number of latent channels for this format
    pub fn channels(&self) -> usize {
        match self {
            Self::SD15 | Self::SDXL | Self::SDXLPlayground25 | Self::SDX4 | Self::SCB => 4,
            Self::Mochi => 12,
            Self::SD3 | Self::Flux | Self::SCPrior | Self::HunyuanVideo | Self::Wan21 => 16,
            Self::HunyuanVideo15 | Self::Flux2 => 32,
            Self::Wan22 => 48,
            Self::LTXV => 128,
        }
    }

    /// Get RGB projection coefficients (channels x 3) and bias
    #[rustfmt::skip]
    fn rgb_coefs(&self) -> (Vec<f32>, [f32; 3]) {
        match self {
            // 4-channel models
            Self::SD15 => (vec![
                 0.3512,  0.2297,  0.3227,
                 0.3250,  0.4974,  0.2350,
                -0.2829,  0.1762,  0.2721,
                -0.2120, -0.2616, -0.7177,
            ], [0.0, 0.0, 0.0]),

            Self::SDXL => (vec![
                 0.3651,  0.4232,  0.4341,
                -0.2533, -0.0042,  0.1068,
                 0.1076,  0.1111, -0.0362,
                -0.3165, -0.2492, -0.2188,
            ], [0.1084, -0.0175, -0.0011]),

            Self::SDXLPlayground25 => (vec![
                 0.3920,  0.4054,  0.4549,
                -0.2634, -0.0196,  0.0653,
                 0.0568,  0.1687, -0.0755,
                -0.3112, -0.2359, -0.2076,
            ], [0.0, 0.0, 0.0]),

            Self::SDX4 => (vec![
                -0.2340, -0.3863, -0.3257,
                 0.0994,  0.0885, -0.0908,
                -0.2833, -0.2349, -0.3741,
                 0.2523, -0.0055, -0.1651,
            ], [0.0, 0.0, 0.0]),

            Self::SCB => (vec![
                 0.1121,  0.2006,  0.1023,
                -0.2093, -0.0222, -0.0195,
                -0.3087, -0.1535,  0.0366,
                 0.0290, -0.1574, -0.4078,
            ], [0.0, 0.0, 0.0]),

            // 12-channel models
            Self::Mochi => (vec![
                -0.0069, -0.0045,  0.0018,
                 0.0154, -0.0692, -0.0274,
                 0.0333,  0.0019,  0.0206,
                -0.1390,  0.0628,  0.1678,
                -0.0725,  0.0134, -0.1898,
                 0.0074, -0.0270, -0.0209,
                -0.0176, -0.0277, -0.0221,
                 0.5294,  0.5204,  0.3852,
                -0.0326, -0.0446, -0.0143,
                -0.0659,  0.0153, -0.0153,
                 0.0185, -0.0217,  0.0014,
                -0.0396, -0.0495, -0.0281,
            ], [-0.0940, -0.1418, -0.1453]),

            // 16-channel models
            Self::SD3 => (vec![
                -0.0645,  0.0177,  0.1052,
                 0.0028,  0.0312,  0.0650,
                 0.1848,  0.0762,  0.0360,
                 0.0944,  0.0360,  0.0889,
                 0.0897,  0.0506, -0.0364,
                -0.0020,  0.1203,  0.0284,
                 0.0855,  0.0118,  0.0283,
                -0.0539,  0.1160,  0.1077,
                -0.0057,  0.0116,  0.0700,
                -0.0412,  0.0281, -0.0039,
                 0.1106,  0.1171,  0.1220,
                -0.0248,  0.0682, -0.0481,
                 0.0815,  0.0846,  0.1207,
                -0.0120, -0.0055, -0.1463,
                 0.0020,  0.0523,  0.1994,
                 0.0339,  0.0254,  0.0459,
            ], [0.2394, 0.2135, 0.1925]),

            Self::Flux => (vec![
                -0.0346,  0.0244,  0.0681,
                 0.0034,  0.0210,  0.0687,
                 0.0437,  0.0689,  0.0471,
                 0.0440,  0.0967,  0.0728,
                 0.0386,  0.0445, -0.0601,
                 0.0444,  0.0921,  0.0335,
                 0.0986,  0.0331,  0.0743,
                -0.0279,  0.0553,  0.0452,
                -0.0060,  0.0298,  0.0636,
                 0.0349,  0.0784,  0.0276,
                 0.0732,  0.0735,  0.0148,
                 0.0091,  0.0420,  0.0073,
                 0.0719,  0.0669,  0.0757,
                 0.0270,  0.0658,  0.0031,
                -0.0156,  0.0353,  0.0604,
                 0.0472,  0.0316,  0.0701,
            ], [-0.0329, -0.0718, -0.0851]),

            Self::SCPrior => (vec![
                -0.0326, -0.0204, -0.0127,
                -0.1592, -0.0427,  0.0216,
                 0.0873,  0.0638, -0.0020,
                -0.0602,  0.0442,  0.1304,
                 0.0800, -0.0313, -0.1796,
                -0.0810, -0.0638, -0.1581,
                 0.1791,  0.1180,  0.0967,
                 0.0740,  0.1416,  0.0432,
                -0.1745, -0.1888, -0.1373,
                 0.2412,  0.1577,  0.0928,
                 0.1908,  0.0998,  0.0682,
                 0.0209,  0.0365, -0.0092,
                 0.0448, -0.0650, -0.1728,
                -0.1658, -0.1045, -0.1308,
                 0.0542,  0.1545,  0.1325,
                -0.0352, -0.1672, -0.2541,
            ], [0.0, 0.0, 0.0]),

            Self::HunyuanVideo => (vec![
                -0.0395, -0.0331,  0.0445,
                 0.0696,  0.0795,  0.0518,
                 0.0135, -0.0945, -0.0282,
                 0.0108, -0.0250, -0.0765,
                -0.0209,  0.0032,  0.0224,
                -0.0804, -0.0254, -0.0639,
                -0.0991,  0.0271, -0.0669,
                -0.0646, -0.0422, -0.0400,
                -0.0696, -0.0595, -0.0894,
                -0.0799, -0.0208, -0.0375,
                 0.1166,  0.1627,  0.0962,
                 0.1165,  0.0432,  0.0407,
                -0.2315, -0.1920, -0.1355,
                -0.0270,  0.0401, -0.0821,
                -0.0616, -0.0997, -0.0727,
                 0.0249, -0.0469, -0.1703,
            ], [0.0259, -0.0192, -0.0761]),

            Self::Wan21 => (vec![
                -0.1299, -0.1692,  0.2932,
                 0.0671,  0.0406,  0.0442,
                 0.3568,  0.2548,  0.1747,
                 0.0372,  0.2344,  0.1420,
                 0.0313,  0.0189, -0.0328,
                 0.0296, -0.0956, -0.0665,
                -0.3477, -0.4059, -0.2925,
                 0.0166,  0.1902,  0.1975,
                -0.0412,  0.0267, -0.1364,
                -0.1293,  0.0740,  0.1636,
                 0.0680,  0.3019,  0.1128,
                 0.0032,  0.0581,  0.0639,
                -0.1251,  0.0927,  0.1699,
                 0.0060, -0.0633,  0.0005,
                 0.3477,  0.2275,  0.2950,
                 0.1984,  0.0913,  0.1861,
            ], [-0.1835, -0.0868, -0.3360]),

            // 32-channel models
            Self::HunyuanVideo15 => (vec![
                 0.0568, -0.0521, -0.0131,
                 0.0014,  0.0735,  0.0326,
                 0.0186,  0.0531, -0.0138,
                -0.0031,  0.0051,  0.0288,
                 0.0110,  0.0556,  0.0432,
                -0.0041, -0.0023, -0.0485,
                 0.0530,  0.0413,  0.0253,
                 0.0283,  0.0251,  0.0339,
                 0.0277, -0.0372, -0.0093,
                 0.0393,  0.0944,  0.1131,
                 0.0020,  0.0251,  0.0037,
                -0.0017,  0.0012,  0.0234,
                 0.0468,  0.0436,  0.0203,
                 0.0354,  0.0439, -0.0233,
                 0.0090,  0.0123,  0.0346,
                 0.0382,  0.0029,  0.0217,
                 0.0261, -0.0300,  0.0030,
                -0.0088, -0.0220, -0.0283,
                -0.0272, -0.0121, -0.0363,
                -0.0664, -0.0622,  0.0144,
                 0.0414,  0.0479,  0.0529,
                 0.0355,  0.0612, -0.0247,
                 0.0147,  0.0264,  0.0174,
                 0.0438,  0.0038,  0.0542,
                 0.0431, -0.0573, -0.0033,
                -0.0162, -0.0211, -0.0406,
                -0.0487, -0.0295, -0.0393,
                 0.0005, -0.0109,  0.0253,
                 0.0296,  0.0591,  0.0353,
                 0.0119,  0.0181, -0.0306,
                -0.0085, -0.0362,  0.0229,
                 0.0005, -0.0106,  0.0242,
            ], [0.0456, -0.0202, -0.0644]),

            Self::Flux2 => (vec![
                 0.0058,  0.0113,  0.0073,
                 0.0495,  0.0443,  0.0836,
                -0.0099,  0.0096,  0.0644,
                 0.2144,  0.3009,  0.3652,
                 0.0166, -0.0039, -0.0054,
                 0.0157,  0.0103, -0.0160,
                -0.0398,  0.0902, -0.0235,
                -0.0052,  0.0095,  0.0109,
                -0.3527, -0.2712, -0.1666,
                -0.0301, -0.0356, -0.0180,
                -0.0107,  0.0078,  0.0013,
                 0.0746,  0.0090, -0.0941,
                 0.0156,  0.0169,  0.0070,
                -0.0034, -0.0040, -0.0114,
                 0.0032,  0.0181,  0.0080,
                -0.0939, -0.0008,  0.0186,
                 0.0018,  0.0043,  0.0104,
                 0.0284,  0.0056, -0.0127,
                -0.0024, -0.0022, -0.0030,
                 0.1207, -0.0026,  0.0065,
                 0.0128,  0.0101,  0.0142,
                 0.0137, -0.0072, -0.0007,
                 0.0095,  0.0092, -0.0059,
                 0.0000, -0.0077, -0.0049,
                -0.0465, -0.0204, -0.0312,
                 0.0095,  0.0012, -0.0066,
                 0.0290, -0.0034,  0.0025,
                 0.0220,  0.0169, -0.0048,
                -0.0332, -0.0457, -0.0468,
                -0.0085,  0.0389,  0.0609,
                -0.0076,  0.0003, -0.0043,
                -0.0111, -0.0460, -0.0614,
            ], [-0.0329, -0.0718, -0.0851]),

            // 48-channel models
            Self::Wan22 => (vec![
                 0.0119,  0.0103,  0.0046,
                -0.1062, -0.0504,  0.0165,
                 0.0140,  0.0409,  0.0491,
                -0.0813, -0.0677,  0.0607,
                 0.0656,  0.0851,  0.0808,
                 0.0264,  0.0463,  0.0912,
                 0.0295,  0.0326,  0.0590,
                -0.0244, -0.0270,  0.0025,
                 0.0443, -0.0102,  0.0288,
                -0.0465, -0.0090, -0.0205,
                 0.0359,  0.0236,  0.0082,
                -0.0776,  0.0854,  0.1048,
                 0.0564,  0.0264,  0.0561,
                 0.0006,  0.0594,  0.0418,
                -0.0319, -0.0542, -0.0637,
                -0.0268,  0.0024,  0.0260,
                 0.0539,  0.0265,  0.0358,
                -0.0359, -0.0312, -0.0287,
                -0.0285, -0.1032, -0.1237,
                 0.1041,  0.0537,  0.0622,
                -0.0086, -0.0374, -0.0051,
                 0.0390,  0.0670,  0.2863,
                 0.0069,  0.0144,  0.0082,
                 0.0006, -0.0167,  0.0079,
                 0.0313, -0.0574, -0.0232,
                -0.1454, -0.0902, -0.0481,
                 0.0714,  0.0827,  0.0447,
                -0.0304, -0.0574, -0.0196,
                 0.0401,  0.0384,  0.0204,
                -0.0758, -0.0297, -0.0014,
                 0.0568,  0.1307,  0.1372,
                -0.0055, -0.0310, -0.0380,
                 0.0239, -0.0305,  0.0325,
                -0.0663, -0.0673, -0.0140,
                -0.0416, -0.0047, -0.0023,
                 0.0166,  0.0112, -0.0093,
                -0.0211,  0.0011,  0.0331,
                 0.1833,  0.1466,  0.2250,
                -0.0368,  0.0370,  0.0295,
                -0.3441, -0.3543, -0.2008,
                -0.0479, -0.0489, -0.0420,
                -0.0660, -0.0153,  0.0800,
                -0.0101,  0.0068,  0.0156,
                -0.0690, -0.0452, -0.0927,
                -0.0145,  0.0041,  0.0015,
                 0.0421,  0.0451,  0.0373,
                 0.0504, -0.0483, -0.0356,
                -0.0837,  0.0168,  0.0055,
            ], [0.0317, -0.0878, -0.1388]),

            // 128-channel models
            Self::LTXV => (vec![
                 0.0112, -0.0006, -0.0100,
                 0.0860,  0.0658,  0.0010,
                -0.0126, -0.0076, -0.0041,
                 0.0094, -0.0022,  0.0026,
                 0.0038,  0.0128,  0.0092,
                 0.0210, -0.0053,  0.0034,
                -0.0089, -0.0197, -0.0188,
                -0.0132, -0.0105,  0.0020,
                -0.0015, -0.0070, -0.0076,
                -0.0017,  0.0005, -0.0034,
                 0.0136,  0.0047, -0.0020,
                 0.0103,  0.0077,  0.0139,
                -0.0161, -0.0062,  0.0012,
                 0.0073,  0.0156,  0.0004,
                 0.0010, -0.0030, -0.0148,
                 0.0191,  0.0109,  0.0123,
                 0.0045,  0.0000, -0.0069,
                -0.0005,  0.0033,  0.0078,
                 0.0339,  0.0334,  0.0375,
                -0.0230, -0.0025, -0.0031,
                 0.0503,  0.0388,  0.0335,
                -0.0041, -0.0011,  0.0016,
                -0.1269, -0.1311, -0.2101,
                 0.0263,  0.0142, -0.0036,
                -0.0049,  0.0088,  0.0078,
                -0.0017, -0.0049, -0.0052,
                -0.0021,  0.0024,  0.0094,
                -0.0225, -0.0213, -0.0151,
                -0.0158, -0.0106, -0.0065,
                -0.0047,  0.0050, -0.0067,
                 0.0120,  0.0207,  0.0162,
                -0.0064, -0.0085, -0.0095,
                 0.0073, -0.0099, -0.0230,
                -0.0009,  0.0063,  0.0096,
                -0.0372, -0.0371, -0.0567,
                -0.1337, -0.1072, -0.0538,
                -0.0054,  0.0081,  0.0088,
                -0.1525, -0.2144, -0.2184,
                 0.0314,  0.0070, -0.0098,
                 0.0022, -0.0090, -0.0210,
                 0.0038, -0.0059, -0.0150,
                -0.0043, -0.0129, -0.0160,
                -0.0055, -0.0108, -0.0030,
                -0.0065,  0.0031, -0.0102,
                -0.0050, -0.0072, -0.0009,
                -0.0086, -0.0024,  0.0011,
                -0.0090, -0.0096,  0.0016,
                 0.0051,  0.0121,  0.0200,
                 0.0138,  0.0117,  0.0082,
                -0.0105, -0.0116, -0.0041,
                -0.0284, -0.0313, -0.0221,
                 0.0029,  0.0365,  0.0187,
                -0.0167, -0.0167, -0.0045,
                 0.0488,  0.0401,  0.0087,
                -0.0151, -0.0006,  0.0030,
                -0.0176, -0.0081,  0.0131,
                -0.0093,  0.0108, -0.0063,
                 0.0031,  0.0005,  0.0123,
                -0.0228, -0.0230, -0.0260,
                -0.0248, -0.0154, -0.0221,
                -0.0236,  0.0011,  0.0124,
                -0.0079, -0.0012, -0.0061,
                -0.0115, -0.0013,  0.0063,
                -0.0542,  0.0266,  0.0063,
                 0.0044, -0.0073, -0.0105,
                -0.0045,  0.0016,  0.0144,
                 0.0137,  0.0089,  0.0041,
                -0.0101,  0.0090,  0.0157,
                -0.0056,  0.0012,  0.0081,
                -0.0037, -0.0054,  0.0013,
                 0.0295,  0.0214,  0.0304,
                -0.0349, -0.0243, -0.0253,
                -0.0341, -0.0224, -0.0106,
                -0.0173, -0.0132, -0.0107,
                -0.0021, -0.0086, -0.0030,
                 0.0012, -0.0042, -0.0069,
                 0.0009, -0.0067, -0.0001,
                 0.0160, -0.0101, -0.0289,
                 0.0012,  0.0102,  0.0189,
                 0.0173,  0.0003,  0.0138,
                -0.0135, -0.0036,  0.0007,
                 0.0047, -0.0052,  0.0024,
                -0.0059, -0.0062, -0.0018,
                 0.0155,  0.0146,  0.0020,
                 0.0075,  0.0016, -0.0082,
                 0.0191,  0.0016, -0.0040,
                -0.0057, -0.0027, -0.0041,
                 0.0017,  0.0146,  0.0258,
                -0.0008,  0.0023,  0.0045,
                 0.0116,  0.0089, -0.0073,
                 0.0076,  0.0027,  0.0114,
                 0.0052,  0.0037,  0.0140,
                -0.0184, -0.0225, -0.0245,
                 0.0006, -0.0058, -0.0148,
                -0.0161, -0.0086, -0.0145,
                 0.0205,  0.0207,  0.0064,
                 0.0034, -0.0112, -0.0164,
                -0.0015, -0.0105,  0.0017,
                 0.0281,  0.0235,  0.0328,
                -0.0185, -0.0128, -0.0088,
                -0.0081, -0.0108, -0.0175,
                -0.0039,  0.0162,  0.0334,
                -0.0075, -0.0142, -0.0062,
                 0.0035, -0.0114, -0.0106,
                 0.0115,  0.0039,  0.0028,
                 0.0072, -0.0015, -0.0038,
                 0.0022, -0.0088, -0.0096,
                 0.0241,  0.0217,  0.0281,
                -0.0054, -0.0243, -0.0178,
                 0.0074,  0.0105,  0.0127,
                 0.0063,  0.0063,  0.0192,
                 0.0164,  0.0095,  0.0067,
                 0.0172,  0.0236,  0.0233,
                -0.0146, -0.0098, -0.0116,
                 0.0144,  0.0144,  0.0066,
                -0.0068,  0.0189,  0.0146,
                 0.0061,  0.0035, -0.0027,
                -0.0027, -0.0059, -0.0092,
                 0.0102,  0.0074, -0.0076,
                -0.0133,  0.0193, -0.0009,
                 0.0024, -0.0048, -0.0158,
                 0.0262,  0.0260,  0.0202,
                 0.0157,  0.0185,  0.0027,
                -0.0022,  0.0047, -0.0224,
                -0.0075,  0.0074,  0.0144,
                -0.0084, -0.0080,  0.0098,
                 0.0383,  0.0097, -0.0193,
                -0.0146, -0.0067,  0.0040,
            ], [-0.0571, -0.1657, -0.2512]),
        }
    }

    /// Get format that matches a given channel count (for auto-detection)
    pub fn for_channels(channels: usize) -> Self {
        match channels {
            4 => Self::SD15,
            12 => Self::Mochi,
            16 => Self::SD3,
            32 => Self::HunyuanVideo15,
            48 => Self::Wan22,
            128 => Self::LTXV,
            _ => Self::SD15, // fallback
        }
    }
}

/// Convert latent to RGB preview without VAE decode
///
/// Uses learned coefficients to project latent channels to RGB.
/// Much faster than VAE decode (~0.1ms vs ~50ms).
///
/// Coefficients from ComfyUI's latent_formats.py (GPL-3.0):
/// https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/latent_formats.py
pub fn latent_to_preview<B: Backend>(latent: Tensor<B, 4>, format: LatentFormat) -> Tensor<B, 4> {
    let [b, c, h, w] = latent.dims();
    let device = latent.device();

    // Use format's coefficients, or auto-detect based on channel count
    let effective_format = if format.channels() == c {
        format
    } else {
        LatentFormat::for_channels(c)
    };

    let (coefs_vec, bias) = effective_format.rgb_coefs();
    let channels = effective_format.channels();

    let coefs: Tensor<B, 2> = Tensor::from_data(TensorData::new(coefs_vec, [channels, 3]), &device);

    // einsum "cxy,cr -> rxy": project latent channels to 3 RGB channels
    let flat = latent.reshape([b, channels, h * w]); // [B, C, H*W]
    let flat_t = flat.swap_dims(1, 2); // [B, H*W, C]
    let coefs_3d = coefs.unsqueeze::<3>().repeat_dim(0, b); // [B, C, 3]
    let rgb_flat = flat_t.matmul(coefs_3d); // [B, H*W, 3]
    let rgb = rgb_flat.swap_dims(1, 2).reshape([b, 3, h, w]); // [B, 3, H, W]

    // Apply bias and normalize to [0, 255]
    let bias_tensor: Tensor<B, 1> = Tensor::from_data(TensorData::new(bias.to_vec(), [3]), &device);
    let rgb_biased = rgb + bias_tensor.reshape([1, 3, 1, 1]);
    let normalized = (rgb_biased + 0.5) * 255.0;
    normalized.clamp(0.0, 255.0)
}

/// Unified interface for diffusion pipelines
pub trait DiffusionPipeline<B: Backend> {
    type Conditioning;

    /// Encode text prompt into conditioning
    fn encode_prompt(&self, prompt: &str, negative_prompt: &str) -> Self::Conditioning;

    /// Sample latent from conditioning
    fn sample_latent(
        &self,
        conditioning: &Self::Conditioning,
        config: &SampleConfig,
    ) -> Tensor<B, 4>;

    /// Decode latent to image
    fn decode(&self, latent: Tensor<B, 4>, config: &SampleConfig) -> Tensor<B, 4>;

    /// Full pipeline: prompt -> image
    fn generate(&self, prompt: &str, negative_prompt: &str, config: &SampleConfig) -> Tensor<B, 4> {
        let conditioning = self.encode_prompt(prompt, negative_prompt);
        let latent = self.sample_latent(&conditioning, config);
        self.decode(latent, config)
    }
}

/// Helper to compute tensor statistics for debugging
pub(crate) fn tensor_stats<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> String {
    let data = tensor.clone().into_data();
    let floats: Vec<f32> = data.convert::<f32>().to_vec().unwrap();

    if floats.is_empty() {
        return "empty".to_string();
    }

    let nan_count = floats.iter().filter(|x| x.is_nan()).count();
    let inf_count = floats.iter().filter(|x| x.is_infinite()).count();
    let min = floats.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = floats.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = floats.iter().sum();
    let mean = sum / floats.len() as f32;
    let var: f32 = floats.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / floats.len() as f32;
    let std = var.sqrt();

    if nan_count > 0 || inf_count > 0 {
        format!(
            "min={:.4}, max={:.4}, mean={:.4}, std={:.4} [NaN={}, Inf={}]",
            min, max, mean, std, nan_count, inf_count
        )
    } else {
        format!(
            "min={:.4}, max={:.4}, mean={:.4}, std={:.4}",
            min, max, mean, std
        )
    }
}

/// Check tensor for NaN/Inf values
///
/// When `enabled` is true, this will panic if NaN or Inf is detected.
/// Use this to catch numerical issues early in the forward pass.
#[inline]
pub(crate) fn check_tensor_if<B: Backend, const D: usize>(
    tensor: &Tensor<B, D>,
    name: &str,
    enabled: bool,
) {
    if !enabled {
        return;
    }

    let data = tensor.clone().into_data();
    let floats: Vec<f32> = data.convert::<f32>().to_vec().unwrap();

    let nan_count = floats.iter().filter(|x| x.is_nan()).count();
    let inf_count = floats.iter().filter(|x| x.is_infinite()).count();

    if nan_count > 0 || inf_count > 0 {
        let total = floats.len();
        panic!(
            "[NaN check failed] {}: {}/{} values are NaN, {}/{} are Inf\nStats: {}",
            name,
            nan_count,
            total,
            inf_count,
            total,
            tensor_stats(tensor)
        );
    }
}

/// Helper to convert output tensor to image bytes (RGB, 0-255)
pub fn tensor_to_rgb<B: Backend>(tensor: Tensor<B, 4>) -> Vec<u8> {
    let [_, _, h, w] = tensor.dims();

    // Clamp to [0, 255] and convert
    let tensor = tensor.clamp(0.0, 255.0);

    // Get data as f32 (convert if backend uses different precision)
    let data = tensor.into_data();
    let floats: Vec<f32> = data.convert::<f32>().to_vec().unwrap();

    // Convert to u8 RGB (assuming tensor is [1, 3, H, W])
    let mut rgb = Vec::with_capacity(h * w * 3);
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let idx = c * h * w + y * w + x;
                rgb.push(floats[idx] as u8);
            }
        }
    }

    rgb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_config_default() {
        let config = SampleConfig::default();
        assert_eq!(config.width, 512);
        assert_eq!(config.height, 512);
        assert_eq!(config.steps, 50);
        assert_eq!(config.guidance_scale, 7.5);
    }
}
