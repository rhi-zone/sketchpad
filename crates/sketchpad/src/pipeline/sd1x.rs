//! Stable Diffusion 1.x pipeline implementations

use burn::prelude::*;
use burn::tensor::Int;

use sketchpad_clip::{ClipConfig, ClipTextEncoder, ClipTokenizer, END_OF_TEXT};
use sketchpad_samplers::{
    DdimConfig, DdimSampler, DpmConfig, DpmPlusPlusSampler, NoiseSchedule, SigmaSchedule,
    apply_guidance,
};
use sketchpad_unet::{UNet, UNetConfig};
use sketchpad_vae::{Decoder, DecoderConfig, Encoder, EncoderConfig};

use super::{
    DiffusionPipeline, LatentFormat, SampleConfig, SamplerType, StepInfo, StepOutput,
    check_tensor_if, latent_to_preview, tensor_stats,
};

/// SD 1.x conditioning (text embeddings)
pub struct Sd1xConditioning<B: Backend> {
    /// Conditional text embedding [batch, seq_len, embed_dim]
    pub cond: Tensor<B, 3>,
    /// Unconditional text embedding [batch, seq_len, embed_dim]
    pub uncond: Tensor<B, 3>,
}

/// Stable Diffusion 1.x Pipeline
pub struct StableDiffusion1x<B: Backend> {
    pub tokenizer: ClipTokenizer,
    pub text_encoder: ClipTextEncoder<B>,
    pub unet: UNet<B>,
    pub vae_decoder: Decoder<B>,
    pub scheduler: NoiseSchedule<B>,
    pub device: B::Device,
}

impl<B: Backend> StableDiffusion1x<B> {
    /// Create a new SD 1.x pipeline with default configs
    pub fn new(tokenizer: ClipTokenizer, device: &B::Device) -> Self {
        let clip_config = ClipConfig::sd1x();
        let unet_config = UNetConfig::sd1x();
        let vae_config = DecoderConfig::sd();

        Self {
            tokenizer,
            text_encoder: ClipTextEncoder::new(&clip_config, device),
            unet: UNet::new(&unet_config, device),
            vae_decoder: Decoder::new(&vae_config, device),
            scheduler: NoiseSchedule::sd1x(device),
            device: device.clone(),
        }
    }

    /// Create with custom configs
    pub fn with_configs(
        tokenizer: ClipTokenizer,
        clip_config: &ClipConfig,
        unet_config: &UNetConfig,
        vae_config: &DecoderConfig,
        device: &B::Device,
    ) -> Self {
        Self {
            tokenizer,
            text_encoder: ClipTextEncoder::new(clip_config, device),
            unet: UNet::new(unet_config, device),
            vae_decoder: Decoder::new(vae_config, device),
            scheduler: NoiseSchedule::sd1x(device),
            device: device.clone(),
        }
    }

    /// Encode a single text prompt to CLIP embeddings
    fn encode_text(&self, text: &str) -> Tensor<B, 3> {
        let tokens = self.tokenizer.encode_padded(text, 77);
        let token_tensor: Tensor<B, 1, Int> = Tensor::from_data(
            TensorData::new(tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(), [77]),
            &self.device,
        );
        let token_tensor = token_tensor.unsqueeze::<2>(); // [1, 77]

        self.text_encoder.forward(token_tensor)
    }

    /// Find the position of the end-of-sequence token in the token list
    #[allow(dead_code)]
    fn find_eos_position(tokens: &[u32]) -> usize {
        tokens
            .iter()
            .position(|&t| t == END_OF_TEXT)
            .unwrap_or(tokens.len() - 1)
    }
}

impl<B: Backend> DiffusionPipeline<B> for StableDiffusion1x<B> {
    type Conditioning = Sd1xConditioning<B>;

    fn encode_prompt(&self, prompt: &str, negative_prompt: &str) -> Self::Conditioning {
        let cond = self.encode_text(prompt);
        let uncond = self.encode_text(if negative_prompt.is_empty() {
            ""
        } else {
            negative_prompt
        });
        Sd1xConditioning { cond, uncond }
    }

    fn sample_latent(
        &self,
        conditioning: &Self::Conditioning,
        config: &SampleConfig,
    ) -> Tensor<B, 4> {
        let debug = config.debug.sampler;
        let debug_nan = config.debug.nan;
        let latent_height = config.height / 8;
        let latent_width = config.width / 8;

        // Create DDIM sampler
        let ddim_config = DdimConfig {
            num_inference_steps: config.steps,
            eta: 0.0,
        };
        let sampler = DdimSampler::new(NoiseSchedule::sd1x(&self.device), ddim_config);

        // Initialize with random noise
        let mut latent = sampler.init_latent(1, 4, latent_height, latent_width, &self.device);

        // Debug: print initial latent stats
        if debug {
            eprintln!("[debug] Initial latent: {}", tensor_stats(&latent));
            eprintln!("[debug] Cond shape: {:?}", conditioning.cond.dims());
        }

        // Precompute all timestep tensors to avoid CPU->GPU transfer in hot loop
        let timestep_tensors: Vec<Tensor<B, 1>> = sampler
            .timesteps()
            .iter()
            .map(|&t| Tensor::<B, 1>::from_data(TensorData::new(vec![t as f32], [1]), &self.device))
            .collect();

        // Sampling loop
        for step_idx in 0..sampler.num_steps() {
            let t = timestep_tensors[step_idx].clone();

            // Predict noise for unconditional
            let noise_uncond =
                self.unet
                    .forward(latent.clone(), t.clone(), conditioning.uncond.clone());
            check_tensor_if(
                &noise_uncond,
                &format!("step_{}_noise_uncond", step_idx),
                debug_nan,
            );

            // Predict noise for conditional
            let noise_cond = self
                .unet
                .forward(latent.clone(), t, conditioning.cond.clone());
            check_tensor_if(
                &noise_cond,
                &format!("step_{}_noise_cond", step_idx),
                debug_nan,
            );

            // Debug: print noise predictions for first few steps
            if debug && step_idx < 3 {
                eprintln!(
                    "[debug] Step {} - noise_uncond: {}, noise_cond: {}",
                    step_idx,
                    tensor_stats(&noise_uncond),
                    tensor_stats(&noise_cond)
                );
            }

            // Apply classifier-free guidance
            let noise_pred = apply_guidance(noise_uncond, noise_cond, config.guidance_scale);

            // DDIM step
            latent = sampler.step(latent, noise_pred, step_idx);
            check_tensor_if(&latent, &format!("step_{}_latent", step_idx), debug_nan);

            // Debug: print latent stats for first few steps
            if debug && step_idx < 3 {
                eprintln!(
                    "[debug] Step {} - latent: {}",
                    step_idx,
                    tensor_stats(&latent)
                );
            }
        }

        if debug {
            eprintln!("[debug] Final latent: {}", tensor_stats(&latent));
        }

        latent
    }

    fn decode(&self, latent: Tensor<B, 4>, config: &SampleConfig) -> Tensor<B, 4> {
        let debug = config.debug.sampler;
        let debug_nan = config.debug.nan;

        check_tensor_if(&latent, "vae_input_latent", debug_nan);

        // Note: decode_to_image internally applies the 0.18215 scaling factor
        let result = self.vae_decoder.decode_to_image(latent);
        check_tensor_if(&result, "vae_output", debug_nan);

        if debug {
            eprintln!("[debug] VAE output: {}", tensor_stats(&result));
        }
        result
    }
}

impl<B: Backend> StableDiffusion1x<B> {
    /// Sample latent with step callback for progress reporting
    ///
    /// The callback is called after each step with step info and optional output.
    /// Use `StepOutput::None` for minimal overhead, or `StepOutput::LatentPreview`
    /// for cheap visual feedback.
    pub fn sample_latent_with_callback<F>(
        &self,
        conditioning: &Sd1xConditioning<B>,
        config: &SampleConfig,
        step_output: StepOutput,
        callback: F,
    ) -> Tensor<B, 4>
    where
        F: FnMut(StepInfo<B>),
    {
        let debug = config.debug.sampler;
        let latent_height = config.height / 8;
        let latent_width = config.width / 8;
        let schedule = NoiseSchedule::sd1x(&self.device);
        let sigma_schedule = config.schedule.to_sigma_schedule();

        // Dispatch to appropriate sampler based on config
        match config.sampler {
            SamplerType::DpmPlusPlus => self.sample_with_dpm_pp(
                conditioning,
                config,
                step_output,
                callback,
                latent_height,
                latent_width,
                &schedule,
                sigma_schedule,
                debug,
            ),
            SamplerType::Ddim => self.sample_with_ddim(
                conditioning,
                config,
                step_output,
                callback,
                latent_height,
                latent_width,
                schedule,
                debug,
            ),
            // Default to DPM++ for other samplers (TODO: implement more)
            _ => {
                if debug {
                    eprintln!(
                        "[debug] Sampler {:?} not implemented, using DPM++ 2M",
                        config.sampler
                    );
                }
                self.sample_with_dpm_pp(
                    conditioning,
                    config,
                    step_output,
                    callback,
                    latent_height,
                    latent_width,
                    &schedule,
                    sigma_schedule,
                    debug,
                )
            }
        }
    }

    /// Sample using DDIM sampler
    #[allow(clippy::too_many_arguments)]
    fn sample_with_ddim<F>(
        &self,
        conditioning: &Sd1xConditioning<B>,
        config: &SampleConfig,
        step_output: StepOutput,
        mut callback: F,
        latent_height: usize,
        latent_width: usize,
        schedule: NoiseSchedule<B>,
        debug: bool,
    ) -> Tensor<B, 4>
    where
        F: FnMut(StepInfo<B>),
    {
        let debug_nan = config.debug.nan;
        let ddim_config = DdimConfig {
            num_inference_steps: config.steps,
            eta: 0.0,
        };
        let sampler = DdimSampler::new(schedule, ddim_config);

        let mut latent = sampler.init_latent(1, 4, latent_height, latent_width, &self.device);

        if debug {
            eprintln!("[debug] DDIM sampler, {} steps", config.steps);
            eprintln!("[debug] Initial latent: {}", tensor_stats(&latent));
        }

        let timestep_tensors: Vec<Tensor<B, 1>> = sampler
            .timesteps()
            .iter()
            .map(|&t| Tensor::<B, 1>::from_data(TensorData::new(vec![t as f32], [1]), &self.device))
            .collect();

        let timesteps = sampler.timesteps();
        let total_steps = sampler.num_steps();

        for step_idx in 0..total_steps {
            let t = timestep_tensors[step_idx].clone();

            let noise_uncond =
                self.unet
                    .forward(latent.clone(), t.clone(), conditioning.uncond.clone());
            check_tensor_if(
                &noise_uncond,
                &format!("step_{}_noise_uncond", step_idx),
                debug_nan,
            );

            let noise_cond = self
                .unet
                .forward(latent.clone(), t, conditioning.cond.clone());
            check_tensor_if(
                &noise_cond,
                &format!("step_{}_noise_cond", step_idx),
                debug_nan,
            );

            if debug && step_idx < 3 {
                eprintln!(
                    "[debug] Step {} t={} - noise_uncond: {}, noise_cond: {}",
                    step_idx,
                    timesteps[step_idx],
                    tensor_stats(&noise_uncond),
                    tensor_stats(&noise_cond)
                );
            }

            let noise_pred = apply_guidance(noise_uncond, noise_cond, config.guidance_scale);
            latent = sampler.step(latent, noise_pred, step_idx);
            check_tensor_if(&latent, &format!("step_{}_latent", step_idx), debug_nan);

            let output = match step_output {
                StepOutput::None => None,
                StepOutput::Latent => Some(latent.clone()),
                StepOutput::LatentPreview => {
                    Some(latent_to_preview(latent.clone(), LatentFormat::SD15))
                }
                StepOutput::Decoded => Some(self.vae_decoder.decode_to_image(latent.clone())),
            };

            callback(StepInfo {
                step: step_idx,
                total_steps,
                timestep: timesteps[step_idx],
                output,
            });
        }

        latent
    }

    /// Sample using DPM++ 2M sampler
    #[allow(clippy::too_many_arguments)]
    fn sample_with_dpm_pp<F>(
        &self,
        conditioning: &Sd1xConditioning<B>,
        config: &SampleConfig,
        step_output: StepOutput,
        mut callback: F,
        latent_height: usize,
        latent_width: usize,
        _schedule: &NoiseSchedule<B>,
        sigma_schedule: SigmaSchedule,
        debug: bool,
    ) -> Tensor<B, 4>
    where
        F: FnMut(StepInfo<B>),
    {
        let debug_nan = config.debug.nan;

        // Create schedule for DPM++
        let dpm_schedule = NoiseSchedule::sd1x(&self.device);
        let dpm_config = DpmConfig {
            num_inference_steps: config.steps,
            solver_order: 2,
            sigma_schedule,
            debug,
        };
        let mut sampler = DpmPlusPlusSampler::new(dpm_schedule, dpm_config, &self.device);

        // Get sigmas and compute proper timesteps
        let total_steps = sampler.num_steps();
        let sigmas = sampler.sigmas().to_vec();

        // For non-Normal schedules (Karras, etc.), the sigmas are respaced and don't
        // correspond to the original evenly-spaced timesteps. We need to find the
        // training timestep that corresponds to each sigma for correct UNet conditioning.
        let timesteps: Vec<usize> = if sigma_schedule == SigmaSchedule::Normal {
            sampler.timesteps().to_vec()
        } else {
            // Convert sigmas to training timesteps (excluding final sigma=0)
            // Recreate schedule just for the conversion (lightweight operation)
            let schedule_for_conversion = NoiseSchedule::<B>::sd1x(&self.device);
            schedule_for_conversion.sigmas_to_timesteps(&sigmas[..sigmas.len() - 1])
        };

        // Initialize latent scaled by initial sigma (using sampler's sigma)
        let mut latent = sampler.init_latent(1, 4, latent_height, latent_width, &self.device);

        if debug {
            eprintln!(
                "[debug] DPM++ 2M sampler, {} steps, init_sigma={:.4}",
                config.steps, sigmas[0]
            );
            eprintln!("[debug] Sigmas: {:?}", sigmas);
            eprintln!("[debug] Timesteps: {:?}", timesteps);
            eprintln!("[debug] Initial latent: {}", tensor_stats(&latent));
        }

        let timestep_tensors: Vec<Tensor<B, 1>> = timesteps
            .iter()
            .map(|&t| Tensor::<B, 1>::from_data(TensorData::new(vec![t as f32], [1]), &self.device))
            .collect();

        for step_idx in 0..total_steps {
            let t = timestep_tensors[step_idx].clone();
            let sigma = sigmas[step_idx];

            // k-diffusion input scaling: normalize variance for the UNet
            // c_in = 1 / sqrt(1 + sigma^2) brings sample variance back to ~1
            let c_in = 1.0 / (1.0 + sigma * sigma).sqrt();
            let latent_scaled = latent.clone() * c_in;

            let noise_uncond = self.unet.forward(
                latent_scaled.clone(),
                t.clone(),
                conditioning.uncond.clone(),
            );
            check_tensor_if(
                &noise_uncond,
                &format!("step_{}_noise_uncond", step_idx),
                debug_nan,
            );

            let noise_cond = self
                .unet
                .forward(latent_scaled, t, conditioning.cond.clone());
            check_tensor_if(
                &noise_cond,
                &format!("step_{}_noise_cond", step_idx),
                debug_nan,
            );

            if debug && step_idx < 3 {
                eprintln!(
                    "[debug] Step {} t={} sigma={:.4} c_in={:.4} - noise_uncond: {}, noise_cond: {}",
                    step_idx,
                    timesteps[step_idx],
                    sigma,
                    c_in,
                    tensor_stats(&noise_uncond),
                    tensor_stats(&noise_cond)
                );
            }

            let noise_pred = apply_guidance(noise_uncond, noise_cond, config.guidance_scale);

            // c_in scaling is ONLY for UNet input normalization.
            // The sampler step uses UNSCALED latent for both:
            //   - denoised = x - noise * sigma (x unscaled)
            //   - x_next = (sigma_next/sigma) * x + (1 - exp(-h)) * denoised
            latent = sampler.step(latent, noise_pred, step_idx);
            check_tensor_if(&latent, &format!("step_{}_latent", step_idx), debug_nan);

            let output = match step_output {
                StepOutput::None => None,
                StepOutput::Latent => Some(latent.clone()),
                StepOutput::LatentPreview => {
                    Some(latent_to_preview(latent.clone(), LatentFormat::SD15))
                }
                StepOutput::Decoded => Some(self.vae_decoder.decode_to_image(latent.clone())),
            };

            callback(StepInfo {
                step: step_idx,
                total_steps,
                timestep: timesteps[step_idx],
                output,
            });
        }

        if debug {
            eprintln!("[debug] Final latent (DPM++): {}", tensor_stats(&latent));
        }

        latent
    }

    /// Generate image with step callback
    pub fn generate_with_callback<F>(
        &self,
        prompt: &str,
        negative_prompt: &str,
        config: &SampleConfig,
        step_output: StepOutput,
        callback: F,
    ) -> Tensor<B, 4>
    where
        F: FnMut(StepInfo<B>),
    {
        let debug = config.debug.sampler;
        let conditioning =
            <Self as DiffusionPipeline<B>>::encode_prompt(self, prompt, negative_prompt);
        let latent = self.sample_latent_with_callback(&conditioning, config, step_output, callback);

        if debug {
            eprintln!("[debug] Latent before VAE: {}", tensor_stats(&latent));
        }

        let result = self.vae_decoder.decode_to_image(latent);
        if debug {
            eprintln!("[debug] VAE output: {}", tensor_stats(&result));
        }
        result
    }
}

/// Configuration for img2img sampling
#[derive(Debug, Clone)]
pub struct Img2ImgConfig {
    pub steps: usize,
    pub guidance_scale: f64,
    /// Strength of the transformation (0.0 = no change, 1.0 = full regeneration)
    pub strength: f64,
    pub seed: Option<u64>,
}

impl Default for Img2ImgConfig {
    fn default() -> Self {
        Self {
            steps: 50,
            guidance_scale: 7.5,
            strength: 0.75,
            seed: None,
        }
    }
}

/// Stable Diffusion 1.x Img2Img Pipeline
pub struct StableDiffusion1xImg2Img<B: Backend> {
    pub tokenizer: ClipTokenizer,
    pub text_encoder: ClipTextEncoder<B>,
    pub unet: UNet<B>,
    pub vae_encoder: Encoder<B>,
    pub vae_decoder: Decoder<B>,
    pub scheduler: NoiseSchedule<B>,
    device: B::Device,
}

impl<B: Backend> StableDiffusion1xImg2Img<B> {
    /// Create a new SD 1.x img2img pipeline with default configs
    pub fn new(tokenizer: ClipTokenizer, device: &B::Device) -> Self {
        let clip_config = ClipConfig::sd1x();
        let unet_config = UNetConfig::sd1x();
        let encoder_config = EncoderConfig::sd();
        let decoder_config = DecoderConfig::sd();

        Self {
            tokenizer,
            text_encoder: ClipTextEncoder::new(&clip_config, device),
            unet: UNet::new(&unet_config, device),
            vae_encoder: Encoder::new(&encoder_config, device),
            vae_decoder: Decoder::new(&decoder_config, device),
            scheduler: NoiseSchedule::sd1x(device),
            device: device.clone(),
        }
    }

    /// Encode a single text prompt
    fn encode_text(&self, text: &str) -> Tensor<B, 3> {
        let tokens = self.tokenizer.encode_padded(text, 77);
        let token_tensor: Tensor<B, 1, Int> = Tensor::from_data(
            TensorData::new(tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(), [77]),
            &self.device,
        );
        let token_tensor = token_tensor.unsqueeze::<2>();
        self.text_encoder.forward(token_tensor)
    }

    /// Generate image from input image and prompt
    ///
    /// # Arguments
    /// * `image` - Input image tensor [1, 3, H, W] with values in [0, 255]
    /// * `prompt` - Text prompt
    /// * `negative_prompt` - Negative prompt
    /// * `config` - Img2img configuration
    pub fn generate(
        &self,
        image: Tensor<B, 4>,
        prompt: &str,
        negative_prompt: &str,
        config: &Img2ImgConfig,
    ) -> Tensor<B, 4> {
        // Encode prompts
        let cond = self.encode_text(prompt);
        let uncond = self.encode_text(if negative_prompt.is_empty() {
            ""
        } else {
            negative_prompt
        });

        // Encode image to latent
        let init_latent = self.vae_encoder.encode_deterministic(image);

        // Calculate start step based on strength
        let num_inference_steps = config.steps;
        let start_step = ((1.0 - config.strength) * num_inference_steps as f64) as usize;

        // Create sampler
        let ddim_config = DdimConfig {
            num_inference_steps,
            eta: 0.0,
        };
        let sampler = DdimSampler::new(NoiseSchedule::sd1x(&self.device), ddim_config);

        // Add noise to init_latent at the start timestep
        let start_timestep = if start_step < sampler.timesteps().len() {
            sampler.timesteps()[start_step]
        } else {
            0
        };

        let noise = Tensor::random(
            init_latent.shape(),
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &self.device,
        );

        // Get alpha for noise addition
        let alpha_t = self.scheduler.alpha_cumprod_at(start_timestep);
        let sqrt_alpha = alpha_t.clone().sqrt();
        let sqrt_one_minus_alpha = (alpha_t.neg() + 1.0).sqrt();

        let mut latent =
            init_latent * sqrt_alpha.unsqueeze() + noise * sqrt_one_minus_alpha.unsqueeze();

        // Precompute timestep tensors
        let timestep_tensors: Vec<Tensor<B, 1>> = sampler
            .timesteps()
            .iter()
            .map(|&t| Tensor::<B, 1>::from_data(TensorData::new(vec![t as f32], [1]), &self.device))
            .collect();

        // Denoising loop from start_step
        for step_idx in start_step..sampler.num_steps() {
            let t = timestep_tensors[step_idx].clone();

            let noise_uncond = self.unet.forward(latent.clone(), t.clone(), uncond.clone());
            let noise_cond = self.unet.forward(latent.clone(), t, cond.clone());
            let noise_pred = apply_guidance(noise_uncond, noise_cond, config.guidance_scale);

            latent = sampler.step(latent, noise_pred, step_idx);
        }

        // Decode (scaling applied by decode_to_image)
        self.vae_decoder.decode_to_image(latent)
    }
}

// ============================================================================
// SD 1.x Inpainting Pipeline
// ============================================================================

/// Inpainting configuration
#[derive(Debug, Clone)]
pub struct InpaintConfig {
    pub steps: usize,
    pub guidance_scale: f64,
    pub seed: Option<u64>,
}

impl Default for InpaintConfig {
    fn default() -> Self {
        Self {
            steps: 50,
            guidance_scale: 7.5,
            seed: None,
        }
    }
}

/// Stable Diffusion 1.x Inpainting Pipeline
///
/// Performs masked image editing - regenerates only the masked regions
/// while preserving unmasked areas.
pub struct StableDiffusion1xInpaint<B: Backend> {
    pub tokenizer: ClipTokenizer,
    pub text_encoder: ClipTextEncoder<B>,
    pub unet: UNet<B>,
    pub vae_encoder: Encoder<B>,
    pub vae_decoder: Decoder<B>,
    pub scheduler: NoiseSchedule<B>,
    device: B::Device,
}

impl<B: Backend> StableDiffusion1xInpaint<B> {
    /// Create a new SD 1.x inpainting pipeline
    pub fn new(tokenizer: ClipTokenizer, device: &B::Device) -> Self {
        let clip_config = ClipConfig::sd1x();
        let unet_config = UNetConfig::sd1x();
        let encoder_config = EncoderConfig::sd();
        let decoder_config = DecoderConfig::sd();

        Self {
            tokenizer,
            text_encoder: ClipTextEncoder::new(&clip_config, device),
            unet: UNet::new(&unet_config, device),
            vae_encoder: Encoder::new(&encoder_config, device),
            vae_decoder: Decoder::new(&decoder_config, device),
            scheduler: NoiseSchedule::sd1x(device),
            device: device.clone(),
        }
    }

    /// Encode a single text prompt to embeddings
    fn encode_text(&self, text: &str) -> Tensor<B, 3> {
        let tokens = self.tokenizer.encode_padded(text, 77);
        let token_tensor: Tensor<B, 1, Int> = Tensor::from_data(
            TensorData::new(tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(), [77]),
            &self.device,
        );
        let token_tensor = token_tensor.unsqueeze::<2>();
        self.text_encoder.forward(token_tensor)
    }

    /// Inpaint masked regions of an image
    ///
    /// # Arguments
    /// * `image` - Input image tensor [1, 3, H, W] with values in [0, 255]
    /// * `mask` - Binary mask tensor [1, 1, H, W] where 1 = regenerate, 0 = preserve
    /// * `prompt` - Text prompt for regenerated regions
    /// * `negative_prompt` - Negative prompt
    /// * `config` - Inpainting configuration
    pub fn inpaint(
        &self,
        image: Tensor<B, 4>,
        mask: Tensor<B, 4>,
        prompt: &str,
        negative_prompt: &str,
        config: &InpaintConfig,
    ) -> Tensor<B, 4> {
        let [_, _, img_h, img_w] = image.dims();

        // Encode prompts
        let cond = self.encode_text(prompt);
        let uncond = self.encode_text(if negative_prompt.is_empty() {
            ""
        } else {
            negative_prompt
        });

        // Encode image to latent
        let init_latent = self.vae_encoder.encode_deterministic(image);

        // Downsample mask to latent size
        let latent_mask = self.downsample_mask(mask, img_h / 8, img_w / 8);

        // Create sampler
        let ddim_config = DdimConfig {
            num_inference_steps: config.steps,
            eta: 0.0,
        };
        let scheduler = NoiseSchedule::sd1x(&self.device);
        let sampler = DdimSampler::new(scheduler, ddim_config);

        // Initialize latent with noise
        let [_, c, h, w] = init_latent.dims();
        let mut latent = sampler.init_latent(1, c, h, w, &self.device);

        // Precompute timestep tensors
        let timestep_tensors: Vec<Tensor<B, 1>> = sampler
            .timesteps()
            .iter()
            .map(|&t| Tensor::<B, 1>::from_data(TensorData::new(vec![t as f32], [1]), &self.device))
            .collect();

        // Inpainting loop
        for step_idx in 0..sampler.num_steps() {
            let timestep = sampler.timesteps()[step_idx];
            let t = timestep_tensors[step_idx].clone();

            // Predict noise
            let noise_uncond = self.unet.forward(latent.clone(), t.clone(), uncond.clone());
            let noise_cond = self.unet.forward(latent.clone(), t.clone(), cond.clone());
            let noise_pred = apply_guidance(noise_uncond, noise_cond, config.guidance_scale);

            // DDIM step for generated regions
            latent = sampler.step(latent, noise_pred, step_idx);

            // Blend: replace unmasked regions with noised original
            let alpha_t = self.scheduler.alpha_cumprod_at(timestep);
            let sqrt_alpha = alpha_t.clone().sqrt();
            let sqrt_one_minus_alpha = (alpha_t.neg() + 1.0).sqrt();

            let noise = Tensor::random(
                init_latent.shape(),
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &self.device,
            );
            let noised_original = init_latent.clone() * sqrt_alpha.unsqueeze()
                + noise * sqrt_one_minus_alpha.unsqueeze();

            // mask = 1 means regenerate (use latent), mask = 0 means preserve (use noised_original)
            latent = latent.clone() * latent_mask.clone()
                + noised_original * (latent_mask.clone().neg() + 1.0);
        }

        // Final blend in latent space (without noise)
        latent = latent.clone() * latent_mask.clone() + init_latent * (latent_mask.neg() + 1.0);

        // Decode (scaling applied by decode_to_image)
        self.vae_decoder.decode_to_image(latent)
    }

    /// Downsample mask from image space to latent space using nearest-neighbor sampling
    fn downsample_mask(
        &self,
        mask: Tensor<B, 4>,
        target_h: usize,
        target_w: usize,
    ) -> Tensor<B, 4> {
        let [b, c, h, w] = mask.dims();

        // Simple nearest-neighbor downsampling via slicing
        // Take every 8th pixel
        let scale_h = h / target_h;
        let scale_w = w / target_w;

        let mut result = Vec::with_capacity(b * c * target_h * target_w);
        let data = mask.into_data();
        let values: Vec<f32> = data.to_vec().unwrap();

        for batch in 0..b {
            for channel in 0..c {
                for th in 0..target_h {
                    for tw in 0..target_w {
                        let src_h = th * scale_h;
                        let src_w = tw * scale_w;
                        let idx = batch * c * h * w + channel * h * w + src_h * w + src_w;
                        result.push(values[idx]);
                    }
                }
            }
        }

        Tensor::from_data(
            TensorData::new(result, [b, c, target_h, target_w]),
            &self.device,
        )
    }
}
