//! Stable Diffusion XL pipeline implementations

use burn::prelude::*;
use burn::tensor::Int;

use sketchpad_clip::{
    ClipConfig, ClipTextEncoder, ClipTokenizer, END_OF_TEXT, OpenClipConfig, OpenClipTextEncoder,
};
use sketchpad_samplers::{DdimConfig, DdimSampler, NoiseSchedule, apply_guidance};
use sketchpad_unet::{UNetXL, UNetXLConfig};
use sketchpad_vae::{Decoder, DecoderConfig, Encoder, EncoderConfig};

use super::compute_size_embedding;

/// SDXL sampling configuration
#[derive(Debug, Clone)]
pub struct SdxlSampleConfig {
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub guidance_scale: f64,
    pub seed: Option<u64>,
}

impl Default for SdxlSampleConfig {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 1024,
            steps: 30,
            guidance_scale: 7.5,
            seed: None,
        }
    }
}

/// SDXL conditioning (dual text embeddings + pooled)
pub struct SdxlConditioning<B: Backend> {
    /// Conditional context [batch, seq_len, 2048]
    pub cond_context: Tensor<B, 3>,
    /// Unconditional context [batch, seq_len, 2048]
    pub uncond_context: Tensor<B, 3>,
    /// Conditional pooled embedding [batch, pooled_dim]
    pub cond_pooled: Tensor<B, 2>,
    /// Unconditional pooled embedding [batch, pooled_dim]
    pub uncond_pooled: Tensor<B, 2>,
}

/// Stable Diffusion XL Pipeline
///
/// Uses dual text encoders (CLIP + OpenCLIP) for conditioning
pub struct StableDiffusionXL<B: Backend> {
    pub tokenizer: ClipTokenizer,
    pub clip_encoder: ClipTextEncoder<B>,
    pub open_clip_encoder: OpenClipTextEncoder<B>,
    pub unet: UNetXL<B>,
    pub vae_decoder: Decoder<B>,
    pub scheduler: NoiseSchedule<B>,
    pub device: B::Device,
}

impl<B: Backend> StableDiffusionXL<B> {
    /// Create a new SDXL pipeline with default configs
    pub fn new(tokenizer: ClipTokenizer, device: &B::Device) -> Self {
        let clip_config = ClipConfig::sd1x(); // SDXL uses same CLIP architecture
        let open_clip_config = OpenClipConfig::sdxl();
        let unet_config = UNetXLConfig::sdxl_base();
        let vae_config = DecoderConfig::sd();

        Self {
            tokenizer,
            clip_encoder: ClipTextEncoder::new(&clip_config, device),
            open_clip_encoder: OpenClipTextEncoder::new(&open_clip_config, device),
            unet: UNetXL::new(&unet_config, device),
            vae_decoder: Decoder::new(&vae_config, device),
            scheduler: NoiseSchedule::sdxl(device),
            device: device.clone(),
        }
    }

    /// Create with custom configs
    pub fn with_configs(
        tokenizer: ClipTokenizer,
        clip_config: &ClipConfig,
        open_clip_config: &OpenClipConfig,
        unet_config: &UNetXLConfig,
        vae_config: &DecoderConfig,
        device: &B::Device,
    ) -> Self {
        Self {
            tokenizer,
            clip_encoder: ClipTextEncoder::new(clip_config, device),
            open_clip_encoder: OpenClipTextEncoder::new(open_clip_config, device),
            unet: UNetXL::new(unet_config, device),
            vae_decoder: Decoder::new(vae_config, device),
            scheduler: NoiseSchedule::sdxl(device),
            device: device.clone(),
        }
    }

    /// Encode text using both CLIP and OpenCLIP encoders, returning concatenated context and pooled embedding
    fn encode_text(&self, text: &str) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let tokens = self.tokenizer.encode_padded(text, 77);
        let token_tensor: Tensor<B, 1, Int> = Tensor::from_data(
            TensorData::new(tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(), [77]),
            &self.device,
        );
        let token_tensor = token_tensor.unsqueeze::<2>(); // [1, 77]

        // CLIP encoder output [1, 77, 768]
        // SDXL always uses penultimate layer (clip_skip=2), NOT final layer.
        // See: https://github.com/Stability-AI/generative-models/issues/37
        // See: docs/architecture.md "SDXL Text Encoder Details"
        let clip_hidden = self.clip_encoder.forward_penultimate(token_tensor.clone());

        // OpenCLIP encoder outputs [1, 77, 1280] and pooled [1, 1280]
        // Must use forward_with_pooled() to apply text_projection for pooled output
        let eos_pos = tokens.iter().position(|&t| t == END_OF_TEXT).unwrap_or(76);
        let (open_clip_hidden, pooled) = self
            .open_clip_encoder
            .forward_with_pooled(token_tensor, &[eos_pos]);

        // Concatenate CLIP and OpenCLIP hidden states [1, 77, 768 + 1280 = 2048]
        let context = Tensor::cat(vec![clip_hidden, open_clip_hidden], 2);

        (context, pooled)
    }

    /// Create add_embed from pooled text embedding
    ///
    /// SDXL add_embed includes:
    /// - Pooled text embedding (1280)
    /// - Original size (height, width) - 256 each
    /// - Crop coords (top, left) - 256 each
    /// - Target size (height, width) - 256 each
    pub fn create_add_embed(
        &self,
        pooled: Tensor<B, 2>,
        original_size: (usize, usize),
        crop_coords: (usize, usize),
        target_size: (usize, usize),
    ) -> Tensor<B, 2> {
        let [_batch, _] = pooled.dims();

        // Time embeddings for size/coord conditioning (each 256 dim)
        let orig_h_emb = self.size_embedding(original_size.0);
        let orig_w_emb = self.size_embedding(original_size.1);
        let crop_t_emb = self.size_embedding(crop_coords.0);
        let crop_l_emb = self.size_embedding(crop_coords.1);
        let target_h_emb = self.size_embedding(target_size.0);
        let target_w_emb = self.size_embedding(target_size.1);

        // Concatenate: pooled (1280) + size embeddings (6 * 256 = 1536) = 2816
        Tensor::cat(
            vec![
                pooled,
                orig_h_emb.unsqueeze::<2>(),
                orig_w_emb.unsqueeze::<2>(),
                crop_t_emb.unsqueeze::<2>(),
                crop_l_emb.unsqueeze::<2>(),
                target_h_emb.unsqueeze::<2>(),
                target_w_emb.unsqueeze::<2>(),
            ],
            1,
        )
    }

    /// Compute a sinusoidal embedding for a size value
    fn size_embedding(&self, value: usize) -> Tensor<B, 1> {
        compute_size_embedding(value, &self.device)
    }

    /// Encode prompt for SDXL
    pub fn encode_prompt(&self, prompt: &str, negative_prompt: &str) -> SdxlConditioning<B> {
        let (cond_context, cond_pooled) = self.encode_text(prompt);
        let (uncond_context, uncond_pooled) = self.encode_text(if negative_prompt.is_empty() {
            ""
        } else {
            negative_prompt
        });

        SdxlConditioning {
            cond_context,
            uncond_context,
            cond_pooled,
            uncond_pooled,
        }
    }

    /// Sample latent from conditioning
    pub fn sample_latent(
        &self,
        conditioning: &SdxlConditioning<B>,
        config: &SdxlSampleConfig,
    ) -> Tensor<B, 4> {
        let latent_height = config.height / 8;
        let latent_width = config.width / 8;

        // Create DDIM sampler
        let ddim_config = DdimConfig {
            num_inference_steps: config.steps,
            eta: 0.0,
        };
        let sampler = DdimSampler::new(NoiseSchedule::sdxl(&self.device), ddim_config);

        // Initialize with random noise
        let mut latent = sampler.init_latent(1, 4, latent_height, latent_width, &self.device);

        // Create add_embed for conditioning
        let cond_add_embed = self.create_add_embed(
            conditioning.cond_pooled.clone(),
            (config.height, config.width),
            (0, 0),
            (config.height, config.width),
        );
        let uncond_add_embed = self.create_add_embed(
            conditioning.uncond_pooled.clone(),
            (config.height, config.width),
            (0, 0),
            (config.height, config.width),
        );

        // Precompute timestep tensors
        let timestep_tensors: Vec<Tensor<B, 1>> = sampler
            .timesteps()
            .iter()
            .map(|&t| Tensor::<B, 1>::from_data(TensorData::new(vec![t as f32], [1]), &self.device))
            .collect();

        // Sampling loop
        for step_idx in 0..sampler.num_steps() {
            let t = timestep_tensors[step_idx].clone();

            // Predict noise for unconditional
            let noise_uncond = self.unet.forward(
                latent.clone(),
                t.clone(),
                conditioning.uncond_context.clone(),
                uncond_add_embed.clone(),
            );

            // Predict noise for conditional
            let noise_cond = self.unet.forward(
                latent.clone(),
                t,
                conditioning.cond_context.clone(),
                cond_add_embed.clone(),
            );

            // Apply classifier-free guidance
            let noise_pred = apply_guidance(noise_uncond, noise_cond, config.guidance_scale);

            // DDIM step
            latent = sampler.step(latent, noise_pred, step_idx);
        }

        latent
    }

    /// Decode latent to image
    pub fn decode(&self, latent: Tensor<B, 4>) -> Tensor<B, 4> {
        // SDXL uses different VAE scaling factor (0.13025 vs 0.18215)
        // decode_to_image_sdxl applies correct scaling and converts to [0, 255]
        self.vae_decoder.decode_to_image_sdxl(latent)
    }

    /// Full generation pipeline
    pub fn generate(
        &self,
        prompt: &str,
        negative_prompt: &str,
        config: &SdxlSampleConfig,
    ) -> Tensor<B, 4> {
        let conditioning = self.encode_prompt(prompt, negative_prompt);
        let latent = self.sample_latent(&conditioning, config);
        self.decode(latent)
    }
}

/// SDXL img2img configuration
#[derive(Debug, Clone)]
pub struct SdxlImg2ImgConfig {
    pub steps: usize,
    pub guidance_scale: f64,
    /// Strength of the transformation (0.0 = no change, 1.0 = full regeneration)
    pub strength: f64,
    pub seed: Option<u64>,
}

impl Default for SdxlImg2ImgConfig {
    fn default() -> Self {
        Self {
            steps: 30,
            guidance_scale: 7.5,
            strength: 0.75,
            seed: None,
        }
    }
}

/// Stable Diffusion XL Img2Img Pipeline
pub struct StableDiffusionXLImg2Img<B: Backend> {
    pub tokenizer: ClipTokenizer,
    pub clip_encoder: ClipTextEncoder<B>,
    pub open_clip_encoder: OpenClipTextEncoder<B>,
    pub unet: UNetXL<B>,
    pub vae_encoder: Encoder<B>,
    pub vae_decoder: Decoder<B>,
    pub scheduler: NoiseSchedule<B>,
    device: B::Device,
}

impl<B: Backend> StableDiffusionXLImg2Img<B> {
    /// Create a new SDXL img2img pipeline
    pub fn new(tokenizer: ClipTokenizer, device: &B::Device) -> Self {
        let clip_config = ClipConfig::sd1x();
        let open_clip_config = OpenClipConfig::sdxl();
        let unet_config = UNetXLConfig::sdxl_base();
        let encoder_config = EncoderConfig::sd();
        let decoder_config = DecoderConfig::sd();

        Self {
            tokenizer,
            clip_encoder: ClipTextEncoder::new(&clip_config, device),
            open_clip_encoder: OpenClipTextEncoder::new(&open_clip_config, device),
            unet: UNetXL::new(&unet_config, device),
            vae_encoder: Encoder::new(&encoder_config, device),
            vae_decoder: Decoder::new(&decoder_config, device),
            scheduler: NoiseSchedule::sdxl(device),
            device: device.clone(),
        }
    }

    /// Encode text using both encoders
    fn encode_text(&self, text: &str) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let tokens = self.tokenizer.encode_padded(text, 77);
        let token_tensor: Tensor<B, 1, Int> = Tensor::from_data(
            TensorData::new(tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(), [77]),
            &self.device,
        );
        let token_tensor = token_tensor.unsqueeze::<2>();

        // CLIP encoder output [1, 77, 768] - SDXL uses penultimate layer (clip_skip=2)
        let clip_hidden = self.clip_encoder.forward_penultimate(token_tensor.clone());
        let eos_pos = tokens.iter().position(|&t| t == END_OF_TEXT).unwrap_or(76);
        let (open_clip_hidden, pooled) = self
            .open_clip_encoder
            .forward_with_pooled(token_tensor, &[eos_pos]);

        let context = Tensor::cat(vec![clip_hidden, open_clip_hidden], 2);
        (context, pooled)
    }

    /// Create add_embed from pooled text embedding
    fn create_add_embed(
        &self,
        pooled: Tensor<B, 2>,
        original_size: (usize, usize),
        crop_coords: (usize, usize),
        target_size: (usize, usize),
    ) -> Tensor<B, 2> {
        let orig_h_emb = self.size_embedding(original_size.0);
        let orig_w_emb = self.size_embedding(original_size.1);
        let crop_t_emb = self.size_embedding(crop_coords.0);
        let crop_l_emb = self.size_embedding(crop_coords.1);
        let target_h_emb = self.size_embedding(target_size.0);
        let target_w_emb = self.size_embedding(target_size.1);

        Tensor::cat(
            vec![
                pooled,
                orig_h_emb.unsqueeze::<2>(),
                orig_w_emb.unsqueeze::<2>(),
                crop_t_emb.unsqueeze::<2>(),
                crop_l_emb.unsqueeze::<2>(),
                target_h_emb.unsqueeze::<2>(),
                target_w_emb.unsqueeze::<2>(),
            ],
            1,
        )
    }

    /// Compute a sinusoidal embedding for a size value
    fn size_embedding(&self, value: usize) -> Tensor<B, 1> {
        compute_size_embedding(value, &self.device)
    }

    /// Generate image from input image and prompt
    pub fn generate(
        &self,
        image: Tensor<B, 4>,
        prompt: &str,
        negative_prompt: &str,
        config: &SdxlImg2ImgConfig,
    ) -> Tensor<B, 4> {
        let [_, _, height, width] = image.dims();

        // Encode prompts
        let (cond_context, cond_pooled) = self.encode_text(prompt);
        let (uncond_context, uncond_pooled) = self.encode_text(if negative_prompt.is_empty() {
            ""
        } else {
            negative_prompt
        });

        // Encode image to latent (SDXL scale factor: 0.13025)
        let init_latent = self.vae_encoder.encode_deterministic(image) * (0.13025 / 0.18215);

        // Calculate start step based on strength
        let num_inference_steps = config.steps;
        let start_step = ((1.0 - config.strength) * num_inference_steps as f64) as usize;

        // Create sampler
        let ddim_config = DdimConfig {
            num_inference_steps,
            eta: 0.0,
        };
        let sampler = DdimSampler::new(NoiseSchedule::sdxl(&self.device), ddim_config);

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

        let alpha_t = self.scheduler.alpha_cumprod_at(start_timestep);
        let sqrt_alpha = alpha_t.clone().sqrt();
        let sqrt_one_minus_alpha = (alpha_t.neg() + 1.0).sqrt();

        let mut latent =
            init_latent * sqrt_alpha.unsqueeze() + noise * sqrt_one_minus_alpha.unsqueeze();

        // Create add_embed
        let cond_add_embed =
            self.create_add_embed(cond_pooled, (height, width), (0, 0), (height, width));
        let uncond_add_embed =
            self.create_add_embed(uncond_pooled, (height, width), (0, 0), (height, width));

        // Precompute timestep tensors
        let timestep_tensors: Vec<Tensor<B, 1>> = sampler
            .timesteps()
            .iter()
            .map(|&t| Tensor::<B, 1>::from_data(TensorData::new(vec![t as f32], [1]), &self.device))
            .collect();

        // Denoising loop
        for step_idx in start_step..sampler.num_steps() {
            let t = timestep_tensors[step_idx].clone();

            let noise_uncond = self.unet.forward(
                latent.clone(),
                t.clone(),
                uncond_context.clone(),
                uncond_add_embed.clone(),
            );
            let noise_cond = self.unet.forward(
                latent.clone(),
                t,
                cond_context.clone(),
                cond_add_embed.clone(),
            );
            let noise_pred = apply_guidance(noise_uncond, noise_cond, config.guidance_scale);

            latent = sampler.step(latent, noise_pred, step_idx);
        }

        // Decode with SDXL scale factor and convert to [0, 255]
        self.vae_decoder.decode_to_image_sdxl(latent)
    }
}

// ============================================================================
// SDXL Inpainting Pipeline
// ============================================================================

/// SDXL Inpainting configuration
#[derive(Debug, Clone)]
pub struct SdxlInpaintConfig {
    pub steps: usize,
    pub guidance_scale: f64,
    pub seed: Option<u64>,
}

impl Default for SdxlInpaintConfig {
    fn default() -> Self {
        Self {
            steps: 30,
            guidance_scale: 7.5,
            seed: None,
        }
    }
}

/// Stable Diffusion XL Inpainting Pipeline
///
/// Performs masked image editing using SDXL - regenerates only the masked regions
/// while preserving unmasked areas.
pub struct StableDiffusionXLInpaint<B: Backend> {
    pub tokenizer: ClipTokenizer,
    pub clip_encoder: ClipTextEncoder<B>,
    pub open_clip_encoder: OpenClipTextEncoder<B>,
    pub unet: UNetXL<B>,
    pub vae_encoder: Encoder<B>,
    pub vae_decoder: Decoder<B>,
    pub scheduler: NoiseSchedule<B>,
    device: B::Device,
}

impl<B: Backend> StableDiffusionXLInpaint<B> {
    /// Create a new SDXL inpainting pipeline
    pub fn new(tokenizer: ClipTokenizer, device: &B::Device) -> Self {
        let clip_config = ClipConfig::sd1x();
        let open_clip_config = OpenClipConfig::sdxl();
        let unet_config = UNetXLConfig::sdxl_base();
        let encoder_config = EncoderConfig::sdxl();
        let decoder_config = DecoderConfig::sdxl();

        Self {
            tokenizer,
            clip_encoder: ClipTextEncoder::new(&clip_config, device),
            open_clip_encoder: OpenClipTextEncoder::new(&open_clip_config, device),
            unet: UNetXL::new(&unet_config, device),
            vae_encoder: Encoder::new(&encoder_config, device),
            vae_decoder: Decoder::new(&decoder_config, device),
            scheduler: NoiseSchedule::sdxl(device),
            device: device.clone(),
        }
    }

    /// Encode text using dual encoders (same as SDXL base)
    fn encode_text(&self, text: &str) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let tokens = self.tokenizer.encode_padded(text, 77);
        let token_tensor: Tensor<B, 1, Int> = Tensor::from_data(
            TensorData::new(tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(), [77]),
            &self.device,
        );
        let token_tensor = token_tensor.unsqueeze::<2>();

        // CLIP hidden states (penultimate layer)
        let clip_hidden = self.clip_encoder.forward_penultimate(token_tensor.clone());

        // OpenCLIP hidden states and pooled
        let eos_pos = tokens.iter().position(|&t| t == END_OF_TEXT).unwrap_or(76);
        let (open_clip_hidden, pooled) = self
            .open_clip_encoder
            .forward_with_pooled(token_tensor, &[eos_pos]);

        // Concatenate hidden states
        let context = Tensor::cat(vec![clip_hidden, open_clip_hidden], 2);

        (context, pooled)
    }

    /// Compute a sinusoidal embedding for a size value
    fn size_embedding(&self, value: usize) -> Tensor<B, 1> {
        compute_size_embedding(value, &self.device)
    }

    /// Create add_embed tensor from pooled embedding and size/crop information
    fn create_add_embed(
        &self,
        pooled: Tensor<B, 2>,
        original_size: (usize, usize),
        crop_coords: (usize, usize),
        target_size: (usize, usize),
    ) -> Tensor<B, 2> {
        let orig_h_emb = self.size_embedding(original_size.0);
        let orig_w_emb = self.size_embedding(original_size.1);
        let crop_t_emb = self.size_embedding(crop_coords.0);
        let crop_l_emb = self.size_embedding(crop_coords.1);
        let target_h_emb = self.size_embedding(target_size.0);
        let target_w_emb = self.size_embedding(target_size.1);

        Tensor::cat(
            vec![
                pooled,
                orig_h_emb.unsqueeze::<2>(),
                orig_w_emb.unsqueeze::<2>(),
                crop_t_emb.unsqueeze::<2>(),
                crop_l_emb.unsqueeze::<2>(),
                target_h_emb.unsqueeze::<2>(),
                target_w_emb.unsqueeze::<2>(),
            ],
            1,
        )
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
        config: &SdxlInpaintConfig,
    ) -> Tensor<B, 4> {
        let [_, _, img_h, img_w] = image.dims();

        // Encode prompts
        let (cond_context, cond_pooled) = self.encode_text(prompt);
        let (uncond_context, uncond_pooled) = self.encode_text(if negative_prompt.is_empty() {
            ""
        } else {
            negative_prompt
        });

        // Create add_embed
        let cond_add_embed =
            self.create_add_embed(cond_pooled, (img_h, img_w), (0, 0), (img_h, img_w));
        let uncond_add_embed =
            self.create_add_embed(uncond_pooled, (img_h, img_w), (0, 0), (img_h, img_w));

        // Encode image to latent
        let init_latent = self.vae_encoder.encode_deterministic_sdxl(image);

        // Downsample mask to latent size
        let latent_mask = self.downsample_mask(mask, img_h / 8, img_w / 8);

        // Create sampler
        let ddim_config = DdimConfig {
            num_inference_steps: config.steps,
            eta: 0.0,
        };
        let scheduler = NoiseSchedule::sdxl(&self.device);
        let sampler = DdimSampler::new(scheduler, ddim_config);

        // For blending, we need the noise schedule
        let blend_scheduler = NoiseSchedule::sdxl(&self.device);

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
            let noise_uncond = self.unet.forward(
                latent.clone(),
                t.clone(),
                uncond_context.clone(),
                uncond_add_embed.clone(),
            );
            let noise_cond = self.unet.forward(
                latent.clone(),
                t.clone(),
                cond_context.clone(),
                cond_add_embed.clone(),
            );
            let noise_pred = apply_guidance(noise_uncond, noise_cond, config.guidance_scale);

            // DDIM step for generated regions
            latent = sampler.step(latent, noise_pred, step_idx);

            // Blend: replace unmasked regions with noised original
            let alpha_t = blend_scheduler.alpha_cumprod_at(timestep);
            let sqrt_alpha = alpha_t.clone().sqrt();
            let sqrt_one_minus_alpha = (alpha_t.neg() + 1.0).sqrt();

            let noise = Tensor::random(
                init_latent.shape(),
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &self.device,
            );
            let noised_original = init_latent.clone() * sqrt_alpha.unsqueeze()
                + noise * sqrt_one_minus_alpha.unsqueeze();

            // mask = 1 means regenerate, mask = 0 means preserve
            latent = latent.clone() * latent_mask.clone()
                + noised_original * (latent_mask.clone().neg() + 1.0);
        }

        // Final blend in latent space (without noise)
        latent = latent.clone() * latent_mask.clone() + init_latent * (latent_mask.neg() + 1.0);

        // Decode with SDXL scaling
        self.vae_decoder.decode_to_image_sdxl(latent)
    }

    /// Downsample mask from image space to latent space
    fn downsample_mask(
        &self,
        mask: Tensor<B, 4>,
        target_h: usize,
        target_w: usize,
    ) -> Tensor<B, 4> {
        let [b, c, h, w] = mask.dims();

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

// ============================================================================
// SDXL Refiner
// ============================================================================

/// SDXL Refiner configuration
#[derive(Debug, Clone)]
pub struct RefinerConfig {
    /// Number of refinement steps
    pub steps: usize,
    /// Guidance scale for the refiner
    pub guidance_scale: f64,
    /// At what fraction of total steps to hand off from base to refiner (0.0-1.0)
    /// E.g., 0.8 means refiner takes over at 80% of the way through denoising
    pub denoise_start: f64,
}

impl Default for RefinerConfig {
    fn default() -> Self {
        Self {
            steps: 20,
            guidance_scale: 7.5,
            denoise_start: 0.8,
        }
    }
}

/// SDXL Refiner Pipeline
///
/// Uses only OpenCLIP text encoder (no CLIP) and a different UNet architecture.
/// Designed to refine the output of the SDXL base model.
pub struct StableDiffusionXLRefiner<B: Backend> {
    pub tokenizer: ClipTokenizer,
    pub text_encoder: OpenClipTextEncoder<B>,
    pub unet: UNetXL<B>,
    pub vae_decoder: Decoder<B>,
    pub scheduler: NoiseSchedule<B>,
    device: B::Device,
}

impl<B: Backend> StableDiffusionXLRefiner<B> {
    /// Create a new SDXL Refiner pipeline
    pub fn new(tokenizer: ClipTokenizer, device: &B::Device) -> Self {
        let open_clip_config = OpenClipConfig::sdxl();
        let unet_config = UNetXLConfig::sdxl_refiner();
        let vae_config = DecoderConfig::sdxl();

        Self {
            tokenizer,
            text_encoder: OpenClipTextEncoder::new(&open_clip_config, device),
            unet: UNetXL::new(&unet_config, device),
            vae_decoder: Decoder::new(&vae_config, device),
            scheduler: NoiseSchedule::sdxl(device),
            device: device.clone(),
        }
    }

    /// Encode text using OpenCLIP only
    fn encode_text(&self, text: &str) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let tokens = self.tokenizer.encode_padded(text, 77);
        let token_tensor: Tensor<B, 1, Int> = Tensor::from_data(
            TensorData::new(tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(), [77]),
            &self.device,
        );
        let token_tensor = token_tensor.unsqueeze::<2>();

        let eos_pos = tokens.iter().position(|&t| t == END_OF_TEXT).unwrap_or(76);
        self.text_encoder
            .forward_with_pooled(token_tensor, &[eos_pos])
    }

    /// Create add_embed for refiner (includes aesthetic score)
    fn create_add_embed(
        &self,
        pooled: Tensor<B, 2>,
        original_size: (usize, usize),
        crop_coords: (usize, usize),
        target_size: (usize, usize),
        aesthetic_score: f64,
    ) -> Tensor<B, 2> {
        let orig_h_emb = self.size_embedding(original_size.0);
        let orig_w_emb = self.size_embedding(original_size.1);
        let crop_t_emb = self.size_embedding(crop_coords.0);
        let crop_l_emb = self.size_embedding(crop_coords.1);
        let target_h_emb = self.size_embedding(target_size.0);
        let target_w_emb = self.size_embedding(target_size.1);
        let aesthetic_emb = self.aesthetic_embedding(aesthetic_score);

        // Refiner add_embed: pooled + sizes + aesthetic = 2560
        Tensor::cat(
            vec![
                pooled,
                orig_h_emb.unsqueeze::<2>(),
                orig_w_emb.unsqueeze::<2>(),
                crop_t_emb.unsqueeze::<2>(),
                crop_l_emb.unsqueeze::<2>(),
                target_h_emb.unsqueeze::<2>(),
                target_w_emb.unsqueeze::<2>(),
                aesthetic_emb.unsqueeze::<2>(),
            ],
            1,
        )
    }

    /// Compute a sinusoidal embedding for a size value
    fn size_embedding(&self, value: usize) -> Tensor<B, 1> {
        compute_size_embedding(value, &self.device)
    }

    /// Compute a sinusoidal embedding for an aesthetic score value
    fn aesthetic_embedding(&self, score: f64) -> Tensor<B, 1> {
        // Aesthetic score embedding (same as size embedding but for score)
        let half_dim = 128;
        let value = score as f32;
        let mut emb = vec![0.0f32; 256];
        for i in 0..half_dim {
            let freq = (-((i as f32) / half_dim as f32) * (10000.0f32).ln()).exp();
            emb[i] = (value * freq).sin();
            emb[i + half_dim] = (value * freq).cos();
        }
        Tensor::from_data(TensorData::new(emb, [256]), &self.device)
    }

    /// Refine a latent from the base model
    ///
    /// Takes a partially denoised latent and continues the denoising process.
    pub fn refine(
        &self,
        latent: Tensor<B, 4>,
        prompt: &str,
        negative_prompt: &str,
        config: &RefinerConfig,
    ) -> Tensor<B, 4> {
        let [_, _, height, width] = latent.dims();
        let image_height = height * 8;
        let image_width = width * 8;

        // Encode prompts
        let (cond_context, cond_pooled) = self.encode_text(prompt);
        let (uncond_context, uncond_pooled) = self.encode_text(if negative_prompt.is_empty() {
            ""
        } else {
            negative_prompt
        });

        // Create add_embed with high aesthetic score
        let aesthetic_score = 6.0; // High aesthetic score for positive
        let neg_aesthetic_score = 2.5; // Low aesthetic score for negative

        let cond_add_embed = self.create_add_embed(
            cond_pooled,
            (image_height, image_width),
            (0, 0),
            (image_height, image_width),
            aesthetic_score,
        );
        let uncond_add_embed = self.create_add_embed(
            uncond_pooled,
            (image_height, image_width),
            (0, 0),
            (image_height, image_width),
            neg_aesthetic_score,
        );

        // Create sampler
        let ddim_config = DdimConfig {
            num_inference_steps: config.steps,
            eta: 0.0,
        };
        let sampler = DdimSampler::new(NoiseSchedule::sdxl(&self.device), ddim_config);

        // Start from denoise_start
        let start_step = ((1.0 - config.denoise_start) * config.steps as f64) as usize;
        let mut latent = latent;

        // Precompute timestep tensors
        let timestep_tensors: Vec<Tensor<B, 1>> = sampler
            .timesteps()
            .iter()
            .map(|&t| Tensor::<B, 1>::from_data(TensorData::new(vec![t as f32], [1]), &self.device))
            .collect();

        // Refinement loop
        for step_idx in start_step..sampler.num_steps() {
            let t = timestep_tensors[step_idx].clone();

            let noise_uncond = self.unet.forward(
                latent.clone(),
                t.clone(),
                uncond_context.clone(),
                uncond_add_embed.clone(),
            );
            let noise_cond = self.unet.forward(
                latent.clone(),
                t,
                cond_context.clone(),
                cond_add_embed.clone(),
            );
            let noise_pred = apply_guidance(noise_uncond, noise_cond, config.guidance_scale);

            latent = sampler.step(latent, noise_pred, step_idx);
        }

        latent
    }

    /// Decode latent to image
    pub fn decode(&self, latent: Tensor<B, 4>) -> Tensor<B, 4> {
        self.vae_decoder.decode_to_image_sdxl(latent)
    }
}

// ============================================================================
// SDXL Base + Refiner Combined Workflow
// ============================================================================

/// Configuration for combined base + refiner workflow
#[derive(Debug, Clone)]
pub struct BaseRefinerConfig {
    /// Image width
    pub width: usize,
    /// Image height
    pub height: usize,
    /// Total number of denoising steps (split between base and refiner)
    pub steps: usize,
    /// Base model guidance scale
    pub base_guidance_scale: f64,
    /// Refiner model guidance scale
    pub refiner_guidance_scale: f64,
    /// Fraction of steps for base model (e.g., 0.8 = base does 80% of steps)
    pub refiner_start: f64,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for BaseRefinerConfig {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 1024,
            steps: 40,
            base_guidance_scale: 7.5,
            refiner_guidance_scale: 7.5,
            refiner_start: 0.8,
            seed: None,
        }
    }
}

/// Combined SDXL Base + Refiner Pipeline
///
/// Runs the base model for initial denoising, then hands off to the refiner
/// for final quality improvements. This is the recommended workflow for
/// highest quality SDXL generation.
pub struct StableDiffusionXLWithRefiner<B: Backend> {
    pub base: StableDiffusionXL<B>,
    pub refiner: StableDiffusionXLRefiner<B>,
    device: B::Device,
}

impl<B: Backend> StableDiffusionXLWithRefiner<B> {
    /// Create a new combined pipeline
    ///
    /// Requires two tokenizers (same vocabulary, different instances) because
    /// the base and refiner pipelines each need their own tokenizer.
    pub fn new(
        base_tokenizer: ClipTokenizer,
        refiner_tokenizer: ClipTokenizer,
        device: &B::Device,
    ) -> Self {
        Self {
            base: StableDiffusionXL::new(base_tokenizer, device),
            refiner: StableDiffusionXLRefiner::new(refiner_tokenizer, device),
            device: device.clone(),
        }
    }

    /// Create from pre-constructed base and refiner pipelines
    pub fn from_pipelines(
        base: StableDiffusionXL<B>,
        refiner: StableDiffusionXLRefiner<B>,
        device: &B::Device,
    ) -> Self {
        Self {
            base,
            refiner,
            device: device.clone(),
        }
    }

    /// Generate image using base + refiner workflow
    pub fn generate(
        &self,
        prompt: &str,
        negative_prompt: &str,
        config: &BaseRefinerConfig,
    ) -> Tensor<B, 4> {
        // Calculate step splits
        let base_steps = ((config.refiner_start) * config.steps as f64) as usize;

        // Run base model for initial denoising
        let base_config = SdxlSampleConfig {
            width: config.width,
            height: config.height,
            steps: config.steps, // Full schedule, but we'll stop early
            guidance_scale: config.base_guidance_scale,
            seed: config.seed,
        };

        let latent = self.sample_base_partial(prompt, negative_prompt, &base_config, base_steps);

        // Run refiner for final quality
        let refiner_config = RefinerConfig {
            steps: config.steps,
            guidance_scale: config.refiner_guidance_scale,
            denoise_start: config.refiner_start,
        };

        let refined = self
            .refiner
            .refine(latent, prompt, negative_prompt, &refiner_config);

        // Decode to image
        self.refiner.decode(refined)
    }

    /// Run base model and stop after specified number of steps
    fn sample_base_partial(
        &self,
        prompt: &str,
        negative_prompt: &str,
        config: &SdxlSampleConfig,
        stop_at_step: usize,
    ) -> Tensor<B, 4> {
        let conditioning = self.base.encode_prompt(prompt, negative_prompt);

        let latent_height = config.height / 8;
        let latent_width = config.width / 8;

        // Create DDIM sampler with full schedule
        let ddim_config = DdimConfig {
            num_inference_steps: config.steps,
            eta: 0.0,
        };
        let sampler = DdimSampler::new(NoiseSchedule::sdxl(&self.device), ddim_config);

        // Initialize with random noise
        let mut latent = sampler.init_latent(1, 4, latent_height, latent_width, &self.device);

        // Create add_embed for conditioning
        let cond_add_embed = self.base.create_add_embed(
            conditioning.cond_pooled.clone(),
            (config.height, config.width),
            (0, 0),
            (config.height, config.width),
        );
        let uncond_add_embed = self.base.create_add_embed(
            conditioning.uncond_pooled.clone(),
            (config.height, config.width),
            (0, 0),
            (config.height, config.width),
        );

        // Precompute timestep tensors
        let timestep_tensors: Vec<Tensor<B, 1>> = sampler
            .timesteps()
            .iter()
            .map(|&t| Tensor::<B, 1>::from_data(TensorData::new(vec![t as f32], [1]), &self.device))
            .collect();

        // Sampling loop - stop at specified step
        for step_idx in 0..stop_at_step {
            let t = timestep_tensors[step_idx].clone();

            // Predict noise for unconditional
            let noise_uncond = self.base.unet.forward(
                latent.clone(),
                t.clone(),
                conditioning.uncond_context.clone(),
                uncond_add_embed.clone(),
            );

            // Predict noise for conditional
            let noise_cond = self.base.unet.forward(
                latent.clone(),
                t,
                conditioning.cond_context.clone(),
                cond_add_embed.clone(),
            );

            // Apply classifier-free guidance
            let noise_pred = apply_guidance(noise_uncond, noise_cond, config.guidance_scale);

            // DDIM step
            latent = sampler.step(latent, noise_pred, step_idx);
        }

        latent
    }

    /// Generate with separate prompts for base and refiner
    ///
    /// Useful when you want different prompts for initial generation vs refinement
    pub fn generate_with_prompts(
        &self,
        base_prompt: &str,
        base_negative: &str,
        refiner_prompt: &str,
        refiner_negative: &str,
        config: &BaseRefinerConfig,
    ) -> Tensor<B, 4> {
        let base_steps = ((config.refiner_start) * config.steps as f64) as usize;

        let base_config = SdxlSampleConfig {
            width: config.width,
            height: config.height,
            steps: config.steps,
            guidance_scale: config.base_guidance_scale,
            seed: config.seed,
        };

        let latent = self.sample_base_partial(base_prompt, base_negative, &base_config, base_steps);

        let refiner_config = RefinerConfig {
            steps: config.steps,
            guidance_scale: config.refiner_guidance_scale,
            denoise_start: config.refiner_start,
        };

        let refined =
            self.refiner
                .refine(latent, refiner_prompt, refiner_negative, &refiner_config);
        self.refiner.decode(refined)
    }
}
