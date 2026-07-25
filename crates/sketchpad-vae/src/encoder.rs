//! VAE Encoder: image -> latent (for img2img)
//!
//! Encodes RGB images to 4-channel latent representations.

use burn::nn::{
    PaddingConfig2d,
    conv::{Conv2d, Conv2dConfig},
};
use burn::prelude::*;

use sketchpad_core::groupnorm::GroupNorm;
use sketchpad_core::silu::silu;

use crate::decoder::{ResnetBlock, SelfAttention};

/// VAE Encoder configuration
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Input image channels (typically 3 for RGB)
    pub in_channels: usize,
    /// Output latent channels (typically 8, then split for mean/var)
    pub latent_channels: usize,
    /// Base channel multiplier
    pub base_channels: usize,
    /// Channel multipliers for each block
    pub channel_mult: Vec<usize>,
    /// Number of resnet blocks per encoder block
    pub num_res_blocks: usize,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            in_channels: 3,
            latent_channels: 8, // 4 for mean, 4 for logvar
            base_channels: 128,
            channel_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
        }
    }
}

impl EncoderConfig {
    /// SD 1.x VAE encoder config
    pub fn sd1x() -> Self {
        Self::default()
    }

    /// SD 1.x VAE encoder config (alias)
    pub fn sd() -> Self {
        Self::sd1x()
    }

    /// SDXL VAE encoder config (same architecture, different scaling)
    pub fn sdxl() -> Self {
        Self::default()
    }
}

/// VAE scaling factors for different model versions
pub mod scaling {
    /// SD 1.x / SD 2.x scaling factor
    pub const SD1X: f64 = 0.18215;
    /// SDXL scaling factor
    pub const SDXL: f64 = 0.13025;
}

/// VAE Encoder
#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    pub conv_in: Conv2d<B>,
    pub down_blocks: Vec<EncoderBlock<B>>,
    pub mid_block1: ResnetBlock<B>,
    pub mid_attn: SelfAttention<B>,
    pub mid_block2: ResnetBlock<B>,
    pub norm_out: GroupNorm<B>,
    pub conv_out: Conv2d<B>,
}

impl<B: Backend> Encoder<B> {
    /// Creates a new VAE encoder
    ///
    /// # Arguments
    ///
    /// * `config` - Encoder configuration
    /// * `device` - Device to create tensors on
    pub fn new(config: &EncoderConfig, device: &B::Device) -> Self {
        let ch = config.base_channels;
        let ch_mult = &config.channel_mult;

        // Input conv: in_channels -> base_channels
        let conv_in = Conv2dConfig::new([config.in_channels, ch], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        // Down blocks
        let mut down_blocks = Vec::new();
        let mut in_ch = ch;

        for (i, &mult) in ch_mult.iter().enumerate() {
            let out_ch = ch * mult;
            let downsample = i < ch_mult.len() - 1; // Don't downsample on last block

            down_blocks.push(EncoderBlock::new(
                in_ch,
                out_ch,
                config.num_res_blocks,
                downsample,
                device,
            ));
            in_ch = out_ch;
        }

        // Mid blocks
        let mid_ch = ch * ch_mult[ch_mult.len() - 1];
        let mid_block1 = ResnetBlock::new(mid_ch, mid_ch, device);
        let mid_attn = SelfAttention::new(mid_ch, device);
        let mid_block2 = ResnetBlock::new(mid_ch, mid_ch, device);

        // Output layers
        let norm_out = GroupNorm::new(32, mid_ch, device);
        let conv_out = Conv2dConfig::new([mid_ch, config.latent_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        Self {
            conv_in,
            down_blocks,
            mid_block1,
            mid_attn,
            mid_block2,
            norm_out,
            conv_out,
        }
    }

    /// Encode image to latent distribution parameters
    ///
    /// Input: [batch, 3, h, w] image (values in [-1, 1])
    /// Output: [batch, 8, h/8, w/8] (mean and logvar concatenated)
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut h = self.conv_in.forward(x);

        // Down blocks
        for block in &self.down_blocks {
            h = block.forward(h);
        }

        // Mid blocks
        h = self.mid_block1.forward(h);
        h = self.mid_attn.forward(h);
        h = self.mid_block2.forward(h);

        // Output
        h = self.norm_out.forward(h);
        h = silu(h);
        self.conv_out.forward(h)
    }

    /// Encode image and sample latent using reparameterization trick (SD 1.x scaling)
    ///
    /// Input: [batch, 3, h, w] image (values in [0, 255])
    /// Output: [batch, 4, h/8, w/8] latent
    pub fn encode(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        self.encode_scaled(image, scaling::SD1X)
    }

    /// Encode with custom scaling factor
    pub fn encode_scaled(&self, image: Tensor<B, 4>, scale: f64) -> Tensor<B, 4> {
        // Normalize to [-1, 1]
        let x = image / 127.5 - 1.0;

        // Get mean and logvar
        let moments = self.forward(x);
        let [b, c, h, w] = moments.dims();
        let half_c = c / 2;

        let mean = moments.clone().slice([0..b, 0..half_c, 0..h, 0..w]);
        let logvar = moments.slice([0..b, half_c..c, 0..h, 0..w]);

        // Clamp logvar for stability
        let logvar = logvar.clamp(-30.0, 20.0);
        let std = (logvar * 0.5).exp();

        // Sample: z = mean + std * noise
        let noise = Tensor::random(
            mean.shape(),
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &mean.device(),
        );

        let latent = mean + std * noise;

        // Apply scaling factor
        latent * scale
    }

    /// Encode with SDXL scaling
    pub fn encode_sdxl(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        self.encode_scaled(image, scaling::SDXL)
    }

    /// Encode without sampling (just return mean, SD 1.x scaling)
    pub fn encode_deterministic(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        self.encode_deterministic_scaled(image, scaling::SD1X)
    }

    /// Encode without sampling with custom scaling
    pub fn encode_deterministic_scaled(&self, image: Tensor<B, 4>, scale: f64) -> Tensor<B, 4> {
        let x = image / 127.5 - 1.0;
        let moments = self.forward(x);
        let [b, c, h, w] = moments.dims();
        let half_c = c / 2;

        let mean = moments.slice([0..b, 0..half_c, 0..h, 0..w]);
        mean * scale
    }

    /// Encode without sampling with SDXL scaling
    pub fn encode_deterministic_sdxl(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        self.encode_deterministic_scaled(image, scaling::SDXL)
    }
}

/// Encoder block with optional downsampling
#[derive(Module, Debug)]
pub struct EncoderBlock<B: Backend> {
    pub res_blocks: Vec<ResnetBlock<B>>,
    pub downsample: Option<Downsample<B>>,
}

impl<B: Backend> EncoderBlock<B> {
    /// Creates a new encoder block with optional downsampling
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        num_blocks: usize,
        downsample: bool,
        device: &B::Device,
    ) -> Self {
        let mut res_blocks = Vec::new();

        // First block handles channel change
        res_blocks.push(ResnetBlock::new(in_channels, out_channels, device));

        // Remaining blocks maintain channels
        for _ in 1..num_blocks {
            res_blocks.push(ResnetBlock::new(out_channels, out_channels, device));
        }

        let downsample = if downsample {
            Some(Downsample::new(out_channels, device))
        } else {
            None
        };

        Self {
            res_blocks,
            downsample,
        }
    }

    /// Forward pass through residual blocks and optional downsampling
    pub fn forward(&self, mut x: Tensor<B, 4>) -> Tensor<B, 4> {
        for block in &self.res_blocks {
            x = block.forward(x);
        }

        if let Some(ds) = &self.downsample {
            x = ds.forward(x);
        }

        x
    }
}

/// 2x Downsampling with strided conv
#[derive(Module, Debug)]
pub struct Downsample<B: Backend> {
    pub conv: Conv2d<B>,
}

impl<B: Backend> Downsample<B> {
    /// Creates a new 2x downsampling layer with strided convolution
    pub fn new(channels: usize, device: &B::Device) -> Self {
        // Asymmetric padding for even dimensions
        let conv = Conv2dConfig::new([channels, channels], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(0, 1))
            .init(device);

        Self { conv }
    }

    /// Forward pass performing 2x spatial downsampling
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // Use strided conv with asymmetric padding for 2x downsampling
        self.conv.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_config() {
        let config = EncoderConfig::sd();
        assert_eq!(config.in_channels, 3);
        assert_eq!(config.latent_channels, 8);
    }
}
