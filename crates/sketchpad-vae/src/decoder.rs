//! VAE Decoder: latent -> image
//!
//! Decodes 4-channel latent representations to 3-channel RGB images.

use burn::nn::{
    PaddingConfig2d,
    conv::{Conv2d, Conv2dConfig},
};
use burn::prelude::*;

use sketchpad_core::groupnorm::GroupNorm;
use sketchpad_core::silu::silu;

/// VAE Decoder configuration
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    /// Input latent channels (typically 4)
    pub latent_channels: usize,
    /// Output image channels (typically 3 for RGB)
    pub out_channels: usize,
    /// Base channel multiplier
    pub base_channels: usize,
    /// Channel multipliers for each block
    pub channel_mult: Vec<usize>,
    /// Number of resnet blocks per decoder block
    pub num_res_blocks: usize,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            latent_channels: 4,
            out_channels: 3,
            base_channels: 128,
            channel_mult: vec![1, 2, 4, 4],
            num_res_blocks: 3, // SD 1.x VAE has 3 res blocks per up block
        }
    }
}

impl DecoderConfig {
    /// SD 1.x VAE decoder config
    pub fn sd1x() -> Self {
        Self::default()
    }

    /// SD 1.x VAE decoder config (alias)
    pub fn sd() -> Self {
        Self::sd1x()
    }

    /// SDXL VAE decoder config (same architecture, different scaling)
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

/// VAE Decoder
#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    /// Post-quantization conv applied to latent before decoding (optional, 1x1 conv)
    pub post_quant_conv: Option<Conv2d<B>>,
    pub conv_in: Conv2d<B>,
    pub mid_block1: ResnetBlock<B>,
    pub mid_attn: SelfAttention<B>,
    pub mid_block2: ResnetBlock<B>,
    pub up_blocks: Vec<DecoderBlock<B>>,
    pub norm_out: GroupNorm<B>,
    pub conv_out: Conv2d<B>,
    /// Skip attention in mid-block to reduce VRAM (minimal quality impact)
    pub skip_attention: bool,
    /// Enable debug output
    pub debug: bool,
    /// Enable aggressive clamping to prevent f16 overflow
    /// Clamps activations after each operation to stay within f16 range.
    pub clamp_overflow: bool,
}

impl<B: Backend> Decoder<B> {
    /// Creates a new VAE decoder
    ///
    /// # Arguments
    ///
    /// * `config` - Decoder configuration
    /// * `device` - Device to create tensors on
    pub fn new(config: &DecoderConfig, device: &B::Device) -> Self {
        let ch = config.base_channels;
        let ch_mult = &config.channel_mult;

        // Start with highest channel count (reversed from encoder)
        let block_in = ch * ch_mult[ch_mult.len() - 1];

        // Input conv: latent_channels -> block_in
        let conv_in = Conv2dConfig::new([config.latent_channels, block_in], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        // Mid blocks
        let mid_block1 = ResnetBlock::new(block_in, block_in, device);
        let mid_attn = SelfAttention::new(block_in, device);
        let mid_block2 = ResnetBlock::new(block_in, block_in, device);

        // Up blocks (reverse order, with upsampling)
        let mut up_blocks = Vec::new();
        let mut in_ch = block_in;

        for (i, &mult) in ch_mult.iter().rev().enumerate() {
            let out_ch = ch * mult;
            let upsample = i < ch_mult.len() - 1; // Don't upsample on last block

            up_blocks.push(DecoderBlock::new(
                in_ch,
                out_ch,
                config.num_res_blocks,
                upsample,
                device,
            ));
            in_ch = out_ch;
        }

        // Output layers
        let norm_out = GroupNorm::new(32, in_ch, device);
        let conv_out = Conv2dConfig::new([in_ch, config.out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        Self {
            post_quant_conv: None,
            conv_in,
            mid_block1,
            mid_attn,
            mid_block2,
            up_blocks,
            norm_out,
            conv_out,
            skip_attention: false,
            debug: false,
            clamp_overflow: false,
        }
    }

    /// Set whether to skip mid-block attention (reduces VRAM usage)
    pub fn with_skip_attention(mut self, skip: bool) -> Self {
        self.skip_attention = skip;
        self
    }

    /// Enable debug output
    pub fn with_debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }

    /// Enable aggressive clamping to prevent f16 overflow
    pub fn with_clamp_overflow(mut self, clamp: bool) -> Self {
        self.clamp_overflow = clamp;
        self
    }

    /// Clamp tensor to prevent f16 overflow (when clamp_overflow is enabled)
    fn clamp_if_enabled(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        if self.clamp_overflow {
            x.clamp(-65000.0, 65000.0)
        } else {
            x
        }
    }

    /// Decode latent to image (raw, no scaling applied)
    ///
    /// Input: [batch, 4, h, w] latent (already unscaled)
    /// Output: [batch, 3, h*8, w*8] image (values in [-1, 1])
    pub fn forward_raw(&self, z: Tensor<B, 4>) -> Tensor<B, 4> {
        if self.debug {
            eprintln!("[vae] forward_raw input: {:?}", z.dims());
            Self::check_nan("input_latent", &z);

            // Check input stats
            let in_data: Vec<f32> = z.clone().into_data().convert::<f32>().to_vec().unwrap();
            let in_min = in_data.iter().cloned().fold(f32::INFINITY, f32::min);
            let in_max = in_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "[vae] input latent stats: min={:.3}, max={:.3}",
                in_min, in_max
            );
        }

        // Apply post_quant_conv if present (transforms latent before decoding)
        let z = match &self.post_quant_conv {
            Some(conv) => {
                if self.debug {
                    eprintln!("[vae] applying post_quant_conv");
                }
                let out = conv.forward(z);
                if self.debug {
                    Self::check_nan("post_quant_conv", &out);
                }
                out
            }
            None => z,
        };

        // Input conv
        if self.debug {
            eprintln!("[vae] conv_in...");
        }
        let mut h = self.conv_in.forward(z);
        h = self.clamp_if_enabled(h);
        if self.debug {
            Self::check_nan("conv_in", &h);
        }

        // Mid blocks
        if self.debug {
            eprintln!("[vae] mid_block1...");
        }
        h = self.mid_block1.forward(h);
        h = self.clamp_if_enabled(h);
        if self.debug {
            Self::check_nan("mid_block1", &h);
        }

        if !self.skip_attention {
            if self.debug {
                eprintln!("[vae] mid_attn...");
            }
            h = self.mid_attn.forward(h);
            h = self.clamp_if_enabled(h);
            if self.debug {
                Self::check_nan("mid_attn", &h);
            }
        } else if self.debug {
            eprintln!("[vae] skipping mid_attn");
        }

        if self.debug {
            eprintln!("[vae] mid_block2...");
        }
        h = self.mid_block2.forward(h);
        h = self.clamp_if_enabled(h);
        if self.debug {
            Self::check_nan("mid_block2", &h);
        }

        // Up blocks
        for (i, block) in self.up_blocks.iter().enumerate() {
            if self.debug {
                eprintln!("[vae] up_block {}...", i);
            }
            h = block.forward(h);
            h = self.clamp_if_enabled(h);
            if self.debug {
                Self::check_nan(&format!("up_block_{}", i), &h);
            }
        }

        // Output
        if self.debug {
            eprintln!("[vae] norm_out...");
        }
        h = self.norm_out.forward(h);
        h = self.clamp_if_enabled(h);
        if self.debug {
            Self::check_nan("norm_out", &h);
        }

        if self.debug {
            eprintln!("[vae] silu...");
        }
        h = silu(h);
        h = self.clamp_if_enabled(h);
        if self.debug {
            Self::check_nan("silu", &h);
        }

        if self.debug {
            eprintln!("[vae] conv_out...");
        }
        let out = self.conv_out.forward(h);
        if self.debug {
            Self::check_nan("conv_out", &out);
        }
        out
    }

    fn check_nan(name: &str, t: &Tensor<B, 4>) {
        let data: Vec<f32> = t.clone().into_data().convert::<f32>().to_vec().unwrap();
        let nan_count = data.iter().filter(|x| x.is_nan()).count();
        let inf_count = data.iter().filter(|x| x.is_infinite()).count();
        if nan_count > 0 || inf_count > 0 {
            eprintln!(
                "[vae] WARNING: {} has {} NaN, {} Inf out of {} values",
                name,
                nan_count,
                inf_count,
                data.len()
            );
        }
    }

    /// Decode latent to image with SD 1.x scaling
    ///
    /// Input: [batch, 4, h, w] latent
    /// Output: [batch, 3, h*8, w*8] image (values in [-1, 1])
    pub fn forward(&self, z: Tensor<B, 4>) -> Tensor<B, 4> {
        let z = z / scaling::SD1X;
        self.forward_raw(z)
    }

    /// Decode latent to image with custom scaling factor
    pub fn forward_scaled(&self, z: Tensor<B, 4>, scale: f64) -> Tensor<B, 4> {
        let z = z / scale;
        self.forward_raw(z)
    }

    /// Decode latent with SDXL scaling
    pub fn forward_sdxl(&self, z: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward_scaled(z, scaling::SDXL)
    }

    /// Decode and convert to [0, 255] range (SD 1.x scaling)
    pub fn decode_to_image(&self, z: Tensor<B, 4>) -> Tensor<B, 4> {
        let img = self.forward(z);
        // Clamp to [-1, 1] range and convert to [0, 255]
        let img = img.clamp(-1.0, 1.0);
        (img + 1.0) * 127.5
    }

    /// Decode and convert to [0, 255] range (SDXL scaling)
    pub fn decode_to_image_sdxl(&self, z: Tensor<B, 4>) -> Tensor<B, 4> {
        let img = self.forward_sdxl(z);
        (img + 1.0) * 127.5
    }

    /// Decode latent using simple 2x2 quadrant tiling to reduce VRAM usage
    ///
    /// Splits the 128x128 latent into 4 quadrants of 64x64 each,
    /// decodes separately, and stitches together.
    pub fn forward_tiled(
        &self,
        z: Tensor<B, 4>,
        _tile_size: usize,
        _overlap: usize,
    ) -> Tensor<B, 4> {
        let [batch, channels, height, width] = z.dims();
        assert_eq!(batch, 1, "Tiled decode only supports batch size 1");

        // For 128x128 latent, split into 2x2 grid of 64x64 tiles
        let half_h = height / 2;
        let half_w = width / 2;

        if self.debug {
            eprintln!(
                "[vae] Tiled decode: {}x{} latent -> 2x2 grid of {}x{} tiles",
                height, width, half_h, half_w
            );
        }

        // Decode each quadrant
        if self.debug {
            eprintln!("[vae] Decoding quadrant 0 (top-left)...");
        }
        let q0 = z.clone().slice([0..1, 0..channels, 0..half_h, 0..half_w]);
        let d0 = self.forward_raw(q0);
        if self.debug {
            Self::debug_tensor_stats("quadrant 0", &d0);
        }

        if self.debug {
            eprintln!("[vae] Decoding quadrant 1 (top-right)...");
        }
        let q1 = z
            .clone()
            .slice([0..1, 0..channels, 0..half_h, half_w..width]);
        let d1 = self.forward_raw(q1);
        if self.debug {
            Self::debug_tensor_stats("quadrant 1", &d1);
        }

        if self.debug {
            eprintln!("[vae] Decoding quadrant 2 (bottom-left)...");
        }
        let q2 = z
            .clone()
            .slice([0..1, 0..channels, half_h..height, 0..half_w]);
        let d2 = self.forward_raw(q2);
        if self.debug {
            Self::debug_tensor_stats("quadrant 2", &d2);
        }

        if self.debug {
            eprintln!("[vae] Decoding quadrant 3 (bottom-right)...");
        }
        let q3 = z
            .clone()
            .slice([0..1, 0..channels, half_h..height, half_w..width]);
        let d3 = self.forward_raw(q3);
        if self.debug {
            Self::debug_tensor_stats("quadrant 3", &d3);
        }

        // Stitch together: concat horizontally then vertically
        if self.debug {
            eprintln!("[vae] Stitching quadrants...");
        }
        let top_row = Tensor::cat(vec![d0, d1], 3); // concat along width
        let bottom_row = Tensor::cat(vec![d2, d3], 3); // concat along width
        let result = Tensor::cat(vec![top_row, bottom_row], 2); // concat along height

        if self.debug {
            Self::debug_tensor_stats("final stitched", &result);
        }
        result
    }

    fn debug_tensor_stats(name: &str, t: &Tensor<B, 4>) {
        let data: Vec<f32> = t.clone().into_data().convert::<f32>().to_vec().unwrap();
        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let nan_count = data.iter().filter(|x| x.is_nan()).count();
        eprintln!(
            "[vae] {} stats: min={:.3}, max={:.3}, nan={}",
            name, min, max, nan_count
        );
    }

    /// Decode with tiled processing and SDXL scaling
    pub fn forward_tiled_sdxl(
        &self,
        z: Tensor<B, 4>,
        tile_size: usize,
        overlap: usize,
    ) -> Tensor<B, 4> {
        let z = z / scaling::SDXL;
        self.forward_tiled(z, tile_size, overlap)
    }

    /// Decode to image with tiled processing (SDXL)
    pub fn decode_to_image_tiled_sdxl(
        &self,
        z: Tensor<B, 4>,
        tile_size: usize,
        overlap: usize,
    ) -> Tensor<B, 4> {
        let img = self.forward_tiled_sdxl(z, tile_size, overlap);
        (img + 1.0) * 127.5
    }
}

/// Decoder block with optional upsampling
#[derive(Module, Debug)]
pub struct DecoderBlock<B: Backend> {
    pub res_blocks: Vec<ResnetBlock<B>>,
    pub upsample: Option<Upsample<B>>,
}

impl<B: Backend> DecoderBlock<B> {
    /// Creates a new decoder block with optional upsampling
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        num_blocks: usize,
        upsample: bool,
        device: &B::Device,
    ) -> Self {
        let mut res_blocks = Vec::new();

        // First block handles channel change
        res_blocks.push(ResnetBlock::new(in_channels, out_channels, device));

        // Remaining blocks maintain channels
        for _ in 1..num_blocks {
            res_blocks.push(ResnetBlock::new(out_channels, out_channels, device));
        }

        let upsample = if upsample {
            Some(Upsample::new(out_channels, device))
        } else {
            None
        };

        Self {
            res_blocks,
            upsample,
        }
    }

    /// Forward pass through residual blocks and optional upsampling
    pub fn forward(&self, mut x: Tensor<B, 4>) -> Tensor<B, 4> {
        for block in &self.res_blocks {
            x = block.forward(x);
        }

        if let Some(up) = &self.upsample {
            x = up.forward(x);
        }

        x
    }
}

/// Resnet block with skip connection
#[derive(Module, Debug)]
pub struct ResnetBlock<B: Backend> {
    pub norm1: GroupNorm<B>,
    pub conv1: Conv2d<B>,
    pub norm2: GroupNorm<B>,
    pub conv2: Conv2d<B>,
    pub skip_conv: Option<Conv2d<B>>,
}

impl<B: Backend> ResnetBlock<B> {
    /// Creates a new resnet block with optional skip connection
    pub fn new(in_channels: usize, out_channels: usize, device: &B::Device) -> Self {
        let norm1 = GroupNorm::new(32, in_channels, device);
        let conv1 = Conv2dConfig::new([in_channels, out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        let norm2 = GroupNorm::new(32, out_channels, device);
        let conv2 = Conv2dConfig::new([out_channels, out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        let skip_conv = if in_channels != out_channels {
            Some(Conv2dConfig::new([in_channels, out_channels], [1, 1]).init(device))
        } else {
            None
        };

        Self {
            norm1,
            conv1,
            norm2,
            conv2,
            skip_conv,
        }
    }

    /// Forward pass with residual connection
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = match &self.skip_conv {
            Some(conv) => conv.forward(x.clone()),
            None => x.clone(),
        };

        let h = self.norm1.forward(x);
        let h = silu(h);
        let h = self.conv1.forward(h);

        let h = self.norm2.forward(h);
        let h = silu(h);
        let h = self.conv2.forward(h);

        h + residual
    }
}

/// 2x Upsampling with conv
#[derive(Module, Debug)]
pub struct Upsample<B: Backend> {
    pub conv: Conv2d<B>,
}

impl<B: Backend> Upsample<B> {
    /// Creates a new 2x upsampling layer
    pub fn new(channels: usize, device: &B::Device) -> Self {
        let conv = Conv2dConfig::new([channels, channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        Self { conv }
    }

    /// Forward pass with nearest neighbor upsampling and convolution
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        use burn::tensor::module::interpolate;
        use burn::tensor::ops::{InterpolateMode, InterpolateOptions};

        let [_b, _c, h, w] = x.dims();

        // Use interpolate for memory-efficient nearest neighbor upsampling
        let x = interpolate(
            x,
            [h * 2, w * 2],
            InterpolateOptions::new(InterpolateMode::Nearest),
        );

        self.conv.forward(x)
    }
}

/// Self-attention for VAE mid block
#[derive(Module, Debug)]
pub struct SelfAttention<B: Backend> {
    pub norm: GroupNorm<B>,
    pub q: Conv2d<B>,
    pub k: Conv2d<B>,
    pub v: Conv2d<B>,
    pub proj_out: Conv2d<B>,
    pub channels: usize,
}

impl<B: Backend> SelfAttention<B> {
    /// Creates a new self-attention layer
    pub fn new(channels: usize, device: &B::Device) -> Self {
        let norm = GroupNorm::new(32, channels, device);

        let q = Conv2dConfig::new([channels, channels], [1, 1]).init(device);
        let k = Conv2dConfig::new([channels, channels], [1, 1]).init(device);
        let v = Conv2dConfig::new([channels, channels], [1, 1]).init(device);
        let proj_out = Conv2dConfig::new([channels, channels], [1, 1]).init(device);

        Self {
            norm,
            q,
            k,
            v,
            proj_out,
            channels,
        }
    }

    /// Forward pass computing scaled dot-product self-attention
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, c, h, w] = x.dims();
        let residual = x.clone();

        let x = self.norm.forward(x);

        let q = self.q.forward(x.clone());
        let k = self.k.forward(x.clone());
        let v = self.v.forward(x);

        // Reshape for attention: [b, c, h, w] -> [b, c, h*w] -> [b, h*w, c]
        let q = q.reshape([b, c, h * w]).swap_dims(1, 2);
        let k = k.reshape([b, c, h * w]); // [b, c, h*w]
        let v = v.reshape([b, c, h * w]).swap_dims(1, 2); // [b, h*w, c]

        // Attention: softmax(Q @ K^T / sqrt(d)) @ V
        // Use stable softmax (max-subtraction) for f16 compatibility
        let scale = (c as f64).powf(-0.5);
        let attn = q.matmul(k) * scale; // [b, h*w, h*w]

        // Stable softmax to prevent f16 overflow
        let attn_max = attn.clone().max_dim(2);
        let attn = (attn - attn_max).exp();
        let attn = attn.clone() / attn.clone().sum_dim(2);

        let out = attn.matmul(v); // [b, h*w, c]

        // Reshape back: [b, h*w, c] -> [b, c, h, w]
        let out = out.swap_dims(1, 2).reshape([b, c, h, w]);

        let out = self.proj_out.forward(out);

        out + residual
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_config() {
        let config = DecoderConfig::sd();
        assert_eq!(config.latent_channels, 4);
        assert_eq!(config.out_channels, 3);
    }
}
