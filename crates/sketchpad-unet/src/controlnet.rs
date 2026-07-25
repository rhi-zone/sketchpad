//! ControlNet implementation for conditional image generation
//!
//! ControlNet adds spatial conditioning to diffusion models via additional
//! input images (edge maps, depth maps, poses, etc.)

use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::prelude::*;

use crate::blocks::{Downsample, ResBlock, SpatialTransformer, timestep_embedding};
use sketchpad_core::silu::silu;

/// Zero convolution - initialized to zero for stable training
#[derive(Module, Debug)]
pub struct ZeroConv<B: Backend> {
    conv: Conv2d<B>,
}

impl<B: Backend> ZeroConv<B> {
    /// Create a new zero convolution
    pub fn new(in_channels: usize, out_channels: usize, device: &B::Device) -> Self {
        // In practice, weights should be initialized to zero
        // For inference, the loaded weights will have the trained values
        let conv = Conv2dConfig::new([in_channels, out_channels], [1, 1])
            .with_bias(true)
            .init(device);
        Self { conv }
    }

    /// Forward pass through the zero convolution
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.conv.forward(x)
    }
}

/// ControlNet configuration
#[derive(Debug, Clone)]
pub struct ControlNetConfig {
    /// Input image channels (usually 3 for RGB hint images)
    pub hint_channels: usize,
    /// Model channels
    pub model_channels: usize,
    /// Channel multipliers for each level
    pub channel_mult: Vec<usize>,
    /// Number of res blocks per level
    pub num_res_blocks: usize,
    /// Attention resolutions (which levels have attention)
    pub attention_resolutions: Vec<usize>,
    /// Number of attention heads
    pub num_heads: usize,
    /// Context dimension (from text encoder)
    pub context_dim: usize,
    /// Whether this is for SDXL
    pub is_sdxl: bool,
}

impl Default for ControlNetConfig {
    fn default() -> Self {
        Self {
            hint_channels: 3,
            model_channels: 320,
            channel_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
            attention_resolutions: vec![4, 2, 1],
            num_heads: 8,
            context_dim: 768,
            is_sdxl: false,
        }
    }
}

impl ControlNetConfig {
    /// Configuration for SD 1.x ControlNet
    pub fn sd1x() -> Self {
        Self::default()
    }

    /// Configuration for SDXL ControlNet
    pub fn sdxl() -> Self {
        Self {
            hint_channels: 3,
            model_channels: 320,
            channel_mult: vec![1, 2, 4],
            num_res_blocks: 2,
            attention_resolutions: vec![4, 2],
            num_heads: 8,
            context_dim: 2048,
            is_sdxl: true,
        }
    }
}

/// ControlNet encoder block with zero convolution output
#[derive(Module, Debug)]
pub struct ControlNetEncoderBlock<B: Backend> {
    res_blocks: Vec<ResBlock<B>>,
    attention: Option<SpatialTransformer<B>>,
    zero_conv: ZeroConv<B>,
}

impl<B: Backend> ControlNetEncoderBlock<B> {
    /// Creates a new encoder block with res blocks, optional attention, and zero conv output
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        time_emb_dim: usize,
        num_res_blocks: usize,
        has_attention: bool,
        num_heads: usize,
        context_dim: usize,
        device: &B::Device,
    ) -> Self {
        let mut res_blocks = Vec::new();

        // First res block may change channels
        res_blocks.push(ResBlock::new(
            in_channels,
            out_channels,
            time_emb_dim,
            device,
        ));

        // Remaining res blocks maintain channels
        for _ in 1..num_res_blocks {
            res_blocks.push(ResBlock::new(
                out_channels,
                out_channels,
                time_emb_dim,
                device,
            ));
        }

        let attention = if has_attention {
            let head_dim = out_channels / num_heads;
            Some(SpatialTransformer::new(
                out_channels,
                num_heads,
                head_dim,
                context_dim,
                1, // depth
                device,
            ))
        } else {
            None
        };

        let zero_conv = ZeroConv::new(out_channels, out_channels, device);

        Self {
            res_blocks,
            attention,
            zero_conv,
        }
    }

    /// Forward pass returning hidden state and control output
    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        time_emb: Tensor<B, 2>,
        context: Tensor<B, 3>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let mut hidden = x;

        for res_block in &self.res_blocks {
            hidden = res_block.forward(hidden, time_emb.clone());
        }

        if let Some(attention) = &self.attention {
            hidden = attention.forward(hidden, context);
        }

        let output = self.zero_conv.forward(hidden.clone());

        (hidden, output)
    }
}

/// ControlNet model
#[derive(Module, Debug)]
pub struct ControlNet<B: Backend> {
    /// Hint image encoder (input conditioning image)
    hint_conv: Conv2d<B>,
    hint_block_0: Conv2d<B>,
    hint_block_1: Conv2d<B>,
    hint_block_2: Conv2d<B>,
    hint_block_3: Conv2d<B>,

    /// Time embedding projection
    time_embed_0: burn::nn::Linear<B>,
    time_embed_2: burn::nn::Linear<B>,

    /// Input convolution
    input_conv: Conv2d<B>,
    input_zero_conv: ZeroConv<B>,

    /// Encoder blocks
    encoder_blocks: Vec<ControlNetEncoderBlock<B>>,

    /// Downsampler blocks
    downsamplers: Vec<Option<Downsample<B>>>,
    downsample_zero_convs: Vec<Option<ZeroConv<B>>>,

    /// Middle block
    mid_block_1: ResBlock<B>,
    mid_attn: SpatialTransformer<B>,
    mid_block_2: ResBlock<B>,
    mid_zero_conv: ZeroConv<B>,

    /// Model channels (stored for forward pass)
    model_channels: usize,
    /// Number of res blocks per level
    num_res_blocks: usize,
    /// Channel multipliers
    num_levels: usize,
}

impl<B: Backend> ControlNet<B> {
    /// Create a new ControlNet
    pub fn new(config: ControlNetConfig, device: &B::Device) -> Self {
        let time_embed_dim = config.model_channels * 4;

        // Hint image encoder - progressive encoding
        let hint_conv = Conv2dConfig::new([config.hint_channels, 16], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        // 16 -> 32
        let hint_block_0 = Conv2dConfig::new([16, 32], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_stride([2, 2])
            .init(device);
        // 32 -> 64
        let hint_block_1 = Conv2dConfig::new([32, 64], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_stride([2, 2])
            .init(device);
        // 64 -> 128
        let hint_block_2 = Conv2dConfig::new([64, 128], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_stride([2, 2])
            .init(device);
        // 128 -> model_channels
        let hint_block_3 = Conv2dConfig::new([128, config.model_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        // Time embedding MLP
        let time_embed_0 =
            burn::nn::LinearConfig::new(config.model_channels, time_embed_dim).init(device);
        let time_embed_2 = burn::nn::LinearConfig::new(time_embed_dim, time_embed_dim).init(device);

        // Input convolution
        let input_conv = Conv2dConfig::new([4, config.model_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let input_zero_conv = ZeroConv::new(config.model_channels, config.model_channels, device);

        // Build encoder blocks
        let mut encoder_blocks = Vec::new();
        let mut downsamplers = Vec::new();
        let mut downsample_zero_convs = Vec::new();

        let mut in_channels = config.model_channels;
        let num_levels = config.channel_mult.len();

        for (level, &mult) in config.channel_mult.iter().enumerate() {
            let out_channels = config.model_channels * mult;
            let has_attention = config.attention_resolutions.contains(&(1 << level));

            // Add res blocks at this level
            for i in 0..config.num_res_blocks {
                let block_in = if i == 0 { in_channels } else { out_channels };
                encoder_blocks.push(ControlNetEncoderBlock::new(
                    block_in,
                    out_channels,
                    time_embed_dim,
                    1,
                    has_attention,
                    config.num_heads,
                    config.context_dim,
                    device,
                ));
            }

            in_channels = out_channels;

            // Add downsampler (except for last level)
            if level < config.channel_mult.len() - 1 {
                downsamplers.push(Some(Downsample::new(out_channels, device)));
                downsample_zero_convs.push(Some(ZeroConv::new(out_channels, out_channels, device)));
            } else {
                downsamplers.push(None);
                downsample_zero_convs.push(None);
            }
        }

        // Middle block
        let mid_channels = config.model_channels * config.channel_mult.last().unwrap_or(&1);
        let mid_block_1 = ResBlock::new(mid_channels, mid_channels, time_embed_dim, device);
        let mid_head_dim = mid_channels / config.num_heads;
        let mid_attn = SpatialTransformer::new(
            mid_channels,
            config.num_heads,
            mid_head_dim,
            config.context_dim,
            1,
            device,
        );
        let mid_block_2 = ResBlock::new(mid_channels, mid_channels, time_embed_dim, device);
        let mid_zero_conv = ZeroConv::new(mid_channels, mid_channels, device);

        Self {
            hint_conv,
            hint_block_0,
            hint_block_1,
            hint_block_2,
            hint_block_3,
            time_embed_0,
            time_embed_2,
            input_conv,
            input_zero_conv,
            encoder_blocks,
            downsamplers,
            downsample_zero_convs,
            mid_block_1,
            mid_attn,
            mid_block_2,
            mid_zero_conv,
            model_channels: config.model_channels,
            num_res_blocks: config.num_res_blocks,
            num_levels,
        }
    }

    /// Encode the hint image
    fn encode_hint(&self, hint: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = silu(self.hint_conv.forward(hint));
        let x = silu(self.hint_block_0.forward(x));
        let x = silu(self.hint_block_1.forward(x));
        let x = silu(self.hint_block_2.forward(x));
        silu(self.hint_block_3.forward(x))
    }

    /// Forward pass
    ///
    /// Returns control signals to be added to UNet decoder at each level.
    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        timesteps: Tensor<B, 1>,
        context: Tensor<B, 3>,
        hint: Tensor<B, 4>,
    ) -> ControlNetOutput<B> {
        let device = x.device();

        // Encode timesteps
        let t_emb = timestep_embedding(timesteps, self.model_channels, &device);
        let t_emb = silu(self.time_embed_0.forward(t_emb));
        let t_emb = self.time_embed_2.forward(t_emb);

        // Encode hint and add to input
        let hint_encoded = self.encode_hint(hint);
        let x = self.input_conv.forward(x);
        let x = x + hint_encoded;

        // Collect control outputs
        let mut control_outputs = Vec::new();

        // Input control
        control_outputs.push(self.input_zero_conv.forward(x.clone()));

        // Process through encoder
        let mut hidden = x;
        let mut block_idx = 0;
        let blocks_per_level = self.num_res_blocks;

        for level in 0..self.num_levels {
            // Process res blocks at this level
            for _ in 0..blocks_per_level {
                let (new_hidden, control) =
                    self.encoder_blocks[block_idx].forward(hidden, t_emb.clone(), context.clone());
                hidden = new_hidden;
                control_outputs.push(control);
                block_idx += 1;
            }

            // Downsample
            if let Some(ref downsampler) = self.downsamplers[level] {
                hidden = downsampler.forward(hidden);
                if let Some(ref zero_conv) = self.downsample_zero_convs[level] {
                    control_outputs.push(zero_conv.forward(hidden.clone()));
                }
            }
        }

        // Middle block
        hidden = self.mid_block_1.forward(hidden, t_emb.clone());
        hidden = self.mid_attn.forward(hidden, context);
        hidden = self.mid_block_2.forward(hidden, t_emb);

        let mid_control = self.mid_zero_conv.forward(hidden);
        control_outputs.push(mid_control);

        ControlNetOutput {
            controls: control_outputs,
        }
    }
}

/// Output from ControlNet containing control signals for each UNet level
#[derive(Debug)]
pub struct ControlNetOutput<B: Backend> {
    /// Control signals for each level (from input to middle block)
    pub controls: Vec<Tensor<B, 4>>,
}

impl<B: Backend> ControlNetOutput<B> {
    /// Scale all control signals by a factor
    pub fn scale(self, factor: f32) -> Self {
        Self {
            controls: self.controls.into_iter().map(|c| c * factor).collect(),
        }
    }

    /// Combine with another ControlNet output (for multi-ControlNet)
    pub fn merge(self, other: Self) -> Self {
        Self {
            controls: self
                .controls
                .into_iter()
                .zip(other.controls)
                .map(|(a, b)| a + b)
                .collect(),
        }
    }
}

/// Preprocessing types for ControlNet inputs
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ControlNetPreprocessor {
    /// Canny edge detection
    Canny,
    /// Depth estimation
    Depth,
    /// Normal map
    Normal,
    /// OpenPose keypoints
    Pose,
    /// Semantic segmentation
    Segmentation,
    /// Scribble / sketch
    Scribble,
    /// Soft edge (HED/PidiNet)
    SoftEdge,
    /// Line art
    LineArt,
    /// No preprocessing (direct image)
    None,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_controlnet_config_default() {
        let config = ControlNetConfig::default();
        assert_eq!(config.model_channels, 320);
        assert_eq!(config.hint_channels, 3);
    }

    #[test]
    fn test_controlnet_config_sdxl() {
        let config = ControlNetConfig::sdxl();
        assert_eq!(config.context_dim, 2048);
        assert!(config.is_sdxl);
    }
}
