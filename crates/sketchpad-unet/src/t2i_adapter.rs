//! T2I-Adapter implementation for structural conditioning
//!
//! T2I-Adapter is a lightweight adapter that provides structural guidance
//! (edges, depth, pose, etc.) to the diffusion process.

use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::prelude::*;

use sketchpad_core::silu::silu;

/// T2I-Adapter configuration
#[derive(Debug, Clone)]
pub struct T2IAdapterConfig {
    /// Input channels (3 for RGB condition images)
    pub in_channels: usize,
    /// Base channel count
    pub channels: Vec<usize>,
    /// Number of residual blocks per level
    pub num_res_blocks: usize,
    /// Downsampling factor
    pub downscale_factor: usize,
    /// Adapter type
    pub adapter_type: T2IAdapterType,
}

/// Types of T2I-Adapter
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum T2IAdapterType {
    /// Full adapter (all UNet levels)
    Full,
    /// Light adapter (fewer parameters)
    Light,
    /// Style adapter (for style transfer)
    Style,
}

impl Default for T2IAdapterConfig {
    fn default() -> Self {
        Self {
            in_channels: 3,
            channels: vec![320, 640, 1280, 1280],
            num_res_blocks: 2,
            downscale_factor: 8,
            adapter_type: T2IAdapterType::Full,
        }
    }
}

impl T2IAdapterConfig {
    /// Configuration for SD 1.x
    pub fn sd1x() -> Self {
        Self::default()
    }

    /// Configuration for SDXL
    pub fn sdxl() -> Self {
        Self {
            in_channels: 3,
            channels: vec![320, 640, 1280],
            num_res_blocks: 2,
            downscale_factor: 8,
            adapter_type: T2IAdapterType::Full,
        }
    }

    /// Light adapter configuration
    pub fn light() -> Self {
        Self {
            in_channels: 3,
            channels: vec![320, 640, 1280, 1280],
            num_res_blocks: 1,
            downscale_factor: 8,
            adapter_type: T2IAdapterType::Light,
        }
    }
}

/// T2I-Adapter residual block
#[derive(Module, Debug)]
pub struct T2IAdapterBlock<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    skip: Option<Conv2d<B>>,
}

impl<B: Backend> T2IAdapterBlock<B> {
    /// Creates a new adapter residual block
    pub fn new(in_channels: usize, out_channels: usize, device: &B::Device) -> Self {
        let conv1 = Conv2dConfig::new([in_channels, out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        let conv2 = Conv2dConfig::new([out_channels, out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        let skip = if in_channels != out_channels {
            Some(Conv2dConfig::new([in_channels, out_channels], [1, 1]).init(device))
        } else {
            None
        };

        Self { conv1, conv2, skip }
    }

    /// Forward pass with residual connection
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = match &self.skip {
            Some(conv) => conv.forward(x.clone()),
            None => x.clone(),
        };

        let h = silu(self.conv1.forward(x));
        let h = self.conv2.forward(h);

        h + residual
    }
}

/// T2I-Adapter encoder level
#[derive(Module, Debug)]
pub struct T2IAdapterLevel<B: Backend> {
    blocks: Vec<T2IAdapterBlock<B>>,
    downsample: Option<Conv2d<B>>,
}

impl<B: Backend> T2IAdapterLevel<B> {
    /// Creates a new adapter encoder level
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        num_blocks: usize,
        downsample: bool,
        device: &B::Device,
    ) -> Self {
        let mut blocks = Vec::new();

        // First block may change channels
        blocks.push(T2IAdapterBlock::new(in_channels, out_channels, device));

        // Remaining blocks maintain channels
        for _ in 1..num_blocks {
            blocks.push(T2IAdapterBlock::new(out_channels, out_channels, device));
        }

        let downsample = if downsample {
            Some(
                Conv2dConfig::new([out_channels, out_channels], [3, 3])
                    .with_stride([2, 2])
                    .with_padding(PaddingConfig2d::Explicit(1, 1))
                    .init(device),
            )
        } else {
            None
        };

        Self { blocks, downsample }
    }

    /// Forward pass returning both downsampled and level outputs
    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let mut h = x;

        for block in &self.blocks {
            h = block.forward(h);
        }

        let output = h.clone();

        let h = if let Some(ref ds) = self.downsample {
            ds.forward(h)
        } else {
            h
        };

        (h, output)
    }
}

/// T2I-Adapter model
#[derive(Module, Debug)]
pub struct T2IAdapter<B: Backend> {
    /// Initial pixel unshuffle / conv
    pixel_unshuffle: Conv2d<B>,
    /// Encoder levels
    levels: Vec<T2IAdapterLevel<B>>,
    /// Output projections (one per level)
    out_convs: Vec<Conv2d<B>>,
    /// Global scale
    scale: f32,
}

impl<B: Backend> T2IAdapter<B> {
    /// Create a new T2I-Adapter
    pub fn new(config: T2IAdapterConfig, device: &B::Device) -> Self {
        // Initial convolution (with pixel unshuffle effect)
        let unshuffle_channels =
            config.in_channels * config.downscale_factor * config.downscale_factor;
        let first_channels = config.channels[0];

        let pixel_unshuffle =
            Conv2dConfig::new([unshuffle_channels, first_channels], [1, 1]).init(device);

        // Build levels
        let mut levels = Vec::new();
        let mut in_ch = first_channels;

        for (i, &out_ch) in config.channels.iter().enumerate() {
            let downsample = i < config.channels.len() - 1;
            levels.push(T2IAdapterLevel::new(
                in_ch,
                out_ch,
                config.num_res_blocks,
                downsample,
                device,
            ));
            in_ch = out_ch;
        }

        // Output projections
        let out_convs = config
            .channels
            .iter()
            .map(|&ch| Conv2dConfig::new([ch, ch], [1, 1]).init(device))
            .collect();

        Self {
            pixel_unshuffle,
            levels,
            out_convs,
            scale: 1.0,
        }
    }

    /// Set the scale factor
    pub fn set_scale(&mut self, scale: f32) {
        self.scale = scale;
    }

    /// Forward pass
    ///
    /// Input: Condition image [batch, 3, height, width]
    /// Output: Features for each UNet level
    pub fn forward(&self, x: Tensor<B, 4>) -> T2IAdapterOutput<B> {
        let [batch, channels, height, width] = x.dims();

        // Pixel unshuffle (rearrange spatial to channel dimension)
        // This is a simplified version - full implementation would use proper pixel unshuffle
        let factor = 8usize; // Assuming downscale_factor = 8
        let new_h = height / factor;
        let new_w = width / factor;
        let new_c = channels * factor * factor;

        // Reshape to simulate pixel unshuffle
        let x = x.reshape([batch, channels, new_h, factor, new_w, factor]);
        let x = x.swap_dims(2, 3).swap_dims(4, 5);
        let x = x.reshape([batch, new_c, new_h, new_w]);

        let x = self.pixel_unshuffle.forward(x);

        // Process through levels
        let mut features = Vec::new();
        let mut h = x;

        for (level, out_conv) in self.levels.iter().zip(self.out_convs.iter()) {
            let (next_h, level_out) = level.forward(h);
            let projected = out_conv.forward(level_out);
            features.push(projected * self.scale);
            h = next_h;
        }

        T2IAdapterOutput { features }
    }
}

/// Output from T2I-Adapter containing features for each UNet level
#[derive(Debug)]
pub struct T2IAdapterOutput<B: Backend> {
    /// Features for each level (from shallow to deep)
    pub features: Vec<Tensor<B, 4>>,
}

impl<B: Backend> T2IAdapterOutput<B> {
    /// Scale all features by a factor
    pub fn scale(self, factor: f32) -> Self {
        Self {
            features: self.features.into_iter().map(|f| f * factor).collect(),
        }
    }

    /// Get feature at level index
    pub fn get(&self, level: usize) -> Option<&Tensor<B, 4>> {
        self.features.get(level)
    }

    /// Number of levels
    pub fn num_levels(&self) -> usize {
        self.features.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_t2i_adapter_config_default() {
        let config = T2IAdapterConfig::default();
        assert_eq!(config.in_channels, 3);
        assert_eq!(config.channels.len(), 4);
    }

    #[test]
    fn test_t2i_adapter_config_sdxl() {
        let config = T2IAdapterConfig::sdxl();
        assert_eq!(config.channels.len(), 3);
    }

    #[test]
    fn test_t2i_adapter_type() {
        let config = T2IAdapterConfig::light();
        assert_eq!(config.adapter_type, T2IAdapterType::Light);
        assert_eq!(config.num_res_blocks, 1);
    }
}
