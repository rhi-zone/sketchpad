//! SD 1.x UNet
//!
//! The diffusion backbone for Stable Diffusion 1.x models.

use burn::nn::{
    Linear, LinearConfig, PaddingConfig2d,
    conv::{Conv2d, Conv2dConfig},
};
use burn::prelude::*;

use sketchpad_core::groupnorm::GroupNorm;
use sketchpad_core::silu::silu;

use crate::blocks::{
    Downsample, ResBlock, SpatialTransformer, Upsample, timestep_embedding_with_freqs,
    timestep_freqs,
};

/// UNet configuration for SD 1.x
#[derive(Debug, Clone)]
pub struct UNetConfig {
    /// Input channels (latent channels, typically 4)
    pub in_channels: usize,
    /// Output channels (same as input)
    pub out_channels: usize,
    /// Base model channels
    pub model_channels: usize,
    /// Channel multipliers per resolution level
    pub channel_mult: Vec<usize>,
    /// Number of attention heads
    pub num_heads: usize,
    /// Attention head dimension
    pub head_dim: usize,
    /// Context dimension (text embedding dim)
    pub context_dim: usize,
    /// Number of transformer blocks per spatial transformer
    pub transformer_depth: usize,
}

impl UNetConfig {
    /// Configuration for SD 1.x
    pub fn sd1x() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            model_channels: 320,
            channel_mult: vec![1, 2, 4, 4],
            num_heads: 8,
            head_dim: 40,
            context_dim: 768, // CLIP embedding dim
            transformer_depth: 1,
        }
    }

    /// Configuration for SD 2.x
    pub fn sd2x() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            model_channels: 320,
            channel_mult: vec![1, 2, 4, 4],
            num_heads: 8,
            head_dim: 80,
            context_dim: 1024, // OpenCLIP embedding dim
            transformer_depth: 1,
        }
    }
}

/// SD 1.x UNet
#[derive(Module, Debug)]
pub struct UNet<B: Backend> {
    /// Time embedding first linear layer
    pub time_embed_0: Linear<B>,
    /// Time embedding second linear layer
    pub time_embed_2: Linear<B>,
    /// Precomputed timestep embedding frequencies
    pub time_freqs: Tensor<B, 1>,

    /// Input convolution
    pub conv_in: Conv2d<B>,

    /// Down blocks (encoder path)
    pub down_blocks: Vec<DownBlock<B>>,

    /// Middle block
    pub mid_block: MidBlock<B>,

    /// Up blocks (decoder path)
    pub up_blocks: Vec<UpBlock<B>>,

    /// Output group normalization
    pub norm_out: GroupNorm<B>,
    /// Output convolution
    pub conv_out: Conv2d<B>,

    /// Base model channels
    pub model_channels: usize,
}

impl<B: Backend> UNet<B> {
    /// Creates a new SD 1.x UNet
    ///
    /// # Arguments
    ///
    /// * `config` - UNet configuration
    /// * `device` - Device to create tensors on
    pub fn new(config: &UNetConfig, device: &B::Device) -> Self {
        let ch = config.model_channels;
        let time_embed_dim = ch * 4;

        // Time embedding MLP
        let time_embed_0 = LinearConfig::new(ch, time_embed_dim).init(device);
        let time_embed_2 = LinearConfig::new(time_embed_dim, time_embed_dim).init(device);

        // Input conv
        let conv_in = Conv2dConfig::new([config.in_channels, ch], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        // Build down blocks
        let mut down_blocks = Vec::new();
        let mut channels = vec![ch]; // Track channel sizes for skip connections
        let mut ch_in = ch;

        for (level, &mult) in config.channel_mult.iter().enumerate() {
            let ch_out = ch * mult;
            let is_last = level == config.channel_mult.len() - 1;
            // SD 1.x: levels 0,1,2 have attention, level 3 does not
            let has_attention = level < 3;

            down_blocks.push(DownBlock::new(
                ch_in,
                ch_out,
                time_embed_dim,
                config.num_heads,
                config.head_dim,
                config.context_dim,
                config.transformer_depth,
                has_attention,
                !is_last, // downsample except last
                device,
            ));

            channels.push(ch_out);
            channels.push(ch_out);
            if !is_last {
                channels.push(ch_out);
            }
            ch_in = ch_out;
        }

        // Mid block
        let mid_block = MidBlock::new(
            ch_in,
            time_embed_dim,
            config.num_heads,
            config.head_dim,
            config.context_dim,
            config.transformer_depth,
            device,
        );

        // Build up blocks (reverse order)
        let mut up_blocks = Vec::new();

        for (level, &mult) in config.channel_mult.iter().rev().enumerate() {
            let ch_out = ch * mult;
            let is_last = level == config.channel_mult.len() - 1;
            // Reversed: level 0 here = original level 3 (no attention)
            // level 1,2,3 here = original levels 2,1,0 (have attention)
            let has_attention = level > 0;

            // Each up block has 3 res blocks that take skip connections
            for i in 0..3 {
                let skip_ch = channels.pop().unwrap();
                let block_in = ch_in + skip_ch;
                let block_out = if i == 2 && !is_last {
                    ch * config.channel_mult[config.channel_mult.len() - 2 - level]
                } else {
                    ch_out
                };

                let upsample = i == 2 && !is_last;

                up_blocks.push(UpBlock::new(
                    block_in,
                    block_out,
                    time_embed_dim,
                    config.num_heads,
                    config.head_dim,
                    config.context_dim,
                    config.transformer_depth,
                    has_attention,
                    upsample,
                    device,
                ));

                ch_in = block_out;
            }
        }

        // Output
        let norm_out = GroupNorm::new(32, ch, device);
        let conv_out = Conv2dConfig::new([ch, config.out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        // Precompute timestep embedding frequencies
        let time_freqs = timestep_freqs(ch, device);

        Self {
            time_embed_0,
            time_embed_2,
            time_freqs,
            conv_in,
            down_blocks,
            mid_block,
            up_blocks,
            norm_out,
            conv_out,
            model_channels: ch,
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Noisy latent [batch, 4, h, w]
    /// * `timesteps` - Timestep for each sample [batch]
    /// * `context` - Text embeddings [batch, seq_len, context_dim]
    ///
    /// # Returns
    /// Predicted noise [batch, 4, h, w]
    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        timesteps: Tensor<B, 1>,
        context: Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        // Time embedding (uses precomputed frequencies)
        let t_emb = timestep_embedding_with_freqs(timesteps, self.time_freqs.clone());
        let t_emb = self.time_embed_0.forward(t_emb);
        let t_emb = silu(t_emb);
        let t_emb = self.time_embed_2.forward(t_emb);

        // Input
        let mut h = self.conv_in.forward(x);

        // Down blocks with skip connections
        let mut skips = vec![h.clone()];
        for block in &self.down_blocks {
            let (out, block_skips) = block.forward(h, t_emb.clone(), context.clone());
            h = out;
            skips.extend(block_skips);
        }

        // Mid block
        h = self.mid_block.forward(h, t_emb.clone(), context.clone());

        // Up blocks with skip connections
        for block in &self.up_blocks {
            let skip = skips.pop().unwrap();
            h = Tensor::cat(vec![h, skip], 1);
            h = block.forward(h, t_emb.clone(), context.clone());
        }

        // Output
        h = self.norm_out.forward(h);
        h = silu(h);
        self.conv_out.forward(h)
    }
}

/// Down block: ResBlock + optional SpatialTransformer + optional Downsample
#[derive(Module, Debug)]
pub struct DownBlock<B: Backend> {
    /// First residual block
    pub res1: ResBlock<B>,
    /// First attention block (None for levels without attention, e.g. SD 1.x level 3)
    pub attn1: Option<SpatialTransformer<B>>,
    /// Second residual block
    pub res2: ResBlock<B>,
    /// Second attention block (None for levels without attention)
    pub attn2: Option<SpatialTransformer<B>>,
    /// Optional downsampling layer
    pub downsample: Option<Downsample<B>>,
}

impl<B: Backend> DownBlock<B> {
    /// Creates a new down block with residual and optional attention layers
    #[allow(clippy::too_many_arguments)]
    fn new(
        in_ch: usize,
        out_ch: usize,
        time_dim: usize,
        num_heads: usize,
        head_dim: usize,
        context_dim: usize,
        transformer_depth: usize,
        has_attention: bool,
        downsample: bool,
        device: &B::Device,
    ) -> Self {
        Self {
            res1: ResBlock::new(in_ch, out_ch, time_dim, device),
            attn1: if has_attention {
                Some(SpatialTransformer::new(
                    out_ch,
                    num_heads,
                    head_dim,
                    context_dim,
                    transformer_depth,
                    device,
                ))
            } else {
                None
            },
            res2: ResBlock::new(out_ch, out_ch, time_dim, device),
            attn2: if has_attention {
                Some(SpatialTransformer::new(
                    out_ch,
                    num_heads,
                    head_dim,
                    context_dim,
                    transformer_depth,
                    device,
                ))
            } else {
                None
            },
            downsample: if downsample {
                Some(Downsample::new(out_ch, device))
            } else {
                None
            },
        }
    }

    /// Forward pass, returns output and skip connections for up blocks
    fn forward(
        &self,
        x: Tensor<B, 4>,
        t_emb: Tensor<B, 2>,
        context: Tensor<B, 3>,
    ) -> (Tensor<B, 4>, Vec<Tensor<B, 4>>) {
        let mut skips = Vec::new();

        let h = self.res1.forward(x, t_emb.clone());
        let h = if let Some(attn) = &self.attn1 {
            attn.forward(h, context.clone())
        } else {
            h
        };
        skips.push(h.clone());

        let h = self.res2.forward(h, t_emb);
        let h = if let Some(attn) = &self.attn2 {
            attn.forward(h, context)
        } else {
            h
        };
        skips.push(h.clone());

        let h = if let Some(ds) = &self.downsample {
            let h = ds.forward(h);
            skips.push(h.clone());
            h
        } else {
            h
        };

        (h, skips)
    }
}

/// Mid block: ResBlock + Attention + ResBlock
#[derive(Module, Debug)]
pub struct MidBlock<B: Backend> {
    /// First residual block
    pub res1: ResBlock<B>,
    /// Attention block
    pub attn: SpatialTransformer<B>,
    /// Second residual block
    pub res2: ResBlock<B>,
}

impl<B: Backend> MidBlock<B> {
    /// Creates a new mid block with ResBlock-Attention-ResBlock structure
    fn new(
        channels: usize,
        time_dim: usize,
        num_heads: usize,
        head_dim: usize,
        context_dim: usize,
        transformer_depth: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            res1: ResBlock::new(channels, channels, time_dim, device),
            attn: SpatialTransformer::new(
                channels,
                num_heads,
                head_dim,
                context_dim,
                transformer_depth,
                device,
            ),
            res2: ResBlock::new(channels, channels, time_dim, device),
        }
    }

    /// Forward pass through the mid block
    fn forward(&self, x: Tensor<B, 4>, t_emb: Tensor<B, 2>, context: Tensor<B, 3>) -> Tensor<B, 4> {
        let h = self.res1.forward(x, t_emb.clone());
        let h = self.attn.forward(h, context);
        self.res2.forward(h, t_emb)
    }
}

/// Up block with residual, optional attention, and optional upsampling
#[derive(Module, Debug)]
pub struct UpBlock<B: Backend> {
    /// Residual block
    pub res: ResBlock<B>,
    /// Attention block (None for levels without attention, e.g. SD 1.x level 3)
    pub attn: Option<SpatialTransformer<B>>,
    /// Optional upsampling layer
    pub upsample: Option<Upsample<B>>,
}

impl<B: Backend> UpBlock<B> {
    /// Creates a new up block
    #[allow(clippy::too_many_arguments)]
    fn new(
        in_ch: usize,
        out_ch: usize,
        time_dim: usize,
        num_heads: usize,
        head_dim: usize,
        context_dim: usize,
        transformer_depth: usize,
        has_attention: bool,
        upsample: bool,
        device: &B::Device,
    ) -> Self {
        Self {
            res: ResBlock::new(in_ch, out_ch, time_dim, device),
            attn: if has_attention {
                Some(SpatialTransformer::new(
                    out_ch,
                    num_heads,
                    head_dim,
                    context_dim,
                    transformer_depth,
                    device,
                ))
            } else {
                None
            },
            upsample: if upsample {
                Some(Upsample::new(out_ch, device))
            } else {
                None
            },
        }
    }

    /// Forward pass through the up block
    fn forward(&self, x: Tensor<B, 4>, t_emb: Tensor<B, 2>, context: Tensor<B, 3>) -> Tensor<B, 4> {
        let h = self.res.forward(x, t_emb);
        let h = if let Some(attn) = &self.attn {
            attn.forward(h, context)
        } else {
            h
        };

        if let Some(up) = &self.upsample {
            up.forward(h)
        } else {
            h
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unet_config() {
        let config = UNetConfig::sd1x();
        assert_eq!(config.in_channels, 4);
        assert_eq!(config.context_dim, 768);
    }
}
