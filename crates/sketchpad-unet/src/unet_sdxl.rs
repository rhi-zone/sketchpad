//! SDXL UNet (larger architecture)
//!
//! The diffusion backbone for Stable Diffusion XL models.
//! Key differences from SD 1.x:
//! - Larger context dimension (2048 for dual text encoders)
//! - Variable transformer depths per resolution
//! - Additional embedding for pooled text + time conditioning

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

/// SDXL UNet configuration
#[derive(Debug, Clone)]
pub struct UNetXLConfig {
    /// Input channels (latent channels, typically 4)
    pub in_channels: usize,
    /// Output channels (same as input)
    pub out_channels: usize,
    /// Base model channels
    pub model_channels: usize,
    /// Channel multipliers per resolution level
    pub channel_mult: Vec<usize>,
    /// Number of transformer blocks at each resolution
    pub transformer_depth: Vec<usize>,
    /// Attention head dimension
    pub head_dim: usize,
    /// Context dimension (concatenated text embedding dim)
    pub context_dim: usize,
    /// Additional embedding dimension (for pooled text + time conditioning)
    pub add_emb_dim: usize,
}

impl Default for UNetXLConfig {
    fn default() -> Self {
        Self::sdxl_base()
    }
}

impl UNetXLConfig {
    /// Configuration for SDXL Base
    pub fn sdxl_base() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            model_channels: 320,
            channel_mult: vec![1, 2, 4],
            transformer_depth: vec![1, 2, 10], // Different depths per resolution
            head_dim: 64,
            context_dim: 2048, // CLIP 768 + OpenCLIP 1280
            add_emb_dim: 2816, // Pooled embedding dim
        }
    }

    /// Configuration for SDXL Refiner
    pub fn sdxl_refiner() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            model_channels: 384,
            channel_mult: vec![1, 2, 4, 4],
            transformer_depth: vec![4, 4, 4, 4],
            head_dim: 64,
            context_dim: 1280, // OpenCLIP only
            add_emb_dim: 2560,
        }
    }

    /// Computes the number of attention heads for a given channel count
    fn num_heads_at(&self, channels: usize) -> usize {
        channels / self.head_dim
    }
}

/// SDXL UNet
#[derive(Module, Debug)]
pub struct UNetXL<B: Backend> {
    /// Time embedding first linear layer
    pub time_embed_0: Linear<B>,
    /// Time embedding second linear layer
    pub time_embed_2: Linear<B>,

    /// Additional embedding first linear (pooled text + size conditioning)
    pub add_embed_0: Linear<B>,
    /// Additional embedding second linear
    pub add_embed_2: Linear<B>,

    /// Precomputed timestep embedding frequencies (for performance)
    pub time_freqs: Tensor<B, 1>,

    /// Input convolution
    pub conv_in: Conv2d<B>,

    /// Down blocks (encoder path)
    pub down_blocks: Vec<DownBlockXL<B>>,

    /// Middle block
    pub mid_block: MidBlockXL<B>,

    /// Up blocks (decoder path)
    pub up_blocks: Vec<UpBlockXL<B>>,

    /// Output group norm
    pub norm_out: GroupNorm<B>,
    /// Output convolution
    pub conv_out: Conv2d<B>,

    /// Base model channels
    pub model_channels: usize,
}

impl<B: Backend> UNetXL<B> {
    /// Creates a new SDXL UNet
    ///
    /// # Arguments
    ///
    /// * `config` - SDXL UNet configuration
    /// * `device` - Device to create tensors on
    pub fn new(config: &UNetXLConfig, device: &B::Device) -> Self {
        let ch = config.model_channels;
        let time_embed_dim = ch * 4;

        // Time embedding MLP
        let time_embed_0 = LinearConfig::new(ch, time_embed_dim).init(device);
        let time_embed_2 = LinearConfig::new(time_embed_dim, time_embed_dim).init(device);

        // Additional embedding MLP
        let add_embed_0 = LinearConfig::new(config.add_emb_dim, time_embed_dim).init(device);
        let add_embed_2 = LinearConfig::new(time_embed_dim, time_embed_dim).init(device);

        // Precompute timestep embedding frequencies
        let time_freqs = timestep_freqs(ch, device);

        // Input conv
        let conv_in = Conv2dConfig::new([config.in_channels, ch], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        // Build down blocks
        let mut down_blocks = Vec::new();
        let mut channels = vec![ch];
        let mut ch_in = ch;

        for (level, &mult) in config.channel_mult.iter().enumerate() {
            let ch_out = ch * mult;
            let is_last = level == config.channel_mult.len() - 1;
            let depth = config.transformer_depth[level];
            let num_heads = config.num_heads_at(ch_out);

            down_blocks.push(DownBlockXL::new(
                ch_in,
                ch_out,
                time_embed_dim,
                num_heads,
                config.context_dim,
                depth,
                !is_last,
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
        let mid_depth = *config.transformer_depth.last().unwrap();
        let mid_heads = config.num_heads_at(ch_in);
        let mid_block = MidBlockXL::new(
            ch_in,
            time_embed_dim,
            mid_heads,
            config.context_dim,
            mid_depth,
            device,
        );

        // Build up blocks (reverse order)
        let mut up_blocks = Vec::new();

        for (level, &mult) in config.channel_mult.iter().rev().enumerate() {
            let ch_out = ch * mult;
            let is_last = level == config.channel_mult.len() - 1;
            let depth_idx = config.channel_mult.len() - 1 - level;
            let depth = config.transformer_depth[depth_idx];
            let num_heads = config.num_heads_at(ch_out);

            for i in 0..3 {
                let skip_ch = channels.pop().unwrap();
                let block_in = ch_in + skip_ch;
                let block_out = if i == 2 && !is_last {
                    ch * config.channel_mult[config.channel_mult.len() - 2 - level]
                } else {
                    ch_out
                };

                let upsample = i == 2 && !is_last;

                up_blocks.push(UpBlockXL::new(
                    block_in,
                    block_out,
                    time_embed_dim,
                    num_heads,
                    config.context_dim,
                    depth,
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

        Self {
            time_embed_0,
            time_embed_2,
            add_embed_0,
            add_embed_2,
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
    /// * `context` - Concatenated text embeddings [batch, seq_len, context_dim]
    /// * `add_embed` - Additional embedding (pooled text + time conditioning) [batch, add_emb_dim]
    ///
    /// # Returns
    /// Predicted noise [batch, 4, h, w]
    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        timesteps: Tensor<B, 1>,
        context: Tensor<B, 3>,
        add_embed: Tensor<B, 2>,
    ) -> Tensor<B, 4> {
        // Time embedding (using precomputed frequencies)
        let t_emb = timestep_embedding_with_freqs(timesteps, self.time_freqs.clone());
        let t_emb = self.time_embed_0.forward(t_emb);
        let t_emb = silu(t_emb);
        let t_emb = self.time_embed_2.forward(t_emb);

        // Additional embedding
        let add_emb = self.add_embed_0.forward(add_embed);
        let add_emb = silu(add_emb);
        let add_emb = self.add_embed_2.forward(add_emb);

        // Combine time and additional embedding
        let emb = t_emb + add_emb;

        // Input
        let mut h = self.conv_in.forward(x);

        // Down blocks with skip connections
        let mut skips = vec![h.clone()];
        for block in &self.down_blocks {
            let (out, block_skips) = block.forward(h, emb.clone(), context.clone());
            h = out;
            skips.extend(block_skips);
        }

        // Mid block
        h = self.mid_block.forward(h, emb.clone(), context.clone());

        // Up blocks with skip connections
        for block in &self.up_blocks {
            let skip = skips.pop().unwrap();
            h = Tensor::cat(vec![h, skip], 1);
            h = block.forward(h, emb.clone(), context.clone());
        }

        // Output
        h = self.norm_out.forward(h);
        h = silu(h);
        self.conv_out.forward(h)
    }

    /// Forward pass with simple conditioning (no add_embed)
    ///
    /// Creates a zero add_embed tensor for compatibility
    pub fn forward_simple(
        &self,
        x: Tensor<B, 4>,
        timesteps: Tensor<B, 1>,
        context: Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        let [batch, _, _, _] = x.dims();
        // Infer add_emb_dim from the add_embed_0 layer
        let add_emb_dim = 2816; // Default for SDXL base
        let add_embed = Tensor::zeros([batch, add_emb_dim], &x.device());
        self.forward(x, timesteps, context, add_embed)
    }
}

/// SDXL Down block with variable transformer depth
#[derive(Module, Debug)]
pub struct DownBlockXL<B: Backend> {
    /// First ResNet block
    pub res1: ResBlock<B>,
    /// First attention (optional, depends on transformer_depth)
    pub attn1: Option<SpatialTransformer<B>>,
    /// Second ResNet block
    pub res2: ResBlock<B>,
    /// Second attention (optional)
    pub attn2: Option<SpatialTransformer<B>>,
    /// Downsampler (optional, not present on last level)
    pub downsample: Option<Downsample<B>>,
}

impl<B: Backend> DownBlockXL<B> {
    /// Creates a new SDXL down block with optional attention layers
    #[allow(clippy::too_many_arguments)]
    fn new(
        in_ch: usize,
        out_ch: usize,
        time_dim: usize,
        num_heads: usize,
        context_dim: usize,
        transformer_depth: usize,
        downsample: bool,
        device: &B::Device,
    ) -> Self {
        let head_dim = out_ch / num_heads;

        // Only add attention if transformer_depth > 0
        let attn1 = if transformer_depth > 0 {
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
        };

        let attn2 = if transformer_depth > 0 {
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
        };

        Self {
            res1: ResBlock::new(in_ch, out_ch, time_dim, device),
            attn1,
            res2: ResBlock::new(out_ch, out_ch, time_dim, device),
            attn2,
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
        emb: Tensor<B, 2>,
        context: Tensor<B, 3>,
    ) -> (Tensor<B, 4>, Vec<Tensor<B, 4>>) {
        let mut skips = Vec::new();

        let h = self.res1.forward(x, emb.clone());
        let h = if let Some(attn) = &self.attn1 {
            attn.forward(h, context.clone())
        } else {
            h
        };
        skips.push(h.clone());

        let h = self.res2.forward(h, emb);
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

/// SDXL Mid block with variable transformer depth
#[derive(Module, Debug)]
pub struct MidBlockXL<B: Backend> {
    /// First ResNet block
    pub res1: ResBlock<B>,
    /// Attention block
    pub attn: SpatialTransformer<B>,
    /// Second ResNet block
    pub res2: ResBlock<B>,
}

impl<B: Backend> MidBlockXL<B> {
    /// Creates a new SDXL mid block
    fn new(
        channels: usize,
        time_dim: usize,
        num_heads: usize,
        context_dim: usize,
        transformer_depth: usize,
        device: &B::Device,
    ) -> Self {
        let head_dim = channels / num_heads;

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
    fn forward(&self, x: Tensor<B, 4>, emb: Tensor<B, 2>, context: Tensor<B, 3>) -> Tensor<B, 4> {
        let h = self.res1.forward(x, emb.clone());
        let h = self.attn.forward(h, context);
        self.res2.forward(h, emb)
    }
}

/// SDXL Up block with variable transformer depth
#[derive(Module, Debug)]
pub struct UpBlockXL<B: Backend> {
    /// ResNet block (takes skip connection input)
    pub res: ResBlock<B>,
    /// Attention block (optional)
    pub attn: Option<SpatialTransformer<B>>,
    /// Upsampler (optional)
    pub upsample: Option<Upsample<B>>,
}

impl<B: Backend> UpBlockXL<B> {
    /// Creates a new SDXL up block with optional attention and upsampling
    #[allow(clippy::too_many_arguments)]
    fn new(
        in_ch: usize,
        out_ch: usize,
        time_dim: usize,
        num_heads: usize,
        context_dim: usize,
        transformer_depth: usize,
        upsample: bool,
        device: &B::Device,
    ) -> Self {
        let head_dim = if num_heads > 0 {
            out_ch / num_heads
        } else {
            out_ch
        };

        let attn = if transformer_depth > 0 && num_heads > 0 {
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
        };

        Self {
            res: ResBlock::new(in_ch, out_ch, time_dim, device),
            attn,
            upsample: if upsample {
                Some(Upsample::new(out_ch, device))
            } else {
                None
            },
        }
    }

    /// Forward pass through the up block
    fn forward(&self, x: Tensor<B, 4>, emb: Tensor<B, 2>, context: Tensor<B, 3>) -> Tensor<B, 4> {
        let h = self.res.forward(x, emb);
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
    fn test_unetxl_config() {
        let config = UNetXLConfig::sdxl_base();
        assert_eq!(config.in_channels, 4);
        assert_eq!(config.context_dim, 2048);
        assert_eq!(config.channel_mult, vec![1, 2, 4]);
        assert_eq!(config.transformer_depth, vec![1, 2, 10]);
    }

    #[test]
    fn test_refiner_config() {
        let config = UNetXLConfig::sdxl_refiner();
        assert_eq!(config.model_channels, 384);
        assert_eq!(config.context_dim, 1280);
    }
}
