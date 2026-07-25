//! Stable Cascade (Würstchen) Implementation
//!
//! Stable Cascade is a latent cascade model that operates in two stages:
//!
//! # Architecture Overview
//!
//! ```text
//! Text Prompt
//!      ↓
//! [Stage C] - High-level semantic latents (16x compressed)
//!      ↓
//! [Stage B] - Decode to lower-level latents (4x compressed)
//!      ↓
//! [Stage A] - VAE decoder to pixels
//! ```
//!
//! # Key Features
//!
//! - **Würstchen architecture**: Two-stage cascade diffusion
//! - **Efficient**: Stage C operates on highly compressed 16x latents
//! - **High quality**: Stage B refines to standard VAE latents
//! - **Fast inference**: Stage C enables fewer steps at high compression

use burn::nn::{
    Linear, LinearConfig, PaddingConfig2d,
    conv::{Conv2d, Conv2dConfig},
};
use burn::prelude::*;

use sketchpad_core::groupnorm::GroupNorm;
use sketchpad_core::layernorm::LayerNorm;
use sketchpad_core::silu::silu;

use crate::blocks::timestep_embedding;

/// Stage C configuration (high-level latent generator)
#[derive(Debug, Clone)]
pub struct StageCConfig {
    /// Input channels (Stage C latent channels)
    pub in_channels: usize,
    /// Output channels (same as input)
    pub out_channels: usize,
    /// Base model channels
    pub model_channels: usize,
    /// Channel multipliers per resolution level
    pub channel_mult: Vec<usize>,
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Text embedding dimension (CLIP)
    pub context_dim: usize,
    /// Number of res blocks per level
    pub num_res_blocks: usize,
    /// Compression factor (16x for Stage C)
    pub compression: usize,
}

impl StageCConfig {
    /// Default Stage C configuration
    pub fn default_c() -> Self {
        Self {
            in_channels: 16, // Stage C latents
            out_channels: 16,
            model_channels: 1536,
            channel_mult: vec![1, 1, 1, 1],
            num_heads: 24,
            head_dim: 64,
            context_dim: 1280, // CLIP embedding
            num_res_blocks: 3,
            compression: 16,
        }
    }

    /// Tiny config for testing
    pub fn tiny() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            model_channels: 256,
            channel_mult: vec![1, 2, 2],
            num_heads: 4,
            head_dim: 64,
            context_dim: 128,
            num_res_blocks: 2,
            compression: 16,
        }
    }
}

/// Stage B configuration (latent decoder)
#[derive(Debug, Clone)]
pub struct StageBConfig {
    /// Input channels (Stage B latent channels, same as VAE)
    pub in_channels: usize,
    /// Output channels
    pub out_channels: usize,
    /// Conditioning channels (Stage C output)
    pub cond_channels: usize,
    /// Base model channels
    pub model_channels: usize,
    /// Channel multipliers per resolution level
    pub channel_mult: Vec<usize>,
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Text embedding dimension
    pub context_dim: usize,
    /// Number of res blocks per level
    pub num_res_blocks: usize,
}

impl StageBConfig {
    /// Default Stage B configuration
    pub fn default_b() -> Self {
        Self {
            in_channels: 4, // Standard VAE latent channels
            out_channels: 4,
            cond_channels: 16, // Stage C output channels
            model_channels: 640,
            channel_mult: vec![1, 2, 2, 4],
            num_heads: 10,
            head_dim: 64,
            context_dim: 1280,
            num_res_blocks: 2,
        }
    }

    /// Tiny config for testing
    pub fn tiny() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            cond_channels: 4,
            model_channels: 128,
            channel_mult: vec![1, 2],
            num_heads: 4,
            head_dim: 32,
            context_dim: 64,
            num_res_blocks: 1,
        }
    }
}

/// Stage C timestep embedding
#[derive(Module, Debug)]
pub struct CascadeTimestepEmbed<B: Backend> {
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
    pub model_channels: usize,
}

impl<B: Backend> CascadeTimestepEmbed<B> {
    pub fn new(model_channels: usize, time_embed_dim: usize, device: &B::Device) -> Self {
        Self {
            linear1: LinearConfig::new(model_channels, time_embed_dim).init(device),
            linear2: LinearConfig::new(time_embed_dim, time_embed_dim).init(device),
            model_channels,
        }
    }

    pub fn forward(&self, timesteps: Tensor<B, 1>) -> Tensor<B, 2> {
        let t_emb = timestep_embedding(
            timesteps,
            self.model_channels,
            &self.linear1.weight.device(),
        );
        let t_emb = silu(self.linear1.forward(t_emb));
        self.linear2.forward(t_emb)
    }
}

/// Cascade Attention Block
#[derive(Module, Debug)]
pub struct CascadeAttention<B: Backend> {
    pub norm: LayerNorm<B>,
    pub to_qkv: Linear<B>,
    pub to_out: Linear<B>,
    #[module(skip)]
    pub num_heads: usize,
    #[module(skip)]
    pub head_dim: usize,
}

impl<B: Backend> CascadeAttention<B> {
    pub fn new(channels: usize, num_heads: usize, head_dim: usize, device: &B::Device) -> Self {
        let inner_dim = num_heads * head_dim;
        Self {
            norm: LayerNorm::new(channels, device),
            to_qkv: LinearConfig::new(channels, 3 * inner_dim)
                .with_bias(true)
                .init(device),
            to_out: LinearConfig::new(inner_dim, channels)
                .with_bias(true)
                .init(device),
            num_heads,
            head_dim,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, channels, height, width] = x.dims();
        let seq_len = height * width;

        // Reshape to [B, H*W, C]
        let x_flat = x
            .clone()
            .permute([0, 2, 3, 1])
            .reshape([batch, seq_len, channels]);
        let x_norm = self.norm.forward(x_flat);

        // Compute Q, K, V
        let qkv = self.to_qkv.forward(x_norm);
        let qkv = qkv.reshape([batch, seq_len, 3, self.num_heads, self.head_dim]);

        let q = qkv
            .clone()
            .slice([
                0..batch,
                0..seq_len,
                0..1,
                0..self.num_heads,
                0..self.head_dim,
            ])
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = qkv
            .clone()
            .slice([
                0..batch,
                0..seq_len,
                1..2,
                0..self.num_heads,
                0..self.head_dim,
            ])
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = qkv
            .slice([
                0..batch,
                0..seq_len,
                2..3,
                0..self.num_heads,
                0..self.head_dim,
            ])
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);

        // Attention
        let scale = (self.head_dim as f32).sqrt().recip();
        let attn = q.matmul(k.swap_dims(2, 3)) * scale;
        let attn = burn::tensor::activation::softmax(attn, 3);
        let out = attn.matmul(v);

        // Reshape back
        let out = out
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * self.head_dim]);
        let out = self.to_out.forward(out);

        // Back to spatial: [B, H, W, C] -> [B, C, H, W]
        let out = out
            .reshape([batch, height, width, channels])
            .permute([0, 3, 1, 2]);

        x + out
    }
}

/// Cascade Cross Attention Block
#[derive(Module, Debug)]
pub struct CascadeCrossAttention<B: Backend> {
    pub norm: LayerNorm<B>,
    pub to_q: Linear<B>,
    pub to_kv: Linear<B>,
    pub to_out: Linear<B>,
    #[module(skip)]
    pub num_heads: usize,
    #[module(skip)]
    pub head_dim: usize,
}

impl<B: Backend> CascadeCrossAttention<B> {
    pub fn new(
        channels: usize,
        context_dim: usize,
        num_heads: usize,
        head_dim: usize,
        device: &B::Device,
    ) -> Self {
        let inner_dim = num_heads * head_dim;
        Self {
            norm: LayerNorm::new(channels, device),
            to_q: LinearConfig::new(channels, inner_dim)
                .with_bias(true)
                .init(device),
            to_kv: LinearConfig::new(context_dim, 2 * inner_dim)
                .with_bias(true)
                .init(device),
            to_out: LinearConfig::new(inner_dim, channels)
                .with_bias(true)
                .init(device),
            num_heads,
            head_dim,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>, context: Tensor<B, 3>) -> Tensor<B, 4> {
        let [batch, channels, height, width] = x.dims();
        let [_, ctx_len, _] = context.dims();
        let seq_len = height * width;

        // Reshape to [B, H*W, C]
        let x_flat = x
            .clone()
            .permute([0, 2, 3, 1])
            .reshape([batch, seq_len, channels]);
        let x_norm = self.norm.forward(x_flat);

        // Q from image, K/V from context
        let q = self.to_q.forward(x_norm);
        let q = q
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);

        let kv = self.to_kv.forward(context);
        let kv = kv.reshape([batch, ctx_len, 2, self.num_heads, self.head_dim]);

        let k = kv
            .clone()
            .slice([
                0..batch,
                0..ctx_len,
                0..1,
                0..self.num_heads,
                0..self.head_dim,
            ])
            .reshape([batch, ctx_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = kv
            .slice([
                0..batch,
                0..ctx_len,
                1..2,
                0..self.num_heads,
                0..self.head_dim,
            ])
            .reshape([batch, ctx_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);

        // Attention
        let scale = (self.head_dim as f32).sqrt().recip();
        let attn = q.matmul(k.swap_dims(2, 3)) * scale;
        let attn = burn::tensor::activation::softmax(attn, 3);
        let out = attn.matmul(v);

        // Reshape back
        let out = out
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * self.head_dim]);
        let out = self.to_out.forward(out);

        // Back to spatial
        let out = out
            .reshape([batch, height, width, channels])
            .permute([0, 3, 1, 2]);

        x + out
    }
}

/// Cascade ResBlock with conditioning
#[derive(Module, Debug)]
pub struct CascadeResBlock<B: Backend> {
    pub norm1: GroupNorm<B>,
    pub conv1: Conv2d<B>,
    pub norm2: GroupNorm<B>,
    pub conv2: Conv2d<B>,
    pub time_emb_proj: Linear<B>,
    pub skip_conv: Option<Conv2d<B>>,
}

impl<B: Backend> CascadeResBlock<B> {
    pub fn new(in_ch: usize, out_ch: usize, time_dim: usize, device: &B::Device) -> Self {
        let skip_conv = if in_ch != out_ch {
            Some(Conv2dConfig::new([in_ch, out_ch], [1, 1]).init(device))
        } else {
            None
        };

        Self {
            norm1: GroupNorm::new(32.min(in_ch), in_ch, device),
            conv1: Conv2dConfig::new([in_ch, out_ch], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            norm2: GroupNorm::new(32.min(out_ch), out_ch, device),
            conv2: Conv2dConfig::new([out_ch, out_ch], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            time_emb_proj: LinearConfig::new(time_dim, out_ch).init(device),
            skip_conv,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>, t_emb: Tensor<B, 2>) -> Tensor<B, 4> {
        let h = self.norm1.forward(x.clone());
        let h = silu(h);
        let h = self.conv1.forward(h);

        // Add time embedding
        let t_proj = silu(self.time_emb_proj.forward(t_emb));
        let [batch, ch] = t_proj.dims();
        let h = h + t_proj.reshape([batch, ch, 1, 1]);

        let h = self.norm2.forward(h);
        let h = silu(h);
        let h = self.conv2.forward(h);

        // Skip connection
        let skip = if let Some(conv) = &self.skip_conv {
            conv.forward(x)
        } else {
            x
        };

        h + skip
    }
}

/// Cascade Down Block
#[derive(Module, Debug)]
pub struct CascadeDownBlock<B: Backend> {
    pub res_blocks: Vec<CascadeResBlock<B>>,
    pub self_attn: Option<CascadeAttention<B>>,
    pub cross_attn: Option<CascadeCrossAttention<B>>,
    pub downsample: Option<Conv2d<B>>,
}

impl<B: Backend> CascadeDownBlock<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        time_dim: usize,
        num_res_blocks: usize,
        num_heads: usize,
        head_dim: usize,
        context_dim: usize,
        has_attention: bool,
        has_downsample: bool,
        device: &B::Device,
    ) -> Self {
        let mut res_blocks = Vec::new();
        for i in 0..num_res_blocks {
            let block_in = if i == 0 { in_ch } else { out_ch };
            res_blocks.push(CascadeResBlock::new(block_in, out_ch, time_dim, device));
        }

        let self_attn = if has_attention {
            Some(CascadeAttention::new(out_ch, num_heads, head_dim, device))
        } else {
            None
        };

        let cross_attn = if has_attention {
            Some(CascadeCrossAttention::new(
                out_ch,
                context_dim,
                num_heads,
                head_dim,
                device,
            ))
        } else {
            None
        };

        let downsample = if has_downsample {
            Some(
                Conv2dConfig::new([out_ch, out_ch], [3, 3])
                    .with_stride([2, 2])
                    .with_padding(PaddingConfig2d::Explicit(1, 1))
                    .init(device),
            )
        } else {
            None
        };

        Self {
            res_blocks,
            self_attn,
            cross_attn,
            downsample,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        t_emb: Tensor<B, 2>,
        context: Tensor<B, 3>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let mut h = x;

        for res_block in &self.res_blocks {
            h = res_block.forward(h, t_emb.clone());
        }

        if let Some(attn) = &self.self_attn {
            h = attn.forward(h);
        }

        if let Some(cross) = &self.cross_attn {
            h = cross.forward(h, context);
        }

        // Store skip BEFORE downsampling (one per level)
        let skip = h.clone();

        if let Some(ds) = &self.downsample {
            h = ds.forward(h);
        }

        (h, skip)
    }
}

/// Cascade Up Block
#[derive(Module, Debug)]
pub struct CascadeUpBlock<B: Backend> {
    pub res_blocks: Vec<CascadeResBlock<B>>,
    pub self_attn: Option<CascadeAttention<B>>,
    pub cross_attn: Option<CascadeCrossAttention<B>>,
    pub upsample: Option<Conv2d<B>>,
    #[module(skip)]
    pub has_upsample: bool,
}

impl<B: Backend> CascadeUpBlock<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        time_dim: usize,
        num_res_blocks: usize,
        num_heads: usize,
        head_dim: usize,
        context_dim: usize,
        has_attention: bool,
        has_upsample: bool,
        device: &B::Device,
    ) -> Self {
        let mut res_blocks = Vec::new();
        for i in 0..num_res_blocks {
            // First block takes skip connection (doubled channels)
            let block_in = if i == 0 { in_ch + out_ch } else { out_ch };
            let block_out = out_ch;
            res_blocks.push(CascadeResBlock::new(block_in, block_out, time_dim, device));
        }

        let self_attn = if has_attention {
            Some(CascadeAttention::new(out_ch, num_heads, head_dim, device))
        } else {
            None
        };

        let cross_attn = if has_attention {
            Some(CascadeCrossAttention::new(
                out_ch,
                context_dim,
                num_heads,
                head_dim,
                device,
            ))
        } else {
            None
        };

        let upsample = if has_upsample {
            Some(
                Conv2dConfig::new([out_ch, out_ch], [3, 3])
                    .with_padding(PaddingConfig2d::Explicit(1, 1))
                    .init(device),
            )
        } else {
            None
        };

        Self {
            res_blocks,
            self_attn,
            cross_attn,
            upsample,
            has_upsample,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        skip: Tensor<B, 4>,
        t_emb: Tensor<B, 2>,
        context: Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        // Concatenate with skip connection (same resolution)
        let mut h = Tensor::cat(vec![x, skip], 1);

        for res_block in &self.res_blocks {
            h = res_block.forward(h, t_emb.clone());
        }

        if let Some(attn) = &self.self_attn {
            h = attn.forward(h);
        }

        if let Some(cross) = &self.cross_attn {
            h = cross.forward(h, context);
        }

        // Upsample AFTER processing (to match next level's skip resolution)
        if self.has_upsample {
            let [batch, channels, height, width] = h.dims();
            h = h
                .reshape([batch, channels, height, 1, width, 1])
                .repeat_dim(3, 2)
                .repeat_dim(5, 2)
                .reshape([batch, channels, height * 2, width * 2]);
            if let Some(up) = &self.upsample {
                h = up.forward(h);
            }
        }

        h
    }
}

/// Stage C Model (high-level latent generator)
#[derive(Module, Debug)]
pub struct StageC<B: Backend> {
    pub conv_in: Conv2d<B>,
    pub time_embed: CascadeTimestepEmbed<B>,
    pub down_blocks: Vec<CascadeDownBlock<B>>,
    pub mid_res1: CascadeResBlock<B>,
    pub mid_attn: CascadeAttention<B>,
    pub mid_cross: CascadeCrossAttention<B>,
    pub mid_res2: CascadeResBlock<B>,
    pub up_blocks: Vec<CascadeUpBlock<B>>,
    pub norm_out: GroupNorm<B>,
    pub conv_out: Conv2d<B>,
}

/// Output from Stage C
pub struct StageCOutput<B: Backend> {
    /// Predicted noise or velocity [batch, 16, h/16, w/16]
    pub output: Tensor<B, 4>,
}

impl<B: Backend> StageC<B> {
    pub fn new(config: &StageCConfig, device: &B::Device) -> Self {
        let ch = config.model_channels;
        let time_embed_dim = ch * 4;

        let conv_in = Conv2dConfig::new([config.in_channels, ch], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        let time_embed = CascadeTimestepEmbed::new(ch, time_embed_dim, device);

        // Build down blocks
        let mut down_blocks = Vec::new();
        let mut ch_in = ch;
        for (level, &mult) in config.channel_mult.iter().enumerate() {
            let ch_out = ch * mult;
            let is_last = level == config.channel_mult.len() - 1;
            down_blocks.push(CascadeDownBlock::new(
                ch_in,
                ch_out,
                time_embed_dim,
                config.num_res_blocks,
                config.num_heads,
                config.head_dim,
                config.context_dim,
                true,
                !is_last,
                device,
            ));
            ch_in = ch_out;
        }

        // Mid block (ch_in is the final channel count after the loop)
        let mid_res1 = CascadeResBlock::new(ch_in, ch_in, time_embed_dim, device);
        let mid_attn = CascadeAttention::new(ch_in, config.num_heads, config.head_dim, device);
        let mid_cross = CascadeCrossAttention::new(
            ch_in,
            config.context_dim,
            config.num_heads,
            config.head_dim,
            device,
        );
        let mid_res2 = CascadeResBlock::new(ch_in, ch_in, time_embed_dim, device);

        // Build up blocks (reverse order)
        let mut up_blocks = Vec::new();
        for (level, &mult) in config.channel_mult.iter().rev().enumerate() {
            let ch_out = ch * mult;
            let is_last = level == config.channel_mult.len() - 1;
            let prev_mult = if level > 0 {
                config.channel_mult[config.channel_mult.len() - level]
            } else {
                *config.channel_mult.last().unwrap()
            };
            let block_in = ch * prev_mult;

            up_blocks.push(CascadeUpBlock::new(
                block_in,
                ch_out,
                time_embed_dim,
                config.num_res_blocks,
                config.num_heads,
                config.head_dim,
                config.context_dim,
                true,
                !is_last,
                device,
            ));
        }

        let norm_out = GroupNorm::new(32.min(ch), ch, device);
        let conv_out = Conv2dConfig::new([ch, config.out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        Self {
            conv_in,
            time_embed,
            down_blocks,
            mid_res1,
            mid_attn,
            mid_cross,
            mid_res2,
            up_blocks,
            norm_out,
            conv_out,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        timesteps: Tensor<B, 1>,
        context: Tensor<B, 3>,
    ) -> StageCOutput<B> {
        // Time embedding
        let t_emb = self.time_embed.forward(timesteps);

        // Input
        let h = self.conv_in.forward(x);

        // Down blocks with skip connections (one skip per level)
        let mut skips = Vec::new();
        let mut h = h;
        for block in &self.down_blocks {
            let (out, skip) = block.forward(h, t_emb.clone(), context.clone());
            h = out;
            skips.push(skip);
        }

        // Mid block
        h = self.mid_res1.forward(h, t_emb.clone());
        h = self.mid_attn.forward(h);
        h = self.mid_cross.forward(h, context.clone());
        h = self.mid_res2.forward(h, t_emb.clone());

        // Up blocks with skip connections (reverse order)
        for block in &self.up_blocks {
            let skip = skips.pop().unwrap_or_else(|| h.clone());
            h = block.forward(h, skip, t_emb.clone(), context.clone());
        }

        // Output
        h = self.norm_out.forward(h);
        h = silu(h);
        let output = self.conv_out.forward(h);

        StageCOutput { output }
    }
}

/// Stage B Model (latent decoder, conditioned on Stage C output)
#[derive(Module, Debug)]
pub struct StageB<B: Backend> {
    pub conv_in: Conv2d<B>,
    pub cond_embed: Conv2d<B>,
    pub time_embed: CascadeTimestepEmbed<B>,
    pub down_blocks: Vec<CascadeDownBlock<B>>,
    pub mid_res1: CascadeResBlock<B>,
    pub mid_attn: CascadeAttention<B>,
    pub mid_res2: CascadeResBlock<B>,
    pub up_blocks: Vec<CascadeUpBlock<B>>,
    pub norm_out: GroupNorm<B>,
    pub conv_out: Conv2d<B>,
}

/// Output from Stage B
pub struct StageBOutput<B: Backend> {
    /// Predicted noise or velocity [batch, 4, h/4, w/4]
    pub output: Tensor<B, 4>,
}

impl<B: Backend> StageB<B> {
    pub fn new(config: &StageBConfig, device: &B::Device) -> Self {
        let ch = config.model_channels;
        let time_embed_dim = ch * 4;

        let conv_in = Conv2dConfig::new([config.in_channels, ch], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        // Conditioning from Stage C (upscaled to match spatial dimensions)
        let cond_embed = Conv2dConfig::new([config.cond_channels, ch], [1, 1]).init(device);

        let time_embed = CascadeTimestepEmbed::new(ch, time_embed_dim, device);

        // Build down blocks
        let mut down_blocks = Vec::new();
        let mut ch_in = ch;
        for (level, &mult) in config.channel_mult.iter().enumerate() {
            let ch_out = ch * mult;
            let is_last = level == config.channel_mult.len() - 1;
            down_blocks.push(CascadeDownBlock::new(
                ch_in,
                ch_out,
                time_embed_dim,
                config.num_res_blocks,
                config.num_heads,
                config.head_dim,
                config.context_dim,
                level >= 2,
                !is_last,
                device,
            ));
            ch_in = ch_out;
        }

        // Mid block
        let mid_res1 = CascadeResBlock::new(ch_in, ch_in, time_embed_dim, device);
        let mid_attn = CascadeAttention::new(ch_in, config.num_heads, config.head_dim, device);
        let mid_res2 = CascadeResBlock::new(ch_in, ch_in, time_embed_dim, device);

        // Build up blocks
        let mut up_blocks = Vec::new();
        for (level, &mult) in config.channel_mult.iter().rev().enumerate() {
            let ch_out = ch * mult;
            let is_last = level == config.channel_mult.len() - 1;
            let prev_mult = if level > 0 {
                config.channel_mult[config.channel_mult.len() - level]
            } else {
                *config.channel_mult.last().unwrap()
            };
            let block_in = ch * prev_mult;

            up_blocks.push(CascadeUpBlock::new(
                block_in,
                ch_out,
                time_embed_dim,
                config.num_res_blocks,
                config.num_heads,
                config.head_dim,
                config.context_dim,
                level < 2,
                !is_last,
                device,
            ));
        }

        let norm_out = GroupNorm::new(32.min(ch), ch, device);
        let conv_out = Conv2dConfig::new([ch, config.out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        Self {
            conv_in,
            cond_embed,
            time_embed,
            down_blocks,
            mid_res1,
            mid_attn,
            mid_res2,
            up_blocks,
            norm_out,
            conv_out,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        timesteps: Tensor<B, 1>,
        stage_c_output: Tensor<B, 4>,
        context: Tensor<B, 3>,
    ) -> StageBOutput<B> {
        // Time embedding
        let t_emb = self.time_embed.forward(timesteps);

        // Input
        let h = self.conv_in.forward(x);

        // Add conditioning from Stage C (upscaled)
        let [_, _, h_c, _] = stage_c_output.dims();
        let [_, _, h_b, _] = h.dims();
        let scale = h_b / h_c;

        let cond = if scale > 1 {
            let [b, c, oh, ow] = stage_c_output.dims();
            stage_c_output
                .reshape([b, c, oh, 1, ow, 1])
                .repeat_dim(3, scale)
                .repeat_dim(5, scale)
                .reshape([b, c, oh * scale, ow * scale])
        } else {
            stage_c_output
        };
        let cond = self.cond_embed.forward(cond);
        let h = h + cond;

        // Down blocks (one skip per level)
        let mut skips = Vec::new();
        let mut h = h;
        for block in &self.down_blocks {
            let (out, skip) = block.forward(h, t_emb.clone(), context.clone());
            h = out;
            skips.push(skip);
        }

        // Mid block
        h = self.mid_res1.forward(h, t_emb.clone());
        h = self.mid_attn.forward(h);
        h = self.mid_res2.forward(h, t_emb.clone());

        // Up blocks (reverse order)
        for block in &self.up_blocks {
            let skip = skips.pop().unwrap_or_else(|| h.clone());
            h = block.forward(h, skip, t_emb.clone(), context.clone());
        }

        // Output
        h = self.norm_out.forward(h);
        h = silu(h);
        let output = self.conv_out.forward(h);

        StageBOutput { output }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_stage_c_config() {
        let config = StageCConfig::default_c();
        assert_eq!(config.model_channels, 1536);
        assert_eq!(config.compression, 16);
    }

    #[test]
    fn test_stage_b_config() {
        let config = StageBConfig::default_b();
        assert_eq!(config.model_channels, 640);
        assert_eq!(config.in_channels, 4);
    }

    #[test]
    fn test_cascade_attention() {
        let device = Default::default();
        let attn = CascadeAttention::new(64, 4, 16, &device);

        let x = Tensor::<TestBackend, 4>::zeros([1, 64, 8, 8], &device);
        let out = attn.forward(x);
        assert_eq!(out.dims(), [1, 64, 8, 8]);
    }

    #[test]
    fn test_cascade_res_block() {
        let device = Default::default();
        let block = CascadeResBlock::new(64, 128, 256, &device);

        let x = Tensor::<TestBackend, 4>::zeros([1, 64, 8, 8], &device);
        let t_emb = Tensor::zeros([1, 256], &device);
        let out = block.forward(x, t_emb);
        assert_eq!(out.dims(), [1, 128, 8, 8]);
    }

    #[test]
    fn test_stage_c_tiny() {
        let device = Default::default();
        let config = StageCConfig::tiny();
        let model = StageC::new(&config, &device);

        let x = Tensor::<TestBackend, 4>::zeros([1, 4, 8, 8], &device);
        let timesteps = Tensor::from_floats([500.0], &device);
        let context = Tensor::zeros([1, 4, 128], &device);

        let output = model.forward(x, timesteps, context);
        assert_eq!(output.output.dims(), [1, 4, 8, 8]);
    }

    #[test]
    fn test_stage_b_tiny() {
        let device = Default::default();
        let config = StageBConfig::tiny();
        let model = StageB::new(&config, &device);

        let x = Tensor::<TestBackend, 4>::zeros([1, 4, 16, 16], &device);
        let timesteps = Tensor::from_floats([500.0], &device);
        let stage_c = Tensor::zeros([1, 4, 4, 4], &device); // 4x smaller
        let context = Tensor::zeros([1, 4, 64], &device);

        let output = model.forward(x, timesteps, stage_c, context);
        assert_eq!(output.output.dims(), [1, 4, 16, 16]);
    }
}
