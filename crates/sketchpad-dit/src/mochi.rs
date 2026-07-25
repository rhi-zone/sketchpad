//! Mochi Model Implementation
//!
//! Mochi is Genmo's open-source video generation model that uses
//! an asymmetric DiT architecture with factorized 3D attention.
//!
//! # Architecture Overview
//!
//! ```text
//! Input: video patches + text embeddings + timestep
//!        ↓
//! [Asymmetric DiT Blocks] - deeper video, shallower text processing
//!        ↓
//! Output: velocity prediction for flow matching
//! ```
//!
//! # Key Features
//!
//! - **Asymmetric**: Different block depths for text and video
//! - **Factorized attention**: Spatial and temporal attention separated
//! - **Flow matching**: Rectified flow objective
//! - **Open source**: Fully open weights

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

use sketchpad_core::glu::SwiGluFfn;
use sketchpad_core::layernorm::LayerNorm;
use sketchpad_core::rope::RotaryEmbedding;

/// Mochi Model Configuration
#[derive(Debug, Clone)]
pub struct MochiConfig {
    /// Number of input video channels (from 3D VAE)
    pub in_channels: usize,
    /// Spatial patch size
    pub patch_size: usize,
    /// Temporal patch size
    pub temporal_patch_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of video DiT blocks
    pub num_video_blocks: usize,
    /// Number of text DiT blocks (usually fewer)
    pub num_text_blocks: usize,
    /// Text embedding dimension (T5)
    pub text_dim: usize,
    /// Timestep embedding dimension
    pub time_embed_dim: usize,
    /// FFN intermediate size multiplier
    pub mlp_ratio: f32,
    /// Maximum spatial sequence length
    pub max_spatial_len: usize,
    /// Maximum temporal length
    pub max_temporal_len: usize,
}

impl MochiConfig {
    /// Mochi-1 configuration
    pub fn mochi_1() -> Self {
        Self {
            in_channels: 12,
            patch_size: 2,
            temporal_patch_size: 1,
            hidden_size: 3072,
            num_heads: 24,
            num_video_blocks: 48,
            num_text_blocks: 6,
            text_dim: 4096, // T5-XXL
            time_embed_dim: 256,
            mlp_ratio: 4.0,
            max_spatial_len: 4096,
            max_temporal_len: 256,
        }
    }

    /// Tiny model for testing
    pub fn tiny() -> Self {
        Self {
            in_channels: 4,
            patch_size: 2,
            temporal_patch_size: 1,
            hidden_size: 256,
            num_heads: 4,
            num_video_blocks: 4,
            num_text_blocks: 2,
            text_dim: 128,
            time_embed_dim: 64,
            mlp_ratio: 4.0,
            max_spatial_len: 256,
            max_temporal_len: 32,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    pub fn intermediate_size(&self) -> usize {
        (self.hidden_size as f32 * self.mlp_ratio) as usize
    }

    /// Initialize the model
    pub fn init<B: Backend>(&self, device: &B::Device) -> (Mochi<B>, MochiRuntime<B>) {
        // Video patch embedding
        let patch_dim =
            self.in_channels * self.patch_size * self.patch_size * self.temporal_patch_size;
        let video_embed = LinearConfig::new(patch_dim, self.hidden_size)
            .with_bias(true)
            .init(device);

        // Text projection
        let text_embed = LinearConfig::new(self.text_dim, self.hidden_size)
            .with_bias(true)
            .init(device);

        // Timestep embedding
        let time_embed = MochiTimestepEmbed {
            linear1: LinearConfig::new(self.time_embed_dim, self.hidden_size)
                .with_bias(true)
                .init(device),
            linear2: LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(true)
                .init(device),
            freqs: mochi_timestep_freqs(self.time_embed_dim, device),
        };

        // Text blocks (fewer, process text context)
        let text_blocks: Vec<MochiTextBlock<B>> = (0..self.num_text_blocks)
            .map(|_| {
                MochiTextBlockConfig::new(
                    self.hidden_size,
                    self.num_heads,
                    self.intermediate_size(),
                )
                .init(device)
            })
            .collect();

        // Video blocks (more, with factorized attention)
        let video_blocks: Vec<MochiVideoBlock<B>> = (0..self.num_video_blocks)
            .map(|_| {
                MochiVideoBlockConfig::new(
                    self.hidden_size,
                    self.num_heads,
                    self.intermediate_size(),
                )
                .init(device)
            })
            .collect();

        // Final layer
        let final_layer = MochiFinalLayer {
            norm: LayerNorm::new(self.hidden_size, device),
            proj: LinearConfig::new(
                self.hidden_size,
                self.patch_size * self.patch_size * self.temporal_patch_size * self.in_channels,
            )
            .with_bias(true)
            .init(device),
            modulation: LinearConfig::new(self.hidden_size, 2 * self.hidden_size)
                .with_bias(true)
                .init(device),
        };

        let model = Mochi {
            video_embed,
            text_embed,
            time_embed,
            text_blocks,
            video_blocks,
            final_layer,
            hidden_size: self.hidden_size,
            patch_size: self.patch_size,
            temporal_patch_size: self.temporal_patch_size,
            in_channels: self.in_channels,
        };

        let runtime = MochiRuntime {
            spatial_rope: RotaryEmbedding::new(self.head_dim(), self.max_spatial_len, device),
            temporal_rope: RotaryEmbedding::new(self.head_dim(), self.max_temporal_len, device),
        };

        (model, runtime)
    }
}

/// Timestep embedding for Mochi
#[derive(Module, Debug)]
pub struct MochiTimestepEmbed<B: Backend> {
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
    /// Precomputed sinusoidal frequencies
    pub freqs: Tensor<B, 1>,
}

/// Precompute sinusoidal frequencies for Mochi timestep embedding
pub fn mochi_timestep_freqs<B: Backend>(embed_dim: usize, device: &B::Device) -> Tensor<B, 1> {
    let half_dim = embed_dim / 2;
    let emb_scale = -(10000.0_f32.ln()) / (half_dim as f32 - 1.0);
    let freqs: Vec<f32> = (0..half_dim)
        .map(|i| (emb_scale * i as f32).exp())
        .collect();
    Tensor::<B, 1>::from_floats(freqs.as_slice(), device)
}

impl<B: Backend> MochiTimestepEmbed<B> {
    pub fn forward(&self, t: Tensor<B, 1>) -> Tensor<B, 2> {
        let [_batch] = t.dims();
        let t_expanded = t.unsqueeze_dim::<2>(1);
        let freqs_expanded = self.freqs.clone().unsqueeze_dim::<2>(0);
        let angles = t_expanded.matmul(freqs_expanded);

        let sin_emb = angles.clone().sin();
        let cos_emb = angles.cos();
        let emb = Tensor::cat(vec![sin_emb, cos_emb], 1);

        let x = self.linear1.forward(emb);
        let x = burn::tensor::activation::silu(x);
        self.linear2.forward(x)
    }

    /// Forward pass for a single scalar timestep (no tensor allocation)
    pub fn forward_scalar(&self, t: f32) -> Tensor<B, 2> {
        let angles = self.freqs.clone() * t;
        let sin_emb = angles.clone().sin();
        let cos_emb = angles.cos();
        let emb = Tensor::cat(vec![sin_emb, cos_emb], 0).unsqueeze_dim(0);
        let x = self.linear1.forward(emb);
        let x = burn::tensor::activation::silu(x);
        self.linear2.forward(x)
    }
}

/// Mochi Text Block Configuration
struct MochiTextBlockConfig {
    hidden_size: usize,
    num_heads: usize,
    intermediate_size: usize,
}

impl MochiTextBlockConfig {
    fn new(hidden_size: usize, num_heads: usize, intermediate_size: usize) -> Self {
        Self {
            hidden_size,
            num_heads,
            intermediate_size,
        }
    }

    fn init<B: Backend>(&self, device: &B::Device) -> MochiTextBlock<B> {
        use sketchpad_core::glu::SwiGluFfnConfig;

        let head_dim = self.hidden_size / self.num_heads;

        MochiTextBlock {
            norm1: LayerNorm::new(self.hidden_size, device),
            attn: MochiAttention {
                to_qkv: LinearConfig::new(self.hidden_size, 3 * self.hidden_size)
                    .with_bias(true)
                    .init(device),
                to_out: LinearConfig::new(self.hidden_size, self.hidden_size)
                    .with_bias(true)
                    .init(device),
                num_heads: self.num_heads,
                head_dim,
            },
            norm2: LayerNorm::new(self.hidden_size, device),
            ffn: SwiGluFfnConfig::new(self.hidden_size, self.intermediate_size).init(device),
        }
    }
}

/// Mochi Video Block Configuration
struct MochiVideoBlockConfig {
    hidden_size: usize,
    num_heads: usize,
    intermediate_size: usize,
}

impl MochiVideoBlockConfig {
    fn new(hidden_size: usize, num_heads: usize, intermediate_size: usize) -> Self {
        Self {
            hidden_size,
            num_heads,
            intermediate_size,
        }
    }

    fn init<B: Backend>(&self, device: &B::Device) -> MochiVideoBlock<B> {
        use sketchpad_core::glu::SwiGluFfnConfig;

        let head_dim = self.hidden_size / self.num_heads;

        MochiVideoBlock {
            norm1: LayerNorm::new(self.hidden_size, device),
            spatial_attn: MochiAttention {
                to_qkv: LinearConfig::new(self.hidden_size, 3 * self.hidden_size)
                    .with_bias(true)
                    .init(device),
                to_out: LinearConfig::new(self.hidden_size, self.hidden_size)
                    .with_bias(true)
                    .init(device),
                num_heads: self.num_heads,
                head_dim,
            },
            norm2: LayerNorm::new(self.hidden_size, device),
            temporal_attn: MochiAttention {
                to_qkv: LinearConfig::new(self.hidden_size, 3 * self.hidden_size)
                    .with_bias(true)
                    .init(device),
                to_out: LinearConfig::new(self.hidden_size, self.hidden_size)
                    .with_bias(true)
                    .init(device),
                num_heads: self.num_heads,
                head_dim,
            },
            norm3: LayerNorm::new(self.hidden_size, device),
            cross_attn: MochiCrossAttention {
                to_q: LinearConfig::new(self.hidden_size, self.hidden_size)
                    .with_bias(true)
                    .init(device),
                to_kv: LinearConfig::new(self.hidden_size, 2 * self.hidden_size)
                    .with_bias(true)
                    .init(device),
                to_out: LinearConfig::new(self.hidden_size, self.hidden_size)
                    .with_bias(true)
                    .init(device),
                num_heads: self.num_heads,
                head_dim,
            },
            norm4: LayerNorm::new(self.hidden_size, device),
            ffn: SwiGluFfnConfig::new(self.hidden_size, self.intermediate_size).init(device),
            modulation: LinearConfig::new(self.hidden_size, 8 * self.hidden_size)
                .with_bias(true)
                .init(device),
            hidden_size: self.hidden_size,
        }
    }
}

/// Mochi Attention (shared by spatial/temporal)
#[derive(Module, Debug)]
pub struct MochiAttention<B: Backend> {
    pub to_qkv: Linear<B>,
    pub to_out: Linear<B>,
    #[module(skip)]
    pub num_heads: usize,
    #[module(skip)]
    pub head_dim: usize,
}

impl<B: Backend> MochiAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>, rope: &RotaryEmbedding<B>) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden] = x.dims();

        let qkv = self.to_qkv.forward(x);
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

        // Apply RoPE
        let (q, k) = rope.forward(q, k, 0);

        let scale = (self.head_dim as f32).sqrt().recip();
        let attn = q.matmul(k.swap_dims(2, 3)) * scale;
        let attn = burn::tensor::activation::softmax(attn, 3);
        let out = attn.matmul(v);

        let out = out
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * self.head_dim]);
        self.to_out.forward(out)
    }
}

/// Mochi Cross-Attention to text
#[derive(Module, Debug)]
pub struct MochiCrossAttention<B: Backend> {
    pub to_q: Linear<B>,
    pub to_kv: Linear<B>,
    pub to_out: Linear<B>,
    #[module(skip)]
    pub num_heads: usize,
    #[module(skip)]
    pub head_dim: usize,
}

impl<B: Backend> MochiCrossAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>, context: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden] = x.dims();
        let [_, ctx_len, _] = context.dims();

        let q = self.to_q.forward(x);
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

        let scale = (self.head_dim as f32).sqrt().recip();
        let attn = q.matmul(k.swap_dims(2, 3)) * scale;
        let attn = burn::tensor::activation::softmax(attn, 3);
        let out = attn.matmul(v);

        let out = out
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * self.head_dim]);
        self.to_out.forward(out)
    }
}

/// Mochi Text Block (simpler, no modulation)
#[derive(Module, Debug)]
pub struct MochiTextBlock<B: Backend> {
    pub norm1: LayerNorm<B>,
    pub attn: MochiAttention<B>,
    pub norm2: LayerNorm<B>,
    pub ffn: SwiGluFfn<B>,
}

impl<B: Backend> MochiTextBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>, rope: &RotaryEmbedding<B>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(self.norm1.forward(x.clone()), rope);
        x.clone() + self.ffn.forward(self.norm2.forward(x))
    }
}

/// Mochi Video Block with factorized attention
#[derive(Module, Debug)]
pub struct MochiVideoBlock<B: Backend> {
    pub norm1: LayerNorm<B>,
    pub spatial_attn: MochiAttention<B>,
    pub norm2: LayerNorm<B>,
    pub temporal_attn: MochiAttention<B>,
    pub norm3: LayerNorm<B>,
    pub cross_attn: MochiCrossAttention<B>,
    pub norm4: LayerNorm<B>,
    pub ffn: SwiGluFfn<B>,
    pub modulation: Linear<B>,
    #[module(skip)]
    pub hidden_size: usize,
}

impl<B: Backend> MochiVideoBlock<B> {
    /// Forward pass with factorized spatial-temporal attention
    ///
    /// # Arguments
    /// * `x` - Video tokens [batch, T*H*W, hidden]
    /// * `cond` - Timestep conditioning [batch, hidden]
    /// * `context` - Text context [batch, text_len, hidden]
    /// * `nt, nh, nw` - Video patch dimensions
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        cond: Tensor<B, 2>,
        context: Tensor<B, 3>,
        nt: usize,
        nh: usize,
        nw: usize,
        spatial_rope: &RotaryEmbedding<B>,
        temporal_rope: &RotaryEmbedding<B>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, hidden] = x.dims();
        let spatial_len = nh * nw;
        let temporal_len = nt;

        // Get modulation (8 params: shift/scale for spatial, temporal, cross, ffn)
        let mod_params = self.modulation.forward(cond);
        let mod_params = mod_params.reshape([batch, 8, hidden]);

        let shift1 = mod_params
            .clone()
            .slice([0..batch, 0..1, 0..hidden])
            .reshape([batch, 1, hidden]);
        let scale1 = mod_params
            .clone()
            .slice([0..batch, 1..2, 0..hidden])
            .reshape([batch, 1, hidden]);
        let shift2 = mod_params
            .clone()
            .slice([0..batch, 2..3, 0..hidden])
            .reshape([batch, 1, hidden]);
        let scale2 = mod_params
            .clone()
            .slice([0..batch, 3..4, 0..hidden])
            .reshape([batch, 1, hidden]);
        let shift3 = mod_params
            .clone()
            .slice([0..batch, 4..5, 0..hidden])
            .reshape([batch, 1, hidden]);
        let scale3 = mod_params
            .clone()
            .slice([0..batch, 5..6, 0..hidden])
            .reshape([batch, 1, hidden]);
        let shift4 = mod_params
            .clone()
            .slice([0..batch, 6..7, 0..hidden])
            .reshape([batch, 1, hidden]);
        let scale4 = mod_params
            .slice([0..batch, 7..8, 0..hidden])
            .reshape([batch, 1, hidden]);

        // Spatial attention: [B, T*H*W, D] -> [B*T, H*W, D]
        let x_norm = self.norm1.forward(x.clone());
        let x_norm = (Tensor::ones_like(&scale1) + scale1) * x_norm + shift1;
        let x_spatial = x_norm.reshape([batch * temporal_len, spatial_len, hidden]);
        let spatial_out = self.spatial_attn.forward(x_spatial, spatial_rope);
        // Back to [B, T*H*W, D] for residual
        let spatial_out = spatial_out.reshape([batch, seq_len, hidden]);
        let x = x + spatial_out;

        // Temporal attention: reshape [B, T*H*W, D] -> [B, T, H*W, D] -> [B*H*W, T, D]
        let x_norm = self.norm2.forward(x.clone());
        let x_norm = (Tensor::ones_like(&scale2) + scale2) * x_norm + shift2;
        // Reshape for temporal: [B, T*H*W, D] -> [B*H*W, T, D]
        // First reshape to [B, T, H*W, D], then [B, H*W, T, D], then [B*H*W, T, D]
        let x_reshaped = x_norm.reshape([batch, temporal_len, spatial_len, hidden]);
        let x_temporal =
            x_reshaped
                .swap_dims(1, 2)
                .reshape([batch * spatial_len, temporal_len, hidden]);
        let temporal_out = self.temporal_attn.forward(x_temporal, temporal_rope);
        // Back to [B, T*H*W, D]
        let temporal_out = temporal_out
            .reshape([batch, spatial_len, temporal_len, hidden])
            .swap_dims(1, 2)
            .reshape([batch, seq_len, hidden]);
        let x = x + temporal_out;

        // Cross-attention to text
        let x_norm = self.norm3.forward(x.clone());
        let x_norm = (Tensor::ones_like(&scale3) + scale3) * x_norm + shift3;
        let x = x + self.cross_attn.forward(x_norm, context);

        // FFN
        let x_norm = self.norm4.forward(x.clone());
        let x_norm = (Tensor::ones_like(&scale4) + scale4) * x_norm + shift4;
        x + self.ffn.forward(x_norm)
    }
}

/// Mochi Final Layer
#[derive(Module, Debug)]
pub struct MochiFinalLayer<B: Backend> {
    pub norm: LayerNorm<B>,
    pub proj: Linear<B>,
    pub modulation: Linear<B>,
}

impl<B: Backend> MochiFinalLayer<B> {
    pub fn forward(&self, x: Tensor<B, 3>, cond: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch, _seq_len, hidden] = x.dims();

        let mod_params = self.modulation.forward(cond);
        let mod_params = mod_params.reshape([batch, 1, hidden * 2]);

        let shift = mod_params.clone().slice([0..batch, 0..1, 0..hidden]);
        let scale = mod_params.slice([0..batch, 0..1, hidden..(hidden * 2)]);

        let x = self.norm.forward(x);
        let x = (Tensor::ones_like(&scale) + scale) * x + shift;

        self.proj.forward(x)
    }
}

/// Mochi Model
#[derive(Module, Debug)]
pub struct Mochi<B: Backend> {
    pub video_embed: Linear<B>,
    pub text_embed: Linear<B>,
    pub time_embed: MochiTimestepEmbed<B>,
    pub text_blocks: Vec<MochiTextBlock<B>>,
    pub video_blocks: Vec<MochiVideoBlock<B>>,
    pub final_layer: MochiFinalLayer<B>,
    #[module(skip)]
    pub hidden_size: usize,
    #[module(skip)]
    pub patch_size: usize,
    #[module(skip)]
    pub temporal_patch_size: usize,
    #[module(skip)]
    pub in_channels: usize,
}

/// Runtime state for Mochi
pub struct MochiRuntime<B: Backend> {
    pub spatial_rope: RotaryEmbedding<B>,
    pub temporal_rope: RotaryEmbedding<B>,
}

/// Output from Mochi
pub struct MochiOutput<B: Backend> {
    /// Velocity prediction [batch, channels, time, height, width]
    pub velocity: Tensor<B, 5>,
}

impl<B: Backend> Mochi<B> {
    /// Patchify video (6D-safe)
    fn patchify(&self, x: Tensor<B, 5>) -> (Tensor<B, 3>, usize, usize, usize, usize) {
        let [batch, channels, time, height, width] = x.dims();
        let ps = self.patch_size;
        let pt = self.temporal_patch_size;

        assert_eq!(pt, 1, "temporal_patch_size must be 1 (NdArray 6D limit)");

        let nt = time / pt;
        let nh = height / ps;
        let nw = width / ps;

        // [B, C, T, H, W] -> [B, T, C, H, W]
        let x = x.permute([0, 2, 1, 3, 4]);
        // [B*T, C, H, W]
        let x = x.reshape([batch * time, channels, height, width]);
        // [B*T, C, nh, ps, nw, ps]
        let x = x.reshape([batch * time, channels, nh, ps, nw, ps]);
        // [B*T, nh, nw, C, ps, ps]
        let x = x.permute([0, 2, 4, 1, 3, 5]);
        // [B*T, nh*nw, C*ps*ps]
        let x = x.reshape([batch * time, nh * nw, channels * ps * ps]);
        // [B, T*nh*nw, C*ps*ps]
        let x = x.reshape([batch, time * nh * nw, channels * ps * ps]);

        // Project
        let x = self.video_embed.forward(x);

        (x, nt, nh, nw, channels)
    }

    /// Unpatchify video (6D-safe)
    fn unpatchify(
        &self,
        x: Tensor<B, 3>,
        nt: usize,
        nh: usize,
        nw: usize,
        channels: usize,
    ) -> Tensor<B, 5> {
        let [batch, _seq_len, _hidden] = x.dims();
        let ps = self.patch_size;
        let time = nt;
        let height = nh * ps;
        let width = nw * ps;

        // [B, T*nh*nw, C*ps*ps] -> [B*T, nh*nw, C*ps*ps]
        let x = x.reshape([batch * time, nh * nw, channels * ps * ps]);
        // [B*T, nh, nw, C, ps, ps]
        let x = x.reshape([batch * time, nh, nw, channels, ps, ps]);
        // [B*T, C, nh, ps, nw, ps]
        let x = x.permute([0, 3, 1, 4, 2, 5]);
        // [B*T, C, H, W]
        let x = x.reshape([batch * time, channels, height, width]);
        // [B, T, C, H, W]
        let x = x.reshape([batch, time, channels, height, width]);
        // [B, C, T, H, W]
        x.permute([0, 2, 1, 3, 4])
    }

    /// Forward pass
    pub fn forward(
        &self,
        video_latents: Tensor<B, 5>,
        timestep: f32,
        text_embeds: Tensor<B, 3>,
        runtime: &MochiRuntime<B>,
    ) -> MochiOutput<B> {
        // Patchify video
        let (x, nt, nh, nw, channels) = self.patchify(video_latents);
        let [_batch, _seq_len, _hidden] = x.dims();

        // Project text
        let mut text = self.text_embed.forward(text_embeds);

        // Timestep embedding (using scalar forward to avoid tensor allocation)
        let cond = self.time_embed.forward_scalar(timestep);

        // Process text through text blocks
        for block in &self.text_blocks {
            text = block.forward(text, &runtime.spatial_rope);
        }

        // Process video through video blocks with cross-attention to text
        let mut x = x;
        for block in &self.video_blocks {
            x = block.forward(
                x,
                cond.clone(),
                text.clone(),
                nt,
                nh,
                nw,
                &runtime.spatial_rope,
                &runtime.temporal_rope,
            );
        }

        // Final layer
        let out = self.final_layer.forward(x, cond);

        // Unpatchify
        let velocity = self.unpatchify(out, nt, nh, nw, channels);

        MochiOutput { velocity }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_mochi_config() {
        let config = MochiConfig::mochi_1();
        assert_eq!(config.hidden_size, 3072);
        assert_eq!(config.num_video_blocks, 48);
        assert_eq!(config.num_text_blocks, 6);
    }

    #[test]
    fn test_mochi_timestep_embed() {
        let device = Default::default();
        let embed = MochiTimestepEmbed {
            linear1: LinearConfig::new(64, 256).with_bias(true).init(&device),
            linear2: LinearConfig::new(256, 256).with_bias(true).init(&device),
            freqs: mochi_timestep_freqs(64, &device),
        };

        let t = Tensor::<TestBackend, 1>::from_floats([0.5], &device);
        let out = embed.forward(t);
        assert_eq!(out.dims(), [1, 256]);
    }

    #[test]
    fn test_mochi_attention() {
        let device = Default::default();
        let attn = MochiAttention {
            to_qkv: LinearConfig::new(256, 768).with_bias(true).init(&device),
            to_out: LinearConfig::new(256, 256).with_bias(true).init(&device),
            num_heads: 4,
            head_dim: 64,
        };
        let rope = RotaryEmbedding::new(64, 256, &device);

        let x = Tensor::<TestBackend, 3>::zeros([2, 16, 256], &device);
        let out = attn.forward(x, &rope);
        assert_eq!(out.dims(), [2, 16, 256]);
    }

    #[test]
    fn test_mochi_text_block() {
        let device = Default::default();
        let block = MochiTextBlockConfig::new(256, 4, 512).init::<TestBackend>(&device);
        let rope = RotaryEmbedding::new(64, 256, &device);

        let x = Tensor::zeros([2, 8, 256], &device);
        let out = block.forward(x, &rope);
        assert_eq!(out.dims(), [2, 8, 256]);
    }

    #[test]
    fn test_mochi_video_block() {
        let device = Default::default();
        let block = MochiVideoBlockConfig::new(256, 4, 512).init::<TestBackend>(&device);
        let spatial_rope = RotaryEmbedding::new(64, 256, &device);
        let temporal_rope = RotaryEmbedding::new(64, 32, &device);

        // 4 temporal * 16 spatial = 64 tokens
        let x = Tensor::zeros([2, 64, 256], &device);
        let cond = Tensor::zeros([2, 256], &device);
        let ctx = Tensor::zeros([2, 8, 256], &device);

        let out = block.forward(x, cond, ctx, 4, 4, 4, &spatial_rope, &temporal_rope);
        assert_eq!(out.dims(), [2, 64, 256]);
    }

    #[test]
    fn test_mochi_tiny_forward() {
        let device = Default::default();
        let config = MochiConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        // [batch=1, channels=4, time=4, height=8, width=8]
        let video = Tensor::zeros([1, 4, 4, 8, 8], &device);
        let text = Tensor::zeros([1, 4, 128], &device);

        let output = model.forward(video, 0.5, text, &runtime);

        assert_eq!(output.velocity.dims(), [1, 4, 4, 8, 8]);
    }
}
