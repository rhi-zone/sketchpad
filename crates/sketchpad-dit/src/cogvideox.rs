//! CogVideoX Model Implementation
//!
//! CogVideoX is a DiT-based video generation model that processes
//! spatiotemporal tokens with 3D attention.
//!
//! # Architecture Overview
//!
//! ```text
//! Input: video patches (from 3D VAE) + text embeddings + timestep
//!        ↓
//! [3D Patch Embedding] - spatial + temporal patchification
//!        ↓
//! [DiT Blocks with 3D Attention] - factorized spatial-temporal attention
//!        ↓
//! Output: velocity prediction
//! ```
//!
//! # Key Features
//!
//! - **3D VAE**: Compresses video to latent space with temporal compression
//! - **Factorized Attention**: Separate spatial and temporal attention for efficiency
//! - **Expert Parallel**: Some variants use MoE for capacity
//!
//! # Variants
//!
//! - **CogVideoX-2B**: Base model
//! - **CogVideoX-5B**: Larger capacity

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

use sketchpad_core::glu::SwiGluFfn;
use sketchpad_core::layernorm::LayerNorm;
use sketchpad_core::rope::RotaryEmbedding;

/// CogVideoX Model Configuration
#[derive(Debug, Clone)]
pub struct CogVideoXConfig {
    /// Number of latent channels from 3D VAE (usually 16)
    pub in_channels: usize,
    /// Spatial patch size
    pub patch_size: usize,
    /// Temporal patch size (frames grouped together)
    pub temporal_patch_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of DiT blocks
    pub num_blocks: usize,
    /// Text embedding dimension
    pub text_dim: usize,
    /// Timestep embedding dimension
    pub time_embed_dim: usize,
    /// Maximum spatial sequence length
    pub max_spatial_len: usize,
    /// Maximum temporal length (frames)
    pub max_temporal_len: usize,
    /// FFN multiplier
    pub mlp_ratio: f32,
    /// Whether to use factorized (spatial then temporal) attention
    pub factorized_attention: bool,
}

impl CogVideoXConfig {
    /// CogVideoX-2B configuration
    pub fn cogvideox_2b() -> Self {
        Self {
            in_channels: 16,
            patch_size: 2,
            temporal_patch_size: 1,
            hidden_size: 1920,
            num_heads: 30,
            num_blocks: 30,
            text_dim: 4096, // T5-XXL
            time_embed_dim: 512,
            max_spatial_len: 4096,
            max_temporal_len: 256,
            mlp_ratio: 4.0,
            factorized_attention: true,
        }
    }

    /// CogVideoX-5B configuration
    pub fn cogvideox_5b() -> Self {
        Self {
            in_channels: 16,
            patch_size: 2,
            temporal_patch_size: 1,
            hidden_size: 3072,
            num_heads: 48,
            num_blocks: 42,
            text_dim: 4096,
            time_embed_dim: 512,
            max_spatial_len: 4096,
            max_temporal_len: 256,
            mlp_ratio: 4.0,
            factorized_attention: true,
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
            num_blocks: 4,
            text_dim: 128,
            time_embed_dim: 64,
            max_spatial_len: 256,
            max_temporal_len: 32,
            mlp_ratio: 4.0,
            factorized_attention: true,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    pub fn intermediate_size(&self) -> usize {
        (self.hidden_size as f32 * self.mlp_ratio) as usize
    }

    /// Initialize the model
    pub fn init<B: Backend>(&self, device: &B::Device) -> (CogVideoX<B>, CogVideoXRuntime<B>) {
        // 3D Patch embedding
        let patch_embed = VideoPatchEmbed {
            proj: LinearConfig::new(
                self.in_channels * self.patch_size * self.patch_size * self.temporal_patch_size,
                self.hidden_size,
            )
            .with_bias(true)
            .init(device),
            patch_size: self.patch_size,
            temporal_patch_size: self.temporal_patch_size,
            in_channels: self.in_channels,
        };

        // Text embedding projection
        let text_embed = LinearConfig::new(self.text_dim, self.hidden_size)
            .with_bias(true)
            .init(device);

        // Timestep embedding
        let time_embed = CogVideoTimestepEmbed {
            linear1: LinearConfig::new(self.time_embed_dim, self.hidden_size)
                .with_bias(true)
                .init(device),
            linear2: LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(true)
                .init(device),
            freqs: cogvideo_timestep_freqs(self.time_embed_dim, device),
        };

        // DiT blocks
        let blocks: Vec<CogVideoBlock<B>> = (0..self.num_blocks)
            .map(|_| {
                CogVideoBlockConfig::new(
                    self.hidden_size,
                    self.num_heads,
                    self.intermediate_size(),
                    self.factorized_attention,
                )
                .init(device)
            })
            .collect();

        // Final layer
        let final_layer = CogVideoFinalLayer {
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

        let model = CogVideoX {
            patch_embed,
            text_embed,
            time_embed,
            blocks,
            final_layer,
            hidden_size: self.hidden_size,
            patch_size: self.patch_size,
            temporal_patch_size: self.temporal_patch_size,
            in_channels: self.in_channels,
        };

        let runtime = CogVideoXRuntime {
            spatial_rope: RotaryEmbedding::new(self.head_dim(), self.max_spatial_len, device),
            temporal_rope: RotaryEmbedding::new(self.head_dim(), self.max_temporal_len, device),
        };

        (model, runtime)
    }
}

/// 3D Video Patch Embedding
#[derive(Module, Debug)]
pub struct VideoPatchEmbed<B: Backend> {
    pub proj: Linear<B>,
    #[module(skip)]
    pub patch_size: usize,
    #[module(skip)]
    pub temporal_patch_size: usize,
    #[module(skip)]
    pub in_channels: usize,
}

impl<B: Backend> VideoPatchEmbed<B> {
    /// Patchify video and project
    ///
    /// Input: [batch, channels, time, height, width]
    /// Output: [batch, num_patches, hidden_size]
    ///
    /// Note: Uses 6D-safe implementation by processing spatial dims with batch merged.
    /// Currently requires temporal_patch_size=1 (which all CogVideoX configs use).
    pub fn forward(&self, x: Tensor<B, 5>) -> (Tensor<B, 3>, VideoShape) {
        let [batch, channels, time, height, width] = x.dims();

        let pt = self.temporal_patch_size;
        let ps = self.patch_size;

        assert_eq!(pt, 1, "temporal_patch_size must be 1 (NdArray 6D limit)");

        let nt = time / pt;
        let nh = height / ps;
        let nw = width / ps;

        // Spatial patchification staying within 6D:
        // [B, C, T, H, W] -> [B, T, C, H, W]
        let x = x.permute([0, 2, 1, 3, 4]);
        // [B*T, C, H, W]
        let x = x.reshape([batch * time, channels, height, width]);
        // [B*T, C, nh, ps, nw, ps] (6D)
        let x = x.reshape([batch * time, channels, nh, ps, nw, ps]);
        // [B*T, nh, nw, C, ps, ps]
        let x = x.permute([0, 2, 4, 1, 3, 5]);
        // [B*T, nh*nw, C*ps*ps]
        let x = x.reshape([batch * time, nh * nw, channels * ps * ps]);
        // [B, T*nh*nw, C*ps*ps]
        let x = x.reshape([batch, time * nh * nw, channels * ps * ps]);

        let shape = VideoShape {
            batch,
            channels,
            time,
            height,
            width,
            nt,
            nh,
            nw,
        };

        (self.proj.forward(x), shape)
    }
}

/// Video shape for unpatchifying
#[derive(Debug, Clone, Copy)]
pub struct VideoShape {
    pub batch: usize,
    pub channels: usize,
    pub time: usize,
    pub height: usize,
    pub width: usize,
    pub nt: usize, // Temporal patches
    pub nh: usize, // Height patches
    pub nw: usize, // Width patches
}

/// Unpatchify video (6D-safe implementation)
///
/// Reverses the patchification process. Requires temporal_patch_size=1.
pub fn unpatchify_video<B: Backend>(
    x: Tensor<B, 3>,
    patch_size: usize,
    temporal_patch_size: usize,
    shape: &VideoShape,
) -> Tensor<B, 5> {
    let VideoShape {
        batch,
        channels,
        time,
        height,
        width,
        nt: _,
        nh,
        nw,
    } = *shape;
    let ps = patch_size;
    let pt = temporal_patch_size;

    assert_eq!(pt, 1, "temporal_patch_size must be 1 (NdArray 6D limit)");

    // x: [B, T*nh*nw, C*ps*ps]
    // Reverse the patchification staying within 6D:

    // [B*T, nh*nw, C*ps*ps]
    let x = x.reshape([batch * time, nh * nw, channels * ps * ps]);
    // [B*T, nh, nw, C, ps, ps] (6D)
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

/// Timestep embedding for CogVideoX
#[derive(Module, Debug)]
pub struct CogVideoTimestepEmbed<B: Backend> {
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
    /// Precomputed sinusoidal frequencies
    pub freqs: Tensor<B, 1>,
}

/// Precompute sinusoidal frequencies for CogVideoX timestep embedding
pub fn cogvideo_timestep_freqs<B: Backend>(embed_dim: usize, device: &B::Device) -> Tensor<B, 1> {
    let half_dim = embed_dim / 2;
    let emb_scale = -(2.0_f32.ln()) / (half_dim as f32 - 1.0);
    let freqs: Vec<f32> = (0..half_dim)
        .map(|i| (emb_scale * i as f32).exp())
        .collect();
    Tensor::<B, 1>::from_floats(freqs.as_slice(), device)
}

impl<B: Backend> CogVideoTimestepEmbed<B> {
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

/// CogVideoX Block Configuration
struct CogVideoBlockConfig {
    hidden_size: usize,
    num_heads: usize,
    intermediate_size: usize,
    factorized: bool,
}

impl CogVideoBlockConfig {
    fn new(
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
        factorized: bool,
    ) -> Self {
        Self {
            hidden_size,
            num_heads,
            intermediate_size,
            factorized,
        }
    }

    fn init<B: Backend>(&self, device: &B::Device) -> CogVideoBlock<B> {
        use sketchpad_core::glu::SwiGluFfnConfig;

        let head_dim = self.hidden_size / self.num_heads;

        CogVideoBlock {
            norm1: LayerNorm::new(self.hidden_size, device),
            spatial_attn: CogVideoAttention {
                to_qkv: LinearConfig::new(self.hidden_size, 3 * self.hidden_size)
                    .with_bias(true)
                    .init(device),
                to_out: LinearConfig::new(self.hidden_size, self.hidden_size)
                    .with_bias(true)
                    .init(device),
                num_heads: self.num_heads,
                head_dim,
            },
            // Always create temporal attention (used only when factorized=true)
            temporal_attn: CogVideoAttention {
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
            modulation: LinearConfig::new(self.hidden_size, 6 * self.hidden_size)
                .with_bias(true)
                .init(device),
            hidden_size: self.hidden_size,
            factorized: self.factorized,
        }
    }
}

/// CogVideoX Attention
#[derive(Module, Debug)]
pub struct CogVideoAttention<B: Backend> {
    pub to_qkv: Linear<B>,
    pub to_out: Linear<B>,
    #[module(skip)]
    pub num_heads: usize,
    #[module(skip)]
    pub head_dim: usize,
}

impl<B: Backend> CogVideoAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>, rope: &RotaryEmbedding<B>) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden] = x.dims();

        // QKV projection
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
            .reshape([batch, seq_len, self.num_heads, self.head_dim]);
        let k = qkv
            .clone()
            .slice([
                0..batch,
                0..seq_len,
                1..2,
                0..self.num_heads,
                0..self.head_dim,
            ])
            .reshape([batch, seq_len, self.num_heads, self.head_dim]);
        let v = qkv
            .slice([
                0..batch,
                0..seq_len,
                2..3,
                0..self.num_heads,
                0..self.head_dim,
            ])
            .reshape([batch, seq_len, self.num_heads, self.head_dim]);

        // [batch, heads, seq, dim]
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        // Apply RoPE
        let (q, k) = rope.forward(q, k, 0);

        // Attention
        let scale = (self.head_dim as f32).sqrt().recip();
        let attn = q.matmul(k.swap_dims(2, 3)) * scale;
        let attn = burn::tensor::activation::softmax(attn, 3);
        let out = attn.matmul(v);

        // Reshape and project
        let out = out
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * self.head_dim]);
        self.to_out.forward(out)
    }
}

/// CogVideoX DiT Block
#[derive(Module, Debug)]
pub struct CogVideoBlock<B: Backend> {
    pub norm1: LayerNorm<B>,
    pub spatial_attn: CogVideoAttention<B>,
    pub temporal_attn: CogVideoAttention<B>,
    pub norm2: LayerNorm<B>,
    pub ffn: SwiGluFfn<B>,
    pub modulation: Linear<B>,
    #[module(skip)]
    pub hidden_size: usize,
    #[module(skip)]
    pub factorized: bool,
}

impl<B: Backend> CogVideoBlock<B> {
    /// Forward pass
    ///
    /// Handles text + video tokens together. For factorized attention,
    /// text tokens use full attention while video uses spatial-then-temporal.
    ///
    /// # Arguments
    /// * `x` - Combined sequence [B, text_len + T*H*W, D]
    /// * `text_len` - Number of text tokens to skip for video reshaping
    /// * `nt, nh, nw` - Video patch dimensions
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        cond: Tensor<B, 2>,
        text_len: usize,
        nt: usize,
        nh: usize,
        nw: usize,
        spatial_rope: &RotaryEmbedding<B>,
        temporal_rope: &RotaryEmbedding<B>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, hidden] = x.dims();
        let video_len = nt * nh * nw;

        // Get modulation (shift, scale, gate x 2)
        let mod_params = self.modulation.forward(cond);
        let mod_params = mod_params.reshape([batch, 6, hidden]);

        let shift1 = mod_params
            .clone()
            .slice([0..batch, 0..1, 0..hidden])
            .reshape([batch, 1, hidden]);
        let scale1 = mod_params
            .clone()
            .slice([0..batch, 1..2, 0..hidden])
            .reshape([batch, 1, hidden]);
        let gate1 = mod_params
            .clone()
            .slice([0..batch, 2..3, 0..hidden])
            .reshape([batch, 1, hidden]);
        let shift2 = mod_params
            .clone()
            .slice([0..batch, 3..4, 0..hidden])
            .reshape([batch, 1, hidden]);
        let scale2 = mod_params
            .clone()
            .slice([0..batch, 4..5, 0..hidden])
            .reshape([batch, 1, hidden]);
        let gate2 = mod_params
            .slice([0..batch, 5..6, 0..hidden])
            .reshape([batch, 1, hidden]);

        // Modulated norm for attention
        let x_norm = self.norm1.forward(x.clone());
        let x_norm = (Tensor::ones_like(&scale1) + scale1) * x_norm + shift1;

        // Attention
        let attn_out = if self.factorized && text_len == 0 {
            // Pure video: factorized spatial-temporal attention
            let spatial_len = nh * nw;
            let temporal_len = nt;

            // Spatial attention: [B, T*H*W, D] → [B*T, H*W, D]
            let x_spatial = x_norm
                .clone()
                .reshape([batch * temporal_len, spatial_len, hidden]);
            let spatial_out = self.spatial_attn.forward(x_spatial, spatial_rope);
            let spatial_out = spatial_out.reshape([batch, temporal_len, spatial_len, hidden]);

            // Temporal attention: [B, T, H*W, D] → permute → [B*H*W, T, D]
            let x_temporal = spatial_out.permute([0, 2, 1, 3]); // [B, H*W, T, D]
            let x_temporal = x_temporal.reshape([batch * spatial_len, temporal_len, hidden]);
            let temporal_out = self.temporal_attn.forward(x_temporal, temporal_rope);

            // Reshape back: [B*H*W, T, D] → [B, H*W, T, D] → [B, T*H*W, D]
            let temporal_out = temporal_out.reshape([batch, spatial_len, temporal_len, hidden]);
            let temporal_out = temporal_out.permute([0, 2, 1, 3]); // [B, T, H*W, D]
            temporal_out.reshape([batch, seq_len, hidden])
        } else if self.factorized {
            // Text + video: split, process video with factorized attention, recombine
            let text_x = x_norm.clone().slice([0..batch, 0..text_len, 0..hidden]);
            let video_x = x_norm.slice([0..batch, text_len..seq_len, 0..hidden]);

            let spatial_len = nh * nw;
            let temporal_len = nt;

            // Spatial attention on video
            let x_spatial = video_x.reshape([batch * temporal_len, spatial_len, hidden]);
            let spatial_out = self.spatial_attn.forward(x_spatial, spatial_rope);
            let spatial_out = spatial_out.reshape([batch, temporal_len, spatial_len, hidden]);

            // Temporal attention on video
            let x_temporal = spatial_out.permute([0, 2, 1, 3]);
            let x_temporal = x_temporal.reshape([batch * spatial_len, temporal_len, hidden]);
            let temporal_out = self.temporal_attn.forward(x_temporal, temporal_rope);
            let temporal_out = temporal_out.reshape([batch, spatial_len, temporal_len, hidden]);
            let video_out = temporal_out
                .permute([0, 2, 1, 3])
                .reshape([batch, video_len, hidden]);

            // Text uses spatial attention only (full sequence attention)
            let text_out = self.spatial_attn.forward(text_x, spatial_rope);

            Tensor::cat(vec![text_out, video_out], 1)
        } else {
            // Joint attention on full sequence
            self.spatial_attn.forward(x_norm, spatial_rope)
        };

        let x = x + gate1 * attn_out;

        // FFN
        let x_norm = self.norm2.forward(x.clone());
        let x_norm = (Tensor::ones_like(&scale2) + scale2) * x_norm + shift2;
        let ffn_out = self.ffn.forward(x_norm);

        x + gate2 * ffn_out
    }
}

/// CogVideoX Final Layer
#[derive(Module, Debug)]
pub struct CogVideoFinalLayer<B: Backend> {
    pub norm: LayerNorm<B>,
    pub proj: Linear<B>,
    pub modulation: Linear<B>,
}

impl<B: Backend> CogVideoFinalLayer<B> {
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

/// CogVideoX Model
#[derive(Module, Debug)]
pub struct CogVideoX<B: Backend> {
    pub patch_embed: VideoPatchEmbed<B>,
    pub text_embed: Linear<B>,
    pub time_embed: CogVideoTimestepEmbed<B>,
    pub blocks: Vec<CogVideoBlock<B>>,
    pub final_layer: CogVideoFinalLayer<B>,
    #[module(skip)]
    pub hidden_size: usize,
    #[module(skip)]
    pub patch_size: usize,
    #[module(skip)]
    pub temporal_patch_size: usize,
    #[module(skip)]
    pub in_channels: usize,
}

/// Runtime state for CogVideoX
pub struct CogVideoXRuntime<B: Backend> {
    pub spatial_rope: RotaryEmbedding<B>,
    pub temporal_rope: RotaryEmbedding<B>,
}

/// Output from CogVideoX
pub struct CogVideoXOutput<B: Backend> {
    /// Velocity prediction [batch, channels, time, height, width]
    pub velocity: Tensor<B, 5>,
}

impl<B: Backend> CogVideoX<B> {
    /// Forward pass
    ///
    /// # Arguments
    ///
    /// * `video_latents` - Noisy video latents [batch, channels, time, height, width]
    /// * `timestep` - Diffusion timestep
    /// * `text_embeds` - Text embeddings [batch, seq_len, text_dim]
    /// * `runtime` - Runtime state
    pub fn forward(
        &self,
        video_latents: Tensor<B, 5>,
        timestep: f32,
        text_embeds: Tensor<B, 3>,
        runtime: &CogVideoXRuntime<B>,
    ) -> CogVideoXOutput<B> {
        // Patchify video
        let (x, shape) = self.patch_embed.forward(video_latents);
        let [batch, _num_patches, _hidden] = x.dims();

        // Project text
        let text = self.text_embed.forward(text_embeds);
        let [_, text_len, _] = text.dims();

        // Concatenate text and video tokens
        let x = Tensor::cat(vec![text, x], 1);

        // Timestep embedding (using scalar forward to avoid tensor allocation)
        let cond = self.time_embed.forward_scalar(timestep);

        // DiT blocks
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(
                x,
                cond.clone(),
                text_len,
                shape.nt,
                shape.nh,
                shape.nw,
                &runtime.spatial_rope,
                &runtime.temporal_rope,
            );
        }

        // Remove text tokens and apply final layer
        let video_tokens = x.slice([
            0..batch,
            text_len..(text_len + shape.nt * shape.nh * shape.nw),
            0..self.hidden_size,
        ]);
        let out = self.final_layer.forward(video_tokens, cond);

        // Unpatchify
        let velocity = unpatchify_video(out, self.patch_size, self.temporal_patch_size, &shape);

        CogVideoXOutput { velocity }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_cogvideox_config() {
        let config_2b = CogVideoXConfig::cogvideox_2b();
        assert_eq!(config_2b.hidden_size, 1920);
        assert_eq!(config_2b.num_blocks, 30);

        let config_5b = CogVideoXConfig::cogvideox_5b();
        assert_eq!(config_5b.hidden_size, 3072);
    }

    #[test]
    fn test_video_patch_embed() {
        let device = Default::default();
        let embed = VideoPatchEmbed {
            proj: LinearConfig::new(4 * 2 * 2, 256)
                .with_bias(true)
                .init(&device),
            patch_size: 2,
            temporal_patch_size: 1,
            in_channels: 4,
        };

        // [batch=1, channels=4, time=4, height=8, width=8]
        let video = Tensor::<TestBackend, 5>::zeros([1, 4, 4, 8, 8], &device);
        let (patches, shape) = embed.forward(video);

        // 4 temporal * (8/2 * 8/2) spatial = 4 * 16 = 64 patches
        assert_eq!(patches.dims(), [1, 64, 256]);
        assert_eq!(shape.nt, 4);
        assert_eq!(shape.nh, 4);
        assert_eq!(shape.nw, 4);
    }

    #[test]
    fn test_unpatchify_video() {
        let device = Default::default();

        let shape = VideoShape {
            batch: 1,
            channels: 4,
            time: 4,
            height: 8,
            width: 8,
            nt: 4,
            nh: 4,
            nw: 4,
        };

        // 64 patches, each projecting to 4*1*2*2 = 16
        let x = Tensor::<TestBackend, 3>::zeros([1, 64, 16], &device);
        let video = unpatchify_video(x, 2, 1, &shape);

        assert_eq!(video.dims(), [1, 4, 4, 8, 8]);
    }

    #[test]
    fn test_cogvideox_tiny_forward() {
        let device = Default::default();
        let config = CogVideoXConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        // [batch=1, channels=4, time=4, height=8, width=8]
        let video = Tensor::zeros([1, 4, 4, 8, 8], &device);
        let text = Tensor::zeros([1, 4, 128], &device); // 4 text tokens

        let output = model.forward(video, 0.5, text, &runtime);

        assert_eq!(output.velocity.dims(), [1, 4, 4, 8, 8]);
    }

    #[test]
    fn test_cogvideo_block() {
        let device = Default::default();
        let block = CogVideoBlockConfig::new(256, 4, 512, true).init::<TestBackend>(&device);

        let spatial_rope = RotaryEmbedding::new(64, 256, &device);
        let temporal_rope = RotaryEmbedding::new(64, 32, &device);

        // 4 temporal * 16 spatial = 64 tokens, no text
        let x = Tensor::zeros([2, 64, 256], &device);
        let cond = Tensor::zeros([2, 256], &device);

        let out = block.forward(x, cond, 0, 4, 4, 4, &spatial_rope, &temporal_rope);
        assert_eq!(out.dims(), [2, 64, 256]);
    }

    #[test]
    fn test_cogvideo_attention() {
        let device = Default::default();
        let attn = CogVideoAttention {
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
}
