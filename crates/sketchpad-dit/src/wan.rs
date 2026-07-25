//! Wan 2.x Video Model Implementation
//!
//! Wan 2.x is Alibaba's state-of-the-art video generation model that uses
//! DiT + MoE (Mixture of Experts) with a 3D VAE for efficient video compression.
//!
//! # Architecture Overview
//!
//! ```text
//! Input: 3D VAE latents + text embeddings + timestep
//!        ↓
//! [DiT + MoE Blocks with Factorized Attention]
//!        ↓
//! Output: velocity prediction for flow matching
//! ```
//!
//! # Key Features
//!
//! - **DiT + MoE**: Sparse mixture of experts for efficiency
//! - **14B active params**: Large model with ~80B total params
//! - **3D VAE**: Temporal compression of video latents
//! - **Factorized attention**: Spatial and temporal attention
//! - **Flow matching**: Rectified flow objective

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use sketchpad_core::layernorm::LayerNorm;
use sketchpad_core::rope::RotaryEmbedding;

/// Wan 2.x Model Configuration
#[derive(Debug, Clone)]
pub struct WanConfig {
    /// Number of input channels (from 3D VAE)
    pub in_channels: usize,
    /// Spatial patch size
    pub patch_size: usize,
    /// Temporal patch size
    pub temporal_patch_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of DiT blocks
    pub num_blocks: usize,
    /// Number of experts per MoE layer
    pub num_experts: usize,
    /// Number of experts to route to per token
    pub top_k: usize,
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

impl WanConfig {
    /// Wan 2.x base configuration (14B active params)
    pub fn base() -> Self {
        Self {
            in_channels: 32, // From 3D VAE
            patch_size: 2,
            temporal_patch_size: 1,
            hidden_size: 3584,
            num_heads: 28,
            num_blocks: 32,
            num_experts: 8,
            top_k: 2,
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
            num_blocks: 4,
            num_experts: 4,
            top_k: 2,
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
    pub fn init<B: Backend>(&self, device: &B::Device) -> (Wan<B>, WanRuntime<B>) {
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
        let time_embed = WanTimestepEmbed {
            linear1: LinearConfig::new(self.time_embed_dim, self.hidden_size)
                .with_bias(true)
                .init(device),
            linear2: LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(true)
                .init(device),
            freqs: wan_timestep_freqs(self.time_embed_dim, device),
        };

        // DiT + MoE blocks
        let blocks: Vec<WanBlock<B>> = (0..self.num_blocks)
            .map(|_| {
                WanBlockConfig::new(
                    self.hidden_size,
                    self.num_heads,
                    self.intermediate_size(),
                    self.num_experts,
                    self.top_k,
                )
                .init(device)
            })
            .collect();

        // Final layer
        let final_layer = WanFinalLayer {
            norm: LayerNorm::new(self.hidden_size, device),
            proj: LinearConfig::new(self.hidden_size, patch_dim)
                .with_bias(true)
                .init(device),
            modulation: LinearConfig::new(self.hidden_size, 2 * self.hidden_size)
                .with_bias(true)
                .init(device),
        };

        let model = Wan {
            video_embed,
            text_embed,
            time_embed,
            blocks,
            final_layer,
            hidden_size: self.hidden_size,
            patch_size: self.patch_size,
            temporal_patch_size: self.temporal_patch_size,
            in_channels: self.in_channels,
        };

        let runtime = WanRuntime {
            spatial_rope: RotaryEmbedding::new(self.head_dim(), self.max_spatial_len, device),
            temporal_rope: RotaryEmbedding::new(self.head_dim(), self.max_temporal_len, device),
        };

        (model, runtime)
    }
}

/// Timestep embedding for Wan
#[derive(Module, Debug)]
pub struct WanTimestepEmbed<B: Backend> {
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
    /// Precomputed sinusoidal frequencies
    pub freqs: Tensor<B, 1>,
}

/// Precompute sinusoidal frequencies for Wan timestep embedding
pub fn wan_timestep_freqs<B: Backend>(embed_dim: usize, device: &B::Device) -> Tensor<B, 1> {
    let half_dim = embed_dim / 2;
    let emb_scale = -(10000.0_f32.ln()) / (half_dim as f32 - 1.0);
    let freqs: Vec<f32> = (0..half_dim)
        .map(|i| (emb_scale * i as f32).exp())
        .collect();
    Tensor::<B, 1>::from_floats(freqs.as_slice(), device)
}

impl<B: Backend> WanTimestepEmbed<B> {
    pub fn forward(&self, t: Tensor<B, 1>) -> Tensor<B, 2> {
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

/// Wan Block Configuration
struct WanBlockConfig {
    hidden_size: usize,
    num_heads: usize,
    intermediate_size: usize,
    num_experts: usize,
    top_k: usize,
}

impl WanBlockConfig {
    fn new(
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
        num_experts: usize,
        top_k: usize,
    ) -> Self {
        Self {
            hidden_size,
            num_heads,
            intermediate_size,
            num_experts,
            top_k,
        }
    }

    fn init<B: Backend>(&self, device: &B::Device) -> WanBlock<B> {
        let head_dim = self.hidden_size / self.num_heads;

        // Create expert FFNs
        let experts: Vec<WanExpert<B>> = (0..self.num_experts)
            .map(|_| WanExpert {
                up: LinearConfig::new(self.hidden_size, self.intermediate_size)
                    .with_bias(false)
                    .init(device),
                gate: LinearConfig::new(self.hidden_size, self.intermediate_size)
                    .with_bias(false)
                    .init(device),
                down: LinearConfig::new(self.intermediate_size, self.hidden_size)
                    .with_bias(false)
                    .init(device),
            })
            .collect();

        WanBlock {
            norm1: LayerNorm::new(self.hidden_size, device),
            spatial_attn: WanAttention {
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
            temporal_attn: WanAttention {
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
            cross_attn: WanCrossAttention {
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
            // MoE components
            router: LinearConfig::new(self.hidden_size, self.num_experts)
                .with_bias(false)
                .init(device),
            experts,
            num_experts: self.num_experts,
            top_k: self.top_k,
            // Modulation
            modulation: LinearConfig::new(self.hidden_size, 8 * self.hidden_size)
                .with_bias(true)
                .init(device),
            hidden_size: self.hidden_size,
        }
    }
}

/// Wan Expert FFN (SwiGLU)
#[derive(Module, Debug)]
pub struct WanExpert<B: Backend> {
    pub up: Linear<B>,
    pub gate: Linear<B>,
    pub down: Linear<B>,
}

impl<B: Backend> WanExpert<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = self.gate.forward(x.clone());
        let gate = burn::tensor::activation::silu(gate);
        let up = self.up.forward(x);
        self.down.forward(gate * up)
    }
}

/// Wan Self-Attention with RoPE
#[derive(Module, Debug)]
pub struct WanAttention<B: Backend> {
    pub to_qkv: Linear<B>,
    pub to_out: Linear<B>,
    #[module(skip)]
    pub num_heads: usize,
    #[module(skip)]
    pub head_dim: usize,
}

impl<B: Backend> WanAttention<B> {
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

/// Wan Cross-Attention to text
#[derive(Module, Debug)]
pub struct WanCrossAttention<B: Backend> {
    pub to_q: Linear<B>,
    pub to_kv: Linear<B>,
    pub to_out: Linear<B>,
    #[module(skip)]
    pub num_heads: usize,
    #[module(skip)]
    pub head_dim: usize,
}

impl<B: Backend> WanCrossAttention<B> {
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

/// Wan DiT + MoE Block
#[derive(Module, Debug)]
pub struct WanBlock<B: Backend> {
    pub norm1: LayerNorm<B>,
    pub spatial_attn: WanAttention<B>,
    pub norm2: LayerNorm<B>,
    pub temporal_attn: WanAttention<B>,
    pub norm3: LayerNorm<B>,
    pub cross_attn: WanCrossAttention<B>,
    pub norm4: LayerNorm<B>,
    // MoE components
    pub router: Linear<B>,
    pub experts: Vec<WanExpert<B>>,
    #[module(skip)]
    pub num_experts: usize,
    #[module(skip)]
    pub top_k: usize,
    // Modulation
    pub modulation: Linear<B>,
    #[module(skip)]
    pub hidden_size: usize,
}

impl<B: Backend> WanBlock<B> {
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

        // Get modulation
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
        let spatial_out = spatial_out.reshape([batch, seq_len, hidden]);
        let x = x + spatial_out;

        // Temporal attention: [B, T*H*W, D] -> [B*H*W, T, D]
        let x_norm = self.norm2.forward(x.clone());
        let x_norm = (Tensor::ones_like(&scale2) + scale2) * x_norm + shift2;
        let x_reshaped = x_norm.reshape([batch, temporal_len, spatial_len, hidden]);
        let x_temporal =
            x_reshaped
                .swap_dims(1, 2)
                .reshape([batch * spatial_len, temporal_len, hidden]);
        let temporal_out = self.temporal_attn.forward(x_temporal, temporal_rope);
        let temporal_out = temporal_out
            .reshape([batch, spatial_len, temporal_len, hidden])
            .swap_dims(1, 2)
            .reshape([batch, seq_len, hidden]);
        let x = x + temporal_out;

        // Cross-attention to text
        let x_norm = self.norm3.forward(x.clone());
        let x_norm = (Tensor::ones_like(&scale3) + scale3) * x_norm + shift3;
        let x = x + self.cross_attn.forward(x_norm, context);

        // MoE FFN
        let x_norm = self.norm4.forward(x.clone());
        let x_norm = (Tensor::ones_like(&scale4) + scale4) * x_norm + shift4;
        let moe_out = self.forward_moe(x_norm);
        x + moe_out
    }

    /// Forward through MoE layer
    fn forward_moe(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, hidden] = x.dims();
        let device = x.device();

        // Compute router logits
        let router_logits = self.router.forward(x.clone()); // [B, seq, num_experts]

        // Get top-k experts
        // For simplicity, we'll use a soft routing approach
        let router_probs = burn::tensor::activation::softmax(router_logits, 2);

        // Sum contributions from all experts weighted by router probs
        // This is a dense MoE approximation for testing (real impl would use sparse routing)
        let mut out = Tensor::zeros([batch, seq_len, hidden], &device);
        for (i, expert) in self.experts.iter().enumerate() {
            let expert_out = expert.forward(x.clone());
            // Get weight for this expert
            let weight = router_probs
                .clone()
                .slice([0..batch, 0..seq_len, i..(i + 1)])
                .reshape([batch, seq_len, 1]);
            out = out + expert_out * weight;
        }

        out
    }
}

/// Wan Final Layer
#[derive(Module, Debug)]
pub struct WanFinalLayer<B: Backend> {
    pub norm: LayerNorm<B>,
    pub proj: Linear<B>,
    pub modulation: Linear<B>,
}

impl<B: Backend> WanFinalLayer<B> {
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

/// Wan Video Model
#[derive(Module, Debug)]
pub struct Wan<B: Backend> {
    pub video_embed: Linear<B>,
    pub text_embed: Linear<B>,
    pub time_embed: WanTimestepEmbed<B>,
    pub blocks: Vec<WanBlock<B>>,
    pub final_layer: WanFinalLayer<B>,
    #[module(skip)]
    pub hidden_size: usize,
    #[module(skip)]
    pub patch_size: usize,
    #[module(skip)]
    pub temporal_patch_size: usize,
    #[module(skip)]
    pub in_channels: usize,
}

/// Runtime state for Wan
pub struct WanRuntime<B: Backend> {
    pub spatial_rope: RotaryEmbedding<B>,
    pub temporal_rope: RotaryEmbedding<B>,
}

/// Output from Wan
pub struct WanOutput<B: Backend> {
    /// Velocity prediction [batch, channels, time, height, width]
    pub velocity: Tensor<B, 5>,
}

impl<B: Backend> Wan<B> {
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
        runtime: &WanRuntime<B>,
    ) -> WanOutput<B> {
        // Patchify video
        let (x, nt, nh, nw, channels) = self.patchify(video_latents);

        // Project text
        let context = self.text_embed.forward(text_embeds);

        // Timestep embedding (using scalar forward to avoid tensor allocation)
        let cond = self.time_embed.forward_scalar(timestep);

        // DiT + MoE blocks
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(
                x,
                cond.clone(),
                context.clone(),
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

        WanOutput { velocity }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_wan_config() {
        let config = WanConfig::base();
        assert_eq!(config.hidden_size, 3584);
        assert_eq!(config.num_blocks, 32);
        assert_eq!(config.num_experts, 8);
    }

    #[test]
    fn test_wan_timestep_embed() {
        let device = Default::default();
        let embed = WanTimestepEmbed {
            linear1: LinearConfig::new(64, 256).with_bias(true).init(&device),
            linear2: LinearConfig::new(256, 256).with_bias(true).init(&device),
            freqs: wan_timestep_freqs(64, &device),
        };

        let t = Tensor::<TestBackend, 1>::from_floats([0.5], &device);
        let out = embed.forward(t);
        assert_eq!(out.dims(), [1, 256]);
    }

    #[test]
    fn test_wan_attention() {
        let device = Default::default();
        let attn = WanAttention {
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
    fn test_wan_expert() {
        let device = Default::default();
        let expert = WanExpert {
            up: LinearConfig::new(256, 512).with_bias(false).init(&device),
            gate: LinearConfig::new(256, 512).with_bias(false).init(&device),
            down: LinearConfig::new(512, 256).with_bias(false).init(&device),
        };

        let x = Tensor::<TestBackend, 3>::zeros([2, 16, 256], &device);
        let out = expert.forward(x);
        assert_eq!(out.dims(), [2, 16, 256]);
    }

    #[test]
    fn test_wan_block() {
        let device = Default::default();
        let block = WanBlockConfig::new(256, 4, 512, 4, 2).init::<TestBackend>(&device);
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
    fn test_wan_tiny_forward() {
        let device = Default::default();
        let config = WanConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        // [batch=1, channels=4, time=4, height=8, width=8]
        let video = Tensor::zeros([1, 4, 4, 8, 8], &device);
        let text = Tensor::zeros([1, 4, 128], &device);

        let output = model.forward(video, 0.5, text, &runtime);

        assert_eq!(output.velocity.dims(), [1, 4, 4, 8, 8]);
    }
}
