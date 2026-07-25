//! Z-Image Model Implementation
//!
//! Z-Image is Alibaba's fast image generation model using a single-stream
//! DiT (S3-DiT) architecture that achieves high quality with fast inference.
//!
//! # Architecture Overview
//!
//! ```text
//! Input: latent patches + text embeddings + timestep
//!        ↓
//! [Single-Stream DiT Blocks - text and image together]
//!        ↓
//! Output: velocity prediction for flow matching
//! ```
//!
//! # Key Features
//!
//! - **Single-stream**: Text and image tokens processed together (not dual-stream)
//! - **6B params**: Large model for high quality
//! - **Fast inference**: Optimized architecture
//! - **Flow matching**: Rectified flow objective

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

use sketchpad_core::glu::SwiGluFfn;
use sketchpad_core::layernorm::LayerNorm;
use sketchpad_core::rope::RotaryEmbedding;

/// Z-Image Model Configuration
#[derive(Debug, Clone)]
pub struct ZImageConfig {
    /// Number of input channels (from VAE)
    pub in_channels: usize,
    /// Patch size for patchifying latents
    pub patch_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of DiT blocks
    pub num_blocks: usize,
    /// Text embedding dimension (T5)
    pub text_dim: usize,
    /// Timestep embedding dimension
    pub time_embed_dim: usize,
    /// FFN intermediate size multiplier
    pub mlp_ratio: f32,
    /// Maximum sequence length for RoPE
    pub max_seq_len: usize,
}

impl ZImageConfig {
    /// Z-Image 6B base configuration
    pub fn base() -> Self {
        Self {
            in_channels: 16, // From SDXL-style VAE
            patch_size: 2,
            hidden_size: 3072,
            num_heads: 24,
            num_blocks: 36,
            text_dim: 4096, // T5-XXL
            time_embed_dim: 256,
            mlp_ratio: 4.0,
            max_seq_len: 4096,
        }
    }

    /// Tiny model for testing
    pub fn tiny() -> Self {
        Self {
            in_channels: 4,
            patch_size: 2,
            hidden_size: 256,
            num_heads: 4,
            num_blocks: 4,
            text_dim: 128,
            time_embed_dim: 64,
            mlp_ratio: 4.0,
            max_seq_len: 256,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    pub fn intermediate_size(&self) -> usize {
        (self.hidden_size as f32 * self.mlp_ratio) as usize
    }

    /// Initialize the model
    pub fn init<B: Backend>(&self, device: &B::Device) -> (ZImage<B>, ZImageRuntime<B>) {
        // Image patch embedding
        let patch_dim = self.in_channels * self.patch_size * self.patch_size;
        let img_embed = LinearConfig::new(patch_dim, self.hidden_size)
            .with_bias(true)
            .init(device);

        // Text projection
        let text_embed = LinearConfig::new(self.text_dim, self.hidden_size)
            .with_bias(true)
            .init(device);

        // Timestep embedding
        let time_embed = ZImageTimestepEmbed {
            linear1: LinearConfig::new(self.time_embed_dim, self.hidden_size)
                .with_bias(true)
                .init(device),
            linear2: LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(true)
                .init(device),
            freqs: zimage_timestep_freqs(self.time_embed_dim, device),
        };

        // Single-stream DiT blocks
        let blocks: Vec<ZImageBlock<B>> = (0..self.num_blocks)
            .map(|_| {
                ZImageBlockConfig::new(self.hidden_size, self.num_heads, self.intermediate_size())
                    .init(device)
            })
            .collect();

        // Final layer
        let final_layer = ZImageFinalLayer {
            norm: LayerNorm::new(self.hidden_size, device),
            proj: LinearConfig::new(self.hidden_size, patch_dim)
                .with_bias(true)
                .init(device),
            modulation: LinearConfig::new(self.hidden_size, 2 * self.hidden_size)
                .with_bias(true)
                .init(device),
        };

        let model = ZImage {
            img_embed,
            text_embed,
            time_embed,
            blocks,
            final_layer,
            hidden_size: self.hidden_size,
            patch_size: self.patch_size,
            in_channels: self.in_channels,
        };

        let runtime = ZImageRuntime {
            rope: RotaryEmbedding::new(self.head_dim(), self.max_seq_len, device),
        };

        (model, runtime)
    }
}

/// Precompute sinusoidal timestep frequencies for Z-Image.
#[rustfmt::skip]
pub fn zimage_timestep_freqs<B: Backend>(embed_dim: usize, device: &B::Device) -> Tensor<B, 1> {
    let half_dim = embed_dim / 2;
    let emb_scale = -(10000.0_f32.ln()) / (half_dim as f32 - 1.0);
    let freqs: Vec<f32> = (0..half_dim)
        .map(|i| (emb_scale * i as f32).exp())
        .collect();
    Tensor::<B, 1>::from_floats(freqs.as_slice(), device)
}

/// Timestep embedding for Z-Image
#[derive(Module, Debug)]
pub struct ZImageTimestepEmbed<B: Backend> {
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
    /// Precomputed sinusoidal frequencies
    pub freqs: Tensor<B, 1>,
}

impl<B: Backend> ZImageTimestepEmbed<B> {
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

    /// Forward pass for a single scalar timestep (no tensor allocation).
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

/// Z-Image Block Configuration
struct ZImageBlockConfig {
    hidden_size: usize,
    num_heads: usize,
    intermediate_size: usize,
}

impl ZImageBlockConfig {
    fn new(hidden_size: usize, num_heads: usize, intermediate_size: usize) -> Self {
        Self {
            hidden_size,
            num_heads,
            intermediate_size,
        }
    }

    fn init<B: Backend>(&self, device: &B::Device) -> ZImageBlock<B> {
        use sketchpad_core::glu::SwiGluFfnConfig;

        let head_dim = self.hidden_size / self.num_heads;

        ZImageBlock {
            norm1: LayerNorm::new(self.hidden_size, device),
            attn: ZImageAttention {
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
            modulation: LinearConfig::new(self.hidden_size, 4 * self.hidden_size)
                .with_bias(true)
                .init(device),
            hidden_size: self.hidden_size,
        }
    }
}

/// Z-Image Self-Attention with RoPE
#[derive(Module, Debug)]
pub struct ZImageAttention<B: Backend> {
    pub to_qkv: Linear<B>,
    pub to_out: Linear<B>,
    #[module(skip)]
    pub num_heads: usize,
    #[module(skip)]
    pub head_dim: usize,
}

impl<B: Backend> ZImageAttention<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        img_len: usize,
    ) -> Tensor<B, 3> {
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

        // Apply RoPE only to image tokens (not text)
        // For single-stream, we apply RoPE to image portion only
        let text_len = seq_len - img_len;

        // Split q and k into text and image parts
        let q_text = q
            .clone()
            .slice([0..batch, 0..self.num_heads, 0..text_len, 0..self.head_dim]);
        let q_img = q.slice([
            0..batch,
            0..self.num_heads,
            text_len..seq_len,
            0..self.head_dim,
        ]);
        let k_text = k
            .clone()
            .slice([0..batch, 0..self.num_heads, 0..text_len, 0..self.head_dim]);
        let k_img = k.slice([
            0..batch,
            0..self.num_heads,
            text_len..seq_len,
            0..self.head_dim,
        ]);

        // Apply RoPE to image tokens
        let (q_img_rope, k_img_rope) = rope.forward(q_img, k_img, 0);

        // Concatenate back
        let q = Tensor::cat(vec![q_text, q_img_rope], 2);
        let k = Tensor::cat(vec![k_text, k_img_rope], 2);

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

/// Z-Image Single-Stream DiT Block
#[derive(Module, Debug)]
pub struct ZImageBlock<B: Backend> {
    pub norm1: LayerNorm<B>,
    pub attn: ZImageAttention<B>,
    pub norm2: LayerNorm<B>,
    pub ffn: SwiGluFfn<B>,
    pub modulation: Linear<B>,
    #[module(skip)]
    pub hidden_size: usize,
}

impl<B: Backend> ZImageBlock<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        cond: Tensor<B, 2>,
        rope: &RotaryEmbedding<B>,
        img_len: usize,
    ) -> Tensor<B, 3> {
        let [batch, _seq_len, hidden] = x.dims();

        // Get modulation (shift1, scale1, shift2, scale2)
        let mod_params = self.modulation.forward(cond);
        let mod_params = mod_params.reshape([batch, 4, hidden]);

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
            .slice([0..batch, 3..4, 0..hidden])
            .reshape([batch, 1, hidden]);

        // Self-attention
        let x_norm = self.norm1.forward(x.clone());
        let x_norm = (Tensor::ones_like(&scale1) + scale1) * x_norm + shift1;
        let x = x + self.attn.forward(x_norm, rope, img_len);

        // FFN
        let x_norm = self.norm2.forward(x.clone());
        let x_norm = (Tensor::ones_like(&scale2) + scale2) * x_norm + shift2;
        x + self.ffn.forward(x_norm)
    }
}

/// Z-Image Final Layer
#[derive(Module, Debug)]
pub struct ZImageFinalLayer<B: Backend> {
    pub norm: LayerNorm<B>,
    pub proj: Linear<B>,
    pub modulation: Linear<B>,
}

impl<B: Backend> ZImageFinalLayer<B> {
    pub fn forward(&self, x: Tensor<B, 3>, cond: Tensor<B, 2>, text_len: usize) -> Tensor<B, 3> {
        let [batch, seq_len, hidden] = x.dims();

        // Only process image tokens
        let x_img = x.slice([0..batch, text_len..seq_len, 0..hidden]);

        let mod_params = self.modulation.forward(cond);
        let mod_params = mod_params.reshape([batch, 1, hidden * 2]);

        let shift = mod_params.clone().slice([0..batch, 0..1, 0..hidden]);
        let scale = mod_params.slice([0..batch, 0..1, hidden..(hidden * 2)]);

        let x_img = self.norm.forward(x_img);
        let x_img = (Tensor::ones_like(&scale) + scale) * x_img + shift;

        self.proj.forward(x_img)
    }
}

/// Z-Image Model
#[derive(Module, Debug)]
pub struct ZImage<B: Backend> {
    pub img_embed: Linear<B>,
    pub text_embed: Linear<B>,
    pub time_embed: ZImageTimestepEmbed<B>,
    pub blocks: Vec<ZImageBlock<B>>,
    pub final_layer: ZImageFinalLayer<B>,
    #[module(skip)]
    pub hidden_size: usize,
    #[module(skip)]
    pub patch_size: usize,
    #[module(skip)]
    pub in_channels: usize,
}

/// Runtime state for Z-Image
pub struct ZImageRuntime<B: Backend> {
    pub rope: RotaryEmbedding<B>,
}

/// Output from Z-Image
pub struct ZImageOutput<B: Backend> {
    /// Velocity prediction [batch, channels, height, width]
    pub velocity: Tensor<B, 4>,
}

impl<B: Backend> ZImage<B> {
    /// Patchify latents
    fn patchify(&self, x: Tensor<B, 4>) -> (Tensor<B, 3>, usize, usize) {
        let [batch, channels, height, width] = x.dims();
        let ps = self.patch_size;

        let nh = height / ps;
        let nw = width / ps;

        // [B, C, H, W] -> [B, C, nh, ps, nw, ps]
        let x = x.reshape([batch, channels, nh, ps, nw, ps]);
        // [B, nh, nw, C, ps, ps]
        let x = x.permute([0, 2, 4, 1, 3, 5]);
        // [B, nh*nw, C*ps*ps]
        let x = x.reshape([batch, nh * nw, channels * ps * ps]);

        let x = self.img_embed.forward(x);

        (x, nh, nw)
    }

    /// Unpatchify output
    fn unpatchify(&self, x: Tensor<B, 3>, nh: usize, nw: usize) -> Tensor<B, 4> {
        let [batch, _seq_len, _hidden] = x.dims();
        let ps = self.patch_size;
        let channels = self.in_channels;
        let height = nh * ps;
        let width = nw * ps;

        // [B, nh*nw, C*ps*ps] -> [B, nh, nw, C, ps, ps]
        let x = x.reshape([batch, nh, nw, channels, ps, ps]);
        // [B, C, nh, ps, nw, ps]
        let x = x.permute([0, 3, 1, 4, 2, 5]);
        // [B, C, H, W]
        x.reshape([batch, channels, height, width])
    }

    /// Forward pass
    pub fn forward(
        &self,
        latents: Tensor<B, 4>,
        timestep: f32,
        text_embeds: Tensor<B, 3>,
        runtime: &ZImageRuntime<B>,
    ) -> ZImageOutput<B> {
        // Patchify image
        let (img_tokens, nh, nw) = self.patchify(latents);
        let [_b, img_len, _hidden] = img_tokens.dims();
        let [_batch, text_len, _text_dim] = text_embeds.dims();

        // Project text
        let text_tokens = self.text_embed.forward(text_embeds);

        // Concatenate text + image (single-stream)
        let x = Tensor::cat(vec![text_tokens, img_tokens], 1);

        // Timestep embedding (using scalar forward to avoid tensor allocation)
        let cond = self.time_embed.forward_scalar(timestep);

        // DiT blocks
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(x, cond.clone(), &runtime.rope, img_len);
        }

        // Final layer (extracts and projects image tokens)
        let out = self.final_layer.forward(x, cond, text_len);

        // Unpatchify
        let velocity = self.unpatchify(out, nh, nw);

        ZImageOutput { velocity }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_zimage_config() {
        let config = ZImageConfig::base();
        assert_eq!(config.hidden_size, 3072);
        assert_eq!(config.num_blocks, 36);
    }

    #[test]
    fn test_zimage_timestep_embed() {
        let device = Default::default();
        let embed = ZImageTimestepEmbed {
            linear1: LinearConfig::new(64, 256).with_bias(true).init(&device),
            linear2: LinearConfig::new(256, 256).with_bias(true).init(&device),
            freqs: zimage_timestep_freqs(64, &device),
        };

        let t = Tensor::<TestBackend, 1>::from_floats([0.5], &device);
        let out = embed.forward(t);
        assert_eq!(out.dims(), [1, 256]);
    }

    #[test]
    fn test_zimage_attention() {
        let device = Default::default();
        let attn = ZImageAttention {
            to_qkv: LinearConfig::new(256, 768).with_bias(true).init(&device),
            to_out: LinearConfig::new(256, 256).with_bias(true).init(&device),
            num_heads: 4,
            head_dim: 64,
        };
        let rope = RotaryEmbedding::new(64, 256, &device);

        // text_len=4, img_len=12 -> total 16
        let x = Tensor::<TestBackend, 3>::zeros([2, 16, 256], &device);
        let out = attn.forward(x, &rope, 12);
        assert_eq!(out.dims(), [2, 16, 256]);
    }

    #[test]
    fn test_zimage_block() {
        let device = Default::default();
        let block = ZImageBlockConfig::new(256, 4, 512).init::<TestBackend>(&device);
        let rope = RotaryEmbedding::new(64, 256, &device);

        let x = Tensor::zeros([2, 20, 256], &device); // text=4 + img=16
        let cond = Tensor::zeros([2, 256], &device);

        let out = block.forward(x, cond, &rope, 16);
        assert_eq!(out.dims(), [2, 20, 256]);
    }

    #[test]
    fn test_zimage_tiny_forward() {
        let device = Default::default();
        let config = ZImageConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        // [batch=1, channels=4, height=8, width=8]
        let latents = Tensor::zeros([1, 4, 8, 8], &device);
        let text = Tensor::zeros([1, 4, 128], &device);

        let output = model.forward(latents, 0.5, text, &runtime);

        assert_eq!(output.velocity.dims(), [1, 4, 8, 8]);
    }
}
