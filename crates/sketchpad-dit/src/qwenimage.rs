//! Qwen-Image Model Implementation
//!
//! Qwen-Image is Alibaba's 20B MMDiT image generation model that uses
//! Qwen2.5-VL as the text encoder, released under Apache 2.0.
//!
//! # Architecture Overview
//!
//! ```text
//! Input: latent patches + Qwen2.5-VL text embeddings + timestep
//!        ↓
//! [MMDiT Blocks with Joint Attention]
//!        ↓
//! Output: velocity prediction for flow matching
//! ```
//!
//! # Key Features
//!
//! - **MMDiT**: Multimodal DiT with joint attention (like SD3/Flux)
//! - **Qwen2.5-VL encoder**: Large vision-language model as text encoder
//! - **20B params**: Large model for high quality
//! - **Apache 2.0**: Permissive license
//! - **Flow matching**: Rectified flow objective

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

use sketchpad_core::glu::SwiGluFfn;
use sketchpad_core::layernorm::LayerNorm;
use sketchpad_core::rope::RotaryEmbedding;

/// Qwen-Image Model Configuration
#[derive(Debug, Clone)]
pub struct QwenImageConfig {
    /// Number of input channels (from VAE)
    pub in_channels: usize,
    /// Patch size for patchifying latents
    pub patch_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of MMDiT blocks
    pub num_blocks: usize,
    /// Text embedding dimension (from Qwen2.5-VL)
    pub text_dim: usize,
    /// Pooled text embedding dimension (for conditioning)
    pub pooled_dim: usize,
    /// Timestep embedding dimension
    pub time_embed_dim: usize,
    /// FFN intermediate size multiplier
    pub mlp_ratio: f32,
    /// Maximum sequence length for RoPE
    pub max_seq_len: usize,
}

impl QwenImageConfig {
    /// Qwen-Image 20B base configuration
    pub fn base() -> Self {
        Self {
            in_channels: 16, // From VAE
            patch_size: 2,
            hidden_size: 4096,
            num_heads: 32,
            num_blocks: 48,
            text_dim: 4096,   // Qwen2.5-VL hidden size
            pooled_dim: 2048, // Pooled embedding
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
            pooled_dim: 64,
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
    pub fn init<B: Backend>(&self, device: &B::Device) -> (QwenImage<B>, QwenImageRuntime<B>) {
        // Image patch embedding
        let patch_dim = self.in_channels * self.patch_size * self.patch_size;
        let img_embed = LinearConfig::new(patch_dim, self.hidden_size)
            .with_bias(true)
            .init(device);

        // Text projection
        let text_embed = LinearConfig::new(self.text_dim, self.hidden_size)
            .with_bias(true)
            .init(device);

        // Timestep embedding with pooled text conditioning
        let time_embed = QwenImageTimestepEmbed {
            linear1: LinearConfig::new(self.time_embed_dim + self.pooled_dim, self.hidden_size)
                .with_bias(true)
                .init(device),
            linear2: LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(true)
                .init(device),
            freqs: qwenimage_timestep_freqs(self.time_embed_dim, device),
        };

        // MMDiT blocks
        let blocks: Vec<QwenImageBlock<B>> = (0..self.num_blocks)
            .map(|_| {
                QwenImageBlockConfig::new(
                    self.hidden_size,
                    self.num_heads,
                    self.intermediate_size(),
                )
                .init(device)
            })
            .collect();

        // Final layer
        let final_layer = QwenImageFinalLayer {
            norm: LayerNorm::new(self.hidden_size, device),
            proj: LinearConfig::new(self.hidden_size, patch_dim)
                .with_bias(true)
                .init(device),
            modulation: LinearConfig::new(self.hidden_size, 2 * self.hidden_size)
                .with_bias(true)
                .init(device),
        };

        let model = QwenImage {
            img_embed,
            text_embed,
            time_embed,
            blocks,
            final_layer,
            hidden_size: self.hidden_size,
            patch_size: self.patch_size,
            in_channels: self.in_channels,
        };

        let runtime = QwenImageRuntime {
            img_rope: RotaryEmbedding::new(self.head_dim(), self.max_seq_len, device),
        };

        (model, runtime)
    }
}

/// Precompute sinusoidal timestep frequencies for Qwen-Image.
#[rustfmt::skip]
pub fn qwenimage_timestep_freqs<B: Backend>(embed_dim: usize, device: &B::Device) -> Tensor<B, 1> {
    let half_dim = embed_dim / 2;
    let emb_scale = -(10000.0_f32.ln()) / (half_dim as f32 - 1.0);
    let freqs: Vec<f32> = (0..half_dim)
        .map(|i| (emb_scale * i as f32).exp())
        .collect();
    Tensor::<B, 1>::from_floats(freqs.as_slice(), device)
}

/// Timestep embedding for Qwen-Image with pooled text conditioning
#[derive(Module, Debug)]
pub struct QwenImageTimestepEmbed<B: Backend> {
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
    /// Precomputed sinusoidal frequencies
    pub freqs: Tensor<B, 1>,
}

impl<B: Backend> QwenImageTimestepEmbed<B> {
    pub fn forward(&self, t: Tensor<B, 1>, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
        let t_expanded = t.unsqueeze_dim::<2>(1);
        let freqs_expanded = self.freqs.clone().unsqueeze_dim::<2>(0);
        let angles = t_expanded.matmul(freqs_expanded);

        let sin_emb = angles.clone().sin();
        let cos_emb = angles.cos();
        let time_emb = Tensor::cat(vec![sin_emb, cos_emb], 1);

        // Concatenate time embedding with pooled text embedding
        let emb = Tensor::cat(vec![time_emb, pooled], 1);

        let x = self.linear1.forward(emb);
        let x = burn::tensor::activation::silu(x);
        self.linear2.forward(x)
    }

    /// Forward pass for a single scalar timestep with pooled embeddings (no tensor allocation for timestep).
    pub fn forward_scalar(&self, t: f32, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
        let angles = self.freqs.clone() * t;
        let sin_emb = angles.clone().sin();
        let cos_emb = angles.cos();
        let time_emb = Tensor::cat(vec![sin_emb, cos_emb], 0).unsqueeze_dim(0);

        // Concatenate time embedding with pooled text embedding
        let emb = Tensor::cat(vec![time_emb, pooled], 1);

        let x = self.linear1.forward(emb);
        let x = burn::tensor::activation::silu(x);
        self.linear2.forward(x)
    }
}

/// Qwen-Image Block Configuration
struct QwenImageBlockConfig {
    hidden_size: usize,
    num_heads: usize,
    intermediate_size: usize,
}

impl QwenImageBlockConfig {
    fn new(hidden_size: usize, num_heads: usize, intermediate_size: usize) -> Self {
        Self {
            hidden_size,
            num_heads,
            intermediate_size,
        }
    }

    fn init<B: Backend>(&self, device: &B::Device) -> QwenImageBlock<B> {
        use sketchpad_core::glu::SwiGluFfnConfig;

        let head_dim = self.hidden_size / self.num_heads;

        QwenImageBlock {
            // Image stream
            img_norm1: LayerNorm::new(self.hidden_size, device),
            img_norm2: LayerNorm::new(self.hidden_size, device),
            img_ffn: SwiGluFfnConfig::new(self.hidden_size, self.intermediate_size).init(device),
            // Text stream
            txt_norm1: LayerNorm::new(self.hidden_size, device),
            txt_norm2: LayerNorm::new(self.hidden_size, device),
            txt_ffn: SwiGluFfnConfig::new(self.hidden_size, self.intermediate_size).init(device),
            // Joint attention
            joint_attn: QwenImageJointAttention {
                img_to_qkv: LinearConfig::new(self.hidden_size, 3 * self.hidden_size)
                    .with_bias(true)
                    .init(device),
                txt_to_qkv: LinearConfig::new(self.hidden_size, 3 * self.hidden_size)
                    .with_bias(true)
                    .init(device),
                img_to_out: LinearConfig::new(self.hidden_size, self.hidden_size)
                    .with_bias(true)
                    .init(device),
                txt_to_out: LinearConfig::new(self.hidden_size, self.hidden_size)
                    .with_bias(true)
                    .init(device),
                num_heads: self.num_heads,
                head_dim,
            },
            // Modulation
            img_modulation: LinearConfig::new(self.hidden_size, 4 * self.hidden_size)
                .with_bias(true)
                .init(device),
            txt_modulation: LinearConfig::new(self.hidden_size, 4 * self.hidden_size)
                .with_bias(true)
                .init(device),
            hidden_size: self.hidden_size,
        }
    }
}

/// Joint attention for MMDiT (image and text attend to each other)
#[derive(Module, Debug)]
pub struct QwenImageJointAttention<B: Backend> {
    pub img_to_qkv: Linear<B>,
    pub txt_to_qkv: Linear<B>,
    pub img_to_out: Linear<B>,
    pub txt_to_out: Linear<B>,
    #[module(skip)]
    pub num_heads: usize,
    #[module(skip)]
    pub head_dim: usize,
}

impl<B: Backend> QwenImageJointAttention<B> {
    pub fn forward(
        &self,
        img: Tensor<B, 3>,
        txt: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [batch, img_len, _hidden] = img.dims();
        let [_, txt_len, _] = txt.dims();
        let total_len = img_len + txt_len;

        // Compute Q, K, V for both streams
        let img_qkv = self.img_to_qkv.forward(img);
        let txt_qkv = self.txt_to_qkv.forward(txt);

        // Reshape to [batch, seq, 3, heads, head_dim]
        let img_qkv = img_qkv.reshape([batch, img_len, 3, self.num_heads, self.head_dim]);
        let txt_qkv = txt_qkv.reshape([batch, txt_len, 3, self.num_heads, self.head_dim]);

        // Extract Q, K, V
        let img_q = img_qkv
            .clone()
            .slice([
                0..batch,
                0..img_len,
                0..1,
                0..self.num_heads,
                0..self.head_dim,
            ])
            .reshape([batch, img_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let img_k = img_qkv
            .clone()
            .slice([
                0..batch,
                0..img_len,
                1..2,
                0..self.num_heads,
                0..self.head_dim,
            ])
            .reshape([batch, img_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let img_v = img_qkv
            .slice([
                0..batch,
                0..img_len,
                2..3,
                0..self.num_heads,
                0..self.head_dim,
            ])
            .reshape([batch, img_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);

        let txt_q = txt_qkv
            .clone()
            .slice([
                0..batch,
                0..txt_len,
                0..1,
                0..self.num_heads,
                0..self.head_dim,
            ])
            .reshape([batch, txt_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let txt_k = txt_qkv
            .clone()
            .slice([
                0..batch,
                0..txt_len,
                1..2,
                0..self.num_heads,
                0..self.head_dim,
            ])
            .reshape([batch, txt_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let txt_v = txt_qkv
            .slice([
                0..batch,
                0..txt_len,
                2..3,
                0..self.num_heads,
                0..self.head_dim,
            ])
            .reshape([batch, txt_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);

        // Apply RoPE to image Q and K only
        let (img_q, img_k) = rope.forward(img_q, img_k, 0);

        // Concatenate for joint attention
        let q = Tensor::cat(vec![img_q, txt_q], 2); // [B, heads, img+txt, head_dim]
        let k = Tensor::cat(vec![img_k, txt_k], 2);
        let v = Tensor::cat(vec![img_v, txt_v], 2);

        // Attention
        let scale = (self.head_dim as f32).sqrt().recip();
        let attn = q.matmul(k.swap_dims(2, 3)) * scale;
        let attn = burn::tensor::activation::softmax(attn, 3);
        let out = attn.matmul(v);

        // Split back to image and text
        let out = out.swap_dims(1, 2); // [B, total, heads, head_dim]
        let img_out = out
            .clone()
            .slice([0..batch, 0..img_len, 0..self.num_heads, 0..self.head_dim])
            .reshape([batch, img_len, self.num_heads * self.head_dim]);
        let txt_out = out
            .slice([
                0..batch,
                img_len..total_len,
                0..self.num_heads,
                0..self.head_dim,
            ])
            .reshape([batch, txt_len, self.num_heads * self.head_dim]);

        let img_out = self.img_to_out.forward(img_out);
        let txt_out = self.txt_to_out.forward(txt_out);

        (img_out, txt_out)
    }
}

/// Qwen-Image MMDiT Block
#[derive(Module, Debug)]
pub struct QwenImageBlock<B: Backend> {
    // Image stream
    pub img_norm1: LayerNorm<B>,
    pub img_norm2: LayerNorm<B>,
    pub img_ffn: SwiGluFfn<B>,
    // Text stream
    pub txt_norm1: LayerNorm<B>,
    pub txt_norm2: LayerNorm<B>,
    pub txt_ffn: SwiGluFfn<B>,
    // Joint attention
    pub joint_attn: QwenImageJointAttention<B>,
    // Modulation
    pub img_modulation: Linear<B>,
    pub txt_modulation: Linear<B>,
    #[module(skip)]
    pub hidden_size: usize,
}

impl<B: Backend> QwenImageBlock<B> {
    pub fn forward(
        &self,
        img: Tensor<B, 3>,
        txt: Tensor<B, 3>,
        cond: Tensor<B, 2>,
        rope: &RotaryEmbedding<B>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [batch, _img_len, hidden] = img.dims();

        // Get modulation for both streams (shift1, scale1, shift2, scale2)
        let img_mod = self.img_modulation.forward(cond.clone());
        let txt_mod = self.txt_modulation.forward(cond);

        let img_mod = img_mod.reshape([batch, 4, hidden]);
        let txt_mod = txt_mod.reshape([batch, 4, hidden]);

        // Image modulation
        let img_shift1 = img_mod
            .clone()
            .slice([0..batch, 0..1, 0..hidden])
            .reshape([batch, 1, hidden]);
        let img_scale1 = img_mod
            .clone()
            .slice([0..batch, 1..2, 0..hidden])
            .reshape([batch, 1, hidden]);
        let img_shift2 = img_mod
            .clone()
            .slice([0..batch, 2..3, 0..hidden])
            .reshape([batch, 1, hidden]);
        let img_scale2 = img_mod
            .slice([0..batch, 3..4, 0..hidden])
            .reshape([batch, 1, hidden]);

        // Text modulation
        let txt_shift1 = txt_mod
            .clone()
            .slice([0..batch, 0..1, 0..hidden])
            .reshape([batch, 1, hidden]);
        let txt_scale1 = txt_mod
            .clone()
            .slice([0..batch, 1..2, 0..hidden])
            .reshape([batch, 1, hidden]);
        let txt_shift2 = txt_mod
            .clone()
            .slice([0..batch, 2..3, 0..hidden])
            .reshape([batch, 1, hidden]);
        let txt_scale2 = txt_mod
            .slice([0..batch, 3..4, 0..hidden])
            .reshape([batch, 1, hidden]);

        // Pre-attention norm with modulation
        let img_normed = self.img_norm1.forward(img.clone());
        let img_normed = (Tensor::ones_like(&img_scale1) + img_scale1) * img_normed + img_shift1;
        let txt_normed = self.txt_norm1.forward(txt.clone());
        let txt_normed = (Tensor::ones_like(&txt_scale1) + txt_scale1) * txt_normed + txt_shift1;

        // Joint attention
        let (img_attn, txt_attn) = self.joint_attn.forward(img_normed, txt_normed, rope);
        let img = img + img_attn;
        let txt = txt + txt_attn;

        // Post-attention FFN with modulation
        let img_normed = self.img_norm2.forward(img.clone());
        let img_normed = (Tensor::ones_like(&img_scale2) + img_scale2) * img_normed + img_shift2;
        let txt_normed = self.txt_norm2.forward(txt.clone());
        let txt_normed = (Tensor::ones_like(&txt_scale2) + txt_scale2) * txt_normed + txt_shift2;

        let img = img + self.img_ffn.forward(img_normed);
        let txt = txt + self.txt_ffn.forward(txt_normed);

        (img, txt)
    }
}

/// Qwen-Image Final Layer
#[derive(Module, Debug)]
pub struct QwenImageFinalLayer<B: Backend> {
    pub norm: LayerNorm<B>,
    pub proj: Linear<B>,
    pub modulation: Linear<B>,
}

impl<B: Backend> QwenImageFinalLayer<B> {
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

/// Qwen-Image Model
#[derive(Module, Debug)]
pub struct QwenImage<B: Backend> {
    pub img_embed: Linear<B>,
    pub text_embed: Linear<B>,
    pub time_embed: QwenImageTimestepEmbed<B>,
    pub blocks: Vec<QwenImageBlock<B>>,
    pub final_layer: QwenImageFinalLayer<B>,
    #[module(skip)]
    pub hidden_size: usize,
    #[module(skip)]
    pub patch_size: usize,
    #[module(skip)]
    pub in_channels: usize,
}

/// Runtime state for Qwen-Image
pub struct QwenImageRuntime<B: Backend> {
    pub img_rope: RotaryEmbedding<B>,
}

/// Output from Qwen-Image
pub struct QwenImageOutput<B: Backend> {
    /// Velocity prediction [batch, channels, height, width]
    pub velocity: Tensor<B, 4>,
}

impl<B: Backend> QwenImage<B> {
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
        pooled_embeds: Tensor<B, 2>,
        runtime: &QwenImageRuntime<B>,
    ) -> QwenImageOutput<B> {
        // Patchify image
        let (img, nh, nw) = self.patchify(latents);

        // Project text
        let txt = self.text_embed.forward(text_embeds);

        // Timestep embedding with pooled text conditioning (using scalar forward to avoid tensor allocation)
        let cond = self.time_embed.forward_scalar(timestep, pooled_embeds);

        // MMDiT blocks
        let mut img = img;
        let mut txt = txt;
        for block in &self.blocks {
            let (new_img, new_txt) = block.forward(img, txt, cond.clone(), &runtime.img_rope);
            img = new_img;
            txt = new_txt;
        }

        // Final layer (image only)
        let out = self.final_layer.forward(img, cond);

        // Unpatchify
        let velocity = self.unpatchify(out, nh, nw);

        QwenImageOutput { velocity }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_qwenimage_config() {
        let config = QwenImageConfig::base();
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_blocks, 48);
    }

    #[test]
    fn test_qwenimage_timestep_embed() {
        let device = Default::default();
        let embed = QwenImageTimestepEmbed {
            linear1: LinearConfig::new(128, 256).with_bias(true).init(&device), // 64 + 64
            linear2: LinearConfig::new(256, 256).with_bias(true).init(&device),
            freqs: qwenimage_timestep_freqs(64, &device),
        };

        let t = Tensor::<TestBackend, 1>::from_floats([0.5], &device);
        let pooled = Tensor::zeros([1, 64], &device);
        let out = embed.forward(t, pooled);
        assert_eq!(out.dims(), [1, 256]);
    }

    #[test]
    fn test_qwenimage_joint_attention() {
        let device = Default::default();
        let attn = QwenImageJointAttention {
            img_to_qkv: LinearConfig::new(256, 768).with_bias(true).init(&device),
            txt_to_qkv: LinearConfig::new(256, 768).with_bias(true).init(&device),
            img_to_out: LinearConfig::new(256, 256).with_bias(true).init(&device),
            txt_to_out: LinearConfig::new(256, 256).with_bias(true).init(&device),
            num_heads: 4,
            head_dim: 64,
        };
        let rope = RotaryEmbedding::new(64, 256, &device);

        let img = Tensor::<TestBackend, 3>::zeros([2, 16, 256], &device);
        let txt = Tensor::zeros([2, 8, 256], &device);

        let (img_out, txt_out) = attn.forward(img, txt, &rope);
        assert_eq!(img_out.dims(), [2, 16, 256]);
        assert_eq!(txt_out.dims(), [2, 8, 256]);
    }

    #[test]
    fn test_qwenimage_block() {
        let device = Default::default();
        let block = QwenImageBlockConfig::new(256, 4, 512).init::<TestBackend>(&device);
        let rope = RotaryEmbedding::new(64, 256, &device);

        let img = Tensor::zeros([2, 16, 256], &device);
        let txt = Tensor::zeros([2, 8, 256], &device);
        let cond = Tensor::zeros([2, 256], &device);

        let (img_out, txt_out) = block.forward(img, txt, cond, &rope);
        assert_eq!(img_out.dims(), [2, 16, 256]);
        assert_eq!(txt_out.dims(), [2, 8, 256]);
    }

    #[test]
    fn test_qwenimage_tiny_forward() {
        let device = Default::default();
        let config = QwenImageConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        // [batch=1, channels=4, height=8, width=8]
        let latents = Tensor::zeros([1, 4, 8, 8], &device);
        let text = Tensor::zeros([1, 4, 128], &device);
        let pooled = Tensor::zeros([1, 64], &device);

        let output = model.forward(latents, 0.5, text, pooled, &runtime);

        assert_eq!(output.velocity.dims(), [1, 4, 8, 8]);
    }
}
