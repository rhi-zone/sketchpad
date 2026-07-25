//! Hunyuan-DiT Model Implementation
//!
//! Hunyuan-DiT is Tencent's bilingual (Chinese/English) DiT model that uses
//! dual text encoders and cross-attention.
//!
//! # Architecture Overview
//!
//! ```text
//! Input: image patches + timestep
//!        ↓
//! [DiT Blocks with Cross-Attention] - cross-attend to dual text embeddings
//!        ↓
//! Output: noise prediction
//! ```
//!
//! # Key Features
//!
//! - **Bilingual**: Native Chinese and English support
//! - **Dual text encoder**: CLIP + MT5 for rich text understanding
//! - **Cross-attention**: Similar to PixArt architecture
//! - **Skip connections**: Long skip connections for better gradients

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

use sketchpad_core::dit::{PatchEmbed, PatchEmbedConfig, unpatchify};
use sketchpad_core::glu::SwiGluFfn;
use sketchpad_core::layernorm::LayerNorm;
use sketchpad_core::rope::RotaryEmbedding;

/// Hunyuan-DiT Model Configuration
#[derive(Debug, Clone)]
pub struct HunyuanDiTConfig {
    /// Number of input image channels (usually 4 for VAE latents)
    pub in_channels: usize,
    /// Patch size
    pub patch_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of DiT blocks
    pub num_blocks: usize,
    /// CLIP text embedding dimension
    pub clip_dim: usize,
    /// MT5 text embedding dimension
    pub mt5_dim: usize,
    /// Timestep embedding dimension
    pub time_embed_dim: usize,
    /// FFN intermediate size multiplier
    pub mlp_ratio: f32,
    /// Maximum sequence length for RoPE
    pub max_seq_len: usize,
    /// Whether to use skip connections
    pub use_skip: bool,
}

impl HunyuanDiTConfig {
    /// Hunyuan-DiT v1.1 configuration
    pub fn v1_1() -> Self {
        Self {
            in_channels: 4,
            patch_size: 2,
            hidden_size: 1408,
            num_heads: 16,
            num_blocks: 40,
            clip_dim: 1024, // CLIP-ViT-L
            mt5_dim: 2048,  // MT5-XL
            time_embed_dim: 256,
            mlp_ratio: 4.0,
            max_seq_len: 4096,
            use_skip: true,
        }
    }

    /// Hunyuan-DiT v1.2 configuration (larger)
    pub fn v1_2() -> Self {
        Self {
            in_channels: 4,
            patch_size: 2,
            hidden_size: 1408,
            num_heads: 16,
            num_blocks: 40,
            clip_dim: 1024,
            mt5_dim: 2048,
            time_embed_dim: 256,
            mlp_ratio: 4.0,
            max_seq_len: 4096,
            use_skip: true,
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
            clip_dim: 64,
            mt5_dim: 128,
            time_embed_dim: 64,
            mlp_ratio: 4.0,
            max_seq_len: 256,
            use_skip: false, // Skip for tiny to simplify
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    pub fn intermediate_size(&self) -> usize {
        (self.hidden_size as f32 * self.mlp_ratio) as usize
    }

    /// Combined text dimension (CLIP + MT5)
    pub fn text_dim(&self) -> usize {
        self.clip_dim + self.mt5_dim
    }

    /// Initialize the model
    pub fn init<B: Backend>(&self, device: &B::Device) -> (HunyuanDiT<B>, HunyuanDiTRuntime<B>) {
        // Patch embedding
        let patch_embed =
            PatchEmbedConfig::new(self.patch_size, self.in_channels, self.hidden_size).init(device);

        // Text projections (separate for CLIP and MT5)
        let clip_proj = LinearConfig::new(self.clip_dim, self.hidden_size)
            .with_bias(true)
            .init(device);
        let mt5_proj = LinearConfig::new(self.mt5_dim, self.hidden_size)
            .with_bias(true)
            .init(device);

        // Timestep embedding with extra style embedding
        let time_embed = HunyuanTimestepEmbed {
            linear1: LinearConfig::new(self.time_embed_dim, self.hidden_size)
                .with_bias(true)
                .init(device),
            linear2: LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(true)
                .init(device),
            freqs: hunyuan_timestep_freqs(self.time_embed_dim, device),
        };

        // Style embedding (pooled CLIP)
        let style_embed = LinearConfig::new(self.clip_dim, self.hidden_size)
            .with_bias(true)
            .init(device);

        // DiT blocks
        let blocks: Vec<HunyuanBlock<B>> = (0..self.num_blocks)
            .map(|i| {
                HunyuanBlockConfig::new(
                    self.hidden_size,
                    self.num_heads,
                    self.intermediate_size(),
                    i < self.num_blocks / 2, // First half can receive skip
                )
                .init(device)
            })
            .collect();

        // Final layer
        let final_layer = HunyuanFinalLayer {
            norm: LayerNorm::new(self.hidden_size, device),
            proj: LinearConfig::new(
                self.hidden_size,
                self.patch_size * self.patch_size * self.in_channels,
            )
            .with_bias(true)
            .init(device),
            modulation: LinearConfig::new(self.hidden_size, 2 * self.hidden_size)
                .with_bias(true)
                .init(device),
        };

        let model = HunyuanDiT {
            patch_embed,
            clip_proj,
            mt5_proj,
            time_embed,
            style_embed,
            blocks,
            final_layer,
            hidden_size: self.hidden_size,
            patch_size: self.patch_size,
            in_channels: self.in_channels,
            use_skip: self.use_skip,
            num_blocks: self.num_blocks,
        };

        let runtime = HunyuanDiTRuntime {
            rope: RotaryEmbedding::new(self.head_dim(), self.max_seq_len, device),
        };

        (model, runtime)
    }
}

/// Timestep embedding for Hunyuan-DiT
#[derive(Module, Debug)]
pub struct HunyuanTimestepEmbed<B: Backend> {
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
    /// Precomputed sinusoidal frequencies
    pub freqs: Tensor<B, 1>,
}

/// Precompute sinusoidal frequencies for Hunyuan timestep embedding
pub fn hunyuan_timestep_freqs<B: Backend>(embed_dim: usize, device: &B::Device) -> Tensor<B, 1> {
    let half_dim = embed_dim / 2;
    let emb_scale = -(10000.0_f32.ln()) / (half_dim as f32 - 1.0);
    let freqs: Vec<f32> = (0..half_dim)
        .map(|i| (emb_scale * i as f32).exp())
        .collect();
    Tensor::<B, 1>::from_floats(freqs.as_slice(), device)
}

impl<B: Backend> HunyuanTimestepEmbed<B> {
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

/// Hunyuan Block Configuration
struct HunyuanBlockConfig {
    hidden_size: usize,
    num_heads: usize,
    intermediate_size: usize,
    has_skip_input: bool,
}

impl HunyuanBlockConfig {
    fn new(
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
        has_skip_input: bool,
    ) -> Self {
        Self {
            hidden_size,
            num_heads,
            intermediate_size,
            has_skip_input,
        }
    }

    fn init<B: Backend>(&self, device: &B::Device) -> HunyuanBlock<B> {
        use sketchpad_core::glu::SwiGluFfnConfig;

        let head_dim = self.hidden_size / self.num_heads;

        // Skip connection projection (if this block receives skip)
        let skip_proj = if self.has_skip_input {
            Some(
                LinearConfig::new(self.hidden_size * 2, self.hidden_size)
                    .with_bias(true)
                    .init(device),
            )
        } else {
            None
        };

        HunyuanBlock {
            norm1: LayerNorm::new(self.hidden_size, device),
            self_attn: HunyuanAttention {
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
            cross_attn: HunyuanCrossAttention {
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
            norm3: LayerNorm::new(self.hidden_size, device),
            ffn: SwiGluFfnConfig::new(self.hidden_size, self.intermediate_size).init(device),
            modulation: LinearConfig::new(self.hidden_size, 6 * self.hidden_size)
                .with_bias(true)
                .init(device),
            skip_proj,
            hidden_size: self.hidden_size,
        }
    }
}

/// Hunyuan Self-Attention with RoPE
#[derive(Module, Debug)]
pub struct HunyuanAttention<B: Backend> {
    pub to_qkv: Linear<B>,
    pub to_out: Linear<B>,
    #[module(skip)]
    pub num_heads: usize,
    #[module(skip)]
    pub head_dim: usize,
}

impl<B: Backend> HunyuanAttention<B> {
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

        // Attention
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

/// Hunyuan Cross-Attention to text
#[derive(Module, Debug)]
pub struct HunyuanCrossAttention<B: Backend> {
    pub to_q: Linear<B>,
    pub to_kv: Linear<B>,
    pub to_out: Linear<B>,
    #[module(skip)]
    pub num_heads: usize,
    #[module(skip)]
    pub head_dim: usize,
}

impl<B: Backend> HunyuanCrossAttention<B> {
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

/// Hunyuan DiT Block with cross-attention and optional skip connection
#[derive(Module, Debug)]
pub struct HunyuanBlock<B: Backend> {
    pub norm1: LayerNorm<B>,
    pub self_attn: HunyuanAttention<B>,
    pub norm2: LayerNorm<B>,
    pub cross_attn: HunyuanCrossAttention<B>,
    pub norm3: LayerNorm<B>,
    pub ffn: SwiGluFfn<B>,
    pub modulation: Linear<B>,
    pub skip_proj: Option<Linear<B>>,
    #[module(skip)]
    pub hidden_size: usize,
}

impl<B: Backend> HunyuanBlock<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        skip: Option<Tensor<B, 3>>,
        cond: Tensor<B, 2>,
        context: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
    ) -> Tensor<B, 3> {
        let [batch, _seq_len, hidden] = x.dims();

        // Apply skip connection if available
        let x = if let (Some(skip_tensor), Some(skip_proj)) = (skip, &self.skip_proj) {
            let combined = Tensor::cat(vec![x, skip_tensor], 2);
            skip_proj.forward(combined)
        } else {
            x
        };

        // Get modulation parameters
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

        // Self-attention with modulation
        let x_norm = self.norm1.forward(x.clone());
        let x_norm = (Tensor::ones_like(&scale1) + scale1) * x_norm + shift1;
        let x = x + gate1 * self.self_attn.forward(x_norm, rope);

        // Cross-attention (no modulation)
        let x_norm = self.norm2.forward(x.clone());
        let x = x + self.cross_attn.forward(x_norm, context);

        // FFN with modulation
        let x_norm = self.norm3.forward(x.clone());
        let x_norm = (Tensor::ones_like(&scale2) + scale2) * x_norm + shift2;
        x + gate2 * self.ffn.forward(x_norm)
    }
}

/// Hunyuan Final Layer
#[derive(Module, Debug)]
pub struct HunyuanFinalLayer<B: Backend> {
    pub norm: LayerNorm<B>,
    pub proj: Linear<B>,
    pub modulation: Linear<B>,
}

impl<B: Backend> HunyuanFinalLayer<B> {
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

/// Hunyuan-DiT Model
#[derive(Module, Debug)]
pub struct HunyuanDiT<B: Backend> {
    pub patch_embed: PatchEmbed<B>,
    pub clip_proj: Linear<B>,
    pub mt5_proj: Linear<B>,
    pub time_embed: HunyuanTimestepEmbed<B>,
    pub style_embed: Linear<B>,
    pub blocks: Vec<HunyuanBlock<B>>,
    pub final_layer: HunyuanFinalLayer<B>,
    #[module(skip)]
    pub hidden_size: usize,
    #[module(skip)]
    pub patch_size: usize,
    #[module(skip)]
    pub in_channels: usize,
    #[module(skip)]
    pub use_skip: bool,
    #[module(skip)]
    pub num_blocks: usize,
}

/// Runtime state for Hunyuan-DiT
pub struct HunyuanDiTRuntime<B: Backend> {
    pub rope: RotaryEmbedding<B>,
}

/// Output from Hunyuan-DiT
pub struct HunyuanDiTOutput<B: Backend> {
    /// Noise prediction [batch, channels, height, width]
    pub noise: Tensor<B, 4>,
}

impl<B: Backend> HunyuanDiT<B> {
    /// Forward pass
    ///
    /// # Arguments
    ///
    /// * `latents` - Noisy latent images [batch, channels, height, width]
    /// * `timestep` - Diffusion timestep
    /// * `clip_embeds` - CLIP text embeddings [batch, seq_len, clip_dim]
    /// * `mt5_embeds` - MT5 text embeddings [batch, seq_len, mt5_dim]
    /// * `clip_pooled` - Pooled CLIP embedding [batch, clip_dim]
    /// * `runtime` - Runtime state with RoPE
    pub fn forward(
        &self,
        latents: Tensor<B, 4>,
        timestep: f32,
        clip_embeds: Tensor<B, 3>,
        mt5_embeds: Tensor<B, 3>,
        clip_pooled: Tensor<B, 2>,
        runtime: &HunyuanDiTRuntime<B>,
    ) -> HunyuanDiTOutput<B> {
        let [_batch, _channels, height, width] = latents.dims();

        // Patchify
        let x = self.patch_embed.forward(latents);

        // Project text embeddings
        let clip_ctx = self.clip_proj.forward(clip_embeds);
        let mt5_ctx = self.mt5_proj.forward(mt5_embeds);
        let context = Tensor::cat(vec![clip_ctx, mt5_ctx], 1);

        // Timestep + style conditioning (using scalar forward to avoid tensor allocation)
        let t_emb = self.time_embed.forward_scalar(timestep);
        let style_emb = self.style_embed.forward(clip_pooled);
        let cond = t_emb + style_emb;

        // DiT blocks with skip connections
        let mut x = x;
        let mut skip_outputs: Vec<Tensor<B, 3>> = Vec::new();
        let half_blocks = self.num_blocks / 2;

        for (i, block) in self.blocks.iter().enumerate() {
            // First half: save outputs for skip connections
            if i < half_blocks {
                x = block.forward(
                    x.clone(),
                    None,
                    cond.clone(),
                    context.clone(),
                    &runtime.rope,
                );
                if self.use_skip {
                    skip_outputs.push(x.clone());
                }
            } else {
                // Second half: use skip connections from corresponding block
                let skip = if self.use_skip {
                    let skip_idx = self.num_blocks - 1 - i;
                    if skip_idx < skip_outputs.len() {
                        Some(skip_outputs[skip_idx].clone())
                    } else {
                        None
                    }
                } else {
                    None
                };
                x = block.forward(x, skip, cond.clone(), context.clone(), &runtime.rope);
            }
        }

        // Final layer
        let out = self.final_layer.forward(x, cond);

        // Unpatchify
        let noise = unpatchify(out, self.patch_size, height, width, self.in_channels);

        HunyuanDiTOutput { noise }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_hunyuan_config() {
        let config = HunyuanDiTConfig::v1_1();
        assert_eq!(config.hidden_size, 1408);
        assert_eq!(config.num_blocks, 40);
        assert_eq!(config.text_dim(), 1024 + 2048);
    }

    #[test]
    fn test_hunyuan_timestep_embed() {
        let device = Default::default();
        let embed = HunyuanTimestepEmbed {
            linear1: LinearConfig::new(64, 256).with_bias(true).init(&device),
            linear2: LinearConfig::new(256, 256).with_bias(true).init(&device),
            freqs: hunyuan_timestep_freqs(64, &device),
        };

        let t = Tensor::<TestBackend, 1>::from_floats([0.5], &device);
        let out = embed.forward(t);
        assert_eq!(out.dims(), [1, 256]);
    }

    #[test]
    fn test_hunyuan_self_attention() {
        let device = Default::default();
        let attn = HunyuanAttention {
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
    fn test_hunyuan_cross_attention() {
        let device = Default::default();
        let attn = HunyuanCrossAttention {
            to_q: LinearConfig::new(256, 256).with_bias(true).init(&device),
            to_kv: LinearConfig::new(256, 512).with_bias(true).init(&device),
            to_out: LinearConfig::new(256, 256).with_bias(true).init(&device),
            num_heads: 4,
            head_dim: 64,
        };

        let x = Tensor::<TestBackend, 3>::zeros([2, 16, 256], &device);
        let ctx = Tensor::<TestBackend, 3>::zeros([2, 8, 256], &device);
        let out = attn.forward(x, ctx);
        assert_eq!(out.dims(), [2, 16, 256]);
    }

    #[test]
    fn test_hunyuan_block() {
        let device = Default::default();
        let block = HunyuanBlockConfig::new(256, 4, 512, false).init::<TestBackend>(&device);
        let rope = RotaryEmbedding::new(64, 256, &device);

        let x = Tensor::zeros([2, 16, 256], &device);
        let cond = Tensor::zeros([2, 256], &device);
        let ctx = Tensor::zeros([2, 8, 256], &device);

        let out = block.forward(x, None, cond, ctx, &rope);
        assert_eq!(out.dims(), [2, 16, 256]);
    }

    #[test]
    fn test_hunyuan_tiny_forward() {
        let device = Default::default();
        let config = HunyuanDiTConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        // [batch=1, channels=4, height=8, width=8]
        let latents = Tensor::zeros([1, 4, 8, 8], &device);
        let clip = Tensor::zeros([1, 4, 64], &device);
        let mt5 = Tensor::zeros([1, 4, 128], &device);
        let clip_pooled = Tensor::zeros([1, 64], &device);

        let output = model.forward(latents, 0.5, clip, mt5, clip_pooled, &runtime);

        assert_eq!(output.noise.dims(), [1, 4, 8, 8]);
    }
}
