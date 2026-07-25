//! PixArt-α/Σ Model Implementation
//!
//! PixArt is an efficient DiT-based image generation model that uses
//! cross-attention to T5 text embeddings.
//!
//! # Architecture Overview
//!
//! ```text
//! Input: image patches + timestep
//!        ↓
//! [DiT Blocks with Cross-Attention] - cross-attend to T5 embeddings
//!        ↓
//! Output: velocity/noise prediction
//! ```
//!
//! # Key Features
//!
//! - **Cross-attention**: Unlike Flux/SD3, uses cross-attention to text instead of concat
//! - **Efficient**: ~600M params vs multi-billion for other models
//! - **T5 text encoder**: Uses T5-XXL for text understanding
//!
//! # Variants
//!
//! - **PixArt-α**: Original model (512px, 1024px)
//! - **PixArt-Σ**: Improved model with better quality

use burn::module::Param;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

use sketchpad_core::dit::{PatchEmbed, PatchEmbedConfig, unpatchify};
use sketchpad_core::glu::SwiGluFfn;
use sketchpad_core::layernorm::LayerNorm;

/// PixArt Model Configuration
#[derive(Debug, Clone)]
pub struct PixArtConfig {
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
    /// Text embedding dimension (T5-XXL = 4096)
    pub text_dim: usize,
    /// Timestep embedding dimension
    pub time_embed_dim: usize,
    /// FFN intermediate size multiplier
    pub mlp_ratio: f32,
    /// Maximum image sequence length
    pub max_seq_len: usize,
}

impl PixArtConfig {
    /// PixArt-α 512px configuration
    pub fn alpha_512() -> Self {
        Self {
            in_channels: 4,
            patch_size: 2,
            hidden_size: 1152,
            num_heads: 16,
            num_blocks: 28,
            text_dim: 4096, // T5-XXL
            time_embed_dim: 256,
            mlp_ratio: 4.0,
            max_seq_len: 1024, // 32x32 patches
        }
    }

    /// PixArt-α 1024px configuration
    pub fn alpha_1024() -> Self {
        Self {
            in_channels: 4,
            patch_size: 2,
            hidden_size: 1152,
            num_heads: 16,
            num_blocks: 28,
            text_dim: 4096,
            time_embed_dim: 256,
            mlp_ratio: 4.0,
            max_seq_len: 4096, // 64x64 patches
        }
    }

    /// PixArt-Σ configuration (improved model)
    pub fn sigma() -> Self {
        Self {
            in_channels: 4,
            patch_size: 2,
            hidden_size: 1152,
            num_heads: 16,
            num_blocks: 28,
            text_dim: 4096,
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
    pub fn init<B: Backend>(&self, device: &B::Device) -> (PixArt<B>, PixArtRuntime<B>) {
        // Patch embedding
        let patch_embed =
            PatchEmbedConfig::new(self.patch_size, self.in_channels, self.hidden_size).init(device);

        // Timestep embedding
        let time_embed = PixArtTimestepEmbed {
            linear1: LinearConfig::new(self.time_embed_dim, self.hidden_size)
                .with_bias(true)
                .init(device),
            linear2: LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(true)
                .init(device),
            freqs: pixart_timestep_freqs(self.time_embed_dim, device),
        };

        // Position embedding (learned)
        let pos_embed = Tensor::zeros([1, self.max_seq_len, self.hidden_size], device);

        // DiT blocks with cross-attention
        let blocks: Vec<PixArtBlock<B>> = (0..self.num_blocks)
            .map(|_| {
                PixArtBlockConfig::new(
                    self.hidden_size,
                    self.num_heads,
                    self.intermediate_size(),
                    self.text_dim,
                )
                .init(device)
            })
            .collect();

        // Final layer
        let final_layer = PixArtFinalLayer {
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

        let model = PixArt {
            patch_embed,
            time_embed,
            pos_embed: Param::from_tensor(pos_embed),
            blocks,
            final_layer,
            hidden_size: self.hidden_size,
            patch_size: self.patch_size,
            in_channels: self.in_channels,
        };

        let runtime = PixArtRuntime {
            _marker: std::marker::PhantomData,
        };

        (model, runtime)
    }
}

/// Timestep embedding for PixArt
#[derive(Module, Debug)]
pub struct PixArtTimestepEmbed<B: Backend> {
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
    /// Precomputed sinusoidal frequencies
    pub freqs: Tensor<B, 1>,
}

/// Precompute sinusoidal frequencies for PixArt timestep embedding
pub fn pixart_timestep_freqs<B: Backend>(embed_dim: usize, device: &B::Device) -> Tensor<B, 1> {
    let half_dim = embed_dim / 2;
    let emb_scale = -(10000.0_f32.ln()) / (half_dim as f32 - 1.0);
    let freqs: Vec<f32> = (0..half_dim)
        .map(|i| (emb_scale * i as f32).exp())
        .collect();
    Tensor::<B, 1>::from_floats(freqs.as_slice(), device)
}

impl<B: Backend> PixArtTimestepEmbed<B> {
    pub fn forward(&self, t: Tensor<B, 1>) -> Tensor<B, 2> {
        let [_batch] = t.dims();
        let t_expanded = t.unsqueeze_dim::<2>(1);
        let freqs_expanded = self.freqs.clone().unsqueeze_dim::<2>(0);
        let angles = t_expanded.matmul(freqs_expanded);

        let sin_emb = angles.clone().sin();
        let cos_emb = angles.cos();
        let emb = Tensor::cat(vec![sin_emb, cos_emb], 1);

        // MLP
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

/// PixArt Block Configuration
struct PixArtBlockConfig {
    hidden_size: usize,
    num_heads: usize,
    intermediate_size: usize,
    text_dim: usize,
}

impl PixArtBlockConfig {
    fn new(
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
        text_dim: usize,
    ) -> Self {
        Self {
            hidden_size,
            num_heads,
            intermediate_size,
            text_dim,
        }
    }

    fn init<B: Backend>(&self, device: &B::Device) -> PixArtBlock<B> {
        use sketchpad_core::glu::SwiGluFfnConfig;

        let head_dim = self.hidden_size / self.num_heads;

        PixArtBlock {
            norm1: LayerNorm::new(self.hidden_size, device),
            self_attn: PixArtAttention {
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
            cross_attn: PixArtCrossAttention {
                to_q: LinearConfig::new(self.hidden_size, self.hidden_size)
                    .with_bias(true)
                    .init(device),
                to_kv: LinearConfig::new(self.text_dim, 2 * self.hidden_size)
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
            hidden_size: self.hidden_size,
        }
    }
}

/// PixArt Self-Attention
#[derive(Module, Debug)]
pub struct PixArtAttention<B: Backend> {
    pub to_qkv: Linear<B>,
    pub to_out: Linear<B>,
    #[module(skip)]
    pub num_heads: usize,
    #[module(skip)]
    pub head_dim: usize,
}

impl<B: Backend> PixArtAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
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

/// PixArt Cross-Attention to text
#[derive(Module, Debug)]
pub struct PixArtCrossAttention<B: Backend> {
    pub to_q: Linear<B>,
    pub to_kv: Linear<B>,
    pub to_out: Linear<B>,
    #[module(skip)]
    pub num_heads: usize,
    #[module(skip)]
    pub head_dim: usize,
}

impl<B: Backend> PixArtCrossAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>, context: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden] = x.dims();
        let [_, ctx_len, _] = context.dims();

        // Q from image, KV from text
        let q = self.to_q.forward(x);
        let q = q.reshape([batch, seq_len, self.num_heads, self.head_dim]);

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
            .reshape([batch, ctx_len, self.num_heads, self.head_dim]);
        let v = kv
            .slice([
                0..batch,
                0..ctx_len,
                1..2,
                0..self.num_heads,
                0..self.head_dim,
            ])
            .reshape([batch, ctx_len, self.num_heads, self.head_dim]);

        // [batch, heads, seq, dim]
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        // Cross-attention
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

/// PixArt DiT Block with Cross-Attention
#[derive(Module, Debug)]
pub struct PixArtBlock<B: Backend> {
    pub norm1: LayerNorm<B>,
    pub self_attn: PixArtAttention<B>,
    pub norm2: LayerNorm<B>,
    pub cross_attn: PixArtCrossAttention<B>,
    pub norm3: LayerNorm<B>,
    pub ffn: SwiGluFfn<B>,
    pub modulation: Linear<B>,
    #[module(skip)]
    pub hidden_size: usize,
}

impl<B: Backend> PixArtBlock<B> {
    /// Forward pass with AdaLN-Zero modulation
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        cond: Tensor<B, 2>,
        context: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [batch, _seq_len, hidden] = x.dims();

        // Get modulation parameters (6 for: shift1, scale1, gate1, shift2, scale2, gate2)
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
        let x = x + gate1 * self.self_attn.forward(x_norm);

        // Cross-attention (no modulation - direct to text)
        let x_norm = self.norm2.forward(x.clone());
        let x = x + self.cross_attn.forward(x_norm, context);

        // FFN with modulation
        let x_norm = self.norm3.forward(x.clone());
        let x_norm = (Tensor::ones_like(&scale2) + scale2) * x_norm + shift2;
        x + gate2 * self.ffn.forward(x_norm)
    }
}

/// PixArt Final Layer
#[derive(Module, Debug)]
pub struct PixArtFinalLayer<B: Backend> {
    pub norm: LayerNorm<B>,
    pub proj: Linear<B>,
    pub modulation: Linear<B>,
}

impl<B: Backend> PixArtFinalLayer<B> {
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

/// PixArt Model
#[derive(Module, Debug)]
pub struct PixArt<B: Backend> {
    pub patch_embed: PatchEmbed<B>,
    pub time_embed: PixArtTimestepEmbed<B>,
    pub pos_embed: Param<Tensor<B, 3>>,
    pub blocks: Vec<PixArtBlock<B>>,
    pub final_layer: PixArtFinalLayer<B>,
    #[module(skip)]
    pub hidden_size: usize,
    #[module(skip)]
    pub patch_size: usize,
    #[module(skip)]
    pub in_channels: usize,
}

/// Runtime state for PixArt
pub struct PixArtRuntime<B: Backend> {
    _marker: std::marker::PhantomData<B>,
}

/// Output from PixArt
pub struct PixArtOutput<B: Backend> {
    /// Predicted noise or velocity [batch, channels, height, width]
    pub prediction: Tensor<B, 4>,
}

impl<B: Backend> PixArt<B> {
    /// Forward pass
    ///
    /// # Arguments
    ///
    /// * `latents` - Noisy latent images [batch, channels, height, width]
    /// * `timestep` - Diffusion timestep
    /// * `text_embeds` - T5 text embeddings [batch, seq_len, text_dim]
    pub fn forward(
        &self,
        latents: Tensor<B, 4>,
        timestep: f32,
        text_embeds: Tensor<B, 3>,
    ) -> PixArtOutput<B> {
        let [batch, _channels, height, width] = latents.dims();

        // Patchify
        let x = self.patch_embed.forward(latents);
        let [_, seq_len, _] = x.dims();

        // Add position embeddings (truncate if needed)
        let pos_embed = self
            .pos_embed
            .val()
            .slice([0..1, 0..seq_len, 0..self.hidden_size]);
        let pos_embed = pos_embed.repeat_dim(0, batch);
        let x = x + pos_embed;

        // Timestep embedding (using scalar forward to avoid tensor allocation)
        let cond = self.time_embed.forward_scalar(timestep);

        // DiT blocks with cross-attention
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(x, cond.clone(), text_embeds.clone());
        }

        // Final layer
        let out = self.final_layer.forward(x, cond);

        // Unpatchify
        let prediction = unpatchify(out, self.patch_size, height, width, self.in_channels);

        PixArtOutput { prediction }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_pixart_config() {
        let alpha = PixArtConfig::alpha_512();
        assert_eq!(alpha.hidden_size, 1152);
        assert_eq!(alpha.num_blocks, 28);

        let sigma = PixArtConfig::sigma();
        assert_eq!(sigma.hidden_size, 1152);
    }

    #[test]
    fn test_pixart_timestep_embed() {
        let device = Default::default();
        let embed = PixArtTimestepEmbed {
            linear1: LinearConfig::new(64, 256).with_bias(true).init(&device),
            linear2: LinearConfig::new(256, 256).with_bias(true).init(&device),
            freqs: pixart_timestep_freqs(64, &device),
        };

        let t = Tensor::<TestBackend, 1>::from_floats([0.5], &device);
        let out = embed.forward(t);
        assert_eq!(out.dims(), [1, 256]);
    }

    #[test]
    fn test_pixart_self_attention() {
        let device = Default::default();
        let attn = PixArtAttention {
            to_qkv: LinearConfig::new(256, 768).with_bias(true).init(&device),
            to_out: LinearConfig::new(256, 256).with_bias(true).init(&device),
            num_heads: 4,
            head_dim: 64,
        };

        let x = Tensor::<TestBackend, 3>::zeros([2, 16, 256], &device);
        let out = attn.forward(x);
        assert_eq!(out.dims(), [2, 16, 256]);
    }

    #[test]
    fn test_pixart_cross_attention() {
        let device = Default::default();
        let attn = PixArtCrossAttention {
            to_q: LinearConfig::new(256, 256).with_bias(true).init(&device),
            to_kv: LinearConfig::new(128, 512).with_bias(true).init(&device),
            to_out: LinearConfig::new(256, 256).with_bias(true).init(&device),
            num_heads: 4,
            head_dim: 64,
        };

        let x = Tensor::<TestBackend, 3>::zeros([2, 16, 256], &device);
        let ctx = Tensor::<TestBackend, 3>::zeros([2, 8, 128], &device); // text context
        let out = attn.forward(x, ctx);
        assert_eq!(out.dims(), [2, 16, 256]);
    }

    #[test]
    fn test_pixart_block() {
        let device = Default::default();
        let block = PixArtBlockConfig::new(256, 4, 512, 128).init::<TestBackend>(&device);

        let x = Tensor::zeros([2, 16, 256], &device);
        let cond = Tensor::zeros([2, 256], &device);
        let ctx = Tensor::zeros([2, 8, 128], &device);

        let out = block.forward(x, cond, ctx);
        assert_eq!(out.dims(), [2, 16, 256]);
    }

    #[test]
    fn test_pixart_tiny_forward() {
        let device = Default::default();
        let config = PixArtConfig::tiny();
        let (model, _runtime) = config.init::<TestBackend>(&device);

        // [batch=1, channels=4, height=8, width=8]
        let latents = Tensor::zeros([1, 4, 8, 8], &device);
        let text = Tensor::zeros([1, 4, 128], &device); // T5 embeddings

        let output = model.forward(latents, 0.5, text);

        assert_eq!(output.prediction.dims(), [1, 4, 8, 8]);
    }
}
