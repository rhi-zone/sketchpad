//! Flux Model Implementation
//!
//! Flux is a DiT-based image generation model from Black Forest Labs.
//! It uses flow matching with a dual-stream then single-stream transformer architecture.
//!
//! # Architecture Overview
//!
//! ```text
//! Input: image patches + text embeddings + timestep
//!        ↓
//! [Dual-stream blocks] - text and image processed separately with cross-attention
//!        ↓
//! [Single-stream blocks] - merged text+image tokens
//!        ↓
//! Output: velocity prediction for flow matching
//! ```
//!
//! # Variants
//!
//! - **Flux.1-dev**: Guidance-distilled, higher quality
//! - **Flux.1-schnell**: Few-step generation (4 steps)

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

use sketchpad_core::dit::{PatchEmbed, PatchEmbedConfig, unpatchify};
use sketchpad_core::glu::SwiGluFfn;
use sketchpad_core::layernorm::LayerNorm;
use sketchpad_core::rope::RotaryEmbedding;

/// Flux Model Configuration
#[derive(Debug, Clone)]
pub struct FluxConfig {
    /// Number of input image channels (usually 16 for VAE latents)
    pub in_channels: usize,
    /// Patch size for patchifying the image
    pub patch_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of dual-stream (parallel) transformer blocks
    pub num_double_blocks: usize,
    /// Number of single-stream (merged) transformer blocks
    pub num_single_blocks: usize,
    /// Text embedding dimension (from T5/CLIP)
    pub text_dim: usize,
    /// Timestep embedding dimension
    pub time_dim: usize,
    /// Guidance embedding dimension (for guidance-distilled models)
    pub guidance_dim: usize,
    /// Maximum sequence length for RoPE
    pub max_seq_len: usize,
    /// FFN intermediate size multiplier
    pub mlp_ratio: f32,
}

impl FluxConfig {
    /// Flux.1-dev configuration
    pub fn flux_dev() -> Self {
        Self {
            in_channels: 16,
            patch_size: 2,
            hidden_size: 3072,
            num_heads: 24,
            num_double_blocks: 19,
            num_single_blocks: 38,
            text_dim: 4096, // T5-XXL
            time_dim: 256,
            guidance_dim: 256,
            max_seq_len: 4096,
            mlp_ratio: 4.0,
        }
    }

    /// Flux.1-schnell configuration (same architecture)
    pub fn flux_schnell() -> Self {
        Self::flux_dev()
    }

    /// Tiny model for testing
    pub fn tiny() -> Self {
        Self {
            in_channels: 4,
            patch_size: 2,
            hidden_size: 256,
            num_heads: 4,
            num_double_blocks: 2,
            num_single_blocks: 2,
            text_dim: 128,
            time_dim: 64,
            guidance_dim: 64,
            max_seq_len: 512,
            mlp_ratio: 4.0,
        }
    }

    /// Initialize the model
    pub fn init<B: Backend>(&self, device: &B::Device) -> (Flux<B>, FluxRuntime<B>) {
        let head_dim = self.hidden_size / self.num_heads;
        let intermediate_size = (self.hidden_size as f32 * self.mlp_ratio) as usize;

        // Patch embedding for image
        let img_embed =
            PatchEmbedConfig::new(self.patch_size, self.in_channels, self.hidden_size).init(device);

        // Text projection (T5 dim → hidden dim)
        let txt_embed = LinearConfig::new(self.text_dim, self.hidden_size)
            .with_bias(true)
            .init(device);

        // Timestep embedding MLP
        let time_embed = TimestepEmbedding {
            linear1: LinearConfig::new(self.time_dim, self.hidden_size)
                .with_bias(true)
                .init(device),
            linear2: LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(true)
                .init(device),
            freqs: timestep_freqs(self.time_dim, device),
        };

        // Guidance embedding (for dev model)
        let guidance_embed = Some(TimestepEmbedding {
            linear1: LinearConfig::new(self.guidance_dim, self.hidden_size)
                .with_bias(true)
                .init(device),
            linear2: LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(true)
                .init(device),
            freqs: timestep_freqs(self.guidance_dim, device),
        });

        // Dual-stream blocks
        let double_blocks: Vec<FluxDoubleBlock<B>> = (0..self.num_double_blocks)
            .map(|_| {
                FluxDoubleBlockConfig::new(self.hidden_size, self.num_heads, intermediate_size)
                    .init(device)
            })
            .collect();

        // Single-stream blocks
        let single_blocks: Vec<FluxSingleBlock<B>> = (0..self.num_single_blocks)
            .map(|_| {
                FluxSingleBlockConfig::new(self.hidden_size, self.num_heads, intermediate_size)
                    .init(device)
            })
            .collect();

        // Final layer
        let final_layer = FinalLayer {
            norm: LayerNorm::new(self.hidden_size, device),
            proj: LinearConfig::new(
                self.hidden_size,
                self.patch_size * self.patch_size * self.in_channels,
            )
            .with_bias(true)
            .init(device),
            ada_ln_modulation: LinearConfig::new(self.hidden_size, 2 * self.hidden_size)
                .with_bias(true)
                .init(device),
        };

        let model = Flux {
            img_embed,
            txt_embed,
            time_embed,
            guidance_embed,
            double_blocks,
            single_blocks,
            final_layer,
            patch_size: self.patch_size,
            in_channels: self.in_channels,
        };

        let runtime = FluxRuntime {
            rope: RotaryEmbedding::new(head_dim, self.max_seq_len, device),
            config: self.clone(),
        };

        (model, runtime)
    }
}

/// Timestep embedding MLP
///
/// Converts scalar timestep to embedding via sinusoidal encoding + MLP
#[derive(Module, Debug)]
pub struct TimestepEmbedding<B: Backend> {
    /// First linear
    pub linear1: Linear<B>,
    /// Second linear
    pub linear2: Linear<B>,
    /// Precomputed sinusoidal frequencies (avoids f32->tensor conversion in forward)
    pub freqs: Tensor<B, 1>,
}

/// Precompute sinusoidal frequencies for timestep embedding
pub fn timestep_freqs<B: Backend>(embed_dim: usize, device: &B::Device) -> Tensor<B, 1> {
    let half_dim = embed_dim / 2;
    let emb_scale = -(2.0_f32.ln()) / (half_dim as f32 - 1.0);

    let freqs: Vec<f32> = (0..half_dim)
        .map(|i| (emb_scale * i as f32).exp())
        .collect();
    Tensor::<B, 1>::from_floats(freqs.as_slice(), device)
}

impl<B: Backend> TimestepEmbedding<B> {
    /// Forward pass for batched timesteps
    ///
    /// Uses sinusoidal positional encoding followed by MLP
    pub fn forward(&self, t: Tensor<B, 1>) -> Tensor<B, 2> {
        // t * freqs (freqs precomputed at init time)
        let [_batch] = t.dims();
        let t_expanded = t.unsqueeze_dim::<2>(1); // [batch, 1]
        let freqs_expanded = self.freqs.clone().unsqueeze_dim::<2>(0); // [1, half_dim]
        let angles = t_expanded.matmul(freqs_expanded); // [batch, half_dim]

        // Concatenate sin and cos
        let sin_emb = angles.clone().sin();
        let cos_emb = angles.cos();
        let emb = Tensor::cat(vec![sin_emb, cos_emb], 1); // [batch, embed_dim]

        // MLP
        let x = self.linear1.forward(emb);
        let x = burn::tensor::activation::silu(x);
        self.linear2.forward(x)
    }

    /// Forward pass for a single scalar timestep (no tensor allocation)
    ///
    /// More efficient than `forward` when processing one timestep at a time.
    pub fn forward_scalar(&self, t: f32) -> Tensor<B, 2> {
        // Multiply precomputed freqs by scalar directly (no tensor creation)
        let angles = self.freqs.clone() * t;

        // Concatenate sin and cos
        let sin_emb = angles.clone().sin();
        let cos_emb = angles.cos();
        let emb = Tensor::cat(vec![sin_emb, cos_emb], 0).unsqueeze_dim(0); // [1, embed_dim]

        // MLP
        let x = self.linear1.forward(emb);
        let x = burn::tensor::activation::silu(x);
        self.linear2.forward(x)
    }
}

/// Flux double-stream (parallel) block configuration
pub struct FluxDoubleBlockConfig {
    hidden_size: usize,
    num_heads: usize,
    intermediate_size: usize,
}

impl FluxDoubleBlockConfig {
    fn new(hidden_size: usize, num_heads: usize, intermediate_size: usize) -> Self {
        Self {
            hidden_size,
            num_heads,
            intermediate_size,
        }
    }

    fn init<B: Backend>(&self, device: &B::Device) -> FluxDoubleBlock<B> {
        use sketchpad_core::glu::SwiGluFfnConfig;

        let head_dim = self.hidden_size / self.num_heads;

        FluxDoubleBlock {
            // Image stream
            img_norm1: LayerNorm::new(self.hidden_size, device),
            img_attn: FluxAttention {
                qkv: LinearConfig::new(self.hidden_size, 3 * self.hidden_size)
                    .with_bias(true)
                    .init(device),
                proj: LinearConfig::new(self.hidden_size, self.hidden_size)
                    .with_bias(true)
                    .init(device),
                num_heads: self.num_heads,
                head_dim,
            },
            img_norm2: LayerNorm::new(self.hidden_size, device),
            img_ffn: SwiGluFfnConfig::new(self.hidden_size, self.intermediate_size).init(device),

            // Text stream
            txt_norm1: LayerNorm::new(self.hidden_size, device),
            txt_attn: FluxAttention {
                qkv: LinearConfig::new(self.hidden_size, 3 * self.hidden_size)
                    .with_bias(true)
                    .init(device),
                proj: LinearConfig::new(self.hidden_size, self.hidden_size)
                    .with_bias(true)
                    .init(device),
                num_heads: self.num_heads,
                head_dim,
            },
            txt_norm2: LayerNorm::new(self.hidden_size, device),
            txt_ffn: SwiGluFfnConfig::new(self.hidden_size, self.intermediate_size).init(device),

            // Modulation
            modulation: LinearConfig::new(self.hidden_size, 6 * self.hidden_size)
                .with_bias(true)
                .init(device),
        }
    }
}

/// Flux attention layer
#[derive(Module, Debug)]
pub struct FluxAttention<B: Backend> {
    /// QKV projection
    pub qkv: Linear<B>,
    /// Output projection
    pub proj: Linear<B>,
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension
    pub head_dim: usize,
}

impl<B: Backend> FluxAttention<B> {
    /// Forward with optional cross-attention
    fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        cross_x: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden] = x.dims();

        // Compute QKV
        let qkv = self.qkv.forward(x);
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

        // Apply RoPE
        let q = q.swap_dims(1, 2); // [batch, heads, seq, dim]
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        let (q, k) = rope.forward(q, k, 0);

        // Joint attention with cross_x if provided
        let (k, v) = if let Some(cross) = cross_x {
            let cross_qkv = self.qkv.forward(cross.clone());
            let [_b, cross_len, _] = cross.dims();
            let cross_qkv = cross_qkv.reshape([batch, cross_len, 3, self.num_heads, self.head_dim]);

            let cross_k = cross_qkv
                .clone()
                .slice([
                    0..batch,
                    0..cross_len,
                    1..2,
                    0..self.num_heads,
                    0..self.head_dim,
                ])
                .reshape([batch, cross_len, self.num_heads, self.head_dim])
                .swap_dims(1, 2);
            let cross_v = cross_qkv
                .slice([
                    0..batch,
                    0..cross_len,
                    2..3,
                    0..self.num_heads,
                    0..self.head_dim,
                ])
                .reshape([batch, cross_len, self.num_heads, self.head_dim])
                .swap_dims(1, 2);

            // Concatenate along sequence dimension
            (
                Tensor::cat(vec![k, cross_k], 2),
                Tensor::cat(vec![v, cross_v], 2),
            )
        } else {
            (k, v)
        };

        // Scaled dot-product attention
        let scale = (self.head_dim as f32).sqrt().recip();
        let attn = q.matmul(k.swap_dims(2, 3)) * scale;
        let attn = burn::tensor::activation::softmax(attn, 3);
        let out = attn.matmul(v);

        // Reshape back
        let out = out
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * self.head_dim]);
        self.proj.forward(out)
    }
}

/// Flux dual-stream block
///
/// Processes text and image streams in parallel with cross-attention
#[derive(Module, Debug)]
pub struct FluxDoubleBlock<B: Backend> {
    // Image stream
    pub img_norm1: LayerNorm<B>,
    pub img_attn: FluxAttention<B>,
    pub img_norm2: LayerNorm<B>,
    pub img_ffn: SwiGluFfn<B>,

    // Text stream
    pub txt_norm1: LayerNorm<B>,
    pub txt_attn: FluxAttention<B>,
    pub txt_norm2: LayerNorm<B>,
    pub txt_ffn: SwiGluFfn<B>,

    // Modulation from timestep
    pub modulation: Linear<B>,
}

impl<B: Backend> FluxDoubleBlock<B> {
    /// Forward pass
    ///
    /// Returns (img_out, txt_out)
    pub fn forward(
        &self,
        img: Tensor<B, 3>,
        txt: Tensor<B, 3>,
        cond: Tensor<B, 2>,
        rope: &RotaryEmbedding<B>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [batch, _img_len, hidden] = img.dims();
        let [_, _txt_len, _] = txt.dims();

        // Get modulation parameters (6 sets: shift, scale, gate for both streams)
        let mod_params = self.modulation.forward(cond);
        let mod_params = mod_params.reshape([batch, 1, 6, hidden]);

        // Image stream with modulation
        let img_mod1_shift = mod_params
            .clone()
            .slice([0..batch, 0..1, 0..1, 0..hidden])
            .reshape([batch, 1, hidden]);
        let img_mod1_scale = mod_params
            .clone()
            .slice([0..batch, 0..1, 1..2, 0..hidden])
            .reshape([batch, 1, hidden]);
        let img_gate = mod_params
            .clone()
            .slice([0..batch, 0..1, 2..3, 0..hidden])
            .reshape([batch, 1, hidden]);

        // Modulated image norm
        let img_normed = self.img_norm1.forward(img.clone());
        let img_normed =
            (Tensor::ones_like(&img_mod1_scale) + img_mod1_scale) * img_normed + img_mod1_shift;

        // Image self-attention with cross-attention to text
        let img_attn_out = self.img_attn.forward(img_normed, rope, Some(txt.clone()));
        let img = img + img_gate.clone() * img_attn_out;

        // Image FFN
        let img = img.clone() + img_gate * self.img_ffn.forward(self.img_norm2.forward(img));

        // Text stream with modulation
        let txt_mod1_shift = mod_params
            .clone()
            .slice([0..batch, 0..1, 3..4, 0..hidden])
            .reshape([batch, 1, hidden]);
        let txt_mod1_scale = mod_params
            .clone()
            .slice([0..batch, 0..1, 4..5, 0..hidden])
            .reshape([batch, 1, hidden]);
        let txt_gate = mod_params
            .slice([0..batch, 0..1, 5..6, 0..hidden])
            .reshape([batch, 1, hidden]);

        // Modulated text norm
        let txt_normed = self.txt_norm1.forward(txt.clone());
        let txt_normed =
            (Tensor::ones_like(&txt_mod1_scale) + txt_mod1_scale) * txt_normed + txt_mod1_shift;

        // Text self-attention with cross-attention to image
        let txt_attn_out = self.txt_attn.forward(txt_normed, rope, Some(img.clone()));
        let txt = txt + txt_gate.clone() * txt_attn_out;

        // Text FFN
        let txt = txt.clone() + txt_gate * self.txt_ffn.forward(self.txt_norm2.forward(txt));

        (img, txt)
    }
}

/// Flux single-stream block configuration
pub struct FluxSingleBlockConfig {
    hidden_size: usize,
    num_heads: usize,
    intermediate_size: usize,
}

impl FluxSingleBlockConfig {
    fn new(hidden_size: usize, num_heads: usize, intermediate_size: usize) -> Self {
        Self {
            hidden_size,
            num_heads,
            intermediate_size,
        }
    }

    fn init<B: Backend>(&self, device: &B::Device) -> FluxSingleBlock<B> {
        use sketchpad_core::glu::SwiGluFfnConfig;

        let head_dim = self.hidden_size / self.num_heads;

        FluxSingleBlock {
            norm: LayerNorm::new(self.hidden_size, device),
            attn: FluxAttention {
                qkv: LinearConfig::new(self.hidden_size, 3 * self.hidden_size)
                    .with_bias(true)
                    .init(device),
                proj: LinearConfig::new(self.hidden_size, self.hidden_size)
                    .with_bias(true)
                    .init(device),
                num_heads: self.num_heads,
                head_dim,
            },
            ffn: SwiGluFfnConfig::new(self.hidden_size, self.intermediate_size).init(device),
            modulation: LinearConfig::new(self.hidden_size, 3 * self.hidden_size)
                .with_bias(true)
                .init(device),
        }
    }
}

/// Flux single-stream block
///
/// Processes merged text+image tokens
#[derive(Module, Debug)]
pub struct FluxSingleBlock<B: Backend> {
    pub norm: LayerNorm<B>,
    pub attn: FluxAttention<B>,
    pub ffn: SwiGluFfn<B>,
    pub modulation: Linear<B>,
}

impl<B: Backend> FluxSingleBlock<B> {
    /// Forward pass
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        cond: Tensor<B, 2>,
        rope: &RotaryEmbedding<B>,
    ) -> Tensor<B, 3> {
        let [batch, _seq_len, hidden] = x.dims();

        // Get modulation (shift, scale, gate)
        let mod_params = self.modulation.forward(cond);
        let mod_params = mod_params.reshape([batch, 1, 3, hidden]);

        let shift = mod_params
            .clone()
            .slice([0..batch, 0..1, 0..1, 0..hidden])
            .reshape([batch, 1, hidden]);
        let scale = mod_params
            .clone()
            .slice([0..batch, 0..1, 1..2, 0..hidden])
            .reshape([batch, 1, hidden]);
        let gate = mod_params
            .slice([0..batch, 0..1, 2..3, 0..hidden])
            .reshape([batch, 1, hidden]);

        // Modulated norm
        let x_normed = self.norm.forward(x.clone());
        let x_normed = (Tensor::ones_like(&scale) + scale) * x_normed + shift;

        // Self-attention
        let attn_out = self.attn.forward(x_normed.clone(), rope, None);
        let x = x + gate.clone() * attn_out;

        // FFN
        x.clone() + gate * self.ffn.forward(self.norm.forward(x))
    }
}

/// Final layer for output projection
#[derive(Module, Debug)]
pub struct FinalLayer<B: Backend> {
    pub norm: LayerNorm<B>,
    pub proj: Linear<B>,
    pub ada_ln_modulation: Linear<B>,
}

impl<B: Backend> FinalLayer<B> {
    /// Forward pass
    pub fn forward(&self, x: Tensor<B, 3>, cond: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch, _seq_len, hidden] = x.dims();

        // Get modulation (shift, scale)
        let mod_params = self.ada_ln_modulation.forward(cond);
        let mod_params = mod_params.reshape([batch, 1, hidden * 2]);

        let shift = mod_params.clone().slice([0..batch, 0..1, 0..hidden]);
        let scale = mod_params.slice([0..batch, 0..1, hidden..(hidden * 2)]);

        // Apply modulated norm
        let x = self.norm.forward(x);
        let x = (Tensor::ones_like(&scale) + scale) * x + shift;

        // Project to patch output
        self.proj.forward(x)
    }
}

/// Flux Model
#[derive(Module, Debug)]
pub struct Flux<B: Backend> {
    /// Image patch embedding
    pub img_embed: PatchEmbed<B>,
    /// Text embedding projection
    pub txt_embed: Linear<B>,
    /// Timestep embedding
    pub time_embed: TimestepEmbedding<B>,
    /// Guidance embedding (optional, for dev model)
    pub guidance_embed: Option<TimestepEmbedding<B>>,
    /// Dual-stream transformer blocks
    pub double_blocks: Vec<FluxDoubleBlock<B>>,
    /// Single-stream transformer blocks
    pub single_blocks: Vec<FluxSingleBlock<B>>,
    /// Final output layer
    pub final_layer: FinalLayer<B>,
    /// Patch size
    pub patch_size: usize,
    /// Input channels
    pub in_channels: usize,
}

/// Runtime state for Flux
pub struct FluxRuntime<B: Backend> {
    /// Rotary embeddings
    pub rope: RotaryEmbedding<B>,
    /// Configuration
    pub config: FluxConfig,
}

/// Output from Flux model
pub struct FluxOutput<B: Backend> {
    /// Velocity prediction for flow matching [batch, channels, height, width]
    pub velocity: Tensor<B, 4>,
}

impl<B: Backend> Flux<B> {
    /// Forward pass
    ///
    /// # Arguments
    ///
    /// * `latents` - Noisy latents [batch, channels, height, width]
    /// * `timestep` - Diffusion timestep (0 to 1 for flow matching)
    /// * `txt_embeds` - Text embeddings from T5 [batch, seq_len, text_dim]
    /// * `guidance` - Optional guidance scale (for dev model)
    /// * `runtime` - Runtime state with RoPE
    pub fn forward(
        &self,
        latents: Tensor<B, 4>,
        timestep: f32,
        txt_embeds: Tensor<B, 3>,
        guidance: Option<f32>,
        runtime: &FluxRuntime<B>,
    ) -> FluxOutput<B> {
        let [batch, _channels, height, width] = latents.dims();

        // Patchify image
        let img = self.img_embed.forward(latents);
        let [_, img_len, _] = img.dims();

        // Project text embeddings
        let txt = self.txt_embed.forward(txt_embeds);
        let [_, _txt_len, _] = txt.dims();

        // Timestep embedding (using scalar forward to avoid tensor allocation)
        let cond = self.time_embed.forward_scalar(timestep);

        // Add guidance embedding if present
        let cond = if let (Some(g_embed), Some(g)) = (&self.guidance_embed, guidance) {
            cond + g_embed.forward_scalar(g)
        } else {
            cond
        };

        // Dual-stream blocks
        let (mut img, mut txt) = (img, txt);
        for block in &self.double_blocks {
            let (new_img, new_txt) = block.forward(img, txt, cond.clone(), &runtime.rope);
            img = new_img;
            txt = new_txt;
        }

        // Merge streams for single-stream blocks
        let merged = Tensor::cat(vec![img, txt], 1);

        // Single-stream blocks
        let mut x = merged;
        for block in &self.single_blocks {
            x = block.forward(x, cond.clone(), &runtime.rope);
        }

        // Extract image tokens (first img_len tokens)
        let img_out = x.slice([0..batch, 0..img_len, 0..runtime.config.hidden_size]);

        // Final layer
        let out = self.final_layer.forward(img_out, cond);

        // Unpatchify to image
        let velocity = unpatchify(out, self.patch_size, height, width, self.in_channels);

        FluxOutput { velocity }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_flux_tiny_forward() {
        let device = Default::default();
        let config = FluxConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        // 32x32 latents with 4 channels, patch_size=2 → 16x16 = 256 patches
        let latents = Tensor::zeros([1, 4, 32, 32], &device);
        let txt = Tensor::zeros([1, 8, 128], &device); // 8 text tokens

        let output = model.forward(latents, 0.5, txt, Some(3.5), &runtime);

        assert_eq!(output.velocity.dims(), [1, 4, 32, 32]);
    }

    #[test]
    fn test_timestep_embedding() {
        let device = Default::default();

        let embed = TimestepEmbedding {
            linear1: LinearConfig::new(64, 256).with_bias(true).init(&device),
            linear2: LinearConfig::new(256, 256).with_bias(true).init(&device),
            freqs: timestep_freqs(64, &device),
        };

        let t = Tensor::<TestBackend, 1>::from_floats([0.5], &device);
        let out = embed.forward(t);

        assert_eq!(out.dims(), [1, 256]);
    }

    #[test]
    fn test_flux_configs() {
        let _ = FluxConfig::flux_dev();
        let _ = FluxConfig::flux_schnell();
        let _ = FluxConfig::tiny();
    }
}
