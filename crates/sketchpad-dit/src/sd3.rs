//! Stable Diffusion 3 (SD3) Model Implementation
//!
//! SD3 uses MMDiT (Multimodal Diffusion Transformer) - a joint transformer
//! architecture where text and image tokens are processed together.
//!
//! # Architecture Overview
//!
//! ```text
//! Input: image patches + text embeddings + pooled embeddings + timestep
//!        ↓
//! [MMDiT Joint Blocks] - text and image processed together with joint attention
//!        ↓
//! Output: velocity prediction for rectified flow
//! ```
//!
//! # Key Differences from Flux
//!
//! - **Joint Attention**: All tokens processed together (vs dual-stream in Flux)
//! - **Pooled Conditioning**: Uses pooled text embeddings for modulation
//! - **Text Encoders**: CLIP-L + CLIP-G + T5-XXL (vs just T5 in Flux)
//!
//! # Variants
//!
//! - **SD3-Medium**: 2B params, good balance of quality and speed
//! - **SD3-Large**: 8B params, higher quality
//! - **SD3.5-Large**: Latest iteration with improved quality
//! - **SD3.5-Turbo**: Distilled for fast generation

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

use sketchpad_core::dit::{PatchEmbed, PatchEmbedConfig, unpatchify};
use sketchpad_core::glu::SwiGluFfn;
use sketchpad_core::layernorm::LayerNorm;
use sketchpad_core::rope::RotaryEmbedding;

/// SD3 Model Configuration
#[derive(Debug, Clone)]
pub struct Sd3Config {
    /// Number of input image channels (usually 16 for VAE latents)
    pub in_channels: usize,
    /// Patch size for patchifying the image
    pub patch_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of MMDiT joint blocks
    pub num_blocks: usize,
    /// Context dimension (from T5)
    pub context_dim: usize,
    /// Pooled projection dimension (CLIP-L + CLIP-G combined)
    pub pooled_dim: usize,
    /// Timestep embedding dimension
    pub time_embed_dim: usize,
    /// Maximum sequence length for RoPE
    pub max_seq_len: usize,
    /// FFN intermediate size multiplier
    pub mlp_ratio: f32,
    /// QK normalization (used in SD3)
    pub qk_norm: bool,
}

impl Sd3Config {
    /// SD3-Medium configuration (2B params)
    pub fn sd3_medium() -> Self {
        Self {
            in_channels: 16,
            patch_size: 2,
            hidden_size: 1536,
            num_heads: 24,
            num_blocks: 24,
            context_dim: 4096, // T5-XXL
            pooled_dim: 2048,  // CLIP-L (768) + CLIP-G (1280)
            time_embed_dim: 256,
            max_seq_len: 4096,
            mlp_ratio: 4.0,
            qk_norm: true,
        }
    }

    /// SD3-Large configuration (8B params)
    pub fn sd3_large() -> Self {
        Self {
            in_channels: 16,
            patch_size: 2,
            hidden_size: 2048,
            num_heads: 32,
            num_blocks: 38,
            context_dim: 4096,
            pooled_dim: 2048,
            time_embed_dim: 256,
            max_seq_len: 4096,
            mlp_ratio: 4.0,
            qk_norm: true,
        }
    }

    /// SD3.5-Large configuration
    pub fn sd3_5_large() -> Self {
        Self::sd3_large()
    }

    /// SD3.5-Large-Turbo (distilled)
    pub fn sd3_5_turbo() -> Self {
        Self::sd3_large()
    }

    /// Tiny model for testing
    pub fn tiny() -> Self {
        Self {
            in_channels: 4,
            patch_size: 2,
            hidden_size: 256,
            num_heads: 4,
            num_blocks: 4,
            context_dim: 128,
            pooled_dim: 128,
            time_embed_dim: 64,
            max_seq_len: 512,
            mlp_ratio: 4.0,
            qk_norm: false,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    pub fn intermediate_size(&self) -> usize {
        (self.hidden_size as f32 * self.mlp_ratio) as usize
    }

    /// Initialize the model
    pub fn init<B: Backend>(&self, device: &B::Device) -> (Sd3<B>, Sd3Runtime<B>) {
        // Patch embedding for image
        let x_embed =
            PatchEmbedConfig::new(self.patch_size, self.in_channels, self.hidden_size).init(device);

        // Context projection (T5 dim → hidden dim)
        let context_embed = LinearConfig::new(self.context_dim, self.hidden_size)
            .with_bias(true)
            .init(device);

        // Pooled text embedding projection
        let y_embed = Sd3TimestepEmbedding {
            linear1: LinearConfig::new(self.pooled_dim, self.hidden_size)
                .with_bias(true)
                .init(device),
            linear2: LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(true)
                .init(device),
        };

        // Timestep embedding
        let t_embed = Sd3TimestepEmbedding {
            linear1: LinearConfig::new(self.time_embed_dim, self.hidden_size)
                .with_bias(true)
                .init(device),
            linear2: LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(true)
                .init(device),
        };

        // MMDiT joint blocks
        let blocks: Vec<MMDiTBlock<B>> = (0..self.num_blocks)
            .map(|_| {
                MMDiTBlockConfig::new(
                    self.hidden_size,
                    self.num_heads,
                    self.intermediate_size(),
                    self.qk_norm,
                )
                .init(device)
            })
            .collect();

        // Final layer
        let final_layer = Sd3FinalLayer {
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

        let model = Sd3 {
            x_embed,
            context_embed,
            y_embed,
            t_embed,
            blocks,
            final_layer,
            patch_size: self.patch_size,
            in_channels: self.in_channels,
            hidden_size: self.hidden_size,
        };

        let runtime = Sd3Runtime {
            rope: RotaryEmbedding::new(self.head_dim(), self.max_seq_len, device),
            time_embed_dim: self.time_embed_dim,
            time_freqs: timestep_freqs(self.time_embed_dim, device),
        };

        (model, runtime)
    }
}

/// Timestep embedding MLP for SD3
#[derive(Module, Debug)]
pub struct Sd3TimestepEmbedding<B: Backend> {
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
}

impl<B: Backend> Sd3TimestepEmbedding<B> {
    /// Forward pass
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(x);
        let x = burn::tensor::activation::silu(x);
        self.linear2.forward(x)
    }
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

/// Get sinusoidal timestep embedding using precomputed frequencies (fast path)
pub fn get_timestep_embedding_with_freqs<B: Backend>(
    timestep: f32,
    freqs: &Tensor<B, 1>,
) -> Tensor<B, 2> {
    let angles = freqs.clone() * timestep;
    let sin_emb = angles.clone().sin();
    let cos_emb = angles.cos();

    Tensor::cat(vec![sin_emb, cos_emb], 0).unsqueeze_dim(0)
}

/// Get sinusoidal timestep embedding (slow path - computes freqs each call)
///
/// Note: For hot paths, prefer `get_timestep_embedding_with_freqs` with precomputed freqs.
pub fn get_timestep_embedding<B: Backend>(
    timestep: f32,
    embed_dim: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let freqs = timestep_freqs(embed_dim, device);
    get_timestep_embedding_with_freqs(timestep, &freqs)
}

/// MMDiT Block Configuration
pub struct MMDiTBlockConfig {
    hidden_size: usize,
    num_heads: usize,
    intermediate_size: usize,
    qk_norm: bool,
}

impl MMDiTBlockConfig {
    fn new(hidden_size: usize, num_heads: usize, intermediate_size: usize, qk_norm: bool) -> Self {
        Self {
            hidden_size,
            num_heads,
            intermediate_size,
            qk_norm,
        }
    }

    fn init<B: Backend>(&self, device: &B::Device) -> MMDiTBlock<B> {
        use sketchpad_core::glu::SwiGluFfnConfig;

        let head_dim = self.hidden_size / self.num_heads;

        // Image (x) stream
        let x_norm1 = LayerNorm::new(self.hidden_size, device);
        let x_attn = MMDiTAttention {
            to_q: LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(true)
                .init(device),
            to_k: LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(true)
                .init(device),
            to_v: LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(true)
                .init(device),
            to_out: LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(true)
                .init(device),
            q_norm: if self.qk_norm {
                Some(LayerNorm::new(head_dim, device))
            } else {
                None
            },
            k_norm: if self.qk_norm {
                Some(LayerNorm::new(head_dim, device))
            } else {
                None
            },
            num_heads: self.num_heads,
            head_dim,
        };
        let x_norm2 = LayerNorm::new(self.hidden_size, device);
        let x_ffn = SwiGluFfnConfig::new(self.hidden_size, self.intermediate_size).init(device);

        // Context (c) stream
        let c_norm1 = LayerNorm::new(self.hidden_size, device);
        let c_attn = MMDiTAttention {
            to_q: LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(true)
                .init(device),
            to_k: LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(true)
                .init(device),
            to_v: LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(true)
                .init(device),
            to_out: LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(true)
                .init(device),
            q_norm: if self.qk_norm {
                Some(LayerNorm::new(head_dim, device))
            } else {
                None
            },
            k_norm: if self.qk_norm {
                Some(LayerNorm::new(head_dim, device))
            } else {
                None
            },
            num_heads: self.num_heads,
            head_dim,
        };
        let c_norm2 = LayerNorm::new(self.hidden_size, device);
        let c_ffn = SwiGluFfnConfig::new(self.hidden_size, self.intermediate_size).init(device);

        // Modulation: predicts (shift, scale, gate) for x and c (6 total for pre-attn/ffn)
        let modulation = LinearConfig::new(self.hidden_size, 6 * self.hidden_size)
            .with_bias(true)
            .init(device);

        MMDiTBlock {
            x_norm1,
            x_attn,
            x_norm2,
            x_ffn,
            c_norm1,
            c_attn,
            c_norm2,
            c_ffn,
            modulation,
            hidden_size: self.hidden_size,
        }
    }
}

/// MMDiT Attention with optional QK normalization
#[derive(Module, Debug)]
pub struct MMDiTAttention<B: Backend> {
    pub to_q: Linear<B>,
    pub to_k: Linear<B>,
    pub to_v: Linear<B>,
    pub to_out: Linear<B>,
    pub q_norm: Option<LayerNorm<B>>,
    pub k_norm: Option<LayerNorm<B>>,
    #[module(skip)]
    pub num_heads: usize,
    #[module(skip)]
    pub head_dim: usize,
}

impl<B: Backend> MMDiTAttention<B> {
    /// Get Q, K, V projections for this stream
    fn get_qkv(&self, x: Tensor<B, 3>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, seq_len, _hidden] = x.dims();

        let q = self.to_q.forward(x.clone());
        let k = self.to_k.forward(x.clone());
        let v = self.to_v.forward(x);

        // Reshape to [batch, seq, heads, head_dim]
        let q = q.reshape([batch, seq_len, self.num_heads, self.head_dim]);
        let k = k.reshape([batch, seq_len, self.num_heads, self.head_dim]);
        let v = v.reshape([batch, seq_len, self.num_heads, self.head_dim]);

        // Apply QK normalization if enabled
        let q = if let Some(ref norm) = self.q_norm {
            // Normalize per head - reshape to 3D for LayerNorm then back
            let q_flat = q.reshape([batch * seq_len * self.num_heads, 1, self.head_dim]);
            let q_norm = norm.forward(q_flat);
            q_norm.reshape([batch, seq_len, self.num_heads, self.head_dim])
        } else {
            q
        };

        let k = if let Some(ref norm) = self.k_norm {
            let k_flat = k.reshape([batch * seq_len * self.num_heads, 1, self.head_dim]);
            let k_norm = norm.forward(k_flat);
            k_norm.reshape([batch, seq_len, self.num_heads, self.head_dim])
        } else {
            k
        };

        // Transpose to [batch, heads, seq, head_dim]
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        (q, k, v)
    }

    /// Project attention output back
    fn proj_out(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch, _heads, seq_len, _head_dim] = x.dims();
        let x = x
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * self.head_dim]);
        self.to_out.forward(x)
    }
}

/// MMDiT Joint Block
///
/// Processes image (x) and context (c) tokens together with joint attention
#[derive(Module, Debug)]
pub struct MMDiTBlock<B: Backend> {
    // Image stream
    pub x_norm1: LayerNorm<B>,
    pub x_attn: MMDiTAttention<B>,
    pub x_norm2: LayerNorm<B>,
    pub x_ffn: SwiGluFfn<B>,

    // Context stream
    pub c_norm1: LayerNorm<B>,
    pub c_attn: MMDiTAttention<B>,
    pub c_norm2: LayerNorm<B>,
    pub c_ffn: SwiGluFfn<B>,

    // Modulation
    pub modulation: Linear<B>,

    #[module(skip)]
    pub hidden_size: usize,
}

impl<B: Backend> MMDiTBlock<B> {
    /// Forward pass with joint attention
    ///
    /// # Arguments
    /// * `x` - Image tokens [batch, x_len, hidden]
    /// * `c` - Context tokens [batch, c_len, hidden]
    /// * `y` - Conditioning vector [batch, hidden]
    /// * `rope` - Rotary embeddings
    ///
    /// # Returns
    /// (x_out, c_out) - Updated image and context tokens
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        c: Tensor<B, 3>,
        y: Tensor<B, 2>,
        rope: &RotaryEmbedding<B>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [batch, _x_len, hidden] = x.dims();
        let [_, _c_len, _] = c.dims();

        // Get modulation parameters (6 values: x_shift, x_scale, x_gate, c_shift, c_scale, c_gate)
        let mod_params = self.modulation.forward(y);
        let mod_params = mod_params.reshape([batch, 6, hidden]);

        let x_shift = mod_params
            .clone()
            .slice([0..batch, 0..1, 0..hidden])
            .reshape([batch, 1, hidden]);
        let x_scale = mod_params
            .clone()
            .slice([0..batch, 1..2, 0..hidden])
            .reshape([batch, 1, hidden]);
        let x_gate = mod_params
            .clone()
            .slice([0..batch, 2..3, 0..hidden])
            .reshape([batch, 1, hidden]);
        let c_shift = mod_params
            .clone()
            .slice([0..batch, 3..4, 0..hidden])
            .reshape([batch, 1, hidden]);
        let c_scale = mod_params
            .clone()
            .slice([0..batch, 4..5, 0..hidden])
            .reshape([batch, 1, hidden]);
        let c_gate = mod_params
            .slice([0..batch, 5..6, 0..hidden])
            .reshape([batch, 1, hidden]);

        // Apply modulated norms
        let x_norm = self.x_norm1.forward(x.clone());
        let x_norm = (Tensor::ones_like(&x_scale) + x_scale) * x_norm + x_shift;

        let c_norm = self.c_norm1.forward(c.clone());
        let c_norm = (Tensor::ones_like(&c_scale) + c_scale) * c_norm + c_shift;

        // Get Q, K, V for both streams
        let (x_q, x_k, x_v) = self.x_attn.get_qkv(x_norm);
        let (c_q, c_k, c_v) = self.c_attn.get_qkv(c_norm);

        // Apply RoPE to image tokens only (context doesn't use positional encoding)
        let (x_q, x_k) = rope.forward(x_q, x_k, 0);

        // Joint attention: concatenate K, V from both streams
        let k = Tensor::cat(vec![x_k, c_k], 2); // [batch, heads, x_len+c_len, head_dim]
        let v = Tensor::cat(vec![x_v, c_v], 2);

        // Attention for x stream (q from x, kv from both)
        let scale = (self.x_attn.head_dim as f32).sqrt().recip();
        let x_attn = x_q.matmul(k.clone().swap_dims(2, 3)) * scale;
        let x_attn = burn::tensor::activation::softmax(x_attn, 3);
        let x_out = x_attn.matmul(v.clone());
        let x_out = self.x_attn.proj_out(x_out);

        // Attention for c stream (q from c, kv from both)
        let c_attn = c_q.matmul(k.swap_dims(2, 3)) * scale;
        let c_attn = burn::tensor::activation::softmax(c_attn, 3);
        let c_out = c_attn.matmul(v);
        let c_out = self.c_attn.proj_out(c_out);

        // Residual with gating
        let x = x + x_gate.clone() * x_out;
        let c = c + c_gate.clone() * c_out;

        // FFN (separate for each stream)
        let x_ffn_out = self.x_ffn.forward(self.x_norm2.forward(x.clone()));
        let c_ffn_out = self.c_ffn.forward(self.c_norm2.forward(c.clone()));

        let x = x + x_gate * x_ffn_out;
        let c = c + c_gate * c_ffn_out;

        (x, c)
    }
}

/// SD3 Final Layer
#[derive(Module, Debug)]
pub struct Sd3FinalLayer<B: Backend> {
    pub norm: LayerNorm<B>,
    pub proj: Linear<B>,
    pub modulation: Linear<B>,
}

impl<B: Backend> Sd3FinalLayer<B> {
    pub fn forward(&self, x: Tensor<B, 3>, y: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch, _seq_len, hidden] = x.dims();

        // Get modulation (shift, scale)
        let mod_params = self.modulation.forward(y);
        let mod_params = mod_params.reshape([batch, 1, hidden * 2]);

        let shift = mod_params.clone().slice([0..batch, 0..1, 0..hidden]);
        let scale = mod_params.slice([0..batch, 0..1, hidden..(hidden * 2)]);

        // Apply modulated norm
        let x = self.norm.forward(x);
        let x = (Tensor::ones_like(&scale) + scale) * x + shift;

        self.proj.forward(x)
    }
}

/// Stable Diffusion 3 Model
#[derive(Module, Debug)]
pub struct Sd3<B: Backend> {
    /// Image patch embedding
    pub x_embed: PatchEmbed<B>,
    /// Context projection
    pub context_embed: Linear<B>,
    /// Pooled text embedding
    pub y_embed: Sd3TimestepEmbedding<B>,
    /// Timestep embedding
    pub t_embed: Sd3TimestepEmbedding<B>,
    /// MMDiT blocks
    pub blocks: Vec<MMDiTBlock<B>>,
    /// Final layer
    pub final_layer: Sd3FinalLayer<B>,
    #[module(skip)]
    pub patch_size: usize,
    #[module(skip)]
    pub in_channels: usize,
    #[module(skip)]
    pub hidden_size: usize,
}

/// Runtime state for SD3
pub struct Sd3Runtime<B: Backend> {
    pub rope: RotaryEmbedding<B>,
    pub time_embed_dim: usize,
    /// Precomputed timestep frequencies (avoids f32->tensor conversion in forward)
    pub time_freqs: Tensor<B, 1>,
}

/// Output from SD3 model
pub struct Sd3Output<B: Backend> {
    /// Velocity prediction for rectified flow [batch, channels, height, width]
    pub velocity: Tensor<B, 4>,
}

impl<B: Backend> Sd3<B> {
    /// Forward pass
    ///
    /// # Arguments
    ///
    /// * `latents` - Noisy latents [batch, channels, height, width]
    /// * `timestep` - Diffusion timestep (0 to 1 for flow matching)
    /// * `context` - Text embeddings from T5 [batch, seq_len, context_dim]
    /// * `pooled` - Pooled CLIP embeddings [batch, pooled_dim]
    /// * `runtime` - Runtime state with RoPE
    pub fn forward(
        &self,
        latents: Tensor<B, 4>,
        timestep: f32,
        context: Tensor<B, 3>,
        pooled: Tensor<B, 2>,
        runtime: &Sd3Runtime<B>,
    ) -> Sd3Output<B> {
        let [_batch, _channels, height, width] = latents.dims();

        // Patchify image
        let x = self.x_embed.forward(latents);
        let [_, _x_len, _] = x.dims();

        // Project context
        let c = self.context_embed.forward(context);

        // Get timestep embedding (using precomputed frequencies)
        let t_emb = get_timestep_embedding_with_freqs(timestep, &runtime.time_freqs);
        let t_emb = self.t_embed.forward(t_emb);

        // Get pooled text embedding
        let y_emb = self.y_embed.forward(pooled);

        // Combined conditioning
        let y = t_emb + y_emb;

        // MMDiT blocks
        let (mut x, mut c) = (x, c);
        for block in &self.blocks {
            let (new_x, new_c) = block.forward(x, c, y.clone(), &runtime.rope);
            x = new_x;
            c = new_c;
        }

        // Final layer (only on image tokens)
        let out = self.final_layer.forward(x, y);

        // Unpatchify to image
        let velocity = unpatchify(out, self.patch_size, height, width, self.in_channels);

        Sd3Output { velocity }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_sd3_config() {
        let medium = Sd3Config::sd3_medium();
        assert_eq!(medium.hidden_size, 1536);
        assert_eq!(medium.num_blocks, 24);

        let large = Sd3Config::sd3_large();
        assert_eq!(large.hidden_size, 2048);
        assert_eq!(large.num_blocks, 38);
    }

    #[test]
    fn test_timestep_embedding() {
        let device = Default::default();
        let emb = get_timestep_embedding::<TestBackend>(0.5, 64, &device);
        assert_eq!(emb.dims(), [1, 64]);
    }

    #[test]
    fn test_sd3_tiny_forward() {
        let device = Default::default();
        let config = Sd3Config::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        // 32x32 latents, patch_size=2 → 16x16 = 256 patches
        let latents = Tensor::zeros([1, 4, 32, 32], &device);
        let context = Tensor::zeros([1, 8, 128], &device); // T5 embeddings
        let pooled = Tensor::zeros([1, 128], &device); // CLIP pooled

        let output = model.forward(latents, 0.5, context, pooled, &runtime);

        assert_eq!(output.velocity.dims(), [1, 4, 32, 32]);
    }

    #[test]
    fn test_mmdit_block() {
        let device = Default::default();
        let block = MMDiTBlockConfig::new(256, 4, 512, false).init::<TestBackend>(&device);
        let rope = RotaryEmbedding::new(64, 512, &device);

        let x = Tensor::zeros([2, 64, 256], &device); // Image tokens
        let c = Tensor::zeros([2, 8, 256], &device); // Context tokens
        let y = Tensor::zeros([2, 256], &device); // Conditioning

        let (x_out, c_out) = block.forward(x, c, y, &rope);

        assert_eq!(x_out.dims(), [2, 64, 256]);
        assert_eq!(c_out.dims(), [2, 8, 256]);
    }

    #[test]
    fn test_mmdit_attention_qk_norm() {
        let device = Default::default();
        let block = MMDiTBlockConfig::new(256, 4, 512, true).init::<TestBackend>(&device);

        // QK norm should be enabled
        assert!(block.x_attn.q_norm.is_some());
        assert!(block.c_attn.k_norm.is_some());
    }

    #[test]
    fn test_sd3_final_layer() {
        let device = Default::default();
        let layer = Sd3FinalLayer {
            norm: LayerNorm::new(256, &device),
            proj: LinearConfig::new(256, 64).with_bias(true).init(&device),
            modulation: LinearConfig::new(256, 512).with_bias(true).init(&device),
        };

        let x = Tensor::<TestBackend, 3>::zeros([2, 16, 256], &device);
        let y = Tensor::zeros([2, 256], &device);

        let out = layer.forward(x, y);
        assert_eq!(out.dims(), [2, 16, 64]);
    }
}
