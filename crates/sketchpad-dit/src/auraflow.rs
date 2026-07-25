//! AuraFlow Model Implementation
//!
//! AuraFlow is an open-source flow-based DiT model that uses joint attention
//! between text and image tokens.
//!
//! # Architecture Overview
//!
//! ```text
//! Input: image patches + text embeddings + timestep
//!        ↓
//! [MMDiT Blocks] - joint attention between text and image
//!        ↓
//! Output: velocity prediction for flow matching
//! ```
//!
//! # Key Features
//!
//! - **Flow matching**: Rectified flow objective
//! - **Joint attention**: Text and image attend to each other (like SD3)
//! - **Open source**: Fully open weights and training

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

use sketchpad_core::dit::{PatchEmbed, PatchEmbedConfig, unpatchify};
use sketchpad_core::glu::SwiGluFfn;
use sketchpad_core::layernorm::LayerNorm;
use sketchpad_core::rope::RotaryEmbedding;

/// AuraFlow Model Configuration
#[derive(Debug, Clone)]
pub struct AuraFlowConfig {
    /// Number of input image channels (usually 4 for VAE latents)
    pub in_channels: usize,
    /// Patch size
    pub patch_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of MMDiT blocks
    pub num_blocks: usize,
    /// Text embedding dimension (T5-XXL = 4096)
    pub text_dim: usize,
    /// Timestep embedding dimension
    pub time_embed_dim: usize,
    /// FFN intermediate size multiplier
    pub mlp_ratio: f32,
    /// Maximum sequence length for RoPE
    pub max_seq_len: usize,
}

impl AuraFlowConfig {
    /// AuraFlow base configuration
    pub fn base() -> Self {
        Self {
            in_channels: 4,
            patch_size: 2,
            hidden_size: 3072,
            num_heads: 24,
            num_blocks: 24,
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
            max_seq_len: 512,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    pub fn intermediate_size(&self) -> usize {
        (self.hidden_size as f32 * self.mlp_ratio) as usize
    }

    /// Initialize the model
    pub fn init<B: Backend>(&self, device: &B::Device) -> (AuraFlow<B>, AuraFlowRuntime<B>) {
        // Patch embedding
        let patch_embed =
            PatchEmbedConfig::new(self.patch_size, self.in_channels, self.hidden_size).init(device);

        // Text projection
        let text_embed = LinearConfig::new(self.text_dim, self.hidden_size)
            .with_bias(true)
            .init(device);

        // Timestep embedding
        let time_embed = AuraFlowTimestepEmbed {
            linear1: LinearConfig::new(self.time_embed_dim, self.hidden_size)
                .with_bias(true)
                .init(device),
            linear2: LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(true)
                .init(device),
            freqs: auraflow_timestep_freqs(self.time_embed_dim, device),
        };

        // MMDiT blocks with joint attention
        let blocks: Vec<AuraFlowBlock<B>> = (0..self.num_blocks)
            .map(|_| {
                AuraFlowBlockConfig::new(self.hidden_size, self.num_heads, self.intermediate_size())
                    .init(device)
            })
            .collect();

        // Final layer
        let final_layer = AuraFlowFinalLayer {
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

        let model = AuraFlow {
            patch_embed,
            text_embed,
            time_embed,
            blocks,
            final_layer,
            hidden_size: self.hidden_size,
            patch_size: self.patch_size,
            in_channels: self.in_channels,
        };

        let runtime = AuraFlowRuntime {
            rope: RotaryEmbedding::new(self.head_dim(), self.max_seq_len, device),
        };

        (model, runtime)
    }
}

/// Timestep embedding for AuraFlow
#[derive(Module, Debug)]
pub struct AuraFlowTimestepEmbed<B: Backend> {
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
    /// Precomputed sinusoidal frequencies
    pub freqs: Tensor<B, 1>,
}

/// Precompute sinusoidal frequencies for AuraFlow timestep embedding
pub fn auraflow_timestep_freqs<B: Backend>(embed_dim: usize, device: &B::Device) -> Tensor<B, 1> {
    let half_dim = embed_dim / 2;
    let emb_scale = -(10000.0_f32.ln()) / (half_dim as f32 - 1.0);
    let freqs: Vec<f32> = (0..half_dim)
        .map(|i| (emb_scale * i as f32).exp())
        .collect();
    Tensor::<B, 1>::from_floats(freqs.as_slice(), device)
}

impl<B: Backend> AuraFlowTimestepEmbed<B> {
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

/// AuraFlow Block Configuration
struct AuraFlowBlockConfig {
    hidden_size: usize,
    num_heads: usize,
    intermediate_size: usize,
}

impl AuraFlowBlockConfig {
    fn new(hidden_size: usize, num_heads: usize, intermediate_size: usize) -> Self {
        Self {
            hidden_size,
            num_heads,
            intermediate_size,
        }
    }

    fn init<B: Backend>(&self, device: &B::Device) -> AuraFlowBlock<B> {
        use sketchpad_core::glu::SwiGluFfnConfig;

        let head_dim = self.hidden_size / self.num_heads;

        AuraFlowBlock {
            // Image stream
            x_norm1: LayerNorm::new(self.hidden_size, device),
            x_attn: AuraFlowAttention {
                to_qkv: LinearConfig::new(self.hidden_size, 3 * self.hidden_size)
                    .with_bias(true)
                    .init(device),
                to_out: LinearConfig::new(self.hidden_size, self.hidden_size)
                    .with_bias(true)
                    .init(device),
                num_heads: self.num_heads,
                head_dim,
            },
            x_norm2: LayerNorm::new(self.hidden_size, device),
            x_ffn: SwiGluFfnConfig::new(self.hidden_size, self.intermediate_size).init(device),
            // Text stream
            c_norm1: LayerNorm::new(self.hidden_size, device),
            c_attn: AuraFlowAttention {
                to_qkv: LinearConfig::new(self.hidden_size, 3 * self.hidden_size)
                    .with_bias(true)
                    .init(device),
                to_out: LinearConfig::new(self.hidden_size, self.hidden_size)
                    .with_bias(true)
                    .init(device),
                num_heads: self.num_heads,
                head_dim,
            },
            c_norm2: LayerNorm::new(self.hidden_size, device),
            c_ffn: SwiGluFfnConfig::new(self.hidden_size, self.intermediate_size).init(device),
            // Modulation
            modulation: LinearConfig::new(self.hidden_size, 12 * self.hidden_size)
                .with_bias(true)
                .init(device),
            hidden_size: self.hidden_size,
            num_heads: self.num_heads,
            head_dim,
        }
    }
}

/// AuraFlow Attention (shared implementation)
#[derive(Module, Debug)]
pub struct AuraFlowAttention<B: Backend> {
    pub to_qkv: Linear<B>,
    pub to_out: Linear<B>,
    #[module(skip)]
    pub num_heads: usize,
    #[module(skip)]
    pub head_dim: usize,
}

impl<B: Backend> AuraFlowAttention<B> {
    /// Project to QKV
    pub fn qkv(&self, x: Tensor<B, 3>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
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
            .swap_dims(1, 2); // [B, H, S, D]
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

        (q, k, v)
    }

    /// Output projection
    pub fn out(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch, _heads, seq_len, _dim] = x.dims();
        let x = x
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * self.head_dim]);
        self.to_out.forward(x)
    }
}

/// AuraFlow MMDiT Block with joint attention
#[derive(Module, Debug)]
pub struct AuraFlowBlock<B: Backend> {
    // Image stream
    pub x_norm1: LayerNorm<B>,
    pub x_attn: AuraFlowAttention<B>,
    pub x_norm2: LayerNorm<B>,
    pub x_ffn: SwiGluFfn<B>,
    // Text stream
    pub c_norm1: LayerNorm<B>,
    pub c_attn: AuraFlowAttention<B>,
    pub c_norm2: LayerNorm<B>,
    pub c_ffn: SwiGluFfn<B>,
    // Modulation
    pub modulation: Linear<B>,
    #[module(skip)]
    pub hidden_size: usize,
    #[module(skip)]
    pub num_heads: usize,
    #[module(skip)]
    pub head_dim: usize,
}

impl<B: Backend> AuraFlowBlock<B> {
    /// Forward pass with joint attention between image and text
    pub fn forward(
        &self,
        x: Tensor<B, 3>,    // Image tokens
        c: Tensor<B, 3>,    // Text tokens
        cond: Tensor<B, 2>, // Timestep conditioning
        rope: &RotaryEmbedding<B>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [batch, _x_len, hidden] = x.dims();
        let [_, _c_len, _] = c.dims();

        // Get all modulation parameters (12 total: 6 for x, 6 for c)
        let mod_params = self.modulation.forward(cond);
        let mod_params = mod_params.reshape([batch, 12, hidden]);

        // X modulation
        let x_shift1 = mod_params
            .clone()
            .slice([0..batch, 0..1, 0..hidden])
            .reshape([batch, 1, hidden]);
        let x_scale1 = mod_params
            .clone()
            .slice([0..batch, 1..2, 0..hidden])
            .reshape([batch, 1, hidden]);
        let x_gate1 = mod_params
            .clone()
            .slice([0..batch, 2..3, 0..hidden])
            .reshape([batch, 1, hidden]);
        let x_shift2 = mod_params
            .clone()
            .slice([0..batch, 3..4, 0..hidden])
            .reshape([batch, 1, hidden]);
        let x_scale2 = mod_params
            .clone()
            .slice([0..batch, 4..5, 0..hidden])
            .reshape([batch, 1, hidden]);
        let x_gate2 = mod_params
            .clone()
            .slice([0..batch, 5..6, 0..hidden])
            .reshape([batch, 1, hidden]);
        // C modulation
        let c_shift1 = mod_params
            .clone()
            .slice([0..batch, 6..7, 0..hidden])
            .reshape([batch, 1, hidden]);
        let c_scale1 = mod_params
            .clone()
            .slice([0..batch, 7..8, 0..hidden])
            .reshape([batch, 1, hidden]);
        let c_gate1 = mod_params
            .clone()
            .slice([0..batch, 8..9, 0..hidden])
            .reshape([batch, 1, hidden]);
        let c_shift2 = mod_params
            .clone()
            .slice([0..batch, 9..10, 0..hidden])
            .reshape([batch, 1, hidden]);
        let c_scale2 = mod_params
            .clone()
            .slice([0..batch, 10..11, 0..hidden])
            .reshape([batch, 1, hidden]);
        let c_gate2 = mod_params
            .slice([0..batch, 11..12, 0..hidden])
            .reshape([batch, 1, hidden]);

        // Modulated norms
        let x_norm = self.x_norm1.forward(x.clone());
        let x_norm = (Tensor::ones_like(&x_scale1) + x_scale1) * x_norm + x_shift1;

        let c_norm = self.c_norm1.forward(c.clone());
        let c_norm = (Tensor::ones_like(&c_scale1) + c_scale1) * c_norm + c_shift1;

        // Get QKV for both streams
        let (x_q, x_k, x_v) = self.x_attn.qkv(x_norm);
        let (c_q, c_k, c_v) = self.c_attn.qkv(c_norm);

        // Apply RoPE to image queries and keys
        let (x_q, x_k) = rope.forward(x_q, x_k, 0);

        // Joint attention: concatenate K, V from both streams
        let k = Tensor::cat(vec![c_k, x_k], 2); // [B, H, c_len + x_len, D]
        let v = Tensor::cat(vec![c_v, x_v], 2);

        // Compute attention for both streams
        let scale = (self.head_dim as f32).sqrt().recip();

        // Image attention (x attends to all)
        let x_attn = x_q.matmul(k.clone().swap_dims(2, 3)) * scale;
        let x_attn = burn::tensor::activation::softmax(x_attn, 3);
        let x_out = x_attn.matmul(v.clone());
        let x_out = self.x_attn.out(x_out);

        // Text attention (c attends to all)
        let c_attn = c_q.matmul(k.swap_dims(2, 3)) * scale;
        let c_attn = burn::tensor::activation::softmax(c_attn, 3);
        let c_out = c_attn.matmul(v);
        let c_out = self.c_attn.out(c_out);

        // Residual + gate
        let x = x + x_gate1 * x_out;
        let c = c + c_gate1 * c_out;

        // FFN
        let x_norm = self.x_norm2.forward(x.clone());
        let x_norm = (Tensor::ones_like(&x_scale2) + x_scale2) * x_norm + x_shift2;
        let x = x + x_gate2 * self.x_ffn.forward(x_norm);

        let c_norm = self.c_norm2.forward(c.clone());
        let c_norm = (Tensor::ones_like(&c_scale2) + c_scale2) * c_norm + c_shift2;
        let c = c + c_gate2 * self.c_ffn.forward(c_norm);

        (x, c)
    }
}

/// AuraFlow Final Layer
#[derive(Module, Debug)]
pub struct AuraFlowFinalLayer<B: Backend> {
    pub norm: LayerNorm<B>,
    pub proj: Linear<B>,
    pub modulation: Linear<B>,
}

impl<B: Backend> AuraFlowFinalLayer<B> {
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

/// AuraFlow Model
#[derive(Module, Debug)]
pub struct AuraFlow<B: Backend> {
    pub patch_embed: PatchEmbed<B>,
    pub text_embed: Linear<B>,
    pub time_embed: AuraFlowTimestepEmbed<B>,
    pub blocks: Vec<AuraFlowBlock<B>>,
    pub final_layer: AuraFlowFinalLayer<B>,
    #[module(skip)]
    pub hidden_size: usize,
    #[module(skip)]
    pub patch_size: usize,
    #[module(skip)]
    pub in_channels: usize,
}

/// Runtime state for AuraFlow
pub struct AuraFlowRuntime<B: Backend> {
    pub rope: RotaryEmbedding<B>,
}

/// Output from AuraFlow
pub struct AuraFlowOutput<B: Backend> {
    /// Velocity prediction [batch, channels, height, width]
    pub velocity: Tensor<B, 4>,
}

impl<B: Backend> AuraFlow<B> {
    /// Forward pass
    ///
    /// # Arguments
    ///
    /// * `latents` - Noisy latent images [batch, channels, height, width]
    /// * `timestep` - Diffusion timestep
    /// * `text_embeds` - T5 text embeddings [batch, seq_len, text_dim]
    /// * `runtime` - Runtime state with RoPE
    pub fn forward(
        &self,
        latents: Tensor<B, 4>,
        timestep: f32,
        text_embeds: Tensor<B, 3>,
        runtime: &AuraFlowRuntime<B>,
    ) -> AuraFlowOutput<B> {
        let [_batch, _channels, height, width] = latents.dims();

        // Patchify
        let x = self.patch_embed.forward(latents);

        // Project text
        let c = self.text_embed.forward(text_embeds);

        // Timestep embedding (using scalar forward to avoid tensor allocation)
        let cond = self.time_embed.forward_scalar(timestep);

        // MMDiT blocks with joint attention
        let mut x = x;
        let mut c = c;
        for block in &self.blocks {
            let (x_new, c_new) = block.forward(x, c, cond.clone(), &runtime.rope);
            x = x_new;
            c = c_new;
        }

        // Final layer (image only)
        let out = self.final_layer.forward(x, cond);

        // Unpatchify
        let velocity = unpatchify(out, self.patch_size, height, width, self.in_channels);

        AuraFlowOutput { velocity }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_auraflow_config() {
        let config = AuraFlowConfig::base();
        assert_eq!(config.hidden_size, 3072);
        assert_eq!(config.num_blocks, 24);
    }

    #[test]
    fn test_auraflow_timestep_embed() {
        let device = Default::default();
        let embed = AuraFlowTimestepEmbed {
            linear1: LinearConfig::new(64, 256).with_bias(true).init(&device),
            linear2: LinearConfig::new(256, 256).with_bias(true).init(&device),
            freqs: auraflow_timestep_freqs(64, &device),
        };

        let t = Tensor::<TestBackend, 1>::from_floats([0.5], &device);
        let out = embed.forward(t);
        assert_eq!(out.dims(), [1, 256]);
    }

    #[test]
    fn test_auraflow_attention_qkv() {
        let device = Default::default();
        let attn = AuraFlowAttention {
            to_qkv: LinearConfig::new(256, 768).with_bias(true).init(&device),
            to_out: LinearConfig::new(256, 256).with_bias(true).init(&device),
            num_heads: 4,
            head_dim: 64,
        };

        let x = Tensor::<TestBackend, 3>::zeros([2, 16, 256], &device);
        let (q, k, v) = attn.qkv(x);
        assert_eq!(q.dims(), [2, 4, 16, 64]);
        assert_eq!(k.dims(), [2, 4, 16, 64]);
        assert_eq!(v.dims(), [2, 4, 16, 64]);
    }

    #[test]
    fn test_auraflow_block() {
        let device = Default::default();
        let block = AuraFlowBlockConfig::new(256, 4, 512).init::<TestBackend>(&device);
        let rope = RotaryEmbedding::new(64, 256, &device);

        let x = Tensor::zeros([2, 16, 256], &device);
        let c = Tensor::zeros([2, 8, 256], &device);
        let cond = Tensor::zeros([2, 256], &device);

        let (x_out, c_out) = block.forward(x, c, cond, &rope);
        assert_eq!(x_out.dims(), [2, 16, 256]);
        assert_eq!(c_out.dims(), [2, 8, 256]);
    }

    #[test]
    fn test_auraflow_tiny_forward() {
        let device = Default::default();
        let config = AuraFlowConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        // [batch=1, channels=4, height=8, width=8]
        let latents = Tensor::zeros([1, 4, 8, 8], &device);
        let text = Tensor::zeros([1, 4, 128], &device); // T5 embeddings

        let output = model.forward(latents, 0.5, text, &runtime);

        assert_eq!(output.velocity.dims(), [1, 4, 8, 8]);
    }
}
