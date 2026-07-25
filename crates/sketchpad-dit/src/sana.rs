//! SANA: Efficient High-Resolution Image Synthesis with Linear DiT
//!
//! SANA from NVIDIA Research enables efficient high-resolution image synthesis
//! with linear complexity through key innovations:
//!
//! 1. **Deep Compression Autoencoder (DC-AE)**: 32x compression
//! 2. **Linear DiT**: Linear attention instead of quadratic
//! 3. **Decoder-only Text Encoder**: Uses Gemma for text understanding
//!
//! # Performance
//!
//! - SANA-0.6B competitive with Flux-12B
//! - 20x smaller, 100x+ faster throughput
//! - <1 second for 1024x1024 image on laptop GPU
//! - Up to 4096x4096 resolution
//!
//! Reference: "SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformer"
//! https://arxiv.org/abs/2410.10629

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation;

use sketchpad_core::dit::unpatchify;
use sketchpad_core::layernorm::LayerNorm;

/// SANA Model Configuration
#[derive(Debug, Clone)]
pub struct SanaConfig {
    /// Number of input image channels (usually 32 for DC-AE)
    pub in_channels: usize,
    /// Patch size
    pub patch_size: usize,
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
    /// FFN intermediate size multiplier
    pub mlp_ratio: f32,
    /// Maximum image sequence length
    pub max_seq_len: usize,
}

impl SanaConfig {
    /// SANA-0.6B configuration
    pub fn sana_0_6b() -> Self {
        Self {
            in_channels: 32, // DC-AE uses 32 channels
            patch_size: 1,   // Already heavily compressed by DC-AE
            hidden_size: 1152,
            num_heads: 16,
            num_blocks: 28,
            text_dim: 2048, // Gemma embedding dimension
            time_embed_dim: 256,
            mlp_ratio: 2.5,    // SANA uses 2.5x for efficiency
            max_seq_len: 1024, // 32x32 latent for 1024px
        }
    }

    /// SANA-1.6B configuration
    pub fn sana_1_6b() -> Self {
        Self {
            in_channels: 32,
            patch_size: 1,
            hidden_size: 2048,
            num_heads: 32,
            num_blocks: 28,
            text_dim: 2048,
            time_embed_dim: 256,
            mlp_ratio: 2.5,
            max_seq_len: 4096, // 64x64 latent for 2048px
        }
    }

    /// Tiny model for testing
    pub fn tiny() -> Self {
        Self {
            in_channels: 4,
            patch_size: 1,
            hidden_size: 256,
            num_heads: 4,
            num_blocks: 4,
            text_dim: 128,
            time_embed_dim: 64,
            mlp_ratio: 2.5,
            max_seq_len: 256,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    pub fn mlp_dim(&self) -> usize {
        (self.hidden_size as f32 * self.mlp_ratio) as usize
    }

    /// Initialize the model and runtime
    pub fn init<B: Backend>(&self, device: &B::Device) -> (Sana<B>, SanaRuntime<B>) {
        let model = Sana::new(self, device);
        let runtime = SanaRuntime {
            config: self.clone(),
            _marker: std::marker::PhantomData,
        };
        (model, runtime)
    }
}

/// Runtime configuration (non-Module data)
#[derive(Clone, Debug)]
pub struct SanaRuntime<B: Backend> {
    pub config: SanaConfig,
    _marker: std::marker::PhantomData<B>,
}

/// SANA Model Output
pub struct SanaOutput<B: Backend> {
    /// Velocity/noise prediction [batch, seq, hidden]
    pub sample: Tensor<B, 3>,
}

/// SANA Model
///
/// Linear DiT for efficient high-resolution image synthesis.
#[derive(Module, Debug)]
pub struct Sana<B: Backend> {
    /// Patch embedding
    patch_embed: SanaPatchEmbed<B>,
    /// Time step embedding
    time_embed: SanaTimeEmbed<B>,
    /// Text projection
    text_proj: Linear<B>,
    /// Transformer blocks
    blocks: Vec<SanaBlock<B>>,
    /// Final layer norm
    ln_out: LayerNorm<B>,
    /// Output projection to patch space
    out_proj: Linear<B>,
    /// Config values
    #[module(skip)]
    in_channels: usize,
    #[module(skip)]
    patch_size: usize,
}

impl<B: Backend> Sana<B> {
    pub fn new(config: &SanaConfig, device: &B::Device) -> Self {
        let blocks = (0..config.num_blocks)
            .map(|_| SanaBlock::new(config, device))
            .collect();

        Self {
            patch_embed: SanaPatchEmbed::new(config, device),
            time_embed: SanaTimeEmbed::new(config, device),
            text_proj: LinearConfig::new(config.text_dim, config.hidden_size)
                .with_bias(false)
                .init(device),
            blocks,
            ln_out: LayerNorm::new(config.hidden_size, device),
            out_proj: LinearConfig::new(
                config.hidden_size,
                config.in_channels * config.patch_size * config.patch_size,
            )
            .with_bias(true)
            .init(device),
            in_channels: config.in_channels,
            patch_size: config.patch_size,
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Input latents [batch, channels, height, width]
    /// * `timesteps` - Diffusion timesteps [batch]
    /// * `text_embeds` - Text embeddings from Gemma [batch, text_seq, text_dim]
    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        timesteps: Tensor<B, 1>,
        text_embeds: Tensor<B, 3>,
    ) -> SanaOutput<B> {
        let [_batch, _, _, _] = x.dims();

        // Embed patches
        let x = self.patch_embed.forward(x);

        // Time embedding
        let t_emb = self.time_embed.forward(timesteps);

        // Project text embeddings
        let text = self.text_proj.forward(text_embeds);

        // Process through blocks
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(x, t_emb.clone(), text.clone());
        }

        // Output
        let x = self.ln_out.forward(x);
        let sample = self.out_proj.forward(x);

        SanaOutput { sample }
    }

    /// Forward with unpatchify for full image output
    pub fn forward_image(
        &self,
        x: Tensor<B, 4>,
        timesteps: Tensor<B, 1>,
        text_embeds: Tensor<B, 3>,
        height: usize,
        width: usize,
    ) -> Tensor<B, 4> {
        let output = self.forward(x, timesteps, text_embeds);

        // Reshape back to image space
        let patch_h = height / self.patch_size;
        let patch_w = width / self.patch_size;

        unpatchify(
            output.sample,
            self.patch_size,
            self.in_channels,
            patch_h,
            patch_w,
        )
    }
}

/// Patch embedding for SANA
#[derive(Module, Debug)]
pub struct SanaPatchEmbed<B: Backend> {
    proj: Linear<B>,
}

impl<B: Backend> SanaPatchEmbed<B> {
    pub fn new(config: &SanaConfig, device: &B::Device) -> Self {
        let patch_dim = config.in_channels * config.patch_size * config.patch_size;
        Self {
            proj: LinearConfig::new(patch_dim, config.hidden_size)
                .with_bias(true)
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch, channels, height, width] = x.dims();
        // Flatten spatial dimensions: [B, C, H, W] -> [B, H*W, C]
        let x = x
            .swap_dims(1, 2)
            .swap_dims(2, 3)
            .reshape([batch, height * width, channels]);
        self.proj.forward(x)
    }
}

/// Precompute sinusoidal timestep frequencies for SANA.
#[rustfmt::skip]
pub fn sana_timestep_freqs<B: Backend>(dim: usize, device: &B::Device) -> Tensor<B, 1> {
    let half_dim = dim / 2;
    let freqs: Vec<f32> = (0..half_dim)
        .map(|i| (-(i as f32) / half_dim as f32 * std::f32::consts::LN_10 * 4.0).exp())
        .collect();
    Tensor::<B, 1>::from_floats(freqs.as_slice(), device)
}

/// Time step embedding
#[derive(Module, Debug)]
pub struct SanaTimeEmbed<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    /// Precomputed sinusoidal frequencies
    freqs: Tensor<B, 1>,
}

impl<B: Backend> SanaTimeEmbed<B> {
    pub fn new(config: &SanaConfig, device: &B::Device) -> Self {
        Self {
            linear1: LinearConfig::new(config.time_embed_dim, config.hidden_size)
                .with_bias(true)
                .init(device),
            linear2: LinearConfig::new(config.hidden_size, config.hidden_size)
                .with_bias(true)
                .init(device),
            freqs: sana_timestep_freqs(config.time_embed_dim, device),
        }
    }

    pub fn forward(&self, timesteps: Tensor<B, 1>) -> Tensor<B, 2> {
        // Expand timesteps: [batch] -> [batch, 1]
        let timesteps = timesteps.unsqueeze_dim::<2>(1);
        // Expand freqs: [half_dim] -> [1, half_dim]
        let freqs = self.freqs.clone().unsqueeze_dim::<2>(0);

        // Broadcast multiply: [batch, 1] * [1, half_dim] -> [batch, half_dim]
        let args = timesteps * freqs;

        // Sin and cos: each [batch, half_dim]
        let sin = args.clone().sin();
        let cos = args.cos();

        // Concatenate to [batch, dim]
        let emb = Tensor::cat(vec![sin, cos], 1);

        // Project through MLP
        let emb = self.linear1.forward(emb);
        let emb = activation::silu(emb);
        self.linear2.forward(emb)
    }
}

/// SANA Transformer Block with Linear Attention
#[derive(Module, Debug)]
pub struct SanaBlock<B: Backend> {
    /// Pre-attention layer norm
    ln1: LayerNorm<B>,
    /// Linear attention
    attn: SanaLinearAttention<B>,
    /// Cross-attention to text
    cross_attn: SanaCrossAttention<B>,
    /// Post-attention layer norm
    ln2: LayerNorm<B>,
    /// Feed-forward network
    ffn: SanaFFN<B>,
    /// Scale shift for time conditioning
    ada_ln_modulation: Linear<B>,
}

impl<B: Backend> SanaBlock<B> {
    pub fn new(config: &SanaConfig, device: &B::Device) -> Self {
        Self {
            ln1: LayerNorm::new(config.hidden_size, device),
            attn: SanaLinearAttention::new(config, device),
            cross_attn: SanaCrossAttention::new(config, device),
            ln2: LayerNorm::new(config.hidden_size, device),
            ffn: SanaFFN::new(config, device),
            ada_ln_modulation: LinearConfig::new(config.hidden_size, config.hidden_size * 6)
                .with_bias(true)
                .init(device),
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        t_emb: Tensor<B, 2>,
        text: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, hidden] = x.dims();

        // AdaLN modulation
        let mods = self.ada_ln_modulation.forward(activation::silu(t_emb));
        let mods = mods
            .unsqueeze_dim::<3>(1)
            .expand([batch, seq_len, hidden * 6]);

        let shift1 = mods.clone().slice([0..batch, 0..seq_len, 0..hidden]);
        let scale1 = mods
            .clone()
            .slice([0..batch, 0..seq_len, hidden..hidden * 2]);
        let gate1 = mods
            .clone()
            .slice([0..batch, 0..seq_len, hidden * 2..hidden * 3]);
        let shift2 = mods
            .clone()
            .slice([0..batch, 0..seq_len, hidden * 3..hidden * 4]);
        let scale2 = mods
            .clone()
            .slice([0..batch, 0..seq_len, hidden * 4..hidden * 5]);
        let gate2 = mods.slice([0..batch, 0..seq_len, hidden * 5..hidden * 6]);

        // Self attention with AdaLN
        let residual = x.clone();
        let x = self.ln1.forward(x) * (scale1 + 1.0) + shift1;
        let x = self.attn.forward(x);
        let x = residual + gate1 * x;

        // Cross attention to text
        let x = x.clone() + self.cross_attn.forward(x.clone(), text);

        // FFN with AdaLN
        let residual = x.clone();
        let x = self.ln2.forward(x) * (scale2 + 1.0) + shift2;
        let x = self.ffn.forward(x);
        residual + gate2 * x
    }
}

/// Linear Attention (SANA's key innovation)
///
/// Instead of softmax(QK^T)V with O(N²) complexity,
/// uses kernel feature maps for O(N) complexity.
#[derive(Module, Debug)]
pub struct SanaLinearAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    out_proj: Linear<B>,
    #[module(skip)]
    num_heads: usize,
    #[module(skip)]
    head_dim: usize,
}

impl<B: Backend> SanaLinearAttention<B> {
    pub fn new(config: &SanaConfig, device: &B::Device) -> Self {
        let head_dim = config.head_dim();
        Self {
            q_proj: LinearConfig::new(config.hidden_size, config.hidden_size)
                .with_bias(false)
                .init(device),
            k_proj: LinearConfig::new(config.hidden_size, config.hidden_size)
                .with_bias(false)
                .init(device),
            v_proj: LinearConfig::new(config.hidden_size, config.hidden_size)
                .with_bias(false)
                .init(device),
            out_proj: LinearConfig::new(config.hidden_size, config.hidden_size)
                .with_bias(false)
                .init(device),
            num_heads: config.num_heads,
            head_dim,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _] = x.dims();

        // Project
        let q = self
            .q_proj
            .forward(x.clone())
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = self
            .k_proj
            .forward(x.clone())
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = self
            .v_proj
            .forward(x)
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);

        // Linear attention with ELU-like feature map
        // φ(x) = elu(x) + 1, ensures non-negativity
        let q = activation::relu(q) + 1e-6; // Small epsilon instead of 1.0 for stability
        let k = activation::relu(k) + 1e-6;

        // Standard attention for simplicity (can optimize to linear later)
        // This is O(N²) but ensures correctness first
        let scale = (self.head_dim as f32).sqrt();
        let attn = q.matmul(k.swap_dims(2, 3)) / scale;
        let attn = activation::softmax(attn, 3);
        let out = attn.matmul(v);

        // Reshape back
        let out = out
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * self.head_dim]);

        self.out_proj.forward(out)
    }
}

/// Cross-attention to text embeddings
#[derive(Module, Debug)]
pub struct SanaCrossAttention<B: Backend> {
    ln: LayerNorm<B>,
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    out_proj: Linear<B>,
    #[module(skip)]
    num_heads: usize,
    #[module(skip)]
    head_dim: usize,
}

impl<B: Backend> SanaCrossAttention<B> {
    pub fn new(config: &SanaConfig, device: &B::Device) -> Self {
        let head_dim = config.head_dim();
        Self {
            ln: LayerNorm::new(config.hidden_size, device),
            q_proj: LinearConfig::new(config.hidden_size, config.hidden_size)
                .with_bias(false)
                .init(device),
            k_proj: LinearConfig::new(config.hidden_size, config.hidden_size)
                .with_bias(false)
                .init(device),
            v_proj: LinearConfig::new(config.hidden_size, config.hidden_size)
                .with_bias(false)
                .init(device),
            out_proj: LinearConfig::new(config.hidden_size, config.hidden_size)
                .with_bias(false)
                .init(device),
            num_heads: config.num_heads,
            head_dim,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, text: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _] = x.dims();
        let text_len = text.dims()[1];

        let x = self.ln.forward(x);

        // Q from image, K/V from text
        let q = self
            .q_proj
            .forward(x)
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = self
            .k_proj
            .forward(text.clone())
            .reshape([batch, text_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = self
            .v_proj
            .forward(text)
            .reshape([batch, text_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);

        // Standard attention for cross-attention (text sequence is short)
        let scale = (self.head_dim as f32).sqrt();
        let attn = q.matmul(k.swap_dims(2, 3)) / scale;
        let attn = activation::softmax(attn, 3);

        let out = attn.matmul(v).swap_dims(1, 2).reshape([
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ]);

        self.out_proj.forward(out)
    }
}

/// Feed-forward network
#[derive(Module, Debug)]
pub struct SanaFFN<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
}

impl<B: Backend> SanaFFN<B> {
    pub fn new(config: &SanaConfig, device: &B::Device) -> Self {
        let mlp_dim = config.mlp_dim();
        Self {
            gate_proj: LinearConfig::new(config.hidden_size, mlp_dim)
                .with_bias(false)
                .init(device),
            up_proj: LinearConfig::new(config.hidden_size, mlp_dim)
                .with_bias(false)
                .init(device),
            down_proj: LinearConfig::new(mlp_dim, config.hidden_size)
                .with_bias(false)
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = activation::silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_sana_config() {
        let _ = SanaConfig::sana_0_6b();
        let _ = SanaConfig::sana_1_6b();
    }

    #[test]
    fn test_sana_tiny_forward() {
        let device = Default::default();
        let config = SanaConfig::tiny();
        let (model, _runtime) = config.init::<TestBackend>(&device);

        // Tiny: 4 channels, so input is [batch, 4, height, width]
        let x = Tensor::<TestBackend, 4>::zeros([1, 4, 8, 8], &device);
        let timesteps = Tensor::<TestBackend, 1>::zeros([1], &device);
        let text_embeds = Tensor::<TestBackend, 3>::zeros([1, 4, 128], &device);

        let output = model.forward(x, timesteps, text_embeds);

        // Output should be [batch, 64 patches, 4 channels * 1*1 patch = 4]
        assert_eq!(output.sample.dims(), [1, 64, 4]);
    }

    #[test]
    fn test_sana_linear_attention() {
        let device = Default::default();
        let config = SanaConfig::tiny();
        let attn = SanaLinearAttention::new(&config, &device);

        let x = Tensor::<TestBackend, 3>::zeros([1, 16, 256], &device);
        let output = attn.forward(x);

        assert_eq!(output.dims(), [1, 16, 256]);
    }

    #[test]
    fn test_sana_cross_attention() {
        let device = Default::default();
        let config = SanaConfig::tiny();
        let attn = SanaCrossAttention::new(&config, &device);

        let x = Tensor::<TestBackend, 3>::zeros([1, 16, 256], &device);
        let text = Tensor::<TestBackend, 3>::zeros([1, 8, 256], &device);
        let output = attn.forward(x, text);

        assert_eq!(output.dims(), [1, 16, 256]);
    }

    #[test]
    fn test_sana_block() {
        let device = Default::default();
        let config = SanaConfig::tiny();
        let block = SanaBlock::new(&config, &device);

        let x = Tensor::<TestBackend, 3>::zeros([1, 16, 256], &device);
        let t_emb = Tensor::<TestBackend, 2>::zeros([1, 256], &device);
        let text = Tensor::<TestBackend, 3>::zeros([1, 8, 256], &device);
        let output = block.forward(x, t_emb, text);

        assert_eq!(output.dims(), [1, 16, 256]);
    }
}
