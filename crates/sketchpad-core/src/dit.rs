//! Diffusion Transformer (DiT) building blocks
//!
//! Provides components for DiT-based image generation models like:
//! - Flux (Black Forest Labs)
//! - Stable Diffusion 3 (Stability AI)
//! - PixArt-α/Σ
//!
//! DiT differs from decoder-only transformers:
//! - Uses adaptive layer norm (AdaLN) conditioned on timestep
//! - Bidirectional attention (no causal mask)
//! - Often uses modulation (scale/shift) instead of simple conditioning

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

use crate::glu::SwiGluFfn;
use crate::layernorm::LayerNorm;
use crate::transformer::MultiHeadAttention;

/// Adaptive Layer Norm with modulation
///
/// Computes: y = (1 + scale) * LayerNorm(x) + shift
/// where scale and shift are predicted from conditioning (e.g., timestep)
///
/// # References
///
/// - [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
#[derive(Module, Debug)]
pub struct AdaLayerNorm<B: Backend> {
    /// Base layer norm
    pub norm: LayerNorm<B>,
    /// Linear to predict scale and shift from conditioning
    pub modulation: Linear<B>,
}

/// Configuration for AdaLayerNorm
pub struct AdaLayerNormConfig {
    /// Hidden dimension
    pub hidden_size: usize,
    /// Conditioning dimension
    pub cond_dim: usize,
}

impl AdaLayerNormConfig {
    /// Creates a new config
    pub fn new(hidden_size: usize, cond_dim: usize) -> Self {
        Self {
            hidden_size,
            cond_dim,
        }
    }

    /// Initialize the module
    pub fn init<B: Backend>(&self, device: &B::Device) -> AdaLayerNorm<B> {
        AdaLayerNorm {
            norm: LayerNorm::new(self.hidden_size, device),
            // Predict scale and shift (2 * hidden_size)
            modulation: LinearConfig::new(self.cond_dim, self.hidden_size * 2)
                .with_bias(true)
                .init(device),
        }
    }
}

impl<B: Backend> AdaLayerNorm<B> {
    /// Forward pass with conditioning
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor [batch, seq_len, hidden_size]
    /// * `cond` - Conditioning tensor [batch, cond_dim]
    ///
    /// # Returns
    ///
    /// Modulated output [batch, seq_len, hidden_size]
    pub fn forward(&self, x: Tensor<B, 3>, cond: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch, _seq_len, hidden_size] = x.dims();

        // Get scale and shift from conditioning
        let modulation = self.modulation.forward(cond);
        let modulation = modulation.reshape([batch, 1, hidden_size * 2]);

        // Split into scale and shift
        let scale = modulation.clone().slice([0..batch, 0..1, 0..hidden_size]);
        let shift = modulation.slice([0..batch, 0..1, hidden_size..(hidden_size * 2)]);

        // Apply: (1 + scale) * norm(x) + shift
        let x_norm = self.norm.forward(x);
        (Tensor::ones_like(&scale) + scale) * x_norm + shift
    }
}

/// AdaLN-Zero: Adaptive Layer Norm with zero initialization
///
/// Like AdaLN but also predicts a gate that's initialized to zero,
/// allowing the network to start as identity.
///
/// Computes: y = gate * ((1 + scale) * LayerNorm(x) + shift)
#[derive(Module, Debug)]
pub struct AdaLayerNormZero<B: Backend> {
    /// Base layer norm
    pub norm: LayerNorm<B>,
    /// Linear to predict scale, shift, and gate
    pub modulation: Linear<B>,
}

/// Configuration for AdaLayerNormZero
pub struct AdaLayerNormZeroConfig {
    /// Hidden dimension
    pub hidden_size: usize,
    /// Conditioning dimension
    pub cond_dim: usize,
}

impl AdaLayerNormZeroConfig {
    /// Creates a new config
    pub fn new(hidden_size: usize, cond_dim: usize) -> Self {
        Self {
            hidden_size,
            cond_dim,
        }
    }

    /// Initialize the module
    pub fn init<B: Backend>(&self, device: &B::Device) -> AdaLayerNormZero<B> {
        AdaLayerNormZero {
            norm: LayerNorm::new(self.hidden_size, device),
            // Predict scale, shift, and gate (3 * hidden_size)
            modulation: LinearConfig::new(self.cond_dim, self.hidden_size * 3)
                .with_bias(true)
                .init(device),
        }
    }
}

impl<B: Backend> AdaLayerNormZero<B> {
    /// Forward pass with conditioning
    ///
    /// # Returns
    ///
    /// Tuple of (modulated_output, gate) for use in residual connection
    pub fn forward(&self, x: Tensor<B, 3>, cond: Tensor<B, 2>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [batch, _seq_len, hidden_size] = x.dims();

        let modulation = self.modulation.forward(cond);
        let modulation = modulation.reshape([batch, 1, hidden_size * 3]);

        let scale = modulation.clone().slice([0..batch, 0..1, 0..hidden_size]);
        let shift = modulation
            .clone()
            .slice([0..batch, 0..1, hidden_size..(hidden_size * 2)]);
        let gate = modulation.slice([0..batch, 0..1, (hidden_size * 2)..(hidden_size * 3)]);

        let x_norm = self.norm.forward(x);
        let modulated = (Tensor::ones_like(&scale) + scale) * x_norm + shift;

        (modulated, gate)
    }
}

/// DiT Block with AdaLN-Zero
///
/// Standard DiT block as used in the original paper and many follow-ups.
///
/// Architecture:
/// ```text
/// x = x + gate1 * attn(adaLN1(x, cond))
/// x = x + gate2 * ffn(adaLN2(x, cond))
/// ```
#[derive(Module, Debug)]
pub struct DiTBlock<B: Backend> {
    /// Pre-attention adaptive norm
    pub norm1: AdaLayerNormZero<B>,
    /// Self-attention
    pub attention: MultiHeadAttention<B>,
    /// Pre-FFN adaptive norm
    pub norm2: AdaLayerNormZero<B>,
    /// Feed-forward network
    pub ffn: SwiGluFfn<B>,
}

/// Configuration for DiTBlock
pub struct DiTBlockConfig {
    /// Hidden dimension
    pub hidden_size: usize,
    /// Intermediate FFN dimension
    pub intermediate_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Conditioning dimension
    pub cond_dim: usize,
}

impl DiTBlockConfig {
    /// Creates a new config
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
        cond_dim: usize,
    ) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            num_heads,
            cond_dim,
        }
    }

    /// Initialize the block
    pub fn init<B: Backend>(&self, device: &B::Device) -> DiTBlock<B> {
        use crate::glu::SwiGluFfnConfig;
        use crate::transformer::MultiHeadAttentionConfig;

        DiTBlock {
            norm1: AdaLayerNormZeroConfig::new(self.hidden_size, self.cond_dim).init(device),
            attention: MultiHeadAttentionConfig::new(self.hidden_size, self.num_heads).init(device),
            norm2: AdaLayerNormZeroConfig::new(self.hidden_size, self.cond_dim).init(device),
            ffn: SwiGluFfnConfig::new(self.hidden_size, self.intermediate_size).init(device),
        }
    }
}

impl<B: Backend> DiTBlock<B> {
    /// Forward pass through the DiT block
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor [batch, seq_len, hidden_size]
    /// * `cond` - Conditioning tensor [batch, cond_dim] (e.g., timestep embedding)
    pub fn forward(&self, x: Tensor<B, 3>, cond: Tensor<B, 2>) -> Tensor<B, 3> {
        // Attention with AdaLN-Zero
        let (x_norm, gate1) = self.norm1.forward(x.clone(), cond.clone());
        let attn_out = self.attention.forward(x_norm, None, 0, None);
        let x = x + gate1 * attn_out;

        // FFN with AdaLN-Zero
        let (x_norm, gate2) = self.norm2.forward(x.clone(), cond);
        let ffn_out = self.ffn.forward(x_norm);
        x + gate2 * ffn_out
    }
}

/// Patchify: Convert image to sequence of patch embeddings
///
/// Takes an image tensor and converts it to a sequence of flattened patches.
#[derive(Module, Debug)]
pub struct PatchEmbed<B: Backend> {
    /// Projection from flattened patch to hidden dim
    pub proj: Linear<B>,
    /// Patch size
    pub patch_size: usize,
    /// Number of input channels
    pub in_channels: usize,
}

/// Configuration for PatchEmbed
pub struct PatchEmbedConfig {
    /// Patch size (height = width)
    pub patch_size: usize,
    /// Number of input channels
    pub in_channels: usize,
    /// Hidden dimension
    pub hidden_size: usize,
}

impl PatchEmbedConfig {
    /// Creates a new config
    pub fn new(patch_size: usize, in_channels: usize, hidden_size: usize) -> Self {
        Self {
            patch_size,
            in_channels,
            hidden_size,
        }
    }

    /// Initialize the module
    pub fn init<B: Backend>(&self, device: &B::Device) -> PatchEmbed<B> {
        let patch_dim = self.patch_size * self.patch_size * self.in_channels;
        PatchEmbed {
            proj: LinearConfig::new(patch_dim, self.hidden_size)
                .with_bias(true)
                .init(device),
            patch_size: self.patch_size,
            in_channels: self.in_channels,
        }
    }
}

impl<B: Backend> PatchEmbed<B> {
    /// Convert image to patch sequence
    ///
    /// # Arguments
    ///
    /// * `x` - Image tensor [batch, channels, height, width]
    ///
    /// # Returns
    ///
    /// Patch embeddings [batch, num_patches, hidden_size]
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch, _channels, height, width] = x.dims();
        let ph = height / self.patch_size;
        let pw = width / self.patch_size;
        let num_patches = ph * pw;

        // Reshape to patches: [B, C, H, W] -> [B, C, ph, ps, pw, ps]
        let x = x.reshape([
            batch,
            self.in_channels,
            ph,
            self.patch_size,
            pw,
            self.patch_size,
        ]);
        // Permute to [B, ph, pw, C, ps, ps]
        let x = x.swap_dims(2, 4).swap_dims(3, 5);
        // Flatten patches: [B, ph*pw, C*ps*ps]
        let patch_dim = self.in_channels * self.patch_size * self.patch_size;
        let x = x.reshape([batch, num_patches, patch_dim]);

        // Project to hidden dim
        self.proj.forward(x)
    }
}

/// Unpatchify: Convert sequence back to image
pub fn unpatchify<B: Backend>(
    x: Tensor<B, 3>,
    patch_size: usize,
    height: usize,
    width: usize,
    channels: usize,
) -> Tensor<B, 4> {
    let [batch, _num_patches, _hidden] = x.dims();
    let ph = height / patch_size;
    let pw = width / patch_size;

    // Reshape: [B, ph*pw, C*ps*ps] -> [B, ph, pw, C, ps, ps]
    let x = x.reshape([batch, ph, pw, channels, patch_size, patch_size]);
    // Permute to [B, C, ph, ps, pw, ps]
    let x = x.swap_dims(2, 4).swap_dims(3, 5);
    // Reshape to image: [B, C, H, W]
    x.reshape([batch, channels, height, width])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_ada_layer_norm() {
        let device = Default::default();
        let config = AdaLayerNormConfig::new(256, 128);
        let norm = config.init::<TestBackend>(&device);

        let x = Tensor::zeros([2, 16, 256], &device);
        let cond = Tensor::zeros([2, 128], &device);
        let y = norm.forward(x, cond);

        assert_eq!(y.dims(), [2, 16, 256]);
    }

    #[test]
    fn test_ada_layer_norm_zero() {
        let device = Default::default();
        let config = AdaLayerNormZeroConfig::new(256, 128);
        let norm = config.init::<TestBackend>(&device);

        let x = Tensor::zeros([2, 16, 256], &device);
        let cond = Tensor::zeros([2, 128], &device);
        let (y, gate) = norm.forward(x, cond);

        assert_eq!(y.dims(), [2, 16, 256]);
        assert_eq!(gate.dims(), [2, 1, 256]);
    }

    #[test]
    fn test_dit_block() {
        let device = Default::default();
        let config = DiTBlockConfig::new(256, 512, 8, 128);
        let block = config.init::<TestBackend>(&device);

        let x = Tensor::zeros([2, 16, 256], &device);
        let cond = Tensor::zeros([2, 128], &device);
        let y = block.forward(x, cond);

        assert_eq!(y.dims(), [2, 16, 256]);
    }

    #[test]
    fn test_patch_embed() {
        let device = Default::default();
        let config = PatchEmbedConfig::new(16, 3, 768);
        let embed = config.init::<TestBackend>(&device);

        // 256x256 image with 16x16 patches = 256 patches
        let x = Tensor::zeros([2, 3, 256, 256], &device);
        let patches = embed.forward(x);

        assert_eq!(patches.dims(), [2, 256, 768]);
    }

    #[test]
    fn test_unpatchify() {
        let device = Default::default();
        let x: Tensor<TestBackend, 3> = Tensor::zeros([2, 256, 768], &device);

        // 768 = 16*16*3
        let image = unpatchify(x, 16, 256, 256, 3);

        assert_eq!(image.dims(), [2, 3, 256, 256]);
    }
}
