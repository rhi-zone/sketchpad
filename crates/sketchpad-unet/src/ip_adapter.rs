//! IP-Adapter implementation for image-conditioned generation
//!
//! IP-Adapter allows using images as prompts by projecting CLIP image
//! embeddings into the cross-attention space of the UNet.

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

/// IP-Adapter configuration
#[derive(Debug, Clone)]
pub struct IpAdapterConfig {
    /// CLIP image embedding dimension (typically 1024 or 1280)
    pub image_embed_dim: usize,
    /// Cross-attention dimension in UNet
    pub cross_attention_dim: usize,
    /// Number of image tokens to generate
    pub num_tokens: usize,
    /// Whether this is for SDXL (uses different projection)
    pub is_sdxl: bool,
}

impl Default for IpAdapterConfig {
    fn default() -> Self {
        Self {
            image_embed_dim: 1024,
            cross_attention_dim: 768,
            num_tokens: 4,
            is_sdxl: false,
        }
    }
}

impl IpAdapterConfig {
    /// Configuration for SD 1.x
    pub fn sd1x() -> Self {
        Self {
            image_embed_dim: 1024,
            cross_attention_dim: 768,
            num_tokens: 4,
            is_sdxl: false,
        }
    }

    /// Configuration for SDXL
    pub fn sdxl() -> Self {
        Self {
            image_embed_dim: 1280,
            cross_attention_dim: 2048,
            num_tokens: 4,
            is_sdxl: true,
        }
    }

    /// Configuration for IP-Adapter Plus (more tokens)
    pub fn plus() -> Self {
        Self {
            image_embed_dim: 1024,
            cross_attention_dim: 768,
            num_tokens: 16,
            is_sdxl: false,
        }
    }
}

/// Image projection module for IP-Adapter
#[derive(Module, Debug)]
pub struct ImageProjection<B: Backend> {
    /// Linear projection layers
    proj: Linear<B>,
    /// Layer normalization
    norm: burn::nn::LayerNorm<B>,
    /// Number of output tokens
    num_tokens: usize,
    /// Output dimension per token
    output_dim: usize,
}

impl<B: Backend> ImageProjection<B> {
    /// Create a new image projection
    pub fn new(config: &IpAdapterConfig, device: &B::Device) -> Self {
        let output_dim = config.cross_attention_dim;
        let proj_dim = config.num_tokens * output_dim;

        let proj = LinearConfig::new(config.image_embed_dim, proj_dim).init(device);
        let norm = burn::nn::LayerNormConfig::new(output_dim).init(device);

        Self {
            proj,
            norm,
            num_tokens: config.num_tokens,
            output_dim,
        }
    }

    /// Project image embeddings to cross-attention space
    ///
    /// Input: [batch, image_embed_dim]
    /// Output: [batch, num_tokens, cross_attention_dim]
    pub fn forward(&self, image_embeds: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch, _] = image_embeds.dims();

        // Project to flattened token space
        let projected = self.proj.forward(image_embeds);

        // Reshape to [batch, num_tokens, dim]
        let tokens = projected.reshape([batch, self.num_tokens, self.output_dim]);

        // Apply layer norm
        self.norm.forward(tokens)
    }
}

/// Resampler module for IP-Adapter Plus
///
/// Uses attention to resample image features into a fixed number of tokens.
#[derive(Module, Debug)]
pub struct Resampler<B: Backend> {
    /// Learnable queries
    queries: Tensor<B, 2>,
    /// Key/Value projection
    kv_proj: Linear<B>,
    /// Query projection
    q_proj: Linear<B>,
    /// Output projection
    out_proj: Linear<B>,
    /// Number of attention heads
    num_heads: usize,
    /// Dimension per head
    head_dim: usize,
}

impl<B: Backend> Resampler<B> {
    /// Create a new resampler
    pub fn new(
        image_embed_dim: usize,
        output_dim: usize,
        num_tokens: usize,
        num_heads: usize,
        device: &B::Device,
    ) -> Self {
        let head_dim = output_dim / num_heads;
        let inner_dim = num_heads * head_dim;

        // Initialize learnable queries
        let queries = Tensor::zeros([num_tokens, output_dim], device);

        let kv_proj = LinearConfig::new(image_embed_dim, inner_dim * 2).init(device);
        let q_proj = LinearConfig::new(output_dim, inner_dim).init(device);
        let out_proj = LinearConfig::new(inner_dim, output_dim).init(device);

        Self {
            queries,
            kv_proj,
            q_proj,
            out_proj,
            num_heads,
            head_dim,
        }
    }

    /// Resample image features
    ///
    /// Input: [batch, seq_len, image_embed_dim]
    /// Output: [batch, num_tokens, output_dim]
    pub fn forward(&self, image_features: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _] = image_features.dims();
        let num_tokens = self.queries.dims()[0];

        // Expand queries for batch
        let queries = self.queries.clone().unsqueeze::<3>().repeat_dim(0, batch);

        // Project queries
        let q = self.q_proj.forward(queries);

        // Project keys and values
        let kv = self.kv_proj.forward(image_features);
        let [_, _, kv_dim] = kv.dims();
        let half_dim = kv_dim / 2;
        let k = kv.clone().slice([0..batch, 0..seq_len, 0..half_dim]);
        let v = kv.slice([0..batch, 0..seq_len, half_dim..kv_dim]);

        // Reshape for multi-head attention
        let q = q
            .reshape([batch, num_tokens, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);

        // Attention
        let scale = (self.head_dim as f64).powf(-0.5);
        let attn = q.matmul(k.transpose()) * scale;
        let attn = burn::tensor::activation::softmax(attn, 3);
        let out = attn.matmul(v);

        // Reshape back
        let out = out
            .swap_dims(1, 2)
            .reshape([batch, num_tokens, self.num_heads * self.head_dim]);

        self.out_proj.forward(out)
    }
}

/// IP-Adapter model
#[derive(Module, Debug)]
pub struct IpAdapter<B: Backend> {
    /// Image projection (simple linear or resampler)
    image_proj: ImageProjection<B>,
    /// Scale factor for IP-Adapter contribution
    scale: f32,
}

impl<B: Backend> IpAdapter<B> {
    /// Create a new IP-Adapter
    pub fn new(config: IpAdapterConfig, device: &B::Device) -> Self {
        let image_proj = ImageProjection::new(&config, device);

        Self {
            image_proj,
            scale: 1.0,
        }
    }

    /// Set the scale factor
    pub fn set_scale(&mut self, scale: f32) {
        self.scale = scale;
    }

    /// Get image prompt embeddings to add to text embeddings
    ///
    /// Input: CLIP image embeddings [batch, image_embed_dim]
    /// Output: Image tokens [batch, num_tokens, cross_attention_dim]
    pub fn get_image_embeds(&self, image_embeds: Tensor<B, 2>) -> Tensor<B, 3> {
        let embeds = self.image_proj.forward(image_embeds);
        embeds * self.scale
    }
}

/// Combine text and image embeddings for cross-attention
///
/// Concatenates text tokens with image tokens along sequence dimension.
pub fn combine_embeddings<B: Backend>(
    text_embeds: Tensor<B, 3>,
    image_embeds: Tensor<B, 3>,
) -> Tensor<B, 3> {
    // text_embeds: [batch, text_seq, dim]
    // image_embeds: [batch, image_tokens, dim]
    // output: [batch, text_seq + image_tokens, dim]
    Tensor::cat(vec![text_embeds, image_embeds], 1)
}

/// IP-Adapter attention processor
///
/// Modifies cross-attention to incorporate image embeddings.
#[derive(Debug, Clone)]
pub struct IpAdapterAttention {
    /// Scale for text attention
    pub text_scale: f32,
    /// Scale for image attention
    pub image_scale: f32,
}

impl Default for IpAdapterAttention {
    fn default() -> Self {
        Self {
            text_scale: 1.0,
            image_scale: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ip_adapter_config_default() {
        let config = IpAdapterConfig::default();
        assert_eq!(config.image_embed_dim, 1024);
        assert_eq!(config.num_tokens, 4);
    }

    #[test]
    fn test_ip_adapter_config_sdxl() {
        let config = IpAdapterConfig::sdxl();
        assert_eq!(config.image_embed_dim, 1280);
        assert_eq!(config.cross_attention_dim, 2048);
        assert!(config.is_sdxl);
    }
}
