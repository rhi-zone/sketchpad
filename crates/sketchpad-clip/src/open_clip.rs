//! OpenCLIP Text Encoder (SDXL second encoder)
//!
//! Implements the OpenCLIP-G/14 text encoder used as the second text encoder
//! in SDXL. Key differences from CLIP:
//! - Larger model (1280 embed_dim, 32 layers, 20 heads)
//! - Uses standard GELU instead of QuickGELU
//! - Includes text projection for pooled embeddings

use burn::module::Param;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::Int;
use burn::tensor::activation::gelu;

use crate::attention::{precompute_causal_mask, scaled_dot_product_attention, slice_causal_mask};
use sketchpad_core::layernorm::LayerNorm;

/// OpenCLIP model configuration
#[derive(Debug, Clone)]
pub struct OpenClipConfig {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub context_length: usize,
    pub intermediate_size: usize,
    /// Dimension for text projection (for pooled output)
    pub projection_dim: usize,
}

impl OpenClipConfig {
    /// Configuration for OpenCLIP-G/14 used in SDXL
    pub fn sdxl() -> Self {
        Self {
            vocab_size: 49408,
            embed_dim: 1280,
            num_heads: 20,
            num_layers: 32,
            context_length: 77,
            intermediate_size: 5120, // 4x embed_dim
            projection_dim: 1280,
        }
    }

    /// Configuration for OpenCLIP ViT-H/14 used in SD 2.x
    ///
    /// SD 2.x uses the ViT-H/14 variant with 1024 embed dim
    pub fn sd2x() -> Self {
        Self {
            vocab_size: 49408,
            embed_dim: 1024,
            num_heads: 16,
            num_layers: 24,
            context_length: 77,
            intermediate_size: 4096, // 4x embed_dim
            projection_dim: 1024,
        }
    }

    /// Configuration for OpenCLIP ViT-H/14 with 768 context (SD 2.1 768)
    pub fn sd2x_768() -> Self {
        Self {
            vocab_size: 49408,
            embed_dim: 1024,
            num_heads: 16,
            num_layers: 24,
            context_length: 77,
            intermediate_size: 4096,
            projection_dim: 1024,
        }
    }
}

/// OpenCLIP Text Encoder
#[derive(Module, Debug)]
pub struct OpenClipTextEncoder<B: Backend> {
    pub token_embedding: Embedding<B>,
    pub position_embedding: Param<Tensor<B, 2>>,
    pub layers: Vec<OpenClipTransformerBlock<B>>,
    pub final_layer_norm: LayerNorm<B>,
    pub text_projection: Linear<B>,
    /// Precomputed causal mask for max context length
    pub causal_mask: Tensor<B, 2>,
    #[module(skip)]
    pub context_length: usize,
}

impl<B: Backend> OpenClipTextEncoder<B> {
    /// Creates a new OpenCLIP text encoder
    ///
    /// # Arguments
    ///
    /// * `config` - OpenCLIP model configuration
    /// * `device` - Device to create tensors on
    pub fn new(config: &OpenClipConfig, device: &B::Device) -> Self {
        let token_embedding =
            EmbeddingConfig::new(config.vocab_size, config.embed_dim).init(device);

        let position_embedding = Param::from_tensor(Tensor::zeros(
            [config.context_length, config.embed_dim],
            device,
        ));

        let layers = (0..config.num_layers)
            .map(|_| OpenClipTransformerBlock::new(config, device))
            .collect();

        let final_layer_norm = LayerNorm::new(config.embed_dim, device);

        // Text projection for pooled output
        let text_projection = LinearConfig::new(config.embed_dim, config.projection_dim)
            .with_bias(false)
            .init(device);

        // Precompute causal mask for max context length
        let causal_mask = precompute_causal_mask(config.context_length, device);

        Self {
            token_embedding,
            position_embedding,
            layers,
            final_layer_norm,
            text_projection,
            causal_mask,
            context_length: config.context_length,
        }
    }

    /// Forward pass through the text encoder
    ///
    /// Input: token_ids [batch, seq_len]
    /// Output: hidden_states [batch, seq_len, embed_dim]
    pub fn forward(&self, token_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_batch, seq_len] = token_ids.dims();

        // Token embeddings
        let x = self.token_embedding.forward(token_ids);

        // Add position embeddings
        let pos_emb = self
            .position_embedding
            .val()
            .slice(0..seq_len)
            .unsqueeze::<3>();
        let mut x = x + pos_emb;

        // Slice precomputed causal mask to actual sequence length
        let mask = slice_causal_mask(&self.causal_mask, seq_len);

        // Pass through transformer layers
        for layer in &self.layers {
            x = layer.forward(x.clone(), Some(mask.clone()));
        }

        // Final layer norm
        self.final_layer_norm.forward(x)
    }

    /// Forward pass returning both hidden states and pooled output
    ///
    /// Returns:
    /// - hidden_states [batch, seq_len, embed_dim]
    /// - pooled [batch, projection_dim] (from EOS token position, projected)
    pub fn forward_with_pooled(
        &self,
        token_ids: Tensor<B, 2, Int>,
        eos_positions: &[usize],
    ) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let hidden = self.forward(token_ids);
        let [_batch, _seq, embed] = hidden.dims();

        // Extract embeddings at EOS positions
        let mut pooled_list = Vec::new();
        for (i, &pos) in eos_positions.iter().enumerate() {
            let emb = hidden
                .clone()
                .slice([i..i + 1, pos..pos + 1, 0..embed])
                .reshape([1, embed]);
            pooled_list.push(emb);
        }

        let pooled = Tensor::cat(pooled_list, 0);

        // Apply text projection
        let pooled = self.text_projection.forward(pooled);

        (hidden, pooled)
    }

    /// Get penultimate layer hidden states (used for SDXL conditioning)
    ///
    /// SDXL uses the penultimate layer output for text conditioning
    pub fn forward_penultimate(&self, token_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_batch, seq_len] = token_ids.dims();

        let x = self.token_embedding.forward(token_ids);
        let pos_emb = self
            .position_embedding
            .val()
            .slice(0..seq_len)
            .unsqueeze::<3>();
        let mut x = x + pos_emb;

        // Slice precomputed causal mask to actual sequence length
        let mask = slice_causal_mask(&self.causal_mask, seq_len);

        // Pass through all but last layer
        let num_layers = self.layers.len();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x.clone(), Some(mask.clone()));
            if i == num_layers - 2 {
                // Return after penultimate layer (before final layer)
                return self.final_layer_norm.forward(x);
            }
        }

        self.final_layer_norm.forward(x)
    }
}

/// OpenCLIP Transformer block with self-attention and feed-forward
#[derive(Module, Debug)]
pub struct OpenClipTransformerBlock<B: Backend> {
    pub attn_norm: LayerNorm<B>,
    pub attn: OpenClipMultiHeadSelfAttention<B>,
    pub ffn_norm: LayerNorm<B>,
    pub ffn: OpenClipFeedForward<B>,
}

impl<B: Backend> OpenClipTransformerBlock<B> {
    /// Creates a new OpenCLIP transformer block
    pub fn new(config: &OpenClipConfig, device: &B::Device) -> Self {
        Self {
            attn_norm: LayerNorm::new(config.embed_dim, device),
            attn: OpenClipMultiHeadSelfAttention::new(config, device),
            ffn_norm: LayerNorm::new(config.embed_dim, device),
            ffn: OpenClipFeedForward::new(config, device),
        }
    }

    /// Forward pass with pre-norm residual connections
    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 2>>) -> Tensor<B, 3> {
        // Self-attention with residual (pre-norm)
        let residual = x.clone();
        let x = self.attn_norm.forward(x);
        let x = residual + self.attn.forward(x, mask);

        // Feed-forward with residual (pre-norm)
        let residual = x.clone();
        let x = self.ffn_norm.forward(x);
        residual + self.ffn.forward(x)
    }
}

/// Multi-head self-attention for OpenCLIP
#[derive(Module, Debug)]
pub struct OpenClipMultiHeadSelfAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub out_proj: Linear<B>,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> OpenClipMultiHeadSelfAttention<B> {
    /// Creates a new multi-head self-attention layer
    pub fn new(config: &OpenClipConfig, device: &B::Device) -> Self {
        let embed_dim = config.embed_dim;
        let num_heads = config.num_heads;
        let head_dim = embed_dim / num_heads;

        Self {
            q_proj: LinearConfig::new(embed_dim, embed_dim).init(device),
            k_proj: LinearConfig::new(embed_dim, embed_dim).init(device),
            v_proj: LinearConfig::new(embed_dim, embed_dim).init(device),
            out_proj: LinearConfig::new(embed_dim, embed_dim).init(device),
            num_heads,
            head_dim,
        }
    }

    /// Forward pass computing scaled dot-product attention
    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 2>>) -> Tensor<B, 3> {
        let [batch, seq_len, _] = x.dims();

        // Project to Q, K, V
        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        // Reshape to [batch, num_heads, seq_len, head_dim]
        let q = q
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);

        // Scaled dot-product attention
        let out = scaled_dot_product_attention(q, k, v, mask, self.head_dim);

        // Reshape back
        let out = out
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * self.head_dim]);

        self.out_proj.forward(out)
    }
}

/// Feed-forward network with standard GELU activation (not QuickGELU)
#[derive(Module, Debug)]
pub struct OpenClipFeedForward<B: Backend> {
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
}

impl<B: Backend> OpenClipFeedForward<B> {
    /// Creates a new feed-forward network
    pub fn new(config: &OpenClipConfig, device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(config.embed_dim, config.intermediate_size).init(device),
            fc2: LinearConfig::new(config.intermediate_size, config.embed_dim).init(device),
        }
    }

    /// Forward pass with standard GELU activation
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.fc1.forward(x);
        let x = gelu(x); // Standard GELU, not QuickGELU
        self.fc2.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openclip_config_sdxl() {
        let config = OpenClipConfig::sdxl();
        assert_eq!(config.vocab_size, 49408);
        assert_eq!(config.embed_dim, 1280);
        assert_eq!(config.num_heads, 20);
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.intermediate_size, 5120);
    }
}
