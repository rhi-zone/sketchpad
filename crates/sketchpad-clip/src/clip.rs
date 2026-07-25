//! CLIP Text Encoder (SD 1.x)
//!
//! Implements the CLIP text encoder used in Stable Diffusion 1.x models.

use burn::module::Param;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::Int;
use burn::tensor::activation::sigmoid;

use crate::attention::{precompute_causal_mask, scaled_dot_product_attention, slice_causal_mask};
use sketchpad_core::layernorm::LayerNorm;

/// CLIP model configuration
#[derive(Debug, Clone)]
pub struct ClipConfig {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub context_length: usize,
    pub intermediate_size: usize,
}

impl ClipConfig {
    /// Configuration for SD 1.x CLIP text encoder
    pub fn sd1x() -> Self {
        Self {
            vocab_size: 49408,
            embed_dim: 768,
            num_heads: 12,
            num_layers: 12,
            context_length: 77,
            intermediate_size: 3072,
        }
    }
}

/// CLIP Text Encoder
#[derive(Module, Debug)]
pub struct ClipTextEncoder<B: Backend> {
    pub token_embedding: Embedding<B>,
    pub position_embedding: Param<Tensor<B, 2>>,
    pub layers: Vec<TransformerBlock<B>>,
    pub final_layer_norm: LayerNorm<B>,
    /// Precomputed causal mask for max context length
    pub causal_mask: Tensor<B, 2>,
    #[module(skip)]
    pub context_length: usize,
}

impl<B: Backend> ClipTextEncoder<B> {
    /// Creates a new CLIP text encoder
    ///
    /// # Arguments
    ///
    /// * `config` - CLIP model configuration
    /// * `device` - Device to create tensors on
    pub fn new(config: &ClipConfig, device: &B::Device) -> Self {
        let token_embedding =
            EmbeddingConfig::new(config.vocab_size, config.embed_dim).init(device);

        let position_embedding = Param::from_tensor(Tensor::zeros(
            [config.context_length, config.embed_dim],
            device,
        ));

        let layers = (0..config.num_layers)
            .map(|_| TransformerBlock::new(config, device))
            .collect();

        let final_layer_norm = LayerNorm::new(config.embed_dim, device);

        // Precompute causal mask for max context length
        let causal_mask = precompute_causal_mask(config.context_length, device);

        Self {
            token_embedding,
            position_embedding,
            layers,
            final_layer_norm,
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
        for layer in self.layers.iter() {
            x = layer.forward(x.clone(), Some(mask.clone()));
        }

        // Final layer norm
        self.final_layer_norm.forward(x)
    }

    /// Get the embedding at the end-of-text token position
    pub fn forward_pooled(
        &self,
        token_ids: Tensor<B, 2, Int>,
        eos_positions: &[usize],
    ) -> Tensor<B, 2> {
        let hidden = self.forward(token_ids);
        let [_batch, _seq, embed] = hidden.dims();

        // Extract embeddings at EOS positions
        let mut pooled = Vec::new();
        for (i, &pos) in eos_positions.iter().enumerate() {
            let emb = hidden
                .clone()
                .slice([i..i + 1, pos..pos + 1, 0..embed])
                .reshape([1, embed]);
            pooled.push(emb);
        }

        Tensor::cat(pooled, 0)
    }

    /// Forward pass returning penultimate layer hidden states
    ///
    /// Used for SDXL which uses penultimate layer output instead of final
    pub fn forward_penultimate(&self, token_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
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

        // Pass through all but the last layer
        let num_layers = self.layers.len();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x.clone(), Some(mask.clone()));
            if i == num_layers - 2 {
                // Return after penultimate layer (with layer norm)
                return self.final_layer_norm.forward(x);
            }
        }

        // Fallback if only one layer
        self.final_layer_norm.forward(x)
    }
}

/// Transformer block with self-attention and feed-forward
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    pub attn_norm: LayerNorm<B>,
    pub attn: MultiHeadSelfAttention<B>,
    pub ffn_norm: LayerNorm<B>,
    pub ffn: FeedForward<B>,
}

impl<B: Backend> TransformerBlock<B> {
    /// Creates a new transformer block
    pub fn new(config: &ClipConfig, device: &B::Device) -> Self {
        Self {
            attn_norm: LayerNorm::new(config.embed_dim, device),
            attn: MultiHeadSelfAttention::new(config, device),
            ffn_norm: LayerNorm::new(config.embed_dim, device),
            ffn: FeedForward::new(config, device),
        }
    }

    /// Forward pass with pre-norm residual connections
    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 2>>) -> Tensor<B, 3> {
        // Self-attention with residual
        let residual = x.clone();
        let x = self.attn_norm.forward(x);
        let x = residual + self.attn.forward(x, mask);

        // Feed-forward with residual
        let residual = x.clone();
        let x = self.ffn_norm.forward(x);
        residual + self.ffn.forward(x)
    }
}

/// Multi-head self-attention
#[derive(Module, Debug)]
pub struct MultiHeadSelfAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub out_proj: Linear<B>,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> MultiHeadSelfAttention<B> {
    /// Creates a new multi-head self-attention layer
    pub fn new(config: &ClipConfig, device: &B::Device) -> Self {
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

/// Feed-forward network with QuickGELU activation
#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    /// Creates a new feed-forward network
    pub fn new(config: &ClipConfig, device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(config.embed_dim, config.intermediate_size).init(device),
            fc2: LinearConfig::new(config.intermediate_size, config.embed_dim).init(device),
        }
    }

    /// Forward pass with QuickGELU activation
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, dim] = x.dims();
        // Burn Linear weights are [in_features, out_features]
        let out_dim = self.fc2.weight.dims()[1];

        // Flatten to 2D for Linear
        let x = x.reshape([batch * seq, dim]);

        // Standard Linear forward
        let x = self.fc1.forward(x);
        let x = quick_gelu(x);
        let x = self.fc2.forward(x);

        // Reshape back to 3D
        x.reshape([batch, seq, out_dim])
    }
}

/// QuickGELU activation: x * sigmoid(1.702 * x)
///
/// An approximation of GELU used in original CLIP that is slightly
/// faster to compute than the standard GELU.
fn quick_gelu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    x.clone() * sigmoid(x * 1.702)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_config_sd1x() {
        let config = ClipConfig::sd1x();
        assert_eq!(config.vocab_size, 49408);
        assert_eq!(config.embed_dim, 768);
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.num_layers, 12);
    }
}
