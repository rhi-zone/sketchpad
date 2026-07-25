//! Temporal Attention for Video Models
//!
//! Video transformers process sequences with both spatial and temporal dimensions.
//! This module provides temporal attention that operates across frames.
//!
//! # Architecture
//!
//! Video input shape: [batch, channels, time, height, width]
//!
//! For temporal attention:
//! 1. Reshape to [batch * height * width, time, channels]
//! 2. Apply self-attention over the time dimension
//! 3. Reshape back to [batch, channels, time, height, width]
//!
//! This allows each spatial position to attend across all frames.

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

/// Configuration for temporal attention
#[derive(Debug, Clone)]
pub struct TemporalAttentionConfig {
    /// Model/embedding dimension
    pub dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension (dim / num_heads if not specified)
    pub head_dim: Option<usize>,
    /// Dropout rate (not used during inference)
    pub dropout: f64,
    /// Whether to use rotary position embeddings
    pub use_rope: bool,
    /// Maximum sequence length for positional encoding
    pub max_seq_len: usize,
}

impl Default for TemporalAttentionConfig {
    fn default() -> Self {
        Self {
            dim: 1024,
            num_heads: 16,
            head_dim: None,
            dropout: 0.0,
            use_rope: true,
            max_seq_len: 256,
        }
    }
}

impl TemporalAttentionConfig {
    pub fn new(dim: usize, num_heads: usize) -> Self {
        Self {
            dim,
            num_heads,
            ..Default::default()
        }
    }

    pub fn with_rope(mut self, use_rope: bool) -> Self {
        self.use_rope = use_rope;
        self
    }

    pub fn with_max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim.unwrap_or(self.dim / self.num_heads)
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> TemporalAttention<B> {
        let head_dim = self.head_dim();
        let inner_dim = self.num_heads * head_dim;

        TemporalAttention {
            to_q: LinearConfig::new(self.dim, inner_dim)
                .with_bias(false)
                .init(device),
            to_k: LinearConfig::new(self.dim, inner_dim)
                .with_bias(false)
                .init(device),
            to_v: LinearConfig::new(self.dim, inner_dim)
                .with_bias(false)
                .init(device),
            to_out: LinearConfig::new(inner_dim, self.dim)
                .with_bias(true)
                .init(device),
            rope_freqs: if self.use_rope {
                Some(compute_rope_freqs::<B>(head_dim, self.max_seq_len, device))
            } else {
                None
            },
            num_heads: self.num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        }
    }
}

/// Temporal attention module
#[derive(Module, Debug)]
pub struct TemporalAttention<B: Backend> {
    pub to_q: Linear<B>,
    pub to_k: Linear<B>,
    pub to_v: Linear<B>,
    pub to_out: Linear<B>,
    /// Precomputed RoPE frequencies [max_seq, head_dim/2, 2]
    pub rope_freqs: Option<Tensor<B, 3>>,
    #[module(skip)]
    pub num_heads: usize,
    #[module(skip)]
    pub head_dim: usize,
    #[module(skip)]
    pub scale: f64,
}

impl<B: Backend> TemporalAttention<B> {
    /// Forward pass for temporal attention
    ///
    /// Input shape: [batch, seq_len, dim]
    /// Output shape: [batch, seq_len, dim]
    ///
    /// For video, reshape from [B, C, T, H, W] to [B*H*W, T, C] before calling.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _dim] = x.dims();

        // Project to Q, K, V
        let q = self.to_q.forward(x.clone());
        let k = self.to_k.forward(x.clone());
        let v = self.to_v.forward(x);

        // Reshape to [batch, seq, heads, head_dim]
        let q = q.reshape([batch, seq_len, self.num_heads, self.head_dim]);
        let k = k.reshape([batch, seq_len, self.num_heads, self.head_dim]);
        let v = v.reshape([batch, seq_len, self.num_heads, self.head_dim]);

        // Apply RoPE if enabled
        let (q, k) = if let Some(ref freqs) = self.rope_freqs {
            let freqs = freqs.clone().slice(0..seq_len);
            (apply_rope(q, freqs.clone()), apply_rope(k, freqs))
        } else {
            (q, k)
        };

        // Transpose to [batch, heads, seq, head_dim]
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        // Scaled dot-product attention
        let attn = q.matmul(k.transpose()) * self.scale;
        let attn = burn::tensor::activation::softmax(attn, 3);
        let out = attn.matmul(v);

        // Transpose back and reshape: [batch, seq, heads * head_dim]
        let out = out.swap_dims(1, 2);
        let out = out.reshape([batch, seq_len, self.num_heads * self.head_dim]);

        self.to_out.forward(out)
    }

    /// Forward with external KV cache for autoregressive decoding
    pub fn forward_with_cache(
        &self,
        x: Tensor<B, 3>,
        k_cache: Option<Tensor<B, 4>>,
        v_cache: Option<Tensor<B, 4>>,
    ) -> TemporalAttentionOutput<B> {
        let [batch, seq_len, _dim] = x.dims();

        let q = self.to_q.forward(x.clone());
        let k = self.to_k.forward(x.clone());
        let v = self.to_v.forward(x);

        let q = q.reshape([batch, seq_len, self.num_heads, self.head_dim]);
        let mut k = k.reshape([batch, seq_len, self.num_heads, self.head_dim]);
        let mut v = v.reshape([batch, seq_len, self.num_heads, self.head_dim]);

        // Apply RoPE to current Q, K
        let pos_offset = k_cache.as_ref().map_or(0, |c| c.dims()[1]);
        let q = if let Some(ref freqs) = self.rope_freqs {
            let freqs = freqs.clone().slice(pos_offset..pos_offset + seq_len);
            let q_rope = apply_rope(q.clone(), freqs.clone());
            k = apply_rope(k, freqs);
            q_rope
        } else {
            q
        };

        // Transpose for attention: [batch, heads, seq, head_dim]
        let q = q.swap_dims(1, 2);
        k = k.swap_dims(1, 2);
        v = v.swap_dims(1, 2);

        // Concatenate with cache
        let (k, v) = match (k_cache, v_cache) {
            (Some(kc), Some(vc)) => (
                Tensor::cat(vec![kc, k.clone()], 2),
                Tensor::cat(vec![vc, v.clone()], 2),
            ),
            _ => (k.clone(), v.clone()),
        };

        // Attention
        let attn = q.matmul(k.clone().transpose()) * self.scale;
        let attn = burn::tensor::activation::softmax(attn, 3);
        let out = attn.matmul(v.clone());

        let out = out.swap_dims(1, 2);
        let out = out.reshape([batch, seq_len, self.num_heads * self.head_dim]);

        TemporalAttentionOutput {
            output: self.to_out.forward(out),
            k_cache: k,
            v_cache: v,
        }
    }
}

/// Output from temporal attention with KV cache
pub struct TemporalAttentionOutput<B: Backend> {
    pub output: Tensor<B, 3>,
    pub k_cache: Tensor<B, 4>,
    pub v_cache: Tensor<B, 4>,
}

/// Compute RoPE frequency tensor
fn compute_rope_freqs<B: Backend>(
    head_dim: usize,
    max_seq_len: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let half_dim = head_dim / 2;
    let theta = 10000.0f32;

    // Compute frequencies: theta^(-2i/d) for i in [0, half_dim)
    let freqs_data: Vec<f32> = (0..half_dim)
        .map(|i| theta.powf(-2.0 * i as f32 / head_dim as f32))
        .collect();
    let freqs = Tensor::<B, 1>::from_floats(freqs_data.as_slice(), device);

    // Compute positions: [0, 1, 2, ..., max_seq_len-1]
    let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
    let pos = Tensor::<B, 1>::from_floats(positions.as_slice(), device);

    // Outer product: [max_seq, half_dim]
    let pos = pos.reshape([max_seq_len, 1]);
    let freqs = freqs.reshape([1, half_dim]);
    let angles = pos * freqs;

    // Compute sin and cos
    let cos = angles.clone().cos();
    let sin = angles.sin();

    // Stack to [max_seq, half_dim, 2] where [:, :, 0] = cos, [:, :, 1] = sin
    Tensor::stack(vec![cos, sin], 2)
}

/// Apply rotary position embeddings
fn apply_rope<B: Backend>(
    x: Tensor<B, 4>,     // [batch, seq, heads, head_dim]
    freqs: Tensor<B, 3>, // [seq, head_dim/2, 2]
) -> Tensor<B, 4> {
    let [batch, seq_len, heads, head_dim] = x.dims();
    let half_dim = head_dim / 2;

    // Split x into even and odd indices
    // x_even = x[:, :, :, 0::2], x_odd = x[:, :, :, 1::2]
    let x_reshaped = x.reshape([batch, seq_len, heads, half_dim, 2]);
    let x_even = x_reshaped
        .clone()
        .slice([0..batch, 0..seq_len, 0..heads, 0..half_dim, 0..1]);
    let x_odd = x_reshaped.slice([0..batch, 0..seq_len, 0..heads, 0..half_dim, 1..2]);
    let x_even = x_even.reshape([batch, seq_len, heads, half_dim]);
    let x_odd = x_odd.reshape([batch, seq_len, heads, half_dim]);

    // Extract cos and sin from freqs
    let cos = freqs.clone().slice([0..seq_len, 0..half_dim, 0..1]);
    let sin = freqs.slice([0..seq_len, 0..half_dim, 1..2]);
    let cos = cos.reshape([1, seq_len, 1, half_dim]);
    let sin = sin.reshape([1, seq_len, 1, half_dim]);

    // Apply rotation: [x_even * cos - x_odd * sin, x_even * sin + x_odd * cos]
    let rot_even = x_even.clone() * cos.clone() - x_odd.clone() * sin.clone();
    let rot_odd = x_even * sin + x_odd * cos;

    // Interleave back
    let rot_even = rot_even.reshape([batch, seq_len, heads, half_dim, 1]);
    let rot_odd = rot_odd.reshape([batch, seq_len, heads, half_dim, 1]);
    let result = Tensor::cat(vec![rot_even, rot_odd], 4);
    result.reshape([batch, seq_len, heads, head_dim])
}

/// Reshape video tensor for temporal attention
///
/// Input: [batch, channels, time, height, width]
/// Output: [batch * height * width, time, channels]
pub fn reshape_for_temporal<B: Backend>(x: Tensor<B, 5>) -> (Tensor<B, 3>, VideoShape) {
    let [batch, channels, time, height, width] = x.dims();

    // Permute to [batch, height, width, time, channels]
    let x = x.permute([0, 3, 4, 2, 1]);
    // Reshape to [batch * height * width, time, channels]
    let x = x.reshape([batch * height * width, time, channels]);

    (
        x,
        VideoShape {
            batch,
            channels,
            time,
            height,
            width,
        },
    )
}

/// Reshape back from temporal attention format
///
/// Input: [batch * height * width, time, channels]
/// Output: [batch, channels, time, height, width]
pub fn reshape_from_temporal<B: Backend>(x: Tensor<B, 3>, shape: &VideoShape) -> Tensor<B, 5> {
    let VideoShape {
        batch,
        channels,
        time,
        height,
        width,
    } = *shape;

    // Reshape to [batch, height, width, time, channels]
    let x = x.reshape([batch, height, width, time, channels]);
    // Permute back to [batch, channels, time, height, width]
    x.permute([0, 4, 3, 1, 2])
}

/// Video tensor shape for reshape operations
#[derive(Debug, Clone, Copy)]
pub struct VideoShape {
    pub batch: usize,
    pub channels: usize,
    pub time: usize,
    pub height: usize,
    pub width: usize,
}

/// Combined spatial-temporal attention block for video transformers
#[derive(Module, Debug)]
pub struct SpatioTemporalBlock<B: Backend> {
    pub temporal_attn: TemporalAttention<B>,
    pub spatial_attn: TemporalAttention<B>,
    pub norm1: burn::nn::LayerNorm<B>,
    pub norm2: burn::nn::LayerNorm<B>,
}

/// Configuration for spatio-temporal block
#[derive(Debug, Clone)]
pub struct SpatioTemporalConfig {
    pub dim: usize,
    pub num_heads: usize,
    pub head_dim: Option<usize>,
    pub use_rope: bool,
    pub max_temporal_len: usize,
    pub max_spatial_len: usize,
}

impl Default for SpatioTemporalConfig {
    fn default() -> Self {
        Self {
            dim: 1024,
            num_heads: 16,
            head_dim: None,
            use_rope: true,
            max_temporal_len: 128,
            max_spatial_len: 4096,
        }
    }
}

impl SpatioTemporalConfig {
    pub fn new(dim: usize, num_heads: usize) -> Self {
        Self {
            dim,
            num_heads,
            ..Default::default()
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> SpatioTemporalBlock<B> {
        let temporal_config = TemporalAttentionConfig {
            dim: self.dim,
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            dropout: 0.0,
            use_rope: self.use_rope,
            max_seq_len: self.max_temporal_len,
        };

        let spatial_config = TemporalAttentionConfig {
            dim: self.dim,
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            dropout: 0.0,
            use_rope: self.use_rope,
            max_seq_len: self.max_spatial_len,
        };

        SpatioTemporalBlock {
            temporal_attn: temporal_config.init(device),
            spatial_attn: spatial_config.init(device),
            norm1: burn::nn::LayerNormConfig::new(self.dim).init(device),
            norm2: burn::nn::LayerNormConfig::new(self.dim).init(device),
        }
    }
}

impl<B: Backend> SpatioTemporalBlock<B> {
    /// Forward pass for spatio-temporal attention
    ///
    /// Input: [batch, channels, time, height, width]
    /// Output: [batch, channels, time, height, width]
    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let [batch, channels, time, height, width] = x.dims();

        // Temporal attention: attend across time at each spatial position
        let (temporal_input, shape) = reshape_for_temporal(x.clone());
        let temporal_out = self.norm1.forward(temporal_input.clone());
        let temporal_out = self.temporal_attn.forward(temporal_out);
        let temporal_out = temporal_input + temporal_out;
        let x = reshape_from_temporal(temporal_out, &shape);

        // Spatial attention: attend across space at each time step
        // Reshape to [batch * time, height * width, channels]
        let x_spatial = x.permute([0, 2, 3, 4, 1]); // [B, T, H, W, C]
        let x_spatial = x_spatial.reshape([batch * time, height * width, channels]);
        let spatial_out = self.norm2.forward(x_spatial.clone());
        let spatial_out = self.spatial_attn.forward(spatial_out);
        let spatial_out = x_spatial + spatial_out;

        // Reshape back to [batch, channels, time, height, width]
        let out = spatial_out.reshape([batch, time, height, width, channels]);
        out.permute([0, 4, 1, 2, 3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_temporal_attention_config() {
        let config = TemporalAttentionConfig::default();
        assert_eq!(config.dim, 1024);
        assert_eq!(config.num_heads, 16);
        assert_eq!(config.head_dim(), 64);

        let config = TemporalAttentionConfig::new(512, 8);
        assert_eq!(config.head_dim(), 64);
    }

    #[test]
    fn test_temporal_attention_forward() {
        let device = Default::default();
        let config = TemporalAttentionConfig::new(64, 4);
        let attn = config.init::<TestBackend>(&device);

        let x = Tensor::zeros([2, 8, 64], &device);
        let out = attn.forward(x);

        assert_eq!(out.dims(), [2, 8, 64]);
    }

    #[test]
    fn test_rope_freqs() {
        let device = Default::default();
        let freqs = compute_rope_freqs::<TestBackend>(64, 100, &device);
        assert_eq!(freqs.dims(), [100, 32, 2]);
    }

    #[test]
    fn test_apply_rope() {
        let device = Default::default();
        let x = Tensor::<TestBackend, 4>::zeros([2, 10, 4, 64], &device);
        let freqs = compute_rope_freqs::<TestBackend>(64, 10, &device);

        let out = apply_rope(x.clone(), freqs);
        assert_eq!(out.dims(), [2, 10, 4, 64]);
    }

    #[test]
    fn test_reshape_for_temporal() {
        let device = Default::default();
        // [batch=2, channels=64, time=8, height=16, width=16]
        let x = Tensor::<TestBackend, 5>::zeros([2, 64, 8, 16, 16], &device);

        let (reshaped, shape) = reshape_for_temporal(x);
        // Should be [2*16*16=512, 8, 64]
        assert_eq!(reshaped.dims(), [512, 8, 64]);
        assert_eq!(shape.batch, 2);
        assert_eq!(shape.time, 8);
    }

    #[test]
    fn test_reshape_from_temporal() {
        let device = Default::default();
        let shape = VideoShape {
            batch: 2,
            channels: 64,
            time: 8,
            height: 16,
            width: 16,
        };
        let x = Tensor::<TestBackend, 3>::zeros([512, 8, 64], &device);

        let restored = reshape_from_temporal(x, &shape);
        assert_eq!(restored.dims(), [2, 64, 8, 16, 16]);
    }

    #[test]
    fn test_reshape_roundtrip() {
        let device = Default::default();
        let original = Tensor::<TestBackend, 5>::ones([2, 32, 4, 8, 8], &device);

        let (reshaped, shape) = reshape_for_temporal(original.clone());
        let restored = reshape_from_temporal(reshaped, &shape);

        assert_eq!(original.dims(), restored.dims());
    }

    #[test]
    fn test_spatio_temporal_config() {
        let config = SpatioTemporalConfig::default();
        assert_eq!(config.dim, 1024);
        assert_eq!(config.max_temporal_len, 128);
        assert_eq!(config.max_spatial_len, 4096);
    }

    #[test]
    fn test_spatio_temporal_forward() {
        let device = Default::default();
        let config = SpatioTemporalConfig {
            dim: 32,
            num_heads: 4,
            head_dim: Some(8),
            use_rope: true,
            max_temporal_len: 16,
            max_spatial_len: 64,
        };
        let block = config.init::<TestBackend>(&device);

        // [batch=1, channels=32, time=4, height=4, width=4]
        let x = Tensor::zeros([1, 32, 4, 4, 4], &device);
        let out = block.forward(x);

        assert_eq!(out.dims(), [1, 32, 4, 4, 4]);
    }

    #[test]
    fn test_attention_with_cache() {
        let device = Default::default();
        let config = TemporalAttentionConfig::new(64, 4).with_rope(false);
        let attn = config.init::<TestBackend>(&device);

        // First token
        let x1 = Tensor::zeros([2, 1, 64], &device);
        let out1 = attn.forward_with_cache(x1, None, None);
        assert_eq!(out1.output.dims(), [2, 1, 64]);
        assert_eq!(out1.k_cache.dims(), [2, 4, 1, 16]);

        // Second token with cache
        let x2 = Tensor::zeros([2, 1, 64], &device);
        let out2 = attn.forward_with_cache(x2, Some(out1.k_cache), Some(out1.v_cache));
        assert_eq!(out2.output.dims(), [2, 1, 64]);
        assert_eq!(out2.k_cache.dims(), [2, 4, 2, 16]); // Cache grew
    }
}
