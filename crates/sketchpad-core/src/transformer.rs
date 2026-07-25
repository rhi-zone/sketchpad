//! Generic Transformer building blocks
//!
//! Provides configurable transformer components that can be composed to build
//! various architectures including LLMs (LLaMA, Qwen), DiT, and encoders.

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

use crate::glu::{SwiGluFfn, SwiGluFfnConfig};
use crate::rmsnorm::RmsNorm;
use crate::rope::RotaryEmbedding;

/// Multi-head self-attention with optional RoPE
///
/// Supports grouped-query attention (GQA) where KV heads can be fewer than Q heads.
#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    /// Query projection
    pub q_proj: Linear<B>,
    /// Key projection
    pub k_proj: Linear<B>,
    /// Value projection
    pub v_proj: Linear<B>,
    /// Output projection
    pub o_proj: Linear<B>,
    /// Number of query heads
    pub num_heads: usize,
    /// Number of KV heads (for GQA)
    pub num_kv_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
}

/// Configuration for MultiHeadAttention
pub struct MultiHeadAttentionConfig {
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of query heads
    pub num_heads: usize,
    /// Number of key-value heads (for GQA, set < num_heads)
    pub num_kv_heads: usize,
    /// Whether to use bias
    pub bias: bool,
}

impl MultiHeadAttentionConfig {
    /// Standard multi-head attention (all heads equal)
    pub fn new(hidden_size: usize, num_heads: usize) -> Self {
        Self {
            hidden_size,
            num_heads,
            num_kv_heads: num_heads,
            bias: false,
        }
    }

    /// Grouped-query attention
    pub fn gqa(hidden_size: usize, num_heads: usize, num_kv_heads: usize) -> Self {
        Self {
            hidden_size,
            num_heads,
            num_kv_heads,
            bias: false,
        }
    }

    /// Initialize the attention module
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadAttention<B> {
        let head_dim = self.hidden_size / self.num_heads;
        let kv_dim = head_dim * self.num_kv_heads;

        MultiHeadAttention {
            q_proj: LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(self.bias)
                .init(device),
            k_proj: LinearConfig::new(self.hidden_size, kv_dim)
                .with_bias(self.bias)
                .init(device),
            v_proj: LinearConfig::new(self.hidden_size, kv_dim)
                .with_bias(self.bias)
                .init(device),
            o_proj: LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(self.bias)
                .init(device),
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim,
        }
    }
}

impl<B: Backend> MultiHeadAttention<B> {
    /// Forward pass with optional RoPE and causal mask
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor [batch, seq_len, hidden_size]
    /// * `rope` - Optional rotary embeddings
    /// * `start_pos` - Position offset for RoPE (for KV cache)
    /// * `mask` - Optional attention mask [seq_len, seq_len]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: Option<&RotaryEmbedding<B>>,
        start_pos: usize,
        mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden] = x.dims();

        // Project to Q, K, V
        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        // Reshape to [batch, seq_len, num_heads, head_dim]
        let q = q.reshape([batch, seq_len, self.num_heads, self.head_dim]);
        let k = k.reshape([batch, seq_len, self.num_kv_heads, self.head_dim]);
        let v = v.reshape([batch, seq_len, self.num_kv_heads, self.head_dim]);

        // Transpose to [batch, num_heads, seq_len, head_dim]
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        // Apply RoPE if provided
        let (q, k) = match rope {
            Some(r) => r.forward(q, k, start_pos),
            None => (q, k),
        };

        // Repeat KV heads for GQA
        let k = self.repeat_kv(k);
        let v = self.repeat_kv(v);

        // Attention
        let scale = (self.head_dim as f64).powf(-0.5);
        let attn = q.matmul(k.transpose()) * scale;

        let attn = match mask {
            Some(m) => attn + m.unsqueeze::<3>().unsqueeze(),
            None => attn,
        };

        let attn = burn::tensor::activation::softmax(attn, 3);
        let out = attn.matmul(v);

        // Reshape back: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden]
        let out = out.swap_dims(1, 2);
        let out = out.reshape([batch, seq_len, self.num_heads * self.head_dim]);

        self.o_proj.forward(out)
    }

    /// Repeat KV heads for grouped-query attention
    fn repeat_kv(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        if self.num_kv_heads == self.num_heads {
            return x;
        }

        let [batch, kv_heads, seq_len, head_dim] = x.dims();
        let n_rep = self.num_heads / self.num_kv_heads;

        x.unsqueeze_dim::<5>(2).repeat_dim(2, n_rep).reshape([
            batch,
            kv_heads * n_rep,
            seq_len,
            head_dim,
        ])
    }
}

/// Pre-norm Transformer Block (LLaMA-style)
///
/// Architecture:
/// ```text
/// x = x + attention(norm1(x))
/// x = x + ffn(norm2(x))
/// ```
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    /// Multi-head attention
    pub attention: MultiHeadAttention<B>,
    /// Feed-forward network
    pub ffn: SwiGluFfn<B>,
    /// Input layer norm (before attention)
    pub input_norm: RmsNorm<B>,
    /// Post-attention layer norm (before FFN)
    pub post_attention_norm: RmsNorm<B>,
}

/// Configuration for TransformerBlock
pub struct TransformerBlockConfig {
    /// Hidden dimension
    pub hidden_size: usize,
    /// Intermediate FFN dimension
    pub intermediate_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of KV heads (for GQA)
    pub num_kv_heads: usize,
    /// RMSNorm epsilon
    pub norm_eps: f64,
}

impl TransformerBlockConfig {
    /// Create config with standard settings
    pub fn new(hidden_size: usize, intermediate_size: usize, num_heads: usize) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            num_heads,
            num_kv_heads: num_heads,
            norm_eps: 1e-6,
        }
    }

    /// Create config with GQA
    pub fn with_gqa(
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
    ) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            num_heads,
            num_kv_heads,
            norm_eps: 1e-6,
        }
    }

    /// Initialize the transformer block
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerBlock<B> {
        TransformerBlock {
            attention: MultiHeadAttentionConfig::gqa(
                self.hidden_size,
                self.num_heads,
                self.num_kv_heads,
            )
            .init(device),
            ffn: SwiGluFfnConfig::new(self.hidden_size, self.intermediate_size).init(device),
            input_norm: RmsNorm::with_eps(self.hidden_size, self.norm_eps, device),
            post_attention_norm: RmsNorm::with_eps(self.hidden_size, self.norm_eps, device),
        }
    }
}

impl<B: Backend> TransformerBlock<B> {
    /// Forward pass through the transformer block
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor [batch, seq_len, hidden_size]
    /// * `rope` - Optional rotary embeddings
    /// * `start_pos` - Position offset for KV cache
    /// * `mask` - Optional attention mask
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: Option<&RotaryEmbedding<B>>,
        start_pos: usize,
        mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        // Pre-norm attention with residual
        let h = x.clone()
            + self
                .attention
                .forward(self.input_norm.forward(x), rope, start_pos, mask);

        // Pre-norm FFN with residual
        h.clone() + self.ffn.forward(self.post_attention_norm.forward(h))
    }
}

/// Creates a causal attention mask for autoregressive decoding
///
/// Returns a mask where future positions are set to -inf
pub fn causal_mask<B: Backend>(seq_len: usize, device: &B::Device) -> Tensor<B, 2> {
    // Create upper triangular matrix with -inf
    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask_data[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::<B, 1>::from_floats(mask_data.as_slice(), device).reshape([seq_len, seq_len])
}

/// Creates a sliding window causal attention mask
///
/// Like causal_mask, but also masks positions beyond the window size.
/// Used by Mistral for efficient long-context attention.
///
/// # Arguments
///
/// * `seq_len` - Sequence length
/// * `window_size` - Sliding window size (e.g., 4096 for Mistral)
/// * `device` - Device to create tensor on
pub fn sliding_window_mask<B: Backend>(
    seq_len: usize,
    window_size: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            // Mask future positions (causal) or positions outside sliding window
            if j > i || i > j + window_size {
                mask_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::<B, 1>::from_floats(mask_data.as_slice(), device).reshape([seq_len, seq_len])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_mha_shape() {
        let device = Default::default();
        let config = MultiHeadAttentionConfig::new(256, 8);
        let mha = config.init::<TestBackend>(&device);

        let x = Tensor::zeros([2, 16, 256], &device);
        let y = mha.forward(x, None, 0, None);

        assert_eq!(y.dims(), [2, 16, 256]);
    }

    #[test]
    fn test_mha_with_rope() {
        let device = Default::default();
        let config = MultiHeadAttentionConfig::new(256, 8);
        let mha = config.init::<TestBackend>(&device);
        let rope = RotaryEmbedding::new(32, 128, &device); // head_dim = 256/8 = 32

        let x = Tensor::zeros([2, 16, 256], &device);
        let y = mha.forward(x, Some(&rope), 0, None);

        assert_eq!(y.dims(), [2, 16, 256]);
    }

    #[test]
    fn test_gqa() {
        let device = Default::default();
        // 8 query heads, 2 KV heads (4x GQA)
        let config = MultiHeadAttentionConfig::gqa(256, 8, 2);
        let mha = config.init::<TestBackend>(&device);

        let x = Tensor::zeros([2, 16, 256], &device);
        let y = mha.forward(x, None, 0, None);

        assert_eq!(y.dims(), [2, 16, 256]);
    }

    #[test]
    fn test_transformer_block() {
        let device = Default::default();
        let config = TransformerBlockConfig::new(256, 512, 8);
        let block = config.init::<TestBackend>(&device);

        let x = Tensor::zeros([2, 16, 256], &device);
        let y = block.forward(x, None, 0, None);

        assert_eq!(y.dims(), [2, 16, 256]);
    }

    #[test]
    fn test_transformer_block_with_mask() {
        let device = Default::default();
        let config = TransformerBlockConfig::new(128, 256, 4);
        let block = config.init::<TestBackend>(&device);

        let x = Tensor::zeros([1, 8, 128], &device);
        let mask = causal_mask::<TestBackend>(8, &device);
        let y = block.forward(x, None, 0, Some(mask));

        assert_eq!(y.dims(), [1, 8, 128]);
    }

    #[test]
    fn test_causal_mask() {
        let device = Default::default();
        let mask: Tensor<TestBackend, 2> = causal_mask(4, &device);
        let mask_data: Vec<f32> = mask.into_data().to_vec().unwrap();

        // Check that upper triangle is -inf
        // mask_data layout: [0,0], [0,1], [0,2], [0,3], [1,0], ...
        assert_eq!(mask_data[0], 0.0); // [0,0] diagonal
        assert_eq!(mask_data[1], f32::NEG_INFINITY); // [0,1] above diagonal
        assert_eq!(mask_data[2], f32::NEG_INFINITY); // [0,2] above diagonal
        assert_eq!(mask_data[4], 0.0); // [1,0] below diagonal
        assert_eq!(mask_data[5], 0.0); // [1,1] diagonal
    }

    #[test]
    fn test_sliding_window_mask() {
        let device = Default::default();
        // Window size 2: position i can attend to positions [i-2, i-1, i]
        let mask: Tensor<TestBackend, 2> = sliding_window_mask(5, 2, &device);
        let mask_data: Vec<f32> = mask.into_data().to_vec().unwrap();

        // Row 0: can attend to position 0 only
        assert_eq!(mask_data[0], 0.0); // [0,0]
        assert_eq!(mask_data[1], f32::NEG_INFINITY); // [0,1] future

        // Row 3: can attend to positions 1, 2, 3 (i=3, window=2: 3-0=3>2, 3-1=2<=2)
        assert_eq!(mask_data[15], f32::NEG_INFINITY); // [3,0] outside window (3>0+2)
        assert_eq!(mask_data[16], 0.0); // [3,1] in window (3<=1+2=3)
        assert_eq!(mask_data[17], 0.0); // [3,2] in window
        assert_eq!(mask_data[18], 0.0); // [3,3] current

        // Row 4: can attend to positions 2, 3, 4
        assert_eq!(mask_data[20], f32::NEG_INFINITY); // [4,0] outside window
        assert_eq!(mask_data[21], f32::NEG_INFINITY); // [4,1] outside window
        assert_eq!(mask_data[22], 0.0); // [4,2] in window
        assert_eq!(mask_data[23], 0.0); // [4,3] in window
        assert_eq!(mask_data[24], 0.0); // [4,4] current
    }
}
