//! Rotary Position Embedding (RoPE)
//!
//! Provides rotary position embeddings as used in modern transformer architectures
//! like LLaMA, Qwen, Mistral, and many others. RoPE encodes positional information
//! by rotating query and key vectors in a way that makes attention scores depend
//! on relative positions.

use burn::prelude::*;

/// Rotary Position Embedding
///
/// RoPE encodes position by rotating pairs of dimensions in the embedding space.
/// The rotation angle depends on both the position and the dimension index,
/// allowing the model to learn relative positional relationships.
///
/// # Formula
///
/// For a vector x at position m, each pair of dimensions (x_i, x_{i+1}) is rotated:
/// ```text
/// x'_i     = x_i * cos(mθ_i) - x_{i+1} * sin(mθ_i)
/// x'_{i+1} = x_i * sin(mθ_i) + x_{i+1} * cos(mθ_i)
/// ```
/// where θ_i = base^(-2i/d) and base is typically 10000.
///
/// # References
///
/// - [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
/// - Used in LLaMA, Qwen, Mistral, Falcon, and many modern LLMs
#[derive(Debug, Clone)]
pub struct RotaryEmbedding<B: Backend> {
    cos_cached: Tensor<B, 2>,
    sin_cached: Tensor<B, 2>,
    head_dim: usize,
}

impl<B: Backend> RotaryEmbedding<B> {
    /// Creates a new rotary embedding with the specified parameters
    ///
    /// # Arguments
    ///
    /// * `head_dim` - Dimension of each attention head (must be even)
    /// * `max_seq_len` - Maximum sequence length to precompute
    /// * `device` - Device to create tensors on
    ///
    /// # Panics
    ///
    /// Panics if `head_dim` is not even
    pub fn new(head_dim: usize, max_seq_len: usize, device: &B::Device) -> Self {
        Self::with_base(head_dim, max_seq_len, 10000.0, device)
    }

    /// Creates rotary embedding with a custom base frequency
    ///
    /// # Arguments
    ///
    /// * `head_dim` - Dimension of each attention head (must be even)
    /// * `max_seq_len` - Maximum sequence length to precompute
    /// * `base` - Base frequency for position encoding (default: 10000.0)
    /// * `device` - Device to create tensors on
    pub fn with_base(head_dim: usize, max_seq_len: usize, base: f32, device: &B::Device) -> Self {
        assert!(head_dim % 2 == 0, "head_dim must be even for RoPE");

        // Compute inverse frequencies: base^(-2i/d) for i in 0..d/2
        let half_dim = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / base.powf((2 * i) as f32 / head_dim as f32))
            .collect();

        let inv_freq = Tensor::<B, 1>::from_floats(inv_freq.as_slice(), device);

        // Compute position indices
        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::<B, 1>::from_floats(positions.as_slice(), device);

        // Compute angles: position * inv_freq -> [seq_len, head_dim/2]
        let angles = positions
            .unsqueeze::<2>()
            .transpose()
            .matmul(inv_freq.unsqueeze());

        // Cache cos and sin values, repeated for pairs: [seq_len, head_dim]
        let cos_cached = angles.clone().cos();
        let sin_cached = angles.sin();

        // Interleave to match dimension pairs
        let cos_cached = Self::repeat_interleave(cos_cached);
        let sin_cached = Self::repeat_interleave(sin_cached);

        Self {
            cos_cached,
            sin_cached,
            head_dim,
        }
    }

    /// Repeats each value to create pairs: [a, b, c] -> [a, a, b, b, c, c]
    fn repeat_interleave(x: Tensor<B, 2>) -> Tensor<B, 2> {
        let [seq_len, half_dim] = x.dims();
        // Stack and reshape to interleave
        let x_expanded = x.unsqueeze_dim::<3>(2);
        let x_repeated = x_expanded.repeat_dim(2, 2);
        x_repeated.reshape([seq_len, half_dim * 2])
    }

    /// Applies rotary embedding to query and key tensors
    ///
    /// # Arguments
    ///
    /// * `q` - Query tensor of shape [batch, heads, seq_len, head_dim]
    /// * `k` - Key tensor of shape [batch, heads, seq_len, head_dim]
    /// * `start_pos` - Starting position for the sequence (for KV cache)
    ///
    /// # Returns
    ///
    /// Tuple of (rotated_q, rotated_k) with same shapes as inputs
    pub fn forward(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        start_pos: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [_batch, _heads, seq_len, _head_dim] = q.dims();

        // Get cached cos/sin for this position range
        let cos = self
            .cos_cached
            .clone()
            .slice([start_pos..start_pos + seq_len, 0..self.head_dim]);
        let sin = self
            .sin_cached
            .clone()
            .slice([start_pos..start_pos + seq_len, 0..self.head_dim]);

        // Broadcast to [1, 1, seq_len, head_dim]
        let cos = cos.unsqueeze::<3>().unsqueeze();
        let sin = sin.unsqueeze::<3>().unsqueeze();

        let q_rotated = Self::apply_rotary(q, cos.clone(), sin.clone());
        let k_rotated = Self::apply_rotary(k, cos, sin);

        (q_rotated, k_rotated)
    }

    /// Applies rotation to a single tensor
    fn apply_rotary(x: Tensor<B, 4>, cos: Tensor<B, 4>, sin: Tensor<B, 4>) -> Tensor<B, 4> {
        // Rotate pairs: for (x0, x1), compute (x0*cos - x1*sin, x0*sin + x1*cos)
        let x_rotated = Self::rotate_half(x.clone());
        x * cos + x_rotated * sin
    }

    /// Rotates adjacent pairs: [x0, x1, x2, x3] -> [-x1, x0, -x3, x2]
    fn rotate_half(x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, heads, seq_len, head_dim] = x.dims();
        let half = head_dim / 2;

        // Reshape to [..., head_dim/2, 2]
        let x_reshaped = x.reshape([batch, heads, seq_len, half, 2]);

        // Split into even and odd
        let x_even = x_reshaped
            .clone()
            .slice([0..batch, 0..heads, 0..seq_len, 0..half, 0..1]);
        let x_odd = x_reshaped.slice([0..batch, 0..heads, 0..seq_len, 0..half, 1..2]);

        // Rotate: (-odd, even)
        let neg_x_odd = x_odd.neg();
        let rotated = Tensor::cat(vec![neg_x_odd, x_even], 4);

        rotated.reshape([batch, heads, seq_len, head_dim])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_rope_shape_preserved() {
        let device = Default::default();
        let rope = RotaryEmbedding::<TestBackend>::new(64, 128, &device);

        let q = Tensor::zeros([2, 8, 16, 64], &device);
        let k = Tensor::zeros([2, 8, 16, 64], &device);

        let (q_rot, k_rot) = rope.forward(q, k, 0);

        assert_eq!(q_rot.dims(), [2, 8, 16, 64]);
        assert_eq!(k_rot.dims(), [2, 8, 16, 64]);
    }

    #[test]
    fn test_rope_with_offset() {
        let device = Default::default();
        let rope = RotaryEmbedding::<TestBackend>::new(32, 256, &device);

        let q = Tensor::zeros([1, 4, 8, 32], &device);
        let k = Tensor::zeros([1, 4, 8, 32], &device);

        // Test with position offset (simulating KV cache)
        let (q_rot, k_rot) = rope.forward(q, k, 100);

        assert_eq!(q_rot.dims(), [1, 4, 8, 32]);
        assert_eq!(k_rot.dims(), [1, 4, 8, 32]);
    }

    #[test]
    fn test_rope_modifies_values() {
        let device = Default::default();
        let rope = RotaryEmbedding::<TestBackend>::new(4, 16, &device);

        let q: Tensor<TestBackend, 4> = Tensor::ones([1, 1, 4, 4], &device);
        let k: Tensor<TestBackend, 4> = Tensor::ones([1, 1, 4, 4], &device);

        let (q_rot, _k_rot) = rope.forward(q.clone(), k, 0);

        // Rotated values should differ from original
        let q_data: Vec<f32> = q.into_data().to_vec().unwrap();
        let q_rot_data: Vec<f32> = q_rot.into_data().to_vec().unwrap();

        // At least some values should change
        let any_changed = q_data
            .iter()
            .zip(q_rot_data.iter())
            .any(|(orig, rotated)| (orig - rotated).abs() > 1e-6);
        assert!(any_changed, "RoPE should modify the input values");
    }
}
