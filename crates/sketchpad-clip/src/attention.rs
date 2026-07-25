//! Shared attention utilities for CLIP encoders

use burn::prelude::*;

/// Precompute a causal attention mask for a given max sequence length.
///
/// Call this once at initialization time and slice the result in forward().
/// This avoids allocating a Vec<f32> on every forward pass.
///
/// Uses -1e9 instead of -inf for bf16 compatibility. -inf in bf16 can cause
/// NaN values in softmax when combined with max-subtraction stabilization.
#[rustfmt::skip]
pub fn precompute_causal_mask<B: Backend>(max_seq_len: usize, device: &B::Device) -> Tensor<B, 2> {
    let mut mask_data = vec![0.0f32; max_seq_len * max_seq_len];
    for i in 0..max_seq_len {
        for j in (i + 1)..max_seq_len {
            // Use large negative value instead of -inf for bf16 compatibility
            mask_data[i * max_seq_len + j] = -1e9;
        }
    }
    let data = TensorData::new(mask_data, [max_seq_len, max_seq_len]);
    Tensor::from_data(data, device)
}

/// Slice a precomputed causal mask to the actual sequence length.
///
/// This is a cheap operation (no allocation) compared to creating a new mask.
pub fn slice_causal_mask<B: Backend>(mask: &Tensor<B, 2>, seq_len: usize) -> Tensor<B, 2> {
    mask.clone().slice([0..seq_len, 0..seq_len])
}

/// Create a causal attention mask (convenience function, allocates on each call)
///
/// For hot paths, prefer `precompute_causal_mask` + `slice_causal_mask`.
pub fn create_causal_mask<B: Backend>(seq_len: usize, device: &B::Device) -> Tensor<B, 2> {
    precompute_causal_mask(seq_len, device)
}

/// Compute scaled dot-product attention
///
/// Shared attention computation used by both CLIP and OpenCLIP.
/// Uses numerically stable softmax (max-subtraction) for f16 compatibility.
pub fn scaled_dot_product_attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    mask: Option<Tensor<B, 2>>,
    head_dim: usize,
) -> Tensor<B, 4> {
    // Scaled dot-product attention
    let scale = (head_dim as f64).powf(-0.5);
    let attn = q.matmul(k.transpose()) * scale;

    // Apply causal mask
    let attn = match mask {
        Some(m) => attn + m.unsqueeze::<4>(),
        None => attn,
    };

    // Stable softmax: subtract max before exp to prevent f16 overflow
    let attn_max = attn.clone().max_dim(3);
    let attn = (attn - attn_max).exp();
    // Add small epsilon to prevent division by zero when all values underflow to 0
    let attn_sum = attn.clone().sum_dim(3) + 1e-8;
    let attn = attn / attn_sum;

    attn.matmul(v)
}
