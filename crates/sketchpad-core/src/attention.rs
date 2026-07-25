use burn::prelude::*;

/// Multi-head self-attention
pub fn qkv_attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    mask: Option<Tensor<B, 2>>,
) -> Tensor<B, 4> {
    let [_batch, _heads, _seq_len, head_dim] = q.dims();
    let scale = (head_dim as f64).powf(-0.25);

    let q = q * scale;
    let k = k * scale;

    // [batch, heads, seq_q, seq_k]
    let attn = q.matmul(k.transpose());

    let attn = match mask {
        Some(m) => attn + m.unsqueeze::<4>(),
        None => attn,
    };

    let attn = burn::tensor::activation::softmax(attn, 3);

    attn.matmul(v)
}

/// Causal attention mask for autoregressive decoding
///
/// Creates an upper triangular matrix with -inf values, which when added to
/// attention scores prevents attending to future positions.
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
