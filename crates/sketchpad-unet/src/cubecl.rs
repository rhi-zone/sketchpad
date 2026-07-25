//! CubeCL-accelerated UNet blocks
//!
//! This module provides optimized versions of UNet building blocks
//! that use fused CubeCL kernels for better GPU performance.
//!
//! # Usage
//!
//! Enable the `cubecl` feature and use `ResBlockCubeCL` instead of `ResBlock`:
//!
//! ```ignore
//! use sketchpad_unet::cubecl::ResBlockCubeCL;
//! use cubecl::wgpu::WgpuRuntime;
//!
//! let block = ResBlockCubeCL::<WgpuRuntime>::new(256, 512, 1024, &device);
//! let output = block.forward(input, time_emb);
//! ```

use burn::nn::{
    Linear, LinearConfig, PaddingConfig2d,
    conv::{Conv2d, Conv2dConfig},
};
use burn::prelude::*;
use burn::tensor::DType;
use burn_cubecl::{
    BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement, tensor::CubeTensor,
};
use sketchpad_cubecl::{
    FlashAttentionOptions, GroupNormSiLuOptions, cube_to_tensor, flash_attention, groupnorm_silu,
    tensor_to_cube,
};

/// CubeCL-accelerated ResNet block with fused GroupNorm+SiLU
///
/// Uses fused GPU kernels for the `norm → silu` pattern, reducing
/// memory traffic and kernel launch overhead.
#[derive(Debug)]
pub struct ResBlockCubeCL<R: CubeRuntime> {
    norm1_weight: CubeTensor<R>,
    norm1_bias: CubeTensor<R>,
    conv1: Conv2d<CubeBackend<R, f32, i32, u32>>,
    time_emb_proj: Linear<CubeBackend<R, f32, i32, u32>>,
    norm2_weight: CubeTensor<R>,
    norm2_bias: CubeTensor<R>,
    conv2: Conv2d<CubeBackend<R, f32, i32, u32>>,
    skip_conv: Option<Conv2d<CubeBackend<R, f32, i32, u32>>>,
    num_groups: usize,
}

type B<R> = CubeBackend<R, f32, i32, u32>;

impl<R: CubeRuntime> ResBlockCubeCL<R> {
    /// Creates a new CubeCL-accelerated residual block
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        time_emb_dim: usize,
        device: &R::Device,
    ) -> Self {
        let num_groups = 32;

        // Initialize norm weights (ones) and biases (zeros)
        let norm1_weight = Tensor::<B<R>, 1>::ones([in_channels], device);
        let norm1_bias = Tensor::<B<R>, 1>::zeros([in_channels], device);

        let conv1 = Conv2dConfig::new([in_channels, out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        let time_emb_proj = LinearConfig::new(time_emb_dim, out_channels).init(device);

        let norm2_weight = Tensor::<B<R>, 1>::ones([out_channels], device);
        let norm2_bias = Tensor::<B<R>, 1>::zeros([out_channels], device);

        let conv2 = Conv2dConfig::new([out_channels, out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        let skip_conv = if in_channels != out_channels {
            Some(Conv2dConfig::new([in_channels, out_channels], [1, 1]).init(device))
        } else {
            None
        };

        Self {
            norm1_weight: tensor_to_cube(norm1_weight),
            norm1_bias: tensor_to_cube(norm1_bias),
            conv1,
            time_emb_proj,
            norm2_weight: tensor_to_cube(norm2_weight),
            norm2_bias: tensor_to_cube(norm2_bias),
            conv2,
            skip_conv,
            num_groups,
        }
    }

    /// Forward pass with fused GroupNorm+SiLU kernels
    pub fn forward(&self, x: Tensor<B<R>, 4>, time_emb: Tensor<B<R>, 2>) -> Tensor<B<R>, 4> {
        let [b, _, _h, _w] = x.dims();

        let residual = match &self.skip_conv {
            Some(conv) => conv.forward(x.clone()),
            None => x.clone(),
        };

        // First conv block with FUSED GroupNorm+SiLU
        let hidden = groupnorm_silu(
            tensor_to_cube(x),
            self.norm1_weight.clone(),
            self.norm1_bias.clone(),
            GroupNormSiLuOptions::with_groups(self.num_groups),
        );
        let hidden: Tensor<B<R>, 4> = cube_to_tensor(hidden);
        let hidden = self.conv1.forward(hidden);

        // Add time embedding (silu applied to time_emb)
        let time_emb = sketchpad_core::silu::silu(time_emb);
        let time_emb = self.time_emb_proj.forward(time_emb);
        let emb_dim = time_emb.dims()[1];
        let time_emb = time_emb.reshape([b, emb_dim, 1, 1]);
        let hidden = hidden + time_emb;

        // Second conv block with FUSED GroupNorm+SiLU
        let hidden = groupnorm_silu(
            tensor_to_cube(hidden),
            self.norm2_weight.clone(),
            self.norm2_bias.clone(),
            GroupNormSiLuOptions::with_groups(self.num_groups),
        );
        let hidden: Tensor<B<R>, 4> = cube_to_tensor(hidden);
        let hidden = self.conv2.forward(hidden);

        hidden + residual
    }
}

/// Convert a standard ResBlock to CubeCL-accelerated version
///
/// This extracts the weights from an existing ResBlock and creates
/// a CubeCL-accelerated version that uses fused kernels.
pub fn convert_resblock<R: CubeRuntime>(
    block: &crate::blocks::ResBlock<B<R>>,
) -> ResBlockCubeCL<R> {
    ResBlockCubeCL {
        norm1_weight: tensor_to_cube(block.norm1.weight.clone()),
        norm1_bias: tensor_to_cube(block.norm1.bias.clone()),
        conv1: block.conv1.clone(),
        time_emb_proj: block.time_emb_proj.clone(),
        norm2_weight: tensor_to_cube(block.norm2.weight.clone()),
        norm2_bias: tensor_to_cube(block.norm2.bias.clone()),
        conv2: block.conv2.clone(),
        skip_conv: block.skip_conv.clone(),
        num_groups: block.norm1.num_groups,
    }
}

/// CubeCL-accelerated cross-attention with Flash Attention
///
/// Uses Flash Attention kernel for O(n) memory attention computation.
/// This is a drop-in replacement for `CrossAttention` that uses GPU-optimized
/// tiled attention instead of materializing the full attention matrix.
///
/// Flash attention uses f32 accumulation internally, which solves f16 overflow
/// issues that cause NaN in the standard attention implementation.
#[derive(Debug)]
pub struct CrossAttentionCubeCL<
    R: CubeRuntime,
    F: FloatElement = f32,
    I: IntElement = i32,
    BT: BoolElement = u32,
> {
    to_q: Linear<CubeBackend<R, F, I, BT>>,
    to_k: Linear<CubeBackend<R, F, I, BT>>,
    to_v: Linear<CubeBackend<R, F, I, BT>>,
    to_out: Linear<CubeBackend<R, F, I, BT>>,
    num_heads: usize,
    head_dim: usize,
}

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement>
    CrossAttentionCubeCL<R, F, I, BT>
{
    /// Creates a new CubeCL-accelerated cross-attention module
    ///
    /// # Arguments
    ///
    /// * `query_dim` - Dimension of query input
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per attention head
    /// * `context_dim` - Dimension of key/value context (None for self-attention)
    /// * `device` - Device to create tensors on
    pub fn new(
        query_dim: usize,
        num_heads: usize,
        head_dim: usize,
        context_dim: Option<usize>,
        device: &R::Device,
    ) -> Self {
        let inner_dim = num_heads * head_dim;
        let context_dim = context_dim.unwrap_or(query_dim);

        Self {
            to_q: LinearConfig::new(query_dim, inner_dim)
                .with_bias(false)
                .init(device),
            to_k: LinearConfig::new(context_dim, inner_dim)
                .with_bias(false)
                .init(device),
            to_v: LinearConfig::new(context_dim, inner_dim)
                .with_bias(false)
                .init(device),
            to_out: LinearConfig::new(inner_dim, query_dim).init(device),
            num_heads,
            head_dim,
        }
    }

    /// Computes attention using Flash Attention kernel
    ///
    /// Flash attention uses f32 accumulation internally, which prevents
    /// overflow issues that cause NaN in standard attention with f16 inputs.
    ///
    /// # Arguments
    ///
    /// * `x` - Query input of shape `[batch, seq_len, query_dim]`
    /// * `context` - Key/value context (None uses x for self-attention)
    ///
    /// # Returns
    ///
    /// Attention output of shape `[batch, seq_len, query_dim]`
    pub fn forward(
        &self,
        x: Tensor<CubeBackend<R, F, I, BT>, 3>,
        context: Option<Tensor<CubeBackend<R, F, I, BT>, 3>>,
    ) -> Tensor<CubeBackend<R, F, I, BT>, 3> {
        let context = context.unwrap_or_else(|| x.clone());

        let [b, seq_len, _] = x.dims();
        let [_, ctx_len, _] = context.dims();

        let q = self.to_q.forward(x);
        let k = self.to_k.forward(context.clone());
        let v = self.to_v.forward(context);

        // Reshape to multi-head: [b, seq, heads*dim] -> [b, heads, seq, dim]
        let q = q
            .reshape([b, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([b, ctx_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([b, ctx_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);

        // Pad dimensions to next power of 2 for cubek-attention compatibility
        // SD 1.x uses head_dim=40 which doesn't align with f16 tiling
        // Also ctx_len=77 (CLIP tokens) may need padding
        let padded_head_dim = next_power_of_2(self.head_dim);
        let padded_ctx_len = next_power_of_2(ctx_len);
        let need_pad_head = padded_head_dim != self.head_dim;
        let need_pad_ctx = padded_ctx_len != ctx_len;

        let debug_attn = std::env::var("DEBUG_ATTENTION").is_ok();
        if debug_attn {
            eprintln!(
                "[attention] seq_len={}, ctx_len={}, head_dim={} → pad_head={} (to {}), pad_ctx={} (to {})",
                seq_len,
                ctx_len,
                self.head_dim,
                need_pad_head,
                padded_head_dim,
                need_pad_ctx,
                padded_ctx_len
            );
        }

        let (q, k, v) = if need_pad_head || need_pad_ctx {
            let device = q.device();

            // When padding head_dim, we need to correct the scale factor.
            // Flash attention uses 1/sqrt(padded_head_dim) but we need 1/sqrt(original_head_dim).
            // Correct by pre-scaling Q: Q' = Q * sqrt(padded_head_dim) / sqrt(original_head_dim)
            // Then: Q' @ K^T / sqrt(padded) = Q @ K^T / sqrt(original)
            let q = if need_pad_head {
                let scale_correction = ((padded_head_dim as f64) / (self.head_dim as f64)).sqrt();
                let q = q * scale_correction;
                pad_last_dim(q, padded_head_dim, &device)
            } else {
                q
            };
            let k = if need_pad_head {
                pad_last_dim(k, padded_head_dim, &device)
            } else {
                k
            };
            let v = if need_pad_head {
                pad_last_dim(v, padded_head_dim, &device)
            } else {
                v
            };
            // Pad K and V along sequence dimension (dim 2) for context alignment
            let k = if need_pad_ctx {
                pad_dim::<_, 4>(k, 2, padded_ctx_len, &device)
            } else {
                k
            };
            let v = if need_pad_ctx {
                pad_dim::<_, 4>(v, 2, padded_ctx_len, &device)
            } else {
                v
            };
            (q, k, v)
        } else {
            (q, k, v)
        };

        // Debug: print shapes right before flash attention
        if debug_attn {
            let q_dims = q.dims();
            let k_dims = k.dims();
            let v_dims = v.dims();
            eprintln!(
                "[attention] Q={:?} K={:?} V={:?} (batch={}, heads={}, seq_q={}, seq_k={}, head_dim={})",
                q_dims, k_dims, v_dims, q_dims[0], q_dims[1], q_dims[2], k_dims[2], q_dims[3]
            );
        }

        // Flash attention (non-causal for diffusion models)
        //
        // WORKAROUND: cubek-attention 0.1.0-pre.1 has a bug where f16/bf16 fail on CUDA
        // due to tile size alignment issues. We cast to f32 for attention computation.
        // See: https://github.com/tracel-ai/cubek/pull/55
        //
        // Memory overhead: ~2x for Q/K/V during attention (temporary)
        // Performance: slight overhead from casting, but attention itself is fast
        let needs_cast = q.dtype() != DType::F32;
        let (q_attn, k_attn, v_attn) = if needs_cast {
            (
                q.clone().cast(DType::F32),
                k.clone().cast(DType::F32),
                v.clone().cast(DType::F32),
            )
        } else {
            (q.clone(), k.clone(), v.clone())
        };

        let out = flash_attention(
            tensor_to_cube(q_attn),
            tensor_to_cube(k_attn),
            tensor_to_cube(v_attn),
            FlashAttentionOptions::default(), // non-causal
        )
        .expect("Flash attention failed");

        let out: Tensor<CubeBackend<R, F, I, BT>, 4> = cube_to_tensor(out);
        // Cast back to original dtype if we casted to f32
        let out = if needs_cast { out.cast(q.dtype()) } else { out };

        // Slice back to original head_dim if we padded
        let out = if need_pad_head {
            out.narrow(3, 0, self.head_dim)
        } else {
            out
        };

        // Reshape back: [b, heads, seq, dim] -> [b, seq, heads*dim]
        let out = out
            .swap_dims(1, 2)
            .reshape([b, seq_len, self.num_heads * self.head_dim]);

        self.to_out.forward(out)
    }
}

/// Round up to next power of 2 (minimum 8)
fn next_power_of_2(n: usize) -> usize {
    if n <= 8 {
        return 8;
    }
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

/// Pad tensor along last dimension with zeros
fn pad_last_dim<B: burn::prelude::Backend, const D: usize>(
    tensor: Tensor<B, D>,
    target_dim: usize,
    device: &B::Device,
) -> Tensor<B, D> {
    pad_dim::<B, D>(tensor, D - 1, target_dim, device)
}

/// Pad tensor along specified dimension with zeros
fn pad_dim<B: burn::prelude::Backend, const D: usize>(
    tensor: Tensor<B, D>,
    dim: usize,
    target_size: usize,
    device: &B::Device,
) -> Tensor<B, D> {
    let dims = tensor.dims();
    let current_size = dims[dim];
    if current_size >= target_size {
        return tensor;
    }

    // Build padded shape
    let mut padded_dims = dims;
    padded_dims[dim] = target_size;

    // Create zeros tensor and assign original into it
    let padded = Tensor::zeros(padded_dims, device);

    // Build slice ranges for assignment
    let ranges: [std::ops::Range<usize>; D] = std::array::from_fn(|i| {
        if i == dim {
            0..current_size
        } else {
            0..dims[i]
        }
    });

    padded.slice_assign(ranges, tensor)
}

/// Convert a standard CrossAttention to CubeCL-accelerated version
///
/// The resulting CrossAttentionCubeCL uses flash attention with f32 accumulation,
/// which prevents NaN issues that occur with f16 in standard attention.
pub fn convert_crossattention<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement>(
    attn: &crate::blocks::CrossAttention<CubeBackend<R, F, I, BT>>,
) -> CrossAttentionCubeCL<R, F, I, BT> {
    CrossAttentionCubeCL {
        to_q: attn.to_q.clone(),
        to_k: attn.to_k.clone(),
        to_v: attn.to_v.clone(),
        to_out: attn.to_out.clone(),
        num_heads: attn.num_heads,
        head_dim: attn.head_dim,
    }
}

/// CubeCL-accelerated transformer block with Flash Attention
///
/// Uses flash attention for both self-attention and cross-attention,
/// preventing f16 overflow issues.
#[derive(Debug)]
pub struct TransformerBlockCubeCL<
    R: CubeRuntime,
    F: FloatElement = f32,
    I: IntElement = i32,
    BT: BoolElement = u32,
> {
    norm1: sketchpad_core::layernorm::LayerNorm<CubeBackend<R, F, I, BT>>,
    attn1: CrossAttentionCubeCL<R, F, I, BT>,
    norm2: sketchpad_core::layernorm::LayerNorm<CubeBackend<R, F, I, BT>>,
    attn2: CrossAttentionCubeCL<R, F, I, BT>,
    norm3: sketchpad_core::layernorm::LayerNorm<CubeBackend<R, F, I, BT>>,
    ff: crate::blocks::FeedForward<CubeBackend<R, F, I, BT>>,
}

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement>
    TransformerBlockCubeCL<R, F, I, BT>
{
    /// Forward pass with flash attention
    pub fn forward(
        &self,
        x: Tensor<CubeBackend<R, F, I, BT>, 3>,
        context: Tensor<CubeBackend<R, F, I, BT>, 3>,
    ) -> Tensor<CubeBackend<R, F, I, BT>, 3> {
        // Self-attention with flash attention
        let x = x.clone() + self.attn1.forward(self.norm1.forward(x.clone()), None);

        // Cross-attention with flash attention
        let x = x.clone()
            + self
                .attn2
                .forward(self.norm2.forward(x.clone()), Some(context));

        // FFN
        x.clone() + self.ff.forward(self.norm3.forward(x))
    }
}

/// Convert a TransformerBlock to CubeCL-accelerated version
pub fn convert_transformer_block<
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
>(
    block: &crate::blocks::TransformerBlock<CubeBackend<R, F, I, BT>>,
) -> TransformerBlockCubeCL<R, F, I, BT> {
    TransformerBlockCubeCL {
        norm1: block.norm1.clone(),
        attn1: convert_crossattention(&block.attn1),
        norm2: block.norm2.clone(),
        attn2: convert_crossattention(&block.attn2),
        norm3: block.norm3.clone(),
        ff: block.ff.clone(),
    }
}

/// CubeCL-accelerated spatial transformer with Flash Attention
///
/// Replaces standard attention with flash attention throughout.
#[derive(Debug)]
pub struct SpatialTransformerCubeCL<
    R: CubeRuntime,
    F: FloatElement = f32,
    I: IntElement = i32,
    BT: BoolElement = u32,
> {
    norm: sketchpad_core::groupnorm::GroupNorm<CubeBackend<R, F, I, BT>>,
    proj_in: burn::nn::conv::Conv2d<CubeBackend<R, F, I, BT>>,
    transformer_blocks: Vec<TransformerBlockCubeCL<R, F, I, BT>>,
    proj_out: burn::nn::conv::Conv2d<CubeBackend<R, F, I, BT>>,
}

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement>
    SpatialTransformerCubeCL<R, F, I, BT>
{
    /// Forward pass with flash attention in all transformer blocks
    pub fn forward(
        &self,
        x: Tensor<CubeBackend<R, F, I, BT>, 4>,
        context: Tensor<CubeBackend<R, F, I, BT>, 3>,
    ) -> Tensor<CubeBackend<R, F, I, BT>, 4> {
        let [b, _c, h, w] = x.dims();
        let residual = x.clone();

        let x = self.norm.forward(x);
        let x = self.proj_in.forward(x);

        // Reshape to sequence: [b, c, h, w] -> [b, h*w, c]
        let inner_dim = x.dims()[1];
        let x = x.reshape([b, inner_dim, h * w]).swap_dims(1, 2);

        // Apply transformer blocks with flash attention
        let mut x = x;
        for block in &self.transformer_blocks {
            x = block.forward(x, context.clone());
        }

        // Reshape back: [b, h*w, c] -> [b, c, h, w]
        let x = x.swap_dims(1, 2).reshape([b, inner_dim, h, w]);
        let x = self.proj_out.forward(x);

        x + residual
    }
}

/// Convert a SpatialTransformer to CubeCL-accelerated version
pub fn convert_spatial_transformer<
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
>(
    st: &crate::blocks::SpatialTransformer<CubeBackend<R, F, I, BT>>,
) -> SpatialTransformerCubeCL<R, F, I, BT> {
    SpatialTransformerCubeCL {
        norm: st.norm.clone(),
        proj_in: st.proj_in.clone(),
        transformer_blocks: st
            .transformer_blocks
            .iter()
            .map(convert_transformer_block)
            .collect(),
        proj_out: st.proj_out.clone(),
    }
}

// ============================================================================
// UNet Block Types with Flash Attention
// ============================================================================

/// CubeCL-accelerated down block with Flash Attention
#[derive(Debug)]
pub struct DownBlockCubeCL<
    R: CubeRuntime,
    F: FloatElement = f32,
    I: IntElement = i32,
    BT: BoolElement = u32,
> {
    res1: crate::blocks::ResBlock<CubeBackend<R, F, I, BT>>,
    attn1: Option<SpatialTransformerCubeCL<R, F, I, BT>>,
    res2: crate::blocks::ResBlock<CubeBackend<R, F, I, BT>>,
    attn2: Option<SpatialTransformerCubeCL<R, F, I, BT>>,
    downsample: Option<crate::blocks::Downsample<CubeBackend<R, F, I, BT>>>,
}

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> DownBlockCubeCL<R, F, I, BT> {
    /// Forward pass with flash attention, returns output and skip connections
    #[allow(clippy::type_complexity)]
    pub fn forward(
        &self,
        x: Tensor<CubeBackend<R, F, I, BT>, 4>,
        t_emb: Tensor<CubeBackend<R, F, I, BT>, 2>,
        context: Tensor<CubeBackend<R, F, I, BT>, 3>,
    ) -> (
        Tensor<CubeBackend<R, F, I, BT>, 4>,
        Vec<Tensor<CubeBackend<R, F, I, BT>, 4>>,
    ) {
        let mut skips = Vec::new();

        let h = self.res1.forward(x, t_emb.clone());
        let h = if let Some(attn) = &self.attn1 {
            attn.forward(h, context.clone())
        } else {
            h
        };
        skips.push(h.clone());

        let h = self.res2.forward(h, t_emb);
        let h = if let Some(attn) = &self.attn2 {
            attn.forward(h, context)
        } else {
            h
        };
        skips.push(h.clone());

        let h = if let Some(ds) = &self.downsample {
            let h = ds.forward(h);
            skips.push(h.clone());
            h
        } else {
            h
        };

        (h, skips)
    }
}

/// Convert a DownBlock to CubeCL-accelerated version
pub fn convert_down_block<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement>(
    block: &crate::unet_sd::DownBlock<CubeBackend<R, F, I, BT>>,
) -> DownBlockCubeCL<R, F, I, BT> {
    DownBlockCubeCL {
        res1: block.res1.clone(),
        attn1: block.attn1.as_ref().map(convert_spatial_transformer),
        res2: block.res2.clone(),
        attn2: block.attn2.as_ref().map(convert_spatial_transformer),
        downsample: block.downsample.clone(),
    }
}

/// CubeCL-accelerated mid block with Flash Attention
#[derive(Debug)]
pub struct MidBlockCubeCL<
    R: CubeRuntime,
    F: FloatElement = f32,
    I: IntElement = i32,
    BT: BoolElement = u32,
> {
    res1: crate::blocks::ResBlock<CubeBackend<R, F, I, BT>>,
    attn: SpatialTransformerCubeCL<R, F, I, BT>,
    res2: crate::blocks::ResBlock<CubeBackend<R, F, I, BT>>,
}

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> MidBlockCubeCL<R, F, I, BT> {
    /// Forward pass with flash attention
    pub fn forward(
        &self,
        x: Tensor<CubeBackend<R, F, I, BT>, 4>,
        t_emb: Tensor<CubeBackend<R, F, I, BT>, 2>,
        context: Tensor<CubeBackend<R, F, I, BT>, 3>,
    ) -> Tensor<CubeBackend<R, F, I, BT>, 4> {
        let h = self.res1.forward(x, t_emb.clone());
        let h = self.attn.forward(h, context);
        self.res2.forward(h, t_emb)
    }
}

/// Convert a MidBlock to CubeCL-accelerated version
pub fn convert_mid_block<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement>(
    block: &crate::unet_sd::MidBlock<CubeBackend<R, F, I, BT>>,
) -> MidBlockCubeCL<R, F, I, BT> {
    MidBlockCubeCL {
        res1: block.res1.clone(),
        attn: convert_spatial_transformer(&block.attn),
        res2: block.res2.clone(),
    }
}

/// CubeCL-accelerated up block with Flash Attention
#[derive(Debug)]
pub struct UpBlockCubeCL<
    R: CubeRuntime,
    F: FloatElement = f32,
    I: IntElement = i32,
    BT: BoolElement = u32,
> {
    res: crate::blocks::ResBlock<CubeBackend<R, F, I, BT>>,
    attn: Option<SpatialTransformerCubeCL<R, F, I, BT>>,
    upsample: Option<crate::blocks::Upsample<CubeBackend<R, F, I, BT>>>,
}

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> UpBlockCubeCL<R, F, I, BT> {
    /// Forward pass with flash attention
    pub fn forward(
        &self,
        x: Tensor<CubeBackend<R, F, I, BT>, 4>,
        t_emb: Tensor<CubeBackend<R, F, I, BT>, 2>,
        context: Tensor<CubeBackend<R, F, I, BT>, 3>,
    ) -> Tensor<CubeBackend<R, F, I, BT>, 4> {
        let h = self.res.forward(x, t_emb);
        let h = if let Some(attn) = &self.attn {
            attn.forward(h, context)
        } else {
            h
        };

        if let Some(up) = &self.upsample {
            up.forward(h)
        } else {
            h
        }
    }
}

/// Convert an UpBlock to CubeCL-accelerated version
pub fn convert_up_block<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement>(
    block: &crate::unet_sd::UpBlock<CubeBackend<R, F, I, BT>>,
) -> UpBlockCubeCL<R, F, I, BT> {
    UpBlockCubeCL {
        res: block.res.clone(),
        attn: block.attn.as_ref().map(convert_spatial_transformer),
        upsample: block.upsample.clone(),
    }
}

// ============================================================================
// UNet with Flash Attention
// ============================================================================

/// CubeCL-accelerated SD 1.x UNet with Flash Attention
///
/// This is a drop-in replacement for [`crate::unet_sd::UNet`] that uses
/// flash attention in all attention blocks, preventing f16 overflow.
///
/// # Usage
///
/// ```ignore
/// use sketchpad_unet::cubecl::{UNetCubeCL, convert_unet};
///
/// // Load standard UNet
/// let unet: UNet<CubeBackend<R, f16, i32, u32>> = load_unet(...);
///
/// // Convert to flash attention version
/// let unet_flash = convert_unet(&unet);
///
/// // Use in pipeline (same interface)
/// let noise = unet_flash.forward(latents, timesteps, context);
/// ```
#[derive(Debug)]
pub struct UNetCubeCL<
    R: CubeRuntime,
    F: FloatElement = f32,
    I: IntElement = i32,
    BT: BoolElement = u32,
> {
    time_embed_0: Linear<CubeBackend<R, F, I, BT>>,
    time_embed_2: Linear<CubeBackend<R, F, I, BT>>,
    time_freqs: Tensor<CubeBackend<R, F, I, BT>, 1>,
    conv_in: Conv2d<CubeBackend<R, F, I, BT>>,
    down_blocks: Vec<DownBlockCubeCL<R, F, I, BT>>,
    mid_block: MidBlockCubeCL<R, F, I, BT>,
    up_blocks: Vec<UpBlockCubeCL<R, F, I, BT>>,
    norm_out: sketchpad_core::groupnorm::GroupNorm<CubeBackend<R, F, I, BT>>,
    conv_out: Conv2d<CubeBackend<R, F, I, BT>>,
}

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> UNetCubeCL<R, F, I, BT> {
    /// Forward pass with flash attention in all attention blocks
    ///
    /// Same interface as [`crate::unet_sd::UNet::forward`].
    pub fn forward(
        &self,
        x: Tensor<CubeBackend<R, F, I, BT>, 4>,
        timesteps: Tensor<CubeBackend<R, F, I, BT>, 1>,
        context: Tensor<CubeBackend<R, F, I, BT>, 3>,
    ) -> Tensor<CubeBackend<R, F, I, BT>, 4> {
        // Time embedding
        let t_emb =
            crate::blocks::timestep_embedding_with_freqs(timesteps, self.time_freqs.clone());
        let t_emb = self.time_embed_0.forward(t_emb);
        let t_emb = sketchpad_core::silu::silu(t_emb);
        let t_emb = self.time_embed_2.forward(t_emb);

        // Input
        let mut h = self.conv_in.forward(x);

        // Down blocks with skip connections
        let mut skips = vec![h.clone()];
        for block in &self.down_blocks {
            let (out, block_skips) = block.forward(h, t_emb.clone(), context.clone());
            h = out;
            skips.extend(block_skips);
        }

        // Mid block (with flash attention)
        h = self.mid_block.forward(h, t_emb.clone(), context.clone());

        // Up blocks with skip connections
        for block in &self.up_blocks {
            let skip = skips.pop().unwrap();
            h = Tensor::cat(vec![h, skip], 1);
            h = block.forward(h, t_emb.clone(), context.clone());
        }

        // Output
        h = self.norm_out.forward(h);
        h = sketchpad_core::silu::silu(h);
        self.conv_out.forward(h)
    }
}

/// Convert a standard UNet to CubeCL-accelerated version with flash attention
///
/// This extracts all weights from the standard UNet and creates a version
/// that uses flash attention in all attention blocks. The result is
/// functionally equivalent but uses f32 accumulation in attention,
/// preventing f16 overflow.
///
/// # Example
///
/// ```ignore
/// let unet = load_unet::<CudaBackend<f16, i32, u32>>(&weights_path, &device)?;
/// let unet_flash = convert_unet(&unet);
/// ```
pub fn convert_unet<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement>(
    unet: &crate::unet_sd::UNet<CubeBackend<R, F, I, BT>>,
) -> UNetCubeCL<R, F, I, BT> {
    UNetCubeCL {
        time_embed_0: unet.time_embed_0.clone(),
        time_embed_2: unet.time_embed_2.clone(),
        time_freqs: unet.time_freqs.clone(),
        conv_in: unet.conv_in.clone(),
        down_blocks: unet.down_blocks.iter().map(convert_down_block).collect(),
        mid_block: convert_mid_block(&unet.mid_block),
        up_blocks: unet.up_blocks.iter().map(convert_up_block).collect(),
        norm_out: unet.norm_out.clone(),
        conv_out: unet.conv_out.clone(),
    }
}

// ============================================================================
// UNetXL (SDXL) with Flash Attention
// ============================================================================

/// CubeCL-accelerated SDXL down block with Flash Attention
#[derive(Debug)]
pub struct DownBlockXLCubeCL<
    R: CubeRuntime,
    F: FloatElement = f32,
    I: IntElement = i32,
    BT: BoolElement = u32,
> {
    res1: crate::blocks::ResBlock<CubeBackend<R, F, I, BT>>,
    attn1: Option<SpatialTransformerCubeCL<R, F, I, BT>>,
    res2: crate::blocks::ResBlock<CubeBackend<R, F, I, BT>>,
    attn2: Option<SpatialTransformerCubeCL<R, F, I, BT>>,
    downsample: Option<crate::blocks::Downsample<CubeBackend<R, F, I, BT>>>,
}

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement>
    DownBlockXLCubeCL<R, F, I, BT>
{
    /// Forward pass with flash attention, returns output and skip connections
    #[allow(clippy::type_complexity)]
    pub fn forward(
        &self,
        x: Tensor<CubeBackend<R, F, I, BT>, 4>,
        t_emb: Tensor<CubeBackend<R, F, I, BT>, 2>,
        context: Tensor<CubeBackend<R, F, I, BT>, 3>,
    ) -> (
        Tensor<CubeBackend<R, F, I, BT>, 4>,
        Vec<Tensor<CubeBackend<R, F, I, BT>, 4>>,
    ) {
        let mut skips = Vec::new();

        let h = self.res1.forward(x, t_emb.clone());
        let h = if let Some(attn) = &self.attn1 {
            attn.forward(h, context.clone())
        } else {
            h
        };
        skips.push(h.clone());

        let h = self.res2.forward(h, t_emb);
        let h = if let Some(attn) = &self.attn2 {
            attn.forward(h, context)
        } else {
            h
        };
        skips.push(h.clone());

        let h = if let Some(ds) = &self.downsample {
            let h = ds.forward(h);
            skips.push(h.clone());
            h
        } else {
            h
        };

        (h, skips)
    }
}

/// Convert an SDXL DownBlockXL to CubeCL-accelerated version
pub fn convert_down_block_xl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement>(
    block: &crate::unet_sdxl::DownBlockXL<CubeBackend<R, F, I, BT>>,
) -> DownBlockXLCubeCL<R, F, I, BT> {
    DownBlockXLCubeCL {
        res1: block.res1.clone(),
        attn1: block.attn1.as_ref().map(convert_spatial_transformer),
        res2: block.res2.clone(),
        attn2: block.attn2.as_ref().map(convert_spatial_transformer),
        downsample: block.downsample.clone(),
    }
}

/// CubeCL-accelerated SDXL mid block with Flash Attention
#[derive(Debug)]
pub struct MidBlockXLCubeCL<
    R: CubeRuntime,
    F: FloatElement = f32,
    I: IntElement = i32,
    BT: BoolElement = u32,
> {
    res1: crate::blocks::ResBlock<CubeBackend<R, F, I, BT>>,
    attn: SpatialTransformerCubeCL<R, F, I, BT>,
    res2: crate::blocks::ResBlock<CubeBackend<R, F, I, BT>>,
}

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement>
    MidBlockXLCubeCL<R, F, I, BT>
{
    /// Forward pass with flash attention
    pub fn forward(
        &self,
        x: Tensor<CubeBackend<R, F, I, BT>, 4>,
        t_emb: Tensor<CubeBackend<R, F, I, BT>, 2>,
        context: Tensor<CubeBackend<R, F, I, BT>, 3>,
    ) -> Tensor<CubeBackend<R, F, I, BT>, 4> {
        let h = self.res1.forward(x, t_emb.clone());
        let h = self.attn.forward(h, context);
        self.res2.forward(h, t_emb)
    }
}

/// Convert an SDXL MidBlockXL to CubeCL-accelerated version
pub fn convert_mid_block_xl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement>(
    block: &crate::unet_sdxl::MidBlockXL<CubeBackend<R, F, I, BT>>,
) -> MidBlockXLCubeCL<R, F, I, BT> {
    MidBlockXLCubeCL {
        res1: block.res1.clone(),
        attn: convert_spatial_transformer(&block.attn),
        res2: block.res2.clone(),
    }
}

/// CubeCL-accelerated SDXL up block with Flash Attention
#[derive(Debug)]
pub struct UpBlockXLCubeCL<
    R: CubeRuntime,
    F: FloatElement = f32,
    I: IntElement = i32,
    BT: BoolElement = u32,
> {
    res: crate::blocks::ResBlock<CubeBackend<R, F, I, BT>>,
    attn: Option<SpatialTransformerCubeCL<R, F, I, BT>>,
    upsample: Option<crate::blocks::Upsample<CubeBackend<R, F, I, BT>>>,
}

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> UpBlockXLCubeCL<R, F, I, BT> {
    /// Forward pass with flash attention
    pub fn forward(
        &self,
        x: Tensor<CubeBackend<R, F, I, BT>, 4>,
        t_emb: Tensor<CubeBackend<R, F, I, BT>, 2>,
        context: Tensor<CubeBackend<R, F, I, BT>, 3>,
    ) -> Tensor<CubeBackend<R, F, I, BT>, 4> {
        let h = self.res.forward(x, t_emb);
        let h = if let Some(attn) = &self.attn {
            attn.forward(h, context)
        } else {
            h
        };

        if let Some(up) = &self.upsample {
            up.forward(h)
        } else {
            h
        }
    }
}

/// Convert an SDXL UpBlockXL to CubeCL-accelerated version
pub fn convert_up_block_xl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement>(
    block: &crate::unet_sdxl::UpBlockXL<CubeBackend<R, F, I, BT>>,
) -> UpBlockXLCubeCL<R, F, I, BT> {
    UpBlockXLCubeCL {
        res: block.res.clone(),
        attn: block.attn.as_ref().map(convert_spatial_transformer),
        upsample: block.upsample.clone(),
    }
}

/// CubeCL-accelerated SDXL UNet with Flash Attention
///
/// This is a drop-in replacement for [`crate::unet_sdxl::UNetXL`] that uses
/// flash attention in all attention blocks, preventing f16 overflow.
#[derive(Debug)]
pub struct UNetXLCubeCL<
    R: CubeRuntime,
    F: FloatElement = f32,
    I: IntElement = i32,
    BT: BoolElement = u32,
> {
    time_embed_0: Linear<CubeBackend<R, F, I, BT>>,
    time_embed_2: Linear<CubeBackend<R, F, I, BT>>,
    add_embed_0: Linear<CubeBackend<R, F, I, BT>>,
    add_embed_2: Linear<CubeBackend<R, F, I, BT>>,
    time_freqs: Tensor<CubeBackend<R, F, I, BT>, 1>,
    conv_in: Conv2d<CubeBackend<R, F, I, BT>>,
    down_blocks: Vec<DownBlockXLCubeCL<R, F, I, BT>>,
    mid_block: MidBlockXLCubeCL<R, F, I, BT>,
    up_blocks: Vec<UpBlockXLCubeCL<R, F, I, BT>>,
    norm_out: sketchpad_core::groupnorm::GroupNorm<CubeBackend<R, F, I, BT>>,
    conv_out: Conv2d<CubeBackend<R, F, I, BT>>,
}

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> UNetXLCubeCL<R, F, I, BT> {
    /// Forward pass with flash attention in all attention blocks
    ///
    /// Same interface as [`crate::unet_sdxl::UNetXL::forward`].
    pub fn forward(
        &self,
        x: Tensor<CubeBackend<R, F, I, BT>, 4>,
        timesteps: Tensor<CubeBackend<R, F, I, BT>, 1>,
        context: Tensor<CubeBackend<R, F, I, BT>, 3>,
        add_embed: Tensor<CubeBackend<R, F, I, BT>, 2>,
    ) -> Tensor<CubeBackend<R, F, I, BT>, 4> {
        // Time embedding
        let t_emb =
            crate::blocks::timestep_embedding_with_freqs(timesteps, self.time_freqs.clone());
        let t_emb = self.time_embed_0.forward(t_emb);
        let t_emb = sketchpad_core::silu::silu(t_emb);
        let t_emb = self.time_embed_2.forward(t_emb);

        // Additional embedding (pooled text + size conditioning)
        let add_emb = self.add_embed_0.forward(add_embed);
        let add_emb = sketchpad_core::silu::silu(add_emb);
        let add_emb = self.add_embed_2.forward(add_emb);

        // Combine time and additional embeddings
        let t_emb = t_emb + add_emb;

        // Input
        let mut h = self.conv_in.forward(x);

        // Down blocks with skip connections
        let mut skips = vec![h.clone()];
        for block in &self.down_blocks {
            let (out, block_skips) = block.forward(h, t_emb.clone(), context.clone());
            h = out;
            skips.extend(block_skips);
        }

        // Mid block (with flash attention)
        h = self.mid_block.forward(h, t_emb.clone(), context.clone());

        // Up blocks with skip connections
        for block in &self.up_blocks {
            let skip = skips.pop().unwrap();
            h = Tensor::cat(vec![h, skip], 1);
            h = block.forward(h, t_emb.clone(), context.clone());
        }

        // Output
        h = self.norm_out.forward(h);
        h = sketchpad_core::silu::silu(h);
        self.conv_out.forward(h)
    }
}

/// Convert an SDXL UNetXL to CubeCL-accelerated version with flash attention
pub fn convert_unet_xl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement>(
    unet: &crate::unet_sdxl::UNetXL<CubeBackend<R, F, I, BT>>,
) -> UNetXLCubeCL<R, F, I, BT> {
    UNetXLCubeCL {
        time_embed_0: unet.time_embed_0.clone(),
        time_embed_2: unet.time_embed_2.clone(),
        add_embed_0: unet.add_embed_0.clone(),
        add_embed_2: unet.add_embed_2.clone(),
        time_freqs: unet.time_freqs.clone(),
        conv_in: unet.conv_in.clone(),
        down_blocks: unet.down_blocks.iter().map(convert_down_block_xl).collect(),
        mid_block: convert_mid_block_xl(&unet.mid_block),
        up_blocks: unet.up_blocks.iter().map(convert_up_block_xl).collect(),
        norm_out: unet.norm_out.clone(),
        conv_out: unet.conv_out.clone(),
    }
}
