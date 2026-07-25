//! Flash Attention kernel
//!
//! GPU-accelerated flash attention via cubek-attention.
//! Computes attention in tiles without materializing the full attention matrix,
//! reducing memory from O(n²) to O(n).
//!
//! # Usage
//!
//! ```ignore
//! use sketchpad_cubecl::{flash_attention, FlashAttentionOptions};
//! use burn_cubecl::tensor::CubeTensor;
//!
//! // Input tensors: [batch, heads, seq_len, head_dim]
//! // Non-causal (bidirectional) - for diffusion models, encoders
//! let output = flash_attention(query, key, value, FlashAttentionOptions::default())?;
//!
//! // Causal (autoregressive) - for LLMs
//! let output = flash_attention(query, key, value, FlashAttentionOptions::causal())?;
//! ```
//!
//! # Known Issues
//!
//! **f16/bf16 is broken on CUDA** - cubek-attention 0.1.0-pre.1 has an alignment bug
//! where half-precision types fail with `assertion failed: unit_tile.layout.num_cols % line_size == 0`.
//! Use f32 precision until this is fixed upstream.
//! See: https://github.com/tracel-ai/cubek/pull/55

use burn::tensor::{DType, Shape};
use burn_cubecl::{CubeRuntime, ops::numeric::empty_device_dtype, tensor::CubeTensor};
use cubek::attention::{
    definition::{AccumulatorPrecision, AttentionGlobalTypes, AttentionOptions},
    launch::{BlueprintStrategy, Strategy},
};

// Re-export the error type for consumers
pub use cubek::attention::definition::AttentionSetupError;

/// Options for flash attention
#[derive(Debug, Clone, Default)]
pub struct FlashAttentionOptions {
    /// Output dtype. If None, uses the query dtype.
    pub out_dtype: Option<DType>,
    /// Whether to use causal (autoregressive) masking.
    /// - `false` (default): Bidirectional attention - for diffusion models, encoders
    /// - `true`: Causal attention - for autoregressive LLMs
    pub causal: bool,
}

impl FlashAttentionOptions {
    /// Create options for causal (autoregressive) attention
    pub fn causal() -> Self {
        Self {
            out_dtype: None,
            causal: true,
        }
    }

    /// Create options with a specific output dtype
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.out_dtype = Some(dtype);
        self
    }

    /// Set causal mode
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }
}

/// Flash attention kernel
///
/// Computes scaled dot-product attention:
/// ```text
/// output = softmax(Q @ K^T / sqrt(d) [+ causal_mask]) @ V
/// ```
///
/// # Arguments
///
/// * `query` - Query tensor [batch, heads, seq_q, head_dim]
/// * `key` - Key tensor [batch, heads, seq_k, head_dim]
/// * `value` - Value tensor [batch, heads, seq_k, val_dim]
/// * `options` - Flash attention options (causal mode, output dtype)
///
/// # Returns
///
/// Output tensor [batch, heads, seq_q, val_dim]
///
/// # Errors
///
/// Returns `AttentionSetupError` if the kernel cannot be launched.
///
/// # Panics
///
/// **Will panic on CUDA with f16/bf16 inputs** due to a bug in cubek-attention 0.1.0-pre.1.
/// Use f32 precision until https://github.com/tracel-ai/cubek/pull/55 is merged.
pub fn flash_attention<R: CubeRuntime>(
    query: CubeTensor<R>,
    key: CubeTensor<R>,
    value: CubeTensor<R>,
    options: FlashAttentionOptions,
) -> Result<CubeTensor<R>, AttentionSetupError> {
    flash_attention_impl(query, key, value, None, options)
}

/// Flash attention with explicit attention mask
///
/// # Arguments
///
/// * `query` - Query tensor [batch, heads, seq_q, head_dim]
/// * `key` - Key tensor [batch, heads, seq_k, head_dim]
/// * `value` - Value tensor [batch, heads, seq_k, val_dim]
/// * `mask` - Attention mask tensor (additive, where masked positions have large negative values)
/// * `options` - Flash attention options
///
/// # Returns
///
/// Output tensor [batch, heads, seq_q, val_dim]
pub fn flash_attention_masked<R: CubeRuntime>(
    query: CubeTensor<R>,
    key: CubeTensor<R>,
    value: CubeTensor<R>,
    mask: CubeTensor<R>,
    options: FlashAttentionOptions,
) -> Result<CubeTensor<R>, AttentionSetupError> {
    flash_attention_impl(query, key, value, Some(mask), options)
}

/// Internal implementation that calls cubek directly
fn flash_attention_impl<R: CubeRuntime>(
    query: CubeTensor<R>,
    key: CubeTensor<R>,
    value: CubeTensor<R>,
    mask: Option<CubeTensor<R>>,
    options: FlashAttentionOptions,
) -> Result<CubeTensor<R>, AttentionSetupError> {
    let client = &query.client;
    let device = &query.device;

    let num_batches = query.shape.dims[0];
    let num_heads = query.shape.dims[1];
    let seq_q = query.shape.dims[2];
    let val_dim = value.shape.dims[3];
    let out_shape = Shape::new([num_batches, num_heads, seq_q, val_dim]);

    let out_dtype = options.out_dtype.unwrap_or(query.dtype);
    let out = empty_device_dtype::<R>(client.clone(), device.clone(), out_shape, out_dtype);

    let dtypes = AttentionGlobalTypes {
        query: query.dtype.into(),
        key: key.dtype.into(),
        value: value.dtype.into(),
        mask: mask.as_ref().map(|m| m.dtype).unwrap_or(DType::U8).into(),
        out: out.dtype.into(),
    };

    // Use Inferred strategy - let cubek compute optimal tile sizes
    let strategy = Strategy::Unit(BlueprintStrategy::Inferred(()));

    cubek::attention::launch::launch_ref::<R>(
        strategy,
        client,
        &query.as_handle_ref(),
        &key.as_handle_ref(),
        &value.as_handle_ref(),
        &mask.as_ref().map(|m| m.as_handle_ref()),
        &out.as_handle_ref(),
        &dtypes,
        AttentionOptions {
            causal: options.causal,
            accumulator_precision: AccumulatorPrecision::Strict(cubecl::ir::StorageType::Scalar(
                cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F32),
            )),
        },
    )?;

    Ok(out)
}
