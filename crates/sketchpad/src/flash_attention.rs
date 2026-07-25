//! Flash Attention for UNet Inference
//!
//! Provides flash attention via cubek-attention for CubeBackends.
//!
//! Flash attention:
//! 1. Tiles computation to avoid materializing the full attention matrix
//! 2. Uses f32 accumulation internally
//! 3. Computes softmax incrementally (online softmax algorithm)
//! 4. Uses O(n) memory instead of O(n²)
//!
//! # Known Issues
//!
//! **f16/bf16 is broken on CUDA** - cubek-attention 0.1.0-pre.1 has an alignment bug.
//! Use f32 precision until https://github.com/tracel-ai/cubek/pull/55 is merged.
//!
//! # Usage
//!
//! ```ignore
//! use sketchpad::flash_attention::convert_unet_flash;
//!
//! // Load standard UNet
//! let unet = load_sd1x_unet(&weights, &device)?;
//!
//! // Convert to flash attention version
//! let unet_flash = convert_unet_flash(&unet);
//!
//! // Use in pipeline - same interface
//! let noise = unet_flash.forward(latents, timesteps, context);
//! ```

// Re-export flash attention types from unet cubecl module
#[cfg(feature = "cubecl")]
pub use sketchpad_unet::cubecl::{
    // Block types (for advanced use)
    CrossAttentionCubeCL,
    DownBlockCubeCL,
    MidBlockCubeCL,
    SpatialTransformerCubeCL,
    TransformerBlockCubeCL,
    // Main UNet type
    UNetCubeCL,
    UpBlockCubeCL,
    // Conversion helpers
    convert_crossattention,
    convert_down_block,
    convert_mid_block,
    convert_spatial_transformer,
    convert_transformer_block,
    // Conversion function
    convert_unet,
    convert_up_block,
};

/// Convenience alias for the flash attention UNet conversion function
#[cfg(feature = "cubecl")]
pub use sketchpad_unet::cubecl::convert_unet as convert_unet_flash;

/// Check if flash attention is available
///
/// Returns true if the `cubecl` feature is enabled, meaning flash attention
/// can be used for f16 inference without NaN issues.
#[inline]
pub const fn is_available() -> bool {
    cfg!(feature = "cubecl")
}
