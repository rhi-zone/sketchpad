//! UNet Diffusion Backbone
//!
//! This crate provides UNet implementations for Stable Diffusion models,
//! including the core denoising network and conditioning adapters.
//!
//! # Models
//!
//! - [`UNet`] - SD 1.x/2.x UNet (~860M params)
//! - [`UNetXL`] - SDXL UNet (~2.6B params)
//! - [`StageC`], [`StageB`] - Stable Cascade stages
//!
//! # Conditioning
//!
//! - [`ControlNet`] - Spatial conditioning (pose, depth, edges)
//! - [`IpAdapter`] - Image prompt conditioning
//! - [`T2IAdapter`] - Lightweight image conditioning
//!
//! # Building Blocks
//!
//! - [`ResBlock`] - Residual convolution block
//! - [`SpatialTransformer`] - Cross-attention to text
//! - [`CrossAttention`] - Text-to-image attention
//! - [`Downsample`], [`Upsample`] - Resolution scaling
//!
//! # Example
//!
//! ```ignore
//! use sketchpad_unet::{UNet, UNetConfig};
//!
//! let config = UNetConfig::sd1x();
//! let unet = config.init::<Backend>(&device);
//!
//! // Forward pass with text conditioning
//! let noise_pred = unet.forward(
//!     latents,
//!     timestep,
//!     text_embeddings,
//!     None, // ControlNet
//! );
//! ```

pub mod blocks;
pub mod conditioning;
pub mod controlnet;
pub mod ip_adapter;
pub mod stable_cascade;
pub mod t2i_adapter;
pub mod unet_sd;
pub mod unet_sdxl;

#[cfg(feature = "cubecl")]
pub mod cubecl;

pub use blocks::{
    CrossAttention, Downsample, FeedForward, ResBlock, SpatialTransformer, TransformerBlock,
    Upsample, timestep_embedding, timestep_embedding_with_freqs, timestep_freqs,
};
pub use controlnet::{ControlNet, ControlNetConfig, ControlNetOutput, ControlNetPreprocessor};
pub use ip_adapter::{ImageProjection, IpAdapter, IpAdapterConfig, combine_embeddings};
pub use stable_cascade::{StageB, StageBConfig, StageBOutput, StageC, StageCConfig, StageCOutput};
pub use t2i_adapter::{T2IAdapter, T2IAdapterConfig, T2IAdapterOutput};
pub use unet_sd::{DownBlock, MidBlock, UNet, UNetConfig, UpBlock};
pub use unet_sdxl::{DownBlockXL, MidBlockXL, UNetXL, UNetXLConfig, UpBlockXL};
