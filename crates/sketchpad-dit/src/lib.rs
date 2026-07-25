//! DiT-based Diffusion Models
//!
//! This crate provides implementations of DiT-based (Diffusion Transformer)
//! image and video generation models.
//!
//! # Supported Models
//!
//! ## Image Generation
//!
//! - **Flux**: Black Forest Labs' flow-matching DiT model
//!   - Flux.1-dev (guidance-distilled)
//!   - Flux.1-schnell (few-step)
//!
//! - **Stable Diffusion 3**: Stability AI's MMDiT model
//!   - SD3-Medium (2B params)
//!   - SD3-Large (8B params)
//!   - SD3.5-Large, SD3.5-Turbo
//!
//! ## Video Generation
//!
//! - **CogVideoX**: Open-source video DiT model
//!   - CogVideoX-2B
//!   - CogVideoX-5B
//!
//! # Architecture
//!
//! DiT models differ from UNet-based diffusion models:
//! - Use transformer blocks instead of convolutional U-Net
//! - Adaptive layer norm (AdaLN) conditioned on timestep
//! - Bidirectional attention (no causal mask)
//! - Flow matching objective (vs noise prediction)

pub mod auraflow;
pub mod cogvideox;
pub mod flux;
pub mod flux_loader;
pub mod hunyuan;
pub mod ltx;
pub mod mochi;
pub mod pixart;
pub mod qwenimage;
pub mod sana;
pub mod sd3;
pub mod wan;
pub mod zimage;

pub use auraflow::{AuraFlow, AuraFlowConfig, AuraFlowOutput, AuraFlowRuntime};
pub use cogvideox::{CogVideoX, CogVideoXConfig, CogVideoXOutput, CogVideoXRuntime};
pub use flux::{Flux, FluxConfig, FluxOutput, FluxRuntime};
pub use flux_loader::{FluxLoadError, load_flux};
pub use hunyuan::{HunyuanDiT, HunyuanDiTConfig, HunyuanDiTOutput, HunyuanDiTRuntime};
pub use ltx::{LtxVideo, LtxVideoConfig, LtxVideoOutput, LtxVideoRuntime};
pub use mochi::{Mochi, MochiConfig, MochiOutput, MochiRuntime};
pub use pixart::{PixArt, PixArtConfig, PixArtOutput, PixArtRuntime};
pub use qwenimage::{QwenImage, QwenImageConfig, QwenImageOutput, QwenImageRuntime};
pub use sana::{Sana, SanaConfig, SanaOutput, SanaRuntime};
pub use sd3::{Sd3, Sd3Config, Sd3Output, Sd3Runtime};
pub use wan::{Wan, WanConfig, WanOutput, WanRuntime};
pub use zimage::{ZImage, ZImageConfig, ZImageOutput, ZImageRuntime};
