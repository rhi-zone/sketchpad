//! Weight Loading and Serialization
//!
//! This crate handles loading model weights from safetensors and GGUF formats
//! and provides serialization utilities for Burn models.
//!
//! # Weight Loading
//!
//! Load weights from safetensors files:
//!
//! ```ignore
//! use sketchpad_convert::{SafeTensorFile, LoadError};
//!
//! let file = SafeTensorFile::open("model.safetensors")?;
//! let tensor = file.load_tensor::<f32>("model.encoder.weight")?;
//! ```
//!
//! # Specialized Loaders
//!
//! - [`load_lora`] - Load LoRA weights (Kohya, Diffusers formats)
//! - [`load_controlnet_info`] - Detect ControlNet type and config
//! - [`load_embedding`] - Load textual inversion embeddings
//!
//! # Serialization
//!
//! Save and load Burn models with configurable precision:
//!
//! ```ignore
//! use sketchpad_convert::{BinFileRecorder, FullPrecisionSettings};
//!
//! // Save model
//! let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
//! recorder.record(model, "model.bin")?;
//!
//! // Load model
//! let model = recorder.load("model.bin", &device)?;
//! ```
//!
//! # Precision Options
//!
//! - [`FullPrecisionSettings`] - f32 weights
//! - [`HalfPrecisionSettings`] - f16/bf16 weights

pub mod controlnet_loader;
pub mod embedding_loader;
pub mod gguf;
pub mod loader;
pub mod lora_loader;
pub mod mapping;
pub mod sd_loader;
pub mod serialize;

pub use controlnet_loader::{
    ControlNetInfo, ControlNetLoadError, ControlNetType, load_controlnet_info,
};
pub use embedding_loader::{EmbeddingFormat, EmbeddingLoadError, load_embedding};
pub use gguf::{GgufError, GgufFile};
pub use loader::{LoadError, SafeTensorFile};
pub use lora_loader::{LoraFormat, LoraLoadError, load_lora};
pub use sd_loader::{SdLoadError, SdWeightLoader};
pub use serialize::{
    BinBytesRecorder, BinFileRecorder, FullPrecisionSettings, HalfPrecisionSettings, Recorder,
    RecorderError, SerializeError, bytes_recorder, full_precision_recorder,
    half_precision_recorder,
};
