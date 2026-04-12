//! RetNet Weight Loading (stub)
//!
//! Stub for loading RetNet model weights from safetensors files.
//! No publicly available RetNet checkpoints exist at the time of writing;
//! this module provides the error type and a placeholder entry-point so that
//! the model can be wired into the inference pipeline now and the actual
//! tensor-name mapping filled in when weights become available.

use std::path::Path;

use burn::prelude::*;
use thiserror::Error;

use crate::retnet::{RetNet, RetNetConfig, RetNetRuntime};

#[derive(Error, Debug)]
pub enum RetNetLoadError {
    #[error("Load error: {0}")]
    Load(#[from] sketchpad_convert::loader::LoadError),

    #[error("Missing tensor: {0}")]
    MissingTensor(String),

    #[error("Config mismatch: {0}")]
    ConfigMismatch(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Load RetNet weights from a safetensors file.
///
/// **Stub** — initialises a fresh model from the provided config rather than
/// loading actual weights. Replace with real tensor-name mapping once
/// checkpoint files are available.
pub fn load_retnet<B: Backend, P: AsRef<Path>>(
    _path: P,
    config: &RetNetConfig,
    device: &B::Device,
) -> Result<(RetNet<B>, RetNetRuntime<B>), RetNetLoadError> {
    // Placeholder: initialise from config with random weights.
    // A real implementation would open the safetensors file and load each
    // projection matrix by its tensor name.
    let (model, runtime) = config.init::<B>(device);
    Ok((model, runtime))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    #[test]
    fn test_load_error_display() {
        let err = RetNetLoadError::MissingTensor("model.layers.0.retention.q_proj.weight".into());
        assert!(err.to_string().contains("model.layers.0"));
    }

    #[test]
    fn test_stub_load() {
        let device = Default::default();
        let config = crate::retnet::RetNetConfig::tiny();
        let result = load_retnet::<NdArray<f32>, _>("dummy_path", &config, &device);
        assert!(result.is_ok());
    }
}
