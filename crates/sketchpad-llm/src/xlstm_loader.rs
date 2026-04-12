//! xLSTM Weight Loading (stub)
//!
//! Weight loading from safetensors / GGUF for xLSTM models.
//! No official pre-trained checkpoints have an established weight layout at
//! time of writing — this stub is the integration point for future loading.

use std::path::Path;

use burn::prelude::*;
use thiserror::Error;

use crate::xlstm::{XLstm, XLstmConfig, XLstmRuntime};

#[derive(Error, Debug)]
pub enum XLstmLoadError {
    #[error("Load error: {0}")]
    Load(String),

    #[error("Missing tensor: {0}")]
    MissingTensor(String),

    #[error("Config mismatch: {0}")]
    ConfigMismatch(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Load xLSTM weights from a safetensors file.
///
/// Currently a stub — returns a freshly initialised model. Replace with
/// actual tensor loading once a canonical weight layout is established.
pub fn load_xlstm<B: Backend, P: AsRef<Path>>(
    _path: P,
    config: &XLstmConfig,
    device: &B::Device,
) -> Result<(XLstm<B>, XLstmRuntime<B>), XLstmLoadError> {
    Ok(config.init(device))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_error_display() {
        let err = XLstmLoadError::MissingTensor("test_tensor".to_string());
        assert!(err.to_string().contains("test_tensor"));
    }
}
