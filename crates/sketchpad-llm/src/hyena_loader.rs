//! Hyena / StripedHyena Weight Loading
//!
//! Loads StripedHyena weights from safetensors files using the HuggingFace
//! naming convention for StripedHyena models.
//!
//! # Status
//!
//! Stub implementation — weight name mapping for StripedHyena 7B has not been
//! finalized against an actual checkpoint. The `init`-based path works for
//! randomly initialized models and testing.
//!
//! TODO: Map HuggingFace StripedHyena weight names to HyenaFilter/HyenaOperator fields.
//!
//! # Expected weight layout (HuggingFace safetensors)
//!
//! ```text
//! backbone.embeddings.word_embeddings.weight
//! backbone.layers.N.pre_norm.weight
//! backbone.layers.N.mixer.in_proj.weight
//! backbone.layers.N.mixer.out_proj.weight
//! backbone.layers.N.mixer.filter.pos_emb               — sinusoidal pos encoding
//! backbone.layers.N.mixer.filter.mlp.0.weight
//! backbone.layers.N.mixer.filter.mlp.0.bias
//! backbone.layers.N.mixer.filter.mlp.2.weight
//! backbone.layers.N.mixer.filter.mlp.2.bias
//! backbone.layers.N.mixer.filter.mlp.4.weight
//! backbone.layers.N.mixer.filter.mlp.4.bias
//! backbone.layers.N.post_norm.weight
//! backbone.layers.N.mlp.l1.weight                      — SwiGLU gate
//! backbone.layers.N.mlp.l2.weight                      — SwiGLU up
//! backbone.layers.N.mlp.l3.weight                      — SwiGLU down
//! backbone.norm_f.weight                               — final norm
//! lm_head.weight
//! ```

use std::path::Path;

use burn::prelude::*;
use thiserror::Error;

use crate::hyena::{Hyena, HyenaConfig, HyenaRuntime};

/// Errors that can occur while loading Hyena weights
#[derive(Error, Debug)]
pub enum HyenaLoadError {
    #[error("Load error: {0}")]
    Load(#[from] sketchpad_convert::loader::LoadError),

    #[error("Missing tensor: {0}")]
    MissingTensor(String),

    #[error("Config mismatch: {0}")]
    ConfigMismatch(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Load Hyena / StripedHyena weights from a safetensors file
///
/// # Arguments
///
/// * `path`   - Path to the `.safetensors` file
/// * `config` - Model configuration
/// * `device` - Device to load tensors onto
///
/// # Note
///
/// This is a stub. Weight loading from actual StripedHyena checkpoints is not
/// yet implemented. Use `config.init(device)` for random initialization.
pub fn load_hyena<B: Backend, P: AsRef<Path>>(
    _path: P,
    config: &HyenaConfig,
    device: &B::Device,
) -> Result<(Hyena<B>, HyenaRuntime<B>), HyenaLoadError> {
    // Stub: initialize randomly. Full weight loading requires mapping
    // HuggingFace StripedHyena checkpoint keys to the module fields.
    Ok(config.init(device))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_error_display() {
        let err = HyenaLoadError::MissingTensor("test_tensor".to_string());
        assert!(err.to_string().contains("test_tensor"));
    }
}
