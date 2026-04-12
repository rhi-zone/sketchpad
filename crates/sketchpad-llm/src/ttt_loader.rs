//! TTT Weight Loading from Safetensors
//!
//! Stub loader for TTT models. There are no publicly released TTT checkpoints
//! as of this implementation; this module provides the scaffolding needed when
//! they become available.
//!
//! Expected weight layout (HuggingFace-style):
//! - `model.embed_tokens.weight`         [vocab_size, d_model]
//! - `model.norm.weight`                 [d_model]
//! - `lm_head.weight`                    [vocab_size, d_model]
//! - `model.layers.{i}.ln1.weight`       [d_model]
//! - `model.layers.{i}.ln1.bias`         [d_model]
//! - `model.layers.{i}.ttt.w_q.weight`   [d_model, d_model]
//! - `model.layers.{i}.ttt.w_k.weight`   [d_model, d_model]
//! - `model.layers.{i}.ttt.w_v.weight`   [d_model, d_model]
//! - `model.layers.{i}.ttt.w_o.weight`   [d_model, d_model]
//! - `model.layers.{i}.ln2.weight`       [d_model]
//! - `model.layers.{i}.ln2.bias`         [d_model]
//! - `model.layers.{i}.ffn.gate_proj.weight`  [intermediate_size, d_model]
//! - `model.layers.{i}.ffn.up_proj.weight`    [intermediate_size, d_model]
//! - `model.layers.{i}.ffn.down_proj.weight`  [d_model, intermediate_size]

use std::path::Path;

use thiserror::Error;

use crate::ttt::{Ttt, TttConfig, TttRuntime};

#[derive(Error, Debug)]
pub enum TttLoadError {
    #[error("Load error: {0}")]
    Load(#[from] sketchpad_convert::loader::LoadError),

    #[error("Missing tensor: {0}")]
    MissingTensor(String),

    #[error("Config mismatch: {0}")]
    ConfigMismatch(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Load a TTT model from safetensors weights.
///
/// This is a stub — implement when checkpoints are available.
pub fn load_ttt<B: burn::prelude::Backend, P: AsRef<Path>>(
    _path: P,
    config: &TttConfig,
    device: &B::Device,
) -> Result<(Ttt<B>, TttRuntime<B>), TttLoadError> {
    // Initialise a randomly-weighted model.
    // Replace with actual weight loading once checkpoints exist.
    let (model, runtime) = config.init::<B>(device);
    Ok((model, runtime))
}
