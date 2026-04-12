//! LLaDA Weight Loading (stub)
//!
//! Stub loader for LLaDA weights. Real loading would follow the same pattern as
//! other loaders in this crate (safetensors, GGUF arch string "llada").
//!
//! GGUF arch string: `"llada"`

use std::path::Path;

use burn::prelude::*;
use thiserror::Error;

use crate::llada::{LLaDa, LLaDaConfig, LLaDaRuntime};

#[derive(Error, Debug)]
pub enum LLaDaLoadError {
    #[error("Load error: {0}")]
    Load(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Load LLaDA weights from a safetensors file (stub — not yet implemented)
///
/// # Arguments
///
/// * `path` - Path to the safetensors file
/// * `config` - Model configuration
/// * `device` - Device to load tensors onto
pub fn load_llada<B: Backend, P: AsRef<Path>>(
    _path: P,
    config: &LLaDaConfig,
    device: &B::Device,
) -> Result<(LLaDa<B>, LLaDaRuntime<B>), LLaDaLoadError> {
    // TODO: implement actual weight loading from safetensors
    // Weight key naming convention (HuggingFace LLaDA):
    //   model.embed_tokens.weight
    //   model.layers.{i}.input_layernorm.weight
    //   model.layers.{i}.self_attn.q_proj.weight
    //   model.layers.{i}.self_attn.k_proj.weight
    //   model.layers.{i}.self_attn.v_proj.weight
    //   model.layers.{i}.self_attn.o_proj.weight
    //   model.layers.{i}.post_attention_layernorm.weight
    //   model.layers.{i}.mlp.gate_proj.weight
    //   model.layers.{i}.mlp.up_proj.weight
    //   model.layers.{i}.mlp.down_proj.weight
    //   model.norm.weight
    //   lm_head.weight
    let (model, runtime) = config.init(device);
    Ok((model, runtime))
}
