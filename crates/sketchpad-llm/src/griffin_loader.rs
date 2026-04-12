//! Griffin / Hawk (RecurrentGemma) Weight Loading
//!
//! Loads RecurrentGemma weights from safetensors files using the HuggingFace
//! naming convention for Griffin/Hawk models.
//!
//! # GGUF
//!
//! RecurrentGemma models are also distributed as GGUF with `llm.architecture = "recurrentgemma"`.
//! The GGUF metadata keys follow the same naming below.
//!
//! # Weight naming (HuggingFace safetensors)
//!
//! ```text
//! model.embed_tokens.weight                         — embedding
//! model.layers.N.temporal_pre_norm.weight           — pre-norm (recurrent layers)
//! model.layers.N.channel_pre_norm.weight            — pre-norm (attention layers, alias)
//! model.layers.N.recurrent_layer.input_gate.weight
//! model.layers.N.recurrent_layer.input_gate.bias
//! model.layers.N.recurrent_layer.recurrent_gate.weight
//! model.layers.N.recurrent_layer.recurrent_gate.bias
//! model.layers.N.recurrent_layer.output_gate.weight
//! model.layers.N.recurrent_layer.output_gate.bias
//! model.layers.N.recurrent_layer.a_param            — a_log parameter
//! model.layers.N.attn.q_proj.weight                 — attention layers
//! model.layers.N.attn.k_proj.weight
//! model.layers.N.attn.v_proj.weight
//! model.layers.N.attn.o_proj.weight
//! model.layers.N.pre_feedforward_layernorm.weight   — post-mixer norm
//! model.layers.N.mlp.gate_proj.weight
//! model.layers.N.mlp.up_proj.weight
//! model.layers.N.mlp.down_proj.weight
//! model.final_norm.weight                           — final norm
//! lm_head.weight
//! ```

use std::path::Path;

use burn::module::Param;
use burn::nn::{EmbeddingConfig, LinearConfig};
use burn::prelude::*;
use sketchpad_convert::loader::SafeTensorFile;
use thiserror::Error;

use crate::griffin::{
    Griffin, GriffinBlock, GriffinConfig, GriffinGeGluFfn, GriffinLocalAttn, GriffinRgLru,
    GriffinRuntime,
};
use sketchpad_core::rmsnorm::RmsNorm;

/// Errors that can occur while loading Griffin weights
#[derive(Error, Debug)]
pub enum GriffinLoadError {
    #[error("Load error: {0}")]
    Load(#[from] sketchpad_convert::loader::LoadError),

    #[error("Missing tensor: {0}")]
    MissingTensor(String),

    #[error("Config mismatch: {0}")]
    ConfigMismatch(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Load Griffin / Hawk weights from a safetensors file
///
/// # Arguments
///
/// * `path`   - Path to the `.safetensors` file
/// * `config` - Model configuration (must match the weights)
/// * `device` - Device to load tensors onto
pub fn load_griffin<B: Backend, P: AsRef<Path>>(
    path: P,
    config: &GriffinConfig,
    device: &B::Device,
) -> Result<(Griffin<B>, GriffinRuntime<B>), GriffinLoadError> {
    let file = SafeTensorFile::open(path)?;

    // Embeddings
    let embed_weight: Tensor<B, 2> = file.load_f32("model.embed_tokens.weight", device)?;
    let [v, h] = embed_weight.dims();
    if v != config.vocab_size || h != config.d_model {
        return Err(GriffinLoadError::ConfigMismatch(format!(
            "embed_tokens: expected [{}, {}], got [{}, {}]",
            config.vocab_size, config.d_model, v, h
        )));
    }
    let mut embed_tokens = EmbeddingConfig::new(config.vocab_size, config.d_model).init(device);
    embed_tokens.weight = Param::from_tensor(embed_weight);

    // Blocks
    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let block = load_griffin_block(&file, i, config, device)?;
        layers.push(block);
    }

    // Final norm
    let norm = load_rmsnorm(
        &file,
        "model.final_norm",
        config.d_model,
        config.norm_eps,
        device,
    )?;

    // LM head
    let lm_head_weight: Tensor<B, 2> = file.load_f32("lm_head.weight", device)?;
    let mut lm_head = LinearConfig::new(config.d_model, config.vocab_size)
        .with_bias(false)
        .init(device);
    lm_head.weight = Param::from_tensor(lm_head_weight);

    let model = Griffin {
        embed_tokens,
        layers,
        norm,
        lm_head,
    };

    let runtime = GriffinRuntime::new(config.clone());

    Ok((model, runtime))
}

fn load_griffin_block<B: Backend>(
    file: &SafeTensorFile,
    layer_idx: usize,
    config: &GriffinConfig,
    device: &B::Device,
) -> Result<GriffinBlock<B>, GriffinLoadError> {
    let is_attn = config.is_attention_layer(layer_idx);
    let prefix = format!("model.layers.{}", layer_idx);

    // Pre-norm: recurrent layers use temporal_pre_norm, attention layers use channel_pre_norm
    let pre_norm_key = if is_attn {
        format!("{}.channel_pre_norm", prefix)
    } else {
        format!("{}.temporal_pre_norm", prefix)
    };
    let pre_norm = load_rmsnorm(file, &pre_norm_key, config.d_model, config.norm_eps, device)?;

    let rg_lru = if is_attn {
        None
    } else {
        Some(load_rg_lru(file, &prefix, config, device)?)
    };

    let attention = if is_attn {
        Some(load_local_attn(file, &prefix, config, device)?)
    } else {
        None
    };

    let post_norm = load_rmsnorm(
        file,
        &format!("{}.pre_feedforward_layernorm", prefix),
        config.d_model,
        config.norm_eps,
        device,
    )?;

    let ffn = load_geglu_ffn(file, &prefix, config, device)?;

    Ok(GriffinBlock {
        pre_norm,
        rg_lru,
        attention,
        post_norm,
        ffn,
        is_attention: is_attn,
    })
}

fn load_rg_lru<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &GriffinConfig,
    device: &B::Device,
) -> Result<GriffinRgLru<B>, GriffinLoadError> {
    let d = config.d_model;
    let rp = format!("{}.recurrent_layer", prefix);

    let input_gate_proj = load_linear(
        file,
        &format!("{}.input_gate.weight", rp),
        Some(&format!("{}.input_gate.bias", rp)),
        d,
        d,
        device,
    )?;

    let recurrence_gate_proj = load_linear(
        file,
        &format!("{}.recurrent_gate.weight", rp),
        Some(&format!("{}.recurrent_gate.bias", rp)),
        d,
        d,
        device,
    )?;

    let output_gate_proj = load_linear(
        file,
        &format!("{}.output_gate.weight", rp),
        Some(&format!("{}.output_gate.bias", rp)),
        d,
        d,
        device,
    )?;

    let a_log: Tensor<B, 1> = file.load_f32(&format!("{}.a_param", rp), device)?;

    Ok(GriffinRgLru {
        input_gate_proj,
        recurrence_gate_proj,
        output_gate_proj,
        a_log: Param::from_tensor(a_log),
    })
}

fn load_local_attn<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &GriffinConfig,
    device: &B::Device,
) -> Result<GriffinLocalAttn<B>, GriffinLoadError> {
    let hidden = config.d_model;
    let kv_dim = config.num_heads * config.head_dim;
    let ap = format!("{}.attn", prefix);

    let q_proj = load_linear(
        file,
        &format!("{}.q_proj.weight", ap),
        None,
        hidden,
        kv_dim,
        device,
    )?;
    let k_proj = load_linear(
        file,
        &format!("{}.k_proj.weight", ap),
        None,
        hidden,
        kv_dim,
        device,
    )?;
    let v_proj = load_linear(
        file,
        &format!("{}.v_proj.weight", ap),
        None,
        hidden,
        kv_dim,
        device,
    )?;
    let o_proj = load_linear(
        file,
        &format!("{}.o_proj.weight", ap),
        None,
        kv_dim,
        hidden,
        device,
    )?;

    Ok(GriffinLocalAttn {
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        num_heads: config.num_heads,
        head_dim: config.head_dim,
        window_size: config.window_size,
    })
}

fn load_geglu_ffn<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &GriffinConfig,
    device: &B::Device,
) -> Result<GriffinGeGluFfn<B>, GriffinLoadError> {
    let d = config.d_model;
    let inter = config.intermediate_size;
    let mp = format!("{}.mlp", prefix);

    let gate_proj = load_linear(
        file,
        &format!("{}.gate_proj.weight", mp),
        None,
        d,
        inter,
        device,
    )?;
    let up_proj = load_linear(
        file,
        &format!("{}.up_proj.weight", mp),
        None,
        d,
        inter,
        device,
    )?;
    let down_proj = load_linear(
        file,
        &format!("{}.down_proj.weight", mp),
        None,
        inter,
        d,
        device,
    )?;

    Ok(GriffinGeGluFfn {
        gate_proj,
        up_proj,
        down_proj,
    })
}

fn load_rmsnorm<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    hidden_size: usize,
    eps: f64,
    device: &B::Device,
) -> Result<RmsNorm<B>, GriffinLoadError> {
    let weight: Tensor<B, 1> = file.load_f32(&format!("{}.weight", prefix), device)?;
    let [size] = weight.dims();
    if size != hidden_size {
        return Err(GriffinLoadError::ConfigMismatch(format!(
            "{}: expected [{}], got [{}]",
            prefix, hidden_size, size
        )));
    }
    Ok(RmsNorm::from_weight(weight, eps))
}

fn load_linear<B: Backend>(
    file: &SafeTensorFile,
    weight_name: &str,
    bias_name: Option<&str>,
    in_features: usize,
    out_features: usize,
    device: &B::Device,
) -> Result<burn::nn::Linear<B>, GriffinLoadError> {
    let weight: Tensor<B, 2> = file.load_f32(weight_name, device)?;
    let [out_f, in_f] = weight.dims();
    if out_f != out_features || in_f != in_features {
        return Err(GriffinLoadError::ConfigMismatch(format!(
            "{}: expected [{}, {}], got [{}, {}]",
            weight_name, out_features, in_features, out_f, in_f
        )));
    }

    let has_bias = bias_name.is_some();
    let mut linear = LinearConfig::new(in_features, out_features)
        .with_bias(has_bias)
        .init(device);
    linear.weight = Param::from_tensor(weight);

    if let Some(bn) = bias_name {
        if file.contains(bn) {
            let bias: Tensor<B, 1> = file.load_f32(bn, device)?;
            linear.bias = Some(Param::from_tensor(bias));
        }
    }

    Ok(linear)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_error_display() {
        let err = GriffinLoadError::MissingTensor("test_tensor".to_string());
        assert!(err.to_string().contains("test_tensor"));
    }
}
