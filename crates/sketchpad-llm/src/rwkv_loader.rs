//! RWKV-7 Weight Loading from Safetensors
//!
//! Loads RWKV-7 "Goose" model weights from HuggingFace-format safetensors files.
//! Compatible with models like RWKV/v7-Goose-*-HF.

use std::path::Path;

use burn::module::Param;
use burn::nn::{EmbeddingConfig, LayerNormConfig, LinearConfig};
use burn::prelude::*;
use sketchpad_convert::loader::SafeTensorFile;
use thiserror::Error;

use crate::rwkv::{Rwkv, RwkvBlock, RwkvChannelMix, RwkvConfig, RwkvRuntime, RwkvTimeMix};

#[derive(Error, Debug)]
pub enum RwkvLoadError {
    #[error("Load error: {0}")]
    Load(#[from] sketchpad_convert::loader::LoadError),

    #[error("Missing tensor: {0}")]
    MissingTensor(String),

    #[error("Config mismatch: {0}")]
    ConfigMismatch(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Loads RWKV-7 weights from a safetensors file
///
/// # Arguments
///
/// * `path` - Path to the safetensors file
/// * `config` - Model configuration (must match the weights)
/// * `device` - Device to load tensors onto
pub fn load_rwkv<B: Backend, P: AsRef<Path>>(
    path: P,
    config: &RwkvConfig,
    device: &B::Device,
) -> Result<(Rwkv<B>, RwkvRuntime<B>), RwkvLoadError> {
    let file = SafeTensorFile::open(path)?;

    // Load embeddings
    let embed_tokens = load_embedding(
        &file,
        "rwkv.embeddings.weight",
        config.vocab_size,
        config.hidden_size,
        device,
    )?;

    // Load transformer layers
    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let layer = load_rwkv_block(&file, i, config, device)?;
        layers.push(layer);
    }

    // Load final layer norm
    let ln_out = load_layer_norm(
        &file,
        "rwkv.ln_out",
        config.hidden_size,
        config.layer_norm_eps,
        device,
    )?;

    // Load LM head (may be tied to embeddings)
    let lm_head = if file.contains("head.weight") {
        load_linear(
            &file,
            "head.weight",
            None,
            config.hidden_size,
            config.vocab_size,
            device,
        )?
    } else {
        // Tied weights - reuse embedding
        let weight: Tensor<B, 2> = file.load_f32("rwkv.embeddings.weight", device)?;
        let mut linear = LinearConfig::new(config.hidden_size, config.vocab_size)
            .with_bias(false)
            .init(device);
        linear.weight = Param::from_tensor(weight);
        linear
    };

    let model = Rwkv {
        embed_tokens,
        layers,
        ln_out,
        lm_head,
    };

    let runtime = RwkvRuntime {
        config: config.clone(),
        _marker: std::marker::PhantomData,
    };

    Ok((model, runtime))
}

fn load_embedding<B: Backend>(
    file: &SafeTensorFile,
    name: &str,
    vocab_size: usize,
    hidden_size: usize,
    device: &B::Device,
) -> Result<burn::nn::Embedding<B>, RwkvLoadError> {
    let weight: Tensor<B, 2> = file.load_f32(name, device)?;

    let [v, h] = weight.dims();
    if v != vocab_size || h != hidden_size {
        return Err(RwkvLoadError::ConfigMismatch(format!(
            "{}: expected [{}, {}], got [{}, {}]",
            name, vocab_size, hidden_size, v, h
        )));
    }

    let mut embedding = EmbeddingConfig::new(vocab_size, hidden_size).init(device);
    embedding.weight = Param::from_tensor(weight);

    Ok(embedding)
}

fn load_linear<B: Backend>(
    file: &SafeTensorFile,
    weight_name: &str,
    bias_name: Option<&str>,
    in_features: usize,
    out_features: usize,
    device: &B::Device,
) -> Result<burn::nn::Linear<B>, RwkvLoadError> {
    let weight: Tensor<B, 2> = file.load_f32(weight_name, device)?;

    let [out_f, in_f] = weight.dims();
    if out_f != out_features || in_f != in_features {
        return Err(RwkvLoadError::ConfigMismatch(format!(
            "{}: expected [{}, {}], got [{}, {}]",
            weight_name, out_features, in_features, out_f, in_f
        )));
    }

    let has_bias = bias_name.is_some();
    let mut linear = LinearConfig::new(in_features, out_features)
        .with_bias(has_bias)
        .init(device);

    linear.weight = Param::from_tensor(weight);

    if let Some(bias_n) = bias_name {
        if file.contains(bias_n) {
            let bias: Tensor<B, 1> = file.load_f32(bias_n, device)?;
            linear.bias = Some(Param::from_tensor(bias));
        }
    }

    Ok(linear)
}

fn load_layer_norm<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    hidden_size: usize,
    eps: f64,
    device: &B::Device,
) -> Result<burn::nn::LayerNorm<B>, RwkvLoadError> {
    let weight: Tensor<B, 1> = file.load_f32(&format!("{}.weight", prefix), device)?;
    let bias: Tensor<B, 1> = file.load_f32(&format!("{}.bias", prefix), device)?;

    let [size] = weight.dims();
    if size != hidden_size {
        return Err(RwkvLoadError::ConfigMismatch(format!(
            "{}: expected [{}], got [{}]",
            prefix, hidden_size, size
        )));
    }

    let mut ln = LayerNormConfig::new(hidden_size)
        .with_epsilon(eps)
        .init(device);
    ln.gamma = Param::from_tensor(weight);
    ln.beta = Some(Param::from_tensor(bias));

    Ok(ln)
}

fn load_rwkv_block<B: Backend>(
    file: &SafeTensorFile,
    layer_idx: usize,
    config: &RwkvConfig,
    device: &B::Device,
) -> Result<RwkvBlock<B>, RwkvLoadError> {
    let prefix = format!("rwkv.blocks.{}", layer_idx);

    let time_mix = load_time_mix(file, &prefix, config, layer_idx, device)?;
    let channel_mix = load_channel_mix(file, &prefix, config, device)?;

    Ok(RwkvBlock {
        time_mix,
        channel_mix,
    })
}

fn load_time_mix<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &RwkvConfig,
    _layer_id: usize,
    device: &B::Device,
) -> Result<RwkvTimeMix<B>, RwkvLoadError> {
    let hidden = config.hidden_size;
    let num_heads = config.num_heads;
    let head_dim = config.head_dim;
    let inner_dim = num_heads * head_dim;

    let ln = load_layer_norm(
        file,
        &format!("{}.ln1", prefix),
        hidden,
        config.layer_norm_eps,
        device,
    )?;

    // Load time mixing parameters
    let time_maa_x: Tensor<B, 1> = file.load_f32(&format!("{}.att.time_maa_x", prefix), device)?;
    let time_maa_r: Tensor<B, 1> = file.load_f32(&format!("{}.att.time_maa_r", prefix), device)?;
    let time_maa_w: Tensor<B, 1> = file.load_f32(&format!("{}.att.time_maa_w", prefix), device)?;
    let time_maa_k: Tensor<B, 1> = file.load_f32(&format!("{}.att.time_maa_k", prefix), device)?;
    let time_maa_v: Tensor<B, 1> = file.load_f32(&format!("{}.att.time_maa_v", prefix), device)?;
    let time_maa_a: Tensor<B, 1> = file.load_f32(&format!("{}.att.time_maa_a", prefix), device)?;
    let time_maa_g: Tensor<B, 1> = file.load_f32(&format!("{}.att.time_maa_g", prefix), device)?;

    let time_decay: Tensor<B, 1> = file.load_f32(&format!("{}.att.time_decay", prefix), device)?;
    let time_faaaa: Tensor<B, 2> = file.load_f32(&format!("{}.att.time_faaaa", prefix), device)?;

    // Load projections
    let receptance = load_linear(
        file,
        &format!("{}.att.receptance.weight", prefix),
        None,
        hidden,
        inner_dim,
        device,
    )?;
    let key = load_linear(
        file,
        &format!("{}.att.key.weight", prefix),
        None,
        hidden,
        inner_dim,
        device,
    )?;
    let value = load_linear(
        file,
        &format!("{}.att.value.weight", prefix),
        None,
        hidden,
        inner_dim,
        device,
    )?;
    let gate = load_linear(
        file,
        &format!("{}.att.gate.weight", prefix),
        None,
        hidden,
        inner_dim,
        device,
    )?;
    let output = load_linear(
        file,
        &format!("{}.att.output.weight", prefix),
        None,
        inner_dim,
        hidden,
        device,
    )?;

    // Load low-rank projections
    let time_maa_w1: Tensor<B, 2> =
        file.load_f32(&format!("{}.att.time_maa_w1", prefix), device)?;
    let time_maa_w2: Tensor<B, 3> =
        file.load_f32(&format!("{}.att.time_maa_w2", prefix), device)?;
    let time_decay_w1: Tensor<B, 2> =
        file.load_f32(&format!("{}.att.time_decay_w1", prefix), device)?;
    let time_decay_w2: Tensor<B, 2> =
        file.load_f32(&format!("{}.att.time_decay_w2", prefix), device)?;

    // Group norm
    let ln_x = burn::nn::GroupNormConfig::new(num_heads, inner_dim).init(device);

    Ok(RwkvTimeMix {
        ln,
        time_maa_x: Param::from_tensor(time_maa_x),
        time_maa_r: Param::from_tensor(time_maa_r),
        time_maa_w: Param::from_tensor(time_maa_w),
        time_maa_k: Param::from_tensor(time_maa_k),
        time_maa_v: Param::from_tensor(time_maa_v),
        time_maa_a: Param::from_tensor(time_maa_a),
        time_maa_g: Param::from_tensor(time_maa_g),
        time_decay: Param::from_tensor(time_decay),
        time_faaaa: Param::from_tensor(time_faaaa),
        receptance,
        key,
        value,
        gate,
        output,
        time_maa_w1: Param::from_tensor(time_maa_w1),
        time_maa_w2: Param::from_tensor(time_maa_w2),
        time_decay_w1: Param::from_tensor(time_decay_w1),
        time_decay_w2: Param::from_tensor(time_decay_w2),
        ln_x,
        num_heads,
        head_dim,
    })
}

fn load_channel_mix<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &RwkvConfig,
    device: &B::Device,
) -> Result<RwkvChannelMix<B>, RwkvLoadError> {
    let hidden = config.hidden_size;
    let intermediate = hidden * config.ffn_multiplier;

    let ln = load_layer_norm(
        file,
        &format!("{}.ln2", prefix),
        hidden,
        config.layer_norm_eps,
        device,
    )?;

    let time_maa_k: Tensor<B, 1> = file.load_f32(&format!("{}.ffn.time_maa_k", prefix), device)?;

    let key = load_linear(
        file,
        &format!("{}.ffn.key.weight", prefix),
        None,
        hidden,
        intermediate,
        device,
    )?;
    let value = load_linear(
        file,
        &format!("{}.ffn.value.weight", prefix),
        None,
        intermediate,
        hidden,
        device,
    )?;

    Ok(RwkvChannelMix {
        ln,
        time_maa_k: Param::from_tensor(time_maa_k),
        key,
        value,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_error_display() {
        let err = RwkvLoadError::MissingTensor("test".to_string());
        assert!(err.to_string().contains("test"));
    }
}
