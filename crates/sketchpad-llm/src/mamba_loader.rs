//! Mamba Weight Loading from Safetensors
//!
//! Loads Mamba model weights from HuggingFace-format safetensors files.
//! Compatible with models like state-spaces/mamba-*.

use std::path::Path;

use burn::module::Param;
use burn::nn::conv::Conv1dConfig;
use burn::nn::{EmbeddingConfig, LayerNormConfig, LinearConfig};
use burn::prelude::*;
use sketchpad_convert::loader::SafeTensorFile;
use thiserror::Error;

use crate::mamba::{Mamba, MambaBlock, MambaConfig, MambaMixer, MambaRuntime};

#[derive(Error, Debug)]
pub enum MambaLoadError {
    #[error("Load error: {0}")]
    Load(#[from] sketchpad_convert::loader::LoadError),

    #[error("Missing tensor: {0}")]
    MissingTensor(String),

    #[error("Config mismatch: {0}")]
    ConfigMismatch(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Loads Mamba weights from a safetensors file
///
/// # Arguments
///
/// * `path` - Path to the safetensors file
/// * `config` - Model configuration (must match the weights)
/// * `device` - Device to load tensors onto
pub fn load_mamba<B: Backend, P: AsRef<Path>>(
    path: P,
    config: &MambaConfig,
    device: &B::Device,
) -> Result<(Mamba<B>, MambaRuntime<B>), MambaLoadError> {
    let file = SafeTensorFile::open(path)?;

    // Load embeddings
    let embed_tokens = load_embedding(
        &file,
        "backbone.embedding.weight",
        config.vocab_size,
        config.d_model,
        device,
    )?;

    // Load Mamba layers
    let mut layers = Vec::with_capacity(config.n_layer);
    for i in 0..config.n_layer {
        let layer = load_mamba_block(&file, i, config, device)?;
        layers.push(layer);
    }

    // Load final layer norm
    let ln_f = load_layer_norm(
        &file,
        "backbone.norm_f",
        config.d_model,
        config.layer_norm_eps,
        device,
    )?;

    // Load LM head
    let lm_head = load_linear(
        &file,
        "lm_head.weight",
        None,
        config.d_model,
        config.vocab_size,
        device,
    )?;

    let model = Mamba {
        embed_tokens,
        layers,
        ln_f,
        lm_head,
    };

    let runtime = MambaRuntime {
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
) -> Result<burn::nn::Embedding<B>, MambaLoadError> {
    let weight: Tensor<B, 2> = file.load_f32(name, device)?;

    let [v, h] = weight.dims();
    if v != vocab_size || h != hidden_size {
        return Err(MambaLoadError::ConfigMismatch(format!(
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
) -> Result<burn::nn::Linear<B>, MambaLoadError> {
    let weight: Tensor<B, 2> = file.load_f32(weight_name, device)?;

    let [out_f, in_f] = weight.dims();
    if out_f != out_features || in_f != in_features {
        return Err(MambaLoadError::ConfigMismatch(format!(
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
) -> Result<burn::nn::LayerNorm<B>, MambaLoadError> {
    let weight: Tensor<B, 1> = file.load_f32(&format!("{}.weight", prefix), device)?;

    let [size] = weight.dims();
    if size != hidden_size {
        return Err(MambaLoadError::ConfigMismatch(format!(
            "{}: expected [{}], got [{}]",
            prefix, hidden_size, size
        )));
    }

    let mut ln = LayerNormConfig::new(hidden_size)
        .with_epsilon(eps)
        .init(device);
    ln.gamma = Param::from_tensor(weight);

    // Mamba uses LayerNorm without bias (RMSNorm-style)
    if file.contains(&format!("{}.bias", prefix)) {
        let bias: Tensor<B, 1> = file.load_f32(&format!("{}.bias", prefix), device)?;
        ln.beta = Some(Param::from_tensor(bias));
    }

    Ok(ln)
}

fn load_mamba_block<B: Backend>(
    file: &SafeTensorFile,
    layer_idx: usize,
    config: &MambaConfig,
    device: &B::Device,
) -> Result<MambaBlock<B>, MambaLoadError> {
    let prefix = format!("backbone.layers.{}", layer_idx);

    let ln = load_layer_norm(
        file,
        &format!("{}.norm", prefix),
        config.d_model,
        config.layer_norm_eps,
        device,
    )?;
    let mixer = load_mamba_mixer(file, &prefix, config, device)?;

    Ok(MambaBlock { ln, mixer })
}

fn load_mamba_mixer<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &MambaConfig,
    device: &B::Device,
) -> Result<MambaMixer<B>, MambaLoadError> {
    let d_inner = config.d_inner();
    let d_state = config.d_state;
    let dt_rank = config.dt_rank;

    // Input projection
    let in_proj = load_linear(
        file,
        &format!("{}.mixer.in_proj.weight", prefix),
        None,
        config.d_model,
        d_inner * 2,
        device,
    )?;

    // Convolution (depthwise)
    let conv1d = load_conv1d(file, prefix, config, device)?;

    // SSM projections
    let x_proj = load_linear(
        file,
        &format!("{}.mixer.x_proj.weight", prefix),
        None,
        d_inner,
        dt_rank + d_state * 2,
        device,
    )?;

    let dt_proj = load_linear(
        file,
        &format!("{}.mixer.dt_proj.weight", prefix),
        Some(&format!("{}.mixer.dt_proj.bias", prefix)),
        dt_rank,
        d_inner,
        device,
    )?;

    // A and D parameters
    let a_log: Tensor<B, 2> = file.load_f32(&format!("{}.mixer.A_log", prefix), device)?;
    let d: Tensor<B, 1> = file.load_f32(&format!("{}.mixer.D", prefix), device)?;

    // Output projection
    let out_proj = load_linear(
        file,
        &format!("{}.mixer.out_proj.weight", prefix),
        None,
        d_inner,
        config.d_model,
        device,
    )?;

    Ok(MambaMixer {
        in_proj,
        conv1d,
        x_proj,
        dt_proj,
        a_log: Param::from_tensor(a_log),
        d: Param::from_tensor(d),
        out_proj,
        d_inner,
        d_state,
        d_conv: config.d_conv,
        dt_rank,
    })
}

fn load_conv1d<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &MambaConfig,
    device: &B::Device,
) -> Result<burn::nn::conv::Conv1d<B>, MambaLoadError> {
    let d_inner = config.d_inner();

    let weight: Tensor<B, 3> = file.load_f32(&format!("{}.mixer.conv1d.weight", prefix), device)?;
    let bias: Tensor<B, 1> = file.load_f32(&format!("{}.mixer.conv1d.bias", prefix), device)?;

    let mut conv = Conv1dConfig::new(d_inner, d_inner, config.d_conv)
        .with_groups(d_inner)
        .with_padding(burn::nn::PaddingConfig1d::Explicit(config.d_conv - 1))
        .with_bias(true)
        .init(device);

    conv.weight = Param::from_tensor(weight);
    conv.bias = Some(Param::from_tensor(bias));

    Ok(conv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_error_display() {
        let err = MambaLoadError::MissingTensor("test".to_string());
        assert!(err.to_string().contains("test"));
    }
}
