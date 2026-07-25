//! Qwen Weight Loading from Safetensors
//!
//! Loads Qwen model weights from HuggingFace-format safetensors files.

use std::path::Path;

use burn::module::Param;
use burn::nn::{EmbeddingConfig, LinearConfig};
use burn::prelude::*;
use sketchpad_convert::loader::SafeTensorFile;
use thiserror::Error;

use crate::qwen::{Qwen, QwenAttention, QwenConfig, QwenLayer, QwenRuntime};
use sketchpad_core::glu::SwiGluFfn;
use sketchpad_core::rmsnorm::RmsNorm;
use sketchpad_core::rope::RotaryEmbedding;

#[derive(Error, Debug)]
pub enum QwenLoadError {
    #[error("Load error: {0}")]
    Load(#[from] sketchpad_convert::loader::LoadError),

    #[error("Missing tensor: {0}")]
    MissingTensor(String),

    #[error("Config mismatch: {0}")]
    ConfigMismatch(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Loads Qwen weights from a safetensors file
pub fn load_qwen<B: Backend, P: AsRef<Path>>(
    path: P,
    config: &QwenConfig,
    device: &B::Device,
) -> Result<(Qwen<B>, QwenRuntime<B>), QwenLoadError> {
    let file = SafeTensorFile::open(path)?;

    let embed_tokens = load_embedding(
        &file,
        "model.embed_tokens.weight",
        config.vocab_size,
        config.hidden_size,
        device,
    )?;

    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let layer = load_qwen_layer(&file, i, config, device)?;
        layers.push(layer);
    }

    let norm = load_rmsnorm(
        &file,
        "model.norm.weight",
        config.hidden_size,
        config.norm_eps,
        device,
    )?;

    // Load lm_head if not tied, otherwise None
    let lm_head = if config.tie_embeddings {
        None
    } else {
        Some(load_linear(
            &file,
            "lm_head.weight",
            None,
            config.hidden_size,
            config.vocab_size,
            device,
        )?)
    };

    let model = Qwen {
        embed_tokens,
        layers,
        norm,
        lm_head,
    };

    let head_dim = config.hidden_size / config.num_heads;
    let runtime = QwenRuntime {
        rope: RotaryEmbedding::with_base(head_dim, config.max_seq_len, config.rope_base, device),
        config: config.clone(),
    };

    Ok((model, runtime))
}

fn load_embedding<B: Backend>(
    file: &SafeTensorFile,
    name: &str,
    vocab_size: usize,
    hidden_size: usize,
    device: &B::Device,
) -> Result<burn::nn::Embedding<B>, QwenLoadError> {
    let weight: Tensor<B, 2> = file.load_f32(name, device)?;

    let [v, h] = weight.dims();
    if v != vocab_size || h != hidden_size {
        return Err(QwenLoadError::ConfigMismatch(format!(
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
) -> Result<burn::nn::Linear<B>, QwenLoadError> {
    let weight: Tensor<B, 2> = file.load_f32(weight_name, device)?;

    let [out_f, in_f] = weight.dims();
    if out_f != out_features || in_f != in_features {
        return Err(QwenLoadError::ConfigMismatch(format!(
            "{}: expected [{}, {}], got [{}, {}]",
            weight_name, out_features, in_features, out_f, in_f
        )));
    }

    let has_bias = bias_name.map(|n| file.contains(n)).unwrap_or(false);
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

fn load_rmsnorm<B: Backend>(
    file: &SafeTensorFile,
    name: &str,
    hidden_size: usize,
    eps: f64,
    device: &B::Device,
) -> Result<RmsNorm<B>, QwenLoadError> {
    let weight: Tensor<B, 1> = file.load_f32(name, device)?;

    let [size] = weight.dims();
    if size != hidden_size {
        return Err(QwenLoadError::ConfigMismatch(format!(
            "{}: expected [{}], got [{}]",
            name, hidden_size, size
        )));
    }

    Ok(RmsNorm::from_weight(weight, eps))
}

fn load_qwen_layer<B: Backend>(
    file: &SafeTensorFile,
    layer_idx: usize,
    config: &QwenConfig,
    device: &B::Device,
) -> Result<QwenLayer<B>, QwenLoadError> {
    let prefix = format!("model.layers.{}", layer_idx);

    let attention = load_qwen_attention(file, &prefix, config, device)?;
    let ffn = load_swiglu_ffn(file, &prefix, config, device)?;

    let input_norm = load_rmsnorm(
        file,
        &format!("{}.input_layernorm.weight", prefix),
        config.hidden_size,
        config.norm_eps,
        device,
    )?;

    let post_attention_norm = load_rmsnorm(
        file,
        &format!("{}.post_attention_layernorm.weight", prefix),
        config.hidden_size,
        config.norm_eps,
        device,
    )?;

    Ok(QwenLayer {
        attention,
        ffn,
        input_norm,
        post_attention_norm,
    })
}

fn load_qwen_attention<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &QwenConfig,
    device: &B::Device,
) -> Result<QwenAttention<B>, QwenLoadError> {
    let head_dim = config.hidden_size / config.num_heads;
    let kv_dim = head_dim * config.num_kv_heads;

    // Qwen has bias in QKV projections
    let q_proj = load_linear(
        file,
        &format!("{}.self_attn.q_proj.weight", prefix),
        Some(&format!("{}.self_attn.q_proj.bias", prefix)),
        config.hidden_size,
        config.hidden_size,
        device,
    )?;

    let k_proj = load_linear(
        file,
        &format!("{}.self_attn.k_proj.weight", prefix),
        Some(&format!("{}.self_attn.k_proj.bias", prefix)),
        config.hidden_size,
        kv_dim,
        device,
    )?;

    let v_proj = load_linear(
        file,
        &format!("{}.self_attn.v_proj.weight", prefix),
        Some(&format!("{}.self_attn.v_proj.bias", prefix)),
        config.hidden_size,
        kv_dim,
        device,
    )?;

    // o_proj has no bias
    let o_proj = load_linear(
        file,
        &format!("{}.self_attn.o_proj.weight", prefix),
        None,
        config.hidden_size,
        config.hidden_size,
        device,
    )?;

    Ok(QwenAttention {
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        num_heads: config.num_heads,
        num_kv_heads: config.num_kv_heads,
        head_dim,
    })
}

fn load_swiglu_ffn<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &QwenConfig,
    device: &B::Device,
) -> Result<SwiGluFfn<B>, QwenLoadError> {
    let gate_proj = load_linear(
        file,
        &format!("{}.mlp.gate_proj.weight", prefix),
        None,
        config.hidden_size,
        config.intermediate_size,
        device,
    )?;

    let up_proj = load_linear(
        file,
        &format!("{}.mlp.up_proj.weight", prefix),
        None,
        config.hidden_size,
        config.intermediate_size,
        device,
    )?;

    let down_proj = load_linear(
        file,
        &format!("{}.mlp.down_proj.weight", prefix),
        None,
        config.intermediate_size,
        config.hidden_size,
        device,
    )?;

    Ok(SwiGluFfn {
        gate_proj,
        up_proj,
        down_proj,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_error_display() {
        let err = QwenLoadError::MissingTensor("test".to_string());
        assert!(err.to_string().contains("test"));
    }
}
