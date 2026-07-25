//! Gemma 2 Weight Loading from Safetensors
//!
//! Loads Gemma 2 model weights from HuggingFace-format safetensors files.

use std::path::Path;

use burn::module::Param;
use burn::nn::{EmbeddingConfig, LinearConfig};
use burn::prelude::*;
use sketchpad_convert::loader::SafeTensorFile;
use thiserror::Error;

use crate::gemma::{Gemma, GemmaAttention, GemmaConfig, GemmaFfn, GemmaLayer, GemmaRuntime};
use sketchpad_core::rmsnorm::RmsNorm;
use sketchpad_core::rope::RotaryEmbedding;

#[derive(Error, Debug)]
pub enum GemmaLoadError {
    #[error("Load error: {0}")]
    Load(#[from] sketchpad_convert::loader::LoadError),

    #[error("Missing tensor: {0}")]
    MissingTensor(String),

    #[error("Config mismatch: {0}")]
    ConfigMismatch(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Loads Gemma 2 weights from a safetensors file
pub fn load_gemma<B: Backend, P: AsRef<Path>>(
    path: P,
    config: &GemmaConfig,
    device: &B::Device,
) -> Result<(Gemma<B>, GemmaRuntime<B>), GemmaLoadError> {
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
        let layer = load_gemma_layer(&file, i, config, device)?;
        layers.push(layer);
    }

    let norm = load_rmsnorm(
        &file,
        "model.norm.weight",
        config.hidden_size,
        config.norm_eps,
        device,
    )?;

    let model = Gemma {
        embed_tokens,
        layers,
        norm,
    };

    let head_dim = config.hidden_size / config.num_heads;
    let runtime = GemmaRuntime {
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
) -> Result<burn::nn::Embedding<B>, GemmaLoadError> {
    let weight: Tensor<B, 2> = file.load_f32(name, device)?;

    let [v, h] = weight.dims();
    if v != vocab_size || h != hidden_size {
        return Err(GemmaLoadError::ConfigMismatch(format!(
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
    in_features: usize,
    out_features: usize,
    device: &B::Device,
) -> Result<burn::nn::Linear<B>, GemmaLoadError> {
    let weight: Tensor<B, 2> = file.load_f32(weight_name, device)?;

    let [out_f, in_f] = weight.dims();
    if out_f != out_features || in_f != in_features {
        return Err(GemmaLoadError::ConfigMismatch(format!(
            "{}: expected [{}, {}], got [{}, {}]",
            weight_name, out_features, in_features, out_f, in_f
        )));
    }

    let mut linear = LinearConfig::new(in_features, out_features)
        .with_bias(false)
        .init(device);

    linear.weight = Param::from_tensor(weight);

    Ok(linear)
}

fn load_rmsnorm<B: Backend>(
    file: &SafeTensorFile,
    name: &str,
    hidden_size: usize,
    eps: f64,
    device: &B::Device,
) -> Result<RmsNorm<B>, GemmaLoadError> {
    let weight: Tensor<B, 1> = file.load_f32(name, device)?;

    let [size] = weight.dims();
    if size != hidden_size {
        return Err(GemmaLoadError::ConfigMismatch(format!(
            "{}: expected [{}], got [{}]",
            name, hidden_size, size
        )));
    }

    Ok(RmsNorm::from_weight(weight, eps))
}

fn load_gemma_layer<B: Backend>(
    file: &SafeTensorFile,
    layer_idx: usize,
    config: &GemmaConfig,
    device: &B::Device,
) -> Result<GemmaLayer<B>, GemmaLoadError> {
    let prefix = format!("model.layers.{}", layer_idx);

    let attention = load_gemma_attention(file, &prefix, config, device)?;
    let ffn = load_gemma_ffn(file, &prefix, config, device)?;

    // Gemma has 4 norms per layer
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

    let pre_ffn_norm = load_rmsnorm(
        file,
        &format!("{}.pre_feedforward_layernorm.weight", prefix),
        config.hidden_size,
        config.norm_eps,
        device,
    )?;

    let post_ffn_norm = load_rmsnorm(
        file,
        &format!("{}.post_feedforward_layernorm.weight", prefix),
        config.hidden_size,
        config.norm_eps,
        device,
    )?;

    // Even layers use sliding window, odd use global
    let use_sliding_window = layer_idx % 2 == 0;

    Ok(GemmaLayer {
        attention,
        ffn,
        input_norm,
        post_attention_norm,
        pre_ffn_norm,
        post_ffn_norm,
        use_sliding_window,
    })
}

fn load_gemma_attention<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &GemmaConfig,
    device: &B::Device,
) -> Result<GemmaAttention<B>, GemmaLoadError> {
    let head_dim = config.hidden_size / config.num_heads;
    let kv_dim = head_dim * config.num_kv_heads;

    // Gemma has no bias in attention projections
    let q_proj = load_linear(
        file,
        &format!("{}.self_attn.q_proj.weight", prefix),
        config.hidden_size,
        config.hidden_size,
        device,
    )?;

    let k_proj = load_linear(
        file,
        &format!("{}.self_attn.k_proj.weight", prefix),
        config.hidden_size,
        kv_dim,
        device,
    )?;

    let v_proj = load_linear(
        file,
        &format!("{}.self_attn.v_proj.weight", prefix),
        config.hidden_size,
        kv_dim,
        device,
    )?;

    let o_proj = load_linear(
        file,
        &format!("{}.self_attn.o_proj.weight", prefix),
        config.hidden_size,
        config.hidden_size,
        device,
    )?;

    Ok(GemmaAttention {
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        num_heads: config.num_heads,
        num_kv_heads: config.num_kv_heads,
        head_dim,
    })
}

fn load_gemma_ffn<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &GemmaConfig,
    device: &B::Device,
) -> Result<GemmaFfn<B>, GemmaLoadError> {
    let gate_proj = load_linear(
        file,
        &format!("{}.mlp.gate_proj.weight", prefix),
        config.hidden_size,
        config.intermediate_size,
        device,
    )?;

    let up_proj = load_linear(
        file,
        &format!("{}.mlp.up_proj.weight", prefix),
        config.hidden_size,
        config.intermediate_size,
        device,
    )?;

    let down_proj = load_linear(
        file,
        &format!("{}.mlp.down_proj.weight", prefix),
        config.intermediate_size,
        config.hidden_size,
        device,
    )?;

    Ok(GemmaFfn {
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
        let err = GemmaLoadError::MissingTensor("test".to_string());
        assert!(err.to_string().contains("test"));
    }
}
