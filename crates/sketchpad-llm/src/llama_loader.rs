//! LLaMA Weight Loading from Safetensors
//!
//! Loads LLaMA model weights from HuggingFace-format safetensors files.
//!
//! # Supported Formats
//!
//! - HuggingFace Transformers format (model.safetensors or model-*.safetensors)
//! - Meta's original format (consolidated.*.safetensors)

use std::path::Path;

use burn::module::Param;
use burn::nn::{EmbeddingConfig, LinearConfig};
use burn::prelude::*;
use sketchpad_convert::loader::SafeTensorFile;
use thiserror::Error;

use crate::llama::{Llama, LlamaConfig, LlamaRuntime};
use sketchpad_core::rmsnorm::RmsNorm;
use sketchpad_core::rope::RotaryEmbedding;
use sketchpad_core::transformer::{MultiHeadAttention, TransformerBlock};

#[derive(Error, Debug)]
pub enum LlamaLoadError {
    #[error("Load error: {0}")]
    Load(#[from] sketchpad_convert::loader::LoadError),

    #[error("Missing tensor: {0}")]
    MissingTensor(String),

    #[error("Config mismatch: {0}")]
    ConfigMismatch(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Loads LLaMA weights from a safetensors file
///
/// # Arguments
///
/// * `path` - Path to the safetensors file
/// * `config` - Model configuration (must match the weights)
/// * `device` - Device to load tensors onto
///
/// # Returns
///
/// Tuple of (model, runtime) ready for inference
pub fn load_llama<B: Backend, P: AsRef<Path>>(
    path: P,
    config: &LlamaConfig,
    device: &B::Device,
) -> Result<(Llama<B>, LlamaRuntime<B>), LlamaLoadError> {
    let file = SafeTensorFile::open(path)?;

    // Load embeddings
    let embed_tokens = load_embedding(
        &file,
        "model.embed_tokens.weight",
        config.vocab_size,
        config.hidden_size,
        device,
    )?;

    // Load transformer layers
    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let layer = load_transformer_layer(&file, i, config, device)?;
        layers.push(layer);
    }

    // Load final norm
    let norm = load_rmsnorm(
        &file,
        "model.norm.weight",
        config.hidden_size,
        config.norm_eps,
        device,
    )?;

    // Load LM head
    let lm_head = load_linear(
        &file,
        "lm_head.weight",
        None,
        config.hidden_size,
        config.vocab_size,
        device,
    )?;

    let model = Llama {
        embed_tokens,
        layers,
        norm,
        lm_head,
    };

    // Create runtime (RoPE is computed, not loaded)
    let head_dim = config.hidden_size / config.num_heads;
    let runtime = LlamaRuntime {
        rope: RotaryEmbedding::with_base(head_dim, config.max_seq_len, config.rope_base, device),
        config: config.clone(),
    };

    Ok((model, runtime))
}

/// Load an embedding layer
fn load_embedding<B: Backend>(
    file: &SafeTensorFile,
    name: &str,
    vocab_size: usize,
    hidden_size: usize,
    device: &B::Device,
) -> Result<burn::nn::Embedding<B>, LlamaLoadError> {
    let weight: Tensor<B, 2> = file.load_f32(name, device)?;

    // Verify shape
    let [v, h] = weight.dims();
    if v != vocab_size || h != hidden_size {
        return Err(LlamaLoadError::ConfigMismatch(format!(
            "{}: expected [{}, {}], got [{}, {}]",
            name, vocab_size, hidden_size, v, h
        )));
    }

    // Create embedding and set weight
    let mut embedding = EmbeddingConfig::new(vocab_size, hidden_size).init(device);
    embedding.weight = Param::from_tensor(weight);

    Ok(embedding)
}

/// Load a linear layer
fn load_linear<B: Backend>(
    file: &SafeTensorFile,
    weight_name: &str,
    bias_name: Option<&str>,
    in_features: usize,
    out_features: usize,
    device: &B::Device,
) -> Result<burn::nn::Linear<B>, LlamaLoadError> {
    let weight: Tensor<B, 2> = file.load_f32(weight_name, device)?;

    // LLaMA stores weights as [out_features, in_features]
    let [out_f, in_f] = weight.dims();
    if out_f != out_features || in_f != in_features {
        return Err(LlamaLoadError::ConfigMismatch(format!(
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

/// Load an RMSNorm layer
fn load_rmsnorm<B: Backend>(
    file: &SafeTensorFile,
    name: &str,
    hidden_size: usize,
    eps: f64,
    device: &B::Device,
) -> Result<RmsNorm<B>, LlamaLoadError> {
    let weight: Tensor<B, 1> = file.load_f32(name, device)?;

    let [size] = weight.dims();
    if size != hidden_size {
        return Err(LlamaLoadError::ConfigMismatch(format!(
            "{}: expected [{}], got [{}]",
            name, hidden_size, size
        )));
    }

    Ok(RmsNorm::from_weight(weight, eps))
}

/// Load a transformer layer
fn load_transformer_layer<B: Backend>(
    file: &SafeTensorFile,
    layer_idx: usize,
    config: &LlamaConfig,
    device: &B::Device,
) -> Result<TransformerBlock<B>, LlamaLoadError> {
    let prefix = format!("model.layers.{}", layer_idx);

    // Load attention
    let attention = load_attention(file, &prefix, config, device)?;

    // Load FFN (SwiGLU)
    let ffn = load_swiglu_ffn(file, &prefix, config, device)?;

    // Load norms
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

    Ok(TransformerBlock {
        attention,
        ffn,
        input_norm,
        post_attention_norm,
    })
}

/// Load multi-head attention
fn load_attention<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &LlamaConfig,
    device: &B::Device,
) -> Result<MultiHeadAttention<B>, LlamaLoadError> {
    let head_dim = config.hidden_size / config.num_heads;
    let kv_dim = head_dim * config.num_kv_heads;

    let q_proj = load_linear(
        file,
        &format!("{}.self_attn.q_proj.weight", prefix),
        None,
        config.hidden_size,
        config.hidden_size,
        device,
    )?;

    let k_proj = load_linear(
        file,
        &format!("{}.self_attn.k_proj.weight", prefix),
        None,
        config.hidden_size,
        kv_dim,
        device,
    )?;

    let v_proj = load_linear(
        file,
        &format!("{}.self_attn.v_proj.weight", prefix),
        None,
        config.hidden_size,
        kv_dim,
        device,
    )?;

    let o_proj = load_linear(
        file,
        &format!("{}.self_attn.o_proj.weight", prefix),
        None,
        config.hidden_size,
        config.hidden_size,
        device,
    )?;

    Ok(MultiHeadAttention {
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        num_heads: config.num_heads,
        num_kv_heads: config.num_kv_heads,
        head_dim,
    })
}

/// Load SwiGLU FFN
fn load_swiglu_ffn<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &LlamaConfig,
    device: &B::Device,
) -> Result<sketchpad_core::glu::SwiGluFfn<B>, LlamaLoadError> {
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

    Ok(sketchpad_core::glu::SwiGluFfn {
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
        let err = LlamaLoadError::MissingTensor("test".to_string());
        assert!(err.to_string().contains("test"));
    }
}
