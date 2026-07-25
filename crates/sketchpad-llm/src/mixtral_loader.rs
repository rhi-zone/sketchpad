//! Mixtral Weight Loading from Safetensors
//!
//! Loads Mixtral MoE model weights from HuggingFace-format safetensors files.
//! Compatible with Mistral AI's Mixtral 8x7B and 8x22B models.

use std::path::Path;

use burn::module::Param;
use burn::nn::{EmbeddingConfig, LinearConfig};
use burn::prelude::*;
use sketchpad_convert::loader::SafeTensorFile;
use sketchpad_core::glu::SwiGluFfn;
use sketchpad_core::moe::{MoeRouter, SparseMoeFfn};
use sketchpad_core::rmsnorm::RmsNorm;
use sketchpad_core::rope::RotaryEmbedding;
use sketchpad_core::transformer::MultiHeadAttention;
use thiserror::Error;

use crate::mixtral::{Mixtral, MixtralConfig, MixtralLayer, MixtralRuntime};

#[derive(Error, Debug)]
pub enum MixtralLoadError {
    #[error("Load error: {0}")]
    Load(#[from] sketchpad_convert::loader::LoadError),

    #[error("Missing tensor: {0}")]
    MissingTensor(String),

    #[error("Config mismatch: {0}")]
    ConfigMismatch(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Loads Mixtral weights from a safetensors file
///
/// # Arguments
///
/// * `path` - Path to the safetensors file
/// * `config` - Model configuration (must match the weights)
/// * `device` - Device to load tensors onto
pub fn load_mixtral<B: Backend, P: AsRef<Path>>(
    path: P,
    config: &MixtralConfig,
    device: &B::Device,
) -> Result<(Mixtral<B>, MixtralRuntime<B>), MixtralLoadError> {
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
        let layer = load_mixtral_layer(&file, i, config, device)?;
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

    let model = Mixtral {
        embed_tokens,
        layers,
        norm,
        lm_head,
    };

    let head_dim = config.hidden_size / config.num_heads;
    let runtime = MixtralRuntime {
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
) -> Result<burn::nn::Embedding<B>, MixtralLoadError> {
    let weight: Tensor<B, 2> = file.load_f32(name, device)?;

    let [v, h] = weight.dims();
    if v != vocab_size || h != hidden_size {
        return Err(MixtralLoadError::ConfigMismatch(format!(
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
) -> Result<burn::nn::Linear<B>, MixtralLoadError> {
    let weight: Tensor<B, 2> = file.load_f32(weight_name, device)?;

    let [out_f, in_f] = weight.dims();
    if out_f != out_features || in_f != in_features {
        return Err(MixtralLoadError::ConfigMismatch(format!(
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

fn load_rmsnorm<B: Backend>(
    file: &SafeTensorFile,
    name: &str,
    hidden_size: usize,
    eps: f64,
    device: &B::Device,
) -> Result<RmsNorm<B>, MixtralLoadError> {
    let weight: Tensor<B, 1> = file.load_f32(name, device)?;

    let [size] = weight.dims();
    if size != hidden_size {
        return Err(MixtralLoadError::ConfigMismatch(format!(
            "{}: expected [{}], got [{}]",
            name, hidden_size, size
        )));
    }

    Ok(RmsNorm::from_weight(weight, eps))
}

fn load_mixtral_layer<B: Backend>(
    file: &SafeTensorFile,
    layer_idx: usize,
    config: &MixtralConfig,
    device: &B::Device,
) -> Result<MixtralLayer<B>, MixtralLoadError> {
    let prefix = format!("model.layers.{}", layer_idx);

    let attention = load_attention(file, &prefix, config, device)?;
    let moe = load_sparse_moe(file, &prefix, config, device)?;

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

    Ok(MixtralLayer {
        attention,
        moe,
        input_norm,
        post_attention_norm,
    })
}

fn load_attention<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &MixtralConfig,
    device: &B::Device,
) -> Result<MultiHeadAttention<B>, MixtralLoadError> {
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

fn load_sparse_moe<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &MixtralConfig,
    device: &B::Device,
) -> Result<SparseMoeFfn<B>, MixtralLoadError> {
    // Load router
    let gate_weight: Tensor<B, 2> =
        file.load_f32(&format!("{}.block_sparse_moe.gate.weight", prefix), device)?;

    let mut gate = LinearConfig::new(config.hidden_size, config.num_experts)
        .with_bias(false)
        .init(device);
    gate.weight = Param::from_tensor(gate_weight);

    let router = MoeRouter {
        gate,
        num_experts: config.num_experts,
        top_k: config.num_experts_per_tok,
    };

    // Load experts
    let mut experts = Vec::with_capacity(config.num_experts);
    for i in 0..config.num_experts {
        let expert = load_expert(file, prefix, i, config, device)?;
        experts.push(expert);
    }

    Ok(SparseMoeFfn {
        router,
        experts,
        num_experts: config.num_experts,
        top_k: config.num_experts_per_tok,
    })
}

fn load_expert<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    expert_idx: usize,
    config: &MixtralConfig,
    device: &B::Device,
) -> Result<SwiGluFfn<B>, MixtralLoadError> {
    let expert_prefix = format!("{}.block_sparse_moe.experts.{}", prefix, expert_idx);

    // Mixtral uses w1=gate, w2=down, w3=up (different naming from LLaMA)
    let gate_proj = load_linear(
        file,
        &format!("{}.w1.weight", expert_prefix),
        None,
        config.hidden_size,
        config.intermediate_size,
        device,
    )?;

    let down_proj = load_linear(
        file,
        &format!("{}.w2.weight", expert_prefix),
        None,
        config.intermediate_size,
        config.hidden_size,
        device,
    )?;

    let up_proj = load_linear(
        file,
        &format!("{}.w3.weight", expert_prefix),
        None,
        config.hidden_size,
        config.intermediate_size,
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
        let err = MixtralLoadError::MissingTensor("test".to_string());
        assert!(err.to_string().contains("test"));
    }
}
