//! Jamba Weight Loading from Safetensors
//!
//! Loads Jamba hybrid Transformer-Mamba-MoE model weights from HuggingFace-format safetensors files.
//! Compatible with AI21's Jamba models.

use std::path::Path;

use burn::module::Param;
use burn::nn::conv::Conv1dConfig;
use burn::nn::{EmbeddingConfig, LayerNormConfig, LinearConfig};
use burn::prelude::*;
use sketchpad_convert::loader::SafeTensorFile;
use thiserror::Error;

use crate::jamba::{
    Jamba, JambaAttention, JambaBlock, JambaConfig, JambaCore, JambaDenseFFN, JambaExpert,
    JambaFFN, JambaMambaMixer, JambaMoEFFN, JambaRuntime,
};

#[derive(Error, Debug)]
pub enum JambaLoadError {
    #[error("Load error: {0}")]
    Load(#[from] sketchpad_convert::loader::LoadError),

    #[error("Missing tensor: {0}")]
    MissingTensor(String),

    #[error("Config mismatch: {0}")]
    ConfigMismatch(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Loads Jamba weights from a safetensors file
///
/// # Arguments
///
/// * `path` - Path to the safetensors file
/// * `config` - Model configuration (must match the weights)
/// * `device` - Device to load tensors onto
pub fn load_jamba<B: Backend, P: AsRef<Path>>(
    path: P,
    config: &JambaConfig,
    device: &B::Device,
) -> Result<(Jamba<B>, JambaRuntime<B>), JambaLoadError> {
    let file = SafeTensorFile::open(path)?;

    // Load embeddings
    let embed_tokens = load_embedding(
        &file,
        "model.embed_tokens.weight",
        config.vocab_size,
        config.d_model,
        device,
    )?;

    // Load Jamba layers
    let mut layers = Vec::with_capacity(config.n_layer);
    for i in 0..config.n_layer {
        let layer = load_jamba_block(&file, i, config, device)?;
        layers.push(layer);
    }

    // Load final layer norm
    let ln_f = load_layer_norm(
        &file,
        "model.final_layernorm",
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

    let model = Jamba {
        embed_tokens,
        layers,
        ln_f,
        lm_head,
    };

    let runtime = JambaRuntime {
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
) -> Result<burn::nn::Embedding<B>, JambaLoadError> {
    let weight: Tensor<B, 2> = file.load_f32(name, device)?;

    let [v, h] = weight.dims();
    if v != vocab_size || h != hidden_size {
        return Err(JambaLoadError::ConfigMismatch(format!(
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
) -> Result<burn::nn::Linear<B>, JambaLoadError> {
    let weight: Tensor<B, 2> = file.load_f32(weight_name, device)?;

    let [out_f, in_f] = weight.dims();
    if out_f != out_features || in_f != in_features {
        return Err(JambaLoadError::ConfigMismatch(format!(
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
) -> Result<burn::nn::LayerNorm<B>, JambaLoadError> {
    let weight: Tensor<B, 1> = file.load_f32(&format!("{}.weight", prefix), device)?;

    let [size] = weight.dims();
    if size != hidden_size {
        return Err(JambaLoadError::ConfigMismatch(format!(
            "{}: expected [{}], got [{}]",
            prefix, hidden_size, size
        )));
    }

    let mut ln = LayerNormConfig::new(hidden_size)
        .with_epsilon(eps)
        .init(device);
    ln.gamma = Param::from_tensor(weight);

    if file.contains(&format!("{}.bias", prefix)) {
        let bias: Tensor<B, 1> = file.load_f32(&format!("{}.bias", prefix), device)?;
        ln.beta = Some(Param::from_tensor(bias));
    }

    Ok(ln)
}

fn load_jamba_block<B: Backend>(
    file: &SafeTensorFile,
    layer_idx: usize,
    config: &JambaConfig,
    device: &B::Device,
) -> Result<JambaBlock<B>, JambaLoadError> {
    let prefix = format!("model.layers.{}", layer_idx);
    let is_attention = (layer_idx + 1) % config.attn_layer_period == 0;
    let is_moe = (layer_idx + 1) % config.moe_layer_period == 0;

    let ln = load_layer_norm(
        file,
        &format!("{}.input_layernorm", prefix),
        config.d_model,
        config.layer_norm_eps,
        device,
    )?;

    let core = if is_attention {
        JambaCore::Attention(load_jamba_attention(file, &prefix, config, device)?)
    } else {
        JambaCore::Mamba(load_jamba_mamba(file, &prefix, config, device)?)
    };

    let ffn = if is_moe {
        JambaFFN::MoE(load_jamba_moe(file, &prefix, config, device)?)
    } else {
        JambaFFN::Dense(load_jamba_dense_ffn(file, &prefix, config, device)?)
    };

    Ok(JambaBlock {
        ln,
        core,
        ffn,
        index: layer_idx,
    })
}

fn load_jamba_attention<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &JambaConfig,
    device: &B::Device,
) -> Result<JambaAttention<B>, JambaLoadError> {
    let q_proj = load_linear(
        file,
        &format!("{}.self_attn.q_proj.weight", prefix),
        None,
        config.d_model,
        config.n_heads * config.head_dim,
        device,
    )?;

    let k_proj = load_linear(
        file,
        &format!("{}.self_attn.k_proj.weight", prefix),
        None,
        config.d_model,
        config.n_kv_heads * config.head_dim,
        device,
    )?;

    let v_proj = load_linear(
        file,
        &format!("{}.self_attn.v_proj.weight", prefix),
        None,
        config.d_model,
        config.n_kv_heads * config.head_dim,
        device,
    )?;

    let o_proj = load_linear(
        file,
        &format!("{}.self_attn.o_proj.weight", prefix),
        None,
        config.n_heads * config.head_dim,
        config.d_model,
        device,
    )?;

    Ok(JambaAttention {
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        n_heads: config.n_heads,
        n_kv_heads: config.n_kv_heads,
        head_dim: config.head_dim,
    })
}

fn load_jamba_mamba<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &JambaConfig,
    device: &B::Device,
) -> Result<JambaMambaMixer<B>, JambaLoadError> {
    let d_inner = config.d_inner();
    let d_state = config.d_state;
    let dt_rank = config.dt_rank;

    let in_proj = load_linear(
        file,
        &format!("{}.mamba.in_proj.weight", prefix),
        None,
        config.d_model,
        d_inner * 2,
        device,
    )?;

    let conv1d = load_conv1d(file, &format!("{}.mamba", prefix), config, device)?;

    let x_proj = load_linear(
        file,
        &format!("{}.mamba.x_proj.weight", prefix),
        None,
        d_inner,
        dt_rank + d_state * 2,
        device,
    )?;

    let dt_proj = load_linear(
        file,
        &format!("{}.mamba.dt_proj.weight", prefix),
        Some(&format!("{}.mamba.dt_proj.bias", prefix)),
        dt_rank,
        d_inner,
        device,
    )?;

    let a_log: Tensor<B, 2> = file.load_f32(&format!("{}.mamba.A_log", prefix), device)?;
    let d: Tensor<B, 1> = file.load_f32(&format!("{}.mamba.D", prefix), device)?;

    let out_proj = load_linear(
        file,
        &format!("{}.mamba.out_proj.weight", prefix),
        None,
        d_inner,
        config.d_model,
        device,
    )?;

    Ok(JambaMambaMixer {
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
    config: &JambaConfig,
    device: &B::Device,
) -> Result<burn::nn::conv::Conv1d<B>, JambaLoadError> {
    let d_inner = config.d_inner();

    let weight: Tensor<B, 3> = file.load_f32(&format!("{}.conv1d.weight", prefix), device)?;
    let bias: Tensor<B, 1> = file.load_f32(&format!("{}.conv1d.bias", prefix), device)?;

    let mut conv = Conv1dConfig::new(d_inner, d_inner, config.d_conv)
        .with_groups(d_inner)
        .with_padding(burn::nn::PaddingConfig1d::Explicit(config.d_conv - 1))
        .with_bias(true)
        .init(device);

    conv.weight = Param::from_tensor(weight);
    conv.bias = Some(Param::from_tensor(bias));

    Ok(conv)
}

fn load_jamba_dense_ffn<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &JambaConfig,
    device: &B::Device,
) -> Result<JambaDenseFFN<B>, JambaLoadError> {
    let ln = load_layer_norm(
        file,
        &format!("{}.post_attention_layernorm", prefix),
        config.d_model,
        config.layer_norm_eps,
        device,
    )?;

    let gate_proj = load_linear(
        file,
        &format!("{}.mlp.gate_proj.weight", prefix),
        None,
        config.d_model,
        config.intermediate_size,
        device,
    )?;

    let up_proj = load_linear(
        file,
        &format!("{}.mlp.up_proj.weight", prefix),
        None,
        config.d_model,
        config.intermediate_size,
        device,
    )?;

    let down_proj = load_linear(
        file,
        &format!("{}.mlp.down_proj.weight", prefix),
        None,
        config.intermediate_size,
        config.d_model,
        device,
    )?;

    Ok(JambaDenseFFN {
        ln,
        gate_proj,
        up_proj,
        down_proj,
    })
}

fn load_jamba_moe<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &JambaConfig,
    device: &B::Device,
) -> Result<JambaMoEFFN<B>, JambaLoadError> {
    let ln = load_layer_norm(
        file,
        &format!("{}.post_attention_layernorm", prefix),
        config.d_model,
        config.layer_norm_eps,
        device,
    )?;

    let router = load_linear(
        file,
        &format!("{}.block_sparse_moe.router.weight", prefix),
        None,
        config.d_model,
        config.n_experts,
        device,
    )?;

    let mut experts = Vec::with_capacity(config.n_experts);
    for i in 0..config.n_experts {
        let expert = load_jamba_expert(file, prefix, i, config, device)?;
        experts.push(expert);
    }

    Ok(JambaMoEFFN {
        ln,
        router,
        experts,
        n_experts_per_tok: config.n_experts_per_tok,
    })
}

fn load_jamba_expert<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    expert_idx: usize,
    config: &JambaConfig,
    device: &B::Device,
) -> Result<JambaExpert<B>, JambaLoadError> {
    let expert_prefix = format!("{}.block_sparse_moe.experts.{}", prefix, expert_idx);

    let gate_proj = load_linear(
        file,
        &format!("{}.gate_proj.weight", expert_prefix),
        None,
        config.d_model,
        config.intermediate_size,
        device,
    )?;

    let up_proj = load_linear(
        file,
        &format!("{}.up_proj.weight", expert_prefix),
        None,
        config.d_model,
        config.intermediate_size,
        device,
    )?;

    let down_proj = load_linear(
        file,
        &format!("{}.down_proj.weight", expert_prefix),
        None,
        config.intermediate_size,
        config.d_model,
        device,
    )?;

    Ok(JambaExpert {
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
        let err = JambaLoadError::MissingTensor("test".to_string());
        assert!(err.to_string().contains("test"));
    }
}
