//! Gemma 4 Weight Loading from Safetensors
//!
//! Loads Gemma 4 model weights from HuggingFace-format safetensors files.

use std::path::Path;

use burn::module::Param;
use burn::nn::{EmbeddingConfig, LinearConfig};
use burn::prelude::*;
use sketchpad_convert::loader::SafeTensorFile;
use thiserror::Error;

use crate::gemma4::{
    ExpertWeights, Gemma4, Gemma4Attention, Gemma4Config, Gemma4DenseFfn, Gemma4ExpertFfn,
    Gemma4Ffn, Gemma4Layer, Gemma4MoE, Gemma4Runtime,
};
use sketchpad_core::rmsnorm::RmsNorm;
use sketchpad_core::rope::RotaryEmbedding;

#[derive(Error, Debug)]
pub enum Gemma4LoadError {
    #[error("Load error: {0}")]
    Load(#[from] sketchpad_convert::loader::LoadError),

    #[error("Missing tensor: {0}")]
    MissingTensor(String),

    #[error("Config mismatch: {0}")]
    ConfigMismatch(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Loads Gemma 4 weights from a safetensors file
pub fn load_gemma4<B: Backend, P: AsRef<Path>>(
    path: P,
    config: &Gemma4Config,
    device: &B::Device,
) -> Result<(Gemma4<B>, Gemma4Runtime<B>), Gemma4LoadError> {
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
        let layer = load_gemma4_layer(&file, i, config, device)?;
        layers.push(layer);
    }

    let norm = load_rmsnorm(
        &file,
        "model.norm.weight",
        config.hidden_size,
        config.norm_eps,
        device,
    )?;

    let model = Gemma4 {
        embed_tokens,
        layers,
        norm,
    };

    let mut ropes = std::collections::HashMap::new();
    ropes.insert(
        config.head_dim,
        RotaryEmbedding::with_base(
            config.head_dim,
            config.max_seq_len,
            config.rope_base,
            device,
        ),
    );
    let runtime = Gemma4Runtime {
        ropes,
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
) -> Result<burn::nn::Embedding<B>, Gemma4LoadError> {
    let weight: Tensor<B, 2> = file.load_f32(name, device)?;

    let [v, h] = weight.dims();
    if v != vocab_size || h != hidden_size {
        return Err(Gemma4LoadError::ConfigMismatch(format!(
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
) -> Result<burn::nn::Linear<B>, Gemma4LoadError> {
    let weight: Tensor<B, 2> = file.load_f32(weight_name, device)?;

    let [out_f, in_f] = weight.dims();
    if out_f != out_features || in_f != in_features {
        return Err(Gemma4LoadError::ConfigMismatch(format!(
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
) -> Result<RmsNorm<B>, Gemma4LoadError> {
    let weight: Tensor<B, 1> = file.load_f32(name, device)?;

    let [size] = weight.dims();
    if size != hidden_size {
        return Err(Gemma4LoadError::ConfigMismatch(format!(
            "{}: expected [{}], got [{}]",
            name, hidden_size, size
        )));
    }

    Ok(RmsNorm::from_weight(weight, eps))
}

fn load_gemma4_layer<B: Backend>(
    file: &SafeTensorFile,
    layer_idx: usize,
    config: &Gemma4Config,
    device: &B::Device,
) -> Result<Gemma4Layer<B>, Gemma4LoadError> {
    let prefix = format!("model.layers.{}", layer_idx);

    let attention = load_gemma4_attention(file, &prefix, config, device)?;

    let ffn = if config.is_moe_layer(layer_idx) {
        Gemma4Ffn::Moe(load_gemma4_moe(file, &prefix, config, device)?)
    } else {
        Gemma4Ffn::Dense(load_gemma4_dense_ffn(file, &prefix, config, device)?)
    };

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

    let use_sliding_window = attention.head_dim < config.head_dim;

    Ok(Gemma4Layer {
        attention,
        ffn,
        input_norm,
        post_attention_norm,
        pre_ffn_norm,
        post_ffn_norm,
        use_sliding_window,
        layer_output_scale: 1.0,
    })
}

fn load_gemma4_attention<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &Gemma4Config,
    device: &B::Device,
) -> Result<Gemma4Attention<B>, Gemma4LoadError> {
    let kv_dim = config.head_dim * config.num_kv_heads;
    let q_dim = config.head_dim * config.num_heads;

    let q_proj = load_linear(
        file,
        &format!("{}.self_attn.q_proj.weight", prefix),
        config.hidden_size,
        q_dim,
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
        q_dim,
        config.hidden_size,
        device,
    )?;

    let q_norm = load_rmsnorm(
        file,
        &format!("{}.self_attn.q_norm.weight", prefix),
        config.head_dim,
        config.norm_eps,
        device,
    )?;
    let k_norm = load_rmsnorm(
        file,
        &format!("{}.self_attn.k_norm.weight", prefix),
        config.head_dim,
        config.norm_eps,
        device,
    )?;

    Ok(Gemma4Attention {
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm,
        k_norm,
        num_heads: config.num_heads,
        num_kv_heads: config.num_kv_heads,
        head_dim: config.head_dim,
    })
}

fn load_gemma4_dense_ffn<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &Gemma4Config,
    device: &B::Device,
) -> Result<Gemma4DenseFfn<B>, Gemma4LoadError> {
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

    Ok(Gemma4DenseFfn {
        gate_proj,
        up_proj,
        down_proj,
    })
}

fn load_gemma4_moe<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &Gemma4Config,
    device: &B::Device,
) -> Result<Gemma4MoE<B>, Gemma4LoadError> {
    // Load router
    let router_weight: Tensor<B, 2> = file.load_f32(
        &format!("{}.block_sparse_moe.router.weight", prefix),
        device,
    )?;

    let mut router = LinearConfig::new(config.hidden_size, config.num_experts)
        .with_bias(false)
        .init(device);
    router.weight = Param::from_tensor(router_weight);

    // Load routed experts (use expert_intermediate_size)
    let mut experts = Vec::with_capacity(config.num_experts);
    for e in 0..config.num_experts {
        let expert = load_expert_ffn(
            file,
            &format!("{}.block_sparse_moe.experts.{}", prefix, e),
            config.hidden_size,
            config.expert_intermediate_size,
            device,
        )?;
        experts.push(expert);
    }

    // Load shared experts (use intermediate_size — shared/dense FFN size)
    let mut shared_experts = Vec::with_capacity(config.num_shared_experts);
    if config.num_shared_experts == 1 {
        let shared = load_expert_ffn(
            file,
            &format!("{}.block_sparse_moe.shared_expert", prefix),
            config.hidden_size,
            config.intermediate_size,
            device,
        )?;
        shared_experts.push(shared);
    } else {
        for e in 0..config.num_shared_experts {
            let shared = load_expert_ffn(
                file,
                &format!("{}.block_sparse_moe.shared_experts.{}", prefix, e),
                config.hidden_size,
                config.intermediate_size,
                device,
            )?;
            shared_experts.push(shared);
        }
    }

    Ok(Gemma4MoE {
        router,
        pre_ffn_norm_moe: None,
        router_extra_scale: None,
        post_ffn_norm_shared: None,
        post_ffn_norm_routed: None,
        expert_down_scale: None,
        experts: ExpertWeights::Full(experts),
        shared_experts,
        top_k: config.num_experts_per_tok,
        num_experts: config.num_experts,
    })
}

fn load_expert_ffn<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    hidden_size: usize,
    intermediate_size: usize,
    device: &B::Device,
) -> Result<Gemma4ExpertFfn<B>, Gemma4LoadError> {
    let gate_proj = load_linear(
        file,
        &format!("{}.gate_proj.weight", prefix),
        hidden_size,
        intermediate_size,
        device,
    )?;

    let up_proj = load_linear(
        file,
        &format!("{}.up_proj.weight", prefix),
        hidden_size,
        intermediate_size,
        device,
    )?;

    let down_proj = load_linear(
        file,
        &format!("{}.down_proj.weight", prefix),
        intermediate_size,
        hidden_size,
        device,
    )?;

    Ok(Gemma4ExpertFfn {
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
        let err = Gemma4LoadError::MissingTensor("test".to_string());
        assert!(err.to_string().contains("test"));
    }
}
