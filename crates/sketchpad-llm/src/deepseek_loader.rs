//! DeepSeek Weight Loading from Safetensors
//!
//! Loads DeepSeek model weights from HuggingFace-format safetensors files.
//! Supports both standard attention (V1) and MLA attention (V2/V3).

use std::path::Path;

use burn::module::Param;
use burn::nn::{EmbeddingConfig, LinearConfig};
use burn::prelude::*;
use sketchpad_convert::loader::SafeTensorFile;
use thiserror::Error;

use crate::deepseek::{
    DeepSeek, DeepSeekAttention, DeepSeekConfig, DeepSeekLayer, DeepSeekRuntime,
};
use sketchpad_core::glu::SwiGluFfn;
use sketchpad_core::rmsnorm::RmsNorm;
use sketchpad_core::rope::RotaryEmbedding;

#[derive(Error, Debug)]
pub enum DeepSeekLoadError {
    #[error("Load error: {0}")]
    Load(#[from] sketchpad_convert::loader::LoadError),

    #[error("Missing tensor: {0}")]
    MissingTensor(String),

    #[error("Config mismatch: {0}")]
    ConfigMismatch(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Loads DeepSeek weights from a safetensors file
pub fn load_deepseek<B: Backend, P: AsRef<Path>>(
    path: P,
    config: &DeepSeekConfig,
    device: &B::Device,
) -> Result<(DeepSeek<B>, DeepSeekRuntime<B>), DeepSeekLoadError> {
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
        let layer = load_deepseek_layer(&file, i, config, device)?;
        layers.push(layer);
    }

    let norm = load_rmsnorm(
        &file,
        "model.norm.weight",
        config.hidden_size,
        config.norm_eps,
        device,
    )?;

    let lm_head = load_linear(
        &file,
        "lm_head.weight",
        None,
        config.hidden_size,
        config.vocab_size,
        device,
    )?;

    let model = DeepSeek {
        embed_tokens,
        layers,
        norm,
        lm_head,
    };

    let head_dim = if config.use_mla {
        config.qk_nope_head_dim + config.qk_rope_head_dim
    } else {
        config.hidden_size / config.num_heads
    };

    let rope_dim = if config.use_mla {
        config.qk_rope_head_dim
    } else {
        head_dim
    };

    let runtime = DeepSeekRuntime {
        rope: RotaryEmbedding::with_base(rope_dim, config.max_seq_len, config.rope_base, device),
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
) -> Result<burn::nn::Embedding<B>, DeepSeekLoadError> {
    let weight: Tensor<B, 2> = file.load_f32(name, device)?;

    let [v, h] = weight.dims();
    if v != vocab_size || h != hidden_size {
        return Err(DeepSeekLoadError::ConfigMismatch(format!(
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
) -> Result<burn::nn::Linear<B>, DeepSeekLoadError> {
    let weight: Tensor<B, 2> = file.load_f32(weight_name, device)?;

    let [out_f, in_f] = weight.dims();
    if out_f != out_features || in_f != in_features {
        return Err(DeepSeekLoadError::ConfigMismatch(format!(
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
) -> Result<RmsNorm<B>, DeepSeekLoadError> {
    let weight: Tensor<B, 1> = file.load_f32(name, device)?;

    let [size] = weight.dims();
    if size != hidden_size {
        return Err(DeepSeekLoadError::ConfigMismatch(format!(
            "{}: expected [{}], got [{}]",
            name, hidden_size, size
        )));
    }

    Ok(RmsNorm::from_weight(weight, eps))
}

fn load_deepseek_layer<B: Backend>(
    file: &SafeTensorFile,
    layer_idx: usize,
    config: &DeepSeekConfig,
    device: &B::Device,
) -> Result<DeepSeekLayer<B>, DeepSeekLoadError> {
    let prefix = format!("model.layers.{}", layer_idx);

    let attention = if config.use_mla {
        load_mla_attention(file, &prefix, config, device)?
    } else {
        load_standard_attention(file, &prefix, config, device)?
    };

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

    Ok(DeepSeekLayer {
        attention,
        ffn,
        input_norm,
        post_attention_norm,
    })
}

fn load_standard_attention<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &DeepSeekConfig,
    device: &B::Device,
) -> Result<DeepSeekAttention<B>, DeepSeekLoadError> {
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

    Ok(DeepSeekAttention {
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_a_proj: None,
        q_b_proj: None,
        kv_a_proj_with_mqa: None,
        kv_b_proj: None,
        num_heads: config.num_heads,
        num_kv_heads: config.num_kv_heads,
        head_dim,
        qk_nope_head_dim: 0,
        qk_rope_head_dim: head_dim,
        use_mla: false,
        kv_lora_rank: 0,
    })
}

fn load_mla_attention<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &DeepSeekConfig,
    device: &B::Device,
) -> Result<DeepSeekAttention<B>, DeepSeekLoadError> {
    let head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim;
    let q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim;
    let v_head_dim = config.qk_nope_head_dim;

    // Load MLA projections
    let q_a_proj = load_linear(
        file,
        &format!("{}.self_attn.q_a_proj.weight", prefix),
        None,
        config.hidden_size,
        config.q_lora_rank,
        device,
    )?;

    let q_b_proj = load_linear(
        file,
        &format!("{}.self_attn.q_b_proj.weight", prefix),
        None,
        config.q_lora_rank,
        config.num_heads * q_head_dim,
        device,
    )?;

    let kv_a_proj_with_mqa = load_linear(
        file,
        &format!("{}.self_attn.kv_a_proj_with_mqa.weight", prefix),
        None,
        config.hidden_size,
        config.kv_lora_rank + config.qk_rope_head_dim,
        device,
    )?;

    let kv_b_proj = load_linear(
        file,
        &format!("{}.self_attn.kv_b_proj.weight", prefix),
        None,
        config.kv_lora_rank,
        config.num_heads * (config.qk_nope_head_dim + v_head_dim),
        device,
    )?;

    let o_proj = load_linear(
        file,
        &format!("{}.self_attn.o_proj.weight", prefix),
        None,
        config.num_heads * v_head_dim,
        config.hidden_size,
        device,
    )?;

    // Placeholder projections (not used in MLA mode)
    let q_proj = LinearConfig::new(config.hidden_size, config.hidden_size)
        .with_bias(false)
        .init(device);
    let kv_dim = head_dim * config.num_kv_heads;
    let k_proj = LinearConfig::new(config.hidden_size, kv_dim)
        .with_bias(false)
        .init(device);
    let v_proj = LinearConfig::new(config.hidden_size, kv_dim)
        .with_bias(false)
        .init(device);

    Ok(DeepSeekAttention {
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_a_proj: Some(q_a_proj),
        q_b_proj: Some(q_b_proj),
        kv_a_proj_with_mqa: Some(kv_a_proj_with_mqa),
        kv_b_proj: Some(kv_b_proj),
        num_heads: config.num_heads,
        num_kv_heads: config.num_kv_heads,
        head_dim,
        qk_nope_head_dim: config.qk_nope_head_dim,
        qk_rope_head_dim: config.qk_rope_head_dim,
        use_mla: true,
        kv_lora_rank: config.kv_lora_rank,
    })
}

fn load_swiglu_ffn<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &DeepSeekConfig,
    device: &B::Device,
) -> Result<SwiGluFfn<B>, DeepSeekLoadError> {
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
        let err = DeepSeekLoadError::MissingTensor("test".to_string());
        assert!(err.to_string().contains("test"));
    }
}
