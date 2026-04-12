//! Zamba/Zamba2 Weight Loading from Safetensors
//!
//! Loads Zamba hybrid Mamba-backbone model weights from HuggingFace-format safetensors files.
//! Compatible with Zyphra's Zamba and Zamba2 models.
//!
//! Weight layout (HuggingFace):
//! - `model.embed_tokens.weight`
//! - `model.layers.{i}.input_layernorm.weight`
//! - `model.layers.{i}.mamba.{in_proj,conv1d,x_proj,dt_proj,A_log,D,out_proj}.{weight,bias}`
//! - `model.layers.{i}.pre_ff_layernorm.weight`
//! - `model.layers.{i}.mlp.{gate_proj,up_proj,down_proj}.weight`
//! - `model.shared_transformer.self_attn.{q,k,v,o}_proj.weight` (shared once)
//! - `model.layers.{i}.self_attn.adapter_down.weight` (Zamba2 LoRA A)
//! - `model.layers.{i}.self_attn.adapter_up.weight` (Zamba2 LoRA B)
//! - `model.norm.weight`
//! - `lm_head.weight`

use std::path::Path;

use burn::module::Param;
use burn::nn::conv::Conv1dConfig;
use burn::nn::{EmbeddingConfig, LinearConfig};
use burn::prelude::*;
use sketchpad_convert::loader::SafeTensorFile;
use sketchpad_core::rmsnorm::RmsNorm;
use thiserror::Error;

use crate::zamba::{
    Zamba, ZambaBlock, ZambaConfig, ZambaLoraAdapter, ZambaMamba, ZambaMlp, ZambaRuntime,
    ZambaSharedAttention,
};

#[derive(Error, Debug)]
pub enum ZambaLoadError {
    #[error("Load error: {0}")]
    Load(#[from] sketchpad_convert::loader::LoadError),

    #[error("Missing tensor: {0}")]
    MissingTensor(String),

    #[error("Config mismatch: {0}")]
    ConfigMismatch(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Load Zamba or Zamba2 weights from a safetensors file.
///
/// # Arguments
///
/// * `path` - Path to the safetensors file
/// * `config` - Model configuration (must match weights)
/// * `device` - Target device
pub fn load_zamba<B: Backend, P: AsRef<Path>>(
    path: P,
    config: &ZambaConfig,
    device: &B::Device,
) -> Result<(Zamba<B>, ZambaRuntime<B>), ZambaLoadError> {
    let file = SafeTensorFile::open(path)?;

    // Embeddings
    let embed_weight: Tensor<B, 2> = file.load_f32("model.embed_tokens.weight", device)?;
    let [vocab, dim] = embed_weight.dims();
    let mut embed_tokens = EmbeddingConfig::new(vocab, dim).init(device);
    embed_tokens.weight = Param::from_tensor(embed_weight);

    // Blocks
    let layers: Vec<ZambaBlock<B>> = (0..config.num_layers)
        .map(|i| load_block(&file, i, config, device))
        .collect::<Result<_, _>>()?;

    // Final norm
    let ln_f = load_rmsnorm(&file, "model.norm", config.d_model, config.norm_eps, device)?;

    // LM head
    let lm_head = load_linear(
        &file,
        "lm_head.weight",
        None,
        config.d_model,
        config.vocab_size,
        device,
    )?;

    let model = Zamba {
        embed_tokens,
        layers,
        ln_f,
        lm_head,
    };

    // Shared attention (single set of weights)
    let shared_attn = load_shared_attention(&file, "model.shared_transformer", config, device)?;

    // LoRA adapters (Zamba2 only)
    let num_attn_layers = (0..config.num_layers)
        .filter(|&i| config.is_attention_layer(i))
        .count();
    let lora_adapters: Vec<Option<ZambaLoraAdapter<B>>> = if config.is_zamba2 {
        (0..config.num_layers)
            .filter(|&i| config.is_attention_layer(i))
            .map(|layer_idx| {
                let prefix = format!("model.layers.{}.self_attn", layer_idx);
                load_lora_adapter(&file, &prefix, config, device)
                    .map(Some)
                    // If not found, fall back to a zeroed adapter
                    .unwrap_or_else(|_| Some(ZambaLoraAdapter::new(config, device)))
            })
            .collect()
    } else {
        (0..num_attn_layers).map(|_| None).collect()
    };

    let runtime = ZambaRuntime {
        config: config.clone(),
        shared_attn,
        lora_adapters,
    };

    Ok((model, runtime))
}

fn load_block<B: Backend>(
    file: &SafeTensorFile,
    layer_idx: usize,
    config: &ZambaConfig,
    device: &B::Device,
) -> Result<ZambaBlock<B>, ZambaLoadError> {
    let prefix = format!("model.layers.{}", layer_idx);

    let pre_norm = load_rmsnorm(
        file,
        &format!("{}.input_layernorm", prefix),
        config.d_model,
        config.norm_eps,
        device,
    )?;

    let mamba = load_mamba(file, &prefix, config, device)?;

    let mlp_norm = load_rmsnorm(
        file,
        &format!("{}.pre_ff_layernorm", prefix),
        config.d_model,
        config.norm_eps,
        device,
    )?;

    let mlp = load_mlp(file, &prefix, config, device)?;

    Ok(ZambaBlock {
        pre_norm,
        mamba,
        mlp_norm,
        mlp,
        is_attn: config.is_attention_layer(layer_idx),
    })
}

fn load_mamba<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &ZambaConfig,
    device: &B::Device,
) -> Result<ZambaMamba<B>, ZambaLoadError> {
    let d_inner = config.d_inner();
    let d_state = config.d_state;
    let dt_rank = config.dt_rank();
    let mp = format!("{}.mamba", prefix);

    let in_proj = load_linear(
        file,
        &format!("{}.in_proj.weight", mp),
        None,
        config.d_model,
        d_inner * 2,
        device,
    )?;

    // Conv1d weight: [d_inner, 1, d_conv], bias: [d_inner]
    let conv_weight: Tensor<B, 3> = file.load_f32(&format!("{}.conv1d.weight", mp), device)?;
    let conv_bias: Tensor<B, 1> = file.load_f32(&format!("{}.conv1d.bias", mp), device)?;
    let mut conv1d = Conv1dConfig::new(d_inner, d_inner, config.d_conv)
        .with_groups(d_inner)
        .with_padding(burn::nn::PaddingConfig1d::Explicit(config.d_conv - 1))
        .with_bias(true)
        .init(device);
    conv1d.weight = Param::from_tensor(conv_weight);
    conv1d.bias = Some(Param::from_tensor(conv_bias));

    let x_proj = load_linear(
        file,
        &format!("{}.x_proj.weight", mp),
        None,
        d_inner,
        dt_rank + d_state * 2,
        device,
    )?;

    let dt_proj = load_linear(
        file,
        &format!("{}.dt_proj.weight", mp),
        Some(&format!("{}.dt_proj.bias", mp)),
        dt_rank,
        d_inner,
        device,
    )?;

    let a_log: Tensor<B, 2> = file.load_f32(&format!("{}.A_log", mp), device)?;
    let d_param: Tensor<B, 1> = file.load_f32(&format!("{}.D", mp), device)?;

    let out_proj = load_linear(
        file,
        &format!("{}.out_proj.weight", mp),
        None,
        d_inner,
        config.d_model,
        device,
    )?;

    Ok(ZambaMamba {
        in_proj,
        conv1d,
        x_proj,
        dt_proj,
        a_log: Param::from_tensor(a_log),
        d: Param::from_tensor(d_param),
        out_proj,
        d_inner,
        d_state,
        d_conv: config.d_conv,
        dt_rank,
    })
}

fn load_mlp<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &ZambaConfig,
    device: &B::Device,
) -> Result<ZambaMlp<B>, ZambaLoadError> {
    let mp = format!("{}.mlp", prefix);

    let gate_proj = load_linear(
        file,
        &format!("{}.gate_proj.weight", mp),
        None,
        config.d_model,
        config.intermediate_size,
        device,
    )?;
    let up_proj = load_linear(
        file,
        &format!("{}.up_proj.weight", mp),
        None,
        config.d_model,
        config.intermediate_size,
        device,
    )?;
    let down_proj = load_linear(
        file,
        &format!("{}.down_proj.weight", mp),
        None,
        config.intermediate_size,
        config.d_model,
        device,
    )?;

    Ok(ZambaMlp {
        gate_proj,
        up_proj,
        down_proj,
    })
}

fn load_shared_attention<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &ZambaConfig,
    device: &B::Device,
) -> Result<ZambaSharedAttention<B>, ZambaLoadError> {
    let ap = format!("{}.self_attn", prefix);

    let q_proj = load_linear(
        file,
        &format!("{}.q_proj.weight", ap),
        None,
        config.d_model,
        config.num_heads * config.head_dim,
        device,
    )?;
    let k_proj = load_linear(
        file,
        &format!("{}.k_proj.weight", ap),
        None,
        config.d_model,
        config.num_kv_heads * config.head_dim,
        device,
    )?;
    let v_proj = load_linear(
        file,
        &format!("{}.v_proj.weight", ap),
        None,
        config.d_model,
        config.num_kv_heads * config.head_dim,
        device,
    )?;
    let o_proj = load_linear(
        file,
        &format!("{}.o_proj.weight", ap),
        None,
        config.num_heads * config.head_dim,
        config.d_model,
        device,
    )?;

    Ok(ZambaSharedAttention {
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        num_heads: config.num_heads,
        num_kv_heads: config.num_kv_heads,
        head_dim: config.head_dim,
    })
}

fn load_lora_adapter<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &ZambaConfig,
    device: &B::Device,
) -> Result<ZambaLoraAdapter<B>, ZambaLoadError> {
    let rank = config.lora_rank;
    let scale = 1.0f32 / rank as f32;

    let lora_a = load_linear(
        file,
        &format!("{}.adapter_down.weight", prefix),
        None,
        config.d_model,
        rank,
        device,
    )?;
    let lora_b = load_linear(
        file,
        &format!("{}.adapter_up.weight", prefix),
        None,
        rank,
        config.d_model,
        device,
    )?;

    Ok(ZambaLoraAdapter {
        lora_a,
        lora_b,
        scale,
    })
}

fn load_rmsnorm<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    size: usize,
    eps: f64,
    device: &B::Device,
) -> Result<RmsNorm<B>, ZambaLoadError> {
    let weight: Tensor<B, 1> = file.load_f32(&format!("{}.weight", prefix), device)?;
    let [s] = weight.dims();
    if s != size {
        return Err(ZambaLoadError::ConfigMismatch(format!(
            "{}.weight: expected [{}], got [{}]",
            prefix, size, s
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
) -> Result<burn::nn::Linear<B>, ZambaLoadError> {
    let weight: Tensor<B, 2> = file.load_f32(weight_name, device)?;
    let [out_f, in_f] = weight.dims();
    if out_f != out_features || in_f != in_features {
        return Err(ZambaLoadError::ConfigMismatch(format!(
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
        let err = ZambaLoadError::MissingTensor("model.embed_tokens.weight".to_string());
        assert!(err.to_string().contains("model.embed_tokens.weight"));
    }
}
