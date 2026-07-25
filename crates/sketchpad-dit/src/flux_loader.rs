//! Flux Weight Loading from Safetensors
//!
//! Loads Flux model weights from HuggingFace diffusers format safetensors files.
//!
//! # Supported Formats
//!
//! - HuggingFace Diffusers format (flux1-dev, flux1-schnell)
//! - Black Forest Labs format

use std::path::Path;

use burn::module::Param;
use burn::nn::LinearConfig;
use burn::prelude::*;
use sketchpad_convert::loader::SafeTensorFile;
use thiserror::Error;

use crate::flux::{
    FinalLayer, Flux, FluxAttention, FluxConfig, FluxDoubleBlock, FluxRuntime, FluxSingleBlock,
    TimestepEmbedding,
};
use sketchpad_core::dit::PatchEmbed;
use sketchpad_core::glu::SwiGluFfn;
use sketchpad_core::layernorm::LayerNorm;
use sketchpad_core::rope::RotaryEmbedding;

#[derive(Error, Debug)]
pub enum FluxLoadError {
    #[error("Load error: {0}")]
    Load(#[from] sketchpad_convert::loader::LoadError),

    #[error("Missing tensor: {0}")]
    MissingTensor(String),

    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Load Flux model from safetensors file
///
/// # Arguments
///
/// * `path` - Path to the transformer safetensors file
/// * `config` - Model configuration
/// * `device` - Device to load onto
pub fn load_flux<B: Backend, P: AsRef<Path>>(
    path: P,
    config: &FluxConfig,
    device: &B::Device,
) -> Result<(Flux<B>, FluxRuntime<B>), FluxLoadError> {
    let file = SafeTensorFile::open(path)?;

    // Load embeddings
    let img_embed = load_patch_embed(&file, "img_embed", config, device)?;
    let txt_embed = load_linear(
        &file,
        "txt_embed",
        config.text_dim,
        config.hidden_size,
        device,
    )?;

    // Load time embedding
    let time_embed = load_timestep_embedding(
        &file,
        "time_embed",
        config.time_dim,
        config.hidden_size,
        device,
    )?;

    // Load guidance embedding (optional for schnell)
    let guidance_embed = if file.contains("guidance_embed.linear1.weight") {
        Some(load_timestep_embedding(
            &file,
            "guidance_embed",
            config.guidance_dim,
            config.hidden_size,
            device,
        )?)
    } else {
        None
    };

    // Load double blocks
    let mut double_blocks = Vec::with_capacity(config.num_double_blocks);
    for i in 0..config.num_double_blocks {
        let block = load_double_block(&file, i, config, device)?;
        double_blocks.push(block);
    }

    // Load single blocks
    let mut single_blocks = Vec::with_capacity(config.num_single_blocks);
    for i in 0..config.num_single_blocks {
        let block = load_single_block(&file, i, config, device)?;
        single_blocks.push(block);
    }

    // Load final layer
    let final_layer = load_final_layer(&file, config, device)?;

    let model = Flux {
        img_embed,
        txt_embed,
        time_embed,
        guidance_embed,
        double_blocks,
        single_blocks,
        final_layer,
        patch_size: config.patch_size,
        in_channels: config.in_channels,
    };

    let head_dim = config.hidden_size / config.num_heads;
    let runtime = FluxRuntime {
        rope: RotaryEmbedding::new(head_dim, config.max_seq_len, device),
        config: config.clone(),
    };

    Ok((model, runtime))
}

fn load_linear<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    in_features: usize,
    out_features: usize,
    device: &B::Device,
) -> Result<burn::nn::Linear<B>, FluxLoadError> {
    let weight_name = format!("{}.weight", prefix);
    let bias_name = format!("{}.bias", prefix);

    let weight: Tensor<B, 2> = file.load_f32(&weight_name, device)?;

    let has_bias = file.contains(&bias_name);
    let mut linear = LinearConfig::new(in_features, out_features)
        .with_bias(has_bias)
        .init(device);

    linear.weight = Param::from_tensor(weight);

    if has_bias {
        let bias: Tensor<B, 1> = file.load_f32(&bias_name, device)?;
        linear.bias = Some(Param::from_tensor(bias));
    }

    Ok(linear)
}

fn load_patch_embed<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &FluxConfig,
    device: &B::Device,
) -> Result<PatchEmbed<B>, FluxLoadError> {
    let patch_dim = config.patch_size * config.patch_size * config.in_channels;
    let proj = load_linear(
        file,
        &format!("{}.proj", prefix),
        patch_dim,
        config.hidden_size,
        device,
    )?;

    Ok(PatchEmbed {
        proj,
        patch_size: config.patch_size,
        in_channels: config.in_channels,
    })
}

fn load_timestep_embedding<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    embed_dim: usize,
    hidden_size: usize,
    device: &B::Device,
) -> Result<TimestepEmbedding<B>, FluxLoadError> {
    let linear1 = load_linear(
        file,
        &format!("{}.linear1", prefix),
        embed_dim,
        hidden_size,
        device,
    )?;
    let linear2 = load_linear(
        file,
        &format!("{}.linear2", prefix),
        hidden_size,
        hidden_size,
        device,
    )?;

    Ok(TimestepEmbedding {
        linear1,
        linear2,
        freqs: super::flux::timestep_freqs(embed_dim, device),
    })
}

fn load_layernorm<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    hidden_size: usize,
    device: &B::Device,
) -> Result<LayerNorm<B>, FluxLoadError> {
    let weight_name = format!("{}.weight", prefix);
    let bias_name = format!("{}.bias", prefix);

    let weight: Tensor<B, 1> = file.load_f32(&weight_name, device)?;
    let bias: Tensor<B, 1> = if file.contains(&bias_name) {
        file.load_f32(&bias_name, device)?
    } else {
        Tensor::zeros([hidden_size], device)
    };

    Ok(LayerNorm::from_weight_bias(weight, bias))
}

fn load_attention<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &FluxConfig,
    device: &B::Device,
) -> Result<FluxAttention<B>, FluxLoadError> {
    let head_dim = config.hidden_size / config.num_heads;

    let qkv = load_linear(
        file,
        &format!("{}.qkv", prefix),
        config.hidden_size,
        3 * config.hidden_size,
        device,
    )?;
    let proj = load_linear(
        file,
        &format!("{}.proj", prefix),
        config.hidden_size,
        config.hidden_size,
        device,
    )?;

    Ok(FluxAttention {
        qkv,
        proj,
        num_heads: config.num_heads,
        head_dim,
    })
}

fn load_swiglu_ffn<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    hidden_size: usize,
    intermediate_size: usize,
    device: &B::Device,
) -> Result<SwiGluFfn<B>, FluxLoadError> {
    let gate_proj = load_linear(
        file,
        &format!("{}.gate_proj", prefix),
        hidden_size,
        intermediate_size,
        device,
    )?;
    let up_proj = load_linear(
        file,
        &format!("{}.up_proj", prefix),
        hidden_size,
        intermediate_size,
        device,
    )?;
    let down_proj = load_linear(
        file,
        &format!("{}.down_proj", prefix),
        intermediate_size,
        hidden_size,
        device,
    )?;

    Ok(SwiGluFfn {
        gate_proj,
        up_proj,
        down_proj,
    })
}

fn load_double_block<B: Backend>(
    file: &SafeTensorFile,
    idx: usize,
    config: &FluxConfig,
    device: &B::Device,
) -> Result<FluxDoubleBlock<B>, FluxLoadError> {
    let prefix = format!("double_blocks.{}", idx);
    let intermediate_size = (config.hidden_size as f32 * config.mlp_ratio) as usize;

    Ok(FluxDoubleBlock {
        img_norm1: load_layernorm(
            file,
            &format!("{}.img_norm1", prefix),
            config.hidden_size,
            device,
        )?,
        img_attn: load_attention(file, &format!("{}.img_attn", prefix), config, device)?,
        img_norm2: load_layernorm(
            file,
            &format!("{}.img_norm2", prefix),
            config.hidden_size,
            device,
        )?,
        img_ffn: load_swiglu_ffn(
            file,
            &format!("{}.img_ffn", prefix),
            config.hidden_size,
            intermediate_size,
            device,
        )?,

        txt_norm1: load_layernorm(
            file,
            &format!("{}.txt_norm1", prefix),
            config.hidden_size,
            device,
        )?,
        txt_attn: load_attention(file, &format!("{}.txt_attn", prefix), config, device)?,
        txt_norm2: load_layernorm(
            file,
            &format!("{}.txt_norm2", prefix),
            config.hidden_size,
            device,
        )?,
        txt_ffn: load_swiglu_ffn(
            file,
            &format!("{}.txt_ffn", prefix),
            config.hidden_size,
            intermediate_size,
            device,
        )?,

        modulation: load_linear(
            file,
            &format!("{}.modulation", prefix),
            config.hidden_size,
            6 * config.hidden_size,
            device,
        )?,
    })
}

fn load_single_block<B: Backend>(
    file: &SafeTensorFile,
    idx: usize,
    config: &FluxConfig,
    device: &B::Device,
) -> Result<FluxSingleBlock<B>, FluxLoadError> {
    let prefix = format!("single_blocks.{}", idx);
    let intermediate_size = (config.hidden_size as f32 * config.mlp_ratio) as usize;

    Ok(FluxSingleBlock {
        norm: load_layernorm(
            file,
            &format!("{}.norm", prefix),
            config.hidden_size,
            device,
        )?,
        attn: load_attention(file, &format!("{}.attn", prefix), config, device)?,
        ffn: load_swiglu_ffn(
            file,
            &format!("{}.ffn", prefix),
            config.hidden_size,
            intermediate_size,
            device,
        )?,
        modulation: load_linear(
            file,
            &format!("{}.modulation", prefix),
            config.hidden_size,
            3 * config.hidden_size,
            device,
        )?,
    })
}

fn load_final_layer<B: Backend>(
    file: &SafeTensorFile,
    config: &FluxConfig,
    device: &B::Device,
) -> Result<FinalLayer<B>, FluxLoadError> {
    let out_dim = config.patch_size * config.patch_size * config.in_channels;

    Ok(FinalLayer {
        norm: load_layernorm(file, "final_layer.norm", config.hidden_size, device)?,
        proj: load_linear(
            file,
            "final_layer.proj",
            config.hidden_size,
            out_dim,
            device,
        )?,
        ada_ln_modulation: load_linear(
            file,
            "final_layer.ada_ln_modulation",
            config.hidden_size,
            2 * config.hidden_size,
            device,
        )?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = FluxLoadError::MissingTensor("test".to_string());
        assert!(err.to_string().contains("test"));
    }
}
