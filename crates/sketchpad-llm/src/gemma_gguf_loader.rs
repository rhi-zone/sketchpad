//! Gemma 2 Weight Loading from GGUF
//!
//! Loads Gemma 2 model weights from GGUF files (llama.cpp quantized format).
//! Extracts model config from GGUF metadata and maps GGUF tensor names
//! to the Gemma model structure.

use burn::module::Param;
use burn::nn::{EmbeddingConfig, LinearConfig};
use burn::prelude::*;
use sketchpad_convert::gguf::{GgufFile, MetadataValue};
use std::path::Path;
use thiserror::Error;

use crate::gemma::{Gemma, GemmaAttention, GemmaConfig, GemmaFfn, GemmaLayer, GemmaRuntime};
use sketchpad_core::rmsnorm::RmsNorm;
use sketchpad_core::rope::RotaryEmbedding;

#[derive(Error, Debug)]
pub enum GemmaGgufLoadError {
    #[error("GGUF error: {0}")]
    Gguf(#[from] sketchpad_convert::gguf::GgufError),

    #[error("Missing metadata key: {0}")]
    MissingMetadata(String),

    #[error("Missing tensor: {0}")]
    MissingTensor(String),

    #[error("Shape mismatch for {name}: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

fn meta_u64(file: &GgufFile, key: &str) -> Result<u64, GemmaGgufLoadError> {
    match file.metadata().get(key) {
        Some(MetadataValue::U64(v)) => Ok(*v),
        Some(MetadataValue::U32(v)) => Ok(*v as u64),
        Some(MetadataValue::I32(v)) => Ok(*v as u64),
        Some(MetadataValue::I64(v)) => Ok(*v as u64),
        Some(MetadataValue::U16(v)) => Ok(*v as u64),
        Some(MetadataValue::U8(v)) => Ok(*v as u64),
        _ => Err(GemmaGgufLoadError::MissingMetadata(key.to_string())),
    }
}

fn meta_u64_or(file: &GgufFile, key: &str, default: u64) -> u64 {
    meta_u64(file, key).unwrap_or(default)
}

fn meta_f32(file: &GgufFile, key: &str) -> Result<f32, GemmaGgufLoadError> {
    match file.metadata().get(key) {
        Some(MetadataValue::F32(v)) => Ok(*v),
        Some(MetadataValue::F64(v)) => Ok(*v as f32),
        _ => Err(GemmaGgufLoadError::MissingMetadata(key.to_string())),
    }
}

fn meta_f32_or(file: &GgufFile, key: &str, default: f32) -> f32 {
    meta_f32(file, key).unwrap_or(default)
}

/// Parse GemmaConfig from GGUF metadata (arch = `gemma2`)
pub fn parse_gguf_config(file: &GgufFile) -> Result<GemmaConfig, GemmaGgufLoadError> {
    let arch = match file.metadata().get("general.architecture") {
        Some(MetadataValue::String(s)) => s.clone(),
        _ => "gemma2".to_string(),
    };

    let num_layers = meta_u64(file, &format!("{arch}.block_count"))? as usize;
    let hidden_size = meta_u64(file, &format!("{arch}.embedding_length"))? as usize;
    let num_heads = meta_u64(file, &format!("{arch}.attention.head_count"))? as usize;
    let num_kv_heads = meta_u64_or(
        file,
        &format!("{arch}.attention.head_count_kv"),
        num_heads as u64,
    ) as usize;
    let intermediate_size =
        meta_u64_or(file, &format!("{arch}.feed_forward_length"), 9216) as usize;

    Ok(GemmaConfig {
        vocab_size: meta_u64_or(file, &format!("{arch}.vocab_size"), 256000) as usize,
        hidden_size,
        intermediate_size,
        num_layers,
        num_heads,
        num_kv_heads,
        max_seq_len: meta_u64_or(file, &format!("{arch}.context_length"), 8192) as usize,
        sliding_window: meta_u64_or(file, &format!("{arch}.attention.sliding_window"), 4096)
            as usize,
        attn_logit_softcap: meta_f32_or(file, &format!("{arch}.attn_logit_softcapping"), 50.0),
        final_logit_softcap: meta_f32_or(file, &format!("{arch}.final_logit_softcapping"), 30.0),
        norm_eps: meta_f32_or(
            file,
            &format!("{arch}.attention.layer_norm_rms_epsilon"),
            1e-6,
        ) as f64,
        rope_base: meta_f32_or(file, &format!("{arch}.rope.freq_base"), 10000.0),
    })
}

/// Load a Gemma 2 model from a GGUF file
pub fn load_gemma_gguf<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<(Gemma<B>, GemmaRuntime<B>, GemmaConfig), GemmaGgufLoadError> {
    let file = GgufFile::open(path)?;
    let config = parse_gguf_config(&file)?;

    let embed_weight: Tensor<B, 2> = file.load_f32("token_embd.weight", device)?;
    let mut embed_tokens = EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device);
    embed_tokens.weight = Param::from_tensor(embed_weight);

    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let layer = load_layer(&file, i, &config, device)?;
        layers.push(layer);
    }

    let norm = load_rmsnorm(
        &file,
        "output_norm.weight",
        config.hidden_size,
        config.norm_eps,
        device,
    )?;

    let head_dim = config.hidden_size / config.num_heads;
    let model = Gemma {
        embed_tokens,
        layers,
        norm,
    };
    let runtime = GemmaRuntime {
        rope: RotaryEmbedding::with_base(head_dim, config.max_seq_len, config.rope_base, device),
        config: config.clone(),
    };

    Ok((model, runtime, config))
}

fn load_rmsnorm<B: Backend>(
    file: &GgufFile,
    name: &str,
    hidden_size: usize,
    eps: f64,
    device: &B::Device,
) -> Result<RmsNorm<B>, GemmaGgufLoadError> {
    let weight: Tensor<B, 1> = file.load_f32(name, device)?;
    let [size] = weight.dims();
    if size != hidden_size {
        return Err(GemmaGgufLoadError::ShapeMismatch {
            name: name.to_string(),
            expected: vec![hidden_size],
            actual: vec![size],
        });
    }
    Ok(RmsNorm::from_weight(weight, eps))
}

fn load_linear<B: Backend>(
    file: &GgufFile,
    name: &str,
    in_features: usize,
    out_features: usize,
    device: &B::Device,
) -> Result<burn::nn::Linear<B>, GemmaGgufLoadError> {
    if !file.contains(name) {
        return Err(GemmaGgufLoadError::MissingTensor(name.to_string()));
    }
    let weight: Tensor<B, 2> = file.load_f32(name, device)?;
    // GGUF shape (after reversal) is [out_features, in_features]; transpose for Burn Linear
    let weight = weight.transpose();
    let mut linear = LinearConfig::new(in_features, out_features)
        .with_bias(false)
        .init(device);
    linear.weight = Param::from_tensor(weight);
    Ok(linear)
}

fn load_layer<B: Backend>(
    file: &GgufFile,
    idx: usize,
    config: &GemmaConfig,
    device: &B::Device,
) -> Result<GemmaLayer<B>, GemmaGgufLoadError> {
    let attention = load_attention(file, idx, config, device)?;

    let ffn = GemmaFfn {
        gate_proj: load_linear(
            file,
            &format!("blk.{idx}.ffn_gate.weight"),
            config.hidden_size,
            config.intermediate_size,
            device,
        )?,
        up_proj: load_linear(
            file,
            &format!("blk.{idx}.ffn_up.weight"),
            config.hidden_size,
            config.intermediate_size,
            device,
        )?,
        down_proj: load_linear(
            file,
            &format!("blk.{idx}.ffn_down.weight"),
            config.intermediate_size,
            config.hidden_size,
            device,
        )?,
    };

    let input_norm = load_rmsnorm(
        file,
        &format!("blk.{idx}.attn_norm.weight"),
        config.hidden_size,
        config.norm_eps,
        device,
    )?;
    let post_attention_norm = load_rmsnorm(
        file,
        &format!("blk.{idx}.post_attention_norm.weight"),
        config.hidden_size,
        config.norm_eps,
        device,
    )?;
    let pre_ffn_norm = load_rmsnorm(
        file,
        &format!("blk.{idx}.ffn_norm.weight"),
        config.hidden_size,
        config.norm_eps,
        device,
    )?;
    let post_ffn_norm = load_rmsnorm(
        file,
        &format!("blk.{idx}.post_ffw_norm.weight"),
        config.hidden_size,
        config.norm_eps,
        device,
    )?;

    // Gemma 2: even-indexed layers use sliding window, odd use global attention
    let use_sliding_window = idx % 2 == 0;

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

fn load_attention<B: Backend>(
    file: &GgufFile,
    idx: usize,
    config: &GemmaConfig,
    device: &B::Device,
) -> Result<GemmaAttention<B>, GemmaGgufLoadError> {
    let head_dim = config.hidden_size / config.num_heads;
    let kv_dim = head_dim * config.num_kv_heads;

    let q_proj = load_linear(
        file,
        &format!("blk.{idx}.attn_q.weight"),
        config.hidden_size,
        config.hidden_size,
        device,
    )?;
    let k_proj = load_linear(
        file,
        &format!("blk.{idx}.attn_k.weight"),
        config.hidden_size,
        kv_dim,
        device,
    )?;
    let v_proj = load_linear(
        file,
        &format!("blk.{idx}.attn_v.weight"),
        config.hidden_size,
        kv_dim,
        device,
    )?;
    let o_proj = load_linear(
        file,
        &format!("blk.{idx}.attn_output.weight"),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = GemmaGgufLoadError::MissingTensor("test".to_string());
        assert!(err.to_string().contains("test"));
    }
}
