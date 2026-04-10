//! Gemma 4 Weight Loading from GGUF
//!
//! Loads Gemma 4 model weights from GGUF files (llama.cpp quantized format).
//! Extracts model config from GGUF metadata and maps GGUF tensor names
//! to the Gemma4 model structure.

use burn::module::Param;
use burn::nn::{EmbeddingConfig, LinearConfig};
use burn::prelude::*;
use sketchpad_convert::gguf::{GgufFile, MetadataValue};
use std::path::Path;
use thiserror::Error;

use crate::gemma4::{
    Gemma4, Gemma4Attention, Gemma4Config, Gemma4DenseFfn, Gemma4ExpertFfn, Gemma4Ffn, Gemma4Layer,
    Gemma4MoE, Gemma4Runtime,
};
use sketchpad_core::rmsnorm::RmsNorm;
use sketchpad_core::rope::RotaryEmbedding;

#[derive(Error, Debug)]
pub enum Gemma4GgufLoadError {
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

/// Extract a u64 from GGUF metadata, with various integer type coercions
fn meta_u64(file: &GgufFile, key: &str) -> Result<u64, Gemma4GgufLoadError> {
    match file.metadata().get(key) {
        Some(MetadataValue::U64(v)) => Ok(*v),
        Some(MetadataValue::U32(v)) => Ok(*v as u64),
        Some(MetadataValue::I32(v)) => Ok(*v as u64),
        Some(MetadataValue::I64(v)) => Ok(*v as u64),
        Some(MetadataValue::U16(v)) => Ok(*v as u64),
        Some(MetadataValue::U8(v)) => Ok(*v as u64),
        _ => Err(Gemma4GgufLoadError::MissingMetadata(key.to_string())),
    }
}

fn meta_u64_or(file: &GgufFile, key: &str, default: u64) -> u64 {
    meta_u64(file, key).unwrap_or(default)
}

fn meta_f32(file: &GgufFile, key: &str) -> Result<f32, Gemma4GgufLoadError> {
    match file.metadata().get(key) {
        Some(MetadataValue::F32(v)) => Ok(*v),
        Some(MetadataValue::F64(v)) => Ok(*v as f32),
        _ => Err(Gemma4GgufLoadError::MissingMetadata(key.to_string())),
    }
}

fn meta_f32_or(file: &GgufFile, key: &str, default: f32) -> f32 {
    meta_f32(file, key).unwrap_or(default)
}

/// Detect the architecture prefix from GGUF metadata (e.g. "gemma4", "gemma2")
fn detect_arch(file: &GgufFile) -> String {
    match file.metadata().get("general.architecture") {
        Some(MetadataValue::String(s)) => s.clone(),
        _ => "gemma4".to_string(),
    }
}

/// Parse Gemma4Config from GGUF metadata
pub fn parse_gguf_config(file: &GgufFile) -> Result<Gemma4Config, Gemma4GgufLoadError> {
    let arch = detect_arch(file);

    let num_layers = meta_u64(file, &format!("{arch}.block_count"))? as usize;
    let hidden_size = meta_u64(file, &format!("{arch}.embedding_length"))? as usize;
    let num_heads = meta_u64(file, &format!("{arch}.attention.head_count"))? as usize;
    let num_kv_heads = meta_u64_or(
        file,
        &format!("{arch}.attention.head_count_kv"),
        num_heads as u64,
    ) as usize;
    let head_dim = meta_u64_or(
        file,
        &format!("{arch}.attention.key_length"),
        (hidden_size / num_heads) as u64,
    ) as usize;

    let num_experts = meta_u64_or(file, &format!("{arch}.expert_count"), 0) as usize;
    let num_experts_per_tok = meta_u64_or(
        file,
        &format!("{arch}.expert_used_count"),
        if num_experts > 0 { 8 } else { 1 },
    ) as usize;

    // Determine MoE layers: if experts exist, odd layers are MoE by default
    let moe_layers = if num_experts > 0 {
        (0..num_layers).filter(|i| i % 2 == 1).collect()
    } else {
        Vec::new()
    };

    // Shared experts: in GGUF, the standard FFN tensors on MoE layers serve as shared expert
    let num_shared_experts = if num_experts > 0 { 1 } else { 0 };

    Ok(Gemma4Config {
        vocab_size: meta_u64_or(file, &format!("{arch}.vocab_size"), 262144) as usize,
        hidden_size,
        intermediate_size: meta_u64_or(file, &format!("{arch}.feed_forward_length"), 24576)
            as usize,
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim,
        max_seq_len: meta_u64_or(file, &format!("{arch}.context_length"), 8192) as usize,
        sliding_window: meta_u64_or(file, &format!("{arch}.attention.sliding_window"), 1024)
            as usize,
        attn_logit_softcap: meta_f32_or(file, &format!("{arch}.attn_logit_softcapping"), 50.0),
        final_logit_softcap: meta_f32_or(file, &format!("{arch}.final_logit_softcapping"), 30.0),
        norm_eps: meta_f32_or(
            file,
            &format!("{arch}.attention.layer_norm_rms_epsilon"),
            1e-6,
        ) as f64,
        rope_base: meta_f32_or(file, &format!("{arch}.rope.freq_base"), 1_000_000.0),
        num_experts: num_experts.max(1),
        num_shared_experts,
        num_experts_per_tok,
        moe_layers,
    })
}

/// Load a Gemma 4 model from a GGUF file
pub fn load_gemma4_gguf<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<(Gemma4<B>, Gemma4Runtime<B>, Gemma4Config), Gemma4GgufLoadError> {
    let file = GgufFile::open(path)?;
    let config = parse_gguf_config(&file)?;

    let embed_tokens = load_embedding(&file, "token_embd.weight", &config, device)?;

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

    let model = Gemma4 {
        embed_tokens,
        layers,
        norm,
    };

    let runtime = Gemma4Runtime {
        rope: RotaryEmbedding::with_base(
            config.head_dim,
            config.max_seq_len,
            config.rope_base,
            device,
        ),
        config: config.clone(),
    };

    Ok((model, runtime, config))
}

fn load_embedding<B: Backend>(
    file: &GgufFile,
    name: &str,
    config: &Gemma4Config,
    device: &B::Device,
) -> Result<burn::nn::Embedding<B>, Gemma4GgufLoadError> {
    let weight: Tensor<B, 2> = file.load_f32(name, device)?;
    let mut embedding = EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device);
    embedding.weight = Param::from_tensor(weight);
    Ok(embedding)
}

fn load_linear<B: Backend>(
    file: &GgufFile,
    name: &str,
    in_features: usize,
    out_features: usize,
    device: &B::Device,
) -> Result<burn::nn::Linear<B>, Gemma4GgufLoadError> {
    if !file.contains(name) {
        return Err(Gemma4GgufLoadError::MissingTensor(name.to_string()));
    }
    let weight: Tensor<B, 2> = file.load_f32(name, device)?;
    let mut linear = LinearConfig::new(in_features, out_features)
        .with_bias(false)
        .init(device);
    linear.weight = Param::from_tensor(weight);
    Ok(linear)
}

fn load_rmsnorm<B: Backend>(
    file: &GgufFile,
    name: &str,
    hidden_size: usize,
    eps: f64,
    device: &B::Device,
) -> Result<RmsNorm<B>, Gemma4GgufLoadError> {
    let weight: Tensor<B, 1> = file.load_f32(name, device)?;
    let [size] = weight.dims();
    if size != hidden_size {
        return Err(Gemma4GgufLoadError::ShapeMismatch {
            name: name.to_string(),
            expected: vec![hidden_size],
            actual: vec![size],
        });
    }
    Ok(RmsNorm::from_weight(weight, eps))
}

fn load_layer<B: Backend>(
    file: &GgufFile,
    idx: usize,
    config: &Gemma4Config,
    device: &B::Device,
) -> Result<Gemma4Layer<B>, Gemma4GgufLoadError> {
    let attention = load_attention(file, idx, config, device)?;

    let ffn = if config.is_moe_layer(idx) {
        Gemma4Ffn::Moe(load_moe(file, idx, config, device)?)
    } else {
        Gemma4Ffn::Dense(load_dense_ffn(file, idx, config, device)?)
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

    let use_sliding_window = idx % 2 == 0;

    Ok(Gemma4Layer {
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
    config: &Gemma4Config,
    device: &B::Device,
) -> Result<Gemma4Attention<B>, Gemma4GgufLoadError> {
    let q_dim = config.head_dim * config.num_heads;
    let kv_dim = config.head_dim * config.num_kv_heads;

    Ok(Gemma4Attention {
        q_proj: load_linear(
            file,
            &format!("blk.{idx}.attn_q.weight"),
            config.hidden_size,
            q_dim,
            device,
        )?,
        k_proj: load_linear(
            file,
            &format!("blk.{idx}.attn_k.weight"),
            config.hidden_size,
            kv_dim,
            device,
        )?,
        v_proj: load_linear(
            file,
            &format!("blk.{idx}.attn_v.weight"),
            config.hidden_size,
            kv_dim,
            device,
        )?,
        o_proj: load_linear(
            file,
            &format!("blk.{idx}.attn_output.weight"),
            q_dim,
            config.hidden_size,
            device,
        )?,
        num_heads: config.num_heads,
        num_kv_heads: config.num_kv_heads,
        head_dim: config.head_dim,
    })
}

fn load_dense_ffn<B: Backend>(
    file: &GgufFile,
    idx: usize,
    config: &Gemma4Config,
    device: &B::Device,
) -> Result<Gemma4DenseFfn<B>, Gemma4GgufLoadError> {
    Ok(Gemma4DenseFfn {
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
    })
}

fn load_moe<B: Backend>(
    file: &GgufFile,
    idx: usize,
    config: &Gemma4Config,
    device: &B::Device,
) -> Result<Gemma4MoE<B>, Gemma4GgufLoadError> {
    // Router
    let router = load_linear(
        file,
        &format!("blk.{idx}.ffn_gate_inp.weight"),
        config.hidden_size,
        config.num_experts,
        device,
    )?;

    // GGUF stores experts as fused tensors:
    // ffn_gate_up_exps.weight: [num_experts, intermediate_size * 2, hidden_size]
    // ffn_down_exps.weight:    [num_experts, hidden_size, intermediate_size]
    let gate_up_name = format!("blk.{idx}.ffn_gate_up_exps.weight");
    let down_name = format!("blk.{idx}.ffn_down_exps.weight");

    let experts = if file.contains(&gate_up_name) {
        // Load fused expert tensors and split per-expert
        let gate_up_all: Tensor<B, 3> = file.load_f32(&gate_up_name, device)?;
        let down_all: Tensor<B, 3> = file.load_f32(&down_name, device)?;

        let mut experts = Vec::with_capacity(config.num_experts);
        for e in 0..config.num_experts {
            // Slice out this expert's weights
            let gate_up = gate_up_all
                .clone()
                .slice([
                    e..e + 1,
                    0..config.intermediate_size * 2,
                    0..config.hidden_size,
                ])
                .reshape([config.intermediate_size * 2, config.hidden_size]);

            // Split gate and up from fused tensor
            let gate_weight = gate_up
                .clone()
                .slice([0..config.intermediate_size, 0..config.hidden_size]);
            let up_weight = gate_up.slice([
                config.intermediate_size..config.intermediate_size * 2,
                0..config.hidden_size,
            ]);

            let down_weight = down_all
                .clone()
                .slice([e..e + 1, 0..config.hidden_size, 0..config.intermediate_size])
                .reshape([config.hidden_size, config.intermediate_size]);

            let mut gate_proj = LinearConfig::new(config.hidden_size, config.intermediate_size)
                .with_bias(false)
                .init(device);
            gate_proj.weight = Param::from_tensor(gate_weight);

            let mut up_proj = LinearConfig::new(config.hidden_size, config.intermediate_size)
                .with_bias(false)
                .init(device);
            up_proj.weight = Param::from_tensor(up_weight);

            let mut down_proj = LinearConfig::new(config.intermediate_size, config.hidden_size)
                .with_bias(false)
                .init(device);
            down_proj.weight = Param::from_tensor(down_weight);

            experts.push(Gemma4ExpertFfn {
                gate_proj,
                up_proj,
                down_proj,
            });
        }
        experts
    } else {
        // Fallback: try per-expert tensors (unlikely for GGUF but handle gracefully)
        let mut experts = Vec::with_capacity(config.num_experts);
        for e in 0..config.num_experts {
            let expert = Gemma4ExpertFfn {
                gate_proj: load_linear(
                    file,
                    &format!("blk.{idx}.ffn_gate.{e}.weight"),
                    config.hidden_size,
                    config.intermediate_size,
                    device,
                )?,
                up_proj: load_linear(
                    file,
                    &format!("blk.{idx}.ffn_up.{e}.weight"),
                    config.hidden_size,
                    config.intermediate_size,
                    device,
                )?,
                down_proj: load_linear(
                    file,
                    &format!("blk.{idx}.ffn_down.{e}.weight"),
                    config.intermediate_size,
                    config.hidden_size,
                    device,
                )?,
            };
            experts.push(expert);
        }
        experts
    };

    // Shared expert: uses the standard FFN tensors on MoE layers
    let shared_expert = Gemma4ExpertFfn {
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

    Ok(Gemma4MoE {
        router,
        experts,
        shared_experts: vec![shared_expert],
        top_k: config.num_experts_per_tok,
        num_experts: config.num_experts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_arch_default() {
        // Without a file, just test the error types compile
        let err = Gemma4GgufLoadError::MissingTensor("test".to_string());
        assert!(err.to_string().contains("test"));
    }

    #[test]
    fn test_shape_mismatch_display() {
        let err = Gemma4GgufLoadError::ShapeMismatch {
            name: "weight".to_string(),
            expected: vec![128],
            actual: vec![256],
        };
        assert!(err.to_string().contains("weight"));
        assert!(err.to_string().contains("128"));
    }
}
