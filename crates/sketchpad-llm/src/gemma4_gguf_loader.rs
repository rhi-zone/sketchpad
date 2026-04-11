//! Gemma 4 Weight Loading from GGUF
//!
//! Loads Gemma 4 model weights from GGUF files (llama.cpp quantized format).
//! Extracts model config from GGUF metadata and maps GGUF tensor names
//! to the Gemma4 model structure.

use burn::module::Param;
use burn::nn::{EmbeddingConfig, LinearConfig};
use burn::prelude::*;
#[cfg(feature = "cubecl")]
use sketchpad_convert::gguf::GgmlType;
use sketchpad_convert::gguf::{GgufFile, MetadataValue};
use std::path::Path;
use thiserror::Error;

use crate::gemma4::{
    ExpertWeights, Gemma4, Gemma4Attention, Gemma4Config, Gemma4DenseFfn, Gemma4ExpertFfn,
    Gemma4Ffn, Gemma4Layer, Gemma4MoE, Gemma4Runtime,
};
use crate::quantized::{
    AttentionOffload, DenseFfnOffload, LayerOffload, QuantizedFusedExperts, QuantizedLinear,
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

    #[error("Unsupported quantization type for GPU dequantization: {0:?}")]
    UnsupportedQuantType(sketchpad_convert::gguf::GgmlType),
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
    let head_dim_swa = meta_u64_or(
        file,
        &format!("{arch}.attention.key_length_swa"),
        head_dim as u64,
    ) as usize;

    let num_experts = meta_u64_or(file, &format!("{arch}.expert_count"), 0) as usize;
    let num_experts_per_tok = meta_u64_or(
        file,
        &format!("{arch}.expert_used_count"),
        if num_experts > 0 { 8 } else { 1 },
    ) as usize;

    let intermediate_size =
        meta_u64_or(file, &format!("{arch}.feed_forward_length"), 24576) as usize;

    // Expert intermediate size may differ from shared/dense FFN size
    let expert_intermediate_size = meta_u64_or(
        file,
        &format!("{arch}.expert_feed_forward_length"),
        intermediate_size as u64,
    ) as usize;

    // Determine MoE layers by checking which layers have expert tensors
    let moe_layers = if num_experts > 0 {
        (0..num_layers)
            .filter(|i| file.contains(&format!("blk.{i}.ffn_gate_up_exps.weight")))
            .collect()
    } else {
        Vec::new()
    };

    // Shared experts: in GGUF, the standard FFN tensors on MoE layers serve as shared expert
    let num_shared_experts = if num_experts > 0 { 1 } else { 0 };

    Ok(Gemma4Config {
        vocab_size: meta_u64_or(file, &format!("{arch}.vocab_size"), 262144) as usize,
        hidden_size,
        intermediate_size,
        expert_intermediate_size,
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
        rope_base_swa: meta_f32_or(file, &format!("{arch}.rope.freq_base_swa"), 10_000.0),
        head_dim_swa,
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

    // Create per-head-dim RoPE with the correct base frequency:
    // global layers use rope_base (1e6), local/SWA layers use rope_base_swa (1e4)
    let mut ropes = std::collections::HashMap::new();
    for layer in &layers {
        let hd = layer.attention.head_dim;
        ropes.entry(hd).or_insert_with(|| {
            let base = if hd == config.head_dim {
                config.rope_base
            } else {
                config.rope_base_swa
            };
            RotaryEmbedding::with_base(hd, config.max_seq_len, base, device)
        });
    }

    let model = Gemma4 {
        embed_tokens,
        layers,
        norm,
    };

    let runtime = Gemma4Runtime {
        ropes,
        config: config.clone(),
        offloaded_layers: None,
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
    // GGUF shape (after reversal) is [out_features, in_features] but
    // Burn's Linear stores weight as [d_input, d_output], so transpose
    let weight = weight.transpose();
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

    // Global attention layers have a larger head_dim (512) than local/SWA layers (256).
    // Local layers use sliding window; global layers use full causal mask.
    let use_sliding_window = attention.head_dim < config.head_dim;

    // Per-layer output scale (used in "heretic"-style distilled models).
    // If the tensor is absent, default to 1.0 (no scaling).
    let layer_output_scale = if file.contains(&format!("blk.{idx}.layer_output_scale.weight")) {
        let t: Tensor<B, 1> =
            file.load_f32(&format!("blk.{idx}.layer_output_scale.weight"), device)?;
        t.into_scalar().elem::<f32>()
    } else {
        1.0
    };

    Ok(Gemma4Layer {
        attention,
        ffn,
        input_norm,
        post_attention_norm,
        pre_ffn_norm,
        post_ffn_norm,
        use_sliding_window,
        layer_output_scale,
    })
}

fn load_attention<B: Backend>(
    file: &GgufFile,
    idx: usize,
    config: &Gemma4Config,
    device: &B::Device,
) -> Result<Gemma4Attention<B>, Gemma4GgufLoadError> {
    let q_name = format!("blk.{idx}.attn_q.weight");
    let k_name = format!("blk.{idx}.attn_k.weight");
    let v_name = format!("blk.{idx}.attn_v.weight");
    let o_name = format!("blk.{idx}.attn_output.weight");

    // Infer per-layer attention dimensions from tensor shapes
    // Q shape: [hidden_size, q_dim] where q_dim = num_heads * head_dim
    // K shape: [hidden_size, kv_dim] where kv_dim = num_kv_heads * head_dim
    let q_info = file
        .tensor_info(&q_name)
        .ok_or_else(|| Gemma4GgufLoadError::MissingTensor(q_name.clone()))?;
    let k_info = file
        .tensor_info(&k_name)
        .ok_or_else(|| Gemma4GgufLoadError::MissingTensor(k_name.clone()))?;

    // After GGUF shape reversal, shapes are [out_features, in_features] in Burn order
    let q_dim = q_info.shape[0];
    let kv_dim = k_info.shape[0];

    // Determine head_dim from Q norm if available, else from config
    let head_dim =
        if let Some(norm_info) = file.tensor_info(&format!("blk.{idx}.attn_q_norm.weight")) {
            norm_info.shape[0]
        } else {
            config.head_dim
        };

    let num_heads = q_dim / head_dim;
    let num_kv_heads = kv_dim / head_dim;

    let q_proj = load_linear(file, &q_name, config.hidden_size, q_dim, device)?;

    // Some layers (global attention) tie V to K — no attn_v.weight exists
    // Load K weight once and share if needed
    let k_weight: Tensor<B, 2> = file.load_f32(&k_name, device)?;
    // Transpose: GGUF [out, in] → Linear [in, out]
    let k_weight = k_weight.transpose();

    let mut k_proj = LinearConfig::new(config.hidden_size, kv_dim)
        .with_bias(false)
        .init(device);
    k_proj.weight = Param::from_tensor(k_weight.clone());

    let v_proj = if file.contains(&v_name) {
        load_linear(file, &v_name, config.hidden_size, kv_dim, device)?
    } else {
        // Tie V to K
        let mut v = LinearConfig::new(config.hidden_size, kv_dim)
            .with_bias(false)
            .init(device);
        v.weight = Param::from_tensor(k_weight);
        v
    };

    let o_proj = load_linear(file, &o_name, q_dim, config.hidden_size, device)?;

    let q_norm = load_rmsnorm(
        file,
        &format!("blk.{idx}.attn_q_norm.weight"),
        head_dim,
        config.norm_eps,
        device,
    )?;
    let k_norm = load_rmsnorm(
        file,
        &format!("blk.{idx}.attn_k_norm.weight"),
        head_dim,
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
        num_heads,
        num_kv_heads,
        head_dim,
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

    let expert_inter = config.expert_intermediate_size;

    // GGUF stores experts as fused tensors:
    // ffn_gate_up_exps.weight: [hidden_size, expert_inter * 2, num_experts]
    // ffn_down_exps.weight:    [expert_inter, hidden_size, num_experts]
    // (expert dimension is last — GGML row-major with least-significant dim first)
    let gate_up_name = format!("blk.{idx}.ffn_gate_up_exps.weight");
    let down_name = format!("blk.{idx}.ffn_down_exps.weight");

    let experts = if file.contains(&gate_up_name) {
        // Keep expert weights quantized — share mmap via Arc, don't copy bytes
        let gate_up_info = file
            .tensor_info(&gate_up_name)
            .ok_or_else(|| Gemma4GgufLoadError::MissingTensor(gate_up_name.clone()))?;
        let gate_up_type = gate_up_info.ggml_type;
        let down_info = file
            .tensor_info(&down_name)
            .ok_or_else(|| Gemma4GgufLoadError::MissingTensor(down_name.clone()))?;
        let down_type = down_info.ggml_type;

        let (gate_up_offset, gate_up_len) = file.tensor_byte_range(&gate_up_name)?;
        let (down_offset, down_len) = file.tensor_byte_range(&down_name)?;

        ExpertWeights::Quantized(QuantizedFusedExperts::new(
            file.mmap_arc(),
            gate_up_offset,
            gate_up_len,
            gate_up_type,
            file.mmap_arc(),
            down_offset,
            down_len,
            down_type,
            config.hidden_size,
            expert_inter,
            config.num_experts,
        ))
    } else {
        // Fallback: try per-expert tensors (unlikely for GGUF but handle gracefully)
        let mut experts = Vec::with_capacity(config.num_experts);
        for e in 0..config.num_experts {
            let expert = Gemma4ExpertFfn {
                gate_proj: load_linear(
                    file,
                    &format!("blk.{idx}.ffn_gate.{e}.weight"),
                    config.hidden_size,
                    expert_inter,
                    device,
                )?,
                up_proj: load_linear(
                    file,
                    &format!("blk.{idx}.ffn_up.{e}.weight"),
                    config.hidden_size,
                    expert_inter,
                    device,
                )?,
                down_proj: load_linear(
                    file,
                    &format!("blk.{idx}.ffn_down.{e}.weight"),
                    expert_inter,
                    config.hidden_size,
                    device,
                )?,
            };
            experts.push(expert);
        }
        ExpertWeights::Full(experts)
    };

    // Shared expert: uses the standard FFN tensors on MoE layers (dequantized to f32)
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

    // Heretic model: separate pre-norm for routed expert path (preFfwNorm2)
    let pre_ffn_norm_moe = {
        let name = format!("blk.{idx}.pre_ffw_norm_2.weight");
        if file.contains(&name) {
            Some(load_rmsnorm(
                file,
                &name,
                config.hidden_size,
                config.norm_eps,
                device,
            )?)
        } else {
            None
        }
    };

    // Heretic model: router extra scale (ffnGateInpScale / sqrt(hidden_size))
    // Router input = pre_ffn_norm_moe(h) * router_extra_scale
    let router_extra_scale = {
        let name = format!("blk.{idx}.ffn_gate_inp.scale");
        if file.contains(&name) {
            let scale: Tensor<B, 1> = file.load_f32(&name, device)?;
            let inv_sqrt = (config.hidden_size as f32).sqrt().recip();
            Some(scale * inv_sqrt)
        } else {
            None
        }
    };

    // Heretic model: post-norm for shared expert output (ffnPostNorm1)
    let post_ffn_norm_shared = {
        let name = format!("blk.{idx}.post_ffw_norm_1.weight");
        if file.contains(&name) {
            Some(load_rmsnorm(
                file,
                &name,
                config.hidden_size,
                config.norm_eps,
                device,
            )?)
        } else {
            None
        }
    };

    // Heretic model: post-norm for routed expert output (ffnPostNorm2)
    let post_ffn_norm_routed = {
        let name = format!("blk.{idx}.post_ffw_norm_2.weight");
        if file.contains(&name) {
            Some(load_rmsnorm(
                file,
                &name,
                config.hidden_size,
                config.norm_eps,
                device,
            )?)
        } else {
            None
        }
    };

    // Heretic model: per-expert down-projection output scale (ffnDownExpsScale)
    let expert_down_scale = {
        let name = format!("blk.{idx}.ffn_down_exps.scale");
        if file.contains(&name) {
            Some(file.load_f32::<B, 1>(&name, device)?)
        } else {
            None
        }
    };

    Ok(Gemma4MoE {
        router,
        pre_ffn_norm_moe,
        router_extra_scale,
        post_ffn_norm_shared,
        post_ffn_norm_routed,
        expert_down_scale,
        experts,
        shared_experts: vec![shared_expert],
        top_k: config.num_experts_per_tok,
        num_experts: config.num_experts,
    })
}

/// Load a tensor's raw quantized bytes from the GGUF mmap into a `QuantizedLinear`.
///
/// Dimensions are read from the tensor info: shape[0] = out_features, shape[1] = in_features
/// (GGUF stores shapes in reversed order relative to Burn/PyTorch convention).
fn load_quantized_linear(
    file: &GgufFile,
    name: &str,
) -> Result<QuantizedLinear, Gemma4GgufLoadError> {
    let info = file
        .tensor_info(name)
        .ok_or_else(|| Gemma4GgufLoadError::MissingTensor(name.to_string()))?;
    let out_features = info.shape[0];
    let in_features = if info.shape.len() > 1 {
        info.shape[1]
    } else {
        1
    };
    let ggml_type = info.ggml_type;

    let (offset, len) = file.tensor_byte_range(name)?;
    let mmap = file.mmap_arc();
    let data = mmap[offset..offset + len].to_vec();

    Ok(QuantizedLinear::new(
        data,
        ggml_type,
        in_features,
        out_features,
    ))
}

/// Load CPU-offloaded attention weights for layer `idx`.
///
/// Reads raw quantized bytes from the GGUF mmap. The corresponding `Gemma4Attention`
/// fields must be initialized with 1×1 dummy `Linear<B>` weights so VRAM use is minimal.
fn load_attention_offload<B: Backend>(
    file: &GgufFile,
    idx: usize,
) -> Result<AttentionOffload<B>, Gemma4GgufLoadError> {
    let q_proj = load_quantized_linear(file, &format!("blk.{idx}.attn_q.weight"))?;
    let k_proj = load_quantized_linear(file, &format!("blk.{idx}.attn_k.weight"))?;
    let v_proj = if file.contains(&format!("blk.{idx}.attn_v.weight")) {
        load_quantized_linear(file, &format!("blk.{idx}.attn_v.weight"))?
    } else {
        // Tied V→K: copy the quantized K data
        load_quantized_linear(file, &format!("blk.{idx}.attn_k.weight"))?
    };
    let o_proj = load_quantized_linear(file, &format!("blk.{idx}.attn_output.weight"))?;
    Ok(AttentionOffload {
        q_proj: Box::new(q_proj),
        k_proj: Box::new(k_proj),
        v_proj: Box::new(v_proj),
        o_proj: Box::new(o_proj),
    })
}

/// Load CPU-offloaded dense FFN weights for layer `idx`.
fn load_dense_ffn_offload<B: Backend>(
    file: &GgufFile,
    idx: usize,
) -> Result<DenseFfnOffload<B>, Gemma4GgufLoadError> {
    Ok(DenseFfnOffload {
        gate_proj: Box::new(load_quantized_linear(
            file,
            &format!("blk.{idx}.ffn_gate.weight"),
        )?),
        up_proj: Box::new(load_quantized_linear(
            file,
            &format!("blk.{idx}.ffn_up.weight"),
        )?),
        down_proj: Box::new(load_quantized_linear(
            file,
            &format!("blk.{idx}.ffn_down.weight"),
        )?),
    })
}

/// Load Gemma 4 from GGUF with attention and dense FFN weights CPU-offloaded.
///
/// Attention and dense FFN weights are kept in RAM as quantized bytes and
/// dequantized per-token. VRAM holds only embeddings, norms, router weights,
/// RoPE tables, and MoE shared expert weights. MoE routed expert weights are
/// already zero-copy mmap'd via `QuantizedFusedExperts`.
///
/// Use this when VRAM is insufficient for full-precision weights.
pub fn load_gemma4_gguf_offloaded<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<(Gemma4<B>, Gemma4Runtime<B>, Gemma4Config), Gemma4GgufLoadError> {
    use burn::nn::LinearConfig;

    let file = GgufFile::open(path)?;
    let config = parse_gguf_config(&file)?;

    let embed_tokens = load_embedding(&file, "token_embd.weight", &config, device)?;

    // Dummy 1×1 linear for offloaded projections — never called during inference,
    // needed only so Gemma4Attention keeps its non-optional Linear<B> fields.
    let dummy_linear = |dev: &B::Device| LinearConfig::new(1, 1).with_bias(false).init::<B>(dev);

    let mut layers = Vec::with_capacity(config.num_layers);
    let mut offloaded_layers = Vec::with_capacity(config.num_layers);

    for i in 0..config.num_layers {
        // Load norms and non-attention/FFN weights normally (these are small)
        let input_norm = load_rmsnorm(
            &file,
            &format!("blk.{i}.attn_norm.weight"),
            config.hidden_size,
            config.norm_eps,
            device,
        )?;
        let post_attention_norm = load_rmsnorm(
            &file,
            &format!("blk.{i}.post_attention_norm.weight"),
            config.hidden_size,
            config.norm_eps,
            device,
        )?;
        let pre_ffn_norm = load_rmsnorm(
            &file,
            &format!("blk.{i}.ffn_norm.weight"),
            config.hidden_size,
            config.norm_eps,
            device,
        )?;
        let post_ffn_norm = load_rmsnorm(
            &file,
            &format!("blk.{i}.post_ffw_norm.weight"),
            config.hidden_size,
            config.norm_eps,
            device,
        )?;

        // Infer head_dim and dims from tensor info (same logic as load_attention)
        let q_name = format!("blk.{i}.attn_q.weight");
        let k_name = format!("blk.{i}.attn_k.weight");
        let q_info = file
            .tensor_info(&q_name)
            .ok_or_else(|| Gemma4GgufLoadError::MissingTensor(q_name.clone()))?;
        let k_info = file
            .tensor_info(&k_name)
            .ok_or_else(|| Gemma4GgufLoadError::MissingTensor(k_name.clone()))?;
        let q_dim = q_info.shape[0];
        let kv_dim = k_info.shape[0];
        let head_dim =
            if let Some(norm_info) = file.tensor_info(&format!("blk.{i}.attn_q_norm.weight")) {
                norm_info.shape[0]
            } else {
                config.head_dim
            };
        let num_heads = q_dim / head_dim;
        let num_kv_heads = kv_dim / head_dim;

        let q_norm = load_rmsnorm(
            &file,
            &format!("blk.{i}.attn_q_norm.weight"),
            head_dim,
            config.norm_eps,
            device,
        )?;
        let k_norm = load_rmsnorm(
            &file,
            &format!("blk.{i}.attn_k_norm.weight"),
            head_dim,
            config.norm_eps,
            device,
        )?;

        // Attention struct with 1×1 dummy projections (real weights go to RAM offload)
        let attention = Gemma4Attention {
            q_proj: dummy_linear(device),
            k_proj: dummy_linear(device),
            v_proj: dummy_linear(device),
            o_proj: dummy_linear(device),
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
        };

        // FFN: MoE layers are loaded normally (experts already zero-copy);
        // dense FFN layers get dummy projections with real weights in RAM offload.
        let (ffn, dense_ffn_offload) = if config.is_moe_layer(i) {
            (Gemma4Ffn::Moe(load_moe(&file, i, &config, device)?), None)
        } else {
            let offload = load_dense_ffn_offload::<B>(&file, i)?;
            let dense = Gemma4DenseFfn {
                gate_proj: dummy_linear(device),
                up_proj: dummy_linear(device),
                down_proj: dummy_linear(device),
            };
            (Gemma4Ffn::Dense(dense), Some(offload))
        };

        let layer_output_scale = if file.contains(&format!("blk.{i}.layer_output_scale.weight")) {
            let t: Tensor<B, 1> =
                file.load_f32(&format!("blk.{i}.layer_output_scale.weight"), device)?;
            t.into_scalar().elem::<f32>()
        } else {
            1.0
        };

        let use_sliding_window = head_dim < config.head_dim;

        layers.push(Gemma4Layer {
            attention,
            ffn,
            input_norm,
            post_attention_norm,
            pre_ffn_norm,
            post_ffn_norm,
            use_sliding_window,
            layer_output_scale,
        });

        offloaded_layers.push(LayerOffload {
            attention: load_attention_offload::<B>(&file, i)?,
            dense_ffn: dense_ffn_offload,
            gpu_experts: None,
        });
    }

    let norm = load_rmsnorm(
        &file,
        "output_norm.weight",
        config.hidden_size,
        config.norm_eps,
        device,
    )?;

    let mut ropes = std::collections::HashMap::new();
    for layer in &layers {
        let hd = layer.attention.head_dim;
        ropes.entry(hd).or_insert_with(|| {
            let base = if hd == config.head_dim {
                config.rope_base
            } else {
                config.rope_base_swa
            };
            RotaryEmbedding::with_base(hd, config.max_seq_len, base, device)
        });
    }

    let model = Gemma4 {
        embed_tokens,
        layers,
        norm,
    };

    let runtime = Gemma4Runtime {
        ropes,
        config: config.clone(),
        offloaded_layers: Some(offloaded_layers),
    };

    Ok((model, runtime, config))
}

// ============================================================================
// GPU-quantized loader (requires `cubecl` feature)
// ============================================================================

/// Wrapper giving `GpuQuantizedLinear<R>` the `LinearForward3d<CubeBackend<R,F,I,BT>>` interface.
#[cfg(feature = "cubecl")]
struct GpuLinearWrap<
    R: sketchpad_cubecl::CubeRuntime,
    F: sketchpad_cubecl::FloatElement,
    I: sketchpad_cubecl::IntElement,
    BT: sketchpad_cubecl::BoolElement,
> {
    inner: sketchpad_cubecl::GpuQuantizedLinear<R>,
    _ph: std::marker::PhantomData<(F, I, BT)>,
}

#[cfg(feature = "cubecl")]
impl<R, F, I, BT> crate::quantized::LinearForward3d<sketchpad_cubecl::CubeBackend<R, F, I, BT>>
    for GpuLinearWrap<R, F, I, BT>
where
    R: sketchpad_cubecl::CubeRuntime + Send + Sync,
    F: sketchpad_cubecl::FloatElement,
    I: sketchpad_cubecl::IntElement,
    BT: sketchpad_cubecl::BoolElement,
{
    fn forward_3d(
        &self,
        x: Tensor<sketchpad_cubecl::CubeBackend<R, F, I, BT>, 3>,
    ) -> Tensor<sketchpad_cubecl::CubeBackend<R, F, I, BT>, 3> {
        self.inner.forward_burn(x)
    }
}

/// Wrapper giving `GpuQuantizedFusedExperts<R>` the
/// `FusedExpertForward<CubeBackend<R,F,I,BT>>` interface.
#[cfg(feature = "cubecl")]
#[allow(dead_code)]
struct GpuFusedExpertsWrap<
    R: sketchpad_cubecl::CubeRuntime,
    F: sketchpad_cubecl::FloatElement,
    I: sketchpad_cubecl::IntElement,
    BT: sketchpad_cubecl::BoolElement,
> {
    inner: sketchpad_cubecl::GpuQuantizedFusedExperts<R>,
    _ph: std::marker::PhantomData<(F, I, BT)>,
}

#[cfg(feature = "cubecl")]
impl<R, F, I, BT> crate::quantized::FusedExpertForward<sketchpad_cubecl::CubeBackend<R, F, I, BT>>
    for GpuFusedExpertsWrap<R, F, I, BT>
where
    R: sketchpad_cubecl::CubeRuntime + Send + Sync,
    F: sketchpad_cubecl::FloatElement,
    I: sketchpad_cubecl::IntElement,
    BT: sketchpad_cubecl::BoolElement,
{
    fn forward_expert(
        &self,
        expert_idx: usize,
        input: Tensor<sketchpad_cubecl::CubeBackend<R, F, I, BT>, 3>,
    ) -> Tensor<sketchpad_cubecl::CubeBackend<R, F, I, BT>, 3> {
        self.inner.forward_expert::<F, I, BT>(expert_idx, input)
    }
}

/// Load Gemma 4 from GGUF with attention and dense FFN weights kept in VRAM as
/// packed quantized bytes, dequantized per-token on the GPU.
///
/// 4–8× VRAM savings vs `load_gemma4_gguf` at the cost of one dequantization
/// kernel per projection per token. MoE routed expert weights are zero-copy
/// mmap'd and unaffected. Norms, router, RoPE, and shared expert weights are
/// loaded at full (f16) precision.
///
/// `dtype` controls the float precision of dequantized outputs (use `DType::F16`
/// to halve activation memory vs `DType::F32`).
///
/// Requires the `cubecl` feature and a `CubeBackend` backend.
#[cfg(feature = "cubecl")]
#[allow(clippy::type_complexity)]
pub fn load_gemma4_gguf_gpu_quant<R, F, I, BT, P>(
    path: P,
    device: &R::Device,
    dtype: burn::tensor::DType,
) -> Result<
    (
        Gemma4<sketchpad_cubecl::CubeBackend<R, F, I, BT>>,
        Gemma4Runtime<sketchpad_cubecl::CubeBackend<R, F, I, BT>>,
        Gemma4Config,
    ),
    Gemma4GgufLoadError,
>
where
    R: sketchpad_cubecl::CubeRuntime + Send + Sync + 'static,
    F: sketchpad_cubecl::FloatElement,
    I: sketchpad_cubecl::IntElement,
    BT: sketchpad_cubecl::BoolElement,
    P: AsRef<Path>,
{
    use burn::nn::LinearConfig;
    use sketchpad_cubecl::{GpuQuantType, GpuQuantizedFusedExperts, GpuQuantizedLinear};

    type B<R, F, I, BT> = sketchpad_cubecl::CubeBackend<R, F, I, BT>;

    let file = GgufFile::open(path)?;
    let config = parse_gguf_config(&file)?;

    let client = R::client(device);

    let embed_tokens =
        load_embedding::<B<R, F, I, BT>>(&file, "token_embd.weight", &config, device)?;

    let dummy = |dev: &R::Device| {
        LinearConfig::new(1, 1)
            .with_bias(false)
            .init::<B<R, F, I, BT>>(dev)
    };

    // Helper: upload raw bytes as GpuQuantizedLinear and box as trait object
    let gpu_proj = |name: &str, out_f: usize, in_f: usize| {
        let (bytes, info) = file.tensor_raw_data(name)?;
        let qt = match info.ggml_type {
            GgmlType::Q8_0 => GpuQuantType::Q8_0,
            GgmlType::Q4K => GpuQuantType::Q4K,
            other => return Err(Gemma4GgufLoadError::UnsupportedQuantType(other)),
        };
        let inner = GpuQuantizedLinear::from_bytes(bytes, qt, out_f, in_f, &client, device, dtype);
        Ok::<Box<dyn crate::quantized::LinearForward3d<B<R, F, I, BT>>>, Gemma4GgufLoadError>(
            Box::new(GpuLinearWrap::<R, F, I, BT> {
                inner,
                _ph: std::marker::PhantomData,
            }),
        )
    };

    let mut layers = Vec::with_capacity(config.num_layers);
    let mut offloaded_layers = Vec::with_capacity(config.num_layers);

    for i in 0..config.num_layers {
        let input_norm = load_rmsnorm::<B<R, F, I, BT>>(
            &file,
            &format!("blk.{i}.attn_norm.weight"),
            config.hidden_size,
            config.norm_eps,
            device,
        )?;
        let post_attention_norm = load_rmsnorm::<B<R, F, I, BT>>(
            &file,
            &format!("blk.{i}.post_attention_norm.weight"),
            config.hidden_size,
            config.norm_eps,
            device,
        )?;
        let pre_ffn_norm = load_rmsnorm::<B<R, F, I, BT>>(
            &file,
            &format!("blk.{i}.ffn_norm.weight"),
            config.hidden_size,
            config.norm_eps,
            device,
        )?;
        let post_ffn_norm = load_rmsnorm::<B<R, F, I, BT>>(
            &file,
            &format!("blk.{i}.post_ffw_norm.weight"),
            config.hidden_size,
            config.norm_eps,
            device,
        )?;

        // Derive head_dim from q_norm weight shape (may differ per layer for SWA layers)
        let q_norm_name = format!("blk.{i}.attn_q_norm.weight");
        let head_dim = file
            .tensor_info(&q_norm_name)
            .map(|info| info.shape[0])
            .unwrap_or(config.head_dim);

        let q_norm =
            load_rmsnorm::<B<R, F, I, BT>>(&file, &q_norm_name, head_dim, config.norm_eps, device)?;
        let k_norm = load_rmsnorm::<B<R, F, I, BT>>(
            &file,
            &format!("blk.{i}.attn_k_norm.weight"),
            head_dim,
            config.norm_eps,
            device,
        )?;

        // Derive q_dim and kv_dim from GGUF tensor shapes (shape[0] = out_features)
        let q_weight_shape = &file
            .tensor_info(&format!("blk.{i}.attn_q.weight"))
            .ok_or_else(|| Gemma4GgufLoadError::MissingTensor(format!("blk.{i}.attn_q.weight")))?
            .shape;
        let q_dim = q_weight_shape[0];
        let kv_weight_shape = &file
            .tensor_info(&format!("blk.{i}.attn_k.weight"))
            .ok_or_else(|| Gemma4GgufLoadError::MissingTensor(format!("blk.{i}.attn_k.weight")))?
            .shape;
        let kv_dim = kv_weight_shape[0];
        let hidden = config.hidden_size;

        let attention_offload = crate::quantized::AttentionOffload {
            q_proj: gpu_proj(&format!("blk.{i}.attn_q.weight"), q_dim, hidden)?,
            k_proj: gpu_proj(&format!("blk.{i}.attn_k.weight"), kv_dim, hidden)?,
            v_proj: if file.contains(&format!("blk.{i}.attn_v.weight")) {
                gpu_proj(&format!("blk.{i}.attn_v.weight"), kv_dim, hidden)?
            } else {
                gpu_proj(&format!("blk.{i}.attn_k.weight"), kv_dim, hidden)?
            },
            o_proj: gpu_proj(&format!("blk.{i}.attn_output.weight"), hidden, q_dim)?,
        };

        let attention = Gemma4Attention {
            q_proj: dummy(device),
            k_proj: dummy(device),
            v_proj: dummy(device),
            o_proj: dummy(device),
            q_norm,
            k_norm,
            num_heads: q_dim / head_dim,
            num_kv_heads: kv_dim / head_dim,
            head_dim,
        };

        let (ffn, dense_ffn_offload, gpu_experts_opt) = if config.is_moe_layer(i) {
            // Build GPU-quantized experts from the fused GGUF tensors.
            // `load_moe` keeps experts as CPU-mmap'd `QuantizedFusedExperts` (used as
            // placeholder); the GPU path is routed through `gpu_experts` in LayerOffload.
            let gate_up_name = format!("blk.{i}.ffn_gate_up_exps.weight");
            let down_name = format!("blk.{i}.ffn_down_exps.weight");
            let gpu_experts_box: Option<
                Box<dyn crate::quantized::FusedExpertForward<B<R, F, I, BT>>>,
            > = if file.contains(&gate_up_name) {
                let gate_up_info = file
                    .tensor_info(&gate_up_name)
                    .ok_or_else(|| Gemma4GgufLoadError::MissingTensor(gate_up_name.clone()))?;
                let down_info = file
                    .tensor_info(&down_name)
                    .ok_or_else(|| Gemma4GgufLoadError::MissingTensor(down_name.clone()))?;

                let gate_up_qt = match gate_up_info.ggml_type {
                    GgmlType::Q8_0 => GpuQuantType::Q8_0,
                    GgmlType::Q4K => GpuQuantType::Q4K,
                    GgmlType::Q2K => GpuQuantType::Q2K,
                    GgmlType::Q3K => GpuQuantType::Q3K,
                    other => return Err(Gemma4GgufLoadError::UnsupportedQuantType(other)),
                };
                let down_qt = match down_info.ggml_type {
                    GgmlType::Q8_0 => GpuQuantType::Q8_0,
                    GgmlType::Q4K => GpuQuantType::Q4K,
                    GgmlType::Q2K => GpuQuantType::Q2K,
                    GgmlType::Q3K => GpuQuantType::Q3K,
                    other => return Err(Gemma4GgufLoadError::UnsupportedQuantType(other)),
                };

                let (gate_up_bytes, _) = file.tensor_raw_data(&gate_up_name)?;
                let (down_bytes, _) = file.tensor_raw_data(&down_name)?;

                let inner = GpuQuantizedFusedExperts::from_bytes(
                    gate_up_bytes,
                    gate_up_qt,
                    down_bytes,
                    down_qt,
                    config.num_experts,
                    config.expert_intermediate_size,
                    config.hidden_size,
                    &client,
                    device,
                    dtype,
                );
                Some(Box::new(GpuFusedExpertsWrap::<R, F, I, BT> {
                    inner,
                    _ph: std::marker::PhantomData,
                })
                    as Box<
                        dyn crate::quantized::FusedExpertForward<B<R, F, I, BT>>,
                    >)
            } else {
                None
            };

            (
                Gemma4Ffn::Moe(load_moe::<B<R, F, I, BT>>(&file, i, &config, device)?),
                None,
                gpu_experts_box,
            )
        } else {
            let inter = config.intermediate_size;
            let ffn_offload = crate::quantized::DenseFfnOffload {
                gate_proj: gpu_proj(&format!("blk.{i}.ffn_gate.weight"), inter, hidden)?,
                up_proj: gpu_proj(&format!("blk.{i}.ffn_up.weight"), inter, hidden)?,
                down_proj: gpu_proj(&format!("blk.{i}.ffn_down.weight"), hidden, inter)?,
            };
            let dense = Gemma4DenseFfn {
                gate_proj: dummy(device),
                up_proj: dummy(device),
                down_proj: dummy(device),
            };
            (Gemma4Ffn::Dense(dense), Some(ffn_offload), None)
        };

        let layer_output_scale = if file.contains(&format!("blk.{i}.layer_output_scale.weight")) {
            let t: Tensor<B<R, F, I, BT>, 1> =
                file.load_f32(&format!("blk.{i}.layer_output_scale.weight"), device)?;
            t.into_scalar().elem::<f32>()
        } else {
            1.0
        };

        layers.push(Gemma4Layer {
            attention,
            ffn,
            input_norm,
            post_attention_norm,
            pre_ffn_norm,
            post_ffn_norm,
            use_sliding_window: head_dim < config.head_dim,
            layer_output_scale,
        });

        offloaded_layers.push(crate::quantized::LayerOffload {
            attention: attention_offload,
            dense_ffn: dense_ffn_offload,
            gpu_experts: gpu_experts_opt,
        });
    }

    let norm = load_rmsnorm::<B<R, F, I, BT>>(
        &file,
        "output_norm.weight",
        config.hidden_size,
        config.norm_eps,
        device,
    )?;

    let mut ropes = std::collections::HashMap::new();
    for layer in &layers {
        let hd = layer.attention.head_dim;
        ropes.entry(hd).or_insert_with(|| {
            let base = if hd == config.head_dim {
                config.rope_base
            } else {
                config.rope_base_swa
            };
            RotaryEmbedding::with_base(hd, config.max_seq_len, base, device)
        });
    }

    let model = Gemma4 {
        embed_tokens,
        layers,
        norm,
    };
    let runtime = Gemma4Runtime {
        ropes,
        config: config.clone(),
        offloaded_layers: Some(offloaded_layers),
    };
    Ok((model, runtime, config))
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
