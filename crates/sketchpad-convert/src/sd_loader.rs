//! Stable Diffusion Weight Loading from Safetensors
//!
//! Loads SD model weights from HuggingFace-format safetensors files.
//!
//! # Supported Formats
//!
//! - Single file models (model.safetensors)
//! - Multi-file models (text_encoder.safetensors, unet.safetensors, vae.safetensors)
//! - Diffusers-style directory structure
//!
//! # Recommended Workflow
//!
//! Since safetensors loading requires name mapping, we recommend a two-step process:
//!
//! 1. **First run**: Convert safetensors to Burn record format
//! 2. **Subsequent runs**: Load directly from Burn record (much faster)
//!
//! ```ignore
//! use burn::record::{BinFileRecorder, FullPrecisionSettings};
//!
//! // First time: convert and save
//! let loader = SdWeightLoader::open("model/")?;
//! let clip = loader.load_clip_text_encoder::<MyBackend>(&ClipConfig::sd1x(), &device)?;
//! clip.save_file("clip.bin", &BinFileRecorder::<FullPrecisionSettings>::new())?;
//!
//! // Later: load directly (fast)
//! let clip = ClipTextEncoder::new(&ClipConfig::sd1x(), &device)
//!     .load_file("clip.bin", &BinFileRecorder::<FullPrecisionSettings>::new(), &device)?;
//! ```
//!
//! # Current Status
//!
//! - [x] CLIP text encoder loader - fully implemented
//! - [x] UNet loader - fully implemented
//! - [x] VAE decoder loader - fully implemented

use std::path::{Path, PathBuf};

use burn::module::Param;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::{EmbeddingConfig, LinearConfig};
use burn::prelude::*;

use crate::loader::{LoadError, SafeTensorFile};

use sketchpad_clip::{ClipConfig, ClipTextEncoder, OpenClipConfig, OpenClipTextEncoder};
use sketchpad_core::groupnorm::GroupNorm;
use sketchpad_core::layernorm::LayerNorm;

/// Error type for SD weight loading
#[derive(Debug, thiserror::Error)]
pub enum SdLoadError {
    #[error("Load error: {0}")]
    Load(#[from] LoadError),

    #[error("Missing tensor: {0}")]
    MissingTensor(String),

    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Shape mismatch for {tensor}: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        tensor: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
}

/// Weight loader for Stable Diffusion models
pub struct SdWeightLoader {
    /// Path to the weights (file or directory)
    path: PathBuf,
    /// Cached text encoder file (CLIP for SD1.x, or embedders.0 for SDXL)
    text_encoder_file: Option<SafeTensorFile>,
    /// Cached second text encoder file (OpenCLIP embedders.1 for SDXL)
    text_encoder_2_file: Option<SafeTensorFile>,
    /// Cached UNet file
    unet_file: Option<SafeTensorFile>,
    /// Cached VAE file
    vae_file: Option<SafeTensorFile>,
}

impl SdWeightLoader {
    /// Open a weights path (file or directory)
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, SdLoadError> {
        let path = path.as_ref().to_path_buf();

        if !path.exists() {
            return Err(SdLoadError::FileNotFound(path.display().to_string()));
        }

        Ok(Self {
            path,
            text_encoder_file: None,
            text_encoder_2_file: None,
            unet_file: None,
            vae_file: None,
        })
    }

    /// Get the SafeTensorFile for text encoder weights
    fn get_text_encoder_file(&mut self) -> Result<&SafeTensorFile, SdLoadError> {
        if self.text_encoder_file.is_none() {
            let file_path = self.find_component_file(&[
                "text_encoder.safetensors",
                "text_encoder/model.safetensors",
            ])?;
            self.text_encoder_file = Some(SafeTensorFile::open(file_path)?);
        }
        Ok(self.text_encoder_file.as_ref().unwrap())
    }

    /// Get the SafeTensorFile for second text encoder weights (OpenCLIP for SDXL)
    fn get_text_encoder_2_file(&mut self) -> Result<&SafeTensorFile, SdLoadError> {
        if self.text_encoder_2_file.is_none() {
            let file_path = self.find_component_file(&[
                "text_encoder_2.safetensors",
                "text_encoder_2/model.safetensors",
            ])?;
            self.text_encoder_2_file = Some(SafeTensorFile::open(file_path)?);
        }
        Ok(self.text_encoder_2_file.as_ref().unwrap())
    }

    /// Get the SafeTensorFile for UNet weights
    fn get_unet_file(&mut self) -> Result<&SafeTensorFile, SdLoadError> {
        if self.unet_file.is_none() {
            let file_path = self.find_component_file(&[
                "unet.safetensors",
                "unet/diffusion_pytorch_model.safetensors",
            ])?;
            self.unet_file = Some(SafeTensorFile::open(file_path)?);
        }
        Ok(self.unet_file.as_ref().unwrap())
    }

    /// Get the SafeTensorFile for VAE weights
    fn get_vae_file(&mut self) -> Result<&SafeTensorFile, SdLoadError> {
        if self.vae_file.is_none() {
            let file_path = self.find_component_file(&[
                "vae.safetensors",
                "vae/diffusion_pytorch_model.safetensors",
            ])?;
            self.vae_file = Some(SafeTensorFile::open(file_path)?);
        }
        Ok(self.vae_file.as_ref().unwrap())
    }

    /// Find a component file from possible locations
    fn find_component_file(&self, candidates: &[&str]) -> Result<PathBuf, SdLoadError> {
        // If path is a file, return it directly (single-file model)
        if self.path.is_file() {
            return Ok(self.path.clone());
        }

        // Search for component files in directory
        for candidate in candidates {
            let full_path = self.path.join(candidate);
            if full_path.exists() {
                return Ok(full_path);
            }
        }

        Err(SdLoadError::FileNotFound(format!(
            "Could not find any of {:?} in {}",
            candidates,
            self.path.display()
        )))
    }

    /// Load CLIP text encoder weights
    ///
    /// Returns a ClipTextEncoder with loaded weights.
    pub fn load_clip_text_encoder<B: Backend>(
        &mut self,
        config: &ClipConfig,
        device: &B::Device,
    ) -> Result<ClipTextEncoder<B>, SdLoadError> {
        let file = self.get_text_encoder_file()?;
        load_clip_from_file(file, config, device)
    }

    /// Load OpenCLIP text encoder weights (SDXL second encoder)
    ///
    /// Returns an OpenClipTextEncoder with loaded weights.
    pub fn load_open_clip_text_encoder<B: Backend>(
        &mut self,
        config: &OpenClipConfig,
        device: &B::Device,
    ) -> Result<OpenClipTextEncoder<B>, SdLoadError> {
        let file = self.get_text_encoder_2_file()?;
        load_open_clip_from_file(file, config, device)
    }

    /// Load UNet weights
    ///
    /// Returns a UNet with loaded weights.
    pub fn load_unet<B: Backend>(
        &mut self,
        config: &sketchpad_unet::UNetConfig,
        device: &B::Device,
    ) -> Result<sketchpad_unet::UNet<B>, SdLoadError> {
        let file = self.get_unet_file()?;
        load_unet_from_file(file, config, device)
    }

    /// Load UNetXL weights (SDXL)
    ///
    /// Returns a UNetXL with loaded weights.
    pub fn load_unet_xl<B: Backend>(
        &mut self,
        config: &sketchpad_unet::UNetXLConfig,
        device: &B::Device,
    ) -> Result<sketchpad_unet::UNetXL<B>, SdLoadError> {
        let file = self.get_unet_file()?;
        load_unet_xl_from_file(file, config, device)
    }

    /// Load VAE decoder weights
    ///
    /// Returns a Decoder with loaded weights.
    pub fn load_vae_decoder<B: Backend>(
        &mut self,
        config: &sketchpad_vae::DecoderConfig,
        device: &B::Device,
    ) -> Result<sketchpad_vae::Decoder<B>, SdLoadError> {
        let file = self.get_vae_file()?;
        load_vae_decoder_from_file(file, config, device)
    }

    /// Load VAE encoder weights
    ///
    /// Returns an Encoder with loaded weights.
    pub fn load_vae_encoder<B: Backend>(
        &mut self,
        config: &sketchpad_vae::EncoderConfig,
        device: &B::Device,
    ) -> Result<sketchpad_vae::Encoder<B>, SdLoadError> {
        let file = self.get_vae_file()?;
        load_vae_encoder_from_file(file, config, device)
    }
}

/// Load CLIP text encoder from a SafeTensorFile
fn load_clip_from_file<B: Backend>(
    file: &SafeTensorFile,
    config: &ClipConfig,
    device: &B::Device,
) -> Result<ClipTextEncoder<B>, SdLoadError> {
    // Detect prefix - could be "text_model" or "text_encoder.text_model"
    let prefix = detect_clip_prefix(file);

    // Load token embedding
    let token_emb_key = format!("{}.embeddings.token_embedding.weight", prefix);
    let token_emb_weight: Tensor<B, 2> = file.load_f32(&token_emb_key, device)?;

    let mut token_embedding =
        EmbeddingConfig::new(config.vocab_size, config.embed_dim).init(device);
    token_embedding.weight = Param::from_tensor(token_emb_weight);

    // Load position embedding
    let pos_emb_key = format!("{}.embeddings.position_embedding.weight", prefix);
    let position_embedding: Tensor<B, 2> = file.load_f32(&pos_emb_key, device)?;
    let position_embedding = Param::from_tensor(position_embedding);

    // Load transformer layers
    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let layer = load_clip_transformer_layer(file, &prefix, i, config, device)?;
        layers.push(layer);
    }

    // Load final layer norm
    let final_ln_weight_key = format!("{}.final_layer_norm.weight", prefix);
    let final_ln_bias_key = format!("{}.final_layer_norm.bias", prefix);
    let final_layer_norm = load_layer_norm(
        file,
        &final_ln_weight_key,
        &final_ln_bias_key,
        config.embed_dim,
        device,
    )?;

    // Precompute causal mask for max context length
    let causal_mask =
        sketchpad_clip::attention::precompute_causal_mask(config.context_length, device);

    Ok(ClipTextEncoder {
        token_embedding,
        position_embedding,
        layers,
        final_layer_norm,
        causal_mask,
        context_length: config.context_length,
    })
}

/// Detect the prefix used for CLIP weights
fn detect_clip_prefix(file: &SafeTensorFile) -> String {
    // Check common prefixes
    let prefixes = [
        "text_model",
        "text_encoder.text_model",
        "cond_stage_model.transformer.text_model",
        // SDXL CompVis format (single-file checkpoints)
        "conditioner.embedders.0.transformer.text_model",
    ];

    for prefix in prefixes {
        let test_key = format!("{}.embeddings.token_embedding.weight", prefix);
        if file.contains(&test_key) {
            return prefix.to_string();
        }
    }

    // Default
    "text_model".to_string()
}

/// Load a single CLIP transformer layer
fn load_clip_transformer_layer<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    layer_idx: usize,
    config: &ClipConfig,
    device: &B::Device,
) -> Result<sketchpad_clip::TransformerBlock<B>, SdLoadError> {
    let layer_prefix = format!("{}.encoder.layers.{}", prefix, layer_idx);

    // Load attention layer norm (layer_norm1)
    let attn_norm = load_layer_norm(
        file,
        &format!("{}.layer_norm1.weight", layer_prefix),
        &format!("{}.layer_norm1.bias", layer_prefix),
        config.embed_dim,
        device,
    )?;

    // Load self-attention
    let attn = load_clip_attention(file, &layer_prefix, config, device)?;

    // Load FFN layer norm (layer_norm2)
    let ffn_norm = load_layer_norm(
        file,
        &format!("{}.layer_norm2.weight", layer_prefix),
        &format!("{}.layer_norm2.bias", layer_prefix),
        config.embed_dim,
        device,
    )?;

    // Load feed-forward
    let ffn = load_clip_ffn(file, &layer_prefix, config, device)?;

    Ok(sketchpad_clip::TransformerBlock {
        attn_norm,
        attn,
        ffn_norm,
        ffn,
    })
}

/// Load CLIP multi-head self-attention
fn load_clip_attention<B: Backend>(
    file: &SafeTensorFile,
    layer_prefix: &str,
    config: &ClipConfig,
    device: &B::Device,
) -> Result<sketchpad_clip::MultiHeadSelfAttention<B>, SdLoadError> {
    let embed_dim = config.embed_dim;

    let q_proj = load_linear(
        file,
        &format!("{}.self_attn.q_proj.weight", layer_prefix),
        Some(&format!("{}.self_attn.q_proj.bias", layer_prefix)),
        embed_dim,
        embed_dim,
        device,
    )?;

    let k_proj = load_linear(
        file,
        &format!("{}.self_attn.k_proj.weight", layer_prefix),
        Some(&format!("{}.self_attn.k_proj.bias", layer_prefix)),
        embed_dim,
        embed_dim,
        device,
    )?;

    let v_proj = load_linear(
        file,
        &format!("{}.self_attn.v_proj.weight", layer_prefix),
        Some(&format!("{}.self_attn.v_proj.bias", layer_prefix)),
        embed_dim,
        embed_dim,
        device,
    )?;

    let out_proj = load_linear(
        file,
        &format!("{}.self_attn.out_proj.weight", layer_prefix),
        Some(&format!("{}.self_attn.out_proj.bias", layer_prefix)),
        embed_dim,
        embed_dim,
        device,
    )?;

    Ok(sketchpad_clip::MultiHeadSelfAttention {
        q_proj,
        k_proj,
        v_proj,
        out_proj,
        num_heads: config.num_heads,
        head_dim: embed_dim / config.num_heads,
    })
}

/// Load CLIP feed-forward network
fn load_clip_ffn<B: Backend>(
    file: &SafeTensorFile,
    layer_prefix: &str,
    config: &ClipConfig,
    device: &B::Device,
) -> Result<sketchpad_clip::FeedForward<B>, SdLoadError> {
    let fc1 = load_linear(
        file,
        &format!("{}.mlp.fc1.weight", layer_prefix),
        Some(&format!("{}.mlp.fc1.bias", layer_prefix)),
        config.embed_dim,
        config.intermediate_size,
        device,
    )?;

    let fc2 = load_linear(
        file,
        &format!("{}.mlp.fc2.weight", layer_prefix),
        Some(&format!("{}.mlp.fc2.bias", layer_prefix)),
        config.intermediate_size,
        config.embed_dim,
        device,
    )?;

    Ok(sketchpad_clip::FeedForward { fc1, fc2 })
}

// ============================================================================
// OpenCLIP Text Encoder Loading (SDXL second encoder)
// ============================================================================

/// Load OpenCLIP text encoder weights from a SafeTensorFile
fn load_open_clip_from_file<B: Backend>(
    file: &SafeTensorFile,
    config: &OpenClipConfig,
    device: &B::Device,
) -> Result<OpenClipTextEncoder<B>, SdLoadError> {
    // Detect prefix - could be "model" or "conditioner.embedders.1.model"
    let prefix = detect_open_clip_prefix(file);

    // Load token embedding
    let token_emb_key = format!("{}.token_embedding.weight", prefix);
    let token_emb_weight: Tensor<B, 2> = file.load_f32(&token_emb_key, device)?;

    let mut token_embedding =
        EmbeddingConfig::new(config.vocab_size, config.embed_dim).init(device);
    token_embedding.weight = Param::from_tensor(token_emb_weight);

    // Load position embedding (OpenCLIP uses positional_embedding, not position_embedding)
    let pos_emb_key = format!("{}.positional_embedding", prefix);
    let position_embedding: Tensor<B, 2> = file.load_f32(&pos_emb_key, device)?;
    let position_embedding = Param::from_tensor(position_embedding);

    // Load transformer layers
    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let layer = load_open_clip_transformer_layer(file, &prefix, i, config, device)?;
        layers.push(layer);
    }

    // Load final layer norm (ln_final)
    let final_ln_weight_key = format!("{}.ln_final.weight", prefix);
    let final_ln_bias_key = format!("{}.ln_final.bias", prefix);
    let final_layer_norm = load_layer_norm(
        file,
        &final_ln_weight_key,
        &final_ln_bias_key,
        config.embed_dim,
        device,
    )?;

    // Load text projection
    let text_proj_key = format!("{}.text_projection", prefix);
    let text_proj_weight: Tensor<B, 2> = file.load_f32(&text_proj_key, device)?;
    let mut text_projection = LinearConfig::new(config.embed_dim, config.projection_dim)
        .with_bias(false)
        .init(device);
    // text_projection is [embed_dim, projection_dim] in OpenCLIP, need to transpose
    text_projection.weight = Param::from_tensor(text_proj_weight.transpose());

    // Precompute causal mask for max context length
    let causal_mask =
        sketchpad_clip::attention::precompute_causal_mask(config.context_length, device);

    Ok(OpenClipTextEncoder {
        token_embedding,
        position_embedding,
        layers,
        final_layer_norm,
        text_projection,
        causal_mask,
        context_length: config.context_length,
    })
}

/// Detect the prefix used for OpenCLIP weights
fn detect_open_clip_prefix(file: &SafeTensorFile) -> String {
    // Check common prefixes for OpenCLIP
    let prefixes = ["model", "conditioner.embedders.1.model", "text_encoder_2"];

    for prefix in prefixes {
        let test_key = format!("{}.token_embedding.weight", prefix);
        if file.contains(&test_key) {
            return prefix.to_string();
        }
    }

    // Default
    "model".to_string()
}

/// Load a single OpenCLIP transformer layer
fn load_open_clip_transformer_layer<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    layer_idx: usize,
    config: &OpenClipConfig,
    device: &B::Device,
) -> Result<sketchpad_clip::open_clip::OpenClipTransformerBlock<B>, SdLoadError> {
    let layer_prefix = format!("{}.transformer.resblocks.{}", prefix, layer_idx);

    // Load attention layer norm (ln_1)
    let attn_norm = load_layer_norm(
        file,
        &format!("{}.ln_1.weight", layer_prefix),
        &format!("{}.ln_1.bias", layer_prefix),
        config.embed_dim,
        device,
    )?;

    // Load self-attention (with fused QKV)
    let attn = load_open_clip_attention(file, &layer_prefix, config, device)?;

    // Load FFN layer norm (ln_2)
    let ffn_norm = load_layer_norm(
        file,
        &format!("{}.ln_2.weight", layer_prefix),
        &format!("{}.ln_2.bias", layer_prefix),
        config.embed_dim,
        device,
    )?;

    // Load feed-forward
    let ffn = load_open_clip_ffn(file, &layer_prefix, config, device)?;

    Ok(sketchpad_clip::open_clip::OpenClipTransformerBlock {
        attn_norm,
        attn,
        ffn_norm,
        ffn,
    })
}

/// Load OpenCLIP multi-head self-attention with fused QKV
fn load_open_clip_attention<B: Backend>(
    file: &SafeTensorFile,
    layer_prefix: &str,
    config: &OpenClipConfig,
    device: &B::Device,
) -> Result<sketchpad_clip::open_clip::OpenClipMultiHeadSelfAttention<B>, SdLoadError> {
    let embed_dim = config.embed_dim;

    // OpenCLIP uses fused in_proj_weight [3*embed_dim, embed_dim] and in_proj_bias [3*embed_dim]
    let in_proj_weight_key = format!("{}.attn.in_proj_weight", layer_prefix);
    let in_proj_bias_key = format!("{}.attn.in_proj_bias", layer_prefix);

    let in_proj_weight: Tensor<B, 2> = file.load_f32(&in_proj_weight_key, device)?;
    let in_proj_bias: Tensor<B, 1> = file.load_f32(&in_proj_bias_key, device)?;

    // Split fused weights into Q, K, V
    // in_proj_weight is [3*embed_dim, embed_dim], split along dim 0
    let q_weight = in_proj_weight.clone().slice([0..embed_dim, 0..embed_dim]);
    let k_weight = in_proj_weight
        .clone()
        .slice([embed_dim..2 * embed_dim, 0..embed_dim]);
    let v_weight = in_proj_weight.slice([2 * embed_dim..3 * embed_dim, 0..embed_dim]);

    let q_bias = in_proj_bias.clone().slice(0..embed_dim);
    let k_bias = in_proj_bias.clone().slice(embed_dim..2 * embed_dim);
    let v_bias = in_proj_bias.slice(2 * embed_dim..3 * embed_dim);

    // Create Q, K, V projections
    let mut q_proj = LinearConfig::new(embed_dim, embed_dim).init(device);
    q_proj.weight = Param::from_tensor(q_weight.transpose());
    q_proj.bias = Some(Param::from_tensor(q_bias));

    let mut k_proj = LinearConfig::new(embed_dim, embed_dim).init(device);
    k_proj.weight = Param::from_tensor(k_weight.transpose());
    k_proj.bias = Some(Param::from_tensor(k_bias));

    let mut v_proj = LinearConfig::new(embed_dim, embed_dim).init(device);
    v_proj.weight = Param::from_tensor(v_weight.transpose());
    v_proj.bias = Some(Param::from_tensor(v_bias));

    // Load output projection
    let out_proj = load_linear(
        file,
        &format!("{}.attn.out_proj.weight", layer_prefix),
        Some(&format!("{}.attn.out_proj.bias", layer_prefix)),
        embed_dim,
        embed_dim,
        device,
    )?;

    Ok(sketchpad_clip::open_clip::OpenClipMultiHeadSelfAttention {
        q_proj,
        k_proj,
        v_proj,
        out_proj,
        num_heads: config.num_heads,
        head_dim: embed_dim / config.num_heads,
    })
}

/// Load OpenCLIP feed-forward network
fn load_open_clip_ffn<B: Backend>(
    file: &SafeTensorFile,
    layer_prefix: &str,
    config: &OpenClipConfig,
    device: &B::Device,
) -> Result<sketchpad_clip::open_clip::OpenClipFeedForward<B>, SdLoadError> {
    // OpenCLIP uses c_fc and c_proj naming
    let fc1 = load_linear(
        file,
        &format!("{}.mlp.c_fc.weight", layer_prefix),
        Some(&format!("{}.mlp.c_fc.bias", layer_prefix)),
        config.embed_dim,
        config.intermediate_size,
        device,
    )?;

    let fc2 = load_linear(
        file,
        &format!("{}.mlp.c_proj.weight", layer_prefix),
        Some(&format!("{}.mlp.c_proj.bias", layer_prefix)),
        config.intermediate_size,
        config.embed_dim,
        device,
    )?;

    Ok(sketchpad_clip::open_clip::OpenClipFeedForward { fc1, fc2 })
}

// ============================================================================
// Helper functions for loading common layer types
// ============================================================================

/// Load a Linear layer from safetensors
fn load_linear<B: Backend>(
    file: &SafeTensorFile,
    weight_key: &str,
    bias_key: Option<&str>,
    in_features: usize,
    out_features: usize,
    device: &B::Device,
) -> Result<burn::nn::Linear<B>, SdLoadError> {
    let weight: Tensor<B, 2> = file
        .load_f32(weight_key, device)
        .map_err(|e| SdLoadError::MissingTensor(format!("{}: {}", weight_key, e)))?;

    // Verify shape [out_features, in_features] (PyTorch convention)
    let [out_f, in_f] = weight.dims();
    if out_f != out_features || in_f != in_features {
        return Err(SdLoadError::ShapeMismatch {
            tensor: weight_key.to_string(),
            expected: vec![out_features, in_features],
            actual: vec![out_f, in_f],
        });
    }

    let has_bias = bias_key.map(|k| file.contains(k)).unwrap_or(false);
    let mut linear = LinearConfig::new(in_features, out_features)
        .with_bias(has_bias)
        .init(device);

    // PyTorch stores Linear weights as [out_features, in_features]
    // Burn expects [in_features, out_features] (Row layout default)
    // Transpose to convert from PyTorch to Burn convention
    linear.weight = Param::from_tensor(weight.transpose());

    if let Some(bias_k) = bias_key {
        if file.contains(bias_k) {
            let bias: Tensor<B, 1> = file.load_f32(bias_k, device)?;
            linear.bias = Some(Param::from_tensor(bias));
        }
    }

    Ok(linear)
}

/// Load a LayerNorm from safetensors
fn load_layer_norm<B: Backend>(
    file: &SafeTensorFile,
    weight_key: &str,
    bias_key: &str,
    _size: usize,
    device: &B::Device,
) -> Result<LayerNorm<B>, SdLoadError> {
    let weight: Tensor<B, 1> = file
        .load_f32(weight_key, device)
        .map_err(|e| SdLoadError::MissingTensor(format!("{}: {}", weight_key, e)))?;
    let bias: Tensor<B, 1> = file
        .load_f32(bias_key, device)
        .map_err(|e| SdLoadError::MissingTensor(format!("{}: {}", bias_key, e)))?;

    Ok(LayerNorm::from_weight_bias(weight, bias))
}

/// Load a GroupNorm from safetensors
#[allow(dead_code)]
fn load_group_norm<B: Backend>(
    file: &SafeTensorFile,
    weight_key: &str,
    bias_key: &str,
    num_groups: usize,
    device: &B::Device,
) -> Result<GroupNorm<B>, SdLoadError> {
    let weight: Tensor<B, 1> = file
        .load_f32(weight_key, device)
        .map_err(|e| SdLoadError::MissingTensor(format!("{}: {}", weight_key, e)))?;
    let bias: Tensor<B, 1> = file
        .load_f32(bias_key, device)
        .map_err(|e| SdLoadError::MissingTensor(format!("{}: {}", bias_key, e)))?;

    Ok(GroupNorm {
        num_groups,
        weight,
        bias,
        eps: 1e-6,
    })
}

/// Load a Conv2d from safetensors
#[allow(clippy::too_many_arguments)]
fn load_conv2d<B: Backend>(
    file: &SafeTensorFile,
    weight_key: &str,
    bias_key: Option<&str>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    padding: usize,
    device: &B::Device,
) -> Result<burn::nn::conv::Conv2d<B>, SdLoadError> {
    let weight: Tensor<B, 4> = file
        .load_f32(weight_key, device)
        .map_err(|e| SdLoadError::MissingTensor(format!("{}: {}", weight_key, e)))?;

    let has_bias = bias_key.map(|k| file.contains(k)).unwrap_or(false);
    let mut conv = Conv2dConfig::new([in_channels, out_channels], [kernel_size, kernel_size])
        .with_bias(has_bias)
        .with_padding(PaddingConfig2d::Explicit(padding, padding))
        .init(device);

    conv.weight = Param::from_tensor(weight);

    if let Some(bias_k) = bias_key {
        if file.contains(bias_k) {
            let bias: Tensor<B, 1> = file.load_f32(bias_k, device)?;
            conv.bias = Some(Param::from_tensor(bias));
        }
    }

    Ok(conv)
}

/// Load a Conv2d with stride from safetensors
#[allow(clippy::too_many_arguments)]
fn load_conv2d_strided<B: Backend>(
    file: &SafeTensorFile,
    weight_key: &str,
    bias_key: Option<&str>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    device: &B::Device,
) -> Result<burn::nn::conv::Conv2d<B>, SdLoadError> {
    let weight: Tensor<B, 4> = file
        .load_f32(weight_key, device)
        .map_err(|e| SdLoadError::MissingTensor(format!("{}: {}", weight_key, e)))?;

    let has_bias = bias_key.map(|k| file.contains(k)).unwrap_or(false);
    let mut conv = Conv2dConfig::new([in_channels, out_channels], [kernel_size, kernel_size])
        .with_bias(has_bias)
        .with_stride([stride, stride])
        .with_padding(PaddingConfig2d::Explicit(padding, padding))
        .init(device);

    conv.weight = Param::from_tensor(weight);

    if let Some(bias_k) = bias_key {
        if file.contains(bias_k) {
            let bias: Tensor<B, 1> = file.load_f32(bias_k, device)?;
            conv.bias = Some(Param::from_tensor(bias));
        }
    }

    Ok(conv)
}

// ============================================================================
// UNet Loading
// ============================================================================

use sketchpad_unet::{
    CrossAttention, DownBlock, DownBlockXL, Downsample, FeedForward, MidBlock, MidBlockXL,
    ResBlock, SpatialTransformer, TransformerBlock, UNet, UNetConfig, UNetXL, UNetXLConfig,
    UpBlock, UpBlockXL, Upsample, timestep_freqs,
};

/// Naming convention used in safetensors files
#[derive(Debug, Clone, Copy, PartialEq)]
enum UNetNaming {
    /// HuggingFace diffusers naming (down_blocks, mid_block, up_blocks)
    HuggingFace,
    /// CompVis/original SD naming (input_blocks, middle_block, output_blocks)
    CompVis,
}

impl UNetNaming {
    /// Detect naming convention from file
    fn detect(file: &SafeTensorFile, prefix: &str) -> Self {
        // Check for CompVis-style input_blocks
        let compvis_test = format!("{}.input_blocks.0.0.weight", prefix);
        if file.contains(&compvis_test) {
            return Self::CompVis;
        }
        Self::HuggingFace
    }
}

/// CompVis UNet block layout mapping
///
/// CompVis uses flat input_blocks indices while HF uses hierarchical down_blocks.
/// This struct tracks the mapping for a standard SD 1.x UNet with channel_mult [1,2,4,4].
///
/// Layout:
/// - input_blocks.0.0 = conv_in
/// - input_blocks.{1,2} = level 0 (320ch), res+attn each
/// - input_blocks.3.0.op = downsample
/// - input_blocks.{4,5} = level 1 (640ch), res+attn each
/// - input_blocks.6.0.op = downsample
/// - input_blocks.{7,8} = level 2 (1280ch), res+attn each
/// - input_blocks.9.0.op = downsample
/// - input_blocks.{10,11} = level 3 (1280ch), res only
#[allow(dead_code)]
struct CompVisBlockMap {
    /// Maps (level, block_in_level) to input_blocks index
    down_res_indices: Vec<Vec<usize>>,
    /// Maps level to downsample input_blocks index (None for last level)
    down_sample_indices: Vec<Option<usize>>,
    /// output_blocks work similarly but in reverse
    up_res_indices: Vec<Vec<usize>>,
    /// Maps level to upsample output_blocks index
    up_sample_indices: Vec<Option<usize>>,
}

impl CompVisBlockMap {
    /// Create mapping for SD 1.x channel_mult [1,2,4,4]
    fn sd1x() -> Self {
        Self {
            // Level 0: blocks 1,2; Level 1: blocks 4,5; Level 2: blocks 7,8; Level 3: blocks 10,11
            down_res_indices: vec![
                vec![1, 2],   // level 0
                vec![4, 5],   // level 1
                vec![7, 8],   // level 2
                vec![10, 11], // level 3 (no attention)
            ],
            down_sample_indices: vec![
                Some(3), // level 0 -> 1
                Some(6), // level 1 -> 2
                Some(9), // level 2 -> 3
                None,    // level 3 (no downsample)
            ],
            // output_blocks: 0-2 = level 3, 3-5 = level 2, 6-8 = level 1, 9-11 = level 0
            up_res_indices: vec![
                vec![9, 10, 11], // level 0 (with upsample at 11)
                vec![6, 7, 8],   // level 1 (with upsample at 8)
                vec![3, 4, 5],   // level 2 (with upsample at 5)
                vec![0, 1, 2],   // level 3 (no upsample)
            ],
            up_sample_indices: vec![
                Some(11), // level 0 upsample
                Some(8),  // level 1 upsample
                Some(5),  // level 2 upsample
                None,     // level 3 (no upsample)
            ],
        }
    }
}

/// Load UNet from a SafeTensorFile
fn load_unet_from_file<B: Backend>(
    file: &SafeTensorFile,
    config: &UNetConfig,
    device: &B::Device,
) -> Result<UNet<B>, SdLoadError> {
    // Detect prefix
    let prefix = detect_unet_prefix(file);

    // Detect naming convention
    let naming = UNetNaming::detect(file, &prefix);

    match naming {
        UNetNaming::CompVis => load_unet_compvis(file, &prefix, config, device),
        UNetNaming::HuggingFace => load_unet_hf(file, &prefix, config, device),
    }
}

/// Load SDXL UNet (UNetXL) from a SafeTensorFile (CompVis format)
///
/// SDXL UNet differs from SD 1.x:
/// - Has `label_emb` for pooled text + size conditioning
/// - Variable transformer depths per resolution
/// - Uses CompVis naming (input_blocks, etc.)
///
/// # Block structure (CompVis naming)
/// - input_blocks.0.0 = conv_in
/// - Level 0 (no attention): input_blocks.1.0/2.0 = res blocks, 3.0.op = downsample
/// - Level 1 (depth 2): input_blocks.4.0/.1, 5.0/.1 = res+attn, 6.0.op = downsample
/// - Level 2 (depth 10): input_blocks.7.0/.1, 8.0/.1 = res+attn, no downsample
fn load_unet_xl_from_file<B: Backend>(
    file: &SafeTensorFile,
    config: &UNetXLConfig,
    device: &B::Device,
) -> Result<UNetXL<B>, SdLoadError> {
    let prefix = detect_unet_prefix(file);
    let ch = config.model_channels;
    let time_embed_dim = ch * 4;

    // Time embedding (same as SD 1.x CompVis: time_embed.0, time_embed.2)
    let time_embed_0 = load_linear(
        file,
        &format!("{}.time_embed.0.weight", prefix),
        Some(&format!("{}.time_embed.0.bias", prefix)),
        ch,
        time_embed_dim,
        device,
    )?;

    let time_embed_2 = load_linear(
        file,
        &format!("{}.time_embed.2.weight", prefix),
        Some(&format!("{}.time_embed.2.bias", prefix)),
        time_embed_dim,
        time_embed_dim,
        device,
    )?;

    // Label embedding (SDXL-specific: label_emb.0.0, label_emb.0.2)
    let add_embed_0 = load_linear(
        file,
        &format!("{}.label_emb.0.0.weight", prefix),
        Some(&format!("{}.label_emb.0.0.bias", prefix)),
        config.add_emb_dim,
        time_embed_dim,
        device,
    )?;

    let add_embed_2 = load_linear(
        file,
        &format!("{}.label_emb.0.2.weight", prefix),
        Some(&format!("{}.label_emb.0.2.bias", prefix)),
        time_embed_dim,
        time_embed_dim,
        device,
    )?;

    // Input conv (input_blocks.0.0)
    let conv_in = load_conv2d(
        file,
        &format!("{}.input_blocks.0.0.weight", prefix),
        Some(&format!("{}.input_blocks.0.0.bias", prefix)),
        config.in_channels,
        ch,
        3,
        1,
        device,
    )?;

    // Build down blocks
    // SDXL Base has 3 levels with channel_mult [1, 2, 4] = [320, 640, 1280]
    // Block indices: [1,2], [4,5], [7,8] with downsamplers at [3, 6]
    let mut down_blocks = Vec::new();
    let mut block_idx = 1; // Start after conv_in
    let mut ch_in = ch;

    for (level, &mult) in config.channel_mult.iter().enumerate() {
        let ch_out = ch * mult;
        let is_last = level == config.channel_mult.len() - 1;
        let depth = config.transformer_depth.get(level).copied().unwrap_or(0);
        let num_heads = ch_out / config.head_dim;

        // Check if this level has attention (by checking if transformer key exists)
        let has_attention = file.contains(&format!(
            "{}.input_blocks.{}.1.transformer_blocks.0.attn1.to_q.weight",
            prefix, block_idx
        ));

        let down_block = load_down_block_xl(
            file,
            &prefix,
            block_idx,
            ch_in,
            ch_out,
            time_embed_dim,
            num_heads,
            config.head_dim,
            config.context_dim,
            depth,
            has_attention,
            !is_last,
            device,
        )?;

        down_blocks.push(down_block);

        // Advance block index: 2 res blocks + 1 optional downsample
        block_idx += 2;
        if !is_last {
            block_idx += 1; // downsample
        }
        ch_in = ch_out;
    }

    // Mid block (middle_block)
    let mid_depth = *config.transformer_depth.last().unwrap_or(&10);
    let mid_heads = ch_in / config.head_dim;
    let mid_block = load_mid_block_xl(
        file,
        &prefix,
        ch_in,
        time_embed_dim,
        mid_heads,
        config.head_dim,
        config.context_dim,
        mid_depth,
        device,
    )?;

    // Build up blocks (output_blocks in reverse order)
    // SDXL has 9 output blocks (3 per level * 3 levels)
    let mut up_blocks = Vec::new();
    let num_up_blocks = 9; // For SDXL Base with 3 levels

    for i in 0..num_up_blocks {
        let up_block = load_up_block_xl(file, &prefix, i, config, device)?;
        up_blocks.push(up_block);
    }

    // Output norm and conv
    let norm_out = load_group_norm(
        file,
        &format!("{}.out.0.weight", prefix),
        &format!("{}.out.0.bias", prefix),
        32,
        device,
    )?;

    let conv_out = load_conv2d(
        file,
        &format!("{}.out.2.weight", prefix),
        Some(&format!("{}.out.2.bias", prefix)),
        ch,
        config.out_channels,
        3,
        1,
        device,
    )?;

    // Precompute timestep embedding frequencies
    let time_freqs = timestep_freqs(ch, device);

    Ok(UNetXL {
        time_embed_0,
        time_embed_2,
        add_embed_0,
        add_embed_2,
        time_freqs,
        conv_in,
        down_blocks,
        mid_block,
        up_blocks,
        norm_out,
        conv_out,
        model_channels: ch,
    })
}

/// Load SDXL down block from CompVis format
#[allow(clippy::too_many_arguments)]
fn load_down_block_xl<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    block_idx: usize,
    in_ch: usize,
    out_ch: usize,
    time_dim: usize,
    num_heads: usize,
    head_dim: usize,
    context_dim: usize,
    transformer_depth: usize,
    has_attention: bool,
    has_downsample: bool,
    device: &B::Device,
) -> Result<DownBlockXL<B>, SdLoadError> {
    // res1 at input_blocks.{block_idx}.0
    let res1 = load_resblock_compvis(
        file,
        &format!("{}.input_blocks.{}.0", prefix, block_idx),
        in_ch,
        out_ch,
        time_dim,
        device,
    )?;

    // attn1 at input_blocks.{block_idx}.1 (optional)
    let attn1 = if has_attention {
        Some(load_spatial_transformer_compvis(
            file,
            &format!("{}.input_blocks.{}.1", prefix, block_idx),
            out_ch,
            num_heads,
            head_dim,
            context_dim,
            transformer_depth,
            device,
        )?)
    } else {
        None
    };

    // res2 at input_blocks.{block_idx+1}.0
    let res2 = load_resblock_compvis(
        file,
        &format!("{}.input_blocks.{}.0", prefix, block_idx + 1),
        out_ch,
        out_ch,
        time_dim,
        device,
    )?;

    // attn2 at input_blocks.{block_idx+1}.1 (optional)
    let attn2 = if has_attention {
        Some(load_spatial_transformer_compvis(
            file,
            &format!("{}.input_blocks.{}.1", prefix, block_idx + 1),
            out_ch,
            num_heads,
            head_dim,
            context_dim,
            transformer_depth,
            device,
        )?)
    } else {
        None
    };

    // downsample at input_blocks.{block_idx+2}.0.op (optional)
    let downsample = if has_downsample {
        Some(load_downsample_compvis(
            file,
            &format!("{}.input_blocks.{}.0.op", prefix, block_idx + 2),
            out_ch,
            device,
        )?)
    } else {
        None
    };

    Ok(DownBlockXL {
        res1,
        attn1,
        res2,
        attn2,
        downsample,
    })
}

/// Load SDXL mid block from CompVis format
#[allow(clippy::too_many_arguments)]
fn load_mid_block_xl<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    channels: usize,
    time_dim: usize,
    num_heads: usize,
    head_dim: usize,
    context_dim: usize,
    transformer_depth: usize,
    device: &B::Device,
) -> Result<MidBlockXL<B>, SdLoadError> {
    // middle_block.0 = res1
    let res1 = load_resblock_compvis(
        file,
        &format!("{}.middle_block.0", prefix),
        channels,
        channels,
        time_dim,
        device,
    )?;

    // middle_block.1 = attention
    let attn = load_spatial_transformer_compvis(
        file,
        &format!("{}.middle_block.1", prefix),
        channels,
        num_heads,
        head_dim,
        context_dim,
        transformer_depth,
        device,
    )?;

    // middle_block.2 = res2
    let res2 = load_resblock_compvis(
        file,
        &format!("{}.middle_block.2", prefix),
        channels,
        channels,
        time_dim,
        device,
    )?;

    Ok(MidBlockXL { res1, attn, res2 })
}

/// Load SDXL up block from CompVis format
fn load_up_block_xl<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    idx: usize,
    config: &UNetXLConfig,
    device: &B::Device,
) -> Result<UpBlockXL<B>, SdLoadError> {
    let ch = config.model_channels;
    let time_dim = ch * 4;

    // Determine channels for this block based on position
    // output_blocks go from high resolution to low: [1280, 1280, 1280, 640, 640, 640, 320, 320, 320]
    let (in_ch, out_ch, level) = match idx {
        0..=2 => (ch * 4 + ch * 4, ch * 4, 2), // 1280+1280 -> 1280
        3..=5 => (ch * 4 + ch * 2, ch * 2, 1), // 1280+640 -> 640
        6..=8 => (ch * 2 + ch, ch, 0),         // 640+320 -> 320
        _ => {
            return Err(SdLoadError::MissingTensor(format!(
                "Invalid up block index: {}",
                idx
            )));
        }
    };

    let depth = config.transformer_depth.get(level).copied().unwrap_or(0);
    let num_heads = if out_ch > 0 {
        out_ch / config.head_dim
    } else {
        1
    };

    // Check if this block has attention
    let has_attention = file.contains(&format!(
        "{}.output_blocks.{}.1.transformer_blocks.0.attn1.to_q.weight",
        prefix, idx
    ));

    // res at output_blocks.{idx}.0
    let res = load_resblock_compvis(
        file,
        &format!("{}.output_blocks.{}.0", prefix, idx),
        in_ch,
        out_ch,
        time_dim,
        device,
    )?;

    // attn at output_blocks.{idx}.1 (optional)
    let attn = if has_attention {
        Some(load_spatial_transformer_compvis(
            file,
            &format!("{}.output_blocks.{}.1", prefix, idx),
            out_ch,
            num_heads,
            config.head_dim,
            config.context_dim,
            depth,
            device,
        )?)
    } else {
        None
    };

    // Detect if upsample exists by checking for the conv weight
    // Upsample is at .2 if attention exists, else at .1
    // In SDXL, upsamples are at the end of each resolution level (blocks 2, 5)
    let upsample_index = if has_attention { 2 } else { 1 };
    let upsample_key = format!(
        "{}.output_blocks.{}.{}.conv.weight",
        prefix, idx, upsample_index
    );
    let has_upsample = file.contains(&upsample_key);

    let upsample = if has_upsample {
        let upsample_prefix = format!("{}.output_blocks.{}.{}.conv", prefix, idx, upsample_index);
        Some(load_upsample_compvis(
            file,
            &upsample_prefix,
            out_ch,
            device,
        )?)
    } else {
        None
    };

    Ok(UpBlockXL {
        res,
        attn,
        upsample,
    })
}

/// Load UNet using HuggingFace diffusers naming
fn load_unet_hf<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &UNetConfig,
    device: &B::Device,
) -> Result<UNet<B>, SdLoadError> {
    let ch = config.model_channels;
    let time_embed_dim = ch * 4;

    // HF uses "linear_1"/"linear_2"
    let time_embed_0 = load_linear(
        file,
        &format!("{}.time_embed.linear_1.weight", prefix),
        Some(&format!("{}.time_embed.linear_1.bias", prefix)),
        ch,
        time_embed_dim,
        device,
    )?;

    let time_embed_2 = load_linear(
        file,
        &format!("{}.time_embed.linear_2.weight", prefix),
        Some(&format!("{}.time_embed.linear_2.bias", prefix)),
        time_embed_dim,
        time_embed_dim,
        device,
    )?;

    // Input conv
    let conv_in = load_conv2d(
        file,
        &format!("{}.conv_in.weight", prefix),
        Some(&format!("{}.conv_in.bias", prefix)),
        config.in_channels,
        ch,
        3,
        1,
        device,
    )?;

    // Build down blocks
    let mut down_blocks = Vec::new();
    let mut ch_in = ch;

    for (level, &mult) in config.channel_mult.iter().enumerate() {
        let ch_out = ch * mult;
        let is_last = level == config.channel_mult.len() - 1;
        // SD 1.x: levels 0,1,2 have attention, level 3 does not
        let has_attention = level < 3;

        let block = load_down_block(
            file,
            prefix,
            level,
            ch_in,
            ch_out,
            time_embed_dim,
            config.num_heads,
            config.head_dim,
            config.context_dim,
            config.transformer_depth,
            has_attention,
            !is_last,
            device,
        )?;

        down_blocks.push(block);
        ch_in = ch_out;
    }

    // Mid block
    let mid_block = load_mid_block(
        file,
        prefix,
        ch_in,
        time_embed_dim,
        config.num_heads,
        config.head_dim,
        config.context_dim,
        config.transformer_depth,
        device,
    )?;

    // Build up blocks
    let mut up_blocks = Vec::new();
    let mut channels: Vec<usize> = vec![ch];
    for (level, &mult) in config.channel_mult.iter().enumerate() {
        let ch_out = ch * mult;
        let is_last = level == config.channel_mult.len() - 1;
        channels.push(ch_out);
        channels.push(ch_out);
        if !is_last {
            channels.push(ch_out);
        }
    }

    let mut ch_in = ch * config.channel_mult[config.channel_mult.len() - 1];
    let mut block_idx = 0;

    for (level, &mult) in config.channel_mult.iter().rev().enumerate() {
        let ch_out = ch * mult;
        let is_last = level == config.channel_mult.len() - 1;
        // Reversed: level 0 = original level 3 (no attention)
        // levels 1,2,3 = original levels 2,1,0 (have attention)
        let has_attention = level > 0;

        for i in 0..3 {
            let skip_ch = channels.pop().unwrap();
            let block_in = ch_in + skip_ch;
            let block_out = if i == 2 && !is_last {
                ch * config.channel_mult[config.channel_mult.len() - 2 - level]
            } else {
                ch_out
            };

            let upsample = i == 2 && !is_last;

            let block = load_up_block(
                file,
                prefix,
                block_idx,
                block_in,
                block_out,
                time_embed_dim,
                config.num_heads,
                config.head_dim,
                config.context_dim,
                config.transformer_depth,
                has_attention,
                upsample,
                device,
            )?;

            up_blocks.push(block);
            ch_in = block_out;
            block_idx += 1;
        }
    }

    // Output
    let norm_out = load_group_norm(
        file,
        &format!("{}.conv_norm_out.weight", prefix),
        &format!("{}.conv_norm_out.bias", prefix),
        32,
        device,
    )?;

    let conv_out = load_conv2d(
        file,
        &format!("{}.conv_out.weight", prefix),
        Some(&format!("{}.conv_out.bias", prefix)),
        ch,
        config.out_channels,
        3,
        1,
        device,
    )?;

    Ok(UNet {
        time_embed_0,
        time_embed_2,
        time_freqs: timestep_freqs(ch, device),
        conv_in,
        down_blocks,
        mid_block,
        up_blocks,
        norm_out,
        conv_out,
        model_channels: ch,
    })
}

/// Detect the prefix used for UNet weights
fn detect_unet_prefix(file: &SafeTensorFile) -> String {
    let prefixes = ["model.diffusion_model", "unet", ""];

    for prefix in prefixes {
        let test_key = if prefix.is_empty() {
            "conv_in.weight".to_string()
        } else {
            format!("{}.conv_in.weight", prefix)
        };
        if file.contains(&test_key) {
            return prefix.to_string();
        }
    }

    "model.diffusion_model".to_string()
}

/// Load UNet using CompVis/original SD naming
fn load_unet_compvis<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    config: &UNetConfig,
    device: &B::Device,
) -> Result<UNet<B>, SdLoadError> {
    let ch = config.model_channels;
    let time_embed_dim = ch * 4;
    let block_map = CompVisBlockMap::sd1x();

    // CompVis uses "time_embed.0"/"time_embed.2"
    let time_embed_0 = load_linear(
        file,
        &format!("{}.time_embed.0.weight", prefix),
        Some(&format!("{}.time_embed.0.bias", prefix)),
        ch,
        time_embed_dim,
        device,
    )?;

    let time_embed_2 = load_linear(
        file,
        &format!("{}.time_embed.2.weight", prefix),
        Some(&format!("{}.time_embed.2.bias", prefix)),
        time_embed_dim,
        time_embed_dim,
        device,
    )?;

    // Input conv from input_blocks.0.0
    let conv_in = load_conv2d(
        file,
        &format!("{}.input_blocks.0.0.weight", prefix),
        Some(&format!("{}.input_blocks.0.0.bias", prefix)),
        config.in_channels,
        ch,
        3,
        1,
        device,
    )?;

    // Build down blocks
    let mut down_blocks = Vec::new();
    let mut ch_in = ch;

    for (level, &mult) in config.channel_mult.iter().enumerate() {
        let ch_out = ch * mult;
        let is_last = level == config.channel_mult.len() - 1;
        let has_attn = level < 3; // Levels 0,1,2 have attention, level 3 doesn't

        let block = load_down_block_compvis(
            file,
            prefix,
            &block_map,
            level,
            ch_in,
            ch_out,
            time_embed_dim,
            config.num_heads,
            config.head_dim,
            config.context_dim,
            config.transformer_depth,
            !is_last, // has_downsample
            has_attn,
            device,
        )?;

        down_blocks.push(block);
        ch_in = ch_out;
    }

    // Mid block from middle_block.{0,1,2}
    let mid_block = load_mid_block_compvis(
        file,
        prefix,
        ch_in,
        time_embed_dim,
        config.num_heads,
        config.head_dim,
        config.context_dim,
        config.transformer_depth,
        device,
    )?;

    // Build up blocks
    let mut up_blocks = Vec::new();
    let mut channels: Vec<usize> = vec![ch];
    for (level, &mult) in config.channel_mult.iter().enumerate() {
        let ch_out = ch * mult;
        let is_last = level == config.channel_mult.len() - 1;
        channels.push(ch_out);
        channels.push(ch_out);
        if !is_last {
            channels.push(ch_out);
        }
    }

    let mut ch_in = ch * config.channel_mult[config.channel_mult.len() - 1];
    let mut block_idx = 0;

    for (level, &mult) in config.channel_mult.iter().rev().enumerate() {
        let ch_out = ch * mult;
        let is_last = level == config.channel_mult.len() - 1;
        let has_attn = level > 0; // Reversed: level 0 = original level 3 (no attn)

        // Get next level's channel count for transition (if not last level)
        let next_ch = if !is_last {
            ch * config.channel_mult[config.channel_mult.len() - 2 - level]
        } else {
            ch_out
        };

        for i in 0..3 {
            let skip_ch = channels.pop().unwrap();
            let block_in = ch_in + skip_ch;

            // CompVis: ResBlocks maintain level channels, transition at first block of next level
            // For block at i=2 with upsample, the block itself still outputs ch_out,
            // but after upsampling we transition to next_ch at the start of next level
            let block_out = ch_out;

            let upsample = i == 2 && !is_last;

            let block = load_up_block_compvis(
                file,
                prefix,
                block_idx,
                block_in,
                block_out,
                time_embed_dim,
                config.num_heads,
                config.head_dim,
                config.context_dim,
                config.transformer_depth,
                upsample,
                has_attn,
                device,
            )?;

            up_blocks.push(block);

            // After upsample, we transition to next level's channel count
            ch_in = if upsample { next_ch } else { block_out };
            block_idx += 1;
        }
    }

    // Output from out.{0,2}
    let norm_out = load_group_norm(
        file,
        &format!("{}.out.0.weight", prefix),
        &format!("{}.out.0.bias", prefix),
        32,
        device,
    )?;

    let conv_out = load_conv2d(
        file,
        &format!("{}.out.2.weight", prefix),
        Some(&format!("{}.out.2.bias", prefix)),
        ch,
        config.out_channels,
        3,
        1,
        device,
    )?;

    Ok(UNet {
        time_embed_0,
        time_embed_2,
        time_freqs: timestep_freqs(ch, device),
        conv_in,
        down_blocks,
        mid_block,
        up_blocks,
        norm_out,
        conv_out,
        model_channels: ch,
    })
}

/// Load a DownBlock
#[allow(clippy::too_many_arguments)]
fn load_down_block<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    level: usize,
    in_ch: usize,
    out_ch: usize,
    time_dim: usize,
    num_heads: usize,
    head_dim: usize,
    context_dim: usize,
    transformer_depth: usize,
    has_attention: bool,
    has_downsample: bool,
    device: &B::Device,
) -> Result<DownBlock<B>, SdLoadError> {
    let block_prefix = format!("{}.down_blocks.{}", prefix, level);

    let res1 = load_resblock(
        file,
        &format!("{}.resnets.0", block_prefix),
        in_ch,
        out_ch,
        time_dim,
        device,
    )?;
    let attn1 = if has_attention {
        Some(load_spatial_transformer(
            file,
            &format!("{}.attentions.0", block_prefix),
            out_ch,
            num_heads,
            head_dim,
            context_dim,
            transformer_depth,
            device,
        )?)
    } else {
        None
    };
    let res2 = load_resblock(
        file,
        &format!("{}.resnets.1", block_prefix),
        out_ch,
        out_ch,
        time_dim,
        device,
    )?;
    let attn2 = if has_attention {
        Some(load_spatial_transformer(
            file,
            &format!("{}.attentions.1", block_prefix),
            out_ch,
            num_heads,
            head_dim,
            context_dim,
            transformer_depth,
            device,
        )?)
    } else {
        None
    };

    let downsample = if has_downsample {
        Some(load_downsample(
            file,
            &format!("{}.downsamplers.0", block_prefix),
            out_ch,
            device,
        )?)
    } else {
        None
    };

    Ok(DownBlock {
        res1,
        attn1,
        res2,
        attn2,
        downsample,
    })
}

/// Load a MidBlock
#[allow(clippy::too_many_arguments)]
fn load_mid_block<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    channels: usize,
    time_dim: usize,
    num_heads: usize,
    head_dim: usize,
    context_dim: usize,
    transformer_depth: usize,
    device: &B::Device,
) -> Result<MidBlock<B>, SdLoadError> {
    let block_prefix = format!("{}.mid_block", prefix);

    let res1 = load_resblock(
        file,
        &format!("{}.resnets.0", block_prefix),
        channels,
        channels,
        time_dim,
        device,
    )?;
    let attn = load_spatial_transformer(
        file,
        &format!("{}.attentions.0", block_prefix),
        channels,
        num_heads,
        head_dim,
        context_dim,
        transformer_depth,
        device,
    )?;
    let res2 = load_resblock(
        file,
        &format!("{}.resnets.1", block_prefix),
        channels,
        channels,
        time_dim,
        device,
    )?;

    Ok(MidBlock { res1, attn, res2 })
}

/// Load an UpBlock
#[allow(clippy::too_many_arguments)]
fn load_up_block<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    block_idx: usize,
    in_ch: usize,
    out_ch: usize,
    time_dim: usize,
    num_heads: usize,
    head_dim: usize,
    context_dim: usize,
    transformer_depth: usize,
    has_attention: bool,
    has_upsample: bool,
    device: &B::Device,
) -> Result<UpBlock<B>, SdLoadError> {
    let block_prefix = format!("{}.up_blocks.{}", prefix, block_idx);

    let res = load_resblock(
        file,
        &format!("{}.resnets.0", block_prefix),
        in_ch,
        out_ch,
        time_dim,
        device,
    )?;
    let attn = if has_attention {
        Some(load_spatial_transformer(
            file,
            &format!("{}.attentions.0", block_prefix),
            out_ch,
            num_heads,
            head_dim,
            context_dim,
            transformer_depth,
            device,
        )?)
    } else {
        None
    };

    let upsample = if has_upsample {
        Some(load_upsample(
            file,
            &format!("{}.upsamplers.0", block_prefix),
            out_ch,
            device,
        )?)
    } else {
        None
    };

    Ok(UpBlock {
        res,
        attn,
        upsample,
    })
}

/// Load a ResBlock
fn load_resblock<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    time_dim: usize,
    device: &B::Device,
) -> Result<ResBlock<B>, SdLoadError> {
    let norm1 = load_group_norm(
        file,
        &format!("{}.norm1.weight", prefix),
        &format!("{}.norm1.bias", prefix),
        32,
        device,
    )?;

    let conv1 = load_conv2d(
        file,
        &format!("{}.conv1.weight", prefix),
        Some(&format!("{}.conv1.bias", prefix)),
        in_ch,
        out_ch,
        3,
        1,
        device,
    )?;

    let time_emb_proj = load_linear(
        file,
        &format!("{}.time_emb_proj.weight", prefix),
        Some(&format!("{}.time_emb_proj.bias", prefix)),
        time_dim,
        out_ch,
        device,
    )?;

    let norm2 = load_group_norm(
        file,
        &format!("{}.norm2.weight", prefix),
        &format!("{}.norm2.bias", prefix),
        32,
        device,
    )?;

    let conv2 = load_conv2d(
        file,
        &format!("{}.conv2.weight", prefix),
        Some(&format!("{}.conv2.bias", prefix)),
        out_ch,
        out_ch,
        3,
        1,
        device,
    )?;

    let skip_conv = if in_ch != out_ch {
        let conv_key = format!("{}.conv_shortcut.weight", prefix);
        if file.contains(&conv_key) {
            Some(load_conv2d(
                file,
                &conv_key,
                Some(&format!("{}.conv_shortcut.bias", prefix)),
                in_ch,
                out_ch,
                1,
                0,
                device,
            )?)
        } else {
            None
        }
    } else {
        None
    };

    Ok(ResBlock {
        norm1,
        conv1,
        time_emb_proj,
        norm2,
        conv2,
        skip_conv,
    })
}

/// Load a SpatialTransformer
#[allow(clippy::too_many_arguments)]
fn load_spatial_transformer<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    channels: usize,
    num_heads: usize,
    head_dim: usize,
    context_dim: usize,
    depth: usize,
    device: &B::Device,
) -> Result<SpatialTransformer<B>, SdLoadError> {
    let inner_dim = num_heads * head_dim;

    let norm = load_group_norm(
        file,
        &format!("{}.norm.weight", prefix),
        &format!("{}.norm.bias", prefix),
        32,
        device,
    )?;

    let proj_in = load_conv2d(
        file,
        &format!("{}.proj_in.weight", prefix),
        Some(&format!("{}.proj_in.bias", prefix)),
        channels,
        inner_dim,
        1,
        0,
        device,
    )?;

    let mut transformer_blocks = Vec::with_capacity(depth);
    for i in 0..depth {
        let block = load_transformer_block(
            file,
            &format!("{}.transformer_blocks.{}", prefix, i),
            inner_dim,
            num_heads,
            head_dim,
            context_dim,
            device,
        )?;
        transformer_blocks.push(block);
    }

    let proj_out = load_conv2d(
        file,
        &format!("{}.proj_out.weight", prefix),
        Some(&format!("{}.proj_out.bias", prefix)),
        inner_dim,
        channels,
        1,
        0,
        device,
    )?;

    Ok(SpatialTransformer {
        norm,
        proj_in,
        transformer_blocks,
        proj_out,
    })
}

/// Load a TransformerBlock
fn load_transformer_block<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    dim: usize,
    num_heads: usize,
    head_dim: usize,
    context_dim: usize,
    device: &B::Device,
) -> Result<TransformerBlock<B>, SdLoadError> {
    let norm1 = load_layer_norm(
        file,
        &format!("{}.norm1.weight", prefix),
        &format!("{}.norm1.bias", prefix),
        dim,
        device,
    )?;

    let attn1 = load_cross_attention(
        file,
        &format!("{}.attn1", prefix),
        dim,
        num_heads,
        head_dim,
        None, // Self-attention
        device,
    )?;

    let norm2 = load_layer_norm(
        file,
        &format!("{}.norm2.weight", prefix),
        &format!("{}.norm2.bias", prefix),
        dim,
        device,
    )?;

    let attn2 = load_cross_attention(
        file,
        &format!("{}.attn2", prefix),
        dim,
        num_heads,
        head_dim,
        Some(context_dim),
        device,
    )?;

    let norm3 = load_layer_norm(
        file,
        &format!("{}.norm3.weight", prefix),
        &format!("{}.norm3.bias", prefix),
        dim,
        device,
    )?;

    let ff = load_feedforward(file, &format!("{}.ff", prefix), dim, dim * 4, device)?;

    Ok(TransformerBlock {
        norm1,
        attn1,
        norm2,
        attn2,
        norm3,
        ff,
    })
}

/// Load a CrossAttention
fn load_cross_attention<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    dim: usize,
    num_heads: usize,
    head_dim: usize,
    context_dim: Option<usize>,
    device: &B::Device,
) -> Result<CrossAttention<B>, SdLoadError> {
    let inner_dim = num_heads * head_dim;
    let kv_dim = context_dim.unwrap_or(dim);

    let to_q = load_linear(
        file,
        &format!("{}.to_q.weight", prefix),
        None, // CLIP-style attention often has no bias
        dim,
        inner_dim,
        device,
    )?;

    let to_k = load_linear(
        file,
        &format!("{}.to_k.weight", prefix),
        None,
        kv_dim,
        inner_dim,
        device,
    )?;

    let to_v = load_linear(
        file,
        &format!("{}.to_v.weight", prefix),
        None,
        kv_dim,
        inner_dim,
        device,
    )?;

    let to_out = load_linear(
        file,
        &format!("{}.to_out.0.weight", prefix),
        Some(&format!("{}.to_out.0.bias", prefix)),
        inner_dim,
        dim,
        device,
    )?;

    Ok(CrossAttention {
        to_q,
        to_k,
        to_v,
        to_out,
        num_heads,
        head_dim,
    })
}

/// Load a FeedForward
fn load_feedforward<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    dim: usize,
    mult_dim: usize,
    device: &B::Device,
) -> Result<FeedForward<B>, SdLoadError> {
    // GEGLU doubles the projection
    let net_0 = load_linear(
        file,
        &format!("{}.net.0.proj.weight", prefix),
        Some(&format!("{}.net.0.proj.bias", prefix)),
        dim,
        mult_dim * 2,
        device,
    )?;

    let net_2 = load_linear(
        file,
        &format!("{}.net.2.weight", prefix),
        Some(&format!("{}.net.2.bias", prefix)),
        mult_dim,
        dim,
        device,
    )?;

    Ok(FeedForward { net_0, net_2 })
}

/// Load a Downsample
fn load_downsample<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    channels: usize,
    device: &B::Device,
) -> Result<Downsample<B>, SdLoadError> {
    let conv = load_conv2d_strided(
        file,
        &format!("{}.conv.weight", prefix),
        Some(&format!("{}.conv.bias", prefix)),
        channels,
        channels,
        3,
        2,
        1,
        device,
    )?;

    Ok(Downsample { conv })
}

/// Load an Upsample
fn load_upsample<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    channels: usize,
    device: &B::Device,
) -> Result<Upsample<B>, SdLoadError> {
    let conv = load_conv2d(
        file,
        &format!("{}.conv.weight", prefix),
        Some(&format!("{}.conv.bias", prefix)),
        channels,
        channels,
        3,
        1,
        device,
    )?;

    Ok(Upsample { conv })
}

// =============================================================================
// CompVis-specific Block Loaders
// =============================================================================

/// Load a DownBlock using CompVis naming
#[allow(clippy::too_many_arguments)]
fn load_down_block_compvis<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    block_map: &CompVisBlockMap,
    level: usize,
    in_ch: usize,
    out_ch: usize,
    time_dim: usize,
    num_heads: usize,
    _head_dim: usize, // Not used - computed per-block
    context_dim: usize,
    transformer_depth: usize,
    has_downsample: bool,
    has_attn: bool,
    device: &B::Device,
) -> Result<DownBlock<B>, SdLoadError> {
    let indices = &block_map.down_res_indices[level];

    // CompVis SD 1.x: head_dim scales with channels (8 heads, head_dim = channels/8)
    let out_head_dim = out_ch / num_heads;

    // First ResBlock + optional Attention
    let res1_prefix = format!("{}.input_blocks.{}.0", prefix, indices[0]);
    let res1 = load_resblock_compvis(file, &res1_prefix, in_ch, out_ch, time_dim, device)?;

    let attn1 = if has_attn {
        let attn1_prefix = format!("{}.input_blocks.{}.1", prefix, indices[0]);
        Some(load_spatial_transformer_compvis(
            file,
            &attn1_prefix,
            out_ch,
            num_heads,
            out_head_dim,
            context_dim,
            transformer_depth,
            device,
        )?)
    } else {
        None
    };

    // Second ResBlock + optional Attention
    let res2_prefix = format!("{}.input_blocks.{}.0", prefix, indices[1]);
    let res2 = load_resblock_compvis(file, &res2_prefix, out_ch, out_ch, time_dim, device)?;

    let attn2 = if has_attn {
        let attn2_prefix = format!("{}.input_blocks.{}.1", prefix, indices[1]);
        Some(load_spatial_transformer_compvis(
            file,
            &attn2_prefix,
            out_ch,
            num_heads,
            out_head_dim,
            context_dim,
            transformer_depth,
            device,
        )?)
    } else {
        None
    };

    // Downsample if needed
    let downsample = if has_downsample {
        let ds_idx = block_map.down_sample_indices[level].unwrap();
        let ds_prefix = format!("{}.input_blocks.{}.0.op", prefix, ds_idx);
        Some(load_downsample_compvis(file, &ds_prefix, out_ch, device)?)
    } else {
        None
    };

    Ok(DownBlock {
        res1,
        attn1,
        res2,
        attn2,
        downsample,
    })
}

/// Load a MidBlock using CompVis naming
#[allow(clippy::too_many_arguments)]
fn load_mid_block_compvis<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    channels: usize,
    time_dim: usize,
    num_heads: usize,
    _head_dim: usize, // Not used - computed from channels
    context_dim: usize,
    transformer_depth: usize,
    device: &B::Device,
) -> Result<MidBlock<B>, SdLoadError> {
    // CompVis SD 1.x: head_dim scales with channels
    let head_dim = channels / num_heads;

    // middle_block.0 = ResBlock
    let res1 = load_resblock_compvis(
        file,
        &format!("{}.middle_block.0", prefix),
        channels,
        channels,
        time_dim,
        device,
    )?;

    // middle_block.1 = SpatialTransformer
    let attn = load_spatial_transformer_compvis(
        file,
        &format!("{}.middle_block.1", prefix),
        channels,
        num_heads,
        head_dim,
        context_dim,
        transformer_depth,
        device,
    )?;

    // middle_block.2 = ResBlock
    let res2 = load_resblock_compvis(
        file,
        &format!("{}.middle_block.2", prefix),
        channels,
        channels,
        time_dim,
        device,
    )?;

    Ok(MidBlock { res1, attn, res2 })
}

/// Load an UpBlock using CompVis naming
#[allow(clippy::too_many_arguments)]
fn load_up_block_compvis<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    block_idx: usize,
    in_ch: usize,
    out_ch: usize,
    time_dim: usize,
    num_heads: usize,
    _head_dim: usize, // Not used - computed from channels
    context_dim: usize,
    transformer_depth: usize,
    has_upsample: bool,
    has_attn: bool,
    device: &B::Device,
) -> Result<UpBlock<B>, SdLoadError> {
    // CompVis SD 1.x: head_dim scales with channels
    let out_head_dim = out_ch / num_heads;

    // output_blocks.{block_idx}.0 = ResBlock
    let res_prefix = format!("{}.output_blocks.{}.0", prefix, block_idx);
    let res = load_resblock_compvis(file, &res_prefix, in_ch, out_ch, time_dim, device)?;

    // output_blocks.{block_idx}.1 = SpatialTransformer (if has_attn) or Upsample (if has_upsample and no attn)
    let attn = if has_attn {
        let attn_prefix = format!("{}.output_blocks.{}.1", prefix, block_idx);
        Some(load_spatial_transformer_compvis(
            file,
            &attn_prefix,
            out_ch,
            num_heads,
            out_head_dim,
            context_dim,
            transformer_depth,
            device,
        )?)
    } else {
        None
    };

    // Upsample: could be at .1 (if no attn) or .2 (if attn present)
    let upsample = if has_upsample {
        let us_suffix = if has_attn { "2" } else { "1" };
        let us_prefix = format!("{}.output_blocks.{}.{}.conv", prefix, block_idx, us_suffix);
        Some(load_upsample_compvis(file, &us_prefix, out_ch, device)?)
    } else {
        None
    };

    Ok(UpBlock {
        res,
        attn,
        upsample,
    })
}

/// Load a ResBlock using CompVis naming
///
/// CompVis uses:
/// - in_layers.0 = GroupNorm (norm1)
/// - in_layers.2 = Conv2d (conv1)
/// - emb_layers.1 = Linear (time_emb_proj)
/// - out_layers.0 = GroupNorm (norm2)
/// - out_layers.3 = Conv2d (conv2)
/// - skip_connection = Conv2d (skip_conv)
fn load_resblock_compvis<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    time_dim: usize,
    device: &B::Device,
) -> Result<ResBlock<B>, SdLoadError> {
    let norm1 = load_group_norm(
        file,
        &format!("{}.in_layers.0.weight", prefix),
        &format!("{}.in_layers.0.bias", prefix),
        32,
        device,
    )?;

    let conv1 = load_conv2d(
        file,
        &format!("{}.in_layers.2.weight", prefix),
        Some(&format!("{}.in_layers.2.bias", prefix)),
        in_ch,
        out_ch,
        3,
        1,
        device,
    )?;

    let time_emb_proj = load_linear(
        file,
        &format!("{}.emb_layers.1.weight", prefix),
        Some(&format!("{}.emb_layers.1.bias", prefix)),
        time_dim,
        out_ch,
        device,
    )?;

    let norm2 = load_group_norm(
        file,
        &format!("{}.out_layers.0.weight", prefix),
        &format!("{}.out_layers.0.bias", prefix),
        32,
        device,
    )?;

    let conv2 = load_conv2d(
        file,
        &format!("{}.out_layers.3.weight", prefix),
        Some(&format!("{}.out_layers.3.bias", prefix)),
        out_ch,
        out_ch,
        3,
        1,
        device,
    )?;

    let skip_conv = if in_ch != out_ch {
        let conv_key = format!("{}.skip_connection.weight", prefix);
        if file.contains(&conv_key) {
            Some(load_conv2d(
                file,
                &conv_key,
                Some(&format!("{}.skip_connection.bias", prefix)),
                in_ch,
                out_ch,
                1,
                0,
                device,
            )?)
        } else {
            None
        }
    } else {
        None
    };

    Ok(ResBlock {
        norm1,
        conv1,
        time_emb_proj,
        norm2,
        conv2,
        skip_conv,
    })
}

/// Load a SpatialTransformer using CompVis naming
///
/// Note: Some CompVis models store proj_in/proj_out as Conv2d, others as Linear.
/// We detect which by checking tensor dimensionality.
#[allow(clippy::too_many_arguments)]
fn load_spatial_transformer_compvis<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    channels: usize,
    num_heads: usize,
    head_dim: usize,
    context_dim: usize,
    depth: usize,
    device: &B::Device,
) -> Result<SpatialTransformer<B>, SdLoadError> {
    let inner_dim = num_heads * head_dim;

    let norm = load_group_norm(
        file,
        &format!("{}.norm.weight", prefix),
        &format!("{}.norm.bias", prefix),
        32,
        device,
    )?;

    // Check if proj_in is stored as Conv2d (4D) or Linear (2D)
    let proj_in_key = format!("{}.proj_in.weight", prefix);
    let proj_in_shape = file
        .shape(&proj_in_key)
        .ok_or_else(|| SdLoadError::MissingTensor(proj_in_key.clone()))?;

    let proj_in = if proj_in_shape.len() == 4 {
        // Already Conv2d format
        load_conv2d(
            file,
            &proj_in_key,
            Some(&format!("{}.proj_in.bias", prefix)),
            channels,
            inner_dim,
            1,
            0,
            device,
        )?
    } else {
        // Linear format - need to reshape
        load_linear_as_conv2d(
            file,
            &proj_in_key,
            Some(&format!("{}.proj_in.bias", prefix)),
            channels,
            inner_dim,
            device,
        )?
    };

    let mut transformer_blocks = Vec::with_capacity(depth);
    for i in 0..depth {
        let block = load_transformer_block(
            file,
            &format!("{}.transformer_blocks.{}", prefix, i),
            inner_dim,
            num_heads,
            head_dim,
            context_dim,
            device,
        )?;
        transformer_blocks.push(block);
    }

    let proj_out_key = format!("{}.proj_out.weight", prefix);
    let proj_out_shape = file
        .shape(&proj_out_key)
        .ok_or_else(|| SdLoadError::MissingTensor(proj_out_key.clone()))?;

    let proj_out = if proj_out_shape.len() == 4 {
        // Already Conv2d format
        load_conv2d(
            file,
            &proj_out_key,
            Some(&format!("{}.proj_out.bias", prefix)),
            inner_dim,
            channels,
            1,
            0,
            device,
        )?
    } else {
        // Linear format - need to reshape
        load_linear_as_conv2d(
            file,
            &proj_out_key,
            Some(&format!("{}.proj_out.bias", prefix)),
            inner_dim,
            channels,
            device,
        )?
    };

    Ok(SpatialTransformer {
        norm,
        proj_in,
        transformer_blocks,
        proj_out,
    })
}

/// Load a Linear layer's weights as a Conv2d 1x1
fn load_linear_as_conv2d<B: Backend>(
    file: &SafeTensorFile,
    weight_key: &str,
    bias_key: Option<&str>,
    in_features: usize,
    out_features: usize,
    device: &B::Device,
) -> Result<burn::nn::conv::Conv2d<B>, SdLoadError> {
    let weight: Tensor<B, 2> = file
        .load_f32(weight_key, device)
        .map_err(|e| SdLoadError::MissingTensor(format!("{}: {}", weight_key, e)))?;

    // Reshape [out, in] -> [out, in, 1, 1]
    let weight = weight.reshape([out_features, in_features, 1, 1]);

    let has_bias = bias_key.map(|k| file.contains(k)).unwrap_or(false);
    let mut conv = Conv2dConfig::new([in_features, out_features], [1, 1])
        .with_bias(has_bias)
        .init(device);

    conv.weight = Param::from_tensor(weight);

    if let Some(bias_k) = bias_key {
        if file.contains(bias_k) {
            let bias: Tensor<B, 1> = file.load_f32(bias_k, device)?;
            conv.bias = Some(Param::from_tensor(bias));
        }
    }

    Ok(conv)
}

/// Load a Downsample using CompVis naming (weight directly at prefix)
fn load_downsample_compvis<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    channels: usize,
    device: &B::Device,
) -> Result<Downsample<B>, SdLoadError> {
    let conv = load_conv2d_strided(
        file,
        &format!("{}.weight", prefix),
        Some(&format!("{}.bias", prefix)),
        channels,
        channels,
        3,
        2,
        1,
        device,
    )?;

    Ok(Downsample { conv })
}

/// Load an Upsample using CompVis naming (weight directly at prefix)
fn load_upsample_compvis<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    channels: usize,
    device: &B::Device,
) -> Result<Upsample<B>, SdLoadError> {
    let conv = load_conv2d(
        file,
        &format!("{}.weight", prefix),
        Some(&format!("{}.bias", prefix)),
        channels,
        channels,
        3,
        1,
        device,
    )?;

    Ok(Upsample { conv })
}

// =============================================================================
// VAE Decoder Loading
// =============================================================================

/// Load VAE decoder from a SafeTensorFile
fn load_vae_decoder_from_file<B: Backend>(
    file: &SafeTensorFile,
    config: &sketchpad_vae::DecoderConfig,
    device: &B::Device,
) -> Result<sketchpad_vae::Decoder<B>, SdLoadError> {
    let debug = std::env::var("SD_DEBUG").is_ok();
    let prefix = detect_vae_decoder_prefix(file);
    if debug {
        eprintln!("[vae_loader] Using prefix: {}", prefix);
        eprintln!(
            "[vae_loader] Config: base_ch={}, ch_mult={:?}, num_res_blocks={}",
            config.base_channels, config.channel_mult, config.num_res_blocks
        );
    }
    let ch = config.base_channels;
    let ch_mult = &config.channel_mult;

    // Start with highest channel count (reversed from encoder)
    let block_in = ch * ch_mult[ch_mult.len() - 1];

    // Input conv: latent_channels -> block_in
    let conv_in = load_conv2d(
        file,
        &format!("{}.conv_in.weight", prefix),
        Some(&format!("{}.conv_in.bias", prefix)),
        config.latent_channels,
        block_in,
        3,
        1,
        device,
    )?;

    // Mid blocks
    let mid_block1 = load_vae_resnet_block(
        file,
        &format!("{}.mid.block_1", prefix),
        block_in,
        block_in,
        device,
    )?;
    let mid_attn =
        load_vae_self_attention(file, &format!("{}.mid.attn_1", prefix), block_in, device)?;
    let mid_block2 = load_vae_resnet_block(
        file,
        &format!("{}.mid.block_2", prefix),
        block_in,
        block_in,
        device,
    )?;

    // Up blocks (reverse order, with upsampling)
    let mut up_blocks = Vec::new();
    let mut in_ch = block_in;

    for (i, &mult) in ch_mult.iter().rev().enumerate() {
        let out_ch = ch * mult;
        let upsample = i < ch_mult.len() - 1; // Don't upsample on last block
        let block_idx = ch_mult.len() - 1 - i; // Map to original block index (3, 2, 1, 0)

        if debug {
            eprintln!(
                "[vae_loader] Loading up_block[{}] from up.{}: in_ch={}, out_ch={}, upsample={}",
                i, block_idx, in_ch, out_ch, upsample
            );
        }

        let block = load_vae_decoder_block(
            file,
            &prefix,
            block_idx,
            in_ch,
            out_ch,
            config.num_res_blocks,
            upsample,
            device,
        )?;
        up_blocks.push(block);
        in_ch = out_ch;
    }

    // Output layers
    let norm_out = load_group_norm(
        file,
        &format!("{}.norm_out.weight", prefix),
        &format!("{}.norm_out.bias", prefix),
        32,
        device,
    )?;

    let conv_out = load_conv2d(
        file,
        &format!("{}.conv_out.weight", prefix),
        Some(&format!("{}.conv_out.bias", prefix)),
        in_ch,
        config.out_channels,
        3,
        1,
        device,
    )?;

    // Load post_quant_conv if present (1x1 conv applied to latent before decoding)
    // The key is at the VAE root level, not under decoder
    let vae_root = if prefix.contains(".decoder") {
        prefix.replace(".decoder", "")
    } else {
        "first_stage_model".to_string()
    };
    let post_quant_key = format!("{}.post_quant_conv.weight", vae_root);
    let post_quant_conv = if file.contains(&post_quant_key) {
        if debug {
            eprintln!(
                "[vae_loader] Loading post_quant_conv from {}",
                post_quant_key
            );
        }
        let weight: Tensor<B, 4> = file.load_f32(&post_quant_key, device)?;
        let bias_key = format!("{}.post_quant_conv.bias", vae_root);
        let bias: Option<Tensor<B, 1>> = if file.contains(&bias_key) {
            Some(file.load_f32(&bias_key, device)?)
        } else {
            None
        };

        // post_quant_conv is typically [4, 4, 1, 1] - a 1x1 conv
        let [out_ch, in_ch, _, _] = weight.dims();
        let mut conv = Conv2dConfig::new([in_ch, out_ch], [1, 1])
            .with_bias(bias.is_some())
            .init(device);
        conv.weight = Param::from_tensor(weight);
        if let Some(b) = bias {
            conv.bias = Some(Param::from_tensor(b));
        }
        Some(conv)
    } else {
        if debug {
            eprintln!("[vae_loader] No post_quant_conv found");
        }
        None
    };

    Ok(sketchpad_vae::Decoder {
        post_quant_conv,
        conv_in,
        mid_block1,
        mid_attn,
        mid_block2,
        up_blocks,
        norm_out,
        conv_out,
        skip_attention: false,
        debug: false,
        clamp_overflow: false,
    })
}

/// Detect the prefix used for VAE decoder weights
fn detect_vae_decoder_prefix(file: &SafeTensorFile) -> String {
    let prefixes = ["decoder", "vae.decoder", "first_stage_model.decoder"];

    for prefix in prefixes {
        let test_key = format!("{}.conv_in.weight", prefix);
        if file.contains(&test_key) {
            return prefix.to_string();
        }
    }

    "decoder".to_string()
}

/// Load a VAE decoder block with residual blocks and optional upsampling
#[allow(clippy::too_many_arguments)]
fn load_vae_decoder_block<B: Backend>(
    file: &SafeTensorFile,
    vae_prefix: &str,
    block_idx: usize,
    in_ch: usize,
    out_ch: usize,
    num_blocks: usize,
    upsample: bool,
    device: &B::Device,
) -> Result<sketchpad_vae::DecoderBlock<B>, SdLoadError> {
    let block_prefix = format!("{}.up.{}", vae_prefix, block_idx);

    let mut res_blocks = Vec::with_capacity(num_blocks);

    // First block handles channel change
    res_blocks.push(load_vae_resnet_block(
        file,
        &format!("{}.block.0", block_prefix),
        in_ch,
        out_ch,
        device,
    )?);

    // Remaining blocks maintain channels
    for i in 1..num_blocks {
        res_blocks.push(load_vae_resnet_block(
            file,
            &format!("{}.block.{}", block_prefix, i),
            out_ch,
            out_ch,
            device,
        )?);
    }

    let upsample_layer = if upsample {
        Some(load_vae_upsample(
            file,
            &format!("{}.upsample", block_prefix),
            out_ch,
            device,
        )?)
    } else {
        None
    };

    Ok(sketchpad_vae::DecoderBlock {
        res_blocks,
        upsample: upsample_layer,
    })
}

/// Load a VAE ResnetBlock
fn load_vae_resnet_block<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    device: &B::Device,
) -> Result<sketchpad_vae::ResnetBlock<B>, SdLoadError> {
    let norm1 = load_group_norm(
        file,
        &format!("{}.norm1.weight", prefix),
        &format!("{}.norm1.bias", prefix),
        32,
        device,
    )?;

    let conv1 = load_conv2d(
        file,
        &format!("{}.conv1.weight", prefix),
        Some(&format!("{}.conv1.bias", prefix)),
        in_ch,
        out_ch,
        3,
        1,
        device,
    )?;

    let norm2 = load_group_norm(
        file,
        &format!("{}.norm2.weight", prefix),
        &format!("{}.norm2.bias", prefix),
        32,
        device,
    )?;

    let conv2 = load_conv2d(
        file,
        &format!("{}.conv2.weight", prefix),
        Some(&format!("{}.conv2.bias", prefix)),
        out_ch,
        out_ch,
        3,
        1,
        device,
    )?;

    let skip_conv = if in_ch != out_ch {
        let conv_key = format!("{}.nin_shortcut.weight", prefix);
        if file.contains(&conv_key) {
            Some(load_conv2d(
                file,
                &conv_key,
                Some(&format!("{}.nin_shortcut.bias", prefix)),
                in_ch,
                out_ch,
                1,
                0,
                device,
            )?)
        } else {
            None
        }
    } else {
        None
    };

    Ok(sketchpad_vae::ResnetBlock {
        norm1,
        conv1,
        norm2,
        conv2,
        skip_conv,
    })
}

/// Load a VAE SelfAttention block
fn load_vae_self_attention<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    channels: usize,
    device: &B::Device,
) -> Result<sketchpad_vae::SelfAttention<B>, SdLoadError> {
    let norm = load_group_norm(
        file,
        &format!("{}.norm.weight", prefix),
        &format!("{}.norm.bias", prefix),
        32,
        device,
    )?;

    let q = load_conv2d(
        file,
        &format!("{}.q.weight", prefix),
        Some(&format!("{}.q.bias", prefix)),
        channels,
        channels,
        1,
        0,
        device,
    )?;

    let k = load_conv2d(
        file,
        &format!("{}.k.weight", prefix),
        Some(&format!("{}.k.bias", prefix)),
        channels,
        channels,
        1,
        0,
        device,
    )?;

    let v = load_conv2d(
        file,
        &format!("{}.v.weight", prefix),
        Some(&format!("{}.v.bias", prefix)),
        channels,
        channels,
        1,
        0,
        device,
    )?;

    let proj_out = load_conv2d(
        file,
        &format!("{}.proj_out.weight", prefix),
        Some(&format!("{}.proj_out.bias", prefix)),
        channels,
        channels,
        1,
        0,
        device,
    )?;

    Ok(sketchpad_vae::SelfAttention {
        norm,
        q,
        k,
        v,
        proj_out,
        channels,
    })
}

/// Load a VAE Upsample layer
fn load_vae_upsample<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    channels: usize,
    device: &B::Device,
) -> Result<sketchpad_vae::Upsample<B>, SdLoadError> {
    let conv = load_conv2d(
        file,
        &format!("{}.conv.weight", prefix),
        Some(&format!("{}.conv.bias", prefix)),
        channels,
        channels,
        3,
        1,
        device,
    )?;

    Ok(sketchpad_vae::Upsample { conv })
}

// =============================================================================
// VAE Encoder Loading
// =============================================================================

/// Load VAE encoder from a SafeTensorFile
fn load_vae_encoder_from_file<B: Backend>(
    file: &SafeTensorFile,
    config: &sketchpad_vae::EncoderConfig,
    device: &B::Device,
) -> Result<sketchpad_vae::Encoder<B>, SdLoadError> {
    let debug = std::env::var("SD_DEBUG").is_ok();
    let prefix = detect_vae_encoder_prefix(file);
    if debug {
        eprintln!("[vae_encoder_loader] Using prefix: {}", prefix);
        eprintln!(
            "[vae_encoder_loader] Config: base_ch={}, ch_mult={:?}, num_res_blocks={}",
            config.base_channels, config.channel_mult, config.num_res_blocks
        );
    }
    let ch = config.base_channels;
    let ch_mult = &config.channel_mult;

    // Input conv: in_channels -> base_channels
    let conv_in = load_conv2d(
        file,
        &format!("{}.conv_in.weight", prefix),
        Some(&format!("{}.conv_in.bias", prefix)),
        config.in_channels,
        ch,
        3,
        1,
        device,
    )?;

    // Down blocks
    let mut down_blocks = Vec::new();
    let mut in_ch = ch;

    for (i, &mult) in ch_mult.iter().enumerate() {
        let out_ch = ch * mult;
        let downsample = i < ch_mult.len() - 1; // Don't downsample on last block

        if debug {
            eprintln!(
                "[vae_encoder_loader] Loading down_block[{}] from down.{}: in_ch={}, out_ch={}, downsample={}",
                i, i, in_ch, out_ch, downsample
            );
        }

        let block = load_vae_encoder_block(
            file,
            &prefix,
            i,
            in_ch,
            out_ch,
            config.num_res_blocks,
            downsample,
            device,
        )?;
        down_blocks.push(block);
        in_ch = out_ch;
    }

    // Mid blocks
    let mid_ch = ch * ch_mult[ch_mult.len() - 1];
    let mid_block1 = load_vae_resnet_block(
        file,
        &format!("{}.mid.block_1", prefix),
        mid_ch,
        mid_ch,
        device,
    )?;
    let mid_attn =
        load_vae_self_attention(file, &format!("{}.mid.attn_1", prefix), mid_ch, device)?;
    let mid_block2 = load_vae_resnet_block(
        file,
        &format!("{}.mid.block_2", prefix),
        mid_ch,
        mid_ch,
        device,
    )?;

    // Output layers
    let norm_out = load_group_norm(
        file,
        &format!("{}.norm_out.weight", prefix),
        &format!("{}.norm_out.bias", prefix),
        32,
        device,
    )?;

    let conv_out = load_conv2d(
        file,
        &format!("{}.conv_out.weight", prefix),
        Some(&format!("{}.conv_out.bias", prefix)),
        mid_ch,
        config.latent_channels,
        3,
        1,
        device,
    )?;

    Ok(sketchpad_vae::Encoder {
        conv_in,
        down_blocks,
        mid_block1,
        mid_attn,
        mid_block2,
        norm_out,
        conv_out,
    })
}

/// Detect the prefix used for VAE encoder weights
fn detect_vae_encoder_prefix(file: &SafeTensorFile) -> String {
    let prefixes = ["encoder", "vae.encoder", "first_stage_model.encoder"];

    for prefix in prefixes {
        let test_key = format!("{}.conv_in.weight", prefix);
        if file.contains(&test_key) {
            return prefix.to_string();
        }
    }

    "encoder".to_string()
}

/// Load a VAE encoder block with residual blocks and optional downsampling
#[allow(clippy::too_many_arguments)]
fn load_vae_encoder_block<B: Backend>(
    file: &SafeTensorFile,
    vae_prefix: &str,
    block_idx: usize,
    in_ch: usize,
    out_ch: usize,
    num_blocks: usize,
    downsample: bool,
    device: &B::Device,
) -> Result<sketchpad_vae::encoder::EncoderBlock<B>, SdLoadError> {
    let block_prefix = format!("{}.down.{}", vae_prefix, block_idx);

    let mut res_blocks = Vec::with_capacity(num_blocks);

    // First block handles channel change
    res_blocks.push(load_vae_resnet_block(
        file,
        &format!("{}.block.0", block_prefix),
        in_ch,
        out_ch,
        device,
    )?);

    // Remaining blocks maintain channels
    for i in 1..num_blocks {
        res_blocks.push(load_vae_resnet_block(
            file,
            &format!("{}.block.{}", block_prefix, i),
            out_ch,
            out_ch,
            device,
        )?);
    }

    let downsample_layer = if downsample {
        Some(load_vae_downsample(
            file,
            &format!("{}.downsample", block_prefix),
            out_ch,
            device,
        )?)
    } else {
        None
    };

    Ok(sketchpad_vae::encoder::EncoderBlock {
        res_blocks,
        downsample: downsample_layer,
    })
}

/// Load a VAE Downsample layer (strided conv with asymmetric padding)
fn load_vae_downsample<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    channels: usize,
    device: &B::Device,
) -> Result<sketchpad_vae::encoder::Downsample<B>, SdLoadError> {
    // VAE downsample uses asymmetric padding (0, 1) for proper spatial halving
    let weight: Tensor<B, 4> = file
        .load_f32(&format!("{}.conv.weight", prefix), device)
        .map_err(|e| SdLoadError::MissingTensor(format!("{}.conv.weight: {}", prefix, e)))?;

    let mut conv = Conv2dConfig::new([channels, channels], [3, 3])
        .with_stride([2, 2])
        .with_padding(PaddingConfig2d::Explicit(0, 1))
        .with_bias(true)
        .init(device);

    conv.weight = Param::from_tensor(weight);

    let bias_key = format!("{}.conv.bias", prefix);
    if file.contains(&bias_key) {
        let bias: Tensor<B, 1> = file.load_f32(&bias_key, device)?;
        conv.bias = Some(Param::from_tensor(bias));
    }

    Ok(sketchpad_vae::encoder::Downsample { conv })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_clip_prefix() {
        // This would need a mock SafeTensorFile to test properly
    }

    #[test]
    fn test_sd_load_error_display() {
        let err = SdLoadError::MissingTensor("text_model.embeddings.weight".to_string());
        assert!(err.to_string().contains("Missing tensor"));
    }
}
