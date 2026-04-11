//! LLM Inference API
//!
//! Provides a unified interface for text generation across all supported LLM models.
//! This module is the main entry point for using burn-models-llm as a library.
//!
//! # Example
//!
//! ```ignore
//! use sketchpad_llm::{LlmInstance, ModelType, GenerationConfig};
//!
//! // Load any model
//! let llm = LlmInstance::load(ModelType::Llama, "./Meta-Llama-3-8B/", &device)?;
//!
//! // Generate text
//! let config = GenerationConfig::default();
//! let output = llm.generate("Once upon a time", &config)?;
//! println!("{}", output);
//! ```

use std::path::Path;

use burn::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokenizers::Tokenizer;

use crate::{
    deepseek::{DeepSeek, DeepSeekConfig, DeepSeekRuntime},
    deepseek_loader::load_deepseek,
    gemma::{Gemma, GemmaConfig, GemmaRuntime},
    gemma_loader::load_gemma,
    gemma4::{Gemma4, Gemma4Config, Gemma4Runtime},
    gemma4_gguf_loader::load_gemma4_gguf,
    gemma4_loader::load_gemma4,
    jamba::{Jamba, JambaConfig, JambaRuntime},
    jamba_loader::load_jamba,
    llama::{Llama, LlamaConfig, LlamaRuntime},
    llama_loader::load_llama,
    mamba::{Mamba, MambaConfig, MambaRuntime},
    mamba_loader::load_mamba,
    mistral::{Mistral, MistralConfig, MistralRuntime},
    mistral_loader::load_mistral,
    mixtral::{Mixtral, MixtralConfig, MixtralRuntime},
    mixtral_loader::load_mixtral,
    phi::{Phi, PhiConfig, PhiRuntime},
    phi_loader::load_phi,
    qwen::{Qwen, QwenConfig, QwenRuntime},
    qwen_loader::load_qwen,
    rwkv::{Rwkv, RwkvConfig, RwkvRuntime},
    rwkv_loader::load_rwkv,
};

/// Errors that can occur during LLM operations
#[derive(Error, Debug)]
pub enum LlmError {
    #[error("Failed to load model: {0}")]
    LoadError(String),

    #[error("Failed to load tokenizer: {0}")]
    TokenizerError(String),

    #[error("Failed to load config: {0}")]
    ConfigError(String),

    #[error("Tokenization error: {0}")]
    TokenizationError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Supported LLM model types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    /// LLaMA 2/3 models
    Llama,
    /// Mistral 7B
    Mistral,
    /// Mixtral MoE
    Mixtral,
    /// Gemma 2
    Gemma,
    /// Gemma 4 MoE
    Gemma4,
    /// Phi-2/3
    Phi,
    /// Qwen 1.5/2
    Qwen,
    /// DeepSeek
    DeepSeek,
    /// RWKV-7
    Rwkv,
    /// Mamba SSM
    Mamba,
    /// Jamba hybrid
    Jamba,
}

impl std::str::FromStr for ModelType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "llama" => Ok(Self::Llama),
            "mistral" => Ok(Self::Mistral),
            "mixtral" => Ok(Self::Mixtral),
            "gemma" | "gemma2" => Ok(Self::Gemma),
            "gemma4" => Ok(Self::Gemma4),
            "phi" => Ok(Self::Phi),
            "qwen" => Ok(Self::Qwen),
            "deepseek" => Ok(Self::DeepSeek),
            "rwkv" => Ok(Self::Rwkv),
            "mamba" => Ok(Self::Mamba),
            "jamba" => Ok(Self::Jamba),
            _ => Err(()),
        }
    }
}

impl ModelType {
    /// Get the model type name as a string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Mistral => "mistral",
            Self::Mixtral => "mixtral",
            Self::Gemma => "gemma",
            Self::Gemma4 => "gemma4",
            Self::Phi => "phi",
            Self::Qwen => "qwen",
            Self::DeepSeek => "deepseek",
            Self::Rwkv => "rwkv",
            Self::Mamba => "mamba",
            Self::Jamba => "jamba",
        }
    }
}

/// Configuration for text generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate
    pub max_tokens: usize,
    /// Sampler configuration (temperature, top-k, top-p, DRY, XTC, etc.)
    #[serde(skip)]
    pub sampler: crate::sampling::SamplerConfig,
    /// Stop sequences - generation stops when any of these are produced
    pub stop_sequences: Vec<String>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            sampler: crate::sampling::SamplerConfig::default(),
            stop_sequences: Vec::new(),
        }
    }
}

impl GenerationConfig {
    /// Create a new config with specified max tokens
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            ..Default::default()
        }
    }

    /// Set the temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.sampler.temperature = temperature;
        self
    }

    /// Set top-p sampling
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.sampler.top_p = top_p;
        self
    }

    /// Set the sampler configuration
    pub fn with_sampler(mut self, sampler: crate::sampling::SamplerConfig) -> Self {
        self.sampler = sampler;
        self
    }

    /// Add a stop sequence
    pub fn with_stop_sequence(mut self, stop: impl Into<String>) -> Self {
        self.stop_sequences.push(stop.into());
        self
    }

    /// Set stop sequences
    pub fn with_stop_sequences(mut self, stops: Vec<String>) -> Self {
        self.stop_sequences = stops;
        self
    }
}

/// Internal enum wrapping model + runtime pairs
enum ModelInstance<B: Backend> {
    Llama(Llama<B>, LlamaRuntime<B>),
    Mistral(Mistral<B>, MistralRuntime<B>),
    Mixtral(Mixtral<B>, MixtralRuntime<B>),
    Gemma(Gemma<B>, GemmaRuntime<B>),
    Gemma4(Gemma4<B>, Gemma4Runtime<B>),
    Phi(Phi<B>, PhiRuntime<B>),
    Qwen(Qwen<B>, QwenRuntime<B>),
    DeepSeek(DeepSeek<B>, DeepSeekRuntime<B>),
    Rwkv(Rwkv<B>, RwkvRuntime<B>),
    Mamba(Mamba<B>, MambaRuntime<B>),
    Jamba(Jamba<B>, JambaRuntime<B>),
}

/// Unified LLM instance that can load and run any supported model
///
/// This struct provides a consistent interface for text generation
/// regardless of the underlying model architecture.
pub struct LlmInstance<B: Backend> {
    model: ModelInstance<B>,
    tokenizer: Tokenizer,
    model_type: ModelType,
}

impl<B: Backend> LlmInstance<B> {
    /// Load a model from a weights directory
    ///
    /// The weights directory should contain:
    /// - Model weights in safetensors format
    /// - `tokenizer.json` for tokenization
    /// - `config.json` for model configuration
    ///
    /// # Arguments
    ///
    /// * `model_type` - The type of model to load
    /// * `weights_path` - Path to the model weights directory
    /// * `device` - Device to load the model onto
    pub fn load<P: AsRef<Path>>(
        model_type: ModelType,
        weights_path: P,
        device: &B::Device,
    ) -> Result<Self, LlmError> {
        let path = weights_path.as_ref();

        // Load tokenizer
        let tokenizer_path = path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            LlmError::TokenizerError(format!("{}: {}", tokenizer_path.display(), e))
        })?;

        // Load config and model based on type
        let model = Self::load_model_instance(model_type, path, device)?;

        Ok(Self {
            model,
            tokenizer,
            model_type,
        })
    }

    /// Load a model from a GGUF file
    ///
    /// GGUF files contain model weights and metadata (config).
    /// A separate tokenizer.json is still required.
    ///
    /// # Arguments
    ///
    /// * `gguf_path` - Path to the .gguf file
    /// * `tokenizer_path` - Path to tokenizer.json
    /// * `device` - Device to load the model onto
    pub fn load_gguf<P: AsRef<Path>>(
        gguf_path: P,
        tokenizer_path: P,
        device: &B::Device,
    ) -> Result<Self, LlmError> {
        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref()).map_err(|e| {
            LlmError::TokenizerError(format!("{}: {}", tokenizer_path.as_ref().display(), e))
        })?;

        // For now, only Gemma4 GGUF loading is supported
        let (model, _runtime, _config) =
            load_gemma4_gguf(gguf_path, device).map_err(|e| LlmError::LoadError(e.to_string()))?;

        Ok(Self {
            model: ModelInstance::Gemma4(model, _runtime),
            tokenizer,
            model_type: ModelType::Gemma4,
        })
    }

    /// Load the model instance based on type
    fn load_model_instance(
        model_type: ModelType,
        path: &Path,
        device: &B::Device,
    ) -> Result<ModelInstance<B>, LlmError> {
        // Find safetensors file(s)
        let safetensors_path = Self::find_safetensors(path)?;

        // Load config
        let config_path = path.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| LlmError::ConfigError(format!("{}: {}", config_path.display(), e)))?;

        match model_type {
            ModelType::Llama => {
                let config = parse_llama_config(&config_str)?;
                let (model, runtime) = load_llama(&safetensors_path, &config, device)
                    .map_err(|e| LlmError::LoadError(e.to_string()))?;
                Ok(ModelInstance::Llama(model, runtime))
            }
            ModelType::Mistral => {
                let config = parse_mistral_config(&config_str)?;
                let (model, runtime) = load_mistral(&safetensors_path, &config, device)
                    .map_err(|e| LlmError::LoadError(e.to_string()))?;
                Ok(ModelInstance::Mistral(model, runtime))
            }
            ModelType::Mixtral => {
                let config = parse_mixtral_config(&config_str)?;
                let (model, runtime) = load_mixtral(&safetensors_path, &config, device)
                    .map_err(|e| LlmError::LoadError(e.to_string()))?;
                Ok(ModelInstance::Mixtral(model, runtime))
            }
            ModelType::Gemma => {
                let config = parse_gemma_config(&config_str)?;
                let (model, runtime) = load_gemma(&safetensors_path, &config, device)
                    .map_err(|e| LlmError::LoadError(e.to_string()))?;
                Ok(ModelInstance::Gemma(model, runtime))
            }
            ModelType::Gemma4 => {
                let config = parse_gemma4_config(&config_str)?;
                let (model, runtime) = load_gemma4(&safetensors_path, &config, device)
                    .map_err(|e| LlmError::LoadError(e.to_string()))?;
                Ok(ModelInstance::Gemma4(model, runtime))
            }
            ModelType::Phi => {
                let config = parse_phi_config(&config_str)?;
                let (model, runtime) = load_phi(&safetensors_path, &config, device)
                    .map_err(|e| LlmError::LoadError(e.to_string()))?;
                Ok(ModelInstance::Phi(model, runtime))
            }
            ModelType::Qwen => {
                let config = parse_qwen_config(&config_str)?;
                let (model, runtime) = load_qwen(&safetensors_path, &config, device)
                    .map_err(|e| LlmError::LoadError(e.to_string()))?;
                Ok(ModelInstance::Qwen(model, runtime))
            }
            ModelType::DeepSeek => {
                let config = parse_deepseek_config(&config_str)?;
                let (model, runtime) = load_deepseek(&safetensors_path, &config, device)
                    .map_err(|e| LlmError::LoadError(e.to_string()))?;
                Ok(ModelInstance::DeepSeek(model, runtime))
            }
            ModelType::Rwkv => {
                let config = parse_rwkv_config(&config_str)?;
                let (model, runtime) = load_rwkv(&safetensors_path, &config, device)
                    .map_err(|e| LlmError::LoadError(e.to_string()))?;
                Ok(ModelInstance::Rwkv(model, runtime))
            }
            ModelType::Mamba => {
                let config = parse_mamba_config(&config_str)?;
                let (model, runtime) = load_mamba(&safetensors_path, &config, device)
                    .map_err(|e| LlmError::LoadError(e.to_string()))?;
                Ok(ModelInstance::Mamba(model, runtime))
            }
            ModelType::Jamba => {
                let config = parse_jamba_config(&config_str)?;
                let (model, runtime) = load_jamba(&safetensors_path, &config, device)
                    .map_err(|e| LlmError::LoadError(e.to_string()))?;
                Ok(ModelInstance::Jamba(model, runtime))
            }
        }
    }

    /// Find the safetensors file(s) in a directory
    fn find_safetensors(path: &Path) -> Result<std::path::PathBuf, LlmError> {
        // Try single file first
        let single = path.join("model.safetensors");
        if single.exists() {
            return Ok(single);
        }

        // Try sharded format (model-00001-of-00002.safetensors)
        for entry in std::fs::read_dir(path).map_err(LlmError::Io)? {
            let entry = entry.map_err(LlmError::Io)?;
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with("model-") && name_str.ends_with(".safetensors") {
                // Return the directory - loader will handle sharding
                return Ok(entry.path());
            }
        }

        Err(LlmError::LoadError(format!(
            "No safetensors file found in {}",
            path.display()
        )))
    }

    /// Get the model type
    pub fn model_type(&self) -> ModelType {
        self.model_type
    }

    /// Get a reference to the tokenizer
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, LlmError> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| LlmError::TokenizationError(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32]) -> Result<String, LlmError> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| LlmError::TokenizationError(e.to_string()))
    }

    /// Generate text from a prompt
    ///
    /// # Arguments
    ///
    /// * `prompt` - The input prompt text
    /// * `config` - Generation configuration
    ///
    /// # Returns
    ///
    /// The generated text (not including the prompt)
    pub fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String, LlmError> {
        // Tokenize prompt
        let input_ids = self.encode(prompt)?;
        let device = self.device();
        let seq_len = input_ids.len();

        // Convert to tensor using TensorData
        let ids_i32: Vec<i32> = input_ids.iter().map(|&id| id as i32).collect();
        let data = burn::tensor::TensorData::new(ids_i32, [1, seq_len]);
        let input_tensor = Tensor::<B, 2, Int>::from_data(data, &device);

        // Generate tokens
        let output_tensor = self.generate_tokens(input_tensor, config);

        // Extract generated portion (skip prompt tokens)
        let output_ids: Vec<i64> = output_tensor.to_data().to_vec().unwrap();
        let generated_ids: Vec<u32> = output_ids
            .into_iter()
            .skip(seq_len)
            .map(|id| id as u32)
            .collect();

        // Decode to text
        let mut output = self.decode(&generated_ids)?;

        // Apply stop sequences
        for stop in &config.stop_sequences {
            if let Some(pos) = output.find(stop) {
                output.truncate(pos);
                break;
            }
        }

        Ok(output)
    }

    /// Get the device the model is on
    fn device(&self) -> B::Device {
        match &self.model {
            ModelInstance::Llama(m, _) => m.embed_tokens.weight.val().device(),
            ModelInstance::Mistral(m, _) => m.embed_tokens.weight.val().device(),
            ModelInstance::Mixtral(m, _) => m.embed_tokens.weight.val().device(),
            ModelInstance::Gemma(m, _) => m.embed_tokens.weight.val().device(),
            ModelInstance::Gemma4(m, _) => m.embed_tokens.weight.val().device(),
            ModelInstance::Phi(m, _) => m.embed_tokens.weight.val().device(),
            ModelInstance::Qwen(m, _) => m.embed_tokens.weight.val().device(),
            ModelInstance::DeepSeek(m, _) => m.embed_tokens.weight.val().device(),
            ModelInstance::Rwkv(m, _) => m.embed_tokens.weight.val().device(),
            ModelInstance::Mamba(m, _) => m.embed_tokens.weight.val().device(),
            ModelInstance::Jamba(m, _) => m.embed_tokens.weight.val().device(),
        }
    }

    /// Generate tokens from input tensor
    fn generate_tokens(
        &self,
        input_ids: Tensor<B, 2, Int>,
        config: &GenerationConfig,
    ) -> Tensor<B, 2, Int> {
        match &self.model {
            ModelInstance::Llama(model, runtime) => {
                model.generate(input_ids, runtime, config.max_tokens, &config.sampler)
            }
            ModelInstance::Gemma4(model, runtime) => {
                model.generate(input_ids, runtime, config.max_tokens, &config.sampler)
            }
            ModelInstance::Mistral(model, runtime) => {
                model.generate(input_ids, runtime, config.max_tokens, &config.sampler)
            }
            ModelInstance::Mixtral(model, runtime) => {
                model.generate(input_ids, runtime, config.max_tokens, &config.sampler)
            }
            ModelInstance::Gemma(model, runtime) => {
                model.generate(input_ids, runtime, config.max_tokens, &config.sampler)
            }
            ModelInstance::Phi(model, runtime) => {
                model.generate(input_ids, runtime, config.max_tokens, &config.sampler)
            }
            ModelInstance::Qwen(model, runtime) => {
                model.generate(input_ids, runtime, config.max_tokens, &config.sampler)
            }
            ModelInstance::DeepSeek(model, runtime) => {
                model.generate(input_ids, runtime, config.max_tokens, &config.sampler)
            }
            ModelInstance::Rwkv(model, runtime) => {
                model.generate(input_ids, runtime, config.max_tokens, &config.sampler)
            }
            ModelInstance::Mamba(model, runtime) => {
                model.generate(input_ids, runtime, config.max_tokens, &config.sampler)
            }
            ModelInstance::Jamba(model, runtime) => {
                model.generate(input_ids, runtime, config.max_tokens, &config.sampler)
            }
        }
    }
}

// Config parsing helpers - extract from HuggingFace config.json format

fn parse_llama_config(json: &str) -> Result<LlamaConfig, LlmError> {
    let v: serde_json::Value =
        serde_json::from_str(json).map_err(|e| LlmError::ConfigError(e.to_string()))?;

    Ok(LlamaConfig {
        vocab_size: v["vocab_size"].as_u64().unwrap_or(32000) as usize,
        hidden_size: v["hidden_size"].as_u64().unwrap_or(4096) as usize,
        intermediate_size: v["intermediate_size"].as_u64().unwrap_or(11008) as usize,
        num_layers: v["num_hidden_layers"].as_u64().unwrap_or(32) as usize,
        num_heads: v["num_attention_heads"].as_u64().unwrap_or(32) as usize,
        num_kv_heads: v["num_key_value_heads"].as_u64().unwrap_or(32) as usize,
        max_seq_len: v["max_position_embeddings"].as_u64().unwrap_or(4096) as usize,
        norm_eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-5),
        rope_base: v["rope_theta"].as_f64().unwrap_or(10000.0) as f32,
    })
}

fn parse_mistral_config(json: &str) -> Result<MistralConfig, LlmError> {
    let v: serde_json::Value =
        serde_json::from_str(json).map_err(|e| LlmError::ConfigError(e.to_string()))?;

    Ok(MistralConfig {
        vocab_size: v["vocab_size"].as_u64().unwrap_or(32000) as usize,
        hidden_size: v["hidden_size"].as_u64().unwrap_or(4096) as usize,
        intermediate_size: v["intermediate_size"].as_u64().unwrap_or(14336) as usize,
        num_layers: v["num_hidden_layers"].as_u64().unwrap_or(32) as usize,
        num_heads: v["num_attention_heads"].as_u64().unwrap_or(32) as usize,
        num_kv_heads: v["num_key_value_heads"].as_u64().unwrap_or(8) as usize,
        max_seq_len: v["max_position_embeddings"].as_u64().unwrap_or(32768) as usize,
        norm_eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-5),
        rope_base: v["rope_theta"].as_f64().unwrap_or(10000.0) as f32,
        sliding_window: v["sliding_window"].as_u64().map(|x| x as usize),
    })
}

fn parse_mixtral_config(json: &str) -> Result<MixtralConfig, LlmError> {
    let v: serde_json::Value =
        serde_json::from_str(json).map_err(|e| LlmError::ConfigError(e.to_string()))?;

    Ok(MixtralConfig {
        vocab_size: v["vocab_size"].as_u64().unwrap_or(32000) as usize,
        hidden_size: v["hidden_size"].as_u64().unwrap_or(4096) as usize,
        intermediate_size: v["intermediate_size"].as_u64().unwrap_or(14336) as usize,
        num_layers: v["num_hidden_layers"].as_u64().unwrap_or(32) as usize,
        num_heads: v["num_attention_heads"].as_u64().unwrap_or(32) as usize,
        num_kv_heads: v["num_key_value_heads"].as_u64().unwrap_or(8) as usize,
        max_seq_len: v["max_position_embeddings"].as_u64().unwrap_or(32768) as usize,
        norm_eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-5),
        rope_base: v["rope_theta"].as_f64().unwrap_or(1000000.0) as f32,
        num_experts: v["num_local_experts"].as_u64().unwrap_or(8) as usize,
        num_experts_per_tok: v["num_experts_per_tok"].as_u64().unwrap_or(2) as usize,
    })
}

fn parse_gemma_config(json: &str) -> Result<GemmaConfig, LlmError> {
    let v: serde_json::Value =
        serde_json::from_str(json).map_err(|e| LlmError::ConfigError(e.to_string()))?;

    Ok(GemmaConfig {
        vocab_size: v["vocab_size"].as_u64().unwrap_or(256000) as usize,
        hidden_size: v["hidden_size"].as_u64().unwrap_or(2048) as usize,
        intermediate_size: v["intermediate_size"].as_u64().unwrap_or(16384) as usize,
        num_layers: v["num_hidden_layers"].as_u64().unwrap_or(18) as usize,
        num_heads: v["num_attention_heads"].as_u64().unwrap_or(8) as usize,
        num_kv_heads: v["num_key_value_heads"].as_u64().unwrap_or(1) as usize,
        max_seq_len: v["max_position_embeddings"].as_u64().unwrap_or(8192) as usize,
        sliding_window: v["sliding_window"].as_u64().unwrap_or(4096) as usize,
        attn_logit_softcap: v["attn_logit_softcapping"].as_f64().unwrap_or(50.0) as f32,
        final_logit_softcap: v["final_logit_softcapping"].as_f64().unwrap_or(30.0) as f32,
        norm_eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-6),
        rope_base: v["rope_theta"].as_f64().unwrap_or(10000.0) as f32,
    })
}

fn parse_gemma4_config(json: &str) -> Result<Gemma4Config, LlmError> {
    let v: serde_json::Value =
        serde_json::from_str(json).map_err(|e| LlmError::ConfigError(e.to_string()))?;

    let num_layers = v["num_hidden_layers"].as_u64().unwrap_or(30) as usize;
    let num_heads = v["num_attention_heads"].as_u64().unwrap_or(32) as usize;
    let hidden_size = v["hidden_size"].as_u64().unwrap_or(3840) as usize;
    let head_dim = v["head_dim"]
        .as_u64()
        .unwrap_or((hidden_size / num_heads) as u64) as usize;

    let num_experts = v["num_local_experts"].as_u64().unwrap_or(0) as usize;

    // Parse moe_layers: explicit list > derive from num_experts > empty (dense)
    let moe_layers = if let Some(arr) = v["moe_layers"].as_array() {
        arr.iter()
            .filter_map(|x| x.as_u64().map(|n| n as usize))
            .collect()
    } else if num_experts > 0 {
        // MoE model without explicit layer list — default to odd layers
        (0..num_layers).filter(|i| i % 2 == 1).collect()
    } else {
        // Dense model — no MoE layers
        Vec::new()
    };

    let intermediate_size = v["intermediate_size"].as_u64().unwrap_or(24576) as usize;

    Ok(Gemma4Config {
        vocab_size: v["vocab_size"].as_u64().unwrap_or(262144) as usize,
        hidden_size,
        intermediate_size,
        expert_intermediate_size: v["expert_intermediate_size"]
            .as_u64()
            .unwrap_or(intermediate_size as u64) as usize,
        num_layers,
        num_heads,
        num_kv_heads: v["num_key_value_heads"].as_u64().unwrap_or(8) as usize,
        head_dim,
        max_seq_len: v["max_position_embeddings"].as_u64().unwrap_or(8192) as usize,
        sliding_window: v["sliding_window"].as_u64().unwrap_or(1024) as usize,
        attn_logit_softcap: v["attn_logit_softcapping"].as_f64().unwrap_or(50.0) as f32,
        final_logit_softcap: v["final_logit_softcapping"].as_f64().unwrap_or(30.0) as f32,
        norm_eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-6),
        rope_base: v["rope_theta"].as_f64().unwrap_or(1000000.0) as f32,
        rope_base_swa: v["rope_theta_swa"].as_f64().unwrap_or(10000.0) as f32,
        head_dim_swa: v["head_dim_swa"].as_u64().unwrap_or(head_dim as u64) as usize,
        num_experts: num_experts.max(1), // at least 1 for dense FFN construction
        num_shared_experts: v["num_shared_experts"]
            .as_u64()
            .unwrap_or(if num_experts > 0 { 1 } else { 0 }) as usize,
        num_experts_per_tok: v["num_experts_per_tok"]
            .as_u64()
            .unwrap_or(if num_experts > 0 { 8 } else { 1 }) as usize,
        moe_layers,
    })
}

fn parse_phi_config(json: &str) -> Result<PhiConfig, LlmError> {
    let v: serde_json::Value =
        serde_json::from_str(json).map_err(|e| LlmError::ConfigError(e.to_string()))?;

    Ok(PhiConfig {
        vocab_size: v["vocab_size"].as_u64().unwrap_or(51200) as usize,
        hidden_size: v["hidden_size"].as_u64().unwrap_or(2560) as usize,
        intermediate_size: v["intermediate_size"].as_u64().unwrap_or(10240) as usize,
        num_layers: v["num_hidden_layers"].as_u64().unwrap_or(32) as usize,
        num_heads: v["num_attention_heads"].as_u64().unwrap_or(32) as usize,
        num_kv_heads: v["num_key_value_heads"].as_u64().unwrap_or(32) as usize,
        max_seq_len: v["max_position_embeddings"].as_u64().unwrap_or(2048) as usize,
        norm_eps: v["layer_norm_eps"].as_f64().unwrap_or(1e-5),
        rope_base: v["rope_theta"].as_f64().unwrap_or(10000.0) as f32,
        rope_scaling: v["rope_scaling"]["factor"].as_f64().map(|f| f as f32),
    })
}

fn parse_qwen_config(json: &str) -> Result<QwenConfig, LlmError> {
    let v: serde_json::Value =
        serde_json::from_str(json).map_err(|e| LlmError::ConfigError(e.to_string()))?;

    Ok(QwenConfig {
        vocab_size: v["vocab_size"].as_u64().unwrap_or(151936) as usize,
        hidden_size: v["hidden_size"].as_u64().unwrap_or(4096) as usize,
        intermediate_size: v["intermediate_size"].as_u64().unwrap_or(22016) as usize,
        num_layers: v["num_hidden_layers"].as_u64().unwrap_or(32) as usize,
        num_heads: v["num_attention_heads"].as_u64().unwrap_or(32) as usize,
        num_kv_heads: v["num_key_value_heads"].as_u64().unwrap_or(32) as usize,
        max_seq_len: v["max_position_embeddings"].as_u64().unwrap_or(32768) as usize,
        norm_eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-6),
        rope_base: v["rope_theta"].as_f64().unwrap_or(1000000.0) as f32,
        tie_embeddings: v["tie_word_embeddings"].as_bool().unwrap_or(false),
    })
}

fn parse_deepseek_config(json: &str) -> Result<DeepSeekConfig, LlmError> {
    let v: serde_json::Value =
        serde_json::from_str(json).map_err(|e| LlmError::ConfigError(e.to_string()))?;

    let hidden_size = v["hidden_size"].as_u64().unwrap_or(4096) as usize;
    let num_heads = v["num_attention_heads"].as_u64().unwrap_or(32) as usize;
    let head_dim = hidden_size / num_heads;

    Ok(DeepSeekConfig {
        vocab_size: v["vocab_size"].as_u64().unwrap_or(102400) as usize,
        hidden_size,
        intermediate_size: v["intermediate_size"].as_u64().unwrap_or(11008) as usize,
        num_layers: v["num_hidden_layers"].as_u64().unwrap_or(30) as usize,
        num_heads,
        num_kv_heads: v["num_key_value_heads"].as_u64().unwrap_or(32) as usize,
        max_seq_len: v["max_position_embeddings"].as_u64().unwrap_or(4096) as usize,
        norm_eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-6),
        rope_base: v["rope_theta"].as_f64().unwrap_or(10000.0) as f32,
        use_mla: v["use_mla"].as_bool().unwrap_or(false),
        q_lora_rank: v["q_lora_rank"].as_u64().unwrap_or(0) as usize,
        kv_lora_rank: v["kv_lora_rank"].as_u64().unwrap_or(0) as usize,
        qk_nope_head_dim: v["qk_nope_head_dim"].as_u64().unwrap_or(head_dim as u64) as usize,
        qk_rope_head_dim: v["qk_rope_head_dim"].as_u64().unwrap_or(head_dim as u64) as usize,
    })
}

fn parse_rwkv_config(json: &str) -> Result<RwkvConfig, LlmError> {
    let v: serde_json::Value =
        serde_json::from_str(json).map_err(|e| LlmError::ConfigError(e.to_string()))?;

    let hidden_size = v["hidden_size"].as_u64().unwrap_or(768) as usize;
    let head_dim = v["head_size"].as_u64().unwrap_or(64) as usize;
    let num_heads = hidden_size / head_dim;

    Ok(RwkvConfig {
        vocab_size: v["vocab_size"].as_u64().unwrap_or(65536) as usize,
        hidden_size,
        num_layers: v["num_hidden_layers"].as_u64().unwrap_or(12) as usize,
        num_heads,
        head_dim,
        ffn_multiplier: v["ffn_multiplier"].as_u64().unwrap_or(4) as usize,
        layer_norm_eps: v["layer_norm_eps"].as_f64().unwrap_or(1e-5),
    })
}

fn parse_mamba_config(json: &str) -> Result<MambaConfig, LlmError> {
    let v: serde_json::Value =
        serde_json::from_str(json).map_err(|e| LlmError::ConfigError(e.to_string()))?;

    let d_model = v["d_model"].as_u64().unwrap_or(768) as usize;

    Ok(MambaConfig {
        vocab_size: v["vocab_size"].as_u64().unwrap_or(50280) as usize,
        d_model,
        n_layer: v["n_layer"].as_u64().unwrap_or(24) as usize,
        d_state: v["d_state"].as_u64().unwrap_or(16) as usize,
        d_conv: v["d_conv"].as_u64().unwrap_or(4) as usize,
        expand: v["expand"].as_u64().unwrap_or(2) as usize,
        dt_rank: v["dt_rank"]
            .as_u64()
            .map(|x| x as usize)
            .unwrap_or_else(|| d_model.div_ceil(16)),
        layer_norm_eps: v["layer_norm_eps"].as_f64().unwrap_or(1e-5),
    })
}

fn parse_jamba_config(json: &str) -> Result<JambaConfig, LlmError> {
    let v: serde_json::Value =
        serde_json::from_str(json).map_err(|e| LlmError::ConfigError(e.to_string()))?;

    Ok(JambaConfig {
        vocab_size: v["vocab_size"].as_u64().unwrap_or(65536) as usize,
        d_model: v["hidden_size"].as_u64().unwrap_or(4096) as usize,
        n_layer: v["num_hidden_layers"].as_u64().unwrap_or(32) as usize,
        n_heads: v["num_attention_heads"].as_u64().unwrap_or(32) as usize,
        n_kv_heads: v["num_key_value_heads"].as_u64().unwrap_or(8) as usize,
        head_dim: v["head_dim"].as_u64().unwrap_or(128) as usize,
        intermediate_size: v["intermediate_size"].as_u64().unwrap_or(14336) as usize,
        d_state: v["mamba_d_state"].as_u64().unwrap_or(16) as usize,
        d_conv: v["mamba_d_conv"].as_u64().unwrap_or(4) as usize,
        expand: v["mamba_expand"].as_u64().unwrap_or(2) as usize,
        dt_rank: v["mamba_dt_rank"].as_u64().unwrap_or(256) as usize,
        n_experts: v["num_experts"].as_u64().unwrap_or(16) as usize,
        n_experts_per_tok: v["num_experts_per_tok"].as_u64().unwrap_or(2) as usize,
        attn_layer_period: v["attn_layer_period"].as_u64().unwrap_or(8) as usize,
        moe_layer_period: v["expert_layer_period"].as_u64().unwrap_or(2) as usize,
        layer_norm_eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-6),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_config_defaults() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_tokens, 256);
        assert!((config.sampler.temperature - 0.7).abs() < 0.001);
        assert!((config.sampler.top_p - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_generation_config_builder() {
        let config = GenerationConfig::new(100)
            .with_temperature(0.5)
            .with_top_p(0.8)
            .with_stop_sequence("\n");

        assert_eq!(config.max_tokens, 100);
        assert!((config.sampler.temperature - 0.5).abs() < 0.001);
        assert!((config.sampler.top_p - 0.8).abs() < 0.001);
        assert_eq!(config.stop_sequences, vec!["\n"]);
    }

    #[test]
    fn test_model_type_parsing() {
        use std::str::FromStr;
        assert_eq!(ModelType::from_str("llama"), Ok(ModelType::Llama));
        assert_eq!(ModelType::from_str("MISTRAL"), Ok(ModelType::Mistral));
        assert!(ModelType::from_str("unknown").is_err());
    }

    #[test]
    fn test_model_type_as_str() {
        assert_eq!(ModelType::Llama.as_str(), "llama");
        assert_eq!(ModelType::Mixtral.as_str(), "mixtral");
    }

    #[test]
    fn test_parse_llama_config() {
        let json = r#"{"vocab_size": 32000, "hidden_size": 4096, "intermediate_size": 11008, "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8}"#;
        let config = parse_llama_config(json).unwrap();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_kv_heads, 8);
    }
}
