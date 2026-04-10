//! LLM implementations for burn-models
//!
//! This crate provides implementations of popular Large Language Models
//! using the Burn deep learning framework.
//!
//! # Supported Models
//!
//! - **LLaMA**: LLaMA 2 (7B, 13B, 70B) and LLaMA 3 (8B, 70B)
//! - **Mistral**: Mistral 7B
//! - **Mixtral**: Mixtral MoE
//! - **Gemma**: Gemma 2
//! - **Gemma 4**: Gemma 4 MoE
//! - **Phi**: Phi-2/3
//! - **Qwen**: Qwen 1.5/2
//! - **DeepSeek**: DeepSeek
//! - **RWKV**: RWKV-7
//! - **Mamba**: Mamba SSM
//! - **Jamba**: Jamba hybrid
//!
//! # High-Level API
//!
//! For most use cases, use the unified [`LlmInstance`] API:
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
//! ```
//!
//! # Chat API
//!
//! For interactive chat sessions:
//!
//! ```ignore
//! use sketchpad_llm::{LlmInstance, ChatSession, GenerationConfig};
//!
//! let llm = LlmInstance::load(ModelType::Mistral, "./model/", &device)?;
//! let mut session = ChatSession::new(llm, Some("You are helpful."));
//!
//! let response = session.send("Hello!", &GenerationConfig::default())?;
//! ```
//!
//! # HTTP Server
//!
//! With the `serve` feature, run an OpenAI-compatible server:
//!
//! ```ignore
//! use sketchpad_llm::{LlmInstance, serve::run_server};
//!
//! let llm = LlmInstance::load(ModelType::Llama, "./model/", &device)?;
//! run_server(llm, "127.0.0.1", 8080).await?;
//! ```

// Model implementations
pub mod deepseek;
pub mod deepseek_loader;
pub mod gemma;
pub mod gemma4;
pub mod gemma4_loader;
pub mod gemma_loader;
pub mod jamba;
pub mod jamba_loader;
pub mod llama;
pub mod llama_loader;
pub mod mamba;
pub mod mamba_loader;
pub mod mistral;
pub mod mistral_loader;
pub mod mixtral;
pub mod mixtral_loader;
pub mod phi;
pub mod phi_loader;
pub mod qwen;
pub mod qwen_loader;
pub mod rwkv;
pub mod rwkv_loader;

// High-level APIs
pub mod chat;
pub mod inference;

#[cfg(feature = "serve")]
pub mod serve;

// Re-export model types
pub use deepseek::{DeepSeek, DeepSeekConfig, DeepSeekOutput, DeepSeekRuntime};
pub use deepseek_loader::{DeepSeekLoadError, load_deepseek};
pub use gemma::{Gemma, GemmaConfig, GemmaOutput, GemmaRuntime};
pub use gemma_loader::{GemmaLoadError, load_gemma};
pub use gemma4::{Gemma4, Gemma4Config, Gemma4Output, Gemma4Runtime};
pub use gemma4_loader::{Gemma4LoadError, load_gemma4};
pub use jamba::{Jamba, JambaConfig, JambaOutput, JambaRuntime, JambaState};
pub use jamba_loader::{JambaLoadError, load_jamba};
pub use llama::{Llama, LlamaConfig, LlamaOutput, LlamaRuntime};
pub use llama_loader::{LlamaLoadError, load_llama};
pub use mamba::{Mamba, MambaConfig, MambaOutput, MambaRuntime, MambaState};
pub use mamba_loader::{MambaLoadError, load_mamba};
pub use mistral::{Mistral, MistralConfig, MistralOutput, MistralRuntime};
pub use mistral_loader::{MistralLoadError, load_mistral};
pub use mixtral::{Mixtral, MixtralConfig, MixtralOutput, MixtralRuntime};
pub use mixtral_loader::{MixtralLoadError, load_mixtral};
pub use phi::{Phi, PhiConfig, PhiOutput, PhiRuntime};
pub use phi_loader::{PhiLoadError, load_phi};
pub use qwen::{Qwen, QwenConfig, QwenOutput, QwenRuntime};
pub use qwen_loader::{QwenLoadError, load_qwen};
pub use rwkv::{Rwkv, RwkvConfig, RwkvOutput, RwkvRuntime, RwkvState};
pub use rwkv_loader::{RwkvLoadError, load_rwkv};

// Re-export high-level APIs
pub use chat::{ChatMessage, ChatSession, ChatTemplate, Role};
pub use inference::{GenerationConfig, LlmError, LlmInstance, ModelType};
