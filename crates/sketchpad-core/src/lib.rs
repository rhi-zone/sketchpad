//! Core Building Blocks for Deep Learning Models
//!
//! This crate provides shared components used across image generation,
//! video generation, and language models in the burn-models ecosystem.
//!
//! # Modules
//!
//! ## Attention Mechanisms
//!
//! - [`attention`] - Multi-head attention with optional GQA (grouped-query attention)
//! - [`paged_attention`] - Block-based KV cache for efficient LLM serving
//! - [`temporal_attention`] - Factorized spatial-temporal attention for video
//!
//! ## Normalization Layers
//!
//! - [`layernorm`] - Layer normalization
//! - [`rmsnorm`] - RMS normalization (used in LLaMA, Mistral)
//! - [`groupnorm`] - Group normalization (used in UNet, VAE)
//!
//! ## Position Encodings
//!
//! - [`rope`] - Rotary position embeddings (RoPE)
//!
//! ## Feed-Forward Networks
//!
//! - [`glu`] - Gated Linear Units (SwiGLU, GeGLU)
//! - [`silu`] - SiLU/Swish activation
//! - [`moe`] - Mixture of Experts routing
//!
//! ## Inference Optimization
//!
//! - [`kv_cache`] - Key-value cache for autoregressive generation
//! - [`continuous_batching`] - Iteration-level scheduling with preemption
//! - [`speculative`] - Speculative decoding (draft/target verification)
//! - [`quantization`] - INT4/INT8/FP8 quantization
//! - [`precision`] - Mixed precision utilities
//!
//! ## Model Components
//!
//! - [`dit`] - DiT building blocks (PatchEmbed, AdaLayerNorm, unpatchify)
//! - [`transformer`] - Generic transformer blocks
//! - [`lora`] - LoRA adapter injection
//! - [`textual_inversion`] - Custom embedding injection
//!
//! ## Video-Specific
//!
//! - [`vae3d`] - 3D VAE for temporal compression
//! - [`frame_interpolation`] - Frame interpolation utilities
//!
//! # Example
//!
//! ```ignore
//! use sketchpad_core::rope::RotaryEmbedding;
//! use sketchpad_core::rmsnorm::RmsNorm;
//!
//! // Create rotary embeddings
//! let rope = RotaryEmbedding::new(head_dim, max_seq_len, &device);
//!
//! // Apply to queries and keys
//! let (q_rot, k_rot) = rope.apply(q, k, position);
//! ```

pub mod attention;
pub mod continuous_batching;
pub mod dit;
pub mod frame_interpolation;
pub mod glu;
pub mod groupnorm;
pub mod kv_cache;
pub mod layernorm;
pub mod lora;
pub mod moe;
pub mod paged_attention;
pub mod precision;
pub mod quantization;
pub mod rmsnorm;
pub mod rope;
pub mod silu;
pub mod speculative;
pub mod temporal_attention;
pub mod textual_inversion;
pub mod transformer;
pub mod vae3d;
