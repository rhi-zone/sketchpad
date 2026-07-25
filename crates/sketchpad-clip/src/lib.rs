//! CLIP and OpenCLIP Text Encoders
//!
//! This crate provides text encoder implementations for conditioning
//! diffusion models on text prompts.
//!
//! # Encoders
//!
//! - [`ClipTextEncoder`] - OpenAI CLIP ViT-L/14 (SD 1.x, SD 2.x)
//! - [`OpenClipTextEncoder`] - OpenCLIP ViT-bigG (SDXL second encoder)
//!
//! # Tokenizer
//!
//! The [`ClipTokenizer`] handles BPE tokenization of text prompts:
//!
//! ```ignore
//! use sketchpad_clip::ClipTokenizer;
//!
//! let tokenizer = ClipTokenizer::from_file("vocab.txt")?;
//! let tokens = tokenizer.encode("a photo of a cat")?;
//! ```
//!
//! # Example
//!
//! ```ignore
//! use sketchpad_clip::{ClipConfig, ClipTextEncoder};
//!
//! // SD 1.x CLIP encoder
//! let config = ClipConfig::sd1x();
//! let encoder = config.init::<Backend>(&device);
//!
//! let embeddings = encoder.forward(token_ids);
//! ```

pub mod attention;
pub mod clip;
pub mod embedder;
pub mod open_clip;
pub mod tokenizer;

pub use attention::{create_causal_mask, scaled_dot_product_attention};
pub use clip::{
    ClipConfig, ClipTextEncoder, FeedForward, MultiHeadSelfAttention, TransformerBlock,
};
pub use open_clip::{OpenClipConfig, OpenClipTextEncoder};
pub use tokenizer::{ClipTokenizer, END_OF_TEXT, START_OF_TEXT, TokenizerError};
