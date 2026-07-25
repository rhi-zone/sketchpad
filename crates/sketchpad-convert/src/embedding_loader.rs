//! Textual Inversion embedding loader
//!
//! Loads embeddings from .pt (PyTorch) or .safetensors files.

use std::path::Path;

use burn::prelude::*;
use sketchpad_core::textual_inversion::{EmbeddingError, TextualInversionEmbedding};

use crate::loader::SafeTensorFile;

/// Embedding file format
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmbeddingFormat {
    /// SafeTensors format (.safetensors)
    SafeTensors,
    /// Auto-detect from extension
    Auto,
}

/// Load a textual inversion embedding from a file
pub fn load_embedding<B: Backend>(
    path: impl AsRef<Path>,
    format: EmbeddingFormat,
    device: &B::Device,
) -> Result<TextualInversionEmbedding<B>, EmbeddingLoadError> {
    let path = path.as_ref();

    let format = if format == EmbeddingFormat::Auto {
        detect_format(path)
    } else {
        format
    };

    match format {
        EmbeddingFormat::SafeTensors => load_safetensors_embedding(path, device),
        EmbeddingFormat::Auto => unreachable!(),
    }
}

/// Detects the embedding format from file extension
fn detect_format(path: &Path) -> EmbeddingFormat {
    match path.extension().and_then(|e| e.to_str()) {
        Some("safetensors") => EmbeddingFormat::SafeTensors,
        _ => EmbeddingFormat::SafeTensors, // Default to safetensors
    }
}

/// Loads an embedding from a safetensors file
fn load_safetensors_embedding<B: Backend>(
    path: &Path,
    device: &B::Device,
) -> Result<TextualInversionEmbedding<B>, EmbeddingLoadError> {
    let file =
        SafeTensorFile::open(path).map_err(|e| EmbeddingLoadError::FileError(e.to_string()))?;

    let names: Vec<String> = file.names().map(|s| s.to_string()).collect();

    // Find the embedding tensor
    // Common patterns: "emb_params", "string_to_param.*", "<token>"
    let emb_key = find_embedding_key(&names).ok_or_else(|| {
        EmbeddingLoadError::ParseError("Could not find embedding tensor in file".to_string())
    })?;

    // Determine token name
    let token = extract_token_name(&emb_key, path);

    // Load the embedding tensor
    let shape = file.shape(&emb_key).ok_or_else(|| {
        EmbeddingLoadError::ParseError(format!("Could not get shape for key: {}", emb_key))
    })?;

    let vectors = match shape.len() {
        1 => {
            // Single vector: reshape to [1, dim]
            let vec: Tensor<B, 1> = file
                .load_f32(&emb_key, device)
                .map_err(|e| EmbeddingLoadError::TensorError(e.to_string()))?;
            let dim = vec.dims()[0];
            vec.reshape([1, dim])
        }
        2 => {
            // Multiple vectors: [num_vectors, dim]
            file.load_f32(&emb_key, device)
                .map_err(|e| EmbeddingLoadError::TensorError(e.to_string()))?
        }
        _ => {
            return Err(EmbeddingLoadError::ParseError(format!(
                "Unexpected embedding shape: {:?}",
                shape
            )));
        }
    };

    Ok(TextualInversionEmbedding::new(token, vectors))
}

/// Finds the embedding tensor key from available tensor names
fn find_embedding_key(names: &[String]) -> Option<String> {
    // Priority order for finding embedding tensor
    let patterns = [
        "emb_params",
        "string_to_param",
        "clip_l", // SDXL
        "clip_g", // SDXL
    ];

    for pattern in patterns {
        if let Some(key) = names.iter().find(|k| k.contains(pattern)) {
            return Some(key.clone());
        }
    }

    // If only one tensor, use it
    if names.len() == 1 {
        return Some(names[0].clone());
    }

    // Look for common embedding-like keys
    names
        .iter()
        .find(|k| k.starts_with('<') || k.contains("embedding") || k.contains("token"))
        .cloned()
}

/// Extracts a token name from the embedding key or filename
fn extract_token_name(key: &str, path: &Path) -> String {
    // Try to extract token from key
    if key.starts_with('<') && key.ends_with('>') {
        return key.to_string();
    }

    if key.contains("string_to_param.") {
        let token = key.replace("string_to_param.", "");
        if !token.is_empty() {
            return format!("<{}>", token);
        }
    }

    // Fall back to filename
    let filename = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("embedding");

    format!("<{}>", filename)
}

/// Errors that can occur when loading embeddings
#[derive(Debug)]
pub enum EmbeddingLoadError {
    /// Error reading the embedding file
    FileError(String),
    /// Error parsing the embedding format
    ParseError(String),
    /// Error loading tensor data
    TensorError(String),
}

impl std::fmt::Display for EmbeddingLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileError(msg) => write!(f, "File error: {}", msg),
            Self::ParseError(msg) => write!(f, "Parse error: {}", msg),
            Self::TensorError(msg) => write!(f, "Tensor error: {}", msg),
        }
    }
}

impl std::error::Error for EmbeddingLoadError {}

impl From<EmbeddingLoadError> for EmbeddingError {
    fn from(e: EmbeddingLoadError) -> Self {
        EmbeddingError::LoadError(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_format() {
        assert_eq!(
            detect_format(Path::new("embedding.safetensors")),
            EmbeddingFormat::SafeTensors
        );
    }

    #[test]
    fn test_extract_token_name() {
        assert_eq!(
            extract_token_name("<my-concept>", Path::new("test.safetensors")),
            "<my-concept>"
        );

        assert_eq!(
            extract_token_name("string_to_param.sks", Path::new("test.safetensors")),
            "<sks>"
        );

        assert_eq!(
            extract_token_name("emb_params", Path::new("my-embedding.safetensors")),
            "<my-embedding>"
        );
    }
}
