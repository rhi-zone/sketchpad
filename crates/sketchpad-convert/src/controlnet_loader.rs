//! ControlNet weight loading from safetensors files
//!
//! Loads ControlNet weights from various formats.

use std::path::Path;

use crate::loader::SafeTensorFile;

/// ControlNet model type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ControlNetType {
    /// SD 1.x ControlNet
    Sd1x,
    /// SDXL ControlNet
    Sdxl,
    /// Auto-detect
    Auto,
}

/// Load ControlNet model info from a safetensors file
pub fn load_controlnet_info(path: impl AsRef<Path>) -> Result<ControlNetInfo, ControlNetLoadError> {
    let file = SafeTensorFile::open(path.as_ref())
        .map_err(|e| ControlNetLoadError::FileError(e.to_string()))?;

    let names: Vec<String> = file.names().map(|s| s.to_string()).collect();

    // Detect model type based on key patterns
    let model_type = detect_controlnet_type(&names);

    // Count control outputs
    let num_zero_convs = names.iter().filter(|k| k.contains("zero_conv")).count();

    Ok(ControlNetInfo {
        model_type,
        num_zero_convs,
        num_tensors: names.len(),
    })
}

/// Detects the ControlNet model type from tensor key names
fn detect_controlnet_type(names: &[String]) -> ControlNetType {
    // Check for SDXL-specific patterns
    let has_sdxl_patterns = names
        .iter()
        .any(|k| k.contains("add_embedding") || k.contains("transformer_blocks.1"));

    if has_sdxl_patterns {
        ControlNetType::Sdxl
    } else {
        ControlNetType::Sd1x
    }
}

/// Information about a ControlNet model
#[derive(Debug, Clone)]
pub struct ControlNetInfo {
    /// Model type (SD 1.x or SDXL)
    pub model_type: ControlNetType,
    /// Number of zero convolution outputs
    pub num_zero_convs: usize,
    /// Total number of tensors
    pub num_tensors: usize,
}

/// Errors that can occur when loading ControlNet
#[derive(Debug)]
pub enum ControlNetLoadError {
    /// Error reading the model file
    FileError(String),
    /// Error parsing the model format
    ParseError(String),
    /// Error loading tensor data
    TensorError(String),
}

impl std::fmt::Display for ControlNetLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileError(msg) => write!(f, "File error: {}", msg),
            Self::ParseError(msg) => write!(f, "Parse error: {}", msg),
            Self::TensorError(msg) => write!(f, "Tensor error: {}", msg),
        }
    }
}

impl std::error::Error for ControlNetLoadError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_controlnet_type_detection() {
        let sd1x_keys = vec![
            "input_hint_block.0.weight".to_string(),
            "zero_convs.0.0.weight".to_string(),
        ];
        assert_eq!(detect_controlnet_type(&sd1x_keys), ControlNetType::Sd1x);

        let sdxl_keys = vec![
            "add_embedding.linear_1.weight".to_string(),
            "transformer_blocks.1.attn1.to_k.weight".to_string(),
        ];
        assert_eq!(detect_controlnet_type(&sdxl_keys), ControlNetType::Sdxl);
    }
}
