//! LoRA weight loading from safetensors files
//!
//! Supports loading LoRA weights in various formats (Kohya, diffusers, etc.)

use burn::prelude::*;
use std::collections::HashMap;
use std::path::Path;

use crate::loader::SafeTensorFile;
use sketchpad_core::lora::{LoraConvWeight, LoraModel, LoraWeight};

/// LoRA file format
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoraFormat {
    /// Kohya-style LoRA (lora_unet_*, lora_te_*)
    Kohya,
    /// Diffusers-style LoRA
    Diffusers,
    /// Auto-detect format
    Auto,
}

/// Load a LoRA model from a safetensors file
pub fn load_lora<B: Backend>(
    path: impl AsRef<Path>,
    scale: f32,
    format: LoraFormat,
    device: &B::Device,
) -> Result<LoraModel<B>, LoraLoadError> {
    let file =
        SafeTensorFile::open(path.as_ref()).map_err(|e| LoraLoadError::FileError(e.to_string()))?;

    let names: Vec<String> = file.names().map(|s| s.to_string()).collect();

    let format = if format == LoraFormat::Auto {
        detect_format(&names)
    } else {
        format
    };

    match format {
        LoraFormat::Kohya => load_kohya_lora(&file, &names, scale, device),
        LoraFormat::Diffusers => load_diffusers_lora(&file, &names, scale, device),
        LoraFormat::Auto => unreachable!(),
    }
}

/// Detects the LoRA format from tensor key names
fn detect_format(names: &[String]) -> LoraFormat {
    // Check for Kohya-style keys
    let has_kohya = names
        .iter()
        .any(|k| k.starts_with("lora_unet_") || k.starts_with("lora_te_"));

    if has_kohya {
        LoraFormat::Kohya
    } else {
        LoraFormat::Diffusers
    }
}

/// Loads a Kohya-style LoRA from safetensors
fn load_kohya_lora<B: Backend>(
    file: &SafeTensorFile,
    names: &[String],
    scale: f32,
    device: &B::Device,
) -> Result<LoraModel<B>, LoraLoadError> {
    let mut model = LoraModel::new(scale);

    // Group keys by base name (down_key, up_key, alpha)
    #[allow(clippy::type_complexity)]
    let mut groups: HashMap<String, (Option<String>, Option<String>, Option<f32>)> = HashMap::new();

    for key in names {
        // Parse Kohya-style keys: lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight
        if let Some(base) = key.strip_suffix(".lora_down.weight") {
            groups
                .entry(base.to_string())
                .or_insert((None, None, None))
                .0 = Some(key.clone());
        } else if let Some(base) = key.strip_suffix(".lora_up.weight") {
            groups
                .entry(base.to_string())
                .or_insert((None, None, None))
                .1 = Some(key.clone());
        } else if let Some(base) = key.strip_suffix(".alpha") {
            // Try to read alpha
            if let Ok(data) = file.load_f32::<B, 1>(key, device) {
                let alpha_data = data.into_data();
                let alpha: f32 = alpha_data.to_vec().unwrap()[0];
                groups
                    .entry(base.to_string())
                    .or_insert((None, None, None))
                    .2 = Some(alpha);
            }
        }
    }

    for (base_name, (down_key, up_key, alpha)) in groups {
        let (down_key, up_key) = match (down_key, up_key) {
            (Some(d), Some(u)) => (d, u),
            _ => continue, // Skip incomplete pairs
        };

        let alpha = alpha.unwrap_or(1.0);

        // Convert Kohya name to our internal name
        let layer_name = kohya_to_internal_name(&base_name);

        // Check shape to determine if linear (2D) or conv (4D)
        let shape = file.shape(&down_key);
        let ndims = shape.map(|s| s.len()).unwrap_or(0);

        if ndims == 2 {
            if let Ok(down) = file.load_f32::<B, 2>(&down_key, device) {
                if let Ok(up) = file.load_f32::<B, 2>(&up_key, device) {
                    let lora = LoraWeight::new(down, up, alpha);
                    model.add_linear(layer_name, lora);
                }
            }
        } else if ndims == 4 {
            if let Ok(down) = file.load_f32::<B, 4>(&down_key, device) {
                if let Ok(up) = file.load_f32::<B, 4>(&up_key, device) {
                    let lora = LoraConvWeight::new(down, up, alpha);
                    model.add_conv(layer_name, lora);
                }
            }
        }
    }

    Ok(model)
}

/// Loads a Diffusers-style LoRA from safetensors
fn load_diffusers_lora<B: Backend>(
    file: &SafeTensorFile,
    names: &[String],
    scale: f32,
    device: &B::Device,
) -> Result<LoraModel<B>, LoraLoadError> {
    let mut model = LoraModel::new(scale);

    // Group keys by base name
    let mut groups: HashMap<String, (Option<String>, Option<String>)> = HashMap::new();

    for key in names {
        // Parse diffusers-style keys
        if key.ends_with(".down.weight") || key.ends_with("_lora.down.weight") {
            let base = key
                .trim_end_matches(".down.weight")
                .trim_end_matches("_lora.down.weight")
                .trim_end_matches("_lora");
            groups.entry(base.to_string()).or_insert((None, None)).0 = Some(key.clone());
        } else if key.ends_with(".up.weight") || key.ends_with("_lora.up.weight") {
            let base = key
                .trim_end_matches(".up.weight")
                .trim_end_matches("_lora.up.weight")
                .trim_end_matches("_lora");
            groups.entry(base.to_string()).or_insert((None, None)).1 = Some(key.clone());
        }
    }

    for (base_name, (down_key, up_key)) in groups {
        let (down_key, up_key) = match (down_key, up_key) {
            (Some(d), Some(u)) => (d, u),
            _ => continue,
        };

        // Check shape to determine if linear (2D) or conv (4D)
        let shape = file.shape(&down_key);
        let ndims = shape.map(|s| s.len()).unwrap_or(0);

        if ndims == 2 {
            if let Ok(down) = file.load_f32::<B, 2>(&down_key, device) {
                if let Ok(up) = file.load_f32::<B, 2>(&up_key, device) {
                    let lora = LoraWeight::new(down, up, 1.0);
                    model.add_linear(base_name, lora);
                }
            }
        } else if ndims == 4 {
            if let Ok(down) = file.load_f32::<B, 4>(&down_key, device) {
                if let Ok(up) = file.load_f32::<B, 4>(&up_key, device) {
                    let lora = LoraConvWeight::new(down, up, 1.0);
                    model.add_conv(base_name, lora);
                }
            }
        }
    }

    Ok(model)
}

/// Convert Kohya-style layer name to internal format
fn kohya_to_internal_name(kohya_name: &str) -> String {
    // lora_unet_down_blocks_0_attentions_0_... -> unet.down_blocks.0.attentions.0...
    // Strip prefix first, then replace underscores, then prepend proper prefix
    let (prefix, rest) = if let Some(rest) = kohya_name.strip_prefix("lora_unet_") {
        ("unet.", rest)
    } else if let Some(rest) = kohya_name.strip_prefix("lora_te2_") {
        ("text_encoder_2.", rest)
    } else if let Some(rest) = kohya_name.strip_prefix("lora_te1_") {
        ("text_encoder_1.", rest)
    } else if let Some(rest) = kohya_name.strip_prefix("lora_te_") {
        ("text_encoder.", rest)
    } else {
        ("", kohya_name)
    };

    format!("{}{}", prefix, rest.replace('_', "."))
}

/// Errors that can occur when loading LoRA
#[derive(Debug)]
pub enum LoraLoadError {
    FileError(String),
    ParseError(String),
    TensorError(String),
}

impl std::fmt::Display for LoraLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileError(msg) => write!(f, "File error: {}", msg),
            Self::ParseError(msg) => write!(f, "Parse error: {}", msg),
            Self::TensorError(msg) => write!(f, "Tensor error: {}", msg),
        }
    }
}

impl std::error::Error for LoraLoadError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kohya_name_conversion() {
        assert_eq!(
            kohya_to_internal_name("lora_unet_down_blocks_0_attentions_0"),
            "unet.down.blocks.0.attentions.0"
        );
        assert_eq!(
            kohya_to_internal_name("lora_te_text_model_encoder_layers_0"),
            "text_encoder.text.model.encoder.layers.0"
        );
    }
}
