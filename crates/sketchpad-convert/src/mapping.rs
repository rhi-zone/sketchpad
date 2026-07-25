//! Map weight names from HuggingFace models to our architecture
//!
//! HuggingFace models use different naming conventions than our crate.
//! This module provides utilities to translate between them.

use std::collections::HashMap;

/// Weight name mapping for a model
pub struct WeightMapping {
    /// Map from our internal name to HuggingFace name
    mappings: HashMap<String, String>,
}

impl WeightMapping {
    /// Creates a new empty weight mapping
    pub fn new() -> Self {
        Self {
            mappings: HashMap::new(),
        }
    }

    /// Add a mapping from internal name to HuggingFace name
    pub fn add(&mut self, internal: impl Into<String>, hf: impl Into<String>) {
        self.mappings.insert(internal.into(), hf.into());
    }

    /// Get the HuggingFace name for an internal name
    pub fn get(&self, internal: &str) -> Option<&str> {
        self.mappings.get(internal).map(|s| s.as_str())
    }

    /// Iterate over all mappings
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.mappings.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }
}

impl Default for WeightMapping {
    fn default() -> Self {
        Self::new()
    }
}

/// Common prefixes in Stable Diffusion models
pub mod prefixes {
    pub const TEXT_ENCODER: &str = "text_model";
    pub const TEXT_ENCODER_2: &str = "text_model_2";
    pub const UNET: &str = "unet";
    pub const VAE: &str = "vae";
    pub const VAE_ENCODER: &str = "encoder";
    pub const VAE_DECODER: &str = "decoder";
}

/// Build weight mapping for CLIP text encoder
pub fn clip_text_encoder_mapping(prefix: &str) -> WeightMapping {
    let mut m = WeightMapping::new();

    // Embeddings
    m.add(
        "embeddings.token_embedding.weight",
        format!("{prefix}.embeddings.token_embedding.weight"),
    );
    m.add(
        "embeddings.position_embedding.weight",
        format!("{prefix}.embeddings.position_embedding.weight"),
    );

    // Final layer norm
    m.add(
        "final_layer_norm.weight",
        format!("{prefix}.final_layer_norm.weight"),
    );
    m.add(
        "final_layer_norm.bias",
        format!("{prefix}.final_layer_norm.bias"),
    );

    m
}
