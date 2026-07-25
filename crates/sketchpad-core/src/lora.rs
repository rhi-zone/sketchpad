//! LoRA (Low-Rank Adaptation) support
//!
//! Implements LoRA weight loading and application for fine-tuning
//! Stable Diffusion models without full model replacement.

use burn::prelude::*;
use std::collections::HashMap;

/// LoRA weight pair (down/up projections)
#[derive(Debug, Clone)]
pub struct LoraWeight<B: Backend> {
    /// Down projection: [rank, in_features]
    pub lora_down: Tensor<B, 2>,
    /// Up projection: [out_features, rank]
    pub lora_up: Tensor<B, 2>,
    /// Alpha scaling factor
    pub alpha: f32,
    /// Rank of the LoRA matrices
    pub rank: usize,
}

impl<B: Backend> LoraWeight<B> {
    /// Create a new LoRA weight pair
    pub fn new(lora_down: Tensor<B, 2>, lora_up: Tensor<B, 2>, alpha: f32) -> Self {
        let rank = lora_down.dims()[0];
        Self {
            lora_down,
            lora_up,
            alpha,
            rank,
        }
    }

    /// Compute the delta weight: scale * (up @ down)
    ///
    /// Returns a tensor that can be added to the original weight.
    pub fn compute_delta(&self, scale: f32) -> Tensor<B, 2> {
        let effective_scale = scale * self.alpha / self.rank as f32;
        self.lora_up.clone().matmul(self.lora_down.clone()) * effective_scale
    }

    /// Apply LoRA to an input tensor (for linear layers)
    ///
    /// Computes: x @ down.T @ up.T * scale
    pub fn forward(&self, x: Tensor<B, 2>, scale: f32) -> Tensor<B, 2> {
        let effective_scale = scale * self.alpha / self.rank as f32;

        // x: [batch, in_features]
        // lora_down: [rank, in_features] -> transpose to [in_features, rank]
        // lora_up: [out_features, rank] -> transpose to [rank, out_features]

        let down = x.matmul(self.lora_down.clone().transpose());
        let up = down.matmul(self.lora_up.clone().transpose());

        up * effective_scale
    }
}

/// LoRA weights for convolutional layers
#[derive(Debug, Clone)]
pub struct LoraConvWeight<B: Backend> {
    /// Down projection weights [out_ch_lora, in_ch, kh, kw]
    pub lora_down: Tensor<B, 4>,
    /// Up projection weights [out_ch, out_ch_lora, 1, 1]
    pub lora_up: Tensor<B, 4>,
    /// Alpha scaling factor
    pub alpha: f32,
    /// Rank of the LoRA matrices
    pub rank: usize,
}

impl<B: Backend> LoraConvWeight<B> {
    /// Create a new LoRA conv weight pair
    pub fn new(lora_down: Tensor<B, 4>, lora_up: Tensor<B, 4>, alpha: f32) -> Self {
        let rank = lora_down.dims()[0];
        Self {
            lora_down,
            lora_up,
            alpha,
            rank,
        }
    }

    /// Compute the delta weight for conv layers
    ///
    /// This is more complex than linear LoRA due to the 4D nature of conv weights.
    pub fn compute_delta(&self, scale: f32) -> Tensor<B, 4> {
        let effective_scale = scale * self.alpha / self.rank as f32;

        // Reshape for matrix multiplication
        let [out_rank, in_ch, kh, kw] = self.lora_down.dims();
        let [out_ch, _, _, _] = self.lora_up.dims();

        // down: [rank, in_ch * kh * kw]
        let down_flat = self.lora_down.clone().reshape([out_rank, in_ch * kh * kw]);

        // up: [out_ch, rank]
        let up_flat = self.lora_up.clone().reshape([out_ch, out_rank]);

        // result: [out_ch, in_ch * kh * kw]
        let delta_flat = up_flat.matmul(down_flat) * effective_scale;

        // Reshape back to conv format
        delta_flat.reshape([out_ch, in_ch, kh, kw])
    }
}

/// Collection of LoRA weights for a model
pub struct LoraModel<B: Backend> {
    /// LoRA weights keyed by layer name
    pub weights: HashMap<String, LoraWeightType<B>>,
    /// Global scale factor for this LoRA
    pub scale: f32,
}

/// Either linear or conv LoRA weight
pub enum LoraWeightType<B: Backend> {
    Linear(LoraWeight<B>),
    Conv(LoraConvWeight<B>),
}

impl<B: Backend> LoraModel<B> {
    /// Create a new empty LoRA model
    pub fn new(scale: f32) -> Self {
        Self {
            weights: HashMap::new(),
            scale,
        }
    }

    /// Add a linear LoRA weight
    pub fn add_linear(&mut self, name: String, weight: LoraWeight<B>) {
        self.weights.insert(name, LoraWeightType::Linear(weight));
    }

    /// Add a conv LoRA weight
    pub fn add_conv(&mut self, name: String, weight: LoraConvWeight<B>) {
        self.weights.insert(name, LoraWeightType::Conv(weight));
    }

    /// Get a LoRA weight by name
    pub fn get(&self, name: &str) -> Option<&LoraWeightType<B>> {
        self.weights.get(name)
    }

    /// Check if a layer has LoRA weights
    pub fn has(&self, name: &str) -> bool {
        self.weights.contains_key(name)
    }

    /// Get the number of LoRA layers
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Set the global scale
    pub fn set_scale(&mut self, scale: f32) {
        self.scale = scale;
    }
}

/// Apply LoRA delta to a linear weight
pub fn apply_lora_to_linear<B: Backend>(
    weight: Tensor<B, 2>,
    lora: &LoraWeight<B>,
    scale: f32,
) -> Tensor<B, 2> {
    let delta = lora.compute_delta(scale);
    weight + delta
}

/// Apply LoRA delta to a conv weight
pub fn apply_lora_to_conv<B: Backend>(
    weight: Tensor<B, 4>,
    lora: &LoraConvWeight<B>,
    scale: f32,
) -> Tensor<B, 4> {
    let delta = lora.compute_delta(scale);
    weight + delta
}

/// Merge multiple LoRA models into weights
///
/// Applies multiple LoRAs with their respective scales.
pub fn merge_loras<B: Backend>(
    loras: &[(&LoraModel<B>, f32)],
    layer_name: &str,
    original_weight: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let mut result = original_weight;

    for (lora, scale) in loras {
        if let Some(LoraWeightType::Linear(lora_weight)) = lora.get(layer_name) {
            let delta = lora_weight.compute_delta(*scale * lora.scale);
            result = result + delta;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_lora_weight_creation() {
        let device = <TestBackend as Backend>::Device::default();

        // down: [rank, in_features], up: [out_features, rank]
        let down: Tensor<TestBackend, 2> = Tensor::zeros([4, 64], &device);
        let up: Tensor<TestBackend, 2> = Tensor::zeros([128, 4], &device);

        let lora = LoraWeight::new(down, up, 1.0);
        assert_eq!(lora.rank, 4);
    }

    #[test]
    fn test_lora_delta_shape() {
        let device = <TestBackend as Backend>::Device::default();

        // down: [rank, in_features] = [4, 64]
        // up: [out_features, rank] = [128, 4]
        let down: Tensor<TestBackend, 2> = Tensor::ones([4, 64], &device);
        let up: Tensor<TestBackend, 2> = Tensor::ones([128, 4], &device);

        let lora = LoraWeight::new(down, up, 1.0);
        let delta = lora.compute_delta(1.0);

        // delta = up @ down = [128, 4] @ [4, 64] = [128, 64]
        assert_eq!(delta.dims(), [128, 64]);
    }

    #[test]
    fn test_lora_model() {
        let device = <TestBackend as Backend>::Device::default();

        let mut model = LoraModel::<TestBackend>::new(1.0);

        let down: Tensor<TestBackend, 2> = Tensor::zeros([4, 64], &device);
        let up: Tensor<TestBackend, 2> = Tensor::zeros([128, 4], &device);
        let lora = LoraWeight::new(down, up, 1.0);

        model.add_linear("layer1".to_string(), lora);

        assert!(model.has("layer1"));
        assert!(!model.has("layer2"));
        assert_eq!(model.len(), 1);
    }
}
