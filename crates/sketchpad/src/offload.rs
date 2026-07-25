//! Model offloading for memory-constrained environments
//!
//! Enables running large models by moving components between CPU and GPU.

/// Offloading strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OffloadStrategy {
    /// Keep everything on GPU (default, fastest)
    #[default]
    None,
    /// Offload VAE to CPU when not in use
    VaeOffload,
    /// Offload text encoder after encoding
    TextEncoderOffload,
    /// Sequential: only one component on GPU at a time
    Sequential,
    /// Aggressive: offload everything possible
    Aggressive,
}

impl OffloadStrategy {
    /// Get a human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            OffloadStrategy::None => "none",
            OffloadStrategy::VaeOffload => "vae_offload",
            OffloadStrategy::TextEncoderOffload => "text_encoder_offload",
            OffloadStrategy::Sequential => "sequential",
            OffloadStrategy::Aggressive => "aggressive",
        }
    }

    /// Estimated VRAM reduction (approximate)
    pub fn vram_savings(&self) -> f32 {
        match self {
            OffloadStrategy::None => 0.0,
            OffloadStrategy::VaeOffload => 0.15,
            OffloadStrategy::TextEncoderOffload => 0.10,
            OffloadStrategy::Sequential => 0.50,
            OffloadStrategy::Aggressive => 0.70,
        }
    }
}

/// Configuration for model offloading
#[derive(Debug, Clone)]
pub struct OffloadConfig {
    /// Which strategy to use
    pub strategy: OffloadStrategy,
    /// Whether to use CPU offload (vs disk/swap)
    pub use_cpu: bool,
    /// Pin memory for faster transfers
    pub pin_memory: bool,
    /// Prefetch next component before current finishes
    pub prefetch: bool,
}

impl Default for OffloadConfig {
    fn default() -> Self {
        Self {
            strategy: OffloadStrategy::None,
            use_cpu: true,
            pin_memory: true,
            prefetch: false,
        }
    }
}

impl OffloadConfig {
    /// No offloading (keep everything on GPU)
    pub fn none() -> Self {
        Self::default()
    }

    /// Offload VAE to save ~1GB VRAM
    pub fn vae_offload() -> Self {
        Self {
            strategy: OffloadStrategy::VaeOffload,
            ..Default::default()
        }
    }

    /// Sequential offloading for low VRAM
    pub fn sequential() -> Self {
        Self {
            strategy: OffloadStrategy::Sequential,
            prefetch: true,
            ..Default::default()
        }
    }

    /// Aggressive offloading for very low VRAM
    pub fn aggressive() -> Self {
        Self {
            strategy: OffloadStrategy::Aggressive,
            prefetch: true,
            ..Default::default()
        }
    }
}

/// Model component that can be offloaded
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelComponent {
    /// Text encoder (CLIP)
    TextEncoder,
    /// Second text encoder (SDXL)
    TextEncoder2,
    /// UNet
    UNet,
    /// VAE encoder
    VaeEncoder,
    /// VAE decoder
    VaeDecoder,
    /// ControlNet
    ControlNet,
    /// LoRA weights
    Lora,
}

impl ModelComponent {
    /// Approximate VRAM usage in GB
    pub fn vram_usage(&self) -> f32 {
        match self {
            ModelComponent::TextEncoder => 0.5,
            ModelComponent::TextEncoder2 => 0.5,
            ModelComponent::UNet => 3.5,
            ModelComponent::VaeEncoder => 0.3,
            ModelComponent::VaeDecoder => 0.3,
            ModelComponent::ControlNet => 1.5,
            ModelComponent::Lora => 0.1,
        }
    }

    /// Whether this component is needed during the diffusion loop
    pub fn needed_during_diffusion(&self) -> bool {
        matches!(
            self,
            ModelComponent::UNet | ModelComponent::ControlNet | ModelComponent::Lora
        )
    }
}

/// Track which components are currently on GPU
#[derive(Debug, Default)]
pub struct OffloadState {
    /// Components currently on GPU
    on_gpu: std::collections::HashSet<ModelComponent>,
    /// Current strategy
    strategy: OffloadStrategy,
}

impl OffloadState {
    /// Create a new offload state
    pub fn new(strategy: OffloadStrategy) -> Self {
        Self {
            on_gpu: std::collections::HashSet::new(),
            strategy,
        }
    }

    /// Check if component is on GPU
    pub fn is_on_gpu(&self, component: ModelComponent) -> bool {
        self.on_gpu.contains(&component)
    }

    /// Mark component as on GPU
    pub fn move_to_gpu(&mut self, component: ModelComponent) {
        self.on_gpu.insert(component);
    }

    /// Mark component as offloaded
    pub fn move_to_cpu(&mut self, component: ModelComponent) {
        self.on_gpu.remove(&component);
    }

    /// Get components that should be offloaded before using the given component
    pub fn get_offload_candidates(&self, needed: ModelComponent) -> Vec<ModelComponent> {
        match self.strategy {
            OffloadStrategy::None => vec![],
            OffloadStrategy::VaeOffload => {
                if needed == ModelComponent::UNet {
                    vec![ModelComponent::VaeEncoder, ModelComponent::VaeDecoder]
                } else {
                    vec![]
                }
            }
            OffloadStrategy::TextEncoderOffload => {
                if needed == ModelComponent::UNet {
                    vec![ModelComponent::TextEncoder, ModelComponent::TextEncoder2]
                } else {
                    vec![]
                }
            }
            OffloadStrategy::Sequential | OffloadStrategy::Aggressive => {
                // Offload everything except what's needed
                self.on_gpu
                    .iter()
                    .filter(|&&c| c != needed)
                    .copied()
                    .collect()
            }
        }
    }

    /// Estimate current VRAM usage
    pub fn estimated_vram(&self) -> f32 {
        self.on_gpu.iter().map(|c| c.vram_usage()).sum()
    }
}

/// Pipeline execution phase (for determining what to offload)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelinePhase {
    /// Encoding text prompt
    TextEncoding,
    /// Preparing latents (VAE encode for img2img)
    LatentPrep,
    /// Main diffusion loop
    Diffusion,
    /// Decoding latents to image
    Decoding,
}

impl PipelinePhase {
    /// Get components needed for this phase
    pub fn required_components(&self) -> Vec<ModelComponent> {
        match self {
            PipelinePhase::TextEncoding => {
                vec![ModelComponent::TextEncoder, ModelComponent::TextEncoder2]
            }
            PipelinePhase::LatentPrep => {
                vec![ModelComponent::VaeEncoder]
            }
            PipelinePhase::Diffusion => {
                vec![
                    ModelComponent::UNet,
                    ModelComponent::ControlNet,
                    ModelComponent::Lora,
                ]
            }
            PipelinePhase::Decoding => {
                vec![ModelComponent::VaeDecoder]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_offload_strategy_default() {
        assert_eq!(OffloadStrategy::default(), OffloadStrategy::None);
    }

    #[test]
    fn test_offload_config_presets() {
        let sequential = OffloadConfig::sequential();
        assert_eq!(sequential.strategy, OffloadStrategy::Sequential);
        assert!(sequential.prefetch);
    }

    #[test]
    fn test_component_vram() {
        assert!(ModelComponent::UNet.vram_usage() > ModelComponent::TextEncoder.vram_usage());
    }

    #[test]
    fn test_offload_state() {
        let mut state = OffloadState::new(OffloadStrategy::Sequential);

        state.move_to_gpu(ModelComponent::UNet);
        assert!(state.is_on_gpu(ModelComponent::UNet));

        state.move_to_cpu(ModelComponent::UNet);
        assert!(!state.is_on_gpu(ModelComponent::UNet));
    }

    #[test]
    fn test_pipeline_phases() {
        let phase = PipelinePhase::Diffusion;
        let required = phase.required_components();
        assert!(required.contains(&ModelComponent::UNet));
    }
}
