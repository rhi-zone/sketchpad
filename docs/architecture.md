# Architecture

## Crate Structure

```
sketchpad/
├── crates/
│   ├── sketchpad/          # Main library, re-exports everything
│   ├── rhi-sketchpad-core/     # Shared layers and utilities
│   ├── rhi-sketchpad-clip/     # Text encoders (CLIP, OpenCLIP)
│   ├── rhi-sketchpad-vae/      # Autoencoder (encoder + decoder)
│   ├── rhi-sketchpad-unet/     # UNet diffusion models (SD 1.x, SDXL)
│   ├── rhi-sketchpad-samplers/ # Diffusion samplers (DDIM, DPM++, Euler, etc.)
│   └── rhi-sketchpad-convert/  # Weight conversion from safetensors
└── src/                     # CLI binary
```

## Module Breakdown

### rhi-sketchpad-core

Shared building blocks:

- `attention.rs` - Multi-head attention, cross-attention
- `groupnorm.rs` - Group normalization
- `layernorm.rs` - Layer normalization
- `silu.rs` - SiLU activation
- `conv.rs` - Convolution helpers
- `linear.rs` - Linear layer helpers
- `timestep.rs` - Timestep embeddings

### rhi-sketchpad-clip

Text encoding:

- `tokenizer.rs` - BPE tokenizer (pure Rust, no Python)
- `clip.rs` - CLIP text encoder (SD 1.x)
- `open_clip.rs` - OpenCLIP encoder (SDXL)
- `embedder.rs` - Unified interface for single/dual encoders

### rhi-sketchpad-vae

Variational Autoencoder:

- `encoder.rs` - Image → latent (for img2img)
- `decoder.rs` - Latent → image
- `autoencoder.rs` - Combined VAE

### rhi-sketchpad-unet

Diffusion backbone:

- `blocks.rs` - ResNet blocks, attention blocks, down/up blocks
- `unet_sd.rs` - SD 1.x UNet
- `unet_sdxl.rs` - SDXL UNet (larger, different architecture)
- `conditioning.rs` - Conditioning types

### rhi-sketchpad-samplers

Noise schedulers and samplers:

- `scheduler.rs` - Noise schedule (linear, cosine, etc.)
- `ddim.rs` - DDIM sampler
- `ddpm.rs` - DDPM sampler
- `dpm.rs` - DPM++ variants
- `euler.rs` - Euler/Euler ancestral

### rhi-sketchpad-convert

Weight conversion:

- `safetensors.rs` - Load .safetensors files
- `mapping.rs` - Map weight names to our architecture
- `sd.rs` - SD 1.x weight mapping
- `sdxl.rs` - SDXL weight mapping

## Pipeline Abstraction

```rust
pub trait DiffusionPipeline<B: Backend> {
    type Conditioning;

    fn encode_prompt(&self, prompt: &str, negative: &str) -> Self::Conditioning;
    fn sample(&self, conditioning: Self::Conditioning, config: SampleConfig) -> Tensor<B, 4>;
    fn decode(&self, latent: Tensor<B, 4>) -> Tensor<B, 4>;
}
```

Both SD 1.x and SDXL implement this trait, enabling unified usage.

## SDXL Text Encoder Details

SDXL uses two text encoders with specific layer selection:

### Dual Encoder Architecture

| Encoder | Model | Output Dim | Usage |
|---------|-------|------------|-------|
| CLIP ViT-L/14 | `conditioner.embedders.0` | 768 | Hidden states only |
| OpenCLIP ViT-bigG/14 | `conditioner.embedders.1` | 1280 | Hidden states + pooled |

The hidden states are concatenated: `[batch, 77, 768] + [batch, 77, 1280] = [batch, 77, 2048]`

### Penultimate Layer (clip_skip=2)

**SDXL always uses the penultimate layer** (second-to-last) from both text encoders, not the final layer. This is equivalent to `clip_skip=2` in other tools.

```rust
// Correct for SDXL
let clip_hidden = clip_encoder.forward_penultimate(tokens);  // Stops at layer 10

// Wrong - would use layer 11 which may have garbage weights
let clip_hidden = clip_encoder.forward(tokens);  // Uses all 12 layers
```

**Why this matters**: Some merged SDXL checkpoints have NaN/garbage values in layer 11 of the first text encoder. Since SDXL never uses layer 11, these checkpoints work correctly in ComfyUI/A1111 but fail if you accidentally use `forward()` instead of `forward_penultimate()`.

### Sources

- [Stability-AI/generative-models#37](https://github.com/Stability-AI/generative-models/issues/37) - Confirms penultimate layer usage
- [huggingface/diffusers#3212](https://github.com/huggingface/diffusers/issues/3212) - clip_skip implementation
- [diffusers SDXL docs](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl) - Official pipeline docs

### Pooled Embedding

Only OpenCLIP provides the pooled embedding for SDXL's `add_embed` conditioning:

```rust
// OpenCLIP returns both hidden states and pooled output
let (hidden, pooled) = open_clip.forward_with_pooled(tokens, &[eos_pos]);

// pooled shape: [batch, 1280]
// This goes into add_embed along with size conditioning
```

The pooled embedding must use `forward_with_pooled()` which applies `text_projection` - manual slicing of hidden states does NOT work.

## SDXL Noise Schedule

**SDXL uses `scaled_linear` beta schedule**, NOT cosine. This is a common misconception.

| Parameter | Value |
|-----------|-------|
| `beta_schedule` | `scaled_linear` |
| `beta_start` | 0.00085 |
| `beta_end` | 0.012 |
| `num_train_timesteps` | 1000 |
| `prediction_type` | `epsilon` |

### scaled_linear vs linear

- **linear**: `betas = linspace(beta_start, beta_end, N)`
- **scaled_linear**: `betas = linspace(sqrt(beta_start), sqrt(beta_end), N) ** 2`

The squared interpolation produces slightly different noise levels. Using the wrong schedule causes completely garbled output.

```rust
// Correct for SDXL
let schedule = NoiseSchedule::sdxl(device);  // Uses scaled_linear

// Wrong - would produce garbled output
let schedule = NoiseSchedule::cosine(&config, device);
```

### Source

- [stabilityai/stable-diffusion-xl-base-1.0 scheduler_config.json](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/scheduler/scheduler_config.json)

## Design Principles

1. **Shared layers** - All common operations in `core`, no duplication
2. **Composable** - Each crate usable independently
3. **Pure Rust** - No Python dependencies
4. **Backend agnostic** - Works with any Burn backend (CUDA, Metal, WGPU, CPU)
5. **Feature flags** - Enable only what you need (`sd`, `sdxl`, `samplers-all`)
