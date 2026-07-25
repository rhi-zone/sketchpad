# burn-models

**Deep learning model inference in pure Rust with Burn.**

burn-models provides implementations of image generation, video generation, and language models using the [Burn](https://burn.dev) deep learning framework.

## Features

- **Pure Rust** - No Python dependencies, runs anywhere Rust runs
- **Multiple Backends** - WGPU, CUDA, and more via Burn
- **Image Generation** - SD 1.x, SDXL, Flux, SD3, PixArt, SANA
- **Video Generation** - CogVideoX, Mochi, LTX-Video, Wan
- **Language Models** - LLaMA, Mistral, Qwen, RWKV, Mamba, Jamba
- **Many Samplers** - DPM++, Euler, DDIM, LCM, and more
- **Model Extensions** - LoRA, ControlNet, IP-Adapter, Textual Inversion

## Quick Start

### Image Generation

```rust
use burn_models::prelude::*;

let pipeline = StableDiffusion1x::load("model.safetensors", &device)?;
let image = pipeline.generate("a photo of a cat", "", &SampleConfig::default());
```

### DiT Models

```rust
use rhi_sketchpad_dit::{Flux, FluxConfig};

let config = FluxConfig::schnell();
let (model, runtime) = config.init::<Backend>(&device);
let output = model.forward(latents, timestep, txt, img_ids, txt_ids, &runtime);
```

### Language Models

```rust
use rhi_sketchpad_llm::{Llama, LlamaConfig};

let config = LlamaConfig::llama3_8b();
let (model, runtime) = config.init::<Backend>(&device);
let generated = model.generate(prompt_ids, &runtime, 100, 0.8);
```

## Crates

| Crate | Description |
|-------|-------------|
| `burn-models` | Main pipeline and high-level API |
| `rhi-sketchpad-core` | Core building blocks (attention, RoPE, quantization) |
| `rhi-sketchpad-unet` | UNet architecture for SD 1.x and SDXL |
| `rhi-sketchpad-dit` | DiT models (Flux, SD3, video generation) |
| `rhi-sketchpad-llm` | Language models (LLaMA, Mistral, Mamba) |
| `rhi-sketchpad-vae` | VAE encoder and decoder |
| `rhi-sketchpad-clip` | CLIP and OpenCLIP text encoders |
| `rhi-sketchpad-samplers` | Diffusion samplers (DPM++, Euler, etc.) |
| `rhi-sketchpad-convert` | Weight loading from safetensors |

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
burn-models = "0.1"
```
