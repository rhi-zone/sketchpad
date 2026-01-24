# sketchpad

Deep learning model inference in pure Rust using the [Burn](https://burn.dev) framework.

> [!WARNING]
> **Experimental** - Development is currently paused. Most model architectures have not been fully tested yet.

## Features

- **Pure Rust** - No Python dependencies, no ONNX runtime
- **Multiple Backends** - CPU (ndarray), CUDA, WebGPU, and PyTorch via libtorch
- **Image Generation** - Stable Diffusion 1.x/2.x, SDXL, Flux, SD3, PixArt, SANA, and more
- **Video Generation** - CogVideoX, Mochi, LTX-Video, Wan
- **Language Models** - LLaMA, Mistral, Qwen, Gemma, Phi, DeepSeek, RWKV, Mamba, Jamba
- **Memory Efficient** - VAE tiling, model offloading, quantization support

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rhi-sketchpad = { version = "0.1", features = ["wgpu"] }
```

Available backend features:
- `ndarray` - CPU backend (no GPU required)
- `tch` - PyTorch backend via libtorch (CUDA, MPS support)
- `wgpu` - WebGPU backend (cross-platform GPU)
- `cuda` - Native CUDA backend (NVIDIA only)

## Quick Start

### Image Generation (Stable Diffusion)

```rust
use rhi_sketchpad::prelude::*;

// Load model and generate
let pipeline = StableDiffusion1x::load("model.safetensors", &device)?;
let image = pipeline.generate("a sunset over mountains", "", &SampleConfig::default());
```

### DiT Models (Flux, SD3)

```rust
use rhi_sketchpad_dit::{Flux, FluxConfig};

let config = FluxConfig::schnell();
let (model, runtime) = config.init::<Backend>(&device);

let output = model.forward(latents, timestep, txt_embeds, img_ids, txt_ids, &runtime);
```

### Language Models

```rust
use rhi_sketchpad_llm::{Llama, LlamaConfig};

let config = LlamaConfig::llama3_8b();
let (model, runtime) = config.init::<Backend>(&device);

let output = model.forward(input_ids, None, &runtime);
let generated = model.generate(prompt_ids, &runtime, 100, 0.8);
```

## Supported Models

### Image Generation (UNet-based)
- Stable Diffusion 1.x, 2.x
- Stable Diffusion XL (base + refiner)
- Stable Cascade

### Image Generation (DiT-based)
- **Flux** - Black Forest Labs (schnell, dev)
- **Stable Diffusion 3/3.5** - MMDiT architecture
- **PixArt-α/Σ** - Efficient DiT with T5 encoder
- **AuraFlow** - Open source flow-based
- **Hunyuan-DiT** - Bilingual (Chinese/English)
- **SANA** - NVIDIA's linear DiT for fast 4K generation
- **Z-Image** - Alibaba's single-stream DiT
- **Qwen-Image** - 20B MMDiT

### Video Generation
- **CogVideoX** - 2B/5B open source video DiT
- **Mochi** - Genmo's asymmetric DiT
- **LTX-Video** - Lightricks video model
- **Wan 2.x** - Alibaba DiT + MoE

### Language Models
- **LLaMA 2/3** - Meta's models (7B to 70B)
- **Mistral/Mixtral** - Sliding window attention, MoE
- **Qwen 2.5** - Multilingual (0.5B to 72B)
- **Gemma 2** - Google's models
- **Phi-3/3.5** - Microsoft's efficient models
- **DeepSeek** - Multi-head Latent Attention
- **RWKV-7** - RNN with transformer performance
- **Mamba/Mamba-2** - Selective state space models
- **Jamba** - Transformer-Mamba-MoE hybrid

## Architecture

```
sketchpad/
├── rhi-sketchpad          # Main crate with pipelines
├── rhi-sketchpad-core     # Shared building blocks
│   ├── attention          # Multi-head, flash, paged attention
│   ├── rope               # Rotary position embeddings
│   ├── quantization       # INT4/INT8/FP8 quantization
│   └── ...
├── rhi-sketchpad-clip     # CLIP/OpenCLIP text encoders
├── rhi-sketchpad-vae      # VAE encoder/decoder (2D, 3D)
├── rhi-sketchpad-unet     # UNet for SD 1.x/2.x/XL
├── rhi-sketchpad-dit      # DiT models (Flux, SD3, video)
├── rhi-sketchpad-llm      # Language models
├── rhi-sketchpad-samplers # Diffusion samplers
└── rhi-sketchpad-convert  # Safetensors weight loading
```

## Samplers

Extensive sampler support for diffusion models:

- DDIM, DDPM, Euler, Euler Ancestral
- DPM++ 2M, DPM++ SDE, DPM++ 2S Ancestral
- Heun, LMS, UniPC, DEIS
- LCM (Latent Consistency)
- And many more...

## Model Extensions

- **LoRA** - Kohya and Diffusers formats
- **ControlNet** - SD 1.x and SDXL
- **IP-Adapter** - Image prompt conditioning
- **T2I-Adapter** - Lightweight conditioning
- **Textual Inversion** - Custom embeddings

## Memory Optimization

```rust
// VAE tiling for large images
let config = MemoryConfig::low_vram();
let tiler = TiledVae::new(config.vae_tile_size, config.vae_tile_overlap);

// Model offloading
let config = OffloadConfig::sequential_cpu_gpu();
```

## Model Weights

Load weights from safetensors format. Obtain weights from:

- [Hugging Face Hub](https://huggingface.co/models)
- [Civitai](https://civitai.com/)

## Documentation

See the [documentation site](https://rhi-zone.github.io/sketchpad) for detailed guides.

## License

MIT OR Apache-2.0
