# TODO

## Next Up

- [x] Rename project from burn-image to burn-models
- [x] Restructure crates for general model inference

## Completed (Image Generation)

- [x] Explore stable-diffusion-burn source to understand architecture
- [x] Explore stable-diffusion-xl-burn source for SDXL differences
- [x] Define crate structure (workspace with 7 crates)
- [x] Implement weight loading from safetensors
- [x] Implement CLIP tokenizer
- [x] Implement CLIP text encoder
- [x] Implement DDIM sampler
- [x] Implement VAE decoder
- [x] Implement UNet

## Roadmap

### Phase 1: Foundation

- [x] Weight loading infrastructure (safetensors → Burn tensors)
- [x] Model serialization (save/load converted weights)
- [x] Basic tensor ops (silu, groupnorm, layernorm)
- [x] Attention mechanism (qkv_attention, causal_mask)

### Phase 2: SD 1.x Components

- [x] CLIP text encoder
- [x] VAE decoder (latent → image)
- [x] UNet (diffusion backbone)
- [x] DDIM sampler

### Phase 3: SD 1.x Pipeline

- [x] Text-to-image generation
- [x] Classifier-free guidance
- [x] VAE encoder (for img2img)
- [x] Image-to-image pipeline

### Phase 4: SDXL Components

- [x] OpenCLIP text encoder (second encoder)
- [x] SDXL UNet (larger, different architecture)
- [x] SDXL VAE (scaling factors: 0.13025 vs 0.18215)
- [x] Refiner model

### Phase 5: SDXL Pipeline

- [x] SDXL text-to-image
- [x] SDXL img2img
- [x] Inpainting
- [x] Base + refiner workflow

### Phase 6: Polish

- [x] Multi-backend support (CUDA, Metal, WGPU)
- [x] Memory optimization
- [x] CLI interface
- [x] Documentation

## Backlog

### Precision Support

**bf16 (Brain Float16)** - IMPLEMENTED ✅
- Same exponent range as f32, so no overflow issues
- Native on Ampere+ GPUs (RTX 30xx/40xx)
- Now the default precision
- Use `--precision bf16` (or just default)

**f32 + Flash Attention** - Working ✅
- Stable, recommended for quality
- Use `--precision f32`

**Half-precision (f16/bf16) with Flash Attention** - BLOCKED (cubek-attention bug)
- cubek-attention 0.1.0-pre.1 has alignment bug with BOTH f16 and bf16
- Unit(Inferred) strategy: Assertion fails `unit_tile.layout.num_cols % line_size == 0`
- Root cause: hardcoded tile_size=4 doesn't align with CUDA's line_size=8 for half-precision
- **Workaround**: Using simple flash attention impl in `burn-models-cubecl/src/flash_attention.rs`
- **Upstream fix**: https://github.com/tracel-ai/cubek/pull/55
- Once merged and released, switch to cubek-attention for better performance

**Tasks**:
- [x] Add `--flash-attention` CLI flag (default enabled)
- [x] bf16 support with Cargo feature flags
- [x] Create pipeline variant using `CrossAttentionCubeCL` with flash attention
- [ ] Test bf16 generation on different GPU architectures (Turing, Ampere, Ada)

**Low Priority** (f16 overflow requires upcasting ALL matmuls, not just attention):
- [x] Upcast GroupNorm/RMSNorm variance to f32 (fixes NaN in f16 precision)
- [x] `Upcasted<B>` Linear wrapper — f32 compute, cast output back to original dtype
- [x] Apply f32 upcast to attention projections in f16 models — inline dtype guard in all 7 attention modules
- [ ] Mixed precision pipeline (different precision per component)

**Cargo feature flags for precision presets**:
```bash
# Default: all precisions
cargo build -p burn-models-cli --features cuda

# bf16 only (minimal binary)
cargo build -p burn-models-cli --no-default-features --features cuda,preset-fast

# f32 only (no JIT overhead)
cargo build -p burn-models-cli --no-default-features --features cuda,preset-quality
```

### Code Organization (High Priority)
- [x] Split pipeline.rs into separate files by model: sd1x.rs, sdxl.rs, mod.rs
- [x] Split large model files — unet_sd.rs (496L), blocks.rs (536L) already small; cubecl.rs (1109L) and stable_cascade.rs (1025L) are the largest remaining if needed

- [x] SD1x CLI defaults: 512x512 (native resolution), f16 precision (2026-01-07)

### Samplers
- [x] DDIM
- [x] Euler
- [x] Euler Ancestral
- [x] DPM++ 2M
- [x] DPM++ SDE
- [x] Heun
- [x] HeunPP2
- [x] DPM 2
- [x] DPM 2 Ancestral
- [x] LMS
- [x] DPM++ 3M SDE
- [x] DDPM
- [x] LCM (Latent Consistency Model)
- [x] iPNDM
- [x] iPNDM-v
- [x] DEIS
- [x] UniPC
- [x] SA-Solver
- [x] Euler CFG++
- [x] Euler Ancestral CFG++
- [x] DPM Fast
- [x] DPM Adaptive
- [x] DPM++ 2S Ancestral
- [x] DPM++ 2S Ancestral CFG++
- [x] DPM++ 2M CFG++
- [x] DPM++ 2M SDE Heun
- [x] Res Multistep

### Model Extensions
- [x] LoRA support (Kohya, Diffusers formats)
- [x] ControlNet integration (SD 1.x, SDXL)
- [x] Textual inversion / embeddings
- [x] IP-Adapter
- [x] T2I-Adapter

### Performance
- [x] fp16/bf16 inference (precision config)
- [x] Flash Attention (tiled, memory-efficient, sliced)
- [x] xFormers-style memory efficient attention
- [x] Model offloading (sequential CPU/GPU)
- [x] Batch processing

### CubeCL Optimization (burn-models-cubecl crate)

Custom GPU kernels for operations that benefit from fusion/specialization.
Decision: Own crate rather than upstream (faster iteration, avoid "vibe code" concerns).

#### Phase 1: Infrastructure
- [x] Create burn-models-cubecl crate with cubecl dependency
- [x] Set up feature flags (cpu default, wgpu, cuda)
- [x] Add benchmark harness for comparing against tensor-ops implementations

#### Phase 2: Conv3d Kernel
- [x] Port conv_transpose3d pattern to conv3d (simple direct kernel, ~200 lines)
- [x] NTHWC layout handling (permute in, permute out)
- [x] Test harness comparing CubeCL vs im2col output (correctness) ✓ PASSING
  - CUDA: `cargo test -p burn-models-cubecl --features cuda --test correctness_cuda -- --ignored`
- [x] Benchmark against im2col implementation ✓ **630-40,900× speedup**
  - Run: `cargo bench -p burn-models-cubecl --features cuda --bench conv3d`
  - Results: see `docs/cubecl-guide.md` for full benchmark table
- [x] Provide CubeCL Conv3d layer (removed im2col from burn-models-core)
  - `Conv3dLayer<R>` in burn-models-cubecl for all CubeBackends
  - Helper functions `to_cube_tensor`/`from_cube_tensor` for type conversion
  - Comprehensive test suite: 9 CPU tests, 8 CUDA tests, 6 WGPU tests (includes NTHWC layout tests)

#### Phase 3: Conv3d Optimization
- [x] Add Line<E> vectorization with NTHWC kernel (`conv3d_nthwc`)
- [x] Recursive kernel_loop with comptime dimension unrolling
- [x] FastDivmod for efficient index calculation
- [x] Benchmark simple vs optimized kernel on CUDA

**Benchmark Results** (RTX 3060):

| Config | Simple (NCTHW) | Optimized (NTHWC) | Speedup | vs im2col |
|--------|----------------|-------------------|---------|-----------|
| tiny (1,2,4,4,8,8) | **12.5 µs** | 24.9 µs | 0.5× | 470× |
| small (1,4,8,8,32,32) | 63.8 µs | **41.3 µs** | 1.5× | 39,800× |
| medium (1,8,16,8,64,64) | 742 µs | **286 µs** | 2.6× | - |
| strided (stride=2) | 107 µs | **97.9 µs** | 1.1× | - |
| deep (1,32,64,4,32,32) | 1.04 ms | **866 µs** | 1.2× | - |

**Findings:**
- Simple kernel wins for tiny tensors (kernel launch overhead dominates)
- Optimized kernel wins 1.1-2.6× for larger tensors
- Permutation overhead is negligible BUT requires contiguous copy (memory allocation)
- Both kernels retained: simple for NCTHW (no memory overhead), optimized for NTHWC

See `docs/cubecl-conv3d.md` for detailed analysis.

#### Phase 4: Additional Kernels (as needed)
- [x] AvgPool3d kernel (5 CPU tests passing)
- [x] MaxPool3d kernel (5 CPU tests passing)
- [x] Expose burn-cubecl's flash attention (cubek-attention) via burn-models-cubecl
  - `flash_attention`, `flash_attention_masked` functions with `FlashAttentionOptions`
  - Supports both causal (LLMs) and non-causal (diffusion) via `FlashAttentionOptions::causal()`
  - Default is non-causal (suitable for diffusion models)
  - Calls cubek::attention directly (bypasses burn-cubecl's hardcoded causal=true)
  - Removed unused tensor-ops fallback from burn-models-core (was dead code)
  - Note: CPU backend has line_size constraints that cause assertion failures (GPU backends work)
  - FlashAttention3: Hopper-only (H100/H800), lower priority
- [x] Custom activation fusions (GroupNorm+SiLU)
  - `groupnorm_silu(input, weight, bias, options)` - fused GroupNorm + SiLU
  - `groupnorm(input, weight, bias, options)` - GroupNorm without SiLU
  - Two-phase kernel: compute group stats, then normalize + affine + SiLU
  - 5 CPU tests passing (tolerance 1e-2 due to biased vs unbiased variance)

#### Phase 5: Integration into Existing Models

Integrate CubeCL kernels into existing model implementations.

- [x] GroupNorm+SiLU → UNet ResNet blocks
  - Added `burn-models-unet/src/cubecl.rs` with `ResBlockCubeCL<R: CubeRuntime>`
  - Uses fused `groupnorm_silu` kernel in forward pass
  - `convert_resblock()` converts existing `ResBlock` to CubeCL version
  - Feature-gated: `--features cubecl`

- [x] FlashAttention → Transformer attention layers
  - Added `CrossAttentionCubeCL<R: CubeRuntime>` to `burn-models-unet/src/cubecl.rs`
  - Uses O(n) memory flash attention instead of materializing full attention matrix
  - `convert_crossattention()` converts existing `CrossAttention` to CubeCL version
  - Non-causal by default (diffusion models)

#### Phase 6: 3D VAE for Video Models

Implement actual 3D VAE encoder/decoder using Conv3d kernels.
Target: CogVideoX, Wan, Mochi video generation.

- [x] 3D VAE Encoder
  - `Vae3dEncoderCubeCL<R: CubeRuntime>` in `burn-models-core/src/vae3d.rs`
  - Uses Conv3d + fused GroupNorm+SiLU kernels
  - ResBlock3dCubeCL, Downsample3dCubeCL building blocks
  - Feature-gated: `--features cubecl`

- [x] 3D VAE Decoder
  - `Vae3dDecoderCubeCL<R: CubeRuntime>` in `burn-models-core/src/vae3d.rs`
  - Uses Conv3d + fused GroupNorm+SiLU kernels
  - Upsample3dCubeCL for temporal/spatial upsampling

- [x] Weight loading infrastructure for CogVideoX/Mochi safetensors
  - `Vae3dKeyMapping` enum for model-specific key naming conventions
  - `expected_vae3d_weights()` for validation
  - Works with existing `SafeTensorFile` from burn-models-convert

See `docs/cubecl-guide.md` for implementation details.

#### Phase 7: Polish & Usability

- [x] GPU integration tests for CubeCL VAE (WGPU/CUDA)
  - UNet: ResBlockCubeCL, CrossAttentionCubeCL shape tests
  - VAE: ResBlock3dCubeCL, Downsample3dCubeCL, Upsample3dCubeCL shape tests
  - VAE encoder/decoder tiny model integration tests
- [x] Benchmark CubeCL integrations
  - `benches/unet_blocks.rs`: ResBlock vs ResBlockCubeCL, CrossAttention vs CrossAttentionCubeCL
  - Run: `cargo bench -p burn-models-cubecl --features cuda --bench unet_blocks`
- [ ] ~~MLX backend (Apple Silicon)~~ - **BLOCKED**: burn-mlx 0.1.2 requires burn ^0.16, we're on 0.20
- [x] Wire up CLI `generate` command - now validates paths and shows helpful messages
- [x] WeightLoader for SD models - enables actual image generation
  - [x] CLIP text encoder loader (`SdWeightLoader::load_clip_text_encoder`)
    - Detects prefix automatically (text_model, text_encoder.text_model, etc.)
    - Loads embeddings, all transformer layers, final layer norm
  - [x] UNet loader (`SdWeightLoader::load_unet`)
    - Detects prefix (model.diffusion_model, unet, etc.)
    - Loads time embedding, conv_in, all down/mid/up blocks, conv_out
    - Each block: ResBlock, SpatialTransformer, Downsample/Upsample
    - **Note**: Currently supports HF diffusers naming only. Single-file checkpoints
      (CivitAI/CompVis format) use different naming and need mapping:
      - CompVis: `input_blocks`/`output_blocks`, `in_layers.0/2`
      - HF: `down_blocks`/`up_blocks`, `norm1/conv1`
  - [x] VAE decoder loader (`SdWeightLoader::load_vae_decoder`)
    - Detects prefix (decoder, vae.decoder, first_stage_model.decoder)
    - Loads conv_in, mid blocks, all up blocks, conv_out
    - Each block: ResnetBlock, SelfAttention, Upsample
  - [x] CompVis naming support for single-file checkpoints (CivitAI models) — `UNetNaming::detect()` auto-detects, routes to `load_unet_compvis()`; VAE supports `first_stage_model.decoder` prefix

### Future Architectures

#### UNet-based (similar to current)
- [x] Stable Diffusion 2.x - OpenCLIP ViT-H/14 text encoder, v-prediction support in samplers
- [x] Stable Cascade - Würstchen architecture, latent cascade

#### DiT-based (new architecture needed)
- [x] Flux (Black Forest Labs) - Flow matching, DiT architecture, weight loading
- [x] Stable Diffusion 3 / 3.5 - MMDiT (Multimodal Diffusion Transformer), rectified flow
- [x] Z-Image (Alibaba) - Single-stream DiT (S3-DiT), 6B params, text+image tokens in single stream
- [x] Qwen-Image (Alibaba) - 20B MMDiT, Qwen2.5-VL text encoder, joint attention with pooled conditioning
- [x] PixArt-α/Σ - Efficient DiT, T5 text encoder, cross-attention to text
- [x] AuraFlow - Open source flow-based, MMDiT joint attention
- [x] Hunyuan-DiT - Bilingual DiT model, dual text encoder (CLIP + MT5), skip connections

### Video Generation

- [x] Wan 2.x (Alibaba) - DiT + MoE with sparse routing, 3D VAE, factorized spatial-temporal attention
- [x] CogVideoX - Open source video DiT, factorized spatial-temporal attention
- [x] Mochi - Genmo's open source video model, asymmetric DiT, factorized attention
- [x] LTX-Video - Lightricks video model, causal temporal attention

### Large Language Models

- [x] LLaMA 2/3 architecture (basic implementation)
- [x] LLaMA weight loading from safetensors
- [x] Mixtral 8x7B/8x22B (LLaMA + MoE)
- [x] Mistral 7B/Nemo - Sliding window attention
- [x] Qwen 2.5 - Alibaba's multilingual LLM (0.5B to 72B)
- [x] Gemma 2 - Google's open models (interleaved sliding/global attention, logit soft-capping, GeGLU)
- [x] Phi-3/3.5 - Microsoft's small models (fused QKV/gate-up, GQA)
- [x] DeepSeek V1/V2 - Multi-head Latent Attention (MLA) for compressed KV cache
- [x] Gemma 4 MoE - 128 routed experts, top-8 routing, shared expert, interleaved attention
- [x] GGUF format loading - GGUF v3 parser, dequant for Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K
- [x] Gemma 4 GGUF loader - auto-config from metadata, fused expert tensor splitting
- [x] Sampler pipeline - DRY, XTC, repetition penalty, temperature, top-k, top-p, min-p
- [x] KV cache for LLaMA (prefill+decode with ModelKvCache)
- [x] KV cache for Gemma 4 (prefill+decode with ModelKvCache)
- [x] KV cache for remaining models (Gemma 2, Mistral, Mixtral, Phi, Qwen, DeepSeek)
- [x] Wire sampler into remaining models (SamplerConfig in all models: LLaMA, Gemma 2/4, Mistral, Mixtral, Phi, Qwen, DeepSeek, RWKV, Mamba, Jamba)
- [ ] End-to-end GGUF test with Gemma 4 26B-A4B (Q4_K_S) — use `load_gemma4_gguf_gpu_quant`; attention/FFN weights stay as Q4K bytes in VRAM (~1–2 GB), MoE experts zero-copy mmap'd. Should fit a 16 GB GPU. Unrun, not blocked.
- [x] Anthropic-compatible HTTP API adapter — `POST /v1/messages`, SSE streaming, `sketchpad llm serve` CLI command (feature: `llm-serve`)
- [x] TurboQuant KV cache compression — `CompressedKvCache<B>` with PolarQuant (~3.9x vs f16, magnitude f16 + 4-bit direction); `--kv-quant` CLI flag; KV cache size included in VRAM budget estimate
- [x] TurboQuant: wire `CompressedKvCache` into model inference — `--kv-quant` active during inference via `KvCacheConfig` in `GenerationConfig`
- [x] TurboQuant: QJL variant — 1-bit sign quantization after random Gaussian projection, LCG-seeded deterministic R matrix, `compression_ratio = (2 + proj_dim/8) / (head_dim * 2)`

#### Memory / Offloading (GGUF inference)

Three strategies compose into a tier system. Strategies 1 and 3 pair well (experts on disk,
attention in VRAM quantized). Strategies 2 and 3 are mutually exclusive for the same weights
(CPU vs VRAM for attention). Strategy 1 can generalize to ALL weights, not just experts.

**`WeightStorage<B>` enum** (unifying abstraction, implement before 2 and 3):
```rust
enum WeightStorage<B: Backend> {
    GpuDequantized(Linear<B>),      // current default: f16/f32 in VRAM
    CpuQuantized(QuantizedLinear),  // strategy 2: bytes in RAM, upload per token
    MmapQuantized(Arc<Mmap>, usize, usize, GgmlType),  // strategy 1: OS page cache
    // GpuQuantized: strategy 3, after CubeCL kernel exists
}
```
**Integration**: avoid restructuring `Module` types. Store offloaded weights in a runtime
side-table `HashMap<(layer_idx, weight_name), WeightStorage<B>>`. Forward pass checks the
table and uses `QuantizedLinear::forward` or normal `Linear::forward`. Hash lookup overhead
is acceptable for CPU-offloaded inference where memory is the constraint anyway.

- [x] **Zero-copy expert bytes** (strategy 1, experts only): `QuantizedFusedExperts` stores
  `Arc<Mmap>` + offset + len. Expert bytes stay in OS page cache, evictable under pressure.
- [x] **`QuantizedLinear` struct**: holds `Vec<u8>` + shape + GgmlType. `forward<B>` dequantizes
  and uploads to GPU each call. Foundation for strategy 2 and the side-table approach.
- [x] **CPU offloading** (`--offload-weights`): runtime side-table `offloaded_layers` in
  `Gemma4Runtime`; attention/FFN weights stay in RAM as `QuantizedLinear`. CLI flag done.
- [x] **GPU-side quantized matmul** (strategy 3, kernels): Q8_0 and Q4_K dequantization kernels
  in `sketchpad-cubecl::quantized_matmul`. `GpuQuantizedLinear<R>` wraps packed i32 tensor in
  VRAM, dequantizes per forward. Works on WGPU/CUDA/CPU via CubeCL backend dispatch.
- [x] **Wire GPU quantized linear into Gemma 4 inference**: `load_gemma4_gguf_gpu_quant` loader
  variant that uploads weight bytes as `GpuQuantizedLinear<R>` instead of dequantizing to f16
  on load. VRAM-aware dispatch via `--vram GB` CLI flag.

#### GGUF Quantization Coverage Gaps
- [x] **Missing GgmlType variants**: Q2_K (10), Q3_K (11), IQ4_NL (20), IQ4_XS (23), IQ2_XXS (16),
  IQ2_XS (17), IQ3_XXS (18), IQ1_S (19), IQ3_S (21), IQ2_S (22), IQ1_M (29) — all added.
- [x] **CPU dequant for Q2_K**: implemented `dequantize_q2_k` (84 bytes/256 elements).
- [x] **CPU dequant for Q3_K**: implemented `dequantize_q3_k` (110 bytes/256 elements).
- [x] **CPU dequant for IQ4_NL and IQ4_XS**: lookup-table-based implementation done.
- [x] **CPU dequant for remaining IQ types**: IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ3_S, IQ2_S,
  IQ1_M all implemented via grid lookup tables ported from llama.cpp ggml-quants.c.
- [x] **GPU kernels for Q2_K, Q3_K**: CubeCL kernels in `sketchpad-cubecl::quantized_matmul`.
- [ ] **GPU kernels for IQ types**: large work, low priority (IQ types are niche).

### Shared Building Blocks (burn-models-core)

#### Layers
- [x] Transformer block (shared by DiT, LLM, encoders)
- [x] RoPE (Rotary Position Embedding)
- [x] RMSNorm
- [x] SwiGLU / GeGLU activations
- [x] Multi-head attention with GQA support
- [x] MoE (Mixture of Experts) routing
- [x] DiT components (AdaLayerNorm, DiTBlock, PatchEmbed)
- [x] Sliding window attention mask

#### Inference Optimization
- [x] KV cache for autoregressive models
- [x] Paged attention (block tables, memory management, scheduler)
- [x] Continuous batching (iteration-level scheduling, preemption)
- [x] Speculative decoding (draft/target verification, acceptance sampling)

#### Quantization
- [x] INT8 dynamic quantization (symmetric/asymmetric)
- [x] INT4 quantization (group-wise, GPTQ style)
- [x] FP8 support

#### Video-specific
- [x] 3D VAE (temporal compression)
- [x] Temporal attention layers
- [x] Frame interpolation

### Pre-Release

- [x] Architecture review - ensure consistent patterns across model implementations
- [x] API surface review - design public API for library consumers
- [x] Documentation - usage examples, model loading, inference patterns

### Future Backlog

#### Alternative Architectures (Production-Ready)
- [x] RWKV-7 "Goose" - RNN with transformer performance, linear time/constant space, no KV cache
- [x] Mamba / Mamba-2 - Selective state space models, linear scaling, 5x faster inference than transformers
- [x] Jamba - AI21's Transformer-Mamba-MoE hybrid, 1:7 attention:Mamba ratio, 256K context, KV cache for attention layers

#### Fast Image Generation
- [x] SANA - NVIDIA's linear DiT, 32x compression, 4K images on laptop GPU in <1s

#### Experimental / Research (Lower Priority)
- [x] xLSTM - sLSTM (exponential gates + normalizer stabilizer) and mLSTM (matrix memory C∈ℝ^{d_v×d_k}) cells
- [x] Zamba/Zamba2 - Mamba backbone + shared GQA attention (weight-tied across attn layers) + per-layer LoRA adapters (Zamba2)
- [x] Griffin/Hawk - Google DeepMind's gated linear recurrences (RecurrentGemma) — RG-LRU + local attention
- [x] RetNet - Microsoft's retentive network — recurrent mode, S_t = γ·S_{t-1} + k⊗v
- [x] Hyena/StripedHyena - Long convolutions + gating; HyenaFilter (sinusoidal MLP), direct_conv, optional interleaved GQA attention
- [x] TTT (Test-Time Training) - TTT-Linear and TTT-MLP; explicit gradient update of W_inner per token
- [x] LLaDA - Masked diffusion LM; bidirectional transformer + iterative unmask-by-confidence loop
- [x] TESS-2 - Simplex diffusion LM; soft embeddings + simplex interpolation steps, forward_soft()

## Issues Log

See [docs/issues-log.md](docs/issues-log.md) for detailed tracking of issues encountered and their resolution, including root cause analysis and prevention strategies.

### Current Open Issues

| Issue | Status | Workaround |
|-------|--------|------------|
| f16 produces NaN in UNet | Low priority (bf16 works) | Use `--precision bf16` (default) or `--precision f32` |
| CompVis single-file checkpoints | Backlog | Use HuggingFace diffusers format |
| SDXL inference slow | Flash attention not yet wired | Use SD 1.x for faster testing |

### SDXL Weight Loading ✅ COMPLETE (2026-01-08)

SDXL pipeline is fully working with CompVis single-file checkpoints (CivitAI format):

1. **CLIP text encoder** ✅ DONE
   - Tensor prefix auto-detected: `conditioner.embedders.0.transformer.text_model.*` (SDXL single-file)
   - Loader: `SdWeightLoader::load_clip_text_encoder()`

2. **OpenCLIP text encoder** ✅ DONE
   - Tensor prefix: `conditioner.embedders.1.model.*` (single-file) or `text_encoder_2/*` (diffusers)
   - Handles fused QKV weights (`in_proj_weight`) by splitting into Q/K/V
   - Loader: `SdWeightLoader::load_open_clip_text_encoder()`

3. **UNetXL** ✅ DONE
   - Tensor prefix: `model.diffusion_model.*` (single-file)
   - Full block mapping for CompVis naming:
     - `input_blocks.1-2` → down block 0 (res + attn)
     - `input_blocks.3.0.op` → downsample
     - Auto-detects transformer depth and upsample locations
   - `label_emb` loading for add_embed
   - Made `DownBlockXL`, `MidBlockXL`, `UpBlockXL` public
   - Loader: `SdWeightLoader::load_unet_xl()`

4. **VAE decoder** ✅ DONE (shared with SD 1.x)

5. **CLI integration** ✅ DONE
   - `run_sdxl_generate()` implemented
   - Use `--model sdxl` (default) with any SDXL checkpoint

**Performance Note**:
- SDXL with flash attention: **~0.6s/step** (bf16, 512x512)
- [x] Flash attention for SDXL UNet ✅ DONE (2026-01-08)

6. **Diffusers format support** - Backlog
   - Multi-file HuggingFace style (text_encoder/, unet/, vae/ directories)
   - Lower priority since CivitAI models are single-file CompVis format

### Upstream Dependencies to Track

| Dependency | Issue | Status | Last Checked |
|------------|-------|--------|--------------|
| cubek-attention | [PR #55](https://github.com/tracel-ai/cubek/pull/55) - f16/bf16 tile alignment | Pending review | 2026-01-08 |

When PR is merged and released to crates.io, update cubek dependency and remove f32 attention workaround.

**Current workaround**: UNet casts Q/K/V to f32 before attention, then casts back. This adds ~2x memory overhead for attention tensors (temporary) but allows f16/bf16 model weights to work.

**Performance impact**: f32 attention is ~4x slower than native f16 would be (1.2s/step vs 0.3s/step). This is significant - prioritize removing the workaround once cubek PR is merged.

## Backlog

### Memory Optimization

- **Smart offloading**: Implement CPU/GPU memory management for large models
  - Move unused model components to CPU during inference
  - Automatic based on available VRAM
  - Priority: High for SDXL on <12GB VRAM cards

## Postmortems

### Dead Code Patterns (2026-01-06)

Cleaned up `let _var = ...` patterns where values were computed but never used.
Root causes:

1. **Incomplete implementations shipped as "good enough"**: Several samplers (DPM2, DEIS,
   DPM3M, DPM Fast) computed intermediate values needed for higher-order methods but then
   fell back to simpler approximations. The computation was left in place, presumably to
   be completed later. Example: `dpm2.rs` computed `sample_mid` for midpoint method but
   comments note "This would need another model call in practice" - then uses simplified
   extrapolation instead.

2. **Placeholder implementations**: `frame_interpolation.rs` computed all four bilinear
   interpolation weights but returns input unchanged with a comment explaining that full
   implementation "requires index_select or gather operations."

3. **Refactoring leftovers**: Variables like `_refiner_steps` in pipeline.rs and
   `_noise_pred` in euler_cfg.rs were computed for logic that was later simplified or
   removed, but the computation wasn't cleaned up.

4. **Speculative code for future features**: `rwkv_loader.rs` computed `lora_dim` for
   LoRA support that was never implemented.

**Lesson**: When deferring implementation, prefer one of:
- Don't write the computation at all - add a TODO comment instead
- Write the full implementation
- If partial implementation is necessary, add `// TODO: ` explaining what's missing

Suppressing warnings with `_` prefix hides technical debt. The warning exists to help.

### Incomplete Implementations (2026-01-06)

Found during dead code audit. These are stubs or broken implementations that need attention:

#### Fixable (no backend changes needed)

- [x] `attention.rs:30-34` `causal_mask` - Returns zeros instead of proper -inf upper triangular mask.
  Would allow model to attend to future tokens. Note: `transformer.rs` has a working version.

- [x] `speculative.rs:268-271` `sample_token` - Non-greedy mode just uses argmax.
  Temperature sampling is fake (comment says "Simplified sampling").

- [x] `dpm_fast.rs:200` `DpmAdaptiveSampler::step` - "accept all steps" comment.
  Adaptive step control is unimplemented, it's just fixed-step pretending to be adaptive.

#### Fixed

- [x] `vae3d.rs:174` `Conv3d::forward` - Now implements proper 3D convolution using im2col.
  Extracts 3D patches, reshapes to columns, matrix multiply with weights.
  **Future optimization**: CubeCL kernel for better GPU performance.

#### Fixed (were marked unfixable but actually solvable)

- [x] `paged_attention.rs:284` `store_single_kv` - Now uses `slice_assign()`.
  Burn has had this since 0.16 - the "needs custom kernels" comment was stale.

- [x] `precision.rs:94-104` `to_fp16`/`to_bf16` - Removed.
  These were no-op functions. Added docs explaining precision is compile-time via backend.

#### Intentional Simplifications (not bugs)

- [x] `main.rs` CLI generate - Fully functional SD 1.x image generation.
  Loads CLIP, UNet, VAE from safetensors and runs full diffusion pipeline.

- [x] `rwkv.rs:277-290` RWKV-7 dynamic mixing - Now implements full low-rank projection.
  Auto-detects RWKV-6 vs RWKV-7 by checking if `time_maa_w1` weights are present.

### Precision and Performance (2026-01-07)

**f16 vs f32 Precision:**
- Added `--precision f16|f32` flag to CLI
- f32: ~57 sec inference, ~11GB VRAM
- f16: ~6GB VRAM, but ops run on CPU (0-1% GPU usage)
- CubeCL issue #984 confirmed f16 should be ~2x faster - our slowness is something else

**Timing (SD 1.x @ 512x512, f32, 30 steps):**
```
[timing] tokenizer: 38ms
[timing] load CLIP: 578ms
[timing] load UNet: 2.8s
[timing] load VAE: 82ms
[timing] inference: 53.4s (~1.78s/step)
[timing] total: 57s
```

Comparison: ComfyUI does SDXL @ 20 steps in ~26s (~1.3s/step). We're 37% slower per step on a smaller model.

**TODO:**
- [x] Add `--debug timing,shapes` flag for diagnostics
- [x] Debug garbled output from CompVis models (2026-01-07)
  - **ROOT CAUSE**: Level 3 dummy SpatialTransformer had RANDOM WEIGHTS
  - **FIX APPLIED**: Made `attn1`/`attn2` `Option<SpatialTransformer<B>>` in DownBlock/UpBlock
    - forward() now skips attention when None
    - Loaders set None for level 3 (both HF and CompVis)
    - Removed `create_dummy_spatial_transformer()` function
  - Remaining issues:
    1. VAE decoder lacks CompVis naming support (separate file prefixes)
    2. Needs testing with actual CompVis checkpoint
- [x] Investigate garbled image output (2026-01-07) - **FIXED**
  - Output had structure but psychedelic colors (heat-map appearance)
  - **ROOT CAUSE**: Missing `post_quant_conv` layer in VAE decoder
  - **Fix applied:**
    1. Added `post_quant_conv: Option<Conv2d<B>>` field to `Decoder` struct
    2. Load `first_stage_model.post_quant_conv` (1x1 conv, [4,4,1,1]) in decoder loader
    3. Apply `post_quant_conv` to latent before main decoder path in `forward_raw()`
  - diffusers VAE has two extra convolutions:
    - `quant_conv` (8→8 channels) - applied after encoder
    - `post_quant_conv` (4→4 channels) - applied before decoder
  - Our decoder now matches diffusers output exactly (verified via roundtrip test)
  - **Other fixes applied during investigation:**
    1. VAE `num_res_blocks` changed from 2 to 3 (matches safetensors)
    2. GroupNorm uses `var_bias` (population variance, N) to match PyTorch
    3. VAE clamping added to [-1, 1] before conversion
  - **Remaining issues:**
    1. f16 produces NaN in UNet (f32 works) - see "f16 NaN Investigation" below
- [x] Fix tokenizer non-determinism (2026-01-07)
  - **ROOT CAUSE**: HashMap iteration order is non-deterministic in Rust
  - Vocab was built by iterating `byte_encoder.values()` and `bpe_ranks.keys()`
  - Each run assigned different token IDs to the same tokens
  - **Fix**: Sort iterators before assigning vocab indices
    - `byte_chars.sort()` for byte-level tokens
    - Sort `bpe_ranks` by rank for merged tokens
  - Verified: Token IDs now match official CLIP vocab.json exactly

### f16 Performance Investigation (2026-01-07)

**Initial hypothesis (partial)**: Timestep tensors created from f32 every step.

**Actual root cause**: CubeCL JIT compilation overhead on first kernel invocation.

**Per-step timing (f16, SD 1.x @ 512x512):**
```
[step 0] 44.6s  ← JIT compilation for all kernels
[step 1] 266ms
[step 2] 336ms
[step 3] 343ms
[step 4] 586ms
```

After warmup, f16 is **~300ms/step** vs f32's ~1.77s/step - **f16 is 5-6x faster!**

**Fixes applied** (help marginally, but JIT dominates first run):
1. Precompute timestep tensors before sampling loop (pipeline.rs)
2. Cache timestep embedding frequencies in UNet struct (unet_sd.rs)
3. Added `timestep_freqs()` and `timestep_embedding_with_freqs()` for hot paths

**Status**: JIT overhead is a CubeCL limitation. No disk caching currently.
Each process startup recompiles kernels. Subsequent steps are fast.

**Workaround options**:
- Use f32 for interactive use (no JIT, consistent ~1.8s/step)
- Use f16 for batch generation (JIT cost amortized over many images)
- ✅ Enable CubeCL kernel caching via `cubecl.toml`

**CubeCL Kernel Caching (2026-01-07):**

CubeCL 0.9.0 supports persistent kernel caching. Added `cubecl.toml` config file:
```toml
[compilation]
cache = "target"  # Store compiled kernels in target/ directory
```

Cache location options:
- `"local"` - current working directory
- `"target"` - project's target directory (default when set)
- `"global"` - system config directory (~/.config/)
- `{ file = "/path/to/cache" }` - custom path

This should eliminate ~45s JIT overhead on subsequent runs. First run compiles
kernels, subsequent runs load from cache. Test with `--debug timing` to verify

**f32<->f16 conversion audit (2026-01-07):**

SD1x hot paths are clean:
- UNet: timestep freqs precomputed in struct (fixed earlier)
- DDIM sampler: uses tensor slicing from precomputed alphas_cumprod
- NoiseSchedule: all values precomputed at construction
- CLIP/VAE: no Vec<f32> in forward passes

DiT models (NOT currently used, future work):
- 12 models create `Vec<f32>` in TimestepEmbedding::forward() every step:
  flux.rs, zimage.rs, qwenimage.rs, ltx.rs, sana.rs, cogvideox.rs,
  auraflow.rs, pixart.rs, sd3.rs, wan.rs, hunyuan.rs, mochi.rs
- Fix: Add `freqs: Tensor<B, 1>` field to each, precompute at construction

### Linear Weight Convention (2026-01-07)

**RESOLVED**: Not a bug. Burn and PyTorch use different weight layouts:
- Burn: `[d_input, d_output]` (Row layout, default)
- PyTorch: `[d_output, d_input]`

Burn's `linear()` does `input.matmul(weight)` - no transpose.
PyTorch's `F.linear()` does `input @ weight.T`.

**Solution**: Transpose Linear weights when loading from PyTorch safetensors:
```rust
// In load_linear():
linear.weight = Param::from_tensor(weight.transpose());
```

Previously observed errors like `[1, 77, 768] @ [1, 3072, 768]` were from missing transpose.


### f16 NaN Investigation (2026-01-07)

**Symptom**: f16 precision produces NaN from step 0 in UNet forward pass.
```
[debug] Step 0 t=666 - noise_uncond: min=inf, max=-inf, mean=NaN
```

**Root cause**: Attention softmax overflow in f16.
- f16 max representable: ~65504
- exp(x) overflows for x > ~11
- Attention scores before softmax can exceed this when Q·K^T accumulates

**Attempted fix #1**: Manual stable softmax (max-subtraction before exp):
```rust
let attn_max = attn.clone().max_dim(3);
let attn = (attn - attn_max).exp();
let attn = attn.clone() / attn.clone().sum_dim(3);
```
Applied to: UNet CrossAttention, CLIP attention, VAE SelfAttention.
**Result**: Still NaN. Overflow likely in matmul itself, not softmax.

**Solution: Flash Attention**

Flash attention solves this by:
1. Tiled computation - never materializes full attention matrix
2. f32 accumulation - even with f16 inputs, uses f32 for intermediate sums
3. Online softmax - computes softmax incrementally without storing full scores

**Implementation exists** in `burn-models-unet/src/cubecl.rs`:
- `CrossAttentionCubeCL<R: CubeRuntime>` - uses `flash_attention()` from cubek
- Uses `AccumulatorPrecision::Strict(f32)` internally
- `convert_crossattention()` converts standard CrossAttention

**Not wired up yet** to main pipeline because:
1. CubeCL blocks use `CubeBackend<R, f32, i32, u32>` (hardcoded f32 inputs)
2. Would need to support f16 input with f32 accumulation
3. Pipeline creates standard UNet, doesn't use CubeCL variants

**To wire up flash attention**:
1. Modify CubeCL CrossAttention to accept f16 inputs
2. Add `--flash-attention` flag to CLI
3. Convert UNet attention layers to CubeCL at pipeline construction
4. Or: Make attention implementation pluggable via trait

**Workaround**: Use f32 precision (works correctly, ~5x slower than f16 would be)

### Session Notes (2026-01-06) - CubeCL Phase 4

#### Key Files Modified

**burn-models-cubecl crate:**
- `src/attention.rs` - Flash attention wrapper calling cubek::attention directly
  - `flash_attention(q, k, v, options)` - main function
  - `flash_attention_masked(q, k, v, mask, options)` - with explicit mask
  - `FlashAttentionOptions { out_dtype, causal }` - default is non-causal (diffusion)
  - `FlashAttentionOptions::causal()` - for LLMs
- `src/groupnorm_silu.rs` - Fused GroupNorm + SiLU kernel
  - `groupnorm_silu(input, weight, bias, options)` - main function
  - `groupnorm(input, weight, bias, options)` - without SiLU
  - `GroupNormSiLuOptions { num_groups, eps }` - configurable groups (default 32)
  - Two-phase: 1) compute mean/var per group, 2) normalize + affine + SiLU
- `src/pool3d.rs` - AvgPool3d/MaxPool3d kernels
- `src/lib.rs` - exports attention, pool3d, groupnorm_silu modules
- `Cargo.toml` - added `cubek = "=0.1.0-pre.1"` dependency with `attention` feature

**Tests:**
- `tests/correctness_cpu.rs` - 21 passing, 3 ignored (flash attention CPU limitation)
- `tests/correctness.rs` (WGPU) - compiles, GPU tests require hardware
- `tests/correctness_cuda.rs` - compiles, GPU tests require hardware

**Removed:**
- `burn-models-core/src/flash_attention.rs` - was dead code (exported but never used)

#### Technical Notes

**Flash Attention:**
- burn-cubecl's `flash_attention` hardcodes `causal: true`
- We bypass it by calling `cubek::attention::launch::launch_ref` directly
- CPU backend fails with assertion `unit_tile.layout.num_cols % line_size == 0`
- GPU backends (CUDA, WGPU) should work

**GroupNorm+SiLU Fusion (DONE):**
- Pattern: `silu(groupnorm(x, gamma, beta, num_groups))`
- Two-phase kernel: 1) compute mean/var per group, 2) normalize + affine + SiLU
- Avoids intermediate tensor allocation between GroupNorm and SiLU
- Uses population variance (n divisor) - 1e-2 tolerance needed vs Burn's unbiased variance

#### Test Commands

```sh
# CPU tests (all kernels)
cargo test -p burn-models-cubecl --features cpu --test correctness_cpu

# WGPU tests (requires GPU)
cargo test -p burn-models-cubecl --features wgpu --test correctness -- --ignored

# CUDA tests (requires CUDA GPU)
cargo test -p burn-models-cubecl --features cuda --test correctness_cuda -- --ignored
```
