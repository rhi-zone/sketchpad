# Issues Log

Running log of issues encountered during development and their resolution.

Each issue includes:
- **Symptom**: How it manifested
- **Root Cause**: Why it happened
- **Fix**: How we resolved it
- **Prevention**: How to avoid similar issues in the future

## 2026-01-09

### [FIXED] f16 GroupNorm Produces All NaN - Reduction Overflow

**Symptom**: SDXL generation with f16 precision produced 100% NaN output. Debug tracing showed:
- UNet inputs (latent, timestep, context): all valid (0 NaN)
- UNet output (noise_pred): 100% NaN
- First NaN appears in the very first ResBlock's GroupNorm

**Root Cause**: f16 reductions overflow when summing large number of elements.

GroupNorm computes: `mean = sum(x) / N` and `var = sum((x-mean)^2) / N`

For SDXL's first down block, input shape is [1, 320, 128, 128]. With 32 groups:
- group_size = 320 / 32 = 10
- elements_per_group = 10 * 128 * 128 = 163,840

Summing 163,840 values with average magnitude ~10 produces sum ~1.6M, but **f16 max is only ~65,504**.
The sum overflows to Inf, then Inf/N = Inf, and subsequent operations produce NaN.

**Debug Evidence**:
```
[groupnorm] input: shape=[1,320,128,128], nan=0, inf=0, min=-31.8, max=35.7
[groupnorm] mean nan=0, var nan=32, var_min=inf, var_max=-inf   # All 32 groups have NaN variance!
[groupnorm] after normalize: nan=5242880/5242880                 # 100% NaN
```

**Fix** (`crates/rhi-sketchpad-core/src/groupnorm.rs`):

Compute mean and variance in f32, then cast back to original dtype:
```rust
pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
    // ... reshape x ...

    // Workaround: compute mean and var in f32, then cast back
    use burn::tensor::DType;
    let original_dtype = x.dtype();
    let x_f32 = x.clone().cast(DType::F32);
    let mean_f32 = x_f32.clone().mean_dim(2);
    let mean_expanded_f32 = mean_f32.clone().unsqueeze::<3>();
    let diff_f32 = x_f32 - mean_expanded_f32.clone();
    let var_f32 = (diff_f32.clone() * diff_f32).mean_dim(2);

    // Cast back to original dtype
    let mean_expanded = mean_expanded_f32.cast(original_dtype);
    let var = var_f32.cast(original_dtype);

    // ... rest of normalization ...
}
```

**Also Added**: Input clamping to prevent Inf values from propagating into GroupNorm:
```rust
let x = x.clamp(-65000.0, 65000.0);
```

**Why var_bias Also Failed**: Burn's `var_bias()` internally uses reduction operations that overflow the same way. Even manual variance computation (`(diff * diff).mean_dim(2)`) fails because `mean_dim` uses sum internally.

**Prevention**:
- For f16/bf16, normalization layers (GroupNorm, LayerNorm, BatchNorm) should compute statistics in f32
- This is standard practice in mixed-precision training (PyTorch's autocast does this automatically)
- Consider adding a `cast_to_f32_for_reduction` utility for reductions over large dimensions
- burn/cubecl could potentially add automatic upcasting for large reductions

**Status**: Fixed. UNet now produces valid output with f16 precision.

---

### [WIP] f16 VAE Decoder Overflow - Conv Accumulation

**Symptom**: With the GroupNorm fix, UNet works but VAE decoder produces Inf values that spread:
```
[vae] up_block_2 has 0 NaN, 11 Inf out of 67108864 values   # Small number of Inf
[vae] up_block_3 has 14947341 NaN, 18607091 Inf out of 33554432 values  # Explosion!
```

**Root Cause**: Convolution accumulation can exceed f16 range.

VAE decoder upsamples from 64x64 → 128x128 → 256x256 → 512x512. At larger spatial sizes:
- More filter taps accumulate
- Values can exceed f16 max (~65504) → become Inf
- Inf propagates through subsequent operations

The Inf values from up_block_2 enter up_block_3's GroupNorm. Even with f32 computation in GroupNorm, computing mean of values including Inf produces Inf, which then causes problems.

**Current Workaround**: Added input clamping to GroupNorm:
```rust
let x = x.clamp(-65000.0, 65000.0);  // Clamp Inf to valid range
```

This helps but doesn't fully fix the issue. Some Inf still leak through conv operations.

**Implemented**: Added `--vae-clamp` flag that enables aggressive per-layer clamping in VAE decoder:
```bash
burn-models generate --prompt "..." --vae-clamp true
```

This clamps activations after each major operation (conv_in, mid_blocks, up_blocks, etc.) to prevent f16 overflow from propagating. Helps prevent NaN/Inf explosion.

Also added `--debug=vae` flag to enable VAE debug output for diagnosing f16 issues.

**Proper Solutions** (not yet implemented):
1. **True VAE in f32**: Load VAE weights in f32 separately from f16 UNet
2. **bf16 for VAE**: bf16 has f32's exponent range, avoids overflow
3. **Mixed-precision compute**: Store weights in f16, compute in f32 (requires burn architecture changes)

**Status**: Partially mitigated. `--vae-clamp` adds clamping that helps prevent overflow. Full fix needs true f32 VAE computation.

**Files Modified**:
- `crates/rhi-sketchpad-core/src/groupnorm.rs` - f32 computation + clamping
- `crates/rhi-sketchpad-vae/src/decoder.rs` - added `clamp_overflow` mode with per-layer clamping
- `crates/burn-models-cli/src/main.rs` - added `--vae-clamp` and `--debug=vae` flags

---

### [WIP] SDXL VRAM OOM at VAE Decode

**Symptom**: SDXL generation fails with OOM at 95% (VAE decode stage):
```
can't allocate buffer of size: 3122610176  (3.1GB)
```
This happens on a 12GB RTX 3060.

**Root Cause**: SDXL's 1024x1024 output requires large intermediate tensors during VAE decode:
- Latent: 128x128x4 → 512x512 → eventually 1024x1024x3
- Intermediate feature maps at 512/256/128 channels
- cubecl's matmul workspace allocates large f32 buffers even for f16 operations

**Attempted Fixes**:
1. **Drop models before VAE**: Drop UNet/CLIP to free VRAM before VAE decode
   - Result: Still OOM - lazy evaluation means tensors aren't released until materialized
2. **Materialize latent before dropping**: Force compute with `into_data()` before drops
   - Result: Helps but still OOM at largest allocation
3. **Skip VAE attention**: Added `--low-vram` flag to skip attention in VAE mid block
   - Result: Reduces peak but still OOM in upsample convolutions
4. **Changed Upsample from repeat_dim to interpolate**: More memory efficient
   - Result: Reduces peak but convolutions still large
5. **Tiled VAE decode**: Process latent in tiles (2x2 quadrants of 64x64)
   - Result: Reduces peak VRAM, but introduces edge artifacts and NaN issues

**Current Workaround**: `--low-vram true` enables:
- Skipping VAE mid_attn
- Tiled decode (2x2 quadrants)
- This works but with quality tradeoffs

**Proper Solutions** (not yet implemented):
1. **Tiled VAE with overlap**: Use overlapping tiles and blend - standard in ComfyUI
2. **VAE in f32 only**: Separate precision for VAE (like `--no-half-vae`)
3. **Gradient checkpointing**: Recompute rather than store intermediates
4. **Smaller batch sizes in VAE**: Process channels/spatial in chunks

**ComfyUI Approaches** (from Gemini's research):
- Tiled VAE for VRAM management
- bf16-vae or no-half-vae options
- Split attention mechanisms

**Status**: Partially working with `--low-vram`. Full fix needs proper tiled VAE implementation.

---

## 2026-01-08

### [FIXED] SDXL White Output - Multiple Causes

**Symptom**: SDXL generation produced completely white images (all pixels 255,255,255).

**Root Causes**: Three separate issues combined:

1. **Wrong CLIP layer**: SDXL always uses the **penultimate layer** (clip_skip=2) from both text encoders, not the final layer. We were using `forward()` (all 12 layers) instead of `forward_penultimate()` (stops at layer 10). This explains why some checkpoints have NaN in layer 11 but work in ComfyUI - layer 11 is never used!

2. **VAE decode wrong scaling**: `StableDiffusionXL::decode()` was calling `vae_decoder.forward()` which uses SD1.x scaling (0.18215), and then manually dividing by SDXL scaling (0.13025) - double scaling! Also returned [-1, 1] range instead of [0, 255].

3. **Pooled embedding extraction** (flash path): Manual slice extraction `slice([0..1, 76..77, 0..1280])` doesn't apply `text_projection`. Must use `forward_with_pooled()`.

**Debugging Process**:
```
[debug] cond_context: 157696 values, 59136 NaN   # 77*768 = CLIP portion all NaN
[debug] cond_pooled: 1280 values, 0 NaN          # OpenCLIP fine
```
59136 NaN = exactly the CLIP encoder's output size. Traced to layer 11 attention producing NaN, then to Q projection weights being all NaN in the safetensors file itself.

**Checkpoint Analysis**:
```python
# Many Illustrious-based checkpoints have corrupted layer 11
waiIllustriousSDXL_v150.safetensors: layer 11 Q nan=589824/589824  # ALL NaN
novaCrossXL_ilVF.safetensors: layer 11 Q nan=589824/589824         # ALL NaN
knkLuminai_v10.safetensors: layer 11 Q nan=0/589824                # Valid ✓
```

**Fixes**:

1. **Use penultimate layer** (`crates/sketchpad/src/pipeline/sdxl.rs`, `crates/burn-models-cli/src/main.rs`):
```rust
// Before (wrong - uses all 12 layers including potentially corrupted layer 11)
let clip_hidden = self.clip_encoder.forward(token_tensor.clone());

// After (correct - SDXL uses penultimate layer, clip_skip=2)
let clip_hidden = self.clip_encoder.forward_penultimate(token_tensor.clone());
```

2. **VAE decode** (`crates/sketchpad/src/pipeline/sdxl.rs`):
```rust
// Before (wrong - double scaling, wrong range)
pub fn decode(&self, latent: Tensor<B, 4>) -> Tensor<B, 4> {
    let latent = latent / 0.13025;
    self.vae_decoder.forward(latent)  // Uses SD1.x scaling internally!
}

// After (correct)
pub fn decode(&self, latent: Tensor<B, 4>) -> Tensor<B, 4> {
    self.vae_decoder.decode_to_image_sdxl(latent)  // SDXL scaling + [0,255]
}
```

2. **Pooled embedding** (`crates/burn-models-cli/src/main.rs`):
```rust
// Before (wrong - no text_projection)
let pooled_cond = open_clip_cond.slice([0..1, 76..77, 0..1280]).reshape([1, 1280]);

// After (correct - uses text_projection)
let pos_eos_pos = pos_tokens.iter().position(|&t| t == 49407).unwrap_or(76);
let (open_clip_cond, pooled_cond) =
    open_clip_encoder.forward_with_pooled(pos_tensor.unsqueeze::<2>(), &[pos_eos_pos]);
```

3. **Attention numerical stability** (`crates/rhi-sketchpad-clip/src/attention.rs`):
```rust
// Changed causal mask from -inf to -1e9 for bf16 compatibility
mask_data[i * max_seq_len + j] = -1e9;  // was f32::NEG_INFINITY

// Added epsilon to prevent division by zero in softmax
let attn_sum = attn.clone().sum_dim(3) + 1e-8;
let attn = attn / attn_sum;
```

**Files Modified**:
- `crates/sketchpad/src/pipeline/sdxl.rs` - VAE decode fix
- `crates/burn-models-cli/src/main.rs` - Pooled embedding fix
- `crates/rhi-sketchpad-clip/src/attention.rs` - Numerical stability

**Prevention**:
- Validate checkpoint weights on load - warn if NaN detected in text encoder
- Use `decode_to_image_sdxl()` not manual scaling + `forward()`
- Always use `forward_with_pooled()` for OpenCLIP pooled output
- Test SDXL with multiple checkpoints - some have corrupted weights
- Consider: add `--validate-weights` flag to check for NaN before generation

**Remaining Issue**: ~~With these fixes, SDXL produces colored but noisy output (not recognizable images). Separate issue - likely conditioning or sampler problem.~~ See next entry - was wrong noise schedule.

---

### [FIXED] SDXL Garbled/Noisy Output - Wrong Noise Schedule

**Symptom**: After fixing white output, SDXL produced colorful but completely incoherent/garbled images - noisy patterns instead of recognizable content.

**Root Cause**: `NoiseSchedule::sdxl()` was using a **cosine schedule**, but SDXL actually uses **scaled_linear** (same as SD 1.x).

The cosine schedule produces completely different alpha_cumprod values at each timestep. Since all sampling math depends on these values, the wrong schedule means:
- Wrong noise levels at each step
- Wrong denoised predictions
- Complete sampling failure

**Evidence**: Official SDXL scheduler config from HuggingFace:
```json
// https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/scheduler/scheduler_config.json
{
  "beta_schedule": "scaled_linear",   // NOT cosine!
  "beta_start": 0.00085,
  "beta_end": 0.012,
  "num_train_timesteps": 1000,
  "prediction_type": "epsilon"
}
```

**Fix** (`crates/rhi-sketchpad-samplers/src/scheduler.rs`):

1. Added `scaled_linear` schedule (different from regular `linear`):
```rust
/// scaled_linear interpolates sqrt(beta) then squares:
/// `betas = linspace(sqrt(beta_start), sqrt(beta_end), num_steps) ** 2`
pub fn scaled_linear(num_steps: usize, beta_start: f64, beta_end: f64, device: &B::Device) -> Self {
    let sqrt_beta_start = beta_start.sqrt();
    let sqrt_beta_end = beta_end.sqrt();

    let betas: Vec<f32> = (0..num_steps)
        .map(|i| {
            let t = i as f64 / (num_steps - 1) as f64;
            let sqrt_beta = sqrt_beta_start + t * (sqrt_beta_end - sqrt_beta_start);
            (sqrt_beta * sqrt_beta) as f32
        })
        .collect();
    // ... rest unchanged
}
```

2. Fixed `sdxl()` to use scaled_linear:
```rust
// Before (WRONG)
pub fn sdxl(device: &B::Device) -> Self {
    Self::cosine(&ScheduleConfig::default(), device)
}

// After (CORRECT)
pub fn sdxl(device: &B::Device) -> Self {
    Self::scaled_linear(1000, 0.00085, 0.012, device)
}
```

**linear vs scaled_linear**:
- `linear`: `betas = linspace(beta_start, beta_end, N)`
- `scaled_linear`: `betas = linspace(sqrt(beta_start), sqrt(beta_end), N) ** 2`

The squared interpolation produces slightly different noise levels, which matters for training/inference consistency.

**Prevention**:
- Always check official scheduler configs, not just assume based on model name
- "SDXL = cosine schedule" is a common misconception
- Validate by comparing alpha_cumprod values against reference at a few timesteps
- Document noise schedule source with URL in code comments

**Sources**:
- https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/scheduler/scheduler_config.json
- https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py

---

## 2026-01-07

### [FIXED] Blurry Output - Missing c_in Input Scaling

**Symptom**: Generated images were extremely blurry compared to ComfyUI/A1111 output.

**Root Cause**: k-diffusion formulation requires input variance normalization before UNet.

The UNet expects input with variance ~1.0. During diffusion, the noisy latent has variance
scaling with sigma. Without normalization, the UNet receives inputs with wrong magnitude.

k-diffusion uses: `c_in = 1 / sqrt(1 + sigma^2)` to normalize input variance.

**Fix**: Apply c_in scaling before each UNet call:
```rust
let c_in = 1.0 / (1.0 + sigma * sigma).sqrt();
let latent_scaled = latent.clone() * c_in;
let noise = unet.forward(latent_scaled, t, conditioning);
```

**Files**: `crates/sketchpad/src/pipeline/sd1x.rs`

**Prevention**:
- When implementing samplers, document the expected input/output formulation
- k-diffusion vs diffusers vs paper formulations differ - be explicit about which is used
- Test: Compare intermediate values (scaled input magnitude) against reference implementation

---

### [FIXED] Gray/Brown Backgrounds - Incorrect Denoised Computation

**Symptom**: After c_in fix, backgrounds were gray/brown (neutral latent color) instead of proper colors.

**Root Cause**: Used scaled input for denoised computation instead of unscaled.

Initial (wrong) fix attempted:
```rust
let denoised = latent_scaled - noise_pred * sigma;  // WRONG
```

This dampens denoised by c_in factor, causing low-variance areas (backgrounds) to collapse toward zero.

**Correct formulation** (from ComfyUI `model_sampling.py`):
- `c_in` scaling is ONLY for UNet input normalization
- `denoised = x - noise * sigma` where x is the **original unscaled** latent
- The sampler's `step()` function already does this correctly

**Fix**: Just use the original sampler step function with unscaled latent:
```rust
// c_in scaling is ONLY for UNet input
let latent_scaled = latent.clone() * c_in;
let noise = unet.forward(latent_scaled, t, cond);

// Sampler uses UNSCALED latent - it computes denoised internally
latent = sampler.step(latent, noise_pred, step_idx);
```

**Files**: `crates/sketchpad/src/pipeline/sd1x.rs`

**Prevention**:
- Read ComfyUI source carefully - `model_input` in `calculate_denoised` is unscaled x
- c_in is variance normalization for neural network input only
- The sampler step formula operates entirely in unscaled space
- Test: Check background colors match reference - gray = signal being dampened

---

### [FIXED] Patchy/Blurry Output with Karras Schedule

**Symptom**: With Karras sigma schedule, output looked coherent but "patchy" - slightly blurry with inconsistent texture.

**Root Cause**: Timestep-sigma mismatch with non-uniform schedules.

With the Karras schedule, sigmas are respaced using the Karras formula, but we were still
passing the original evenly-spaced timesteps to the UNet. For example:
- Karras sigma[1] = 15.01 (corresponds to training t≈910)
- We were passing t=900 (the original evenly-spaced timestep)

The UNet's timestep embedding is trained to expect a specific noise level for each timestep.
Passing the wrong timestep causes subtle conditioning errors.

**Fix**: Convert sigmas back to training timesteps using `sigma_to_timestep()`:
```rust
// For Karras/other non-uniform schedules, find the training timestep
// that corresponds to each sigma value
let timesteps = if sigma_schedule == SigmaSchedule::Normal {
    sampler.timesteps().to_vec()
} else {
    schedule.sigmas_to_timesteps(&sigmas[..sigmas.len() - 1])
};
```

The conversion uses: `alpha_cumprod = 1 / (1 + sigma^2)`, then finds the closest match.

**Files**:
- `crates/rhi-sketchpad-samplers/src/scheduler.rs` - Added `sigma_to_timestep()` method
- `crates/sketchpad/src/pipeline/sd1x.rs` - Use sigma-derived timesteps for non-Normal schedules

**Prevention**:
- In k-diffusion/ComfyUI, the model wrapper converts sigma→timestep internally
- When implementing samplers with non-uniform sigma schedules, always convert sigma to training timestep
- Test: Compare timestep sequence with ComfyUI output for same schedule

---

### [FIXED] f16 Type Mismatch in Samplers

**Symptom**: Runtime panic when using f16 precision:
```
TypeMismatch("Invalid target element type (expected F16, got F32)")
```

**Root Cause**: Tensor `.to_vec()` requires element type to match.

Code like this fails with f16 tensors:
```rust
let alpha: f32 = tensor.into_data().to_vec().unwrap()[0];
```

**Fix**: Add `.convert::<f32>()` before `.to_vec()`:
```rust
let alpha: f32 = tensor.into_data().convert::<f32>().to_vec().unwrap()[0];
```

**Files**: Multiple samplers - `scheduler.rs`, `lcm.rs`, `ddpm.rs`, `ipndm.rs`, `dpm_fast.rs`

**Prevention**:
- Always use `.convert::<f32>()` when extracting scalar values from tensors
- Test samplers with f16 backend explicitly
- Consider: wrapper function `fn scalar_f32<B: Backend>(t: Tensor<B, 0>) -> f32`

---

### [FIXED] Garbled/Psychedelic Output from VAE

**Symptom**: Generated images had structure but psychedelic/heat-map colors.

**Root Cause**: VAE decoder missing `post_quant_conv` layer.

The diffusers VAE has two extra 1x1 convolutions:
- `quant_conv` (8→8 channels) - applied after encoder
- `post_quant_conv` (4→4 channels) - applied before decoder

We were missing `post_quant_conv`, which transforms the latent before decoding.

**Fix**:
1. Added `post_quant_conv: Option<Conv2d<B>>` field to `Decoder` struct
2. Load `first_stage_model.post_quant_conv` weights in decoder loader
3. Apply in `forward_raw()` before main decoder path

**Files**: `crates/rhi-sketchpad-vae/src/decoder.rs`, `crates/rhi-sketchpad-convert/src/sd_loader.rs`

**Prevention**:
- When porting models, enumerate ALL layers in reference implementation
- Create checklist of expected weight names from safetensors before coding
- Test: VAE roundtrip test comparing output to reference (diffusers) on same input
- Could catch with: `expected_weights()` function that validates all required tensors are present

---

### [FIXED] Non-deterministic Tokenization

**Symptom**: Same prompt produced different outputs across runs.

**Root Cause**: HashMap iteration order is non-deterministic in Rust.

Vocab was built by iterating `byte_encoder.values()` and `bpe_ranks.keys()`.
Each run assigned different token IDs to the same tokens.

**Fix**: Sort iterators before assigning vocab indices:
```rust
byte_chars.sort();  // Sort byte-level tokens
pairs.sort_by_key(|&(_, rank)| rank);  // Sort BPE merges by rank
```

**Files**: `crates/rhi-sketchpad-clip/src/tokenizer.rs`

**Prevention**:
- Never iterate HashMap/HashSet when order matters
- Prefer BTreeMap/BTreeSet or explicit sorting
- Test: Run tokenizer twice on same input, assert outputs are identical
- Test: Compare token IDs against known-good reference (official vocab)
- Clippy lint: Consider adding custom lint for HashMap iteration in determinism-critical code

---

### [FIXED] Token ID Mismatch vs Official CLIP

**Symptom**: Tokens like "cute" got wrong IDs, affecting text conditioning quality.

**Root Cause**: Built vocab programmatically from BPE merges instead of using official vocab.

The programmatic approach produced different token assignments than OpenAI's original.

**Fix**: Load official CLIP `vocab.json` directly:
1. Downloaded official vocab from HuggingFace
2. Added simple JSON parser (no serde dependency)
3. Embed vocab in binary with `include_str!`

**Files**: `crates/rhi-sketchpad-clip/src/tokenizer.rs`, `crates/rhi-sketchpad-clip/data/vocab.json`

**Verification**: Token IDs now match exactly:
- "a</w>" = 320
- "cute</w>" = 2242
- "cat</w>" = 2368

**Prevention**:
- Use official assets (vocab files, configs) rather than reconstructing
- When reconstruction is necessary, validate against official output
- Test: Golden test with known prompt → known token IDs
- General: Prefer loading authoritative data over deriving it

---

### [FIXED] f16 Produces NaN in UNet

**Symptom**: f16 precision produces NaN from step 0 in UNet forward pass.
```
[debug] Step 0 t=666 - noise_uncond: min=inf, max=-inf, mean=NaN
```

**Root Cause**: Attention softmax overflow in f16.
- f16 max representable: ~65504
- exp(x) overflows for x > ~11
- Attention scores before softmax can exceed this
- More specifically: Q @ K^T matmul itself can overflow, not just softmax

**Failed Attempt**: Manual stable softmax (max-subtraction before exp).
```rust
let attn_max = attn.clone().max_dim(3);
let attn = (attn - attn_max).exp();
let attn = attn.clone() / attn.clone().sum_dim(3);
```
**Result**: Still NaN. Overflow happens in matmul, before softmax.

**Solution**: Flash attention with f32 accumulation.
- Implemented in `rhi-sketchpad-unet/src/cubecl.rs`
- `UNetCubeCL` type with `convert_unet()` function
- Uses flash attention kernel with f32 accumulation internally
- Wired to CLI with `--flash-attention` flag (enabled by default)

**Fix Details**:
1. Created `SpatialTransformerCubeCL`, `TransformerBlockCubeCL`, `CrossAttentionCubeCL`
2. Created `UNetCubeCL` that uses flash attention in all attention blocks
3. Added `run_sd1x_generate_flash` function in CLI
4. Flash attention uses `cubek-attention` kernel with f32 accumulation

**Files Modified**:
- `crates/rhi-sketchpad-unet/src/cubecl.rs` - UNet and block types with flash attention
- `crates/burn-models-cli/src/main.rs` - Flash attention generation path
- `crates/burn-models-cli/Cargo.toml` - Added cubecl feature

**Usage**:
```bash
# Flash attention enabled by default for f16
cargo run -p burn-models-cli --features cuda -- generate --prompt "..." --weights /path

# Disable if needed (will likely produce NaN with f16)
cargo run -p burn-models-cli --features cuda -- generate --flash-attention=false ...
```

**Prevention**:
- Test attention implementations with f16 specifically, not just f32
- Test with large sequence lengths where overflow is more likely
- When using half-precision, prefer flash attention or algorithms with f32 accumulation
- Add tensor stats assertions in debug builds to catch NaN/inf early
- General: f16 has ~5 orders of magnitude less range than f32 - consider this in algorithm design

---

### [KNOWN] f16 Produces NaN in UNet Operations (Not Just Attention)

**Symptom**: f16 precision with flash attention produces NaN in output. Debug shows NaN in Q/K/V tensors BEFORE flash attention is called.

**Root Cause**: f16 overflow occurs in UNet operations BEFORE attention, not just in attention softmax.

Investigation found NaN appearing in:
1. Linear projections (to_q, to_k, to_v)
2. ResBlock operations (GroupNorm+SiLU+Conv)
3. Possibly other matmul operations

Flash attention with f32 accumulation only fixes overflow in the attention softmax computation itself. It cannot fix NaN that already exists in the input tensors.

**Attempted Fixes**:
1. ✅ Padding head_dim to power-of-2 (40→64) fixes cubek tiling assertion
2. ❌ But f16 still produces NaN because other operations overflow
3. ❌ Forced blueprint with tile_size=8 produces NaN (even with f32)
4. ❌ Forced blueprint with line_size=4 doesn't fix f16 NaN

**Why f32 Works**: f32 has ~5 orders of magnitude more range than f16. Operations that overflow in f16 remain within f32's representable range.

**Workaround**: Use f32 precision:
```bash
# f32 + flash attention (recommended)
cargo run -p burn-models-cli --features cuda -- generate --model sd1x --precision f32 --flash-attention true ...

# f32 without flash attention
cargo run -p burn-models-cli --features cuda -- generate --model sd1x --precision f32 --flash-attention false ...
```

**Files**:
- `crates/rhi-sketchpad-unet/src/cubecl.rs` - CrossAttentionCubeCL with padding
- `crates/rhi-sketchpad-cubecl/src/attention.rs` - flash_attention wrapper

**Future Work - How Existing Software Solves This**:

1. **ComfyUI's "upcast attention"**: Only compute Q@K^T in f32, keep everything else in f16:
   ```python
   if attn_precision == torch.float32:
       sim = einsum('b i d, b j d -> b i j', q.float(), k.float()) * scale
   ```
   This targeted approach avoids overflow while maintaining speed.

2. **Diffusers recommends bfloat16**: bf16 has f32's exponent range (8 bits) but only
   8 bits of mantissa. It can represent the same magnitude as f32 without overflow.

3. **PyTorch autocast**: Automatically keeps LayerNorm, GroupNorm, Softmax in f32.

**Implementation Options for burn-models**:

| Approach | Effort | Status |
|----------|--------|--------|
| **bf16 support** | Low | burn-cubecl supports bf16 - just need to wire it up |
| **Upcast attention Q@K^T** | Medium | Compute similarity matrix in f32 only |
| **Upcast all Linear matmuls** | Medium | Keep nn::Linear forward in f32 |
| **Mixed precision pipeline** | High | Store weights f16, compute f32 |

**Recommended**: Try bf16 first - burn-cubecl already implements `FloatElement` and
`MatmulElement` for bf16. bf16 has f32's dynamic range so overflow is unlikely.

**Note**: SDXL uses `head_dim=64` (power-of-2) which avoids the cubek tiling issue. But SDXL may still have f16 overflow in other operations - needs testing.

---

### [KNOWN] cubek-attention BlackboxAccelerated wmma Compilation Error

**Symptom**: Using BlackboxAccelerated strategy produces CUDA compilation error:
```
error: incomplete type "nvcuda::wmma::fragment<...>" is not allowed
```

**Root Cause**: cubek-attention's BlackboxAccelerated routine uses WMMA (Tensor Core) operations that fail to compile on certain GPU/driver combinations.

This was observed on RTX 3060 (Ampere architecture, compute capability 8.6) which does support Tensor Cores, so the error is likely a cubek bug, not a hardware limitation.

**Workaround**: Use Unit strategy instead of BlackboxAccelerated (this is the default).

**Status**: Known cubek-attention issue. Unit strategy works correctly.

---

## Other Issues (Resolved)

### VAE num_res_blocks Mismatch

**Issue**: Config had `num_res_blocks=2`, safetensors had 3 blocks.

**Fix**: Changed `DecoderConfig::default()` to use 3.

**Prevention**:
- Validate model structure against reference implementation before hardcoding defaults
- Test: Load weights, check all layers have matching weight shapes

### GroupNorm Variance Calculation

**Issue**: Our GroupNorm used unbiased variance (n-1 divisor), PyTorch uses biased (n divisor).

**Fix**: Use `var_bias` for population variance.

**Prevention**:
- Document which variance formula reference implementations use
- Test: Compare layer output to PyTorch reference on same input

---

## Debug Tools

### --debug nan Flag

Use `--debug nan` to enable NaN/Inf checking at key points in the pipeline.
When a NaN or Inf value is detected, the program will panic with details:

```
[NaN check failed] step_0_noise_uncond: 1234/65536 values are NaN, 0/65536 are Inf
Stats: min=inf, max=-inf, mean=NaN, std=NaN [NaN=1234, Inf=0]
```

Checks are placed at:
- After CLIP encoding (text embeddings)
- After each UNet forward pass (noise predictions)
- After each sampling step (latent update)
- Before and after VAE decode

Usage:
```bash
cargo run -p burn-models-cli --features cuda -- generate --debug nan --weights /path/to/model ...
```

Debug modes can be combined: `--debug timing,nan,shapes` or use `--debug all`.

This is invaluable for debugging f16 overflow issues - it pinpoints exactly where NaN first appears.

---

## Testing Strategy (Derived from Issues)

These issues suggest the following testing priorities:

1. **Golden tests with reference outputs**: Compare our output to diffusers/PyTorch on identical inputs
2. **Weight completeness validation**: Assert all expected weights are loaded
3. **Determinism tests**: Run twice, outputs must be identical
4. **Precision-specific tests**: Test f16 and f32 separately
5. **Numeric stability tests**: Check for NaN/inf in intermediate tensors
