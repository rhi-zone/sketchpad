# CubeCL Conv3d Kernels

Custom GPU kernels for 3D convolution, optimized for video model inference (3D VAE, temporal layers).

## Overview

The `rhi-sketchpad-cubecl` crate provides two Conv3d kernel implementations:

1. **Simple kernel** (`conv3d`) - Direct implementation for NCTHW (channels-first) layout
2. **Optimized kernel** (`conv3d_nthwc`) - Vectorized implementation for NTHWC (channels-last) layout

Both are dramatically faster than the im2col-based reference implementation.

## When to Use Each Kernel

| Scenario | Recommended Kernel | Reason |
|----------|-------------------|--------|
| Data already in NCTHW | `conv3d` (simple) | No layout conversion needed |
| Data already in NTHWC | `conv3d_nthwc` (optimized) | Faster kernel, no conversion |
| Tiny tensors (<1K elements) | `conv3d` (simple) | Kernel launch overhead dominates |
| Large tensors, memory constrained | `conv3d` (simple) | Avoids contiguous copy |
| Large tensors, speed priority | `conv3d_nthwc` (optimized) | 1.5-2.6× faster |

## Benchmark Results

Tested on RTX 3060, CUDA 12.8, burn-cubecl 0.20.0-pre.6.

### Kernel Comparison

| Config | Shape (B,C,T,H,W) | Simple | Optimized | Speedup |
|--------|-------------------|--------|-----------|---------|
| tiny | (1,2,4,8,8) | **12.5 µs** | 24.9 µs | 0.5× |
| small | (1,4,8,32,32) | 63.8 µs | **41.3 µs** | 1.5× |
| medium | (1,8,8,64,64) | 742 µs | **286 µs** | 2.6× |
| strided | (1,8,8,64,64) s=2 | 107 µs | **97.9 µs** | 1.1× |
| deep | (1,32,4,32,32) | 1.04 ms | **866 µs** | 1.2× |

### vs im2col Reference

| Config | CubeCL Simple | im2col | Speedup |
|--------|---------------|--------|---------|
| tiny | 12.5 µs | 5.87 ms | **470×** |
| small | 63.8 µs | 2.54 s | **39,800×** |

The im2col approach is O(n³) in spatial dimensions and becomes prohibitively slow for larger tensors.

## Technical Details

### Simple Kernel (`conv3d`)

- **Layout**: NCTHW (channels-first)
- **Implementation**: Direct nested loops over kernel dimensions
- **Memory**: Works on non-contiguous tensors, no extra allocation
- **Best for**: Small tensors, memory-constrained scenarios, NCTHW pipelines

```rust
use rhi_sketchpad_cubecl::{conv3d, Conv3dOptions, Layout};

let output = conv3d::<CudaRuntime>(
    input,   // [batch, channels, time, height, width]
    weight,  // [out_ch, in_ch, k_t, k_h, k_w]
    bias,    // [out_ch] or None
    Conv3dOptions {
        stride: [1, 1, 1],
        padding: [1, 1, 1],
        dilation: [1, 1, 1],
        groups: 1,
        layout: Layout::NCTHW,
    },
)?;
```

### Optimized Kernel (`conv3d_nthwc`)

- **Layout**: NTHWC (channels-last)
- **Implementation**: Based on burn-cubecl's direct_conv2d patterns
  - `Line<E>` vectorization for coalesced memory access
  - `FastDivmod` for efficient index calculation
  - Comptime recursive kernel loop for dimension unrolling
- **Memory**: Requires channels to be contiguous (stride=1); will copy if not
- **Best for**: Large tensors where speed is priority, NTHWC pipelines

```rust
use rhi_sketchpad_cubecl::{conv3d_nthwc, Conv3dOptimizedOptions};

let output = conv3d_nthwc::<CudaRuntime>(
    input,   // [batch, time, height, width, channels]
    weight,  // [out_ch, k_t, k_h, k_w, in_ch]
    bias,    // [out_ch] or None
    Conv3dOptimizedOptions {
        stride: [1, 1, 1],
        padding: [1, 1, 1],
        dilation: [1, 1, 1],
        groups: 1,
    },
)?;
```

## Memory Considerations

### Layout Conversion Cost

If your data is in NCTHW and you want to use the optimized kernel:

```rust
// Permute creates a view (no copy)
let input_nthwc = input.permute([0, 2, 3, 4, 1]);

// BUT: kernel requires contiguous channels, so it will copy internally
// This allocates memory equal to the input tensor size
```

**Memory impact**: For a tensor of size `[1, 32, 16, 64, 64]`:
- Simple kernel: 0 extra bytes (works in-place on non-contiguous data)
- Optimized kernel: +8 MB extra (contiguous copy of input)

### Speed vs Memory Trade-off

| Priority | Kernel | Why |
|----------|--------|-----|
| **Speed** | `conv3d_nthwc` | 1.5-2.6× faster, worth the memory copy |
| **Memory** | `conv3d` | Zero extra allocation, works on any stride layout |

The benchmarks show optimized is faster even with copy overhead, but if you're memory-constrained (e.g., running near GPU VRAM limits), the simple kernel avoids allocation spikes.

### Recommendation

**Memory-constrained scenarios** (near VRAM limit):
- Use simple kernel (`conv3d`) to avoid memory spikes
- Accept the ~1.5-2.6× speed trade-off

**Speed-priority scenarios** (plenty of VRAM):
- Use optimized kernel (`conv3d_nthwc`)
- If data is NCTHW, the copy overhead is worth it for larger tensors

**Best of both worlds**:
- Store tensors in NTHWC format throughout the pipeline
- Use optimized kernel directly (no conversion, no extra memory, fastest speed)

## Running Benchmarks

```bash
# Requires CUDA
cargo bench -p rhi-sketchpad-cubecl --features cuda --bench conv3d
```

## Supported Backends

- **CUDA**: Full support, benchmarked
- **WGPU**: Supported (not benchmarked)
- **CPU**: Supported via burn-cpu (Linux only, uses CubeCL MLIR backend)

## Test Coverage

- 11 CPU tests (NCTHW + NTHWC layouts, optimized kernel)
- 8 CUDA tests (NCTHW + NTHWC layouts)
- 6 WGPU tests
