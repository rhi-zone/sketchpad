# CubeCL Kernel Development Guide

A living document for writing custom GPU kernels with CubeCL for Burn.

## Overview

CubeCL is Burn's multi-platform compute language that compiles to CUDA, ROCm, WebGPU, Metal, Vulkan, and even CPU with SIMD. Write GPU kernels in Rust syntax, get optimized code for any target.

### Supported Platforms

| Platform | Runtime | Compiler | Hardware |
|----------|---------|----------|----------|
| WebGPU | wgpu | WGSL | Most GPUs |
| CUDA | CUDA | C++ | NVIDIA GPUs |
| ROCm | HIP | C++ | AMD GPUs |
| Metal | wgpu | C++ | Apple GPUs |
| Vulkan | wgpu | SPIR-V | Linux/Windows |
| CPU | cpu | Rust | All CPUs + SIMD |

### The Cube Topology

CubeCL organizes compute using a hierarchical cuboid model:
- **Units**: Individual compute threads
- **Cubes**: Groups of units (analogous to CUDA blocks)
- **Hypercubes**: Groups of cubes (analogous to grids)

This abstraction maps cleanly across CUDA, WebGPU, and Metal.

### Three Pillars

**Automatic Vectorization**: The `Line<T>` type enables SIMD. Specify vector widths at launch time; the compiler generates appropriate SIMD code without manual unrolling.

**Comptime**: Execute calculations during kernel compilation using `#[comptime]` and `comptime![]`. Results are injected as constants, enabling instruction specialization, loop unrolling, and shape-specific optimization.

**Autotuning**: The runtime benchmarks kernel variants and caches optimal configurations per device.

## Core Concepts

### The `#[cube]` Macro

Every kernel function needs the `cube` attribute:

```rust
use cubecl::prelude::*;

// Simple helper function (can be called from kernels)
#[cube]
fn my_helper<E: Numeric>(x: E) -> E {
    x * x
}

// Launchable kernel (entry point)
#[cube(launch_unchecked)]
fn my_kernel<E: Numeric>(
    input: &Tensor<Line<E>>,
    output: &mut Tensor<Line<E>>,
) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = my_helper(input[ABSOLUTE_POS]);
    }
}
```

### Key Types

| Type | Description |
|------|-------------|
| `Tensor<Line<E>>` | GPU tensor with vectorized elements |
| `Line<E>` | SIMD vector of elements (auto-vectorized) |
| `Sequence<T>` | Compile-time fixed-size array |
| `CubeOption<T>` | Optional value (like `Option` but for kernels) |

### Position Variables

| Variable | Meaning |
|----------|---------|
| `ABSOLUTE_POS` | Global thread index across all cubes |
| `UNIT_POS` / `UNIT_POS_X` | Thread index within cube (block) |
| `CUBE_POS` / `CUBE_POS_X` | Cube (block) index |
| `CUBE_DIM` | Cube (block) dimensions |

### Compile-Time Computation

Use `#[comptime]` on parameters and `comptime![]` for expressions:

```rust
#[cube(launch)]
fn sum_kernel<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
    #[comptime] use_fast_path: bool,
    #[comptime] loop_count: Option<u32>,
) {
    // Branches are eliminated at compile-time - different kernels generated
    if use_fast_path {
        output[UNIT_POS] = plane_sum(input[UNIT_POS]);
    } else {
        // Unroll loops with comptime-known bounds
        #[unroll(loop_count)]
        for i in 0..loop_count.unwrap_or(8) {
            // ...
        }
    }
}
```

## Tutorial Examples (from CubeCL Book)

### Simple Reduction

```rust
#[cube(launch_unchecked)]
fn reduce_matrix<F: Float>(input: &Tensor<F>, output: &mut Tensor<F>) {
    // Each thread processes one row
    let row = UNIT_POS_X;
    let mut acc = F::new(0.0f32);

    for col in 0..input.shape(1) {
        acc += input[row * input.stride(0) + col];
    }
    output[row] = acc;
}
```

### Vectorized Reduction with Line<F>

```rust
const LINE_SIZE: u32 = 4;  // Process 4 elements per operation

#[cube(launch_unchecked)]
fn reduce_matrix<F: Float>(input: &Tensor<Line<F>>, output: &mut Tensor<Line<F>>) {
    let mut acc = Line::new(F::new(0.0f32));

    // Divide iterations by LINE_SIZE since each access gets 4 elements
    for i in 0..input.shape(1) / LINE_SIZE {
        acc = acc + input[UNIT_POS_X * input.stride(0) + i];
    }
    output[UNIT_POS_X] = acc;
}
```

### 3D Parallel Reduction (multi-level parallelism)

```rust
#[cube(launch_unchecked)]
fn reduce_3d<F: Float>(input: &Tensor<Line<F>>, output: &mut Tensor<Line<F>>) {
    let mut acc = Line::new(F::new(0.0f32));

    // CUBE_POS_X: which cube (outer parallelism)
    // UNIT_POS_X: which thread in cube (inner parallelism)
    for i in 0..input.shape(2) / LINE_SIZE {
        acc = acc + input[CUBE_POS_X * input.stride(0) + UNIT_POS_X * input.stride(1) + i];
    }
    output[CUBE_POS_X * output.stride(0) + UNIT_POS_X] = acc;
}

// Launch with:
// CubeCount::Static(num_rows, 1, 1)     - outer parallelism (cubes)
// CubeDim::new(cols_per_row, 1, 1)      - inner parallelism (threads per cube)
```

## Kernel Structure (from Burn's conv2d)

### 1. Define Launch Arguments

```rust
#[derive(CubeLaunch, CubeType, Clone)]
pub struct ConvParam {
    pub stride: u32,
    pub dilation: u32,
    pub padding: i32,
}

#[derive(CubeLaunch, CubeType)]
struct Conv2dArgs {
    conv_params: Sequence<ConvParam>,
    channels_per_group: u32,
}
```

### 2. Write the Kernel

```rust
#[cube(launch_unchecked)]
fn direct_conv2d_kernel<E: Numeric>(
    input: &Tensor<Line<E>>,
    weight: &Tensor<Line<E>>,
    bias: CubeOption<Tensor<Line<E>>>,
    output: &mut Tensor<Line<E>>,
    args: Conv2dArgs,
    #[comptime] has_padding: bool,
) {
    // Bounds check - terminate early if out of range
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    // Compute output position from linear index
    let pos = ABSOLUTE_POS;

    // ... convolution logic ...

    output[pos] = sum;
}
```

### 3. Launch Configuration

```rust
pub fn conv_direct<R: CubeRuntime>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
) -> CubeTensor<R> {
    // Allocate output
    let output = empty_device(...);

    // Compute launch parameters
    let cube_dim = CubeDim::new(&input.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&input.client, working_units, cube_dim);

    // Launch!
    unsafe {
        my_kernel::launch_unchecked(
            &input.client,
            cube_count,
            cube_dim,
            input.as_tensor_arg(line_size),
            output.as_tensor_arg(line_size),
        )?;
    }

    Ok(output)
}
```

## Conv2d Implementation Patterns

Burn's direct conv2d kernel uses these patterns:

### Recursive Spatial Loops

Instead of nested for-loops, use recursive compile-time unrolled functions:

```rust
#[cube]
fn kernel_loop<E: Numeric>(
    input: &Tensor<Line<E>>,
    weight: &Tensor<Line<E>>,
    sum: &mut Line<E>,
    params: &LoopParams,
    #[comptime] dim: u32,  // Which spatial dimension
) {
    if comptime![dim < num_spatial_dims] {
        // Loop over kernel positions in this dimension
        for pos in 0..kernel_size[dim] {
            // Recurse to next dimension
            kernel_loop(input, weight, sum, params, comptime![dim + 1]);
        }
    } else {
        // Base case: accumulate over channels
        inner_product(input, weight, sum);
    }
}
```

### FastDivmod for Index Calculation

Convert linear index to N-dimensional position efficiently:

```rust
#[cube]
fn div_mod_seq(pos: u32, shape: &Sequence<FastDivmod>) -> (u32, Sequence<u32>) {
    let mut offs = pos;
    let mut out = Sequence::new();

    #[unroll]
    for i in 0..shape.len() {
        let (rem, local) = shape.index(i).div_mod(offs);
        out.push(local);
        offs = rem;
    }

    (offs, out.rev())
}
```

## Writing Conv3d Kernel

Based on the Conv2d pattern, a Conv3d kernel would:

1. **Arguments**: Same structure but with 3 spatial dimensions
2. **Kernel loop**: Recurse over T, H, W dimensions
3. **Index calculation**: 5D position from linear index
4. **Accumulation**: Inner product over input channels

### Pseudo-structure

```rust
#[cube(launch_unchecked)]
fn conv3d_kernel<E: Numeric>(
    input: &Tensor<Line<E>>,      // [B, T, H, W, C] NHWC-style
    weight: &Tensor<Line<E>>,     // [O, T, H, W, C]
    bias: CubeOption<Tensor<Line<E>>>,
    output: &mut Tensor<Line<E>>, // [B, T', H', W', O]
    args: Conv3dArgs,
    #[comptime] has_padding: bool,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    // Decompose linear position to [b, t, h, w, c]
    let (b, t, h, w, out_c) = decompose_position(ABSOLUTE_POS);

    // Initialize sum with bias
    let mut sum = get_bias_or_zero(bias, out_c);

    // Triple nested loop over kernel
    for kt in 0..kernel_t {
        for kh in 0..kernel_h {
            for kw in 0..kernel_w {
                // Check bounds if padding
                if in_bounds(t, h, w, kt, kh, kw) {
                    // Accumulate over input channels
                    for ic in 0..in_channels {
                        sum += input[...] * weight[...];
                    }
                }
            }
        }
    }

    output[ABSOLUTE_POS] = sum;
}
```

## Integration with Burn

To use a CubeCL kernel in Burn:

1. **Define in burn-cubecl crate** (or your own crate with cubecl dependency)
2. **Create wrapper function** that takes `CubeTensor<R>` and returns `CubeTensor<R>`
3. **Register as backend operation** via the ops traits

## Performance Tips

1. **Vectorization**: Use `Line<E>` for SIMD (2, 4, 8 elements per thread)
2. **Coalescing**: Access memory in contiguous patterns
3. **Shared memory**: Use `SharedMemory<E>` for tile-based algorithms
4. **Compile-time**: Use `#[comptime]` to eliminate branches
5. **Unroll**: Use `#[unroll]` for small fixed loops

## Current Status for Conv3d

### Benchmark Results (RTX 3060, CUDA 12.8, 2025-01-06)

CubeCL direct kernel vs im2col+matmul reference implementation:

| Config | CubeCL | im2col | Speedup |
|--------|--------|--------|---------|
| small (4→8ch, 8×32×32) | **63 µs** | 2.58 s | **40,900×** |
| medium (8→16ch, 8×64×64) | **682 µs** | 14.7 s | **21,500×** |
| strided (8→16ch, stride 2) | **101 µs** | 765 ms | **7,600×** |
| deep (32→64ch, 4×32×32) | **950 µs** | 598 ms | **630×** |

**Why im2col is so slow:** Each slice operation in the im2col loop is a separate kernel launch. For a 8×64×64 output, that's 32,768 kernel launches vs 1 for CubeCL.

**Conclusion:** CubeCL kernel is **orders of magnitude faster** - the im2col implementation should be replaced.

Run benchmarks: `cargo bench -p rhi-sketchpad-cubecl --features cuda --bench conv3d`

### Architecture

Our CubeCL Conv3d kernel:
- Single kernel launch (fused im2col + accumulation)
- Direct convolution pattern (no implicit GEMM)
- NCTHW layout (matches Burn's tensor layout)
- ~200 lines, adapted from burn-cubecl's conv_transpose3d

## Readiness Assessment

### What We Have

1. **CubeCL fundamentals** - cube macro, position vars, Line<E>, comptime ✓
2. **Tutorial examples** - reduction patterns, vectorization, benchmarking ✓
3. **Conv2d patterns** - recursive kernel_loop, FastDivmod, ConvParam ✓
4. **Conv3d pseudo-code** - basic structure outlined ✓
5. **Working Conv3d** - im2col reference implementation to test against ✓

### What's Missing

1. **LinearView** - Used in conv2d for output, not documented
   ```rust
   // In conv2d direct.rs
   output: &mut LinearView<Line<E>, ReadWrite>
   ```

2. **Integration path** - Conv kernels live in burn-cubecl itself
   - Options: contribute upstream OR create rhi-sketchpad-cubecl crate
   - Need to understand backend trait registration

3. **cubek crate** - High-level GEMM-based convolution components
   - `cubek::convolution::components::ConvSetupError`
   - Used for ImplicitGemm strategy (tensor core optimized)
   - We probably don't need this for a simple direct kernel

4. **Layout handling** - Burn uses NHWC internally
   ```rust
   // base.rs pattern
   let input = permute_nchw_to_nhwc(input);
   let out = conv_forward_nhwc(...)?;
   Ok(permute_nhwc_to_nchw(out))
   ```

5. **Strategy pattern** - How to organize multiple implementations
   ```rust
   pub enum ConvStrategy {
       Direct,          // Simple, works everywhere
       Autotune,        // Benchmark and choose
       ImplicitGemm,    // Tensor cores (CMMA)
   }
   ```

### Reference Implementations in burn-cubecl

| Kernel | Complexity | Vectorized | Uses cubek | Notes |
|--------|------------|------------|------------|-------|
| conv_transpose3d | Simple | No | No | Good starting template |
| conv2d direct | Medium | Yes (Line<E>) | No | Recursive kernel_loop |
| conv2d ImplicitGemm | Complex | Yes | Yes | Tensor core path |

### Recommended Path for Conv3d

**Phase 1: Simple direct kernel (like conv_transpose3d)**
```rust
#[cube(launch)]
fn conv3d_kernel<E: Numeric>(
    input: &Tensor<E>,
    weight: &Tensor<E>,
    bias: &Tensor<E>,
    output: &mut Tensor<E>,
    args: Conv3dArgs,
) {
    // ABSOLUTE_POS for each output element
    // Nested loops over kernel dimensions
    // No vectorization yet
}
```
- Simplest to implement and debug
- Compare against our im2col implementation
- ~200 lines, mostly adapting conv_transpose3d

**Phase 2: Optimized direct kernel (like conv2d)**
- Add Line<E> vectorization
- Recursive kernel_loop with comptime dimension
- FastDivmod for index calculation
- More complex but still no cubek dependency

**Phase 3: ImplicitGemm (optional)**
- Requires cubek understanding
- Hardware-specific (needs CMMA)
- Biggest perf gains but most complex

### Integration Decision: rhi-sketchpad-cubecl crate

**Chosen: Option B** - Create `rhi-sketchpad-cubecl` crate

Rationale:
- Full control over iteration speed
- Avoid "vibe coded" concerns with upstream contributions
- Can always upstream later once battle-tested
- Clear separation of concerns

Trade-offs accepted:
- Maintenance burden (we own it)
- Some duplication of patterns from burn-cubecl

### Crate Structure

```
crates/rhi-sketchpad-cubecl/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── conv3d.rs          # Conv3d kernel
│   ├── utils.rs           # Shared utilities (permute, etc.)
│   └── bench.rs           # Benchmark infrastructure
└── benches/
    └── conv3d.rs          # Criterion benchmarks
```

**Cargo.toml skeleton:**
```toml
[package]
name = "rhi-sketchpad-cubecl"
version = "0.1.0"
edition.workspace = true

[features]
default = ["wgpu"]
wgpu = ["cubecl/wgpu"]
cuda = ["cubecl/cuda"]

[dependencies]
cubecl = "0.5"  # Match burn's version
burn = { workspace = true }

[dev-dependencies]
criterion = "0.5"
rhi-sketchpad-core = { workspace = true }  # For im2col reference

[[bench]]
name = "conv3d"
harness = false
```

### Queued Work (see TODO.md)

**Phase 1: Infrastructure**
- Create crate with cubecl dependency
- Feature flags (wgpu, cuda)
- Benchmark harness

**Phase 2: Conv3d Kernel**
- Port conv_transpose3d → conv3d
- NHWC layout handling
- Correctness tests vs im2col
- Performance benchmarks

**Phase 3: Optimization** (if justified)
- Line<E> vectorization
- Recursive kernel_loop
- FastDivmod

## Configuration

CubeCL uses `cubecl.toml` for configuration. Create in project root:

```toml
[compilation]
log_level = "basic"  # disabled | basic | full

[autotune]
level = "balanced"   # minimal | balanced | extensive | full
```

Environment variable overrides:
- `CUBECL_DEBUG_LOG`: stdout, stderr, or file path
- `CUBECL_DEBUG_OPTION`: "debug" or "debug-full"
- `CUBECL_AUTOTUNE_LEVEL`: minimal | balanced | extensive | full

## Benchmarking

Use the `Benchmark` trait to properly measure GPU kernel performance:

```rust
use cubecl::benchmark::{Benchmark, TimingMethod};

pub struct ConvBench<R: Runtime, F: Float + CubeElement> {
    input_shape: Vec<usize>,
    client: ComputeClient<R::Server>,
    _f: PhantomData<F>,
}

impl<R: Runtime, F: Float + CubeElement> Benchmark for ConvBench<R, F> {
    type Input = GpuTensor<R, F>;
    type Output = GpuTensor<R, F>;

    fn prepare(&self) -> Self::Input {
        // Create input data (not timed)
        GpuTensor::<R, F>::random(self.input_shape.clone(), &self.client)
    }

    fn execute(&self, input: Self::Input) -> Self::Output {
        // Run kernel (timed)
        my_kernel_launch(input)
    }

    fn sync(&self) {
        // CRITICAL: GPU is async, must sync before measuring
        future::block_on(self.client.sync())
    }
}

// Run benchmark
let bench = ConvBench { ... };
println!("{}", bench.run(TimingMethod::System));
```

## Resources

- [CubeCL Book](https://burn.dev/books/cubecl/) - Official documentation
- [CubeCL GitHub](https://github.com/tracel-ai/cubecl)
- [Burn cubecl crate](https://github.com/tracel-ai/burn/tree/main/crates/burn-cubecl)
- [Burn's SOTA matmul blog](https://burn.dev/blog/sota-multiplatform-matmul/)
