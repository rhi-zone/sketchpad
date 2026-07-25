//! Benchmark UNet blocks: CubeCL vs standard implementations
//!
//! Run with:
//!   cargo bench -p burn-models-cubecl --features cuda --bench unet_blocks
//!
//! For WGPU:
//!   cargo bench -p burn-models-cubecl --features wgpu --bench unet_blocks

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

#[cfg(feature = "cuda")]
mod cuda_bench {
    use super::*;
    use burn::prelude::*;
    use burn_cubecl::CubeBackend;
    use burn_cuda::CudaDevice;
    use cubecl::cuda::CudaRuntime;

    type BenchBackend = CubeBackend<CudaRuntime, f32, i32, u32>;

    /// Benchmark ResBlock vs ResBlockCubeCL
    pub fn bench_resblock(c: &mut Criterion) {
        use sketchpad_unet::blocks::ResBlock;
        use sketchpad_unet::cubecl::ResBlockCubeCL;

        let device = CudaDevice::default();

        let configs = [
            ("sd1x_down", 320, 320, 1280),  // SD 1.x first down block
            ("sd1x_mid", 1280, 1280, 1280), // SD 1.x mid block
            ("sdxl_down", 640, 640, 2816),  // SDXL down block
        ];

        let mut group = c.benchmark_group("ResBlock_CUDA");
        group.sample_size(20);

        for (name, in_ch, out_ch, time_dim) in configs {
            let batch = 1;
            let height = 64;
            let width = 64;

            // Standard ResBlock
            let std_block = ResBlock::<BenchBackend>::new(in_ch, out_ch, time_dim, &device);
            let input_std = Tensor::<BenchBackend, 4>::random(
                [batch, in_ch, height, width],
                burn::tensor::Distribution::Uniform(-1.0, 1.0),
                &device,
            );
            let time_emb_std = Tensor::<BenchBackend, 2>::random(
                [batch, time_dim],
                burn::tensor::Distribution::Uniform(-1.0, 1.0),
                &device,
            );

            group.bench_with_input(
                BenchmarkId::new("Standard", name),
                &(in_ch, out_ch, time_dim),
                |b, _| {
                    b.iter(|| {
                        let out = std_block.forward(
                            black_box(input_std.clone()),
                            black_box(time_emb_std.clone()),
                        );
                        // Force sync
                        let _ = out.into_data();
                    });
                },
            );

            // CubeCL ResBlock
            let cubecl_block = ResBlockCubeCL::<CudaRuntime>::new(in_ch, out_ch, time_dim, &device);
            let input_cubecl = Tensor::<BenchBackend, 4>::random(
                [batch, in_ch, height, width],
                burn::tensor::Distribution::Uniform(-1.0, 1.0),
                &device,
            );
            let time_emb_cubecl = Tensor::<BenchBackend, 2>::random(
                [batch, time_dim],
                burn::tensor::Distribution::Uniform(-1.0, 1.0),
                &device,
            );

            group.bench_with_input(
                BenchmarkId::new("CubeCL", name),
                &(in_ch, out_ch, time_dim),
                |b, _| {
                    b.iter(|| {
                        let out = cubecl_block.forward(
                            black_box(input_cubecl.clone()),
                            black_box(time_emb_cubecl.clone()),
                        );
                        // Force sync
                        let _ = out.into_data();
                    });
                },
            );
        }

        group.finish();
    }

    /// Benchmark CrossAttention vs CrossAttentionCubeCL
    pub fn bench_crossattention(c: &mut Criterion) {
        use sketchpad_unet::blocks::CrossAttention;
        use sketchpad_unet::cubecl::CrossAttentionCubeCL;

        let device = CudaDevice::default();

        let configs = [
            ("sd1x_self", 320, 8, 40, None),         // SD 1.x self-attention
            ("sd1x_cross", 320, 8, 40, Some(768)),   // SD 1.x cross-attention (CLIP)
            ("sdxl_cross", 640, 10, 64, Some(2048)), // SDXL cross-attention (OpenCLIP)
        ];

        let mut group = c.benchmark_group("CrossAttention_CUDA");
        group.sample_size(20);

        for (name, dim, heads, head_dim, ctx_dim) in configs {
            let batch = 1;
            let seq_len = 64 * 64; // 64x64 latent
            let ctx_len = 77; // CLIP tokens

            // Standard CrossAttention
            let std_attn =
                CrossAttention::<BenchBackend>::new(dim, heads, head_dim, ctx_dim, &device);
            let query_std = Tensor::<BenchBackend, 3>::random(
                [batch, seq_len, dim],
                burn::tensor::Distribution::Uniform(-1.0, 1.0),
                &device,
            );
            let context_std = ctx_dim.map(|cd| {
                Tensor::<BenchBackend, 3>::random(
                    [batch, ctx_len, cd],
                    burn::tensor::Distribution::Uniform(-1.0, 1.0),
                    &device,
                )
            });

            group.bench_with_input(
                BenchmarkId::new("Standard", name),
                &(dim, heads, head_dim),
                |b, _| {
                    b.iter(|| {
                        let out = std_attn
                            .forward(black_box(query_std.clone()), black_box(context_std.clone()));
                        let _ = out.into_data();
                    });
                },
            );

            // CubeCL CrossAttention
            let cubecl_attn =
                CrossAttentionCubeCL::<CudaRuntime>::new(dim, heads, head_dim, ctx_dim, &device);
            let query_cubecl = Tensor::<BenchBackend, 3>::random(
                [batch, seq_len, dim],
                burn::tensor::Distribution::Uniform(-1.0, 1.0),
                &device,
            );
            let context_cubecl = ctx_dim.map(|cd| {
                Tensor::<BenchBackend, 3>::random(
                    [batch, ctx_len, cd],
                    burn::tensor::Distribution::Uniform(-1.0, 1.0),
                    &device,
                )
            });

            group.bench_with_input(
                BenchmarkId::new("CubeCL", name),
                &(dim, heads, head_dim),
                |b, _| {
                    b.iter(|| {
                        let out = cubecl_attn.forward(
                            black_box(query_cubecl.clone()),
                            black_box(context_cubecl.clone()),
                        );
                        let _ = out.into_data();
                    });
                },
            );
        }

        group.finish();
    }

    criterion_group!(benches, bench_resblock, bench_crossattention);
}

#[cfg(feature = "wgpu")]
mod wgpu_bench {
    use super::*;
    use burn::prelude::*;
    use burn_cubecl::CubeBackend;
    use burn_wgpu::{WgpuDevice, WgpuRuntime};

    #[allow(dead_code)]
    type BenchBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;

    /// Benchmark ResBlock vs ResBlockCubeCL
    #[allow(dead_code)]
    pub fn bench_resblock(c: &mut Criterion) {
        use sketchpad_unet::blocks::ResBlock;
        use sketchpad_unet::cubecl::ResBlockCubeCL;

        let device = WgpuDevice::default();

        // Smaller configs for WGPU (typically slower)
        let configs = [("small", 128, 128, 512), ("medium", 256, 256, 1024)];

        let mut group = c.benchmark_group("ResBlock_WGPU");
        group.sample_size(10);

        for (name, in_ch, out_ch, time_dim) in configs {
            let batch = 1;
            let height = 32;
            let width = 32;

            // Standard ResBlock
            let std_block = ResBlock::<BenchBackend>::new(in_ch, out_ch, time_dim, &device);
            let input_std = Tensor::<BenchBackend, 4>::random(
                [batch, in_ch, height, width],
                burn::tensor::Distribution::Uniform(-1.0, 1.0),
                &device,
            );
            let time_emb_std = Tensor::<BenchBackend, 2>::random(
                [batch, time_dim],
                burn::tensor::Distribution::Uniform(-1.0, 1.0),
                &device,
            );

            group.bench_with_input(
                BenchmarkId::new("Standard", name),
                &(in_ch, out_ch, time_dim),
                |b, _| {
                    b.iter(|| {
                        let out = std_block.forward(
                            black_box(input_std.clone()),
                            black_box(time_emb_std.clone()),
                        );
                        let _ = out.into_data();
                    });
                },
            );

            // CubeCL ResBlock
            let cubecl_block = ResBlockCubeCL::<WgpuRuntime>::new(in_ch, out_ch, time_dim, &device);
            let input_cubecl = Tensor::<BenchBackend, 4>::random(
                [batch, in_ch, height, width],
                burn::tensor::Distribution::Uniform(-1.0, 1.0),
                &device,
            );
            let time_emb_cubecl = Tensor::<BenchBackend, 2>::random(
                [batch, time_dim],
                burn::tensor::Distribution::Uniform(-1.0, 1.0),
                &device,
            );

            group.bench_with_input(
                BenchmarkId::new("CubeCL", name),
                &(in_ch, out_ch, time_dim),
                |b, _| {
                    b.iter(|| {
                        let out = cubecl_block.forward(
                            black_box(input_cubecl.clone()),
                            black_box(time_emb_cubecl.clone()),
                        );
                        let _ = out.into_data();
                    });
                },
            );
        }

        group.finish();
    }

    criterion_group!(benches, bench_resblock);
}

#[cfg(feature = "cuda")]
criterion_main!(cuda_bench::benches);

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
criterion_main!(wgpu_bench::benches);

#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
fn main() {
    eprintln!("This benchmark requires either 'cuda' or 'wgpu' feature");
}
