//! Benchmark Conv3d: CubeCL kernel vs im2col reference
//!
//! Run with:
//!   CUDA_PATH=/path/to/cuda cargo bench -p burn-models-cubecl --features cuda --bench conv3d

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

#[cfg(feature = "cuda")]
mod cuda_bench {
    use super::*;
    use burn::prelude::*;
    use burn_cubecl::{CubeBackend, tensor::CubeTensor};
    use burn_cuda::CudaDevice;
    use cubecl::cuda::CudaRuntime;
    use sketchpad_cubecl::{Conv3dOptimizedOptions, Conv3dOptions, Layout, conv3d, conv3d_nthwc};

    type BenchBackend = CubeBackend<CudaRuntime, f32, i32, u8>;

    /// Reference im2col convolution (same as in correctness tests)
    mod reference {
        use burn::prelude::*;

        fn pad_5d<B: Backend>(x: Tensor<B, 5>, padding: [usize; 3]) -> Tensor<B, 5> {
            let [batch, channels, time, height, width] = x.dims();
            let [p_t, p_h, p_w] = padding;
            let device = x.device();

            let new_t = time + 2 * p_t;
            let new_h = height + 2 * p_h;
            let new_w = width + 2 * p_w;

            let mut padded = Tensor::zeros([batch, channels, new_t, new_h, new_w], &device);
            padded = padded.slice_assign(
                [
                    0..batch,
                    0..channels,
                    p_t..p_t + time,
                    p_h..p_h + height,
                    p_w..p_w + width,
                ],
                x,
            );
            padded
        }

        fn im2col_3d<B: Backend>(
            x: Tensor<B, 5>,
            kernel_size: [usize; 3],
            stride: [usize; 3],
            out_size: [usize; 3],
        ) -> Tensor<B, 3> {
            let [batch, in_ch, _, _, _] = x.dims();
            let [k_t, k_h, k_w] = kernel_size;
            let [s_t, s_h, s_w] = stride;
            let [out_t, out_h, out_w] = out_size;

            let kernel_elements = k_t * k_h * k_w;
            let out_positions = out_t * out_h * out_w;
            let col_size = in_ch * kernel_elements;

            let device = x.device();
            let mut patches = Vec::with_capacity(out_positions);

            for ot in 0..out_t {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let t_start = ot * s_t;
                        let h_start = oh * s_h;
                        let w_start = ow * s_w;

                        let patch = x.clone().slice([
                            0..batch,
                            0..in_ch,
                            t_start..t_start + k_t,
                            h_start..h_start + k_h,
                            w_start..w_start + k_w,
                        ]);

                        let patch_flat = patch.reshape([batch, col_size]);
                        let patch_col = patch_flat.unsqueeze_dim::<3>(2);
                        patches.push(patch_col);
                    }
                }
            }

            if patches.is_empty() {
                Tensor::zeros([batch, col_size, 0], &device)
            } else {
                Tensor::cat(patches, 2)
            }
        }

        pub fn conv3d_im2col<B: Backend>(
            input: Tensor<B, 5>,
            weight: Tensor<B, 5>,
            bias: Option<Tensor<B, 1>>,
            stride: [usize; 3],
            padding: [usize; 3],
        ) -> Tensor<B, 5> {
            let [batch, _in_ch, _time, _height, _width] = input.dims();
            let [out_ch, in_ch_per_group, k_t, k_h, k_w] = weight.dims();

            let kernel_elements = k_t * k_h * k_w;
            let weight_2d = weight.reshape([out_ch, in_ch_per_group * kernel_elements]);

            let x_padded = if padding[0] > 0 || padding[1] > 0 || padding[2] > 0 {
                pad_5d(input, padding)
            } else {
                input
            };
            let [_, _, pad_t, pad_h, pad_w] = x_padded.dims();

            let out_t = (pad_t - k_t) / stride[0] + 1;
            let out_h = (pad_h - k_h) / stride[1] + 1;
            let out_w = (pad_w - k_w) / stride[2] + 1;

            let cols = im2col_3d(x_padded, [k_t, k_h, k_w], stride, [out_t, out_h, out_w]);

            let weight_expanded = weight_2d.unsqueeze_dim::<3>(0).expand([
                batch,
                out_ch,
                in_ch_per_group * kernel_elements,
            ]);

            let out = weight_expanded.matmul(cols);

            let out = if let Some(bias) = bias {
                let bias_expanded = bias.reshape([1, out_ch, 1]);
                out + bias_expanded
            } else {
                out
            };

            out.reshape([batch, out_ch, out_t, out_h, out_w])
        }
    }

    fn to_cube_tensor<const D: usize>(tensor: Tensor<BenchBackend, D>) -> CubeTensor<CudaRuntime> {
        match tensor.into_primitive() {
            burn::tensor::TensorPrimitive::Float(t) => t,
            _ => panic!("Expected float tensor"),
        }
    }

    fn from_cube_tensor<const D: usize>(
        tensor: CubeTensor<CudaRuntime>,
    ) -> Tensor<BenchBackend, D> {
        Tensor::from_primitive(burn::tensor::TensorPrimitive::Float(tensor))
    }

    pub fn bench_conv3d(c: &mut Criterion) {
        let device = CudaDevice::default();

        let mut group = c.benchmark_group("conv3d");
        group.sample_size(50);

        // Test configurations: (batch, in_ch, out_ch, t, h, w, kernel, stride, padding, name)
        let configs = [
            // Tiny - for im2col comparison (fast enough to run)
            (1, 2, 4, 4, 8, 8, 3, 1, 1, "tiny_3x3x3"),
            // Small - typical video VAE layer
            (1, 4, 8, 8, 32, 32, 3, 1, 1, "small_3x3x3"),
            // Medium - larger spatial
            (1, 8, 16, 8, 64, 64, 3, 1, 1, "medium_3x3x3"),
            // Strided - downsampling
            (1, 8, 16, 8, 64, 64, 3, 2, 1, "strided_3x3x3"),
            // Deeper channels
            (1, 32, 64, 4, 32, 32, 3, 1, 1, "deep_3x3x3"),
        ];

        for (batch, in_ch, out_ch, t, h, w, k, s, p, name) in configs {
            let input = Tensor::<BenchBackend, 5>::random(
                [batch, in_ch, t, h, w],
                burn::tensor::Distribution::Uniform(-1.0, 1.0),
                &device,
            );

            let weight = Tensor::<BenchBackend, 5>::random(
                [out_ch, in_ch, k, k, k],
                burn::tensor::Distribution::Uniform(-0.5, 0.5),
                &device,
            );

            let bias = Tensor::<BenchBackend, 1>::random(
                [out_ch],
                burn::tensor::Distribution::Uniform(-0.1, 0.1),
                &device,
            );

            // Benchmark CubeCL simple kernel (NCTHW)
            group.bench_with_input(
                BenchmarkId::new("cubecl_simple", name),
                &(&input, &weight, &bias),
                |b, (input, weight, bias)| {
                    b.iter(|| {
                        let input_cube = to_cube_tensor((*input).clone());
                        let weight_cube = to_cube_tensor((*weight).clone());
                        let bias_cube = Some(to_cube_tensor((*bias).clone()));

                        let options = Conv3dOptions {
                            stride: [s, s, s],
                            padding: [p, p, p],
                            dilation: [1, 1, 1],
                            groups: 1,
                            layout: Layout::NCTHW,
                        };

                        let output =
                            conv3d::<CudaRuntime>(input_cube, weight_cube, bias_cube, options)
                                .expect("CubeCL conv3d failed");

                        black_box(from_cube_tensor::<5>(output))
                    })
                },
            );

            // Prepare NTHWC tensors for optimized kernel
            // Input: permute NCTHW -> NTHWC
            let input_nthwc = input.clone().permute([0, 2, 3, 4, 1]);
            // Weight: permute [out_ch, in_ch, k_t, k_h, k_w] -> [out_ch, k_t, k_h, k_w, in_ch]
            let weight_nthwc = weight.clone().permute([0, 2, 3, 4, 1]);

            // Benchmark CubeCL optimized kernel (NTHWC with vectorization)
            // This assumes data is already in NTHWC format
            group.bench_with_input(
                BenchmarkId::new("cubecl_optimized", name),
                &(&input_nthwc, &weight_nthwc, &bias),
                |b, (input, weight, bias)| {
                    b.iter(|| {
                        let input_cube = to_cube_tensor((*input).clone());
                        let weight_cube = to_cube_tensor((*weight).clone());
                        let bias_cube = Some(to_cube_tensor((*bias).clone()));

                        let options = Conv3dOptimizedOptions {
                            stride: [s, s, s],
                            padding: [p, p, p],
                            dilation: [1, 1, 1],
                            groups: 1,
                        };

                        let output = conv3d_nthwc::<CudaRuntime>(
                            input_cube,
                            weight_cube,
                            bias_cube,
                            options,
                        )
                        .expect("CubeCL optimized conv3d failed");

                        black_box(from_cube_tensor::<5>(output))
                    })
                },
            );

            // Fair comparison: NCTHW input -> permute -> optimized kernel -> permute back
            // This is what you'd pay if your data starts in channels-first format
            group.bench_with_input(
                BenchmarkId::new("cubecl_opt_with_permute", name),
                &(&input, &weight, &bias),
                |b, (input, weight, bias)| {
                    b.iter(|| {
                        // Permute NCTHW -> NTHWC
                        let input_nthwc = (*input).clone().permute([0, 2, 3, 4, 1]);
                        let weight_nthwc = (*weight).clone().permute([0, 2, 3, 4, 1]);

                        let input_cube = to_cube_tensor(input_nthwc);
                        let weight_cube = to_cube_tensor(weight_nthwc);
                        let bias_cube = Some(to_cube_tensor((*bias).clone()));

                        let options = Conv3dOptimizedOptions {
                            stride: [s, s, s],
                            padding: [p, p, p],
                            dilation: [1, 1, 1],
                            groups: 1,
                        };

                        let output = conv3d_nthwc::<CudaRuntime>(
                            input_cube,
                            weight_cube,
                            bias_cube,
                            options,
                        )
                        .expect("CubeCL optimized conv3d failed");

                        // Permute NTHWC -> NCTHW
                        let output_ncthw: Tensor<BenchBackend, 5> = from_cube_tensor(output);
                        let output_ncthw = output_ncthw.permute([0, 4, 1, 2, 3]);

                        black_box(output_ncthw)
                    })
                },
            );

            // Benchmark im2col reference (only on tiny config - too slow for larger)
            if name == "tiny_3x3x3" {
                group.bench_with_input(
                    BenchmarkId::new("im2col", name),
                    &(&input, &weight, &bias),
                    |b, (input, weight, bias)| {
                        b.iter(|| {
                            let output = reference::conv3d_im2col(
                                (*input).clone(),
                                (*weight).clone(),
                                Some((*bias).clone()),
                                [s, s, s],
                                [p, p, p],
                            );
                            black_box(output)
                        })
                    },
                );
            }
        }

        group.finish();
    }
}

#[cfg(feature = "cuda")]
fn conv3d_benchmarks(c: &mut Criterion) {
    cuda_bench::bench_conv3d(c);
}

#[cfg(not(feature = "cuda"))]
fn conv3d_benchmarks(_c: &mut Criterion) {
    eprintln!("Conv3d benchmarks require --features cuda");
    eprintln!(
        "Run: CUDA_PATH=/path/to/cuda cargo bench -p burn-models-cubecl --features cuda --bench conv3d"
    );
}

criterion_group!(benches, conv3d_benchmarks);
criterion_main!(benches);
