//! CUDA-based correctness tests for Conv3d and Pool3d
//!
//! Run with: `cargo test -p burn-models-cubecl --features cuda --test correctness_cuda -- --ignored`

#![cfg(feature = "cuda")]

use burn::prelude::*;
use burn_cubecl::{CubeBackend, tensor::CubeTensor};
use burn_cuda::CudaDevice;
use cubecl::cuda::CudaRuntime;
use sketchpad_cubecl::{
    Conv3dOptions, FlashAttentionOptions, GroupNormSiLuOptions, Layout, Pool3dOptions, avg_pool3d,
    conv3d, flash_attention, groupnorm, groupnorm_silu, max_pool3d,
};

// Use CubeBackend directly to avoid FusionTensor wrapper
type TestBackend = CubeBackend<CudaRuntime, f32, i32, u32>;

/// Reference im2col-based 3D convolution
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

    pub fn conv3d_reference<B: Backend>(
        input: Tensor<B, 5>,
        weight: Tensor<B, 5>,
        bias: Option<Tensor<B, 1>>,
        stride: [usize; 3],
        padding: [usize; 3],
        _dilation: [usize; 3],
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

fn to_cube_tensor<const D: usize>(tensor: Tensor<TestBackend, D>) -> CubeTensor<CudaRuntime> {
    match tensor.into_primitive() {
        burn::tensor::TensorPrimitive::Float(t) => t,
        _ => panic!("Expected float tensor"),
    }
}

fn from_cube_tensor<const D: usize>(tensor: CubeTensor<CudaRuntime>) -> Tensor<TestBackend, D> {
    Tensor::from_primitive(burn::tensor::TensorPrimitive::Float(tensor))
}

fn run_cubecl_conv3d(
    input: Tensor<TestBackend, 5>,
    weight: Tensor<TestBackend, 5>,
    bias: Option<Tensor<TestBackend, 1>>,
    options: Conv3dOptions,
) -> Tensor<TestBackend, 5> {
    let input_cube = to_cube_tensor(input);
    let weight_cube = to_cube_tensor(weight);
    let bias_cube = bias.map(to_cube_tensor);

    let output_cube = conv3d::<CudaRuntime>(input_cube, weight_cube, bias_cube, options)
        .expect("CubeCL conv3d failed");

    from_cube_tensor(output_cube)
}

fn assert_tensors_approx_eq<const D: usize>(
    actual: Tensor<TestBackend, D>,
    expected: Tensor<TestBackend, D>,
    tolerance: f32,
    test_name: &str,
) {
    let actual_data: Vec<f32> = actual.into_data().to_vec().unwrap();
    let expected_data: Vec<f32> = expected.into_data().to_vec().unwrap();

    assert_eq!(
        actual_data.len(),
        expected_data.len(),
        "{}: tensor sizes don't match",
        test_name
    );

    let mut max_diff: f32 = 0.0;
    let mut max_diff_idx = 0;

    for (i, (a, e)) in actual_data.iter().zip(expected_data.iter()).enumerate() {
        let diff = (a - e).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }

    assert!(
        max_diff <= tolerance,
        "{}: max difference {} at index {} exceeds tolerance {} (actual={}, expected={})",
        test_name,
        max_diff,
        max_diff_idx,
        tolerance,
        actual_data[max_diff_idx],
        expected_data[max_diff_idx]
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_cuda_conv3d_1x1x1_kernel() {
    let device = CudaDevice::default();

    let batch = 1;
    let in_ch = 2;
    let out_ch = 3;
    let t = 4;
    let h = 4;
    let w = 4;

    let input = Tensor::<TestBackend, 5>::random(
        [batch, in_ch, t, h, w],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    let weight = Tensor::<TestBackend, 5>::random(
        [out_ch, in_ch, 1, 1, 1],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &device,
    );

    let bias = Tensor::<TestBackend, 1>::random(
        [out_ch],
        burn::tensor::Distribution::Uniform(-0.1, 0.1),
        &device,
    );

    let options = Conv3dOptions {
        stride: [1, 1, 1],
        padding: [0, 0, 0],
        dilation: [1, 1, 1],
        groups: 1,
        layout: Layout::NCTHW,
    };

    let cubecl_output =
        run_cubecl_conv3d(input.clone(), weight.clone(), Some(bias.clone()), options);

    let reference_output =
        reference::conv3d_reference(input, weight, Some(bias), [1, 1, 1], [0, 0, 0], [1, 1, 1]);

    assert_eq!(cubecl_output.dims(), reference_output.dims());
    assert_tensors_approx_eq(cubecl_output, reference_output, 1e-5, "1x1x1 kernel (CUDA)");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_cuda_conv3d_3x3x3_same_padding() {
    let device = CudaDevice::default();

    let input = Tensor::<TestBackend, 5>::random(
        [1, 3, 8, 8, 8],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );
    let weight = Tensor::<TestBackend, 5>::random(
        [8, 3, 3, 3, 3],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &device,
    );
    let bias = Tensor::<TestBackend, 1>::random(
        [8],
        burn::tensor::Distribution::Uniform(-0.1, 0.1),
        &device,
    );

    let options = Conv3dOptions {
        stride: [1, 1, 1],
        padding: [1, 1, 1],
        dilation: [1, 1, 1],
        groups: 1,
        layout: Layout::NCTHW,
    };

    let cubecl_output =
        run_cubecl_conv3d(input.clone(), weight.clone(), Some(bias.clone()), options);
    let reference_output =
        reference::conv3d_reference(input, weight, Some(bias), [1, 1, 1], [1, 1, 1], [1, 1, 1]);

    assert_eq!(cubecl_output.dims(), [1, 8, 8, 8, 8]);
    assert_tensors_approx_eq(cubecl_output, reference_output, 1e-4, "3x3x3 same padding");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_cuda_conv3d_3x3x3_no_padding() {
    let device = CudaDevice::default();

    let input = Tensor::<TestBackend, 5>::random(
        [1, 2, 6, 6, 6],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );
    let weight = Tensor::<TestBackend, 5>::random(
        [4, 2, 3, 3, 3],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &device,
    );

    let options = Conv3dOptions {
        stride: [1, 1, 1],
        padding: [0, 0, 0],
        dilation: [1, 1, 1],
        groups: 1,
        layout: Layout::NCTHW,
    };

    let cubecl_output = run_cubecl_conv3d(input.clone(), weight.clone(), None, options);
    let reference_output =
        reference::conv3d_reference(input, weight, None, [1, 1, 1], [0, 0, 0], [1, 1, 1]);

    assert_eq!(cubecl_output.dims(), [1, 4, 4, 4, 4]);
    assert_tensors_approx_eq(cubecl_output, reference_output, 1e-4, "3x3x3 no padding");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_cuda_conv3d_stride_2() {
    let device = CudaDevice::default();

    let input = Tensor::<TestBackend, 5>::random(
        [1, 2, 8, 8, 8],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );
    let weight = Tensor::<TestBackend, 5>::random(
        [4, 2, 3, 3, 3],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &device,
    );

    let options = Conv3dOptions {
        stride: [2, 2, 2],
        padding: [1, 1, 1],
        dilation: [1, 1, 1],
        groups: 1,
        layout: Layout::NCTHW,
    };

    let cubecl_output = run_cubecl_conv3d(input.clone(), weight.clone(), None, options);
    let reference_output =
        reference::conv3d_reference(input, weight, None, [2, 2, 2], [1, 1, 1], [1, 1, 1]);

    assert_eq!(cubecl_output.dims(), [1, 4, 4, 4, 4]);
    assert_tensors_approx_eq(cubecl_output, reference_output, 1e-4, "stride 2");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_cuda_conv3d_batch_2() {
    let device = CudaDevice::default();

    let input = Tensor::<TestBackend, 5>::random(
        [2, 2, 4, 4, 4],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );
    let weight = Tensor::<TestBackend, 5>::random(
        [3, 2, 3, 3, 3],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &device,
    );
    let bias = Tensor::<TestBackend, 1>::random(
        [3],
        burn::tensor::Distribution::Uniform(-0.1, 0.1),
        &device,
    );

    let options = Conv3dOptions {
        stride: [1, 1, 1],
        padding: [1, 1, 1],
        dilation: [1, 1, 1],
        groups: 1,
        layout: Layout::NCTHW,
    };

    let cubecl_output =
        run_cubecl_conv3d(input.clone(), weight.clone(), Some(bias.clone()), options);
    let reference_output =
        reference::conv3d_reference(input, weight, Some(bias), [1, 1, 1], [1, 1, 1], [1, 1, 1]);

    assert_eq!(cubecl_output.dims(), [2, 3, 4, 4, 4]);
    assert_tensors_approx_eq(cubecl_output, reference_output, 1e-4, "batch 2");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_cuda_conv3d_asymmetric() {
    let device = CudaDevice::default();

    let input = Tensor::<TestBackend, 5>::random(
        [1, 3, 4, 8, 6],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );
    // 1x3x3 kernel
    let weight = Tensor::<TestBackend, 5>::random(
        [4, 3, 1, 3, 3],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &device,
    );

    let options = Conv3dOptions {
        stride: [1, 1, 1],
        padding: [0, 1, 1],
        dilation: [1, 1, 1],
        groups: 1,
        layout: Layout::NCTHW,
    };

    let cubecl_output = run_cubecl_conv3d(input.clone(), weight.clone(), None, options);
    let reference_output =
        reference::conv3d_reference(input, weight, None, [1, 1, 1], [0, 1, 1], [1, 1, 1]);

    assert_eq!(cubecl_output.dims(), [1, 4, 4, 8, 6]);
    assert_tensors_approx_eq(cubecl_output, reference_output, 1e-4, "asymmetric");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_cuda_conv3d_deep_channels() {
    let device = CudaDevice::default();

    let input = Tensor::<TestBackend, 5>::random(
        [1, 32, 4, 4, 4],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );
    let weight = Tensor::<TestBackend, 5>::random(
        [64, 32, 3, 3, 3],
        burn::tensor::Distribution::Uniform(-0.1, 0.1),
        &device,
    );

    let options = Conv3dOptions {
        stride: [1, 1, 1],
        padding: [1, 1, 1],
        dilation: [1, 1, 1],
        groups: 1,
        layout: Layout::NCTHW,
    };

    let cubecl_output = run_cubecl_conv3d(input.clone(), weight.clone(), None, options);
    let reference_output =
        reference::conv3d_reference(input, weight, None, [1, 1, 1], [1, 1, 1], [1, 1, 1]);

    assert_eq!(cubecl_output.dims(), [1, 64, 4, 4, 4]);
    assert_tensors_approx_eq(cubecl_output, reference_output, 1e-3, "deep channels");
}

// ============================================================================
// NTHWC Layout Tests
// ============================================================================

/// Helper to permute NCTHW -> NTHWC for creating test inputs
fn ncthw_to_nthwc(tensor: Tensor<TestBackend, 5>) -> Tensor<TestBackend, 5> {
    tensor.permute([0, 2, 3, 4, 1])
}

/// Helper to permute NTHWC -> NCTHW for comparing with reference
fn nthwc_to_ncthw(tensor: Tensor<TestBackend, 5>) -> Tensor<TestBackend, 5> {
    tensor.permute([0, 4, 1, 2, 3])
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_cuda_conv3d_nthwc_3x3x3_same_padding() {
    let device = CudaDevice::default();

    // Create NCTHW input, then permute to NTHWC for the CubeCL test
    let input_ncthw = Tensor::<TestBackend, 5>::random(
        [1, 3, 8, 8, 8],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );
    let weight = Tensor::<TestBackend, 5>::random(
        [8, 3, 3, 3, 3],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &device,
    );
    let bias = Tensor::<TestBackend, 1>::random(
        [8],
        burn::tensor::Distribution::Uniform(-0.1, 0.1),
        &device,
    );

    // CubeCL: input in NTHWC format
    let input_nthwc = ncthw_to_nthwc(input_ncthw.clone());

    let options = Conv3dOptions {
        stride: [1, 1, 1],
        padding: [1, 1, 1],
        dilation: [1, 1, 1],
        groups: 1,
        layout: Layout::NTHWC,
    };

    let cubecl_output_nthwc =
        run_cubecl_conv3d(input_nthwc, weight.clone(), Some(bias.clone()), options);

    // Output should be NTHWC: [1, 8, 8, 8, 8] -> batch, time, height, width, channels
    assert_eq!(cubecl_output_nthwc.dims(), [1, 8, 8, 8, 8]);

    // Convert back to NCTHW for comparison with reference
    let cubecl_output_ncthw = nthwc_to_ncthw(cubecl_output_nthwc);
    let ref_output = reference::conv3d_reference(
        input_ncthw,
        weight,
        Some(bias),
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    );

    assert_tensors_approx_eq(
        cubecl_output_ncthw,
        ref_output,
        1e-4,
        "NTHWC 3x3x3 same padding (CUDA)",
    );
}

// ============================================================================
// Pool3d Tests
// ============================================================================

/// Reference implementation for Pool3d
mod pool_reference {
    use burn::prelude::*;

    pub fn avg_pool3d_reference<B: Backend>(
        input: Tensor<B, 5>,
        kernel_size: [usize; 3],
        stride: [usize; 3],
        padding: [usize; 3],
    ) -> Tensor<B, 5> {
        let [batch, channels, in_t, in_h, in_w] = input.dims();
        let device = input.device();

        let out_t = (in_t + 2 * padding[0] - kernel_size[0]) / stride[0] + 1;
        let out_h = (in_h + 2 * padding[1] - kernel_size[1]) / stride[1] + 1;
        let out_w = (in_w + 2 * padding[2] - kernel_size[2]) / stride[2] + 1;

        let mut output_data = vec![0.0f32; batch * channels * out_t * out_h * out_w];
        let input_data: Vec<f32> = input.into_data().to_vec().unwrap();

        for b in 0..batch {
            for c in 0..channels {
                for ot in 0..out_t {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let mut sum = 0.0f32;
                            let mut count = 0u32;

                            for kt in 0..kernel_size[0] {
                                for kh in 0..kernel_size[1] {
                                    for kw in 0..kernel_size[2] {
                                        let it =
                                            (ot * stride[0] + kt) as isize - padding[0] as isize;
                                        let ih =
                                            (oh * stride[1] + kh) as isize - padding[1] as isize;
                                        let iw =
                                            (ow * stride[2] + kw) as isize - padding[2] as isize;

                                        if it >= 0
                                            && (it as usize) < in_t
                                            && ih >= 0
                                            && (ih as usize) < in_h
                                            && iw >= 0
                                            && (iw as usize) < in_w
                                        {
                                            let in_idx = b * channels * in_t * in_h * in_w
                                                + c * in_t * in_h * in_w
                                                + (it as usize) * in_h * in_w
                                                + (ih as usize) * in_w
                                                + (iw as usize);
                                            sum += input_data[in_idx];
                                            count += 1;
                                        }
                                    }
                                }
                            }

                            let out_idx = b * channels * out_t * out_h * out_w
                                + c * out_t * out_h * out_w
                                + ot * out_h * out_w
                                + oh * out_w
                                + ow;
                            output_data[out_idx] = sum / count as f32;
                        }
                    }
                }
            }
        }

        Tensor::from_data(
            burn::tensor::TensorData::new(output_data, [batch, channels, out_t, out_h, out_w]),
            &device,
        )
    }

    pub fn max_pool3d_reference<B: Backend>(
        input: Tensor<B, 5>,
        kernel_size: [usize; 3],
        stride: [usize; 3],
        padding: [usize; 3],
    ) -> Tensor<B, 5> {
        let [batch, channels, in_t, in_h, in_w] = input.dims();
        let device = input.device();

        let out_t = (in_t + 2 * padding[0] - kernel_size[0]) / stride[0] + 1;
        let out_h = (in_h + 2 * padding[1] - kernel_size[1]) / stride[1] + 1;
        let out_w = (in_w + 2 * padding[2] - kernel_size[2]) / stride[2] + 1;

        let mut output_data = vec![f32::NEG_INFINITY; batch * channels * out_t * out_h * out_w];
        let input_data: Vec<f32> = input.into_data().to_vec().unwrap();

        for b in 0..batch {
            for c in 0..channels {
                for ot in 0..out_t {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let mut max_val = f32::NEG_INFINITY;

                            for kt in 0..kernel_size[0] {
                                for kh in 0..kernel_size[1] {
                                    for kw in 0..kernel_size[2] {
                                        let it =
                                            (ot * stride[0] + kt) as isize - padding[0] as isize;
                                        let ih =
                                            (oh * stride[1] + kh) as isize - padding[1] as isize;
                                        let iw =
                                            (ow * stride[2] + kw) as isize - padding[2] as isize;

                                        if it >= 0
                                            && (it as usize) < in_t
                                            && ih >= 0
                                            && (ih as usize) < in_h
                                            && iw >= 0
                                            && (iw as usize) < in_w
                                        {
                                            let in_idx = b * channels * in_t * in_h * in_w
                                                + c * in_t * in_h * in_w
                                                + (it as usize) * in_h * in_w
                                                + (ih as usize) * in_w
                                                + (iw as usize);
                                            if input_data[in_idx] > max_val {
                                                max_val = input_data[in_idx];
                                            }
                                        }
                                    }
                                }
                            }

                            let out_idx = b * channels * out_t * out_h * out_w
                                + c * out_t * out_h * out_w
                                + ot * out_h * out_w
                                + oh * out_w
                                + ow;
                            output_data[out_idx] = max_val;
                        }
                    }
                }
            }
        }

        Tensor::from_data(
            burn::tensor::TensorData::new(output_data, [batch, channels, out_t, out_h, out_w]),
            &device,
        )
    }
}

fn run_cubecl_avg_pool3d(
    input: Tensor<TestBackend, 5>,
    options: Pool3dOptions,
) -> Tensor<TestBackend, 5> {
    let input_cube = to_cube_tensor(input);
    let output_cube = avg_pool3d::<CudaRuntime>(input_cube, options);
    from_cube_tensor(output_cube)
}

fn run_cubecl_max_pool3d(
    input: Tensor<TestBackend, 5>,
    options: Pool3dOptions,
) -> Tensor<TestBackend, 5> {
    let input_cube = to_cube_tensor(input);
    let output_cube = max_pool3d::<CudaRuntime>(input_cube, options);
    from_cube_tensor(output_cube)
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_cuda_avg_pool3d_2x2x2() {
    let device = CudaDevice::default();

    let input = Tensor::<TestBackend, 5>::random(
        [1, 2, 4, 4, 4],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    let options = Pool3dOptions {
        kernel_size: [2, 2, 2],
        stride: [2, 2, 2],
        padding: [0, 0, 0],
    };

    let cubecl_output = run_cubecl_avg_pool3d(input.clone(), options);
    let ref_output = pool_reference::avg_pool3d_reference(input, [2, 2, 2], [2, 2, 2], [0, 0, 0]);

    assert_eq!(cubecl_output.dims(), [1, 2, 2, 2, 2]);
    assert_tensors_approx_eq(cubecl_output, ref_output, 1e-5, "avg_pool3d 2x2x2 (CUDA)");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_cuda_avg_pool3d_with_padding() {
    let device = CudaDevice::default();

    let input = Tensor::<TestBackend, 5>::random(
        [1, 3, 6, 6, 6],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    let options = Pool3dOptions {
        kernel_size: [3, 3, 3],
        stride: [2, 2, 2],
        padding: [1, 1, 1],
    };

    let cubecl_output = run_cubecl_avg_pool3d(input.clone(), options);
    let ref_output = pool_reference::avg_pool3d_reference(input, [3, 3, 3], [2, 2, 2], [1, 1, 1]);

    assert_eq!(cubecl_output.dims(), [1, 3, 3, 3, 3]);
    assert_tensors_approx_eq(
        cubecl_output,
        ref_output,
        1e-5,
        "avg_pool3d with padding (CUDA)",
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_cuda_max_pool3d_2x2x2() {
    let device = CudaDevice::default();

    let input = Tensor::<TestBackend, 5>::random(
        [1, 2, 4, 4, 4],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    let options = Pool3dOptions {
        kernel_size: [2, 2, 2],
        stride: [2, 2, 2],
        padding: [0, 0, 0],
    };

    let cubecl_output = run_cubecl_max_pool3d(input.clone(), options);
    let ref_output = pool_reference::max_pool3d_reference(input, [2, 2, 2], [2, 2, 2], [0, 0, 0]);

    assert_eq!(cubecl_output.dims(), [1, 2, 2, 2, 2]);
    assert_tensors_approx_eq(cubecl_output, ref_output, 1e-5, "max_pool3d 2x2x2 (CUDA)");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_cuda_max_pool3d_with_padding() {
    let device = CudaDevice::default();

    let input = Tensor::<TestBackend, 5>::random(
        [2, 4, 8, 8, 8],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    let options = Pool3dOptions {
        kernel_size: [3, 3, 3],
        stride: [2, 2, 2],
        padding: [1, 1, 1],
    };

    let cubecl_output = run_cubecl_max_pool3d(input.clone(), options);
    let ref_output = pool_reference::max_pool3d_reference(input, [3, 3, 3], [2, 2, 2], [1, 1, 1]);

    assert_eq!(cubecl_output.dims(), [2, 4, 4, 4, 4]);
    assert_tensors_approx_eq(
        cubecl_output,
        ref_output,
        1e-5,
        "max_pool3d with padding (CUDA)",
    );
}

// ============================================================================
// Flash Attention Tests
// ============================================================================

/// Reference implementation for causal attention
mod attention_reference {
    use burn::prelude::*;

    /// Standard causal attention: softmax(Q @ K^T / sqrt(d) + causal_mask) @ V
    pub fn causal_attention_reference<B: Backend>(
        q: Tensor<B, 4>, // [batch, heads, seq_q, head_dim]
        k: Tensor<B, 4>, // [batch, heads, seq_k, head_dim]
        v: Tensor<B, 4>, // [batch, heads, seq_k, val_dim]
    ) -> Tensor<B, 4> {
        let [batch, heads, seq_q, head_dim] = q.dims();
        let [_, _, seq_k, _] = k.dims();
        let device = q.device();

        let scale = (head_dim as f64).powf(-0.5);

        // Q @ K^T -> [batch, heads, seq_q, seq_k]
        let scores = q.matmul(k.transpose()) * scale;

        // Create causal mask (lower triangular)
        let mut mask_data = vec![0.0f32; seq_q * seq_k];
        for i in 0..seq_q {
            for j in 0..seq_k {
                if j > i {
                    mask_data[i * seq_k + j] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(mask_data, [seq_q, seq_k]),
            &device,
        );

        let mask = mask.unsqueeze::<4>().expand([batch, heads, seq_q, seq_k]);
        let scores = scores + mask;
        let attn = burn::tensor::activation::softmax(scores, 3);
        attn.matmul(v)
    }
}

/// Helper to run flash attention (causal mode for comparison with reference)
fn run_cuda_flash_attention(
    q: Tensor<TestBackend, 4>,
    k: Tensor<TestBackend, 4>,
    v: Tensor<TestBackend, 4>,
) -> Tensor<TestBackend, 4> {
    let q_cube = q.into_primitive().tensor();
    let k_cube = k.into_primitive().tensor();
    let v_cube = v.into_primitive().tensor();

    // Use causal mode to match reference implementation
    let output = flash_attention(q_cube, k_cube, v_cube, FlashAttentionOptions::causal())
        .expect("flash attention failed");

    Tensor::from_primitive(burn::tensor::TensorPrimitive::Float(output))
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_cuda_flash_attention_basic() {
    let device = CudaDevice::default();

    let batch = 1;
    let heads = 4;
    let seq_len = 16;
    let head_dim = 32;

    let q = Tensor::<TestBackend, 4>::random(
        [batch, heads, seq_len, head_dim],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &device,
    );
    let k = Tensor::<TestBackend, 4>::random(
        [batch, heads, seq_len, head_dim],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &device,
    );
    let v = Tensor::<TestBackend, 4>::random(
        [batch, heads, seq_len, head_dim],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &device,
    );

    let flash_output = run_cuda_flash_attention(q.clone(), k.clone(), v.clone());
    let ref_output = attention_reference::causal_attention_reference(q, k, v);

    assert_eq!(flash_output.dims(), ref_output.dims());
    assert_tensors_approx_eq(
        flash_output,
        ref_output,
        1e-3,
        "flash attention basic (CUDA)",
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_cuda_flash_attention_longer_sequence() {
    let device = CudaDevice::default();

    let batch = 2;
    let heads = 8;
    let seq_len = 64;
    let head_dim = 64;

    let q = Tensor::<TestBackend, 4>::random(
        [batch, heads, seq_len, head_dim],
        burn::tensor::Distribution::Uniform(-0.3, 0.3),
        &device,
    );
    let k = Tensor::<TestBackend, 4>::random(
        [batch, heads, seq_len, head_dim],
        burn::tensor::Distribution::Uniform(-0.3, 0.3),
        &device,
    );
    let v = Tensor::<TestBackend, 4>::random(
        [batch, heads, seq_len, head_dim],
        burn::tensor::Distribution::Uniform(-0.3, 0.3),
        &device,
    );

    let flash_output = run_cuda_flash_attention(q.clone(), k.clone(), v.clone());
    let ref_output = attention_reference::causal_attention_reference(q, k, v);

    assert_eq!(flash_output.dims(), ref_output.dims());
    assert_tensors_approx_eq(
        flash_output,
        ref_output,
        5e-3,
        "flash attention longer seq (CUDA)",
    );
}

// ============================================================================
// GroupNorm + SiLU Tests
// ============================================================================

/// Reference implementation for GroupNorm + SiLU
mod groupnorm_reference {
    use burn::prelude::*;

    fn silu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
        x.clone() * burn::tensor::activation::sigmoid(x)
    }

    /// GroupNorm: (x - mean) / sqrt(var + eps) * weight + bias
    pub fn groupnorm_reference<B: Backend>(
        input: Tensor<B, 4>,
        weight: Tensor<B, 1>,
        bias: Tensor<B, 1>,
        num_groups: usize,
        eps: f64,
    ) -> Tensor<B, 4> {
        let [batch, channels, height, width] = input.dims();
        let group_size = channels / num_groups;

        // Reshape to [batch, num_groups, group_size * height * width]
        let x = input.reshape([batch, num_groups, group_size * height * width]);

        // Compute mean and variance over the last dimension
        let mean = x.clone().mean_dim(2);
        let var = x.clone().var(2);

        // Expand for broadcasting: [batch, num_groups, 1]
        let mean = mean.unsqueeze::<3>();
        let var = var.unsqueeze::<3>();

        // Normalize
        let x = (x - mean) / (var + eps).sqrt();

        // Reshape back to [batch, channels, height, width]
        let x = x.reshape([batch, channels, height, width]);

        // Apply weight and bias
        let weight = weight.reshape([1, channels, 1, 1]);
        let bias = bias.reshape([1, channels, 1, 1]);

        x * weight + bias
    }

    pub fn groupnorm_silu_reference<B: Backend>(
        input: Tensor<B, 4>,
        weight: Tensor<B, 1>,
        bias: Tensor<B, 1>,
        num_groups: usize,
        eps: f64,
    ) -> Tensor<B, 4> {
        silu(groupnorm_reference(input, weight, bias, num_groups, eps))
    }
}

fn run_cubecl_groupnorm(
    input: Tensor<TestBackend, 4>,
    weight: Tensor<TestBackend, 1>,
    bias: Tensor<TestBackend, 1>,
    options: GroupNormSiLuOptions,
) -> Tensor<TestBackend, 4> {
    let input_cube = to_cube_tensor(input);
    let weight_cube = to_cube_tensor(weight);
    let bias_cube = to_cube_tensor(bias);
    let output_cube = groupnorm::<CudaRuntime>(input_cube, weight_cube, bias_cube, options);
    from_cube_tensor(output_cube)
}

fn run_cubecl_groupnorm_silu(
    input: Tensor<TestBackend, 4>,
    weight: Tensor<TestBackend, 1>,
    bias: Tensor<TestBackend, 1>,
    options: GroupNormSiLuOptions,
) -> Tensor<TestBackend, 4> {
    let input_cube = to_cube_tensor(input);
    let weight_cube = to_cube_tensor(weight);
    let bias_cube = to_cube_tensor(bias);
    let output_cube = groupnorm_silu::<CudaRuntime>(input_cube, weight_cube, bias_cube, options);
    from_cube_tensor(output_cube)
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_cuda_groupnorm_basic() {
    let device = CudaDevice::default();

    let batch = 1;
    let channels = 32;
    let height = 8;
    let width = 8;
    let num_groups = 8;

    let input = Tensor::<TestBackend, 4>::random(
        [batch, channels, height, width],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );
    let weight = Tensor::<TestBackend, 1>::random(
        [channels],
        burn::tensor::Distribution::Uniform(0.5, 1.5),
        &device,
    );
    let bias = Tensor::<TestBackend, 1>::random(
        [channels],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &device,
    );

    let options = GroupNormSiLuOptions::with_groups(num_groups);

    let cubecl_output = run_cubecl_groupnorm(input.clone(), weight.clone(), bias.clone(), options);
    let ref_output =
        groupnorm_reference::groupnorm_reference(input, weight, bias, num_groups, 1e-5);

    assert_eq!(cubecl_output.dims(), ref_output.dims());
    // Tolerance accounts for variance calculation differences (biased vs unbiased)
    assert_tensors_approx_eq(cubecl_output, ref_output, 1e-2, "groupnorm basic (CUDA)");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_cuda_groupnorm_silu_basic() {
    let device = CudaDevice::default();

    let batch = 1;
    let channels = 32;
    let height = 8;
    let width = 8;
    let num_groups = 8;

    let input = Tensor::<TestBackend, 4>::random(
        [batch, channels, height, width],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );
    let weight = Tensor::<TestBackend, 1>::random(
        [channels],
        burn::tensor::Distribution::Uniform(0.5, 1.5),
        &device,
    );
    let bias = Tensor::<TestBackend, 1>::random(
        [channels],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &device,
    );

    let options = GroupNormSiLuOptions::with_groups(num_groups);

    let cubecl_output =
        run_cubecl_groupnorm_silu(input.clone(), weight.clone(), bias.clone(), options);
    let ref_output =
        groupnorm_reference::groupnorm_silu_reference(input, weight, bias, num_groups, 1e-5);

    assert_eq!(cubecl_output.dims(), ref_output.dims());
    // Tolerance accounts for variance calculation differences (biased vs unbiased)
    assert_tensors_approx_eq(
        cubecl_output,
        ref_output,
        1e-2,
        "groupnorm_silu basic (CUDA)",
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_cuda_groupnorm_silu_batch() {
    let device = CudaDevice::default();

    let batch = 4;
    let channels = 64;
    let height = 16;
    let width = 16;
    let num_groups = 32;

    let input = Tensor::<TestBackend, 4>::random(
        [batch, channels, height, width],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );
    let weight = Tensor::<TestBackend, 1>::random(
        [channels],
        burn::tensor::Distribution::Uniform(0.5, 1.5),
        &device,
    );
    let bias = Tensor::<TestBackend, 1>::random(
        [channels],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &device,
    );

    let options = GroupNormSiLuOptions::with_groups(num_groups);

    let cubecl_output =
        run_cubecl_groupnorm_silu(input.clone(), weight.clone(), bias.clone(), options);
    let ref_output =
        groupnorm_reference::groupnorm_silu_reference(input, weight, bias, num_groups, 1e-5);

    assert_eq!(cubecl_output.dims(), ref_output.dims());
    // Tolerance accounts for variance calculation differences (biased vs unbiased)
    assert_tensors_approx_eq(
        cubecl_output,
        ref_output,
        1e-2,
        "groupnorm_silu batch (CUDA)",
    );
}

// =============================================================================
// UNet CubeCL Integration Tests (CUDA)
// =============================================================================

#[test]
#[ignore = "requires CUDA GPU"]
fn test_cuda_resblock_cubecl_shape() {
    use sketchpad_unet::cubecl::ResBlockCubeCL;

    let device = CudaDevice::default();

    let in_channels = 256;
    let out_channels = 512;
    let time_emb_dim = 1024;

    let block =
        ResBlockCubeCL::<CudaRuntime>::new(in_channels, out_channels, time_emb_dim, &device);

    let batch = 2;
    let height = 32;
    let width = 32;

    let input = Tensor::<TestBackend, 4>::random(
        [batch, in_channels, height, width],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );
    let time_emb = Tensor::<TestBackend, 2>::random(
        [batch, time_emb_dim],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    let output = block.forward(input, time_emb);

    assert_eq!(output.dims(), [batch, out_channels, height, width]);
    println!("ResBlockCubeCL (CUDA) output shape: {:?}", output.dims());
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_cuda_crossattention_cubecl() {
    use sketchpad_unet::cubecl::CrossAttentionCubeCL;

    let device = CudaDevice::default();

    let query_dim = 320;
    let context_dim = 768;
    let num_heads = 8;
    let head_dim = 40;

    let attn = CrossAttentionCubeCL::<CudaRuntime>::new(
        query_dim,
        num_heads,
        head_dim,
        Some(context_dim),
        &device,
    );

    let batch = 2;
    let seq_len = 64;
    let ctx_len = 77;

    let query = Tensor::<TestBackend, 3>::random(
        [batch, seq_len, query_dim],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );
    let context = Tensor::<TestBackend, 3>::random(
        [batch, ctx_len, context_dim],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    let output = attn.forward(query, Some(context));

    assert_eq!(output.dims(), [batch, seq_len, query_dim]);
    println!(
        "CrossAttentionCubeCL (CUDA) output shape: {:?}",
        output.dims()
    );
}

// =============================================================================
// 3D VAE CubeCL Integration Tests (CUDA)
// =============================================================================

#[test]
#[ignore = "requires CUDA GPU"]
fn test_cuda_vae3d_resblock_shape() {
    use sketchpad_core::vae3d::cubecl::ResBlock3dCubeCL;

    let device = CudaDevice::default();

    let in_channels = 64;
    let out_channels = 128;

    let block = ResBlock3dCubeCL::<CudaRuntime>::new(in_channels, out_channels, &device);

    let batch = 1;
    let time = 4;
    let height = 16;
    let width = 16;

    let input = Tensor::<TestBackend, 5>::random(
        [batch, in_channels, time, height, width],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    let cube_input = sketchpad_cubecl::tensor_to_cube(input);
    let output = block.forward(cube_input);
    let output: Tensor<TestBackend, 5> = sketchpad_cubecl::cube_to_tensor(output);

    assert_eq!(output.dims(), [batch, out_channels, time, height, width]);
    println!("ResBlock3dCubeCL (CUDA) output shape: {:?}", output.dims());
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_cuda_vae3d_encoder_tiny() {
    use sketchpad_core::vae3d::{Vae3dConfig, cubecl::Vae3dEncoderCubeCL};

    let device = CudaDevice::default();

    let config = Vae3dConfig {
        in_channels: 3,
        latent_channels: 4,
        base_channels: 32,
        channel_mults: vec![1, 2],
        temporal_compression: 2,
        spatial_compression: 4,
        num_res_blocks: 1,
    };

    let encoder = Vae3dEncoderCubeCL::<CudaRuntime>::new(config.clone(), &device);

    let batch = 1;
    let time = 4;
    let height = 32;
    let width = 32;

    let input = Tensor::<TestBackend, 5>::random(
        [batch, config.in_channels, time, height, width],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    let output = encoder.forward(input);

    println!(
        "Vae3dEncoderCubeCL (CUDA) mean shape: {:?}",
        output.mean.dims()
    );
    assert_eq!(output.mean.dims()[0], batch);
    assert_eq!(output.mean.dims()[1], config.latent_channels);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_cuda_vae3d_decoder_tiny() {
    use sketchpad_core::vae3d::{Vae3dConfig, cubecl::Vae3dDecoderCubeCL};

    let device = CudaDevice::default();

    let config = Vae3dConfig {
        in_channels: 3,
        latent_channels: 4,
        base_channels: 32,
        channel_mults: vec![1, 2],
        temporal_compression: 2,
        spatial_compression: 4,
        num_res_blocks: 1,
    };

    let decoder = Vae3dDecoderCubeCL::<CudaRuntime>::new(config.clone(), &device);

    let batch = 1;
    let lat_time = 2;
    let lat_height = 8;
    let lat_width = 8;

    let latent = Tensor::<TestBackend, 5>::random(
        [
            batch,
            config.latent_channels,
            lat_time,
            lat_height,
            lat_width,
        ],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    let output = decoder.forward(latent);

    println!(
        "Vae3dDecoderCubeCL (CUDA) output shape: {:?}",
        output.dims()
    );
    assert_eq!(output.dims()[0], batch);
    assert_eq!(output.dims()[1], config.in_channels);
}
