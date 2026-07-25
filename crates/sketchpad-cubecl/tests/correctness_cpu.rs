//! CPU-based correctness tests for Conv3d and Pool3d
//!
//! Run with: `cargo test -p burn-models-cubecl --test correctness_cpu`
//!
//! These tests run on CPU (no GPU required).
//! Note: burn-cpu only works on Linux.

#![cfg(all(feature = "cpu", target_os = "linux"))]

use burn::prelude::*;
use burn_cubecl::CubeBackend;
use burn_cubecl::tensor::CubeTensor;
use burn_ndarray::NdArray;
use cubecl::cpu::{CpuDevice, CpuRuntime};
use sketchpad_cubecl::{
    Conv3dOptimizedOptions, Conv3dOptions, FlashAttentionOptions, GroupNormSiLuOptions, Layout,
    Pool3dOptions, avg_pool3d, conv3d, conv3d_nthwc, flash_attention, groupnorm, groupnorm_silu,
    max_pool3d,
};

// Use CubeBackend directly to avoid Fusion wrapper
type TestBackend = CubeBackend<CpuRuntime, f32, i32, u8>;
type RefBackend = NdArray<f32>;

/// Reference im2col-based 3D convolution for verification
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

fn to_cube_tensor<const D: usize>(tensor: Tensor<TestBackend, D>) -> CubeTensor<CpuRuntime> {
    match tensor.into_primitive() {
        burn::tensor::TensorPrimitive::Float(t) => t,
        _ => panic!("Expected float tensor"),
    }
}

fn from_cube_tensor<const D: usize>(tensor: CubeTensor<CpuRuntime>) -> Tensor<TestBackend, D> {
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

    let output_cube = conv3d::<CpuRuntime>(input_cube, weight_cube, bias_cube, options)
        .expect("CubeCL conv3d failed");

    from_cube_tensor(output_cube)
}

fn assert_tensors_approx_eq<const D: usize>(
    actual: Tensor<TestBackend, D>,
    expected: Tensor<RefBackend, D>,
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

// ============================================================================
// Tests
// ============================================================================

#[test]
fn test_cpu_conv3d_1x1x1_kernel() {
    let cpu_device = CpuDevice;
    let ref_device = Default::default();

    let input = Tensor::<TestBackend, 5>::random(
        [1, 2, 4, 4, 4],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &cpu_device,
    );
    let weight = Tensor::<TestBackend, 5>::random(
        [3, 2, 1, 1, 1],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &cpu_device,
    );
    let bias = Tensor::<TestBackend, 1>::random(
        [3],
        burn::tensor::Distribution::Uniform(-0.1, 0.1),
        &cpu_device,
    );

    // Create matching reference tensors
    let input_ref = Tensor::<RefBackend, 5>::from_data(input.to_data(), &ref_device);
    let weight_ref = Tensor::<RefBackend, 5>::from_data(weight.to_data(), &ref_device);
    let bias_ref = Tensor::<RefBackend, 1>::from_data(bias.to_data(), &ref_device);

    let options = Conv3dOptions {
        stride: [1, 1, 1],
        padding: [0, 0, 0],
        dilation: [1, 1, 1],
        groups: 1,
        layout: Layout::NCTHW,
    };

    let cubecl_output = run_cubecl_conv3d(input, weight, Some(bias), options);
    let ref_output =
        reference::conv3d_reference(input_ref, weight_ref, Some(bias_ref), [1, 1, 1], [0, 0, 0]);

    assert_eq!(cubecl_output.dims(), ref_output.dims());
    assert_tensors_approx_eq(cubecl_output, ref_output, 1e-5, "1x1x1 kernel (CPU)");
}

#[test]
fn test_cpu_conv3d_3x3x3_same_padding() {
    let cpu_device = CpuDevice;
    let ref_device = Default::default();

    let input = Tensor::<TestBackend, 5>::random(
        [1, 3, 8, 8, 8],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &cpu_device,
    );
    let weight = Tensor::<TestBackend, 5>::random(
        [8, 3, 3, 3, 3],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &cpu_device,
    );
    let bias = Tensor::<TestBackend, 1>::random(
        [8],
        burn::tensor::Distribution::Uniform(-0.1, 0.1),
        &cpu_device,
    );

    let input_ref = Tensor::<RefBackend, 5>::from_data(input.to_data(), &ref_device);
    let weight_ref = Tensor::<RefBackend, 5>::from_data(weight.to_data(), &ref_device);
    let bias_ref = Tensor::<RefBackend, 1>::from_data(bias.to_data(), &ref_device);

    let options = Conv3dOptions {
        stride: [1, 1, 1],
        padding: [1, 1, 1],
        dilation: [1, 1, 1],
        groups: 1,
        layout: Layout::NCTHW,
    };

    let cubecl_output = run_cubecl_conv3d(input, weight, Some(bias), options);
    let ref_output =
        reference::conv3d_reference(input_ref, weight_ref, Some(bias_ref), [1, 1, 1], [1, 1, 1]);

    assert_eq!(cubecl_output.dims(), [1, 8, 8, 8, 8]);
    assert_tensors_approx_eq(cubecl_output, ref_output, 1e-4, "3x3x3 same padding (CPU)");
}

#[test]
fn test_cpu_conv3d_3x3x3_no_padding() {
    let cpu_device = CpuDevice;
    let ref_device = Default::default();

    let input = Tensor::<TestBackend, 5>::random(
        [1, 2, 6, 6, 6],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &cpu_device,
    );
    let weight = Tensor::<TestBackend, 5>::random(
        [4, 2, 3, 3, 3],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &cpu_device,
    );

    let input_ref = Tensor::<RefBackend, 5>::from_data(input.to_data(), &ref_device);
    let weight_ref = Tensor::<RefBackend, 5>::from_data(weight.to_data(), &ref_device);

    let options = Conv3dOptions {
        stride: [1, 1, 1],
        padding: [0, 0, 0],
        dilation: [1, 1, 1],
        groups: 1,
        layout: Layout::NCTHW,
    };

    let cubecl_output = run_cubecl_conv3d(input, weight, None, options);
    let ref_output = reference::conv3d_reference(input_ref, weight_ref, None, [1, 1, 1], [0, 0, 0]);

    assert_eq!(cubecl_output.dims(), [1, 4, 4, 4, 4]);
    assert_tensors_approx_eq(cubecl_output, ref_output, 1e-4, "3x3x3 no padding (CPU)");
}

#[test]
fn test_cpu_conv3d_stride_2() {
    let cpu_device = CpuDevice;
    let ref_device = Default::default();

    let input = Tensor::<TestBackend, 5>::random(
        [1, 2, 8, 8, 8],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &cpu_device,
    );
    let weight = Tensor::<TestBackend, 5>::random(
        [4, 2, 3, 3, 3],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &cpu_device,
    );

    let input_ref = Tensor::<RefBackend, 5>::from_data(input.to_data(), &ref_device);
    let weight_ref = Tensor::<RefBackend, 5>::from_data(weight.to_data(), &ref_device);

    let options = Conv3dOptions {
        stride: [2, 2, 2],
        padding: [1, 1, 1],
        dilation: [1, 1, 1],
        groups: 1,
        layout: Layout::NCTHW,
    };

    let cubecl_output = run_cubecl_conv3d(input, weight, None, options);
    let ref_output = reference::conv3d_reference(input_ref, weight_ref, None, [2, 2, 2], [1, 1, 1]);

    assert_eq!(cubecl_output.dims(), [1, 4, 4, 4, 4]);
    assert_tensors_approx_eq(cubecl_output, ref_output, 1e-4, "stride 2 (CPU)");
}

#[test]
fn test_cpu_conv3d_batch_2() {
    let cpu_device = CpuDevice;
    let ref_device = Default::default();

    let input = Tensor::<TestBackend, 5>::random(
        [2, 2, 4, 4, 4],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &cpu_device,
    );
    let weight = Tensor::<TestBackend, 5>::random(
        [3, 2, 3, 3, 3],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &cpu_device,
    );
    let bias = Tensor::<TestBackend, 1>::random(
        [3],
        burn::tensor::Distribution::Uniform(-0.1, 0.1),
        &cpu_device,
    );

    let input_ref = Tensor::<RefBackend, 5>::from_data(input.to_data(), &ref_device);
    let weight_ref = Tensor::<RefBackend, 5>::from_data(weight.to_data(), &ref_device);
    let bias_ref = Tensor::<RefBackend, 1>::from_data(bias.to_data(), &ref_device);

    let options = Conv3dOptions {
        stride: [1, 1, 1],
        padding: [1, 1, 1],
        dilation: [1, 1, 1],
        groups: 1,
        layout: Layout::NCTHW,
    };

    let cubecl_output = run_cubecl_conv3d(input, weight, Some(bias), options);
    let ref_output =
        reference::conv3d_reference(input_ref, weight_ref, Some(bias_ref), [1, 1, 1], [1, 1, 1]);

    assert_eq!(cubecl_output.dims(), [2, 3, 4, 4, 4]);
    assert_tensors_approx_eq(cubecl_output, ref_output, 1e-4, "batch 2 (CPU)");
}

#[test]
fn test_cpu_conv3d_asymmetric() {
    let cpu_device = CpuDevice;
    let ref_device = Default::default();

    let input = Tensor::<TestBackend, 5>::random(
        [1, 3, 4, 8, 6],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &cpu_device,
    );
    let weight = Tensor::<TestBackend, 5>::random(
        [4, 3, 1, 3, 3],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &cpu_device,
    );

    let input_ref = Tensor::<RefBackend, 5>::from_data(input.to_data(), &ref_device);
    let weight_ref = Tensor::<RefBackend, 5>::from_data(weight.to_data(), &ref_device);

    let options = Conv3dOptions {
        stride: [1, 1, 1],
        padding: [0, 1, 1],
        dilation: [1, 1, 1],
        groups: 1,
        layout: Layout::NCTHW,
    };

    let cubecl_output = run_cubecl_conv3d(input, weight, None, options);
    let ref_output = reference::conv3d_reference(input_ref, weight_ref, None, [1, 1, 1], [0, 1, 1]);

    assert_eq!(cubecl_output.dims(), [1, 4, 4, 8, 6]);
    assert_tensors_approx_eq(cubecl_output, ref_output, 1e-4, "asymmetric (CPU)");
}

#[test]
fn test_cpu_conv3d_deep_channels() {
    let cpu_device = CpuDevice;
    let ref_device = Default::default();

    let input = Tensor::<TestBackend, 5>::random(
        [1, 32, 4, 4, 4],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &cpu_device,
    );
    let weight = Tensor::<TestBackend, 5>::random(
        [64, 32, 3, 3, 3],
        burn::tensor::Distribution::Uniform(-0.1, 0.1),
        &cpu_device,
    );

    let input_ref = Tensor::<RefBackend, 5>::from_data(input.to_data(), &ref_device);
    let weight_ref = Tensor::<RefBackend, 5>::from_data(weight.to_data(), &ref_device);

    let options = Conv3dOptions {
        stride: [1, 1, 1],
        padding: [1, 1, 1],
        dilation: [1, 1, 1],
        groups: 1,
        layout: Layout::NCTHW,
    };

    let cubecl_output = run_cubecl_conv3d(input, weight, None, options);
    let ref_output = reference::conv3d_reference(input_ref, weight_ref, None, [1, 1, 1], [1, 1, 1]);

    assert_eq!(cubecl_output.dims(), [1, 64, 4, 4, 4]);
    assert_tensors_approx_eq(cubecl_output, ref_output, 1e-3, "deep channels (CPU)");
}

// ============================================================================
// NTHWC Layout Tests
// ============================================================================

/// Helper to permute NCTHW -> NTHWC for creating test inputs
fn ncthw_to_nthwc<B: Backend>(tensor: Tensor<B, 5>) -> Tensor<B, 5> {
    tensor.permute([0, 2, 3, 4, 1])
}

/// Helper to permute NTHWC -> NCTHW for comparing with reference
fn nthwc_to_ncthw<B: Backend>(tensor: Tensor<B, 5>) -> Tensor<B, 5> {
    tensor.permute([0, 4, 1, 2, 3])
}

#[test]
fn test_cpu_conv3d_nthwc_3x3x3_same_padding() {
    let cpu_device = CpuDevice;
    let ref_device = Default::default();

    // Create NCTHW input, then permute to NTHWC for the CubeCL test
    let input_ncthw = Tensor::<TestBackend, 5>::random(
        [1, 3, 8, 8, 8],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &cpu_device,
    );
    let weight = Tensor::<TestBackend, 5>::random(
        [8, 3, 3, 3, 3],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &cpu_device,
    );
    let bias = Tensor::<TestBackend, 1>::random(
        [8],
        burn::tensor::Distribution::Uniform(-0.1, 0.1),
        &cpu_device,
    );

    // Reference: NCTHW throughout
    let input_ref = Tensor::<RefBackend, 5>::from_data(input_ncthw.to_data(), &ref_device);
    let weight_ref = Tensor::<RefBackend, 5>::from_data(weight.to_data(), &ref_device);
    let bias_ref = Tensor::<RefBackend, 1>::from_data(bias.to_data(), &ref_device);

    // CubeCL: input in NTHWC format
    let input_nthwc = ncthw_to_nthwc(input_ncthw);

    let options = Conv3dOptions {
        stride: [1, 1, 1],
        padding: [1, 1, 1],
        dilation: [1, 1, 1],
        groups: 1,
        layout: Layout::NTHWC,
    };

    let cubecl_output_nthwc = run_cubecl_conv3d(input_nthwc, weight, Some(bias), options);

    // Output should be NTHWC: [1, 8, 8, 8, 8] -> batch, time, height, width, channels
    assert_eq!(cubecl_output_nthwc.dims(), [1, 8, 8, 8, 8]);

    // Convert back to NCTHW for comparison with reference
    let cubecl_output_ncthw = nthwc_to_ncthw(cubecl_output_nthwc);
    let ref_output =
        reference::conv3d_reference(input_ref, weight_ref, Some(bias_ref), [1, 1, 1], [1, 1, 1]);

    assert_tensors_approx_eq(
        cubecl_output_ncthw,
        ref_output,
        1e-4,
        "NTHWC 3x3x3 same padding (CPU)",
    );
}

#[test]
fn test_cpu_conv3d_nthwc_stride_2() {
    let cpu_device = CpuDevice;
    let ref_device = Default::default();

    let input_ncthw = Tensor::<TestBackend, 5>::random(
        [2, 4, 8, 8, 8],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &cpu_device,
    );
    let weight = Tensor::<TestBackend, 5>::random(
        [8, 4, 3, 3, 3],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &cpu_device,
    );

    let input_ref = Tensor::<RefBackend, 5>::from_data(input_ncthw.to_data(), &ref_device);
    let weight_ref = Tensor::<RefBackend, 5>::from_data(weight.to_data(), &ref_device);

    let input_nthwc = ncthw_to_nthwc(input_ncthw);

    let options = Conv3dOptions {
        stride: [2, 2, 2],
        padding: [1, 1, 1],
        dilation: [1, 1, 1],
        groups: 1,
        layout: Layout::NTHWC,
    };

    let cubecl_output_nthwc = run_cubecl_conv3d(input_nthwc, weight, None, options);

    // Output: [batch=2, time=4, height=4, width=4, channels=8]
    assert_eq!(cubecl_output_nthwc.dims(), [2, 4, 4, 4, 8]);

    let cubecl_output_ncthw = nthwc_to_ncthw(cubecl_output_nthwc);
    let ref_output = reference::conv3d_reference(input_ref, weight_ref, None, [2, 2, 2], [1, 1, 1]);

    assert_tensors_approx_eq(
        cubecl_output_ncthw,
        ref_output,
        1e-4,
        "NTHWC stride 2 (CPU)",
    );
}

// ============================================================================
// Optimized NTHWC Kernel Tests
// ============================================================================

fn run_optimized_conv3d(
    input: Tensor<TestBackend, 5>,
    weight: Tensor<TestBackend, 5>,
    bias: Option<Tensor<TestBackend, 1>>,
    options: Conv3dOptimizedOptions,
) -> Tensor<TestBackend, 5> {
    let input_cube = to_cube_tensor(input);
    let weight_cube = to_cube_tensor(weight);
    let bias_cube = bias.map(to_cube_tensor);

    let output_cube = conv3d_nthwc::<CpuRuntime>(input_cube, weight_cube, bias_cube, options)
        .expect("CubeCL optimized conv3d failed");

    from_cube_tensor(output_cube)
}

/// Convert weight from NCTHW to optimized layout: [out_ch, kernel_t, kernel_h, kernel_w, in_ch]
fn weight_to_optimized<B: Backend>(weight: Tensor<B, 5>) -> Tensor<B, 5> {
    // From: [out_ch, in_ch, kernel_t, kernel_h, kernel_w]
    // To:   [out_ch, kernel_t, kernel_h, kernel_w, in_ch]
    weight.permute([0, 2, 3, 4, 1])
}

#[test]
fn test_cpu_optimized_conv3d_3x3x3_same_padding() {
    let cpu_device = CpuDevice;
    let ref_device = Default::default();

    // Create NCTHW input for reference, NTHWC for optimized kernel
    let input_ncthw = Tensor::<TestBackend, 5>::random(
        [1, 3, 8, 8, 8],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &cpu_device,
    );
    let weight_ncthw = Tensor::<TestBackend, 5>::random(
        [8, 3, 3, 3, 3],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &cpu_device,
    );
    let bias = Tensor::<TestBackend, 1>::random(
        [8],
        burn::tensor::Distribution::Uniform(-0.1, 0.1),
        &cpu_device,
    );

    // Reference computation
    let input_ref = Tensor::<RefBackend, 5>::from_data(input_ncthw.to_data(), &ref_device);
    let weight_ref = Tensor::<RefBackend, 5>::from_data(weight_ncthw.to_data(), &ref_device);
    let bias_ref = Tensor::<RefBackend, 1>::from_data(bias.to_data(), &ref_device);
    let ref_output =
        reference::conv3d_reference(input_ref, weight_ref, Some(bias_ref), [1, 1, 1], [1, 1, 1]);

    // Optimized kernel: NTHWC layout
    let input_nthwc = ncthw_to_nthwc(input_ncthw);
    let weight_opt = weight_to_optimized(weight_ncthw);

    let options = Conv3dOptimizedOptions {
        stride: [1, 1, 1],
        padding: [1, 1, 1],
        dilation: [1, 1, 1],
        groups: 1,
    };

    let opt_output_nthwc = run_optimized_conv3d(input_nthwc, weight_opt, Some(bias), options);

    // Output is NTHWC, convert to NCTHW for comparison
    let opt_output_ncthw = nthwc_to_ncthw(opt_output_nthwc);

    assert_eq!(opt_output_ncthw.dims(), [1, 8, 8, 8, 8]);
    assert_tensors_approx_eq(
        opt_output_ncthw,
        ref_output,
        1e-4,
        "optimized 3x3x3 same padding (CPU)",
    );
}

#[test]
fn test_cpu_optimized_conv3d_no_bias() {
    let cpu_device = CpuDevice;
    let ref_device = Default::default();

    let input_ncthw = Tensor::<TestBackend, 5>::random(
        [1, 2, 6, 6, 6],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &cpu_device,
    );
    let weight_ncthw = Tensor::<TestBackend, 5>::random(
        [4, 2, 3, 3, 3],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &cpu_device,
    );

    let input_ref = Tensor::<RefBackend, 5>::from_data(input_ncthw.to_data(), &ref_device);
    let weight_ref = Tensor::<RefBackend, 5>::from_data(weight_ncthw.to_data(), &ref_device);
    let ref_output = reference::conv3d_reference(input_ref, weight_ref, None, [1, 1, 1], [0, 0, 0]);

    let input_nthwc = ncthw_to_nthwc(input_ncthw);
    let weight_opt = weight_to_optimized(weight_ncthw);

    let options = Conv3dOptimizedOptions {
        stride: [1, 1, 1],
        padding: [0, 0, 0],
        dilation: [1, 1, 1],
        groups: 1,
    };

    let opt_output_nthwc = run_optimized_conv3d(input_nthwc, weight_opt, None, options);
    let opt_output_ncthw = nthwc_to_ncthw(opt_output_nthwc);

    assert_eq!(opt_output_ncthw.dims(), [1, 4, 4, 4, 4]);
    assert_tensors_approx_eq(
        opt_output_ncthw,
        ref_output,
        1e-4,
        "optimized no bias (CPU)",
    );
}

// ============================================================================
// Pool3d Tests
// ============================================================================

/// Reference implementation for avg_pool3d
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
    let output_cube = avg_pool3d::<CpuRuntime>(input_cube, options);
    from_cube_tensor(output_cube)
}

fn run_cubecl_max_pool3d(
    input: Tensor<TestBackend, 5>,
    options: Pool3dOptions,
) -> Tensor<TestBackend, 5> {
    let input_cube = to_cube_tensor(input);
    let output_cube = max_pool3d::<CpuRuntime>(input_cube, options);
    from_cube_tensor(output_cube)
}

#[test]
fn test_cpu_avg_pool3d_2x2x2_no_padding() {
    let cpu_device = CpuDevice;
    let ref_device = Default::default();

    let input = Tensor::<TestBackend, 5>::random(
        [1, 2, 4, 4, 4],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &cpu_device,
    );

    let input_ref = Tensor::<RefBackend, 5>::from_data(input.to_data(), &ref_device);

    let options = Pool3dOptions {
        kernel_size: [2, 2, 2],
        stride: [2, 2, 2],
        padding: [0, 0, 0],
    };

    let cubecl_output = run_cubecl_avg_pool3d(input, options);
    let ref_output =
        pool_reference::avg_pool3d_reference(input_ref, [2, 2, 2], [2, 2, 2], [0, 0, 0]);

    assert_eq!(cubecl_output.dims(), [1, 2, 2, 2, 2]);
    assert_tensors_approx_eq(
        cubecl_output,
        ref_output,
        1e-5,
        "avg_pool3d 2x2x2 no padding (CPU)",
    );
}

#[test]
fn test_cpu_avg_pool3d_with_padding() {
    let cpu_device = CpuDevice;
    let ref_device = Default::default();

    let input = Tensor::<TestBackend, 5>::random(
        [1, 3, 6, 6, 6],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &cpu_device,
    );

    let input_ref = Tensor::<RefBackend, 5>::from_data(input.to_data(), &ref_device);

    let options = Pool3dOptions {
        kernel_size: [3, 3, 3],
        stride: [2, 2, 2],
        padding: [1, 1, 1],
    };

    let cubecl_output = run_cubecl_avg_pool3d(input, options);
    let ref_output =
        pool_reference::avg_pool3d_reference(input_ref, [3, 3, 3], [2, 2, 2], [1, 1, 1]);

    assert_eq!(cubecl_output.dims(), [1, 3, 3, 3, 3]);
    assert_tensors_approx_eq(
        cubecl_output,
        ref_output,
        1e-5,
        "avg_pool3d with padding (CPU)",
    );
}

#[test]
fn test_cpu_max_pool3d_2x2x2_no_padding() {
    let cpu_device = CpuDevice;
    let ref_device = Default::default();

    let input = Tensor::<TestBackend, 5>::random(
        [1, 2, 4, 4, 4],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &cpu_device,
    );

    let input_ref = Tensor::<RefBackend, 5>::from_data(input.to_data(), &ref_device);

    let options = Pool3dOptions {
        kernel_size: [2, 2, 2],
        stride: [2, 2, 2],
        padding: [0, 0, 0],
    };

    let cubecl_output = run_cubecl_max_pool3d(input, options);
    let ref_output =
        pool_reference::max_pool3d_reference(input_ref, [2, 2, 2], [2, 2, 2], [0, 0, 0]);

    assert_eq!(cubecl_output.dims(), [1, 2, 2, 2, 2]);
    assert_tensors_approx_eq(
        cubecl_output,
        ref_output,
        1e-5,
        "max_pool3d 2x2x2 no padding (CPU)",
    );
}

#[test]
fn test_cpu_max_pool3d_with_padding() {
    let cpu_device = CpuDevice;
    let ref_device = Default::default();

    let input = Tensor::<TestBackend, 5>::random(
        [2, 4, 8, 8, 8],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &cpu_device,
    );

    let input_ref = Tensor::<RefBackend, 5>::from_data(input.to_data(), &ref_device);

    let options = Pool3dOptions {
        kernel_size: [3, 3, 3],
        stride: [2, 2, 2],
        padding: [1, 1, 1],
    };

    let cubecl_output = run_cubecl_max_pool3d(input, options);
    let ref_output =
        pool_reference::max_pool3d_reference(input_ref, [3, 3, 3], [2, 2, 2], [1, 1, 1]);

    assert_eq!(cubecl_output.dims(), [2, 4, 4, 4, 4]);
    assert_tensors_approx_eq(
        cubecl_output,
        ref_output,
        1e-5,
        "max_pool3d with padding (CPU)",
    );
}

#[test]
fn test_cpu_avg_pool3d_asymmetric_kernel() {
    let cpu_device = CpuDevice;
    let ref_device = Default::default();

    let input = Tensor::<TestBackend, 5>::random(
        [1, 2, 8, 6, 4],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &cpu_device,
    );

    let input_ref = Tensor::<RefBackend, 5>::from_data(input.to_data(), &ref_device);

    let options = Pool3dOptions {
        kernel_size: [2, 3, 2],
        stride: [2, 2, 2],
        padding: [0, 0, 0],
    };

    let cubecl_output = run_cubecl_avg_pool3d(input, options);
    let ref_output =
        pool_reference::avg_pool3d_reference(input_ref, [2, 3, 2], [2, 2, 2], [0, 0, 0]);

    assert_eq!(cubecl_output.dims(), [1, 2, 4, 2, 2]);
    assert_tensors_approx_eq(
        cubecl_output,
        ref_output,
        1e-5,
        "avg_pool3d asymmetric kernel (CPU)",
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

/// Helper to run flash attention on CPU (causal mode for comparison with reference)
fn run_cpu_flash_attention(
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
#[ignore = "cubek-attention has line_size constraints that fail on CPU backend"]
fn test_cpu_flash_attention_basic() {
    let cpu_device = CpuDevice;
    let ref_device = Default::default();

    // head_dim must be divisible by line_size (typically 4 or 8 on CPU)
    let batch = 1;
    let heads = 4;
    let seq_len = 16;
    let head_dim = 64;

    let q = Tensor::<TestBackend, 4>::random(
        [batch, heads, seq_len, head_dim],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &cpu_device,
    );
    let k = Tensor::<TestBackend, 4>::random(
        [batch, heads, seq_len, head_dim],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &cpu_device,
    );
    let v = Tensor::<TestBackend, 4>::random(
        [batch, heads, seq_len, head_dim],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &cpu_device,
    );

    // Reference on NdArray
    let q_ref = Tensor::<RefBackend, 4>::from_data(q.to_data(), &ref_device);
    let k_ref = Tensor::<RefBackend, 4>::from_data(k.to_data(), &ref_device);
    let v_ref = Tensor::<RefBackend, 4>::from_data(v.to_data(), &ref_device);

    let flash_output = run_cpu_flash_attention(q, k, v);
    let ref_output = attention_reference::causal_attention_reference(q_ref, k_ref, v_ref);

    assert_eq!(flash_output.dims(), ref_output.dims());
    assert_tensors_approx_eq(
        flash_output,
        ref_output,
        1e-3,
        "flash attention basic (CPU)",
    );
}

#[test]
#[ignore = "cubek-attention has line_size constraints that fail on CPU backend"]
fn test_cpu_flash_attention_longer_sequence() {
    let cpu_device = CpuDevice;
    let ref_device = Default::default();

    let batch = 2;
    let heads = 8;
    let seq_len = 64;
    let head_dim = 64;

    let q = Tensor::<TestBackend, 4>::random(
        [batch, heads, seq_len, head_dim],
        burn::tensor::Distribution::Uniform(-0.3, 0.3),
        &cpu_device,
    );
    let k = Tensor::<TestBackend, 4>::random(
        [batch, heads, seq_len, head_dim],
        burn::tensor::Distribution::Uniform(-0.3, 0.3),
        &cpu_device,
    );
    let v = Tensor::<TestBackend, 4>::random(
        [batch, heads, seq_len, head_dim],
        burn::tensor::Distribution::Uniform(-0.3, 0.3),
        &cpu_device,
    );

    let q_ref = Tensor::<RefBackend, 4>::from_data(q.to_data(), &ref_device);
    let k_ref = Tensor::<RefBackend, 4>::from_data(k.to_data(), &ref_device);
    let v_ref = Tensor::<RefBackend, 4>::from_data(v.to_data(), &ref_device);

    let flash_output = run_cpu_flash_attention(q, k, v);
    let ref_output = attention_reference::causal_attention_reference(q_ref, k_ref, v_ref);

    assert_eq!(flash_output.dims(), ref_output.dims());
    assert_tensors_approx_eq(
        flash_output,
        ref_output,
        5e-3,
        "flash attention longer seq (CPU)",
    );
}

#[test]
#[ignore = "cubek-attention has line_size constraints that fail on CPU backend"]
fn test_cpu_flash_attention_small() {
    let cpu_device = CpuDevice;
    let ref_device = Default::default();

    // Small dimensions - head_dim must be divisible by line_size (typically 4 or 8)
    let batch = 1;
    let heads = 2;
    let seq_len = 8;
    let head_dim = 64; // Must be divisible by line_size

    let q = Tensor::<TestBackend, 4>::random(
        [batch, heads, seq_len, head_dim],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &cpu_device,
    );
    let k = Tensor::<TestBackend, 4>::random(
        [batch, heads, seq_len, head_dim],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &cpu_device,
    );
    let v = Tensor::<TestBackend, 4>::random(
        [batch, heads, seq_len, head_dim],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &cpu_device,
    );

    let q_ref = Tensor::<RefBackend, 4>::from_data(q.to_data(), &ref_device);
    let k_ref = Tensor::<RefBackend, 4>::from_data(k.to_data(), &ref_device);
    let v_ref = Tensor::<RefBackend, 4>::from_data(v.to_data(), &ref_device);

    let flash_output = run_cpu_flash_attention(q, k, v);
    let ref_output = attention_reference::causal_attention_reference(q_ref, k_ref, v_ref);

    assert_eq!(flash_output.dims(), ref_output.dims());
    assert_tensors_approx_eq(
        flash_output,
        ref_output,
        1e-4,
        "flash attention small (CPU)",
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
    let output_cube = groupnorm::<CpuRuntime>(input_cube, weight_cube, bias_cube, options);
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
    let output_cube = groupnorm_silu::<CpuRuntime>(input_cube, weight_cube, bias_cube, options);
    from_cube_tensor(output_cube)
}

#[test]
fn test_cpu_groupnorm_basic() {
    let cpu_device = CpuDevice;
    let ref_device = Default::default();

    let batch = 1;
    let channels = 32;
    let height = 8;
    let width = 8;
    let num_groups = 8;

    let input = Tensor::<TestBackend, 4>::random(
        [batch, channels, height, width],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &cpu_device,
    );
    let weight = Tensor::<TestBackend, 1>::random(
        [channels],
        burn::tensor::Distribution::Uniform(0.5, 1.5),
        &cpu_device,
    );
    let bias = Tensor::<TestBackend, 1>::random(
        [channels],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &cpu_device,
    );

    let input_ref = Tensor::<RefBackend, 4>::from_data(input.to_data(), &ref_device);
    let weight_ref = Tensor::<RefBackend, 1>::from_data(weight.to_data(), &ref_device);
    let bias_ref = Tensor::<RefBackend, 1>::from_data(bias.to_data(), &ref_device);

    let options = GroupNormSiLuOptions::with_groups(num_groups);

    let cubecl_output = run_cubecl_groupnorm(input, weight, bias, options);
    let ref_output =
        groupnorm_reference::groupnorm_reference(input_ref, weight_ref, bias_ref, num_groups, 1e-5);

    assert_eq!(cubecl_output.dims(), ref_output.dims());
    // Tolerance accounts for variance calculation differences (biased vs unbiased)
    assert_tensors_approx_eq(cubecl_output, ref_output, 1e-2, "groupnorm basic (CPU)");
}

#[test]
fn test_cpu_groupnorm_silu_basic() {
    let cpu_device = CpuDevice;
    let ref_device = Default::default();

    let batch = 1;
    let channels = 32;
    let height = 8;
    let width = 8;
    let num_groups = 8;

    let input = Tensor::<TestBackend, 4>::random(
        [batch, channels, height, width],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &cpu_device,
    );
    let weight = Tensor::<TestBackend, 1>::random(
        [channels],
        burn::tensor::Distribution::Uniform(0.5, 1.5),
        &cpu_device,
    );
    let bias = Tensor::<TestBackend, 1>::random(
        [channels],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &cpu_device,
    );

    let input_ref = Tensor::<RefBackend, 4>::from_data(input.to_data(), &ref_device);
    let weight_ref = Tensor::<RefBackend, 1>::from_data(weight.to_data(), &ref_device);
    let bias_ref = Tensor::<RefBackend, 1>::from_data(bias.to_data(), &ref_device);

    let options = GroupNormSiLuOptions::with_groups(num_groups);

    let cubecl_output = run_cubecl_groupnorm_silu(input, weight, bias, options);
    let ref_output = groupnorm_reference::groupnorm_silu_reference(
        input_ref, weight_ref, bias_ref, num_groups, 1e-5,
    );

    assert_eq!(cubecl_output.dims(), ref_output.dims());
    // Tolerance accounts for variance calculation differences (biased vs unbiased)
    assert_tensors_approx_eq(
        cubecl_output,
        ref_output,
        1e-2,
        "groupnorm_silu basic (CPU)",
    );
}

#[test]
fn test_cpu_groupnorm_silu_batch() {
    let cpu_device = CpuDevice;
    let ref_device = Default::default();

    let batch = 4;
    let channels = 64;
    let height = 16;
    let width = 16;
    let num_groups = 32;

    let input = Tensor::<TestBackend, 4>::random(
        [batch, channels, height, width],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &cpu_device,
    );
    let weight = Tensor::<TestBackend, 1>::random(
        [channels],
        burn::tensor::Distribution::Uniform(0.5, 1.5),
        &cpu_device,
    );
    let bias = Tensor::<TestBackend, 1>::random(
        [channels],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &cpu_device,
    );

    let input_ref = Tensor::<RefBackend, 4>::from_data(input.to_data(), &ref_device);
    let weight_ref = Tensor::<RefBackend, 1>::from_data(weight.to_data(), &ref_device);
    let bias_ref = Tensor::<RefBackend, 1>::from_data(bias.to_data(), &ref_device);

    let options = GroupNormSiLuOptions::with_groups(num_groups);

    let cubecl_output = run_cubecl_groupnorm_silu(input, weight, bias, options);
    let ref_output = groupnorm_reference::groupnorm_silu_reference(
        input_ref, weight_ref, bias_ref, num_groups, 1e-5,
    );

    assert_eq!(cubecl_output.dims(), ref_output.dims());
    // Tolerance accounts for variance calculation differences (biased vs unbiased)
    assert_tensors_approx_eq(
        cubecl_output,
        ref_output,
        1e-2,
        "groupnorm_silu batch (CPU)",
    );
}

#[test]
fn test_cpu_groupnorm_silu_unit_weight_bias() {
    let cpu_device = CpuDevice;
    let ref_device = Default::default();

    let batch = 2;
    let channels = 16;
    let height = 4;
    let width = 4;
    let num_groups = 4;

    let input = Tensor::<TestBackend, 4>::random(
        [batch, channels, height, width],
        burn::tensor::Distribution::Uniform(-2.0, 2.0),
        &cpu_device,
    );
    // Unit weight and zero bias - should just be normalized + silu
    let weight = Tensor::<TestBackend, 1>::ones([channels], &cpu_device);
    let bias = Tensor::<TestBackend, 1>::zeros([channels], &cpu_device);

    let input_ref = Tensor::<RefBackend, 4>::from_data(input.to_data(), &ref_device);
    let weight_ref = Tensor::<RefBackend, 1>::from_data(weight.to_data(), &ref_device);
    let bias_ref = Tensor::<RefBackend, 1>::from_data(bias.to_data(), &ref_device);

    let options = GroupNormSiLuOptions::with_groups(num_groups);

    let cubecl_output = run_cubecl_groupnorm_silu(input, weight, bias, options);
    let ref_output = groupnorm_reference::groupnorm_silu_reference(
        input_ref, weight_ref, bias_ref, num_groups, 1e-5,
    );

    assert_eq!(cubecl_output.dims(), ref_output.dims());
    // Tolerance accounts for variance calculation differences (biased vs unbiased)
    assert_tensors_approx_eq(
        cubecl_output,
        ref_output,
        2e-2,
        "groupnorm_silu unit weight (CPU)",
    );
}

#[test]
fn test_cpu_groupnorm_silu_single_group() {
    let cpu_device = CpuDevice;
    let ref_device = Default::default();

    // Single group = LayerNorm over all channels
    let batch = 1;
    let channels = 8;
    let height = 4;
    let width = 4;
    let num_groups = 1;

    let input = Tensor::<TestBackend, 4>::random(
        [batch, channels, height, width],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &cpu_device,
    );
    let weight = Tensor::<TestBackend, 1>::random(
        [channels],
        burn::tensor::Distribution::Uniform(0.8, 1.2),
        &cpu_device,
    );
    let bias = Tensor::<TestBackend, 1>::random(
        [channels],
        burn::tensor::Distribution::Uniform(-0.2, 0.2),
        &cpu_device,
    );

    let input_ref = Tensor::<RefBackend, 4>::from_data(input.to_data(), &ref_device);
    let weight_ref = Tensor::<RefBackend, 1>::from_data(weight.to_data(), &ref_device);
    let bias_ref = Tensor::<RefBackend, 1>::from_data(bias.to_data(), &ref_device);

    let options = GroupNormSiLuOptions::with_groups(num_groups);

    let cubecl_output = run_cubecl_groupnorm_silu(input, weight, bias, options);
    let ref_output = groupnorm_reference::groupnorm_silu_reference(
        input_ref, weight_ref, bias_ref, num_groups, 1e-5,
    );

    assert_eq!(cubecl_output.dims(), ref_output.dims());
    // Tolerance accounts for variance calculation differences (biased vs unbiased)
    assert_tensors_approx_eq(
        cubecl_output,
        ref_output,
        1e-2,
        "groupnorm_silu single group (CPU)",
    );
}
