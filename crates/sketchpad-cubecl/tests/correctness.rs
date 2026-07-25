//! Correctness tests comparing CubeCL conv3d and pool3d against reference implementations (WGPU)
//!
//! These tests verify that the CubeCL kernels produce numerically equivalent
//! results to the reference implementations.
//!
//! # Running the tests
//!
//! These tests require a GPU and the `wgpu` feature:
//!
//! ```sh
//! cargo test -p burn-models-cubecl --features wgpu --test correctness -- --ignored
//! ```

#![cfg(feature = "wgpu")]

use burn::prelude::*;
use burn_cubecl::CubeBackend;
use burn_wgpu::{WgpuDevice, WgpuRuntime};
use sketchpad_cubecl::{
    Conv3dOptions, FlashAttentionOptions, GroupNormSiLuOptions, Layout, Pool3dOptions, avg_pool3d,
    conv3d, flash_attention, groupnorm, groupnorm_silu, max_pool3d,
};

// Use CubeBackend directly to avoid FusionTensor wrapper
// The 4th generic is BoolElement (u32 for wgpu)
type TestBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;

/// Reference im2col-based 3D convolution (copied from burn-models-core/src/vae3d.rs)
/// This is the "known good" implementation we're testing against.
mod reference {
    use burn::prelude::*;

    /// Pad a 5D tensor [B, C, T, H, W] with zeros
    fn pad_5d<B: Backend>(x: Tensor<B, 5>, padding: [usize; 3]) -> Tensor<B, 5> {
        let [batch, channels, time, height, width] = x.dims();
        let [p_t, p_h, p_w] = padding;
        let device = x.device();

        let new_t = time + 2 * p_t;
        let new_h = height + 2 * p_h;
        let new_w = width + 2 * p_w;

        // Create padded tensor
        let mut padded = Tensor::zeros([batch, channels, new_t, new_h, new_w], &device);

        // Copy original data to center using slice_assign
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

    /// Extract 3D patches from input (im2col for 3D convolution)
    ///
    /// Input: [batch, in_channels, T, H, W]
    /// Output: [batch, in_channels * k_t * k_h * k_w, out_t * out_h * out_w]
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

        // Collect all patches
        let mut patches = Vec::with_capacity(out_positions);

        for ot in 0..out_t {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let t_start = ot * s_t;
                    let h_start = oh * s_h;
                    let w_start = ow * s_w;

                    // Extract patch [batch, in_ch, k_t, k_h, k_w]
                    let patch = x.clone().slice([
                        0..batch,
                        0..in_ch,
                        t_start..t_start + k_t,
                        h_start..h_start + k_h,
                        w_start..w_start + k_w,
                    ]);

                    // Flatten to [batch, in_ch * k_t * k_h * k_w]
                    let patch_flat = patch.reshape([batch, col_size]);
                    // Add position dimension [batch, col_size, 1]
                    let patch_col = patch_flat.unsqueeze_dim::<3>(2);
                    patches.push(patch_col);
                }
            }
        }

        if patches.is_empty() {
            Tensor::zeros([batch, col_size, 0], &device)
        } else {
            // Concatenate along position dimension: [batch, col_size, out_positions]
            Tensor::cat(patches, 2)
        }
    }

    /// Reference 3D convolution using im2col approach
    ///
    /// # Arguments
    /// * `input` - [batch, in_channels, T, H, W]
    /// * `weight` - [out_channels, in_channels, k_t, k_h, k_w]
    /// * `bias` - [out_channels] (optional)
    /// * `stride` - [stride_t, stride_h, stride_w]
    /// * `padding` - [pad_t, pad_h, pad_w]
    /// * `dilation` - [dilation_t, dilation_h, dilation_w] (must be [1,1,1] for now)
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

        // Reshape weight from [out_ch, in_ch, k_t, k_h, k_w] to [out_ch, in_ch * k_t * k_h * k_w]
        let kernel_elements = k_t * k_h * k_w;
        let weight_2d = weight.reshape([out_ch, in_ch_per_group * kernel_elements]);

        // Pad input if needed
        let x_padded = if padding[0] > 0 || padding[1] > 0 || padding[2] > 0 {
            pad_5d(input, padding)
        } else {
            input
        };
        let [_, _, pad_t, pad_h, pad_w] = x_padded.dims();

        // Calculate output dimensions
        let out_t = (pad_t - k_t) / stride[0] + 1;
        let out_h = (pad_h - k_h) / stride[1] + 1;
        let out_w = (pad_w - k_w) / stride[2] + 1;

        // im2col: extract patches and reshape to columns
        let cols = im2col_3d(x_padded, [k_t, k_h, k_w], stride, [out_t, out_h, out_w]);

        // Matrix multiply: weight @ cols
        // weight_2d is [out_ch, in_ch*k]
        // cols is [batch, in_ch*k, out_positions]
        let weight_expanded = weight_2d.unsqueeze_dim::<3>(0).expand([
            batch,
            out_ch,
            in_ch_per_group * kernel_elements,
        ]);

        // Batched matmul: [batch, out_ch, in_ch*k] @ [batch, in_ch*k, out_positions] = [batch, out_ch, out_positions]
        let out = weight_expanded.matmul(cols);

        // Add bias if present
        let out = if let Some(bias) = bias {
            let bias_expanded = bias.reshape([1, out_ch, 1]);
            out + bias_expanded
        } else {
            out
        };

        // Reshape to [batch, out_ch, out_t, out_h, out_w]
        out.reshape([batch, out_ch, out_t, out_h, out_w])
    }
}

/// Convert Burn tensor to CubeTensor for the CubeCL kernel
fn to_cube_tensor<const D: usize>(
    tensor: Tensor<TestBackend, D>,
) -> burn_wgpu::CubeTensor<WgpuRuntime> {
    // into_primitive() returns TensorPrimitive::Float(CubeTensor<R>)
    match tensor.into_primitive() {
        burn::tensor::TensorPrimitive::Float(t) => t,
        _ => panic!("Expected float tensor"),
    }
}

/// Convert CubeTensor back to Burn tensor
fn from_cube_tensor<const D: usize>(
    tensor: burn_wgpu::CubeTensor<WgpuRuntime>,
) -> Tensor<TestBackend, D> {
    Tensor::from_primitive(burn::tensor::TensorPrimitive::Float(tensor))
}

/// Run CubeCL conv3d kernel
fn run_cubecl_conv3d(
    input: Tensor<TestBackend, 5>,
    weight: Tensor<TestBackend, 5>,
    bias: Option<Tensor<TestBackend, 1>>,
    options: Conv3dOptions,
) -> Tensor<TestBackend, 5> {
    let input_cube = to_cube_tensor(input);
    let weight_cube = to_cube_tensor(weight);
    let bias_cube = bias.map(to_cube_tensor);

    let output_cube = conv3d::<WgpuRuntime>(input_cube, weight_cube, bias_cube, options)
        .expect("CubeCL conv3d failed");

    from_cube_tensor(output_cube)
}

/// Compare two tensors for approximate equality
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
#[ignore = "requires GPU"]
fn test_conv3d_1x1x1_kernel() {
    // Simplest case: 1x1x1 kernel (effectively a 1x1 conv in all dimensions)
    let device = WgpuDevice::default();

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
    assert_tensors_approx_eq(cubecl_output, reference_output, 1e-5, "1x1x1 kernel");
}

#[test]
#[ignore = "requires GPU"]
fn test_conv3d_3x3x3_kernel_no_padding() {
    // 3x3x3 kernel without padding
    let device = WgpuDevice::default();

    let batch = 1;
    let in_ch = 2;
    let out_ch = 4;
    let t = 6;
    let h = 6;
    let w = 6;

    let input = Tensor::<TestBackend, 5>::random(
        [batch, in_ch, t, h, w],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    let weight = Tensor::<TestBackend, 5>::random(
        [out_ch, in_ch, 3, 3, 3],
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

    // Expected output: (6-3)/1 + 1 = 4 in each spatial dimension
    assert_eq!(cubecl_output.dims(), [1, 4, 4, 4, 4]);
    assert_eq!(reference_output.dims(), [1, 4, 4, 4, 4]);
    assert_tensors_approx_eq(cubecl_output, reference_output, 1e-4, "3x3x3 no padding");
}

#[test]
#[ignore = "requires GPU"]
fn test_conv3d_3x3x3_kernel_same_padding() {
    // 3x3x3 kernel with "same" padding
    let device = WgpuDevice::default();

    let batch = 1;
    let in_ch = 3;
    let out_ch = 8;
    let t = 8;
    let h = 8;
    let w = 8;

    let input = Tensor::<TestBackend, 5>::random(
        [batch, in_ch, t, h, w],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    let weight = Tensor::<TestBackend, 5>::random(
        [out_ch, in_ch, 3, 3, 3],
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
        padding: [1, 1, 1], // Same padding
        dilation: [1, 1, 1],
        groups: 1,
        layout: Layout::NCTHW,
    };

    let cubecl_output =
        run_cubecl_conv3d(input.clone(), weight.clone(), Some(bias.clone()), options);

    let reference_output =
        reference::conv3d_reference(input, weight, Some(bias), [1, 1, 1], [1, 1, 1], [1, 1, 1]);

    // With same padding, output shape equals input shape
    assert_eq!(cubecl_output.dims(), [1, 8, 8, 8, 8]);
    assert_eq!(reference_output.dims(), [1, 8, 8, 8, 8]);
    assert_tensors_approx_eq(cubecl_output, reference_output, 1e-4, "3x3x3 same padding");
}

#[test]
#[ignore = "requires GPU"]
fn test_conv3d_stride_2() {
    // Test with stride 2 (downsampling)
    let device = WgpuDevice::default();

    let batch = 1;
    let in_ch = 2;
    let out_ch = 4;
    let t = 8;
    let h = 8;
    let w = 8;

    let input = Tensor::<TestBackend, 5>::random(
        [batch, in_ch, t, h, w],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    let weight = Tensor::<TestBackend, 5>::random(
        [out_ch, in_ch, 3, 3, 3],
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

    // Expected output: (8 + 2*1 - 3) / 2 + 1 = 4 in each dimension
    assert_eq!(cubecl_output.dims(), [1, 4, 4, 4, 4]);
    assert_eq!(reference_output.dims(), [1, 4, 4, 4, 4]);
    assert_tensors_approx_eq(cubecl_output, reference_output, 1e-4, "stride 2");
}

#[test]
#[ignore = "requires GPU"]
fn test_conv3d_batch_size_2() {
    // Test with batch size > 1
    let device = WgpuDevice::default();

    let batch = 2;
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
        [out_ch, in_ch, 3, 3, 3],
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
    assert_eq!(reference_output.dims(), [2, 3, 4, 4, 4]);
    assert_tensors_approx_eq(cubecl_output, reference_output, 1e-4, "batch size 2");
}

#[test]
#[ignore = "requires GPU"]
fn test_conv3d_asymmetric() {
    // Test with asymmetric dimensions
    let device = WgpuDevice::default();

    let batch = 1;
    let in_ch = 3;
    let out_ch = 4;
    let t = 4; // Time dimension smaller
    let h = 8;
    let w = 6;

    let input = Tensor::<TestBackend, 5>::random(
        [batch, in_ch, t, h, w],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    // Asymmetric kernel
    let weight = Tensor::<TestBackend, 5>::random(
        [out_ch, in_ch, 1, 3, 3], // 1x3x3 kernel (no temporal extent)
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

    // Expected: T stays 4, H and W stay 8 and 6 with same padding
    assert_eq!(cubecl_output.dims(), [1, 4, 4, 8, 6]);
    assert_eq!(reference_output.dims(), [1, 4, 4, 8, 6]);
    assert_tensors_approx_eq(cubecl_output, reference_output, 1e-4, "asymmetric");
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

fn run_wgpu_avg_pool3d(
    input: Tensor<TestBackend, 5>,
    options: Pool3dOptions,
) -> Tensor<TestBackend, 5> {
    let input_cube = to_cube_tensor(input);
    let output_cube = avg_pool3d::<WgpuRuntime>(input_cube, options);
    from_cube_tensor(output_cube)
}

fn run_wgpu_max_pool3d(
    input: Tensor<TestBackend, 5>,
    options: Pool3dOptions,
) -> Tensor<TestBackend, 5> {
    let input_cube = to_cube_tensor(input);
    let output_cube = max_pool3d::<WgpuRuntime>(input_cube, options);
    from_cube_tensor(output_cube)
}

#[test]
#[ignore = "requires GPU"]
fn test_wgpu_avg_pool3d_2x2x2() {
    let device = WgpuDevice::default();

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

    let wgpu_output = run_wgpu_avg_pool3d(input.clone(), options);
    let ref_output = pool_reference::avg_pool3d_reference(input, [2, 2, 2], [2, 2, 2], [0, 0, 0]);

    assert_eq!(wgpu_output.dims(), [1, 2, 2, 2, 2]);
    assert_tensors_approx_eq(wgpu_output, ref_output, 1e-5, "avg_pool3d 2x2x2 (WGPU)");
}

#[test]
#[ignore = "requires GPU"]
fn test_wgpu_avg_pool3d_with_padding() {
    let device = WgpuDevice::default();

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

    let wgpu_output = run_wgpu_avg_pool3d(input.clone(), options);
    let ref_output = pool_reference::avg_pool3d_reference(input, [3, 3, 3], [2, 2, 2], [1, 1, 1]);

    assert_eq!(wgpu_output.dims(), [1, 3, 3, 3, 3]);
    assert_tensors_approx_eq(
        wgpu_output,
        ref_output,
        1e-5,
        "avg_pool3d with padding (WGPU)",
    );
}

#[test]
#[ignore = "requires GPU"]
fn test_wgpu_max_pool3d_2x2x2() {
    let device = WgpuDevice::default();

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

    let wgpu_output = run_wgpu_max_pool3d(input.clone(), options);
    let ref_output = pool_reference::max_pool3d_reference(input, [2, 2, 2], [2, 2, 2], [0, 0, 0]);

    assert_eq!(wgpu_output.dims(), [1, 2, 2, 2, 2]);
    assert_tensors_approx_eq(wgpu_output, ref_output, 1e-5, "max_pool3d 2x2x2 (WGPU)");
}

#[test]
#[ignore = "requires GPU"]
fn test_wgpu_max_pool3d_with_padding() {
    let device = WgpuDevice::default();

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

    let wgpu_output = run_wgpu_max_pool3d(input.clone(), options);
    let ref_output = pool_reference::max_pool3d_reference(input, [3, 3, 3], [2, 2, 2], [1, 1, 1]);

    assert_eq!(wgpu_output.dims(), [2, 4, 4, 4, 4]);
    assert_tensors_approx_eq(
        wgpu_output,
        ref_output,
        1e-5,
        "max_pool3d with padding (WGPU)",
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
        // mask[i,j] = 0 if j <= i, else -inf
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

        // Broadcast mask to [batch, heads, seq_q, seq_k]
        let mask = mask.unsqueeze::<4>().expand([batch, heads, seq_q, seq_k]);

        // Apply mask and softmax
        let scores = scores + mask;
        let attn = burn::tensor::activation::softmax(scores, 3);

        // Attention @ V
        attn.matmul(v)
    }
}

/// Helper to run flash attention (causal mode for comparison with reference)
fn run_wgpu_flash_attention(
    q: Tensor<TestBackend, 4>,
    k: Tensor<TestBackend, 4>,
    v: Tensor<TestBackend, 4>,
) -> Tensor<TestBackend, 4> {
    // CubeBackend tensors are already CubeTensors internally
    let q_cube = q.into_primitive().tensor();
    let k_cube = k.into_primitive().tensor();
    let v_cube = v.into_primitive().tensor();

    // Use causal mode to match reference implementation
    let output = flash_attention(q_cube, k_cube, v_cube, FlashAttentionOptions::causal())
        .expect("flash attention failed");

    // Convert back to Burn Tensor
    Tensor::from_primitive(burn::tensor::TensorPrimitive::Float(output))
}

#[test]
#[ignore = "requires GPU"]
fn test_wgpu_flash_attention_basic() {
    let device = WgpuDevice::default();

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

    let flash_output = run_wgpu_flash_attention(q.clone(), k.clone(), v.clone());
    let ref_output = attention_reference::causal_attention_reference(q, k, v);

    assert_eq!(flash_output.dims(), ref_output.dims());
    assert_tensors_approx_eq(
        flash_output,
        ref_output,
        1e-3,
        "flash attention basic (WGPU)",
    );
}

#[test]
#[ignore = "requires GPU"]
fn test_wgpu_flash_attention_longer_sequence() {
    let device = WgpuDevice::default();

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

    let flash_output = run_wgpu_flash_attention(q.clone(), k.clone(), v.clone());
    let ref_output = attention_reference::causal_attention_reference(q, k, v);

    assert_eq!(flash_output.dims(), ref_output.dims());
    // Longer sequences may accumulate more numerical error
    assert_tensors_approx_eq(
        flash_output,
        ref_output,
        5e-3,
        "flash attention longer seq (WGPU)",
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

fn run_wgpu_groupnorm(
    input: Tensor<TestBackend, 4>,
    weight: Tensor<TestBackend, 1>,
    bias: Tensor<TestBackend, 1>,
    options: GroupNormSiLuOptions,
) -> Tensor<TestBackend, 4> {
    let input_cube = to_cube_tensor(input);
    let weight_cube = to_cube_tensor(weight);
    let bias_cube = to_cube_tensor(bias);
    let output_cube = groupnorm::<WgpuRuntime>(input_cube, weight_cube, bias_cube, options);
    from_cube_tensor(output_cube)
}

fn run_wgpu_groupnorm_silu(
    input: Tensor<TestBackend, 4>,
    weight: Tensor<TestBackend, 1>,
    bias: Tensor<TestBackend, 1>,
    options: GroupNormSiLuOptions,
) -> Tensor<TestBackend, 4> {
    let input_cube = to_cube_tensor(input);
    let weight_cube = to_cube_tensor(weight);
    let bias_cube = to_cube_tensor(bias);
    let output_cube = groupnorm_silu::<WgpuRuntime>(input_cube, weight_cube, bias_cube, options);
    from_cube_tensor(output_cube)
}

#[test]
#[ignore = "requires GPU"]
fn test_wgpu_groupnorm_basic() {
    let device = WgpuDevice::default();

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

    let cubecl_output = run_wgpu_groupnorm(input.clone(), weight.clone(), bias.clone(), options);
    let ref_output =
        groupnorm_reference::groupnorm_reference(input, weight, bias, num_groups, 1e-5);

    assert_eq!(cubecl_output.dims(), ref_output.dims());
    // Tolerance accounts for variance calculation differences (biased vs unbiased)
    assert_tensors_approx_eq(cubecl_output, ref_output, 1e-2, "groupnorm basic (WGPU)");
}

#[test]
#[ignore = "requires GPU"]
fn test_wgpu_groupnorm_silu_basic() {
    let device = WgpuDevice::default();

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
        run_wgpu_groupnorm_silu(input.clone(), weight.clone(), bias.clone(), options);
    let ref_output =
        groupnorm_reference::groupnorm_silu_reference(input, weight, bias, num_groups, 1e-5);

    assert_eq!(cubecl_output.dims(), ref_output.dims());
    // Tolerance accounts for variance calculation differences (biased vs unbiased)
    assert_tensors_approx_eq(
        cubecl_output,
        ref_output,
        1e-2,
        "groupnorm_silu basic (WGPU)",
    );
}

#[test]
#[ignore = "requires GPU"]
fn test_wgpu_groupnorm_silu_batch() {
    let device = WgpuDevice::default();

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
        run_wgpu_groupnorm_silu(input.clone(), weight.clone(), bias.clone(), options);
    let ref_output =
        groupnorm_reference::groupnorm_silu_reference(input, weight, bias, num_groups, 1e-5);

    assert_eq!(cubecl_output.dims(), ref_output.dims());
    // Tolerance accounts for variance calculation differences (biased vs unbiased)
    assert_tensors_approx_eq(
        cubecl_output,
        ref_output,
        1e-2,
        "groupnorm_silu batch (WGPU)",
    );
}

// =============================================================================
// UNet CubeCL Integration Tests
// =============================================================================

#[test]
#[ignore = "requires GPU"]
fn test_wgpu_resblock_cubecl_shape() {
    use sketchpad_unet::cubecl::ResBlockCubeCL;

    let device = WgpuDevice::default();

    let in_channels = 256;
    let out_channels = 512;
    let time_emb_dim = 1024;

    let block =
        ResBlockCubeCL::<WgpuRuntime>::new(in_channels, out_channels, time_emb_dim, &device);

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
    println!("ResBlockCubeCL output shape: {:?}", output.dims());
}

#[test]
#[ignore = "requires GPU"]
fn test_wgpu_resblock_cubecl_same_channels() {
    use sketchpad_unet::cubecl::ResBlockCubeCL;

    let device = WgpuDevice::default();

    // Test with same in/out channels (no skip conv)
    let channels = 256;
    let time_emb_dim = 512;

    let block = ResBlockCubeCL::<WgpuRuntime>::new(channels, channels, time_emb_dim, &device);

    let batch = 1;
    let height = 16;
    let width = 16;

    let input = Tensor::<TestBackend, 4>::random(
        [batch, channels, height, width],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );
    let time_emb = Tensor::<TestBackend, 2>::random(
        [batch, time_emb_dim],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    let output = block.forward(input, time_emb);

    assert_eq!(output.dims(), [batch, channels, height, width]);
    println!(
        "ResBlockCubeCL same channels output shape: {:?}",
        output.dims()
    );
}

#[test]
#[ignore = "requires GPU"]
fn test_wgpu_crossattention_cubecl_self_attention() {
    use sketchpad_unet::cubecl::CrossAttentionCubeCL;

    let device = WgpuDevice::default();

    let dim = 320;
    let num_heads = 8;
    let head_dim = 40;

    let attn = CrossAttentionCubeCL::<WgpuRuntime>::new(dim, num_heads, head_dim, None, &device);

    let batch = 2;
    let seq_len = 64;

    let input = Tensor::<TestBackend, 3>::random(
        [batch, seq_len, dim],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    // Self-attention (context = None)
    let output = attn.forward(input.clone(), None);

    assert_eq!(output.dims(), [batch, seq_len, dim]);
    println!(
        "CrossAttentionCubeCL self-attn output shape: {:?}",
        output.dims()
    );
}

#[test]
#[ignore = "requires GPU"]
fn test_wgpu_crossattention_cubecl_cross_attention() {
    use sketchpad_unet::cubecl::CrossAttentionCubeCL;

    let device = WgpuDevice::default();

    let query_dim = 320;
    let context_dim = 768; // CLIP text embedding dim
    let num_heads = 8;
    let head_dim = 40;

    let attn = CrossAttentionCubeCL::<WgpuRuntime>::new(
        query_dim,
        num_heads,
        head_dim,
        Some(context_dim),
        &device,
    );

    let batch = 2;
    let seq_len = 64; // image tokens
    let ctx_len = 77; // text tokens

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
        "CrossAttentionCubeCL cross-attn output shape: {:?}",
        output.dims()
    );
}

// =============================================================================
// 3D VAE CubeCL Integration Tests
// =============================================================================

#[test]
#[ignore = "requires GPU"]
fn test_wgpu_vae3d_resblock_shape() {
    use sketchpad_core::vae3d::cubecl::ResBlock3dCubeCL;

    let device = WgpuDevice::default();

    let in_channels = 64;
    let out_channels = 128;

    let block = ResBlock3dCubeCL::<WgpuRuntime>::new(in_channels, out_channels, &device);

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
    println!("ResBlock3dCubeCL output shape: {:?}", output.dims());
}

#[test]
#[ignore = "requires GPU"]
fn test_wgpu_vae3d_downsample() {
    use sketchpad_core::vae3d::cubecl::Downsample3dCubeCL;

    let device = WgpuDevice::default();

    let channels = 128;
    let temporal_stride = 2;
    let spatial_stride = 2;

    let down =
        Downsample3dCubeCL::<WgpuRuntime>::new(channels, temporal_stride, spatial_stride, &device);

    let batch = 1;
    let time = 8;
    let height = 32;
    let width = 32;

    let input = Tensor::<TestBackend, 5>::random(
        [batch, channels, time, height, width],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    let cube_input = sketchpad_cubecl::tensor_to_cube(input);
    let output = down.forward(cube_input);
    let output: Tensor<TestBackend, 5> = sketchpad_cubecl::cube_to_tensor(output);

    // Strided conv with padding=1, kernel=3, stride=2: out = (in + 2*1 - 3)/2 + 1 = in/2
    let expected_time = time / temporal_stride;
    let expected_height = height / spatial_stride;
    let expected_width = width / spatial_stride;

    assert_eq!(
        output.dims(),
        [
            batch,
            channels,
            expected_time,
            expected_height,
            expected_width
        ]
    );
    println!("Downsample3dCubeCL output shape: {:?}", output.dims());
}

#[test]
#[ignore = "requires GPU"]
fn test_wgpu_vae3d_upsample() {
    use sketchpad_core::vae3d::cubecl::Upsample3dCubeCL;

    let device = WgpuDevice::default();

    let channels = 128;
    let temporal_factor = 2;
    let spatial_factor = 2;

    let up =
        Upsample3dCubeCL::<WgpuRuntime>::new(channels, temporal_factor, spatial_factor, &device);

    let batch = 1;
    let time = 4;
    let height = 16;
    let width = 16;

    let input = Tensor::<TestBackend, 5>::random(
        [batch, channels, time, height, width],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    let cube_input = sketchpad_cubecl::tensor_to_cube(input);
    let output = up.forward(cube_input);
    let output: Tensor<TestBackend, 5> = sketchpad_cubecl::cube_to_tensor(output);

    // Upsample then conv (same padding): out = in * factor
    let expected_time = time * temporal_factor;
    let expected_height = height * spatial_factor;
    let expected_width = width * spatial_factor;

    assert_eq!(
        output.dims(),
        [
            batch,
            channels,
            expected_time,
            expected_height,
            expected_width
        ]
    );
    println!("Upsample3dCubeCL output shape: {:?}", output.dims());
}

#[test]
#[ignore = "requires GPU - slow, allocates large model"]
fn test_wgpu_vae3d_encoder_tiny() {
    use sketchpad_core::vae3d::{Vae3dConfig, cubecl::Vae3dEncoderCubeCL};

    let device = WgpuDevice::default();

    // Tiny config for testing
    let config = Vae3dConfig {
        in_channels: 3,
        latent_channels: 4,
        base_channels: 32,         // Small for testing
        channel_mults: vec![1, 2], // Only 2 levels
        temporal_compression: 2,
        spatial_compression: 4,
        num_res_blocks: 1,
    };

    let encoder = Vae3dEncoderCubeCL::<WgpuRuntime>::new(config.clone(), &device);

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

    // Output should be latent_channels * 2 (mean + logvar)
    println!("Vae3dEncoderCubeCL mean shape: {:?}", output.mean.dims());
    println!(
        "Vae3dEncoderCubeCL logvar shape: {:?}",
        output.logvar.dims()
    );

    assert_eq!(output.mean.dims()[0], batch);
    assert_eq!(output.mean.dims()[1], config.latent_channels);
}

#[test]
#[ignore = "requires GPU - slow, allocates large model"]
fn test_wgpu_vae3d_decoder_tiny() {
    use sketchpad_core::vae3d::{Vae3dConfig, cubecl::Vae3dDecoderCubeCL};

    let device = WgpuDevice::default();

    // Tiny config for testing
    let config = Vae3dConfig {
        in_channels: 3,
        latent_channels: 4,
        base_channels: 32,
        channel_mults: vec![1, 2],
        temporal_compression: 2,
        spatial_compression: 4,
        num_res_blocks: 1,
    };

    let decoder = Vae3dDecoderCubeCL::<WgpuRuntime>::new(config.clone(), &device);

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

    println!("Vae3dDecoderCubeCL output shape: {:?}", output.dims());

    assert_eq!(output.dims()[0], batch);
    assert_eq!(output.dims()[1], config.in_channels);
}
