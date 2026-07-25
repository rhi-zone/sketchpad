//! 3D Pooling kernels for CubeCL
//!
//! Provides AvgPool3d and MaxPool3d for NCTHW layout tensors.

use burn::tensor::Shape;
use burn_cubecl::{CubeRuntime, ops::numeric::empty_device_dtype, tensor::CubeTensor};
use cubecl::prelude::*;
use cubecl::{CubeDim, calculate_cube_count_elemwise};

/// Options for 3D pooling operations
#[derive(Debug, Clone, Copy)]
pub struct Pool3dOptions {
    /// Kernel size [t, h, w]
    pub kernel_size: [usize; 3],
    /// Stride [t, h, w]
    pub stride: [usize; 3],
    /// Padding [t, h, w]
    pub padding: [usize; 3],
}

impl Default for Pool3dOptions {
    fn default() -> Self {
        Self {
            kernel_size: [2, 2, 2],
            stride: [2, 2, 2],
            padding: [0, 0, 0],
        }
    }
}

/// Calculate output size for a single dimension
fn calculate_pool_output_size(
    kernel_size: usize,
    stride: usize,
    padding: usize,
    input_size: usize,
) -> usize {
    (input_size + 2 * padding - kernel_size) / stride + 1
}

/// Pool3d arguments passed to the kernel
#[derive(CubeLaunch, CubeType)]
struct Pool3dArgs {
    stride_t: u32,
    stride_h: u32,
    stride_w: u32,
    pad_t: u32,
    pad_h: u32,
    pad_w: u32,
}

// ============================================================================
// Average Pool3d Kernel
// ============================================================================

#[cube(launch)]
fn avg_pool3d_kernel<E: Numeric>(
    input: &Tensor<E>,
    output: &mut Tensor<E>,
    args: Pool3dArgs,
    #[comptime] kernel_t: u32,
    #[comptime] kernel_h: u32,
    #[comptime] kernel_w: u32,
    #[define(E)] _dtype: StorageType,
) {
    // Output layout: NCTHW
    let out_w = output.shape(4);
    let out_h = output.shape(3);
    let out_t = output.shape(2);
    let channels = output.shape(1);

    let in_w = input.shape(4);
    let in_h = input.shape(3);
    let in_t = input.shape(2);

    // Compute position in output tensor
    let idx = ABSOLUTE_POS;
    if idx >= output.len() {
        terminate!();
    }

    // Decompose linear index to NCTHW coordinates
    let ow = idx % out_w;
    let tmp = idx / out_w;
    let oh = tmp % out_h;
    let tmp = tmp / out_h;
    let ot = tmp % out_t;
    let tmp = tmp / out_t;
    let c = tmp % channels;
    let b = tmp / channels;

    // Input strides
    let in_stride_w = input.stride(4);
    let in_stride_h = input.stride(3);
    let in_stride_t = input.stride(2);
    let in_stride_c = input.stride(1);
    let in_stride_b = input.stride(0);

    // Accumulate sum over kernel window
    let mut sum = E::from_int(0);
    let mut valid_count: u32 = 0;

    for kt in 0..kernel_t {
        // Compute input position (using signed arithmetic for padding)
        let it_signed = (ot * args.stride_t + kt) as i32 - args.pad_t as i32;

        if it_signed >= 0 && (it_signed as u32) < in_t {
            let it = it_signed as u32;

            for kh in 0..kernel_h {
                let ih_signed = (oh * args.stride_h + kh) as i32 - args.pad_h as i32;

                if ih_signed >= 0 && (ih_signed as u32) < in_h {
                    let ih = ih_signed as u32;

                    for kw in 0..kernel_w {
                        let iw_signed = (ow * args.stride_w + kw) as i32 - args.pad_w as i32;

                        if iw_signed >= 0 && (iw_signed as u32) < in_w {
                            let iw = iw_signed as u32;

                            let in_idx = b * in_stride_b
                                + c * in_stride_c
                                + it * in_stride_t
                                + ih * in_stride_h
                                + iw * in_stride_w;

                            sum += input[in_idx];
                            valid_count += 1;
                        }
                    }
                }
            }
        }
    }

    // Divide by count (valid_count is always > 0 for valid pool configs)
    output[idx] = sum / E::cast_from(valid_count)
}

/// Performs 3D average pooling
///
/// Input/output layout: NCTHW (batch, channels, time, height, width)
pub fn avg_pool3d<R: CubeRuntime>(input: CubeTensor<R>, options: Pool3dOptions) -> CubeTensor<R> {
    let [batch, channels, in_t, in_h, in_w] = input.shape.dims();

    let out_t = calculate_pool_output_size(
        options.kernel_size[0],
        options.stride[0],
        options.padding[0],
        in_t,
    );
    let out_h = calculate_pool_output_size(
        options.kernel_size[1],
        options.stride[1],
        options.padding[1],
        in_h,
    );
    let out_w = calculate_pool_output_size(
        options.kernel_size[2],
        options.stride[2],
        options.padding[2],
        in_w,
    );

    let out_shape = Shape::new([batch, channels, out_t, out_h, out_w]);
    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        out_shape.clone(),
        input.dtype,
    );

    let num_elements = out_shape.num_elements();
    let cube_dim = CubeDim::new(&input.client, num_elements);
    let cube_count = calculate_cube_count_elemwise(&input.client, num_elements, cube_dim);

    avg_pool3d_kernel::launch::<R>(
        &input.client,
        cube_count,
        cube_dim,
        input.as_tensor_arg(1),
        output.as_tensor_arg(1),
        Pool3dArgsLaunch::new(
            ScalarArg::new(options.stride[0] as u32),
            ScalarArg::new(options.stride[1] as u32),
            ScalarArg::new(options.stride[2] as u32),
            ScalarArg::new(options.padding[0] as u32),
            ScalarArg::new(options.padding[1] as u32),
            ScalarArg::new(options.padding[2] as u32),
        ),
        options.kernel_size[0] as u32,
        options.kernel_size[1] as u32,
        options.kernel_size[2] as u32,
        input.dtype.into(),
    )
    .expect("avg_pool3d kernel launch failed");

    output
}

// ============================================================================
// Max Pool3d Kernel
// ============================================================================

#[cube(launch)]
fn max_pool3d_kernel<E: Numeric>(
    input: &Tensor<E>,
    output: &mut Tensor<E>,
    args: Pool3dArgs,
    #[comptime] kernel_t: u32,
    #[comptime] kernel_h: u32,
    #[comptime] kernel_w: u32,
    #[define(E)] _dtype: StorageType,
) {
    // Output layout: NCTHW
    let out_w = output.shape(4);
    let out_h = output.shape(3);
    let out_t = output.shape(2);
    let channels = output.shape(1);

    let in_w = input.shape(4);
    let in_h = input.shape(3);
    let in_t = input.shape(2);

    // Compute position in output tensor
    let idx = ABSOLUTE_POS;
    if idx >= output.len() {
        terminate!();
    }

    // Decompose linear index to NCTHW coordinates
    let ow = idx % out_w;
    let tmp = idx / out_w;
    let oh = tmp % out_h;
    let tmp = tmp / out_h;
    let ot = tmp % out_t;
    let tmp = tmp / out_t;
    let c = tmp % channels;
    let b = tmp / channels;

    // Input strides
    let in_stride_w = input.stride(4);
    let in_stride_h = input.stride(3);
    let in_stride_t = input.stride(2);
    let in_stride_c = input.stride(1);
    let in_stride_b = input.stride(0);

    // Track maximum value - initialize with minimum
    let mut max_val = E::min_value();

    for kt in 0..kernel_t {
        // Compute input position (using signed arithmetic for padding)
        let it_signed = (ot * args.stride_t + kt) as i32 - args.pad_t as i32;

        if it_signed >= 0 && (it_signed as u32) < in_t {
            let it = it_signed as u32;

            for kh in 0..kernel_h {
                let ih_signed = (oh * args.stride_h + kh) as i32 - args.pad_h as i32;

                if ih_signed >= 0 && (ih_signed as u32) < in_h {
                    let ih = ih_signed as u32;

                    for kw in 0..kernel_w {
                        let iw_signed = (ow * args.stride_w + kw) as i32 - args.pad_w as i32;

                        if iw_signed >= 0 && (iw_signed as u32) < in_w {
                            let iw = iw_signed as u32;

                            let in_idx = b * in_stride_b
                                + c * in_stride_c
                                + it * in_stride_t
                                + ih * in_stride_h
                                + iw * in_stride_w;

                            let val = input[in_idx];
                            if val > max_val {
                                max_val = val;
                            }
                        }
                    }
                }
            }
        }
    }

    output[idx] = max_val;
}

/// Performs 3D max pooling
///
/// Input/output layout: NCTHW (batch, channels, time, height, width)
pub fn max_pool3d<R: CubeRuntime>(input: CubeTensor<R>, options: Pool3dOptions) -> CubeTensor<R> {
    let [batch, channels, in_t, in_h, in_w] = input.shape.dims();

    let out_t = calculate_pool_output_size(
        options.kernel_size[0],
        options.stride[0],
        options.padding[0],
        in_t,
    );
    let out_h = calculate_pool_output_size(
        options.kernel_size[1],
        options.stride[1],
        options.padding[1],
        in_h,
    );
    let out_w = calculate_pool_output_size(
        options.kernel_size[2],
        options.stride[2],
        options.padding[2],
        in_w,
    );

    let out_shape = Shape::new([batch, channels, out_t, out_h, out_w]);
    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        out_shape.clone(),
        input.dtype,
    );

    let num_elements = out_shape.num_elements();
    let cube_dim = CubeDim::new(&input.client, num_elements);
    let cube_count = calculate_cube_count_elemwise(&input.client, num_elements, cube_dim);

    max_pool3d_kernel::launch::<R>(
        &input.client,
        cube_count,
        cube_dim,
        input.as_tensor_arg(1),
        output.as_tensor_arg(1),
        Pool3dArgsLaunch::new(
            ScalarArg::new(options.stride[0] as u32),
            ScalarArg::new(options.stride[1] as u32),
            ScalarArg::new(options.stride[2] as u32),
            ScalarArg::new(options.padding[0] as u32),
            ScalarArg::new(options.padding[1] as u32),
            ScalarArg::new(options.padding[2] as u32),
        ),
        options.kernel_size[0] as u32,
        options.kernel_size[1] as u32,
        options.kernel_size[2] as u32,
        input.dtype.into(),
    )
    .expect("max_pool3d kernel launch failed");

    output
}
