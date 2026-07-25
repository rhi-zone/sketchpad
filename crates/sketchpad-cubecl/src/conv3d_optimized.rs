//! Optimized 3D Convolution kernel using channels-last (NTHWC) layout
//!
//! This kernel uses burn-cubecl patterns for better performance:
//! - Line<E> vectorization for coalesced memory access
//! - FastDivmod for efficient index calculation
//! - Comptime recursive kernel loop for dimension unrolling
//! - Channels-last layout where channels have stride=1

use burn::tensor::Shape;
use burn_cubecl::{
    CubeRuntime, kernel::into_contiguous_aligned, ops::numeric::empty_device_optimized_dtype,
    tensor::CubeTensor,
};
use cubecl::{
    calculate_cube_count_elemwise,
    prelude::*,
    std::{FastDivmod, FastDivmodArgs},
    tensor_line_size_parallel,
};

/// Convolution parameters for one spatial dimension
#[derive(CubeLaunch, CubeType, Clone)]
pub struct ConvParam {
    pub stride: u32,
    pub dilation: u32,
    pub padding: i32,
}

/// Arguments for the Conv3d kernel
#[derive(CubeLaunch, CubeType)]
struct Conv3dArgs {
    conv_params: Sequence<ConvParam>,
    channels_per_group: u32,
}

/// Parameters for the recursive kernel loop
#[derive(CubeType, Clone)]
struct LoopParams {
    out_pos: Sequence<u32>,
    in_shape: Sequence<u32>,
    in_strides: Sequence<u32>,
    kernel_shape: Sequence<u32>,
    kernel_strides: Sequence<u32>,
    conv_params: Sequence<ConvParam>,
    in_c_per_group: u32,
    stride_oc: u32,
}

/// Optimized Conv3d kernel for NTHWC layout
///
/// Uses Line<E> vectorization over the channel dimension.
/// Each thread computes one output position (potentially vectorized over channels).
#[cube(launch_unchecked)]
fn conv3d_nthwc_kernel<E: Numeric>(
    input: &Tensor<Line<E>>,
    weight: &Tensor<Line<E>>,
    bias: &Tensor<Line<E>>,
    output: &mut Tensor<Line<E>>,
    args: Conv3dArgs,
    shape_out: Sequence<FastDivmod>,
    shape_out_c: FastDivmod,
    #[comptime] has_padding: bool,
    #[comptime] has_bias: bool,
    #[define(E)] _dtype: StorageType,
) {
    let num_out = output.len();
    if ABSOLUTE_POS >= num_out {
        terminate!();
    }

    let line_size_out = output.line_size();
    let pos = ABSOLUTE_POS * line_size_out;

    // Weight layout: [out_channels, kernel_t, kernel_h, kernel_w, in_channels/groups]
    let in_c_per_group = weight.shape(weight.rank() - 1);

    // Decompose position into batch, spatial, and channel indices
    let (rem, out_c) = shape_out_c.div_mod(pos);
    let (b, spatial_pos) = div_mod_seq(rem, &shape_out);

    // Grouped convolution support
    let g = out_c / args.channels_per_group;
    let ic_start = in_c_per_group * g;

    // Initialize accumulator with bias or zero
    let mut sum = if has_bias {
        bias[out_c / line_size_out]
    } else {
        Line::empty(line_size_out).fill(E::from_int(0))
    };

    // Input offset for this batch, starting at the group's input channels
    let in_offs = b * input.stride(0) + ic_start;

    let stride_oc = weight.stride(0);

    // Build sequences of shapes and strides for spatial dimensions
    let mut in_shape = Sequence::new();
    let mut in_strides = Sequence::new();
    let mut kernel_shape = Sequence::new();
    let mut kernel_strides = Sequence::new();

    // 3 spatial dimensions: T, H, W (indices 1, 2, 3 in NTHWC layout)
    #[unroll]
    for i in 0..3u32 {
        in_shape.push(input.shape(i + 1));
        in_strides.push(input.stride(i + 1));
        kernel_shape.push(weight.shape(i + 1));
        kernel_strides.push(weight.stride(i + 1));
    }

    let weight_offs = out_c * stride_oc;

    let loop_params = LoopParams {
        out_pos: spatial_pos,
        in_shape,
        in_strides,
        kernel_shape,
        kernel_strides,
        conv_params: args.conv_params,
        in_c_per_group,
        stride_oc,
    };

    // Recursive kernel loop over spatial dimensions
    kernel_loop(
        input,
        weight,
        &mut sum,
        in_offs,
        true,
        weight_offs,
        &loop_params,
        0u32,
        has_padding,
    );

    output[ABSOLUTE_POS] = sum;
}

/// Recursive kernel loop over spatial dimensions
///
/// Uses comptime recursion to unroll the loop over kernel dimensions.
#[cube]
fn kernel_loop<E: Numeric>(
    input: &Tensor<Line<E>>,
    weight: &Tensor<Line<E>>,
    sum: &mut Line<E>,
    in_offs: u32,
    in_bounds: bool,
    weight_offs: u32,
    params: &LoopParams,
    #[comptime] kernel_dim: u32,
    #[comptime] has_padding: bool,
) {
    if comptime![kernel_dim < 3] {
        let out_idx = *params.out_pos.index(kernel_dim);
        let conv = params.conv_params.index(kernel_dim);
        let shape = *params.in_shape.index(kernel_dim);
        let stride = *params.in_strides.index(kernel_dim);
        let k_stride = *params.kernel_strides.index(kernel_dim);

        for pos in 0..*params.kernel_shape.index(kernel_dim) {
            let in_pos = (out_idx * conv.stride + pos * conv.dilation) as i32 - conv.padding;
            let in_offs = in_offs + in_pos as u32 * stride;
            let weight_offs = weight_offs + pos * k_stride;
            let mut in_bounds = in_bounds;

            if has_padding {
                in_bounds &= in_pos >= 0 && (in_pos as u32) < shape;
            }

            kernel_loop(
                input,
                weight,
                sum,
                in_offs,
                in_bounds,
                weight_offs,
                params,
                comptime![kernel_dim + 1],
                has_padding,
            );
        }
    } else {
        // Base case: innermost loop over input channels
        kernel_loop_inner(
            input,
            weight,
            sum,
            in_offs,
            in_bounds,
            weight_offs,
            params.in_c_per_group,
            params.stride_oc,
        );
    }
}

/// Innermost kernel loop: accumulate over input channels
///
/// This is where the actual computation happens, using vectorized access.
#[cube]
fn kernel_loop_inner<E: Numeric>(
    input: &Tensor<Line<E>>,
    weight: &Tensor<Line<E>>,
    sum: &mut Line<E>,
    in_offs: u32,
    in_bounds: bool,
    weight_offs: u32,
    in_c_per_group: u32,
    stride_oc: u32,
) {
    let line_size_in = input.line_size();
    let line_size_out = sum.size();

    if in_bounds {
        for in_c in range_stepped(0, in_c_per_group, line_size_in) {
            let in_pos = in_offs + in_c;
            let mut weight_pos = weight_offs + in_c;

            let val = input[in_pos / line_size_in];

            #[unroll]
            for v in 0..line_size_out {
                let w = weight[weight_pos / line_size_in];
                let prod = val * w;

                #[unroll]
                for i in 0..line_size_in {
                    sum[v] += prod[i];
                }
                weight_pos += stride_oc;
            }
        }
    }
}

/// Divide and modulo by a sequence of FastDivmod values
#[cube]
fn div_mod_seq(pos: u32, shape: &Sequence<FastDivmod>) -> (u32, Sequence<u32>) {
    let rank = comptime![shape.len()];
    let mut offs = pos;
    let mut out = Sequence::new();

    #[unroll]
    for i in 0..rank {
        let dim = comptime![rank - i - 1];
        let (rem, offs_local) = shape.index(dim).div_mod(offs);
        out.push(offs_local);
        offs = rem;
    }

    (offs, out.rev())
}

/// Options for the optimized Conv3d operation
#[derive(Debug, Clone)]
pub struct Conv3dOptimizedOptions {
    pub stride: [usize; 3],
    pub padding: [usize; 3],
    pub dilation: [usize; 3],
    pub groups: usize,
}

impl Default for Conv3dOptimizedOptions {
    fn default() -> Self {
        Self {
            stride: [1, 1, 1],
            padding: [0, 0, 0],
            dilation: [1, 1, 1],
            groups: 1,
        }
    }
}

/// Calculate output size for one dimension
fn calc_out_size(
    in_size: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> usize {
    (in_size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
}

/// Perform optimized 3D convolution for NTHWC layout
///
/// This kernel expects:
/// - Input: [batch, time, height, width, in_channels] (NTHWC)
/// - Weight: [out_channels, kernel_t, kernel_h, kernel_w, in_channels/groups]
/// - Bias: [out_channels] (optional)
/// - Output: [batch, out_t, out_h, out_w, out_channels] (NTHWC)
///
/// The channels dimension must have stride=1 (contiguous).
pub fn conv3d_nthwc<R: CubeRuntime>(
    mut input: CubeTensor<R>,
    mut weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: Conv3dOptimizedOptions,
) -> Result<CubeTensor<R>, LaunchError> {
    let out_dtype = input.dtype;
    let rank = input.shape.num_dims();
    let dim_c = rank - 1; // Channels are last

    // Ensure channels are contiguous (stride=1)
    if input.strides[dim_c] != 1 {
        input = into_contiguous_aligned(input);
    }
    if weight.strides[dim_c] != 1 {
        weight = into_contiguous_aligned(weight);
    }

    let batch_size = input.shape.dims[0];
    let in_t = input.shape.dims[1];
    let in_h = input.shape.dims[2];
    let in_w = input.shape.dims[3];
    let _in_channels = input.shape.dims[4];

    let out_channels = weight.shape.dims[0];
    let kernel_t = weight.shape.dims[1];
    let kernel_h = weight.shape.dims[2];
    let kernel_w = weight.shape.dims[3];

    let channels_per_group = out_channels / options.groups;

    // Calculate output spatial dimensions
    let out_t = calc_out_size(
        in_t,
        kernel_t,
        options.stride[0],
        options.padding[0],
        options.dilation[0],
    );
    let out_h = calc_out_size(
        in_h,
        kernel_h,
        options.stride[1],
        options.padding[1],
        options.dilation[1],
    );
    let out_w = calc_out_size(
        in_w,
        kernel_w,
        options.stride[2],
        options.padding[2],
        options.dilation[2],
    );

    // Output shape: [batch, out_t, out_h, out_w, out_channels]
    let shape_out = Shape::new([batch_size, out_t, out_h, out_w, out_channels]);

    let output = empty_device_optimized_dtype(
        input.client.clone(),
        input.device.clone(),
        shape_out.clone(),
        out_dtype,
    );

    // Calculate optimal line sizes for vectorization
    let mut grouped_out_shape = output.shape.clone();
    grouped_out_shape.dims[dim_c] = channels_per_group;
    let line_size_out = tensor_line_size_parallel(
        R::supported_line_sizes().iter().copied(),
        &grouped_out_shape.dims,
        &output.strides,
        dim_c,
    );

    // Use the weight's channel dimension for input line size
    let line_size_in = tensor_line_size_parallel(
        R::supported_line_sizes().iter().copied(),
        &weight.shape.dims,
        &weight.strides,
        dim_c,
    );

    // Build FastDivmod for output shape (spatial dimensions only)
    let mut shape_out_seq = SequenceArg::new();
    shape_out_seq.push(FastDivmodArgs::new(&input.client, out_t as u32));
    shape_out_seq.push(FastDivmodArgs::new(&input.client, out_h as u32));
    shape_out_seq.push(FastDivmodArgs::new(&input.client, out_w as u32));
    let shape_out_c = FastDivmodArgs::new(&input.client, out_channels as u32);

    // Build convolution parameters
    let mut conv_params = SequenceArg::new();
    for i in 0..3 {
        conv_params.push(ConvParamLaunch::new(
            ScalarArg::new(options.stride[i] as u32),
            ScalarArg::new(options.dilation[i] as u32),
            ScalarArg::new(options.padding[i] as i32),
        ));
    }

    let has_padding = options.padding.iter().any(|&p| p != 0);
    let has_bias = bias.is_some();

    // Create zero bias if none provided
    let bias = bias.unwrap_or_else(|| {
        use burn_cubecl::ops::numeric::zeros_client;
        zeros_client(
            input.client.clone(),
            input.device.clone(),
            Shape::from([out_channels]),
            input.dtype,
        )
    });

    let working_units = output.shape.num_elements() / line_size_out as usize;
    let cube_dim = CubeDim::new(&input.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&input.client, working_units, cube_dim);

    unsafe {
        conv3d_nthwc_kernel::launch_unchecked::<R>(
            &input.client,
            cube_count,
            cube_dim,
            input.as_tensor_arg(line_size_in),
            weight.as_tensor_arg(line_size_in),
            bias.as_tensor_arg(line_size_out),
            output.as_tensor_arg(line_size_out),
            Conv3dArgsLaunch::new(conv_params, ScalarArg::new(channels_per_group as u32)),
            shape_out_seq,
            shape_out_c,
            has_padding,
            has_bias,
            out_dtype.into(),
        )
    }?;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_shape_calculation() {
        // Input: [1, 16, 64, 64, 3] (NTHWC)
        // Kernel: [32, 3, 3, 3, 3]
        // Stride: [1, 1, 1], Padding: [1, 1, 1], Dilation: [1, 1, 1]
        // Expected output: [1, 16, 64, 64, 32]

        let out_t = calc_out_size(16, 3, 1, 1, 1);
        let out_h = calc_out_size(64, 3, 1, 1, 1);
        let out_w = calc_out_size(64, 3, 1, 1, 1);

        assert_eq!(out_t, 16);
        assert_eq!(out_h, 64);
        assert_eq!(out_w, 64);
    }
}
