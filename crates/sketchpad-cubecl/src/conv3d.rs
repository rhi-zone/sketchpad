//! 3D Convolution kernel
//!
//! Direct convolution implementation for 3D inputs (video, volumetric data).
//! Based on burn-cubecl's conv_transpose3d pattern.

use burn::tensor::Shape;
use burn_cubecl::{
    CubeRuntime,
    ops::numeric::{empty_device_dtype, zeros_client},
    tensor::CubeTensor,
};
use cubecl::{calculate_cube_count_elemwise, prelude::*};

/// Convolution arguments passed to the kernel
#[derive(CubeLaunch, CubeType)]
struct ConvArgs {
    stride_t: u32,
    stride_h: u32,
    stride_w: u32,
    dilation_t: u32,
    dilation_h: u32,
    dilation_w: u32,
    padding_t: i32,
    padding_h: i32,
    padding_w: i32,
    groups: u32,
}

/// Memory layout for 5D tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Layout {
    /// Channels first: [batch, channels, time, height, width]
    #[default]
    NCTHW,
    /// Channels last: [batch, time, height, width, channels]
    NTHWC,
}

/// Options for Conv3d operation
#[derive(Debug, Clone)]
pub struct Conv3dOptions {
    pub stride: [usize; 3],
    pub padding: [usize; 3],
    pub dilation: [usize; 3],
    pub groups: usize,
    /// Input/output layout. Weights are always NCTHW (out_ch, in_ch, T, H, W).
    pub layout: Layout,
}

impl Default for Conv3dOptions {
    fn default() -> Self {
        Self {
            stride: [1, 1, 1],
            padding: [0, 0, 0],
            dilation: [1, 1, 1],
            groups: 1,
            layout: Layout::NCTHW,
        }
    }
}

/// Direct 3D convolution kernel
///
/// Layout: NCTHW (batch, channels, time, height, width)
/// Each thread computes one output element.
#[cube(launch)]
fn conv3d_kernel<E: Numeric>(
    input: &Tensor<E>,
    weight: &Tensor<E>,
    bias: &Tensor<E>,
    output: &mut Tensor<E>,
    args: ConvArgs,
    #[define(E)] _dtype: StorageType,
) {
    // Output shape: [batch, out_channels, out_t, out_h, out_w]
    let out_channels = output.shape(1);
    let out_t = output.shape(2);
    let out_h = output.shape(3);
    let out_w = output.shape(4);

    // Weight shape: [out_channels, in_channels/groups, kernel_t, kernel_h, kernel_w]
    let in_c_per_group = weight.shape(1);
    let kernel_t = weight.shape(2);
    let kernel_h = weight.shape(3);
    let kernel_w = weight.shape(4);

    // Decompose ABSOLUTE_POS into output coordinates
    let pos = ABSOLUTE_POS;
    let batch = pos / output.stride(0) % output.shape(0);
    let out_c = pos / output.stride(1) % out_channels;
    let ot = pos / output.stride(2) % out_t;
    let oh = pos / output.stride(3) % out_h;
    let ow = pos / output.stride(4) % out_w;

    // Grouped convolution support
    let group = out_c / (out_channels / args.groups);
    let in_c_start = group * in_c_per_group;
    let in_c_end = in_c_start + in_c_per_group;

    // Initialize accumulator with bias
    let mut sum = bias[out_c];

    // Precompute input base offset for this batch
    let input_batch_offset = batch * input.stride(0);
    let weight_oc_offset = out_c * weight.stride(0);

    // Loop over kernel dimensions
    for kt in 0..kernel_t {
        // Input position in time dimension
        let it = (ot * args.stride_t) as i32 + (kt * args.dilation_t) as i32 - args.padding_t;

        // Bounds check for time
        if it >= 0 && (it as u32) < input.shape(2) {
            let it = it as u32;

            for kh in 0..kernel_h {
                // Input position in height dimension
                let ih =
                    (oh * args.stride_h) as i32 + (kh * args.dilation_h) as i32 - args.padding_h;

                // Bounds check for height
                if ih >= 0 && (ih as u32) < input.shape(3) {
                    let ih = ih as u32;

                    for kw in 0..kernel_w {
                        // Input position in width dimension
                        let iw = (ow * args.stride_w) as i32 + (kw * args.dilation_w) as i32
                            - args.padding_w;

                        // Bounds check for width
                        if iw >= 0 && (iw as u32) < input.shape(4) {
                            let iw = iw as u32;

                            // Accumulate over input channels in this group
                            for ic in in_c_start..in_c_end {
                                let input_idx = input_batch_offset
                                    + ic * input.stride(1)
                                    + it * input.stride(2)
                                    + ih * input.stride(3)
                                    + iw * input.stride(4);

                                let weight_idx = weight_oc_offset
                                    + (ic - in_c_start) * weight.stride(1)
                                    + kt * weight.stride(2)
                                    + kh * weight.stride(3)
                                    + kw * weight.stride(4);

                                sum += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    output[ABSOLUTE_POS] = sum;
}

/// Permute a 5D tensor
///
/// Creates a new tensor with permuted dimensions and strides.
fn permute_5d<R: CubeRuntime>(tensor: CubeTensor<R>, perm: [usize; 5]) -> CubeTensor<R> {
    let old_shape = tensor.shape.dims;
    let old_strides = &tensor.strides;

    let new_shape = [
        old_shape[perm[0]],
        old_shape[perm[1]],
        old_shape[perm[2]],
        old_shape[perm[3]],
        old_shape[perm[4]],
    ];
    let new_strides = vec![
        old_strides[perm[0]],
        old_strides[perm[1]],
        old_strides[perm[2]],
        old_strides[perm[3]],
        old_strides[perm[4]],
    ];

    CubeTensor {
        client: tensor.client,
        handle: tensor.handle,
        device: tensor.device,
        shape: Shape::from(new_shape),
        strides: new_strides,
        dtype: tensor.dtype,
        qparams: tensor.qparams,
    }
}

/// Perform 3D convolution using CubeCL
///
/// # Arguments
/// * `input` - Input tensor. Shape depends on layout:
///   - NCTHW: [batch, in_channels, time, height, width]
///   - NTHWC: [batch, time, height, width, in_channels]
/// * `weight` - Weight tensor [out_channels, in_channels/groups, kernel_t, kernel_h, kernel_w]
///   (always NCTHW layout)
/// * `bias` - Optional bias tensor [out_channels]
/// * `options` - Convolution options (stride, padding, dilation, groups, layout)
///
/// # Returns
/// Output tensor with same layout as input
pub fn conv3d<R: CubeRuntime>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: Conv3dOptions,
) -> Result<CubeTensor<R>, LaunchError> {
    // Convert NTHWC -> NCTHW if needed
    let input_ncthw = match options.layout {
        Layout::NCTHW => input,
        Layout::NTHWC => permute_5d(input, [0, 4, 1, 2, 3]), // NTHWC -> NCTHW
    };

    let [batch_size, _in_channels, in_t, in_h, in_w] = input_ncthw.shape.dims();
    let [out_channels, _, kernel_t, kernel_h, kernel_w] = weight.shape.dims();

    // Calculate output dimensions
    let out_t = (in_t + 2 * options.padding[0] - options.dilation[0] * (kernel_t - 1) - 1)
        / options.stride[0]
        + 1;
    let out_h = (in_h + 2 * options.padding[1] - options.dilation[1] * (kernel_h - 1) - 1)
        / options.stride[1]
        + 1;
    let out_w = (in_w + 2 * options.padding[2] - options.dilation[2] * (kernel_w - 1) - 1)
        / options.stride[2]
        + 1;

    let shape_out = Shape::new([batch_size, out_channels, out_t, out_h, out_w]);

    // Allocate output (always NCTHW internally)
    let output = empty_device_dtype(
        input_ncthw.client.clone(),
        input_ncthw.device.clone(),
        shape_out.clone(),
        input_ncthw.dtype,
    );

    // Handle optional bias - create zeros if not provided
    let bias = match bias {
        Some(b) => b,
        None => zeros_client(
            input_ncthw.client.clone(),
            input_ncthw.device.clone(),
            Shape::from([out_channels]),
            input_ncthw.dtype,
        ),
    };

    // Launch configuration
    let num_elems = shape_out.num_elements();
    let cube_dim = CubeDim::new(&input_ncthw.client, num_elems);
    let cube_count = calculate_cube_count_elemwise(&input_ncthw.client, num_elems, cube_dim);

    // Launch kernel
    conv3d_kernel::launch::<R>(
        &input_ncthw.client,
        cube_count,
        cube_dim,
        input_ncthw.as_tensor_arg(1),
        weight.as_tensor_arg(1),
        bias.as_tensor_arg(1),
        output.as_tensor_arg(1),
        ConvArgsLaunch::new(
            ScalarArg::new(options.stride[0] as u32),
            ScalarArg::new(options.stride[1] as u32),
            ScalarArg::new(options.stride[2] as u32),
            ScalarArg::new(options.dilation[0] as u32),
            ScalarArg::new(options.dilation[1] as u32),
            ScalarArg::new(options.dilation[2] as u32),
            ScalarArg::new(options.padding[0] as i32),
            ScalarArg::new(options.padding[1] as i32),
            ScalarArg::new(options.padding[2] as i32),
            ScalarArg::new(options.groups as u32),
        ),
        input_ncthw.dtype.into(),
    )?;

    // Convert NCTHW -> NTHWC if needed
    let output = match options.layout {
        Layout::NCTHW => output,
        Layout::NTHWC => permute_5d(output, [0, 2, 3, 4, 1]), // NCTHW -> NTHWC
    };

    Ok(output)
}

/// 3D Convolution layer using CubeCL
///
/// A stateful convolution layer that holds weights and bias.
/// Requires a CubeRuntime backend (CUDA or WGPU).
///
/// # Example
///
/// ```ignore
/// use sketchpad_cubecl::{Conv3dLayer, Conv3dOptions};
///
/// // Create layer
/// let layer = Conv3dLayer::<CudaRuntime>::new(
///     weight_tensor,  // [out_ch, in_ch, k_t, k_h, k_w]
///     Some(bias_tensor),  // [out_ch]
///     Conv3dOptions { stride: [1,1,1], padding: [1,1,1], ..Default::default() },
/// );
///
/// // Forward pass
/// let output = layer.forward(input)?;
/// ```
pub struct Conv3dLayer<R: CubeRuntime> {
    /// Weight tensor [out_channels, in_channels, kernel_t, kernel_h, kernel_w]
    pub weight: CubeTensor<R>,
    /// Optional bias [out_channels]
    pub bias: Option<CubeTensor<R>>,
    /// Convolution options
    pub options: Conv3dOptions,
}

impl<R: CubeRuntime> std::fmt::Debug for Conv3dLayer<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Conv3dLayer")
            .field("weight_shape", &self.weight.shape.dims)
            .field("bias", &self.bias.as_ref().map(|b| b.shape.dims.to_vec()))
            .field("options", &self.options)
            .finish()
    }
}

impl<R: CubeRuntime> Conv3dLayer<R> {
    /// Create a new Conv3d layer
    pub fn new(weight: CubeTensor<R>, bias: Option<CubeTensor<R>>, options: Conv3dOptions) -> Self {
        Self {
            weight,
            bias,
            options,
        }
    }

    /// Forward pass
    pub fn forward(&self, input: CubeTensor<R>) -> Result<CubeTensor<R>, LaunchError> {
        conv3d(
            input,
            self.weight.clone(),
            self.bias.clone(),
            self.options.clone(),
        )
    }
}

/// Helper to convert Burn tensor to CubeTensor
///
/// Works with any CubeBackend (CUDA, WGPU).
pub fn to_cube_tensor<R, F, I, U, const D: usize>(
    tensor: burn::tensor::Tensor<burn_cubecl::CubeBackend<R, F, I, U>, D>,
) -> CubeTensor<R>
where
    R: CubeRuntime,
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    U: burn_cubecl::BoolElement,
{
    match tensor.into_primitive() {
        burn::tensor::TensorPrimitive::Float(t) => t,
        _ => panic!("Expected float tensor"),
    }
}

/// Helper to convert CubeTensor back to Burn tensor
pub fn from_cube_tensor<R, F, I, U, const D: usize>(
    tensor: CubeTensor<R>,
) -> burn::tensor::Tensor<burn_cubecl::CubeBackend<R, F, I, U>, D>
where
    R: CubeRuntime,
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    U: burn_cubecl::BoolElement,
{
    burn::tensor::Tensor::from_primitive(burn::tensor::TensorPrimitive::Float(tensor))
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_output_shape_calculation() {
        // Input: [1, 3, 16, 64, 64]
        // Kernel: [32, 3, 3, 3, 3]
        // Stride: [1, 1, 1], Padding: [1, 1, 1], Dilation: [1, 1, 1]
        // Expected output: [1, 32, 16, 64, 64]

        let in_t = 16;
        let in_h = 64;
        let in_w = 64;
        let kernel_t = 3;
        let kernel_h = 3;
        let kernel_w = 3;
        let padding = [1, 1, 1];
        let stride = [1, 1, 1];
        let dilation = [1, 1, 1];

        let out_t = (in_t + 2 * padding[0] - dilation[0] * (kernel_t - 1) - 1) / stride[0] + 1;
        let out_h = (in_h + 2 * padding[1] - dilation[1] * (kernel_h - 1) - 1) / stride[1] + 1;
        let out_w = (in_w + 2 * padding[2] - dilation[2] * (kernel_w - 1) - 1) / stride[2] + 1;

        assert_eq!(out_t, 16);
        assert_eq!(out_h, 64);
        assert_eq!(out_w, 64);
    }
}
