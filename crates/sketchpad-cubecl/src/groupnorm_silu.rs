//! Fused GroupNorm + SiLU kernel
//!
//! Combines group normalization and SiLU activation into a single operation,
//! avoiding intermediate tensor allocation. This is a common pattern in UNet:
//! `let h = silu(self.norm.forward(x));`
//!
//! # Implementation
//!
//! Uses a two-phase approach:
//! 1. Compute mean and variance per (batch, group) pair
//! 2. Normalize, apply affine transform, apply SiLU
//!
//! The second phase fuses normalize + scale + bias + silu into one kernel.

use burn::tensor::Shape;
use burn_cubecl::{CubeRuntime, ops::numeric::empty_device_dtype, tensor::CubeTensor};
use cubecl::prelude::*;
use cubecl::{CubeDim, calculate_cube_count_elemwise};

/// Options for GroupNorm + SiLU
#[derive(Debug, Clone, Copy)]
pub struct GroupNormSiLuOptions {
    /// Number of groups (typically 32)
    pub num_groups: usize,
    /// Epsilon for numerical stability (typically 1e-5)
    pub eps: f32,
}

impl Default for GroupNormSiLuOptions {
    fn default() -> Self {
        Self {
            num_groups: 32,
            eps: 1e-5,
        }
    }
}

impl GroupNormSiLuOptions {
    /// Create options with custom number of groups
    pub fn with_groups(num_groups: usize) -> Self {
        Self {
            num_groups,
            eps: 1e-5,
        }
    }
}

// ============================================================================
// Phase 1: Compute mean and variance per group
// ============================================================================

/// Arguments for the stats kernel
#[derive(CubeLaunch, CubeType)]
struct GroupStatsArgs {
    spatial_size: u32, // H * W (unused but kept for clarity)
    num_groups: u32,
    num_channels: u32,
}

/// Kernel to compute mean and variance per (batch, group)
///
/// One thread per (batch, group) pair. Each thread loops over all elements
/// in its group to compute statistics.
#[cube(launch)]
fn compute_group_stats_kernel<E: Float>(
    input: &Tensor<E>,
    mean_out: &mut Tensor<E>,
    var_out: &mut Tensor<E>,
    args: GroupStatsArgs,
    #[define(E)] _dtype: StorageType,
) {
    // mean_out and var_out have shape [batch, num_groups]
    let idx = ABSOLUTE_POS;
    let total_groups = mean_out.len();

    if idx >= total_groups {
        terminate!();
    }

    // Decompose idx to (batch, group)
    let g = idx % args.num_groups;
    let b = idx / args.num_groups;

    // Input layout: [B, C, H, W]
    let channels_per_group = args.num_channels / args.num_groups;
    let c_start = g * channels_per_group;

    // Input strides (NCHW layout)
    let in_stride_w = input.stride(3);
    let in_stride_h = input.stride(2);
    let in_stride_c = input.stride(1);
    let in_stride_b = input.stride(0);

    let height = input.shape(2);
    let width = input.shape(3);

    // Compute mean: sum all elements in this group
    let mut sum = E::from_int(0);
    let count = channels_per_group * args.spatial_size;

    for c_local in 0..channels_per_group {
        let c = c_start + c_local;
        for h in 0..height {
            for w in 0..width {
                let in_idx = b * in_stride_b + c * in_stride_c + h * in_stride_h + w * in_stride_w;
                sum += input[in_idx];
            }
        }
    }

    let mean = sum / E::cast_from(count);
    mean_out[idx] = mean;

    // Compute variance: sum of squared differences from mean
    let mut var_sum = E::from_int(0);

    for c_local in 0..channels_per_group {
        let c = c_start + c_local;
        for h in 0..height {
            for w in 0..width {
                let in_idx = b * in_stride_b + c * in_stride_c + h * in_stride_h + w * in_stride_w;
                let diff = input[in_idx] - mean;
                var_sum += diff * diff;
            }
        }
    }

    var_out[idx] = var_sum / E::cast_from(count);
}

// ============================================================================
// Phase 2: Normalize + Affine + SiLU (fused)
// ============================================================================

/// Arguments for the fused norm+silu kernel
#[derive(CubeLaunch, CubeType)]
struct NormSiLuArgs {
    num_groups: u32,
    channels_per_group: u32,
    eps: f32,
}

/// Fused kernel: normalize, apply weight/bias, apply SiLU
///
/// One thread per output element. Each thread:
/// 1. Looks up its group's mean/variance
/// 2. Normalizes the input value
/// 3. Applies weight and bias (per-channel)
/// 4. Applies SiLU activation: x * sigmoid(x)
#[cube(launch)]
fn norm_silu_kernel<E: Float>(
    input: &Tensor<E>,
    mean: &Tensor<E>,
    var: &Tensor<E>,
    weight: &Tensor<E>,
    bias: &Tensor<E>,
    output: &mut Tensor<E>,
    args: NormSiLuArgs,
    #[define(E)] _dtype: StorageType,
) {
    let idx = ABSOLUTE_POS;
    if idx >= output.len() {
        terminate!();
    }

    // Output layout: [B, C, H, W]
    let out_w = output.shape(3);
    let out_h = output.shape(2);
    let out_c = output.shape(1);

    // Decompose linear index to NCHW coordinates
    let w = idx % out_w;
    let tmp = idx / out_w;
    let h = tmp % out_h;
    let tmp = tmp / out_h;
    let c = tmp % out_c;
    let b = tmp / out_c;

    // Determine which group this channel belongs to
    let g = c / args.channels_per_group;

    // Look up mean and variance for this (batch, group)
    let stats_idx = b * args.num_groups + g;
    let group_mean = mean[stats_idx];
    let group_var = var[stats_idx];

    // Input strides (for non-contiguous support)
    let in_stride_w = input.stride(3);
    let in_stride_h = input.stride(2);
    let in_stride_c = input.stride(1);
    let in_stride_b = input.stride(0);
    let in_idx = b * in_stride_b + c * in_stride_c + h * in_stride_h + w * in_stride_w;

    // 1. Normalize: (x - mean) / sqrt(var + eps)
    let x = input[in_idx];
    let eps = E::cast_from(args.eps);
    let std = E::sqrt(group_var + eps);
    let normalized = (x - group_mean) / std;

    // 2. Apply affine: weight[c] * normalized + bias[c]
    let gamma = weight[c];
    let beta = bias[c];
    let affine = gamma * normalized + beta;

    // 3. Apply SiLU: x * sigmoid(x)
    let neg_affine = E::from_int(0) - affine;
    let sigmoid = E::from_int(1) / (E::from_int(1) + E::exp(neg_affine));
    let silu = affine * sigmoid;

    output[idx] = silu;
}

/// Fused GroupNorm + SiLU operation
///
/// Computes: `silu(groupnorm(x, weight, bias))`
///
/// # Arguments
///
/// * `input` - Input tensor [batch, channels, height, width]
/// * `weight` - Scale parameter (gamma) [channels]
/// * `bias` - Bias parameter (beta) [channels]
/// * `options` - GroupNorm options (num_groups, eps)
///
/// # Returns
///
/// Output tensor with same shape as input
///
/// # Panics
///
/// - If channels is not divisible by num_groups
/// - If weight/bias length doesn't match channels
pub fn groupnorm_silu<R: CubeRuntime>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: CubeTensor<R>,
    options: GroupNormSiLuOptions,
) -> CubeTensor<R> {
    let [batch, channels, height, width] = input.shape.dims();

    assert!(
        channels % options.num_groups == 0,
        "channels ({}) must be divisible by num_groups ({})",
        channels,
        options.num_groups
    );

    let channels_per_group = channels / options.num_groups;
    let spatial_size = height * width;

    // Phase 1: Compute mean and variance per (batch, group)
    let stats_shape = Shape::new([batch, options.num_groups]);
    let mean = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        stats_shape.clone(),
        input.dtype,
    );
    let var = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        stats_shape.clone(),
        input.dtype,
    );

    let num_stats = batch * options.num_groups;
    let stats_cube_dim = CubeDim::new(&input.client, num_stats);
    let stats_cube_count = calculate_cube_count_elemwise(&input.client, num_stats, stats_cube_dim);

    compute_group_stats_kernel::launch::<R>(
        &input.client,
        stats_cube_count,
        stats_cube_dim,
        input.as_tensor_arg(1),
        mean.as_tensor_arg(1),
        var.as_tensor_arg(1),
        GroupStatsArgsLaunch::new(
            ScalarArg::new(spatial_size as u32),
            ScalarArg::new(options.num_groups as u32),
            ScalarArg::new(channels as u32),
        ),
        input.dtype.into(),
    )
    .expect("compute_group_stats kernel launch failed");

    // Phase 2: Normalize + Affine + SiLU
    let out_shape = Shape::new([batch, channels, height, width]);
    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        out_shape.clone(),
        input.dtype,
    );

    let num_elements = out_shape.num_elements();
    let cube_dim = CubeDim::new(&input.client, num_elements);
    let cube_count = calculate_cube_count_elemwise(&input.client, num_elements, cube_dim);

    norm_silu_kernel::launch::<R>(
        &input.client,
        cube_count,
        cube_dim,
        input.as_tensor_arg(1),
        mean.as_tensor_arg(1),
        var.as_tensor_arg(1),
        weight.as_tensor_arg(1),
        bias.as_tensor_arg(1),
        output.as_tensor_arg(1),
        NormSiLuArgsLaunch::new(
            ScalarArg::new(options.num_groups as u32),
            ScalarArg::new(channels_per_group as u32),
            ScalarArg::new(options.eps),
        ),
        input.dtype.into(),
    )
    .expect("norm_silu kernel launch failed");

    output
}

/// GroupNorm without SiLU (for cases where only normalization is needed)
///
/// This is provided for completeness but the main use case is `groupnorm_silu`.
pub fn groupnorm<R: CubeRuntime>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: CubeTensor<R>,
    options: GroupNormSiLuOptions,
) -> CubeTensor<R> {
    let [batch, channels, height, width] = input.shape.dims();

    assert!(
        channels % options.num_groups == 0,
        "channels ({}) must be divisible by num_groups ({})",
        channels,
        options.num_groups
    );

    let channels_per_group = channels / options.num_groups;
    let spatial_size = height * width;

    // Phase 1: Compute mean and variance per (batch, group)
    let stats_shape = Shape::new([batch, options.num_groups]);
    let mean = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        stats_shape.clone(),
        input.dtype,
    );
    let var = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        stats_shape.clone(),
        input.dtype,
    );

    let num_stats = batch * options.num_groups;
    let stats_cube_dim = CubeDim::new(&input.client, num_stats);
    let stats_cube_count = calculate_cube_count_elemwise(&input.client, num_stats, stats_cube_dim);

    compute_group_stats_kernel::launch::<R>(
        &input.client,
        stats_cube_count,
        stats_cube_dim,
        input.as_tensor_arg(1),
        mean.as_tensor_arg(1),
        var.as_tensor_arg(1),
        GroupStatsArgsLaunch::new(
            ScalarArg::new(spatial_size as u32),
            ScalarArg::new(options.num_groups as u32),
            ScalarArg::new(channels as u32),
        ),
        input.dtype.into(),
    )
    .expect("compute_group_stats kernel launch failed");

    // Phase 2: Normalize + Affine (no SiLU)
    let out_shape = Shape::new([batch, channels, height, width]);
    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        out_shape.clone(),
        input.dtype,
    );

    let num_elements = out_shape.num_elements();
    let cube_dim = CubeDim::new(&input.client, num_elements);
    let cube_count = calculate_cube_count_elemwise(&input.client, num_elements, cube_dim);

    norm_kernel::launch::<R>(
        &input.client,
        cube_count,
        cube_dim,
        input.as_tensor_arg(1),
        mean.as_tensor_arg(1),
        var.as_tensor_arg(1),
        weight.as_tensor_arg(1),
        bias.as_tensor_arg(1),
        output.as_tensor_arg(1),
        NormSiLuArgsLaunch::new(
            ScalarArg::new(options.num_groups as u32),
            ScalarArg::new(channels_per_group as u32),
            ScalarArg::new(options.eps),
        ),
        input.dtype.into(),
    )
    .expect("norm kernel launch failed");

    output
}

// ============================================================================
// Layer API (for use in models)
// ============================================================================

use burn_cubecl::CubeBackend;

/// Fused GroupNorm + SiLU layer for CubeBackend
///
/// Drop-in replacement for separate GroupNorm + silu() calls.
/// Uses fused GPU kernel for better performance.
///
/// # Example
///
/// ```ignore
/// use sketchpad_cubecl::{GroupNormSiLuLayer, tensor_to_cube};
///
/// // Create from existing GroupNorm weights
/// let layer = GroupNormSiLuLayer::from_tensors(
///     32, // num_groups
///     tensor_to_cube(groupnorm.weight),
///     tensor_to_cube(groupnorm.bias),
/// );
/// let output = layer.forward(input); // Fused GroupNorm + SiLU
/// ```
#[derive(Debug)]
pub struct GroupNormSiLuLayer<R: CubeRuntime> {
    /// Number of groups
    pub num_groups: usize,
    /// Scale parameter (gamma)
    pub weight: CubeTensor<R>,
    /// Bias parameter (beta)
    pub bias: CubeTensor<R>,
    /// Epsilon for numerical stability
    pub eps: f32,
}

impl<R: CubeRuntime> GroupNormSiLuLayer<R> {
    /// Creates a layer from existing weight and bias CubeTensors
    pub fn from_tensors(num_groups: usize, weight: CubeTensor<R>, bias: CubeTensor<R>) -> Self {
        Self {
            num_groups,
            weight,
            bias,
            eps: 1e-5,
        }
    }

    /// Creates a layer from existing weight and bias CubeTensors with custom epsilon
    pub fn from_tensors_with_eps(
        num_groups: usize,
        weight: CubeTensor<R>,
        bias: CubeTensor<R>,
        eps: f32,
    ) -> Self {
        Self {
            num_groups,
            weight,
            bias,
            eps,
        }
    }

    /// Forward pass: GroupNorm + SiLU fused
    pub fn forward(&self, input: CubeTensor<R>) -> CubeTensor<R> {
        groupnorm_silu(
            input,
            self.weight.clone(),
            self.bias.clone(),
            GroupNormSiLuOptions {
                num_groups: self.num_groups,
                eps: self.eps,
            },
        )
    }
}

/// Convenience function to convert Burn tensor to CubeTensor for layer integration
///
/// For use when integrating with existing Burn models that use `Tensor<CubeBackend<R, F, I, BT>, D>`
pub fn tensor_to_cube<R, F, I, BT, const D: usize>(
    tensor: burn::tensor::Tensor<CubeBackend<R, F, I, BT>, D>,
) -> CubeTensor<R>
where
    R: CubeRuntime,
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::BoolElement,
{
    tensor.into_primitive().tensor()
}

/// Convenience function to convert CubeTensor back to Burn tensor
pub fn cube_to_tensor<R, F, I, BT, const D: usize>(
    tensor: CubeTensor<R>,
) -> burn::tensor::Tensor<CubeBackend<R, F, I, BT>, D>
where
    R: CubeRuntime,
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::BoolElement,
{
    burn::tensor::Tensor::from_primitive(burn::tensor::TensorPrimitive::Float(tensor))
}

// ============================================================================
// Norm-only kernel (without SiLU)
// ============================================================================

/// Normalize + Affine kernel (without SiLU)
#[cube(launch)]
fn norm_kernel<E: Float>(
    input: &Tensor<E>,
    mean: &Tensor<E>,
    var: &Tensor<E>,
    weight: &Tensor<E>,
    bias: &Tensor<E>,
    output: &mut Tensor<E>,
    args: NormSiLuArgs,
    #[define(E)] _dtype: StorageType,
) {
    let idx = ABSOLUTE_POS;
    if idx >= output.len() {
        terminate!();
    }

    // Output layout: [B, C, H, W]
    let out_w = output.shape(3);
    let out_h = output.shape(2);
    let out_c = output.shape(1);

    // Decompose linear index to NCHW coordinates
    let w = idx % out_w;
    let tmp = idx / out_w;
    let h = tmp % out_h;
    let tmp = tmp / out_h;
    let c = tmp % out_c;
    let b = tmp / out_c;

    // Determine which group this channel belongs to
    let g = c / args.channels_per_group;

    // Look up mean and variance for this (batch, group)
    let stats_idx = b * args.num_groups + g;
    let group_mean = mean[stats_idx];
    let group_var = var[stats_idx];

    // Input strides (for non-contiguous support)
    let in_stride_w = input.stride(3);
    let in_stride_h = input.stride(2);
    let in_stride_c = input.stride(1);
    let in_stride_b = input.stride(0);
    let in_idx = b * in_stride_b + c * in_stride_c + h * in_stride_h + w * in_stride_w;

    // 1. Normalize: (x - mean) / sqrt(var + eps)
    let x = input[in_idx];
    let eps = E::cast_from(args.eps);
    let std = E::sqrt(group_var + eps);
    let normalized = (x - group_mean) / std;

    // 2. Apply affine: weight[c] * normalized + bias[c]
    let gamma = weight[c];
    let beta = bias[c];

    output[idx] = gamma * normalized + beta;
}
