//! Quantization Support for Model Compression
//!
//! Provides INT8, INT4, and FP8 quantization for reduced memory usage
//! and faster inference on supported hardware.
//!
//! # Quantization Types
//!
//! - **Dynamic Quantization**: Weights quantized, activations quantized at runtime
//! - **Static Quantization**: Both weights and activations pre-quantized (requires calibration)
//!
//! # Supported Formats
//!
//! - INT8 symmetric: scale only, zero_point = 0
//! - INT8 asymmetric: scale + zero_point
//! - INT4 (GPTQ/AWQ style): group-wise quantization
//! - FP8 E4M3: 4 exponent bits, 3 mantissa bits (better for weights)
//! - FP8 E5M2: 5 exponent bits, 2 mantissa bits (better for activations)

use burn::prelude::*;

/// Quantization configuration
#[derive(Debug, Clone)]
pub struct QuantConfig {
    /// Number of bits (4 or 8)
    pub bits: usize,
    /// Whether to use symmetric quantization
    pub symmetric: bool,
    /// Group size for group-wise quantization (0 = per-tensor)
    pub group_size: usize,
}

impl QuantConfig {
    /// INT8 symmetric per-tensor quantization
    pub fn int8_symmetric() -> Self {
        Self {
            bits: 8,
            symmetric: true,
            group_size: 0,
        }
    }

    /// INT8 asymmetric per-tensor quantization
    pub fn int8_asymmetric() -> Self {
        Self {
            bits: 8,
            symmetric: false,
            group_size: 0,
        }
    }

    /// INT4 group-wise quantization (GPTQ style)
    pub fn int4_grouped(group_size: usize) -> Self {
        Self {
            bits: 4,
            symmetric: true,
            group_size,
        }
    }

    /// Get the quantization range
    pub fn range(&self) -> (i64, i64) {
        if self.symmetric {
            let max = (1i64 << (self.bits - 1)) - 1;
            (-max, max)
        } else {
            let max = (1i64 << self.bits) - 1;
            (0, max)
        }
    }
}

/// Quantization parameters for a tensor
#[derive(Debug, Clone)]
pub struct QuantParams<B: Backend> {
    /// Scale factor(s) - [1] for per-tensor, [num_groups] for group-wise
    pub scale: Tensor<B, 1>,
    /// Zero point(s) - [1] for per-tensor, [num_groups] for group-wise
    pub zero_point: Tensor<B, 1>,
    /// Original shape
    pub shape: Vec<usize>,
    /// Configuration
    pub config: QuantConfig,
}

/// Quantized tensor representation
pub struct QuantizedTensor<B: Backend> {
    /// Quantized data stored as i32 (INT8 values in lower bits)
    pub data: Tensor<B, 2, Int>,
    /// Quantization parameters
    pub params: QuantParams<B>,
}

impl<B: Backend> QuantizedTensor<B> {
    /// Dequantize back to float
    pub fn dequantize(&self) -> Tensor<B, 2> {
        let [rows, cols] = self.data.dims();

        if self.params.config.group_size == 0 {
            // Per-tensor quantization
            let scale = self.params.scale.clone().reshape([1, 1]);
            let zero_point = self.params.zero_point.clone().reshape([1, 1]);

            let data_float = self.data.clone().float();
            (data_float - zero_point) * scale
        } else {
            // Group-wise quantization
            let group_size = self.params.config.group_size;
            let num_groups = cols.div_ceil(group_size);

            let mut result_parts = Vec::new();

            for g in 0..num_groups {
                let start = g * group_size;
                let end = (start + group_size).min(cols);

                let group_data = self.data.clone().slice([0..rows, start..end]);
                let scale = self.params.scale.clone().slice(g..g + 1);
                let zero_point = self.params.zero_point.clone().slice(g..g + 1);

                let scale = scale.reshape([1, 1]);
                let zero_point = zero_point.reshape([1, 1]);

                let group_float = group_data.float();
                let dequant = (group_float - zero_point) * scale;
                result_parts.push(dequant);
            }

            Tensor::cat(result_parts, 1)
        }
    }
}

/// Quantize a tensor to INT8/INT4
pub fn quantize<B: Backend>(tensor: Tensor<B, 2>, config: &QuantConfig) -> QuantizedTensor<B> {
    let [rows, cols] = tensor.dims();
    let device = tensor.device();

    if config.group_size == 0 {
        // Per-tensor quantization
        let (scale, zero_point, quantized) = quantize_tensor(&tensor, config, &device);

        QuantizedTensor {
            data: quantized,
            params: QuantParams {
                scale,
                zero_point,
                shape: vec![rows, cols],
                config: config.clone(),
            },
        }
    } else {
        // Group-wise quantization
        let group_size = config.group_size;
        let num_groups = cols.div_ceil(group_size);

        let mut scales = Vec::new();
        let mut zero_points = Vec::new();
        let mut quantized_groups = Vec::new();

        for g in 0..num_groups {
            let start = g * group_size;
            let end = (start + group_size).min(cols);

            let group = tensor.clone().slice([0..rows, start..end]);
            let (scale, zp, quant) = quantize_tensor(&group, config, &device);

            scales.push(scale);
            zero_points.push(zp);
            quantized_groups.push(quant);
        }

        let scale = Tensor::cat(scales, 0);
        let zero_point = Tensor::cat(zero_points, 0);
        let quantized = Tensor::cat(quantized_groups, 1);

        QuantizedTensor {
            data: quantized,
            params: QuantParams {
                scale,
                zero_point,
                shape: vec![rows, cols],
                config: config.clone(),
            },
        }
    }
}

fn quantize_tensor<B: Backend>(
    tensor: &Tensor<B, 2>,
    config: &QuantConfig,
    device: &B::Device,
) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 2, Int>) {
    let (qmin, qmax) = config.range();

    // Find min/max - use reduce operations on 2D tensor
    let min_per_row = tensor.clone().min_dim(1);
    let max_per_row = tensor.clone().max_dim(1);
    let min_val: f32 = min_per_row.min().into_scalar().elem();
    let max_val: f32 = max_per_row.max().into_scalar().elem();

    let (scale, zero_point) = if config.symmetric {
        // Symmetric: scale = max(|min|, |max|) / qmax
        let abs_max = min_val.abs().max(max_val.abs());
        let scale = if abs_max > 0.0 {
            abs_max / qmax as f32
        } else {
            1.0
        };
        (scale, 0.0f32)
    } else {
        // Asymmetric: scale = (max - min) / (qmax - qmin)
        let scale = if (max_val - min_val).abs() > 1e-10 {
            (max_val - min_val) / (qmax - qmin) as f32
        } else {
            1.0
        };
        let zero_point = qmin as f32 - min_val / scale;
        (scale, zero_point.round())
    };

    // Quantize
    let scale_tensor = Tensor::<B, 1>::from_floats([scale], device);
    let zp_tensor = Tensor::<B, 1>::from_floats([zero_point], device);

    let scaled = tensor.clone() / scale;
    let shifted = scaled + zero_point;
    let clamped = shifted.clamp(qmin as f32, qmax as f32);
    let quantized = clamped.int();

    (scale_tensor, zp_tensor, quantized)
}

/// Quantized linear layer
pub struct QuantizedLinear<B: Backend> {
    /// Quantized weight
    pub weight: QuantizedTensor<B>,
    /// Optional bias (not quantized)
    pub bias: Option<Tensor<B, 1>>,
}

impl<B: Backend> QuantizedLinear<B> {
    /// Create from a regular linear layer's weights
    pub fn from_weight(
        weight: Tensor<B, 2>,
        bias: Option<Tensor<B, 1>>,
        config: &QuantConfig,
    ) -> Self {
        Self {
            weight: quantize(weight, config),
            bias,
        }
    }

    /// Forward pass with dequantization
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        // Dequantize weights
        let weight = self.weight.dequantize();

        // Standard linear: x @ weight.T + bias
        let out = x.matmul(weight.transpose());

        match &self.bias {
            Some(b) => out + b.clone().unsqueeze(),
            None => out,
        }
    }

    /// Memory savings compared to f32
    pub fn memory_ratio(&self) -> f32 {
        let bits = self.weight.params.config.bits;
        bits as f32 / 32.0
    }
}

/// Calculate memory usage for quantized vs original
pub fn memory_savings(num_params: usize, config: &QuantConfig) -> QuantMemoryStats {
    let original_bytes = num_params * 4; // f32 = 4 bytes
    let quantized_bytes = num_params * config.bits / 8;

    // Add overhead for scales/zero_points
    let num_groups = if config.group_size > 0 {
        num_params.div_ceil(config.group_size)
    } else {
        1
    };
    let param_overhead = num_groups * 8; // 2 x f32 per group

    let total_quantized = quantized_bytes + param_overhead;

    QuantMemoryStats {
        original_bytes,
        quantized_bytes: total_quantized,
        compression_ratio: original_bytes as f32 / total_quantized as f32,
    }
}

/// Memory statistics for quantization
#[derive(Debug, Clone)]
pub struct QuantMemoryStats {
    pub original_bytes: usize,
    pub quantized_bytes: usize,
    pub compression_ratio: f32,
}

// ============================================================================
// FP8 Quantization Support
// ============================================================================

/// FP8 format variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Format {
    /// E4M3: 4 exponent bits, 3 mantissa bits
    /// Range: ±448, better precision for weights
    E4M3,
    /// E5M2: 5 exponent bits, 2 mantissa bits
    /// Range: ±57344, better range for activations
    E5M2,
}

impl Fp8Format {
    /// Get the number of exponent bits
    pub fn exponent_bits(&self) -> u32 {
        match self {
            Fp8Format::E4M3 => 4,
            Fp8Format::E5M2 => 5,
        }
    }

    /// Get the number of mantissa bits
    pub fn mantissa_bits(&self) -> u32 {
        match self {
            Fp8Format::E4M3 => 3,
            Fp8Format::E5M2 => 2,
        }
    }

    /// Get the exponent bias
    pub fn bias(&self) -> i32 {
        match self {
            Fp8Format::E4M3 => 7,  // 2^(4-1) - 1
            Fp8Format::E5M2 => 15, // 2^(5-1) - 1
        }
    }

    /// Get the maximum representable value
    pub fn max_value(&self) -> f32 {
        match self {
            Fp8Format::E4M3 => 448.0,   // (1 + 7/8) * 2^8 = 448
            Fp8Format::E5M2 => 57344.0, // (1 + 3/4) * 2^15 = 57344
        }
    }

    /// Get the minimum positive normal value
    pub fn min_normal(&self) -> f32 {
        match self {
            Fp8Format::E4M3 => 2f32.powi(-6),  // 2^(1-7) = 2^-6
            Fp8Format::E5M2 => 2f32.powi(-14), // 2^(1-15) = 2^-14
        }
    }

    /// Get the smallest subnormal value
    pub fn min_subnormal(&self) -> f32 {
        match self {
            Fp8Format::E4M3 => 2f32.powi(-9),  // 2^(-6-3)
            Fp8Format::E5M2 => 2f32.powi(-16), // 2^(-14-2)
        }
    }
}

/// FP8 quantization configuration
#[derive(Debug, Clone)]
pub struct Fp8Config {
    /// FP8 format to use
    pub format: Fp8Format,
    /// Whether to use per-tensor or per-channel scaling
    pub per_channel: bool,
    /// Scaling factor (computed from calibration or max value)
    pub scale: Option<f32>,
}

impl Default for Fp8Config {
    fn default() -> Self {
        Self {
            format: Fp8Format::E4M3,
            per_channel: false,
            scale: None,
        }
    }
}

impl Fp8Config {
    /// E4M3 format (better for weights)
    pub fn e4m3() -> Self {
        Self {
            format: Fp8Format::E4M3,
            ..Default::default()
        }
    }

    /// E5M2 format (better for activations)
    pub fn e5m2() -> Self {
        Self {
            format: Fp8Format::E5M2,
            ..Default::default()
        }
    }

    /// Set per-channel quantization
    pub fn with_per_channel(mut self, per_channel: bool) -> Self {
        self.per_channel = per_channel;
        self
    }

    /// Set explicit scale factor
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = Some(scale);
        self
    }
}

/// Simulated FP8 tensor (stored as f32 but with FP8 precision)
pub struct Fp8Tensor<B: Backend> {
    /// Data stored in f32 but quantized to FP8 precision
    pub data: Tensor<B, 2>,
    /// Scale factor(s) used for quantization
    pub scale: Tensor<B, 1>,
    /// FP8 format
    pub format: Fp8Format,
    /// Original shape
    pub shape: Vec<usize>,
}

impl<B: Backend> Fp8Tensor<B> {
    /// Get the dequantized tensor (already stored as f32)
    pub fn dequantize(&self) -> Tensor<B, 2> {
        // Data is already in f32, just needs to be scaled back
        let [rows, _cols] = self.data.dims();
        if self.scale.dims()[0] == 1 {
            // Per-tensor scale
            let scale = self.scale.clone().reshape([1, 1]);
            self.data.clone() * scale
        } else {
            // Per-channel scale
            let scale = self.scale.clone().reshape([rows, 1]);
            self.data.clone() * scale
        }
    }
}

/// Quantize a single f32 value to FP8 precision
fn quantize_fp8_scalar(value: f32, format: Fp8Format) -> f32 {
    if value.is_nan() {
        return f32::NAN;
    }
    if value.is_infinite() {
        return if value > 0.0 {
            format.max_value()
        } else {
            -format.max_value()
        };
    }

    let sign = value.signum();
    let abs_val = value.abs();

    // Clamp to representable range
    let max_val = format.max_value();
    if abs_val > max_val {
        return sign * max_val;
    }

    let min_subnormal = format.min_subnormal();
    if abs_val < min_subnormal {
        return 0.0;
    }

    // Round to FP8 precision
    let mantissa_bits = format.mantissa_bits();

    // Get the exponent
    let exponent = abs_val.log2().floor() as i32;
    let mantissa_scale = 2f32.powi(exponent);

    // Normalize to [1, 2) range
    let normalized = abs_val / mantissa_scale;

    // Quantize mantissa (only keep mantissa_bits precision)
    let mantissa_steps = 2u32.pow(mantissa_bits) as f32;
    let quantized_mantissa = ((normalized - 1.0) * mantissa_steps).round() / mantissa_steps + 1.0;

    sign * quantized_mantissa * mantissa_scale
}

/// Quantize a tensor to FP8
pub fn quantize_fp8<B: Backend>(tensor: Tensor<B, 2>, config: &Fp8Config) -> Fp8Tensor<B> {
    let [rows, cols] = tensor.dims();
    let device = tensor.device();
    let format = config.format;
    let max_fp8 = format.max_value();

    if config.per_channel {
        // Per-channel quantization
        let mut scales_vec = Vec::new();
        let mut quantized_rows = Vec::new();

        for r in 0..rows {
            let row = tensor.clone().slice([r..r + 1, 0..cols]);
            let row_max: f32 = row.clone().abs().max().into_scalar().elem();

            let scale = if row_max > 0.0 {
                row_max / max_fp8
            } else {
                1.0
            };
            scales_vec.push(scale);

            // Scale down, quantize to FP8, then the data stays scaled
            let scaled = row / scale;

            // Apply FP8 quantization to each element
            let data: Vec<f32> = scaled.to_data().to_vec().unwrap();
            let quantized_data: Vec<f32> = data
                .iter()
                .map(|&v| quantize_fp8_scalar(v, format))
                .collect();

            // Create 1D tensor then reshape to 2D
            let quantized_row =
                Tensor::<B, 1>::from_floats(quantized_data.as_slice(), &device).reshape([1, cols]);

            quantized_rows.push(quantized_row);
        }

        let scale = Tensor::<B, 1>::from_floats(scales_vec.as_slice(), &device);
        let data = Tensor::cat(quantized_rows, 0);

        Fp8Tensor {
            data,
            scale,
            format,
            shape: vec![rows, cols],
        }
    } else {
        // Per-tensor quantization
        let scale = config.scale.unwrap_or_else(|| {
            let tensor_max: f32 = tensor.clone().abs().max().into_scalar().elem();
            if tensor_max > 0.0 {
                tensor_max / max_fp8
            } else {
                1.0
            }
        });

        // Scale down
        let scaled = tensor / scale;

        // Apply FP8 quantization
        let data: Vec<f32> = scaled.to_data().to_vec().unwrap();
        let quantized_data: Vec<f32> = data
            .iter()
            .map(|&v| quantize_fp8_scalar(v, format))
            .collect();

        // Create 1D tensor then reshape to 2D
        let quantized =
            Tensor::<B, 1>::from_floats(quantized_data.as_slice(), &device).reshape([rows, cols]);

        let scale_tensor = Tensor::<B, 1>::from_floats([scale], &device);

        Fp8Tensor {
            data: quantized,
            scale: scale_tensor,
            format,
            shape: vec![rows, cols],
        }
    }
}

/// FP8 quantized linear layer
pub struct Fp8Linear<B: Backend> {
    /// FP8 quantized weight
    pub weight: Fp8Tensor<B>,
    /// Optional bias (not quantized)
    pub bias: Option<Tensor<B, 1>>,
}

impl<B: Backend> Fp8Linear<B> {
    /// Create from regular weights
    pub fn from_weight(
        weight: Tensor<B, 2>,
        bias: Option<Tensor<B, 1>>,
        config: &Fp8Config,
    ) -> Self {
        Self {
            weight: quantize_fp8(weight, config),
            bias,
        }
    }

    /// Forward pass with dequantization
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let weight = self.weight.dequantize();
        let out = x.matmul(weight.transpose());

        match &self.bias {
            Some(b) => out + b.clone().unsqueeze(),
            None => out,
        }
    }

    /// Memory ratio compared to f32
    pub fn memory_ratio(&self) -> f32 {
        8.0 / 32.0 // FP8 is 8 bits, f32 is 32 bits
    }
}

/// Calculate FP8 memory savings
pub fn fp8_memory_savings(num_params: usize, config: &Fp8Config) -> QuantMemoryStats {
    let original_bytes = num_params * 4; // f32 = 4 bytes
    let quantized_bytes = num_params; // FP8 = 1 byte

    // Add overhead for scales
    let num_scales = if config.per_channel {
        num_params / 1000 // Rough estimate: one scale per row
    } else {
        1
    };
    let param_overhead = num_scales * 4; // f32 scale

    let total_quantized = quantized_bytes + param_overhead;

    QuantMemoryStats {
        original_bytes,
        quantized_bytes: total_quantized,
        compression_ratio: original_bytes as f32 / total_quantized as f32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_quant_config() {
        let config = QuantConfig::int8_symmetric();
        assert_eq!(config.bits, 8);
        assert!(config.symmetric);
        assert_eq!(config.range(), (-127, 127));

        let config = QuantConfig::int8_asymmetric();
        assert_eq!(config.range(), (0, 255));

        let config = QuantConfig::int4_grouped(128);
        assert_eq!(config.bits, 4);
        assert_eq!(config.group_size, 128);
        assert_eq!(config.range(), (-7, 7));
    }

    #[test]
    fn test_quantize_dequantize_symmetric() {
        let device = Default::default();
        let config = QuantConfig::int8_symmetric();

        // Create a tensor with known values
        let tensor = Tensor::<TestBackend, 2>::from_floats(
            [[1.0, -2.0, 3.0, -4.0], [0.5, -0.5, 1.5, -1.5]],
            &device,
        );

        let quantized = quantize(tensor.clone(), &config);
        let dequantized = quantized.dequantize();

        // Check approximate reconstruction
        let diff = (tensor - dequantized).abs();
        let max_diff: f32 = diff.max().into_scalar().elem();

        // Should be close (within quantization error)
        assert!(max_diff < 0.1, "Max diff {} too large", max_diff);
    }

    #[test]
    fn test_quantize_dequantize_asymmetric() {
        let device = Default::default();
        let config = QuantConfig::int8_asymmetric();

        let tensor = Tensor::<TestBackend, 2>::from_floats(
            [[0.0, 1.0, 2.0, 3.0], [0.5, 1.5, 2.5, 3.5]],
            &device,
        );

        let quantized = quantize(tensor.clone(), &config);
        let dequantized = quantized.dequantize();

        let diff = (tensor - dequantized).abs();
        let max_diff: f32 = diff.max().into_scalar().elem();

        assert!(max_diff < 0.1, "Max diff {} too large", max_diff);
    }

    #[test]
    fn test_quantize_grouped() {
        let device = Default::default();
        let config = QuantConfig::int4_grouped(2); // Group size 2

        let tensor = Tensor::<TestBackend, 2>::from_floats(
            [[1.0, 2.0, 10.0, 20.0]], // Two groups with different scales
            &device,
        );

        let quantized = quantize(tensor.clone(), &config);

        // Should have 2 scales (one per group)
        assert_eq!(quantized.params.scale.dims(), [2]);

        let dequantized = quantized.dequantize();
        let diff = (tensor - dequantized).abs();
        let max_diff: f32 = diff.max().into_scalar().elem();

        // INT4 has more quantization error
        assert!(max_diff < 3.0, "Max diff {} too large for INT4", max_diff);
    }

    #[test]
    fn test_quantized_linear() {
        let device = Default::default();
        let config = QuantConfig::int8_symmetric();

        // Create weight [out_features, in_features] = [3, 4]
        let weight = Tensor::<TestBackend, 2>::from_floats(
            [
                [1.0, 2.0, 3.0, 4.0],
                [0.1, 0.2, 0.3, 0.4],
                [-1.0, -2.0, -3.0, -4.0],
            ],
            &device,
        );

        let linear = QuantizedLinear::from_weight(weight, None, &config);

        // Forward pass
        let x = Tensor::<TestBackend, 2>::from_floats([[1.0, 1.0, 1.0, 1.0]], &device);
        let out = linear.forward(x);

        assert_eq!(out.dims(), [1, 3]);

        // Check memory ratio
        assert!((linear.memory_ratio() - 0.25).abs() < 0.01); // 8/32 = 0.25
    }

    #[test]
    fn test_memory_savings() {
        // 1M params
        let stats = memory_savings(1_000_000, &QuantConfig::int8_symmetric());
        assert_eq!(stats.original_bytes, 4_000_000);
        assert!(stats.compression_ratio > 3.9); // Close to 4x

        let stats = memory_savings(1_000_000, &QuantConfig::int4_grouped(128));
        assert!(stats.compression_ratio > 7.0); // Close to 8x
    }

    // FP8 Tests

    #[test]
    fn test_fp8_format() {
        let e4m3 = Fp8Format::E4M3;
        assert_eq!(e4m3.exponent_bits(), 4);
        assert_eq!(e4m3.mantissa_bits(), 3);
        assert_eq!(e4m3.bias(), 7);
        assert!((e4m3.max_value() - 448.0).abs() < 1.0);

        let e5m2 = Fp8Format::E5M2;
        assert_eq!(e5m2.exponent_bits(), 5);
        assert_eq!(e5m2.mantissa_bits(), 2);
        assert_eq!(e5m2.bias(), 15);
        assert!((e5m2.max_value() - 57344.0).abs() < 1.0);
    }

    #[test]
    fn test_fp8_config() {
        let config = Fp8Config::e4m3();
        assert_eq!(config.format, Fp8Format::E4M3);
        assert!(!config.per_channel);

        let config = Fp8Config::e5m2().with_per_channel(true);
        assert_eq!(config.format, Fp8Format::E5M2);
        assert!(config.per_channel);
    }

    #[test]
    fn test_quantize_fp8_scalar() {
        // Test E4M3 quantization
        let val = quantize_fp8_scalar(1.5, Fp8Format::E4M3);
        assert!((val - 1.5).abs() < 0.01); // 1.5 is exactly representable

        let val = quantize_fp8_scalar(1.234, Fp8Format::E4M3);
        assert!((val - 1.25).abs() < 0.1); // Rounds to nearest FP8

        // Test clamping
        let val = quantize_fp8_scalar(1000.0, Fp8Format::E4M3);
        assert!((val - 448.0).abs() < 1.0); // Clamped to max

        // Test zero handling
        let val = quantize_fp8_scalar(0.0, Fp8Format::E4M3);
        assert_eq!(val, 0.0);

        // Test negative values
        let val = quantize_fp8_scalar(-2.0, Fp8Format::E4M3);
        assert!((val - (-2.0)).abs() < 0.1);
    }

    #[test]
    fn test_quantize_fp8_tensor() {
        let device = Default::default();
        let config = Fp8Config::e4m3();

        let tensor = Tensor::<TestBackend, 2>::from_floats(
            [[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]],
            &device,
        );

        let quantized = quantize_fp8(tensor.clone(), &config);
        let dequantized = quantized.dequantize();

        // Check approximate reconstruction
        let diff = (tensor - dequantized).abs();
        let max_diff: f32 = diff.max().into_scalar().elem();
        assert!(max_diff < 0.5, "Max diff {} too large for FP8", max_diff);
    }

    #[test]
    fn test_quantize_fp8_per_channel() {
        let device = Default::default();
        let config = Fp8Config::e4m3().with_per_channel(true);

        // Different scales per row
        let tensor = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0], [100.0, 200.0]], &device);

        let quantized = quantize_fp8(tensor.clone(), &config);
        assert_eq!(quantized.scale.dims(), [2]); // One scale per row

        let dequantized = quantized.dequantize();
        let diff = (tensor - dequantized).abs();
        let max_diff: f32 = diff.max().into_scalar().elem();
        assert!(max_diff < 5.0, "Max diff {} too large", max_diff);
    }

    #[test]
    fn test_fp8_linear() {
        let device = Default::default();
        let config = Fp8Config::e4m3();

        let weight = Tensor::<TestBackend, 2>::from_floats(
            [
                [1.0, 2.0, 3.0, 4.0],
                [0.1, 0.2, 0.3, 0.4],
                [-1.0, -2.0, -3.0, -4.0],
            ],
            &device,
        );

        let linear = Fp8Linear::from_weight(weight, None, &config);

        let x = Tensor::<TestBackend, 2>::from_floats([[1.0, 1.0, 1.0, 1.0]], &device);
        let out = linear.forward(x);

        assert_eq!(out.dims(), [1, 3]);
        assert!((linear.memory_ratio() - 0.25).abs() < 0.01); // 8/32 = 0.25
    }

    #[test]
    fn test_fp8_memory_savings() {
        let config = Fp8Config::e4m3();
        let stats = fp8_memory_savings(1_000_000, &config);

        assert_eq!(stats.original_bytes, 4_000_000);
        assert!(stats.compression_ratio > 3.9); // Close to 4x
    }

    #[test]
    fn test_fp8_e5m2() {
        let device = Default::default();
        let config = Fp8Config::e5m2();

        // E5M2 has larger range, test with bigger values
        let tensor =
            Tensor::<TestBackend, 2>::from_floats([[100.0, 500.0, 1000.0, 5000.0]], &device);

        let quantized = quantize_fp8(tensor.clone(), &config);
        let dequantized = quantized.dequantize();

        let diff = (tensor - dequantized).abs();
        let max_diff: f32 = diff.max().into_scalar().elem();
        // E5M2 has less precision but larger range
        assert!(max_diff < 200.0, "Max diff {} too large for E5M2", max_diff);
    }
}
