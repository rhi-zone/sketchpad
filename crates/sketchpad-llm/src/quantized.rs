//! Quantized tensor storage and computation
//!
//! Keeps weights in their quantized format (Q4_K, Q5_K, Q6_K, Q8_0, etc.)
//! and only dequantizes per-row during matrix multiply. This keeps memory
//! usage proportional to the quantized size (~15GB for 26B params in Q4_K)
//! instead of the full f32 size (~104GB).

use std::sync::Arc;

use burn::prelude::*;
use sketchpad_convert::gguf::{GgmlType, dequantize};

/// A quantized weight matrix stored as raw GGUF quantization blocks.
///
/// Represents a 2D weight matrix [out_features, in_features] in quantized form.
/// Each row of `in_features` elements is stored as quantization blocks.
#[derive(Debug)]
pub struct QuantizedTensor {
    /// Raw quantized bytes (owned copy of mmap'd data)
    data: Vec<u8>,
    /// Quantization type
    ggml_type: GgmlType,
    /// Number of output features (rows)
    out_features: usize,
    /// Number of input features (columns) — must be divisible by block_size
    in_features: usize,
    /// Bytes per row (precomputed)
    row_bytes: usize,
}

impl QuantizedTensor {
    /// Create from raw quantized data and dimensions.
    ///
    /// `data` must contain exactly `out_features` rows of quantized blocks,
    /// where each row has `in_features / block_size * type_size` bytes.
    pub fn new(
        data: Vec<u8>,
        ggml_type: GgmlType,
        out_features: usize,
        in_features: usize,
    ) -> Self {
        let block_size = ggml_type.block_size();
        let type_size = ggml_type.type_size();
        assert!(
            in_features % block_size == 0,
            "in_features ({in_features}) must be divisible by block_size ({block_size})"
        );
        let row_bytes = (in_features / block_size) * type_size;
        assert_eq!(
            data.len(),
            out_features * row_bytes,
            "data length {} doesn't match {} rows × {} bytes/row",
            data.len(),
            out_features,
            row_bytes
        );
        Self {
            data,
            ggml_type,
            out_features,
            in_features,
            row_bytes,
        }
    }

    /// Matrix multiply: input [n_tokens, in_features] × weight^T → [n_tokens, out_features]
    ///
    /// Dequantizes one row of the weight matrix at a time, keeping peak memory
    /// overhead at just `in_features` floats (~11KB for hidden_size=2816).
    pub fn matmul_f32(&self, input: &[f32], n_tokens: usize) -> Vec<f32> {
        debug_assert_eq!(input.len(), n_tokens * self.in_features);

        let mut output = vec![0.0f32; n_tokens * self.out_features];
        let mut row_buf = vec![0.0f32; self.in_features];

        for out_idx in 0..self.out_features {
            let row_start = out_idx * self.row_bytes;
            let row_data = &self.data[row_start..row_start + self.row_bytes];

            // Dequantize this row into the reusable buffer
            dequantize_into(row_data, self.ggml_type, &mut row_buf);

            // Dot product with each token's input
            for tok in 0..n_tokens {
                let inp_start = tok * self.in_features;
                let dot = dot_product(&input[inp_start..inp_start + self.in_features], &row_buf);
                output[tok * self.out_features + out_idx] = dot;
            }
        }

        output
    }

    /// Forward pass: [batch, seq_len, in_features] → [batch, seq_len, out_features]
    pub fn forward<B: Backend>(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _in_feat] = input.dims();
        let device = input.device();

        let input_data: Vec<f32> = input.to_data().to_vec().unwrap();
        let n_tokens = batch * seq_len;

        let output_data = self.matmul_f32(&input_data, n_tokens);

        let tensor_data = TensorData::new(output_data, [batch, seq_len, self.out_features]);
        Tensor::from_data(tensor_data, &device)
    }

    /// Forward pass for 2D: [n_tokens, in_features] → [n_tokens, out_features]
    pub fn forward_2d<B: Backend>(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let [n_tokens, _in_feat] = input.dims();
        let device = input.device();

        let input_data: Vec<f32> = input.to_data().to_vec().unwrap();

        let output_data = self.matmul_f32(&input_data, n_tokens);

        let tensor_data = TensorData::new(output_data, [n_tokens, self.out_features]);
        Tensor::from_data(tensor_data, &device)
    }
}

/// Quantized storage for fused expert tensors from GGUF.
///
/// GGUF stores MoE experts as fused 3D tensors:
/// - `gate_up_exps`: [hidden_size, expert_inter×2, num_experts] — gate and up projections
/// - `down_exps`: [expert_inter, hidden_size, num_experts] — down projection
///
/// In GGML row-major layout, the expert dimension is outermost, so each expert's
/// data is contiguous in memory. This struct references the mmap directly via Arc,
/// avoiding a copy of the expert weight bytes.
#[derive(Debug)]
pub struct QuantizedFusedExperts {
    gate_up_mmap: Arc<memmap2::Mmap>,
    gate_up_offset: usize,
    gate_up_type: GgmlType,
    down_mmap: Arc<memmap2::Mmap>,
    down_offset: usize,
    down_type: GgmlType,
    /// Model dimensions
    hidden_size: usize,
    expert_inter: usize,
    num_experts: usize,
    /// Precomputed byte sizes per expert
    gate_up_bytes_per_expert: usize,
    down_bytes_per_expert: usize,
}

impl QuantizedFusedExperts {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gate_up_mmap: Arc<memmap2::Mmap>,
        gate_up_offset: usize,
        gate_up_len: usize,
        gate_up_type: GgmlType,
        down_mmap: Arc<memmap2::Mmap>,
        down_offset: usize,
        down_len: usize,
        down_type: GgmlType,
        hidden_size: usize,
        expert_inter: usize,
        num_experts: usize,
    ) -> Self {
        let gate_up_rows_per_expert = expert_inter * 2;
        let gate_up_row_bytes =
            (hidden_size / gate_up_type.block_size()) * gate_up_type.type_size();
        let gate_up_bytes_per_expert = gate_up_rows_per_expert * gate_up_row_bytes;

        let down_rows_per_expert = hidden_size;
        let down_row_bytes = (expert_inter / down_type.block_size()) * down_type.type_size();
        let down_bytes_per_expert = down_rows_per_expert * down_row_bytes;

        assert_eq!(
            gate_up_len,
            gate_up_bytes_per_expert * num_experts,
            "gate_up data size mismatch"
        );
        assert_eq!(
            down_len,
            down_bytes_per_expert * num_experts,
            "down data size mismatch"
        );

        Self {
            gate_up_mmap,
            gate_up_offset,
            gate_up_type,
            down_mmap,
            down_offset,
            down_type,
            hidden_size,
            expert_inter,
            num_experts,
            gate_up_bytes_per_expert,
            down_bytes_per_expert,
        }
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    /// Run a single expert's GeGLU FFN on the input.
    ///
    /// Input: [n_tokens, 1, hidden_size]
    /// Output: [n_tokens, 1, hidden_size]
    ///
    /// Dequantizes only this expert's weights during computation.
    pub fn forward_expert<B: Backend>(
        &self,
        expert_idx: usize,
        input: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [n_tokens, one, _hidden] = input.dims();
        debug_assert_eq!(one, 1);
        let device = input.device();

        let input_data: Vec<f32> = input.to_data().to_vec().unwrap();

        // Extract this expert's gate_up data
        let gu_start = expert_idx * self.gate_up_bytes_per_expert;
        let gu_end = gu_start + self.gate_up_bytes_per_expert;
        let gu_data =
            &self.gate_up_mmap[self.gate_up_offset + gu_start..self.gate_up_offset + gu_end];

        // gate_up weight is [expert_inter*2, hidden_size] for this expert
        // After matmul: [n_tokens, expert_inter*2]
        let gate_up_flat = quantized_matmul_f32(
            gu_data,
            self.gate_up_type,
            self.expert_inter * 2,
            self.hidden_size,
            &input_data,
            n_tokens,
        );

        // Split into gate and up, apply GeGLU
        let mut hidden_buf = vec![0.0f32; n_tokens * self.expert_inter];
        for tok in 0..n_tokens {
            for i in 0..self.expert_inter {
                let gate_val = gate_up_flat[tok * self.expert_inter * 2 + i];
                let up_val = gate_up_flat[tok * self.expert_inter * 2 + self.expert_inter + i];
                // GeGLU: GELU(gate) * up
                hidden_buf[tok * self.expert_inter + i] = gelu_f32(gate_val) * up_val;
            }
        }

        // Down projection
        let dn_start = expert_idx * self.down_bytes_per_expert;
        let dn_end = dn_start + self.down_bytes_per_expert;
        let dn_data = &self.down_mmap[self.down_offset + dn_start..self.down_offset + dn_end];

        let output = quantized_matmul_f32(
            dn_data,
            self.down_type,
            self.hidden_size,
            self.expert_inter,
            &hidden_buf,
            n_tokens,
        );

        let tensor_data = TensorData::new(output, [n_tokens, 1, self.hidden_size]);
        Tensor::from_data(tensor_data, &device)
    }
}

/// A single linear layer with weights stored as quantized bytes in CPU RAM.
///
/// On each `forward` call, dequantizes the weights and uploads them to the
/// compute device. This keeps VRAM usage near zero for this layer at the
/// cost of per-token CPU dequantization and PCIe upload overhead.
///
/// Use when VRAM is the hard constraint and inference speed is secondary.
/// For best performance, use `burn::nn::Linear` with weights in VRAM instead.
pub struct QuantizedLinear {
    /// Raw quantized weight bytes, shape [out_features, in_features] in GGUF row-major layout
    data: Vec<u8>,
    ggml_type: GgmlType,
    in_features: usize,
    out_features: usize,
}

impl QuantizedLinear {
    /// Create from raw quantized bytes.
    ///
    /// `data` must contain `out_features` rows, each of
    /// `(in_features / block_size) * type_size` bytes.
    pub fn new(
        data: Vec<u8>,
        ggml_type: GgmlType,
        in_features: usize,
        out_features: usize,
    ) -> Self {
        let row_bytes = (in_features / ggml_type.block_size()) * ggml_type.type_size();
        assert_eq!(
            data.len(),
            out_features * row_bytes,
            "QuantizedLinear data length mismatch: got {}, expected {} rows × {} bytes",
            data.len(),
            out_features,
            row_bytes,
        );
        Self {
            data,
            ggml_type,
            in_features,
            out_features,
        }
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Forward: [batch, seq, in] → [batch, seq, out]
    ///
    /// Dequantizes weights on CPU, uploads to `device`, and runs matmul.
    pub fn forward<B: Backend>(&self, input: Tensor<B, 3>, device: &B::Device) -> Tensor<B, 3> {
        let [batch, seq, _] = input.dims();
        let input_data: Vec<f32> = input.to_data().to_vec().unwrap();
        let n_tokens = batch * seq;
        let output_data = quantized_matmul_f32(
            &self.data,
            self.ggml_type,
            self.out_features,
            self.in_features,
            &input_data,
            n_tokens,
        );
        let tensor_data = TensorData::new(output_data, [batch, seq, self.out_features]);
        Tensor::from_data(tensor_data, device)
    }

    /// Forward for 2D input: [n_tokens, in] → [n_tokens, out]
    pub fn forward_2d<B: Backend>(&self, input: Tensor<B, 2>, device: &B::Device) -> Tensor<B, 2> {
        let [n_tokens, _] = input.dims();
        let input_data: Vec<f32> = input.to_data().to_vec().unwrap();
        let output_data = quantized_matmul_f32(
            &self.data,
            self.ggml_type,
            self.out_features,
            self.in_features,
            &input_data,
            n_tokens,
        );
        let tensor_data = TensorData::new(output_data, [n_tokens, self.out_features]);
        Tensor::from_data(tensor_data, device)
    }
}

/// Matrix multiply without allocating a `QuantizedTensor`.
///
/// `data` is a row-major quantized weight matrix [out_features, in_features].
/// `input` is [n_tokens, in_features] in f32.
/// Returns [n_tokens, out_features] in f32.
fn quantized_matmul_f32(
    data: &[u8],
    ggml_type: GgmlType,
    out_features: usize,
    in_features: usize,
    input: &[f32],
    n_tokens: usize,
) -> Vec<f32> {
    let row_bytes = (in_features / ggml_type.block_size()) * ggml_type.type_size();
    let mut output = vec![0.0f32; n_tokens * out_features];
    let mut row_buf = vec![0.0f32; in_features];
    for out_idx in 0..out_features {
        let row_start = out_idx * row_bytes;
        dequantize_into(
            &data[row_start..row_start + row_bytes],
            ggml_type,
            &mut row_buf,
        );
        for tok in 0..n_tokens {
            let inp_start = tok * in_features;
            output[tok * out_features + out_idx] =
                dot_product(&input[inp_start..inp_start + in_features], &row_buf);
        }
    }
    output
}

/// Dequantize raw block data into an existing buffer.
fn dequantize_into(data: &[u8], ggml_type: GgmlType, output: &mut [f32]) {
    let result = dequantize(data, ggml_type, output.len()).expect("dequantize failed");
    output.copy_from_slice(&result);
}

/// Dot product of two f32 slices
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// GELU activation (approximate, matching PyTorch/Burn default)
fn gelu_f32(x: f32) -> f32 {
    // Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    // Approximation used by most frameworks:
    let cdf = 0.5 * (1.0 + ((0.797_884_6_f32 * (x + 0.044715 * x * x * x)).tanh()));
    x * cdf
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convert f32 to IEEE 754 binary16 (f16) bytes, little-endian
    fn f32_to_f16_bytes(val: f32) -> [u8; 2] {
        // Simple conversion: just use the half crate via burn's re-export
        // or do it manually with bit manipulation
        let bits = val.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let frac = bits & 0x7FFFFF;

        let h = if exp > 15 {
            // Overflow → infinity
            (sign << 15) | 0x7C00
        } else if exp < -14 {
            // Underflow → zero (skip denormals for test simplicity)
            sign << 15
        } else {
            let h_exp = ((exp + 15) as u32) & 0x1F;
            let h_frac = frac >> 13;
            (sign << 15) | (h_exp << 10) | h_frac
        };
        (h as u16).to_le_bytes()
    }

    fn make_q8_0_data(weights: &[f32]) -> Vec<u8> {
        // Q8_0: block_size=32, type_size=34 (f16 scale + 32 x i8)
        assert!(weights.len() % 32 == 0);
        let mut data = Vec::new();
        for block in weights.chunks(32) {
            let max_abs = block.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };
            data.extend_from_slice(&f32_to_f16_bytes(scale));
            for &val in block {
                let q = if scale > 0.0 {
                    (val / scale).round().clamp(-128.0, 127.0) as i8
                } else {
                    0i8
                };
                data.push(q as u8);
            }
        }
        data
    }

    #[test]
    fn test_quantized_tensor_matmul() {
        // Simple 2×4 weight matrix: identity-like with known values
        // out_features=2, in_features=32 (min for Q8_0)
        let mut row0 = vec![0.0f32; 32];
        let mut row1 = vec![0.0f32; 32];
        row0[0] = 1.0;
        row0[1] = 2.0;
        row1[0] = 3.0;
        row1[1] = 4.0;

        let mut weights = Vec::new();
        weights.extend_from_slice(&row0);
        weights.extend_from_slice(&row1);

        let data = make_q8_0_data(&weights);
        let qt = QuantizedTensor::new(data, GgmlType::Q8_0, 2, 32);

        // Input: 1 token × 32 features
        let mut input = vec![0.0f32; 32];
        input[0] = 1.0;
        input[1] = 1.0;

        let output = qt.matmul_f32(&input, 1);

        // row0 · input ≈ 1.0 + 2.0 = 3.0
        // row1 · input ≈ 3.0 + 4.0 = 7.0
        assert!(
            (output[0] - 3.0).abs() < 0.1,
            "got {} expected ~3.0",
            output[0]
        );
        assert!(
            (output[1] - 7.0).abs() < 0.1,
            "got {} expected ~7.0",
            output[1]
        );
    }

    #[test]
    fn test_gelu() {
        assert!((gelu_f32(0.0) - 0.0).abs() < 1e-6);
        assert!(gelu_f32(1.0) > 0.8);
        assert!(gelu_f32(-1.0).abs() < 0.2);
    }

    #[test]
    fn test_dot_product() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert!((dot_product(&a, &b) - 32.0).abs() < 1e-6);
    }
}
