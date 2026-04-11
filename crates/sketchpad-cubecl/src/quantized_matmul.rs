//! GPU-side dequantization kernels for GGUF quantization formats.
//!
//! Two-step approach: dequantize packed bytes in a GPU kernel to a float tensor,
//! then use Burn's native matmul. This keeps quantized weights in VRAM as packed
//! i32 tensors, avoiding CPU dequantization and PCIe upload overhead.
//!
//! # Supported formats
//! - Q8_0: 34 bytes/block (f16 scale + 32 × int8), 32 values per block
//! - Q4_K: 144 bytes/super-block (f16 d + f16 dmin + 12 B scales/mins + 128 B nibbles), 256 values
//! - Q2_K: 84 bytes/super-block (16 B scales/mins + 64 B quants + f16 d + f16 dmin), 256 values
//! - Q3_K: 110 bytes/super-block (32 B hmask + 64 B quants + 12 B scales + f16 d), 256 values
//!
//! # Usage
//! ```ignore
//! let w = GpuQuantizedLinear::from_bytes(&bytes, GpuQuantType::Q4K, out_f, in_f, &client, &device, dtype);
//! let out = w.forward(input);  // input: CubeTensor [batch, in_f]; out: CubeTensor [batch, out_f]
//! ```

use burn::tensor::{DType, Shape};
use burn_cubecl::{
    CubeRuntime,
    kernel::matmul::{MatmulStrategy, matmul},
    ops::numeric::empty_device_dtype,
    tensor::CubeTensor,
};
use cubecl::prelude::*;
use cubecl::{CubeDim, calculate_cube_count_elemwise};

/// Quantization format selector for [`GpuQuantizedLinear`].
/// Only formats with GPU dequantization kernels are listed here.
/// Convert from `sketchpad_convert::gguf::GgmlType` at the call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuQuantType {
    /// Q8_0: 34 bytes/block (f16 scale + 32 × int8)
    Q8_0,
    /// Q4_K: 144 bytes/super-block (f16 d + f16 dmin + 12 B scales/mins + 128 B nibbles)
    Q4K,
    /// Q2_K: 84 bytes/super-block (16 B scales/mins nibbles + 64 B 2-bit quants + f16 d + f16 dmin)
    Q2K,
    /// Q3_K: 110 bytes/super-block (32 B high-bit mask + 64 B 2-bit low quants + 12 B 6-bit scales + f16 d)
    Q3K,
}

// ============================================================================
// Byte-level helpers (shared between Q8_0 and Q4_K kernels)
// ============================================================================

/// Read one byte from an i32-packed tensor (little-endian packing).
/// Byte at position `byte_pos` is in word `byte_pos/4`, at bit `(byte_pos%4)*8`.
#[cube]
fn read_byte(data: &Tensor<i32>, byte_pos: u32) -> u32 {
    let word_idx = byte_pos / 4u32;
    let bit_shift = (byte_pos % 4u32) * 8u32;
    let word = u32::reinterpret(data[word_idx]);
    (word >> bit_shift) & 0xFFu32
}

/// Decode two little-endian bytes at `byte_pos` as an f16 value, returning f32.
/// Uses bitwise conversion; only handles normal f16 values (zero returns 0.0).
/// GGUF scales are always normal, so NaN/Inf handling is not needed.
#[cube]
fn read_f16_le(data: &Tensor<i32>, byte_pos: u32) -> f32 {
    let lo = read_byte(data, byte_pos);
    let hi = read_byte(data, byte_pos + 1u32);
    let f16_bits = lo | (hi << 8u32);
    let sign = (f16_bits >> 15u32) & 1u32;
    let exp = (f16_bits >> 10u32) & 0x1Fu32;
    let frac = f16_bits & 0x3FFu32;
    let mut result = 0.0f32;
    if exp != 0u32 {
        // f32 bias (127) - f16 bias (15) = 112; shift mantissa left by 13 bits
        let f32_bits = (sign << 31u32) | ((exp + 112u32) << 23u32) | (frac << 13u32);
        result = f32::reinterpret(f32_bits);
    }
    result
}

// ============================================================================
// Q8_0 dequantization
// ============================================================================

/// Dequantize Q8_0 packed bytes to float.
/// One thread per output element. Output is flat [n_elements].
/// Block format: 2-byte f16 scale + 32 × int8 quants = 34 bytes/block.
#[cube(launch)]
fn dequantize_q8_0_kernel<F: Float>(
    packed: &Tensor<i32>,
    output: &mut Tensor<F>,
    #[define(F)] _dtype: StorageType,
) {
    let idx = ABSOLUTE_POS;
    if idx >= output.len() {
        terminate!();
    }

    let block_idx = idx / 32u32;
    let val_in_block = idx % 32u32;
    // Q8_0 block: 34 bytes = 2 (f16 scale) + 32 (int8 quants)
    let byte_offset = block_idx * 34u32;

    let scale = read_f16_le(packed, byte_offset);

    // Read quant byte and sign-extend from 8 bits
    let raw = read_byte(packed, byte_offset + 2u32 + val_in_block);
    // raw ∈ [0, 255]; interpret as signed int8: subtract 256 if ≥ 128
    // raw >> 7 is 0 for [0,127] and 1 for [128,255]
    let quant: f32 = f32::cast_from(i32::cast_from(raw) - i32::cast_from(raw >> 7u32) * 256i32);

    output[idx] = F::cast_from(scale * quant);
}

// ============================================================================
// Q4_K dequantization
// ============================================================================

/// Decode one 6-bit scale from the 12-byte Q4_K scales/mins buffer.
/// `buf_byte` = absolute byte offset of the 12-byte buffer in `packed`.
/// `sub` = sub-block index (0..8).
#[cube]
fn q4k_scale(packed: &Tensor<i32>, buf_byte: u32, sub: u32) -> u32 {
    let mut result = read_byte(packed, buf_byte + sub) & 63u32;
    if sub >= 4u32 {
        let i = sub - 4u32;
        let lo = read_byte(packed, buf_byte + 8u32 + i) & 0x0Fu32;
        let hi = (read_byte(packed, buf_byte + i) >> 6u32) << 4u32;
        result = lo | hi;
    }
    result
}

/// Decode one 6-bit min from the 12-byte Q4_K scales/mins buffer.
#[cube]
fn q4k_min(packed: &Tensor<i32>, buf_byte: u32, sub: u32) -> u32 {
    let mut result = read_byte(packed, buf_byte + 4u32 + sub) & 63u32;
    if sub >= 4u32 {
        let i = sub - 4u32;
        let lo = read_byte(packed, buf_byte + 8u32 + i) >> 4u32;
        let hi = (read_byte(packed, buf_byte + 4u32 + i) >> 6u32) << 4u32;
        result = lo | hi;
    }
    result
}

/// Dequantize Q4_K packed bytes to float.
/// One thread per output element. Output is flat [n_elements].
///
/// Super-block layout (144 bytes = 256 values):
/// - bytes 0..2:    f16 d (global scale)
/// - bytes 2..4:    f16 dmin (global min scale)
/// - bytes 4..16:   12-byte packed scales/mins (8 sub-block scales + 8 mins)
/// - bytes 16..144: 128 nibble bytes (256 × 4-bit quants, lo nibble first)
#[cube(launch)]
fn dequantize_q4k_kernel<F: Float>(
    packed: &Tensor<i32>,
    output: &mut Tensor<F>,
    #[define(F)] _dtype: StorageType,
) {
    let idx = ABSOLUTE_POS;
    if idx >= output.len() {
        terminate!();
    }

    // Map flat index to super-block + position within super-block
    let sblock_idx = idx / 256u32;
    let pos_in_sblock = idx % 256u32;

    // Super-block byte offset: 144 bytes each
    let sblock_byte = sblock_idx * 144u32;

    // Read global d and dmin
    let d = read_f16_le(packed, sblock_byte);
    let dmin = read_f16_le(packed, sblock_byte + 2u32);

    // 12-byte scales/mins buffer starts at byte 4 of the super-block
    let scales_buf_byte = sblock_byte + 4u32;

    // Each sub-block covers 32 values; 8 sub-blocks per super-block
    let sub = pos_in_sblock / 32u32;
    let pos_in_sub = pos_in_sblock % 32u32;

    let sc = q4k_scale(packed, scales_buf_byte, sub);
    let mn = q4k_min(packed, scales_buf_byte, sub);

    // Nibble data starts at byte 16 of the super-block
    // Within a sub-block of 32 values: bytes sub*16..(sub+1)*16
    // lo nibble → values 0..16, hi nibble → values 16..32 within sub-block
    let nibble_base = sblock_byte + 16u32 + sub * 16u32;
    let is_hi = pos_in_sub >= 16u32;
    // byte index within the 16 nibble bytes: same for both halves (pos % 16)
    let nibble_byte_offset = pos_in_sub % 16u32;
    let byte_val = read_byte(packed, nibble_base + nibble_byte_offset);
    let mut nibble = byte_val & 0x0Fu32;
    if is_hi {
        nibble = (byte_val >> 4u32) & 0x0Fu32;
    }

    // Dequantize: d * sc * nibble - dmin * mn
    let val = d * f32::cast_from(sc) * f32::cast_from(nibble) - dmin * f32::cast_from(mn);
    output[idx] = F::cast_from(val);
}

// ============================================================================
// Q2_K dequantization
// ============================================================================

/// Dequantize Q2_K packed bytes to float.
/// One thread per output element. Output is flat [n_elements].
///
/// Super-block layout (84 bytes = 256 values):
/// - bytes 0..16:  scales[16] — each byte: low 4 bits = scale nibble, high 4 bits = min nibble
/// - bytes 16..80: qs[64] — 256 × 2-bit quants, 4 per byte
/// - bytes 80..82: f16 d (super-block scale)
/// - bytes 82..84: f16 dmin (super-block min)
#[cube(launch)]
fn dequantize_q2k_kernel<F: Float>(
    packed: &Tensor<i32>,
    output: &mut Tensor<F>,
    #[define(F)] _dtype: StorageType,
) {
    let idx = ABSOLUTE_POS;
    if idx >= output.len() {
        terminate!();
    }

    let sblock_idx = idx / 256u32;
    let pos = idx % 256u32;

    // Super-block byte offset: 84 bytes each
    let sblock_byte = sblock_idx * 84u32;

    // Read super-block d and dmin (at bytes 80 and 82)
    let d = read_f16_le(packed, sblock_byte + 80u32);
    let dmin = read_f16_le(packed, sblock_byte + 82u32);

    // Sub-block index (16-element sub-blocks → 16 sub-blocks per super-block)
    let ib = pos / 16u32;

    // scales[ib]: low nibble = scale, high nibble = min
    let scale_byte = read_byte(packed, sblock_byte + ib);
    let sc = scale_byte & 0x0Fu32;
    let mn = (scale_byte >> 4u32) & 0x0Fu32;

    // 2-bit quant: qs starts at byte 16; 4 quants per byte
    let qs_byte = read_byte(packed, sblock_byte + 16u32 + pos / 4u32);
    let q = (qs_byte >> ((pos % 4u32) * 2u32)) & 0x3u32;

    // val = d * sc * q - dmin * mn
    let val = d * f32::cast_from(sc) * f32::cast_from(q) - dmin * f32::cast_from(mn);
    output[idx] = F::cast_from(val);
}

// ============================================================================
// Q3_K dequantization
// ============================================================================

/// Extract one 6-bit scale for sub-block `is` (0..15) from the Q3_K 12-byte scales buffer.
/// The buffer starts at `buf_byte` in `packed`.
///
/// Layout of the 12-byte buffer:
/// - bytes 0..8:  low 6 bits of scales[0..7] in a packed scheme
/// - bytes 8..12: upper bits packed in the high nibbles of bytes 0..7
///
/// The extraction follows the GGML utmp scheme:
/// - is 0..3:  byte[is] low 6 bits
/// - is 4..7:  byte[is-4+4] low 6 bits (i.e. byte[is]) -- different quadrant
/// - Actually: construct two 32-bit words w0, w1 from bytes 0..8 masked to 0x3f,
///             and w2, w3 from the same bytes shifted right 6 then masked.
///   scale[is]: is<4 → w0 byte[is], is<8 → w1 byte[is-4], is<12 → w2 byte[is-8], else w3 byte[is-12]
///   Signed value = raw - 32.
#[allow(unused_assignments)] // CubeCL requires branch-written variables to be initialized
#[cube]
fn q3k_scale_signed(packed: &Tensor<i32>, buf_byte: u32, is: u32) -> i32 {
    // Read all 8 bytes of the low-part scales buffer (bytes 0..8 of the 12-byte buffer)
    let b0 = read_byte(packed, buf_byte);
    let b1 = read_byte(packed, buf_byte + 1u32);
    let b2 = read_byte(packed, buf_byte + 2u32);
    let b3 = read_byte(packed, buf_byte + 3u32);
    let b4 = read_byte(packed, buf_byte + 4u32);
    let b5 = read_byte(packed, buf_byte + 5u32);
    let b6 = read_byte(packed, buf_byte + 6u32);
    let b7 = read_byte(packed, buf_byte + 7u32);

    // Four pseudo-words, each byte lane holds one 6-bit scale:
    // w0: low 6 bits of bytes 0..3  → scales[0..3]
    let w0 = (b0 & 0x3Fu32)
        | ((b1 & 0x3Fu32) << 8u32)
        | ((b2 & 0x3Fu32) << 16u32)
        | ((b3 & 0x3Fu32) << 24u32);
    // w1: low 6 bits of bytes 4..7  → scales[4..7]
    let w1 = (b4 & 0x3Fu32)
        | ((b5 & 0x3Fu32) << 8u32)
        | ((b6 & 0x3Fu32) << 16u32)
        | ((b7 & 0x3Fu32) << 24u32);
    // w2: bits [11:6] of bytes 0..3 → scales[8..11]
    let w2 = ((b0 >> 6u32) & 0x3Fu32)
        | (((b1 >> 6u32) & 0x3Fu32) << 8u32)
        | (((b2 >> 6u32) & 0x3Fu32) << 16u32)
        | (((b3 >> 6u32) & 0x3Fu32) << 24u32);
    // w3: bits [11:6] of bytes 4..7 → scales[12..15]
    let w3 = ((b4 >> 6u32) & 0x3Fu32)
        | (((b5 >> 6u32) & 0x3Fu32) << 8u32)
        | (((b6 >> 6u32) & 0x3Fu32) << 16u32)
        | (((b7 >> 6u32) & 0x3Fu32) << 24u32);

    // Extract the right byte lane; CubeCL requires 'mut' for variables written in branches
    let mut raw = 0u32;
    if is < 4u32 {
        raw = (w0 >> (is * 8u32)) & 0xFFu32;
    } else if is < 8u32 {
        raw = (w1 >> ((is - 4u32) * 8u32)) & 0xFFu32;
    } else if is < 12u32 {
        raw = (w2 >> ((is - 8u32) * 8u32)) & 0xFFu32;
    } else {
        raw = (w3 >> ((is - 12u32) * 8u32)) & 0xFFu32;
    }
    i32::cast_from(raw) - 32i32
}

/// Dequantize Q3_K packed bytes to float.
/// One thread per output element. Output is flat [n_elements].
///
/// Super-block layout (110 bytes = 256 values):
/// - bytes 0..32:   hmask[32] — high bit of each 3-bit quant: (hmask[i/8] >> (i%8)) & 1
/// - bytes 32..96:  qs[64]    — low 2 bits: (qs[i/4] >> (2*(i%4))) & 0x3
/// - bytes 96..108: scales[12] — 16 × 6-bit scales via utmp scheme
/// - bytes 108..110: f16 d (super-block scale)
#[cube(launch)]
fn dequantize_q3k_kernel<F: Float>(
    packed: &Tensor<i32>,
    output: &mut Tensor<F>,
    #[define(F)] _dtype: StorageType,
) {
    let idx = ABSOLUTE_POS;
    if idx >= output.len() {
        terminate!();
    }

    let sblock_idx = idx / 256u32;
    let pos = idx % 256u32;

    // Super-block byte offset: 110 bytes each
    let sblock_byte = sblock_idx * 110u32;

    // Read super-block d (at bytes 108..110)
    let d = read_f16_le(packed, sblock_byte + 108u32);

    // High bit from hmask[pos/8], bit (pos%8)
    let hmask_byte = read_byte(packed, sblock_byte + pos / 8u32);
    let hbit = (hmask_byte >> (pos % 8u32)) & 1u32;

    // Low 2 bits from qs[pos/4], bits (2*(pos%4))
    let qs_byte = read_byte(packed, sblock_byte + 32u32 + pos / 4u32);
    let lo2 = (qs_byte >> ((pos % 4u32) * 2u32)) & 0x3u32;

    // 3-bit raw value, then sign: raw=0..7 → signed_raw = raw - 4
    let raw3 = lo2 | (hbit << 2u32);
    let signed_raw = i32::cast_from(raw3) - 4i32;

    // 6-bit scale for this sub-block (16 elements per sub-block → 16 sub-blocks)
    let is = pos / 16u32;
    let scale_signed = q3k_scale_signed(packed, sblock_byte + 96u32, is);

    // val = d * scale_signed * signed_raw
    let val = d * f32::cast_from(scale_signed) * f32::cast_from(signed_raw);
    output[idx] = F::cast_from(val);
}

// ============================================================================
// Public dequantize functions
// ============================================================================

/// Dequantize a Q8_0 packed tensor (i32 storage) to a float tensor.
///
/// `packed` must contain the raw GGUF bytes packed as little-endian i32 words.
/// The output shape is `[out_features, in_features]` (row-major weight matrix).
pub fn dequantize_q8_0<R: CubeRuntime>(
    packed: CubeTensor<R>,
    out_features: usize,
    in_features: usize,
) -> CubeTensor<R> {
    let n_elements = out_features * in_features;
    let out_dtype = packed.dtype;
    let output = empty_device_dtype(
        packed.client.clone(),
        packed.device.clone(),
        Shape::new([n_elements]),
        out_dtype,
    );

    let cube_dim = CubeDim::new(&packed.client, n_elements);
    let cube_count = calculate_cube_count_elemwise(&packed.client, n_elements, cube_dim);

    dequantize_q8_0_kernel::launch::<R>(
        &packed.client,
        cube_count,
        cube_dim,
        packed.as_tensor_arg(1),
        output.as_tensor_arg(1),
        out_dtype.into(),
    )
    .expect("dequantize_q8_0 kernel launch failed");

    CubeTensor {
        shape: Shape::new([out_features, in_features]),
        strides: vec![in_features, 1],
        ..output
    }
}

/// Dequantize a Q4_K packed tensor (i32 storage) to a float tensor.
///
/// `packed` must contain the raw GGUF bytes packed as little-endian i32 words.
/// The output shape is `[out_features, in_features]` (row-major weight matrix).
pub fn dequantize_q4k<R: CubeRuntime>(
    packed: CubeTensor<R>,
    out_features: usize,
    in_features: usize,
) -> CubeTensor<R> {
    let n_elements = out_features * in_features;
    let out_dtype = packed.dtype;
    let output = empty_device_dtype(
        packed.client.clone(),
        packed.device.clone(),
        Shape::new([n_elements]),
        out_dtype,
    );

    let cube_dim = CubeDim::new(&packed.client, n_elements);
    let cube_count = calculate_cube_count_elemwise(&packed.client, n_elements, cube_dim);

    dequantize_q4k_kernel::launch::<R>(
        &packed.client,
        cube_count,
        cube_dim,
        packed.as_tensor_arg(1),
        output.as_tensor_arg(1),
        out_dtype.into(),
    )
    .expect("dequantize_q4k kernel launch failed");

    CubeTensor {
        shape: Shape::new([out_features, in_features]),
        strides: vec![in_features, 1],
        ..output
    }
}

/// Dequantize a Q2_K packed tensor (i32 storage) to a float tensor.
///
/// `packed` must contain the raw GGUF bytes packed as little-endian i32 words.
/// The output shape is `[out_features, in_features]` (row-major weight matrix).
pub fn dequantize_q2k<R: CubeRuntime>(
    packed: CubeTensor<R>,
    out_features: usize,
    in_features: usize,
) -> CubeTensor<R> {
    let n_elements = out_features * in_features;
    let out_dtype = packed.dtype;
    let output = empty_device_dtype(
        packed.client.clone(),
        packed.device.clone(),
        Shape::new([n_elements]),
        out_dtype,
    );

    let cube_dim = CubeDim::new(&packed.client, n_elements);
    let cube_count = calculate_cube_count_elemwise(&packed.client, n_elements, cube_dim);

    dequantize_q2k_kernel::launch::<R>(
        &packed.client,
        cube_count,
        cube_dim,
        packed.as_tensor_arg(1),
        output.as_tensor_arg(1),
        out_dtype.into(),
    )
    .expect("dequantize_q2k kernel launch failed");

    CubeTensor {
        shape: Shape::new([out_features, in_features]),
        strides: vec![in_features, 1],
        ..output
    }
}

/// Dequantize a Q3_K packed tensor (i32 storage) to a float tensor.
///
/// `packed` must contain the raw GGUF bytes packed as little-endian i32 words.
/// The output shape is `[out_features, in_features]` (row-major weight matrix).
pub fn dequantize_q3k<R: CubeRuntime>(
    packed: CubeTensor<R>,
    out_features: usize,
    in_features: usize,
) -> CubeTensor<R> {
    let n_elements = out_features * in_features;
    let out_dtype = packed.dtype;
    let output = empty_device_dtype(
        packed.client.clone(),
        packed.device.clone(),
        Shape::new([n_elements]),
        out_dtype,
    );

    let cube_dim = CubeDim::new(&packed.client, n_elements);
    let cube_count = calculate_cube_count_elemwise(&packed.client, n_elements, cube_dim);

    dequantize_q3k_kernel::launch::<R>(
        &packed.client,
        cube_count,
        cube_dim,
        packed.as_tensor_arg(1),
        output.as_tensor_arg(1),
        out_dtype.into(),
    )
    .expect("dequantize_q3k kernel launch failed");

    CubeTensor {
        shape: Shape::new([out_features, in_features]),
        strides: vec![in_features, 1],
        ..output
    }
}

// ============================================================================
// GpuQuantizedLinear
// ============================================================================

/// A linear projection with weights stored in VRAM as packed quantized bytes.
///
/// Weights are uploaded once (as raw bytes from the GGUF) and stored as an
/// i32 CubeTensor in VRAM. Per-token forward: dequantize → matmul.
///
/// This is roughly 4–8× VRAM savings vs f16 weights, at the cost of one
/// dequantization kernel per forward pass.
pub struct GpuQuantizedLinear<R: CubeRuntime> {
    /// Packed quantized bytes in VRAM (i32 storage, little-endian packing)
    data: CubeTensor<R>,
    /// Quantization format
    quant_type: GpuQuantType,
    out_features: usize,
    in_features: usize,
}

impl<R: CubeRuntime> GpuQuantizedLinear<R> {
    /// Upload raw GGUF bytes to VRAM and create a `GpuQuantizedLinear`.
    ///
    /// `bytes` is the raw tensor data from the GGUF file (little-endian GGML blocks).
    /// It will be zero-padded to the nearest word boundary and uploaded as i32.
    /// `dtype` is the float precision to use for dequantized output (f16 or f32).
    pub fn from_bytes(
        bytes: &[u8],
        quant_type: GpuQuantType,
        out_features: usize,
        in_features: usize,
        client: &cubecl::client::ComputeClient<R>,
        device: &R::Device,
        dtype: DType,
    ) -> Self {
        // Pack bytes as i32 words (little-endian, zero-pad to word boundary)
        let word_count = bytes.len().div_ceil(4);
        let mut word_bytes = vec![0u8; word_count * 4];
        word_bytes[..bytes.len()].copy_from_slice(bytes);

        let handle = client.create_from_slice(&word_bytes);
        let data = CubeTensor {
            client: client.clone(),
            handle,
            shape: Shape::new([word_count]),
            device: device.clone(),
            strides: vec![1],
            dtype,
            qparams: None,
        };

        Self {
            data,
            quant_type,
            out_features,
            in_features,
        }
    }

    /// Dequantize the weight matrix to a float `CubeTensor<R>` of shape
    /// `[out_features, in_features]`.
    pub fn dequantize(&self) -> CubeTensor<R> {
        match self.quant_type {
            GpuQuantType::Q8_0 => {
                dequantize_q8_0(self.data.clone(), self.out_features, self.in_features)
            }
            GpuQuantType::Q4K => {
                dequantize_q4k(self.data.clone(), self.out_features, self.in_features)
            }
            GpuQuantType::Q2K => {
                dequantize_q2k(self.data.clone(), self.out_features, self.in_features)
            }
            GpuQuantType::Q3K => {
                dequantize_q3k(self.data.clone(), self.out_features, self.in_features)
            }
        }
    }

    /// Run a linear projection on a 3D Burn tensor: [batch, seq, in] → [batch, seq, out].
    ///
    /// Converts to `CubeTensor<R>`, dequantizes, runs matmul, converts back.
    /// For use when integrating with Burn model code using `CubeBackend`.
    pub fn forward_burn<F, I, BT>(
        &self,
        input: burn::tensor::Tensor<burn_cubecl::CubeBackend<R, F, I, BT>, 3>,
    ) -> burn::tensor::Tensor<burn_cubecl::CubeBackend<R, F, I, BT>, 3>
    where
        F: burn_cubecl::FloatElement,
        I: burn_cubecl::IntElement,
        BT: burn_cubecl::BoolElement,
    {
        use crate::groupnorm_silu::{cube_to_tensor, tensor_to_cube};

        let [batch, seq, _] = input.dims();
        // Flatten to [batch*seq, in_features] for 2D matmul
        let flat: burn::tensor::Tensor<burn_cubecl::CubeBackend<R, F, I, BT>, 2> =
            input.reshape([batch * seq, self.in_features]);
        let cube_input = tensor_to_cube(flat);
        let cube_out = self.forward(cube_input);
        let out_flat: burn::tensor::Tensor<burn_cubecl::CubeBackend<R, F, I, BT>, 2> =
            cube_to_tensor(cube_out);
        out_flat.reshape([batch, seq, self.out_features])
    }

    /// Run a linear projection: `output = input @ weight.T`
    ///
    /// - `input` shape: `[batch, seq, in_features]` or `[seq, in_features]` (2D)
    /// - Returns same rank as input, last dim replaced with `out_features`
    pub fn forward(&self, input: CubeTensor<R>) -> CubeTensor<R> {
        // Dequantize: [out_features, in_features]
        let weight = self.dequantize();

        // Logical transpose: [in_features, out_features] (zero-copy, swap shape/strides)
        let weight_t = CubeTensor {
            shape: Shape::new([self.in_features, self.out_features]),
            strides: vec![weight.strides[1], weight.strides[0]],
            ..weight
        };

        let out_dtype = input.dtype;
        matmul(input, weight_t, None, MatmulStrategy::default(), out_dtype)
            .expect("GpuQuantizedLinear matmul failed")
    }
}
