//! Load tensors from .gguf files (llama.cpp quantized format)

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use burn::prelude::*;
use half::f16;
use memmap2::MmapOptions;
use thiserror::Error;

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" as bytes [0x47,0x47,0x55,0x46] read as LE u32
const GGUF_VERSION: u32 = 3;
const ALIGNMENT: usize = 32;

// ggml_type enum values
const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_Q4_0: u32 = 2;
const GGML_TYPE_Q4_1: u32 = 3;
const GGML_TYPE_Q5_0: u32 = 6;
const GGML_TYPE_Q5_1: u32 = 7;
const GGML_TYPE_Q8_0: u32 = 8;
const GGML_TYPE_Q4_K: u32 = 12;
const GGML_TYPE_Q5_K: u32 = 13;
const GGML_TYPE_Q6_K: u32 = 14;

#[derive(Error, Debug)]
pub enum GgufError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid GGUF magic: expected 0x46554747, got 0x{0:08X}")]
    InvalidMagic(u32),

    #[error("Unsupported GGUF version: {0} (expected {GGUF_VERSION})")]
    UnsupportedVersion(u32),

    #[error("Tensor not found: {0}")]
    TensorNotFound(String),

    #[error("Unsupported quantization type: {0}")]
    UnsupportedQuantType(u32),

    #[error("Unexpected end of file at offset {0}")]
    UnexpectedEof(usize),

    #[error("Invalid metadata value type: {0}")]
    InvalidValueType(u32),

    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
}

/// Quantization type for a GGUF tensor
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
}

impl GgmlType {
    fn from_u32(v: u32) -> Result<Self, GgufError> {
        match v {
            GGML_TYPE_F32 => Ok(Self::F32),
            GGML_TYPE_F16 => Ok(Self::F16),
            GGML_TYPE_Q4_0 => Ok(Self::Q4_0),
            GGML_TYPE_Q4_1 => Ok(Self::Q4_1),
            GGML_TYPE_Q5_0 => Ok(Self::Q5_0),
            GGML_TYPE_Q5_1 => Ok(Self::Q5_1),
            GGML_TYPE_Q8_0 => Ok(Self::Q8_0),
            GGML_TYPE_Q4_K => Ok(Self::Q4K),
            GGML_TYPE_Q5_K => Ok(Self::Q5K),
            GGML_TYPE_Q6_K => Ok(Self::Q6K),
            _ => Err(GgufError::UnsupportedQuantType(v)),
        }
    }

    /// Number of elements per quantization block
    fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 => 32,
            Self::Q4K | Self::Q5K | Self::Q6K => 256,
        }
    }

    /// Size in bytes of one quantization block
    fn type_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 => 2 + 16,               // f16 scale + 32 x 4-bit
            Self::Q4_1 => 2 + 2 + 16,           // f16 scale + f16 min + 32 x 4-bit
            Self::Q5_0 => 2 + 4 + 16,           // f16 scale + 32-bit high + 32 x 4-bit
            Self::Q5_1 => 2 + 2 + 4 + 16,       // f16 scale + f16 min + 32-bit high + 32 x 4-bit
            Self::Q8_0 => 2 + 32,               // f16 scale + 32 x i8
            Self::Q4K => 2 + 2 + 12 + 128,      // f16 d + f16 dmin + scales + quants = 144
            Self::Q5K => 2 + 2 + 12 + 128 + 32, // f16 d + f16 dmin + scales + ql + qh = 176
            Self::Q6K => 128 + 64 + 16 + 2,     // ql + qh + scales + f16 d = 210
        }
    }
}

/// Metadata value from a GGUF file
#[derive(Debug, Clone)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
}

/// Information about a tensor in the GGUF file
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub shape: Vec<usize>,
    pub ggml_type: GgmlType,
    /// Offset from the start of the tensor data section
    offset: u64,
}

/// A loaded GGUF file with memory-mapped data
pub struct GgufFile {
    _mmap: memmap2::Mmap,
    metadata: HashMap<String, MetadataValue>,
    tensors: HashMap<String, GgufTensorInfo>,
    /// Pointer to the start of the mmap
    data_ptr: *const u8,
    /// Total length of the mmap
    data_len: usize,
    /// Offset where tensor data begins (after header + metadata + tensor infos + padding)
    tensor_data_offset: usize,
}

// Safety: The mmap is read-only and lives as long as GgufFile
unsafe impl Send for GgufFile {}
unsafe impl Sync for GgufFile {}

/// Cursor into a byte slice for reading GGUF structures
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn check(&self, n: usize) -> Result<(), GgufError> {
        if self.remaining() < n {
            return Err(GgufError::UnexpectedEof(self.pos));
        }
        Ok(())
    }

    fn read_u8(&mut self) -> Result<u8, GgufError> {
        self.check(1)?;
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn read_i8(&mut self) -> Result<i8, GgufError> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16, GgufError> {
        self.check(2)?;
        let v = u16::from_le_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        self.pos += 2;
        Ok(v)
    }

    fn read_i16(&mut self) -> Result<i16, GgufError> {
        Ok(self.read_u16()? as i16)
    }

    fn read_u32(&mut self) -> Result<u32, GgufError> {
        self.check(4)?;
        let v = u32::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(v)
    }

    fn read_i32(&mut self) -> Result<i32, GgufError> {
        Ok(self.read_u32()? as i32)
    }

    fn read_u64(&mut self) -> Result<u64, GgufError> {
        self.check(8)?;
        let v = u64::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
            self.data[self.pos + 4],
            self.data[self.pos + 5],
            self.data[self.pos + 6],
            self.data[self.pos + 7],
        ]);
        self.pos += 8;
        Ok(v)
    }

    fn read_i64(&mut self) -> Result<i64, GgufError> {
        Ok(self.read_u64()? as i64)
    }

    fn read_f32(&mut self) -> Result<f32, GgufError> {
        let bits = self.read_u32()?;
        Ok(f32::from_bits(bits))
    }

    fn read_f64(&mut self) -> Result<f64, GgufError> {
        let bits = self.read_u64()?;
        Ok(f64::from_bits(bits))
    }

    fn read_bool(&mut self) -> Result<bool, GgufError> {
        Ok(self.read_u8()? != 0)
    }

    fn read_string(&mut self) -> Result<String, GgufError> {
        let len = self.read_u64()? as usize;
        self.check(len)?;
        let s = String::from_utf8_lossy(&self.data[self.pos..self.pos + len]).into_owned();
        self.pos += len;
        Ok(s)
    }

    fn read_metadata_value(&mut self) -> Result<MetadataValue, GgufError> {
        let value_type = self.read_u32()?;
        self.read_metadata_value_of_type(value_type)
    }

    fn read_metadata_value_of_type(&mut self, value_type: u32) -> Result<MetadataValue, GgufError> {
        match value_type {
            0 => Ok(MetadataValue::U8(self.read_u8()?)),
            1 => Ok(MetadataValue::I8(self.read_i8()?)),
            2 => Ok(MetadataValue::U16(self.read_u16()?)),
            3 => Ok(MetadataValue::I16(self.read_i16()?)),
            4 => Ok(MetadataValue::U32(self.read_u32()?)),
            5 => Ok(MetadataValue::I32(self.read_i32()?)),
            6 => Ok(MetadataValue::F32(self.read_f32()?)),
            7 => Ok(MetadataValue::Bool(self.read_bool()?)),
            8 => Ok(MetadataValue::String(self.read_string()?)),
            9 => {
                // Array: element_type (u32) + count (u64) + values
                let elem_type = self.read_u32()?;
                let count = self.read_u64()? as usize;
                let mut values = Vec::with_capacity(count);
                for _ in 0..count {
                    values.push(self.read_metadata_value_of_type(elem_type)?);
                }
                Ok(MetadataValue::Array(values))
            }
            10 => Ok(MetadataValue::U64(self.read_u64()?)),
            11 => Ok(MetadataValue::I64(self.read_i64()?)),
            12 => Ok(MetadataValue::F64(self.read_f64()?)),
            _ => Err(GgufError::InvalidValueType(value_type)),
        }
    }
}

impl GgufFile {
    /// Open and parse a GGUF file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, GgufError> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        let data_ptr = mmap.as_ptr();
        let data_len = mmap.len();

        let mut reader = Reader::new(&mmap);

        // Read header
        let magic = reader.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic(magic));
        }

        let version = reader.read_u32()?;
        if version != GGUF_VERSION {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count = reader.read_u64()? as usize;
        let metadata_kv_count = reader.read_u64()? as usize;

        // Read metadata
        let mut metadata = HashMap::with_capacity(metadata_kv_count);
        for _ in 0..metadata_kv_count {
            let key = reader.read_string()?;
            let value = reader.read_metadata_value()?;
            metadata.insert(key, value);
        }

        // Read tensor infos
        let mut tensors = HashMap::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = reader.read_string()?;
            let n_dims = reader.read_u32()? as usize;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(reader.read_u64()? as usize);
            }
            let type_id = reader.read_u32()?;
            let ggml_type = GgmlType::from_u32(type_id)?;
            let offset = reader.read_u64()?;

            tensors.insert(
                name,
                GgufTensorInfo {
                    shape,
                    ggml_type,
                    offset,
                },
            );
        }

        // Tensor data starts after padding to alignment boundary
        let tensor_data_offset = align_offset(reader.pos, ALIGNMENT);

        Ok(Self {
            _mmap: mmap,
            metadata,
            tensors,
            data_ptr,
            data_len,
            tensor_data_offset,
        })
    }

    /// Access metadata key-value pairs
    pub fn metadata(&self) -> &HashMap<String, MetadataValue> {
        &self.metadata
    }

    /// List all tensor names
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    /// Check if a tensor exists
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Get tensor info (shape and type)
    pub fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.get(name)
    }

    /// Load a tensor and dequantize to f32
    pub fn load_f32<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &B::Device,
    ) -> Result<Tensor<B, D>, GgufError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))?;

        if info.shape.len() != D {
            return Err(GgufError::ShapeMismatch {
                expected: vec![0; D],
                actual: info.shape.clone(),
            });
        }

        let n_elements: usize = info.shape.iter().product();
        let block_size = info.ggml_type.block_size();
        let n_blocks = n_elements / block_size;
        let data_size = n_blocks * info.ggml_type.type_size();

        let abs_offset = self.tensor_data_offset + info.offset as usize;
        if abs_offset + data_size > self.data_len {
            return Err(GgufError::UnexpectedEof(abs_offset + data_size));
        }

        // Safety: mmap is valid for lifetime of self
        let data = unsafe { std::slice::from_raw_parts(self.data_ptr.add(abs_offset), data_size) };

        let floats = dequantize(data, info.ggml_type, n_elements)?;

        let shape: [usize; D] = info.shape.clone().try_into().unwrap();
        let tensor_data = TensorData::new(floats, shape);

        Ok(Tensor::from_data(tensor_data, device))
    }

    /// Load a tensor with expected shape, dequantizing to f32
    pub fn load_f32_checked<B: Backend, const D: usize>(
        &self,
        name: &str,
        expected_shape: [usize; D],
        device: &B::Device,
    ) -> Result<Tensor<B, D>, GgufError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))?;

        if info.shape.as_slice() != expected_shape.as_slice() {
            return Err(GgufError::ShapeMismatch {
                expected: expected_shape.to_vec(),
                actual: info.shape.clone(),
            });
        }

        self.load_f32::<B, D>(name, device)
    }
}

fn align_offset(offset: usize, alignment: usize) -> usize {
    offset.div_ceil(alignment) * alignment
}

/// Dequantize raw block data to f32
fn dequantize(data: &[u8], ggml_type: GgmlType, n_elements: usize) -> Result<Vec<f32>, GgufError> {
    let mut output = vec![0.0f32; n_elements];

    match ggml_type {
        GgmlType::F32 => {
            for (i, chunk) in data.chunks_exact(4).enumerate() {
                output[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
        GgmlType::F16 => {
            for (i, chunk) in data.chunks_exact(2).enumerate() {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                output[i] = f16::from_bits(bits).to_f32();
            }
        }
        GgmlType::Q8_0 => dequantize_q8_0(data, &mut output),
        GgmlType::Q4_0 => dequantize_q4_0(data, &mut output),
        GgmlType::Q4_1 => dequantize_q4_1(data, &mut output),
        GgmlType::Q5_0 => dequantize_q5_0(data, &mut output),
        GgmlType::Q5_1 => dequantize_q5_1(data, &mut output),
        GgmlType::Q4K => dequantize_q4_k(data, &mut output),
        GgmlType::Q5K => dequantize_q5_k(data, &mut output),
        GgmlType::Q6K => dequantize_q6_k(data, &mut output),
    }

    Ok(output)
}

fn read_f16(data: &[u8], offset: usize) -> f32 {
    let bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
    f16::from_bits(bits).to_f32()
}

/// Q8_0: block of 32 values. Layout: f16 scale (2 bytes) + 32 x i8 quants
fn dequantize_q8_0(data: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 2 + 32; // 34 bytes per block

    for (block_idx, block) in data.chunks_exact(BLOCK_SIZE).enumerate() {
        let scale = read_f16(block, 0);
        let out = &mut output[block_idx * 32..(block_idx + 1) * 32];

        for i in 0..32 {
            let quant = block[2 + i] as i8;
            out[i] = scale * quant as f32;
        }
    }
}

/// Q4_0: block of 32 values. Layout: f16 scale (2 bytes) + 16 bytes (32 x 4-bit quants)
fn dequantize_q4_0(data: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 2 + 16; // 18 bytes per block

    for (block_idx, block) in data.chunks_exact(BLOCK_SIZE).enumerate() {
        let scale = read_f16(block, 0);
        let out = &mut output[block_idx * 32..(block_idx + 1) * 32];

        for i in 0..16 {
            let byte = block[2 + i];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;
            out[i] = scale * lo as f32;
            out[i + 16] = scale * hi as f32;
        }
    }
}

/// Q4_1: block of 32 values. Layout: f16 scale + f16 min + 16 bytes quants
fn dequantize_q4_1(data: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 2 + 2 + 16; // 20 bytes per block

    for (block_idx, block) in data.chunks_exact(BLOCK_SIZE).enumerate() {
        let scale = read_f16(block, 0);
        let min = read_f16(block, 2);
        let out = &mut output[block_idx * 32..(block_idx + 1) * 32];

        for i in 0..16 {
            let byte = block[4 + i];
            let lo = (byte & 0x0F) as f32;
            let hi = ((byte >> 4) & 0x0F) as f32;
            out[i] = scale * lo + min;
            out[i + 16] = scale * hi + min;
        }
    }
}

/// Q5_0: block of 32 values. Layout: f16 scale + 4 bytes high bits + 16 bytes quants
fn dequantize_q5_0(data: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 2 + 4 + 16; // 22 bytes per block

    for (block_idx, block) in data.chunks_exact(BLOCK_SIZE).enumerate() {
        let scale = read_f16(block, 0);
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let out = &mut output[block_idx * 32..(block_idx + 1) * 32];

        for i in 0..16 {
            let byte = block[6 + i];
            let lo_4 = (byte & 0x0F) as i32;
            let hi_4 = ((byte >> 4) & 0x0F) as i32;
            let lo_h = ((qh >> i) & 1) as i32;
            let hi_h = ((qh >> (i + 16)) & 1) as i32;
            out[i] = scale * ((lo_4 | (lo_h << 4)) - 16) as f32;
            out[i + 16] = scale * ((hi_4 | (hi_h << 4)) - 16) as f32;
        }
    }
}

/// Q5_1: block of 32 values. Layout: f16 scale + f16 min + 4 bytes high bits + 16 bytes quants
fn dequantize_q5_1(data: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 2 + 2 + 4 + 16; // 24 bytes per block

    for (block_idx, block) in data.chunks_exact(BLOCK_SIZE).enumerate() {
        let scale = read_f16(block, 0);
        let min = read_f16(block, 2);
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let out = &mut output[block_idx * 32..(block_idx + 1) * 32];

        for i in 0..16 {
            let byte = block[8 + i];
            let lo_4 = (byte & 0x0F) as u32;
            let hi_4 = ((byte >> 4) & 0x0F) as u32;
            let lo_h = (qh >> i) & 1;
            let hi_h = (qh >> (i + 16)) & 1;
            out[i] = scale * (lo_4 | (lo_h << 4)) as f32 + min;
            out[i + 16] = scale * (hi_4 | (hi_h << 4)) as f32 + min;
        }
    }
}

/// Q4_K: super-block of 256 values
/// Layout: f16 d + f16 dmin + 12 bytes scales_and_mins + 128 bytes quants
fn dequantize_q4_k(data: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 144;

    for (block_idx, block) in data.chunks_exact(BLOCK_SIZE).enumerate() {
        let d = read_f16(block, 0);
        let dmin = read_f16(block, 2);
        let scales_buf = &block[4..16];
        let quants = &block[16..144];
        let out = &mut output[block_idx * 256..(block_idx + 1) * 256];

        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        decode_q4k_scales_mins(scales_buf, &mut scales, &mut mins);

        for sub_block in 0..8 {
            let sc = scales[sub_block] as f32;
            let m = mins[sub_block] as f32;
            let base = sub_block * 32;
            let q_offset = sub_block * 16;

            for i in 0..16 {
                let byte = quants[q_offset + i];
                let lo = (byte & 0x0F) as f32;
                let hi = ((byte >> 4) & 0x0F) as f32;
                out[base + i] = d * sc * lo - dmin * m;
                out[base + i + 16] = d * sc * hi - dmin * m;
            }
        }
    }
}

/// Decode the packed 6-bit scales and mins from Q4_K/Q5_K 12-byte buffer.
///
/// The 12 bytes encode 8 scales and 8 mins. Layout (from ggml source):
/// - buf[0..4]: lower 6 bits of scales[0..4]
/// - buf[4..8]: lower 6 bits of mins[0..4]
/// - buf[8..12]: upper 2 bits of scales[4..8] and mins[4..8], packed with lower nibbles
fn decode_q4k_scales_mins(buf: &[u8], scales: &mut [u8; 8], mins: &mut [u8; 8]) {
    for i in 0..4 {
        scales[i] = buf[i] & 63;
        mins[i] = buf[i + 4] & 63;
    }
    for i in 4..8 {
        scales[i] = (buf[i + 4] & 0x0F) | ((buf[i - 4] >> 6) << 4);
        mins[i] = (buf[i + 4] >> 4) | ((buf[i] >> 6) << 4);
    }
}

/// Q5_K: super-block of 256 values
/// Layout: f16 d + f16 dmin + 12 bytes scales_and_mins + 128 bytes ql + 32 bytes qh
fn dequantize_q5_k(data: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 176;

    for (block_idx, block) in data.chunks_exact(BLOCK_SIZE).enumerate() {
        let d = read_f16(block, 0);
        let dmin = read_f16(block, 2);
        let scales_buf = &block[4..16];
        let ql = &block[16..144];
        let qh = &block[144..176];
        let out = &mut output[block_idx * 256..(block_idx + 1) * 256];

        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        decode_q4k_scales_mins(scales_buf, &mut scales, &mut mins);

        for sub_block in 0..8 {
            let sc = scales[sub_block] as f32;
            let m = mins[sub_block] as f32;
            let base = sub_block * 32;
            let q_offset = sub_block * 16;
            let qh_offset = sub_block * 4;

            for i in 0..16 {
                let byte = ql[q_offset + i];
                let lo = (byte & 0x0F) as u32;
                let hi = ((byte >> 4) & 0x0F) as u32;

                // Extract high bit from qh
                let qh_byte = qh[qh_offset + i / 4];
                let bit_lo = ((qh_byte >> (i % 4)) & 1) as u32;
                let bit_hi = ((qh_byte >> ((i % 4) + 4)) & 1) as u32;

                out[base + i] = d * sc * (lo | (bit_lo << 4)) as f32 - dmin * m;
                out[base + i + 16] = d * sc * (hi | (bit_hi << 4)) as f32 - dmin * m;
            }
        }
    }
}

/// Q6_K: super-block of 256 values
/// Layout: 128 bytes ql + 64 bytes qh + 16 bytes scales + f16 d
fn dequantize_q6_k(data: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 210;

    for (block_idx, block) in data.chunks_exact(BLOCK_SIZE).enumerate() {
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = read_f16(block, 208);
        let out = &mut output[block_idx * 256..(block_idx + 1) * 256];

        // 16 sub-blocks of 16 values each
        for (sub_block, &scale_byte) in scales.iter().enumerate() {
            let scale = scale_byte as i8 as f32;
            let base = sub_block * 16;

            for i in 0..16 {
                let idx = base + i;
                // ql has lower 4 bits, qh has upper 2 bits
                let ql_val = (ql[idx / 2] >> ((idx % 2) * 4)) & 0x0F;
                let qh_val = (qh[idx / 4] >> ((idx % 4) * 2)) & 0x03;
                let q = ((ql_val as i32) | ((qh_val as i32) << 4)) - 32;
                out[idx] = d * scale * q as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_offset() {
        assert_eq!(align_offset(0, 32), 0);
        assert_eq!(align_offset(1, 32), 32);
        assert_eq!(align_offset(31, 32), 32);
        assert_eq!(align_offset(32, 32), 32);
        assert_eq!(align_offset(33, 32), 64);
    }

    #[test]
    fn test_ggml_type_from_u32() {
        assert_eq!(GgmlType::from_u32(0).unwrap(), GgmlType::F32);
        assert_eq!(GgmlType::from_u32(1).unwrap(), GgmlType::F16);
        assert_eq!(GgmlType::from_u32(8).unwrap(), GgmlType::Q8_0);
        assert_eq!(GgmlType::from_u32(12).unwrap(), GgmlType::Q4K);
        assert_eq!(GgmlType::from_u32(13).unwrap(), GgmlType::Q5K);
        assert_eq!(GgmlType::from_u32(14).unwrap(), GgmlType::Q6K);
        assert!(GgmlType::from_u32(99).is_err());
    }

    #[test]
    fn test_dequantize_f32() {
        let values: Vec<f32> = vec![1.0, 2.0, -3.5, 0.0];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result = dequantize(&data, GgmlType::F32, 4).unwrap();
        assert_eq!(result, values);
    }

    #[test]
    fn test_dequantize_f16() {
        let values: Vec<f32> = vec![1.0, 2.0, -3.5, 0.0];
        let data: Vec<u8> = values
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_le_bytes())
            .collect();
        let result = dequantize(&data, GgmlType::F16, 4).unwrap();
        for (a, b) in result.iter().zip(values.iter()) {
            assert!((a - b).abs() < 0.01, "{a} != {b}");
        }
    }

    #[test]
    fn test_dequantize_q8_0() {
        // Build a Q8_0 block: f16 scale + 32 x i8 quants
        let scale = 0.5f32;
        let scale_f16 = f16::from_f32(scale);
        let mut block = Vec::new();
        block.extend_from_slice(&scale_f16.to_le_bytes());

        let quants: Vec<i8> = (0..32).map(|i| (i as i8) - 16).collect();
        for &q in &quants {
            block.push(q as u8);
        }

        let result = dequantize(&block, GgmlType::Q8_0, 32).unwrap();
        let scale_actual = scale_f16.to_f32();
        for (i, &q) in quants.iter().enumerate() {
            let expected = scale_actual * q as f32;
            assert!(
                (result[i] - expected).abs() < 1e-6,
                "index {i}: {} != {expected}",
                result[i]
            );
        }
    }

    #[test]
    fn test_dequantize_q4_0() {
        // Build a Q4_0 block: f16 scale + 16 bytes (32 x 4-bit quants)
        let scale = 1.0f32;
        let scale_f16 = f16::from_f32(scale);
        let mut block = Vec::new();
        block.extend_from_slice(&scale_f16.to_le_bytes());

        // 16 bytes, each containing two 4-bit values
        // lo nibble = lower 16 values, hi nibble = upper 16 values
        for i in 0..16 {
            let byte = (i as u8) | ((i as u8) << 4);
            block.push(byte);
        }

        let result = dequantize(&block, GgmlType::Q4_0, 32).unwrap();
        let scale_actual = scale_f16.to_f32();

        // Check first 16 (lo nibbles) — i is used both as index and as value in the formula
        #[allow(clippy::needless_range_loop)]
        for i in 0..16 {
            let expected = scale_actual * (i as i32 - 8) as f32;
            assert!(
                (result[i] - expected).abs() < 1e-6,
                "lo index {i}: {} != {expected}",
                result[i]
            );
        }
        // Check next 16 (hi nibbles)
        #[allow(clippy::needless_range_loop)]
        for i in 0..16 {
            let expected = scale_actual * (i as i32 - 8) as f32;
            assert!(
                (result[i + 16] - expected).abs() < 1e-6,
                "hi index {i}: {} != {expected}",
                result[i + 16]
            );
        }
    }

    #[test]
    fn test_dequantize_q4_k_zero_block() {
        // Q4_K block with scale=0 should produce all zeros
        let block = vec![0u8; 144];
        let result = dequantize(&block, GgmlType::Q4K, 256).unwrap();
        for (i, &v) in result.iter().enumerate() {
            assert!(v.abs() < 1e-6, "index {i}: expected 0.0, got {v}");
        }
    }

    #[test]
    fn test_dequantize_q6_k_zero_block() {
        // Q6_K block with d=0 should produce all zeros
        let block = vec![0u8; 210];
        let result = dequantize(&block, GgmlType::Q6K, 256).unwrap();
        for (i, &v) in result.iter().enumerate() {
            assert!(v.abs() < 1e-6, "index {i}: expected 0.0, got {v}");
        }
    }

    #[test]
    fn test_reader_string() {
        // GGUF string: u64 length + bytes
        let s = "hello";
        let mut data = Vec::new();
        data.extend_from_slice(&(s.len() as u64).to_le_bytes());
        data.extend_from_slice(s.as_bytes());

        let mut reader = Reader::new(&data);
        assert_eq!(reader.read_string().unwrap(), "hello");
    }

    #[test]
    fn test_reader_eof() {
        let data = [0u8; 2];
        let mut reader = Reader::new(&data);
        assert!(reader.read_u32().is_err());
    }

    #[test]
    fn test_decode_q4k_scales_mins() {
        let mut buf = [0u8; 12];
        // Set simple values for first 4 sub-blocks
        buf[0] = 10; // scale[0] = 10 (lower 6 bits)
        buf[1] = 20;
        buf[2] = 30;
        buf[3] = 40;
        buf[4] = 5; // min[0] = 5
        buf[5] = 15;
        buf[6] = 25;
        buf[7] = 35;

        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        decode_q4k_scales_mins(&buf, &mut scales, &mut mins);

        assert_eq!(scales[0], 10);
        assert_eq!(scales[1], 20);
        assert_eq!(scales[2], 30);
        assert_eq!(scales[3], 40);
        assert_eq!(mins[0], 5);
        assert_eq!(mins[1], 15);
        assert_eq!(mins[2], 25);
        assert_eq!(mins[3], 35);
    }

    #[test]
    fn test_block_sizes() {
        assert_eq!(GgmlType::F32.block_size(), 1);
        assert_eq!(GgmlType::F16.block_size(), 1);
        assert_eq!(GgmlType::Q8_0.block_size(), 32);
        assert_eq!(GgmlType::Q4K.block_size(), 256);
        assert_eq!(GgmlType::Q5K.block_size(), 256);
        assert_eq!(GgmlType::Q6K.block_size(), 256);
    }

    #[test]
    fn test_type_sizes() {
        assert_eq!(GgmlType::F32.type_size(), 4);
        assert_eq!(GgmlType::F16.type_size(), 2);
        assert_eq!(GgmlType::Q8_0.type_size(), 34);
        assert_eq!(GgmlType::Q4K.type_size(), 144);
        assert_eq!(GgmlType::Q5K.type_size(), 176);
        assert_eq!(GgmlType::Q6K.type_size(), 210);
    }

    #[test]
    fn test_dequantize_q8_0_known_values() {
        // Verify Q8_0 with scale=1.0, quants = [1, -1, 127, -128, 0, ...]
        let scale_f16 = f16::from_f32(1.0);
        let mut block = Vec::new();
        block.extend_from_slice(&scale_f16.to_le_bytes());

        let mut quants = vec![0i8; 32];
        quants[0] = 1;
        quants[1] = -1;
        quants[2] = 127;
        quants[3] = -128;
        for &q in &quants {
            block.push(q as u8);
        }

        let result = dequantize(&block, GgmlType::Q8_0, 32).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-3);
        assert!((result[1] - (-1.0)).abs() < 1e-3);
        assert!((result[2] - 127.0).abs() < 1e-3);
        assert!((result[3] - (-128.0)).abs() < 1e-3);
        assert!((result[4] - 0.0).abs() < 1e-3);
    }

    #[test]
    fn test_reader_metadata_types() {
        let mut data = Vec::new();

        // Type 4 = U32, value = 42
        data.extend_from_slice(&4u32.to_le_bytes());
        data.extend_from_slice(&42u32.to_le_bytes());

        let mut reader = Reader::new(&data);
        let val = reader.read_metadata_value().unwrap();
        match val {
            MetadataValue::U32(v) => assert_eq!(v, 42),
            other => panic!("expected U32, got {other:?}"),
        }
    }
}
