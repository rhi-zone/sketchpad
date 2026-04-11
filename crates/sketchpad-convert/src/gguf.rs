//! Load tensors from .gguf files (llama.cpp quantized format)

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

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
const GGML_TYPE_Q2_K: u32 = 10;
const GGML_TYPE_Q3_K: u32 = 11;
const GGML_TYPE_Q4_K: u32 = 12;
const GGML_TYPE_Q5_K: u32 = 13;
const GGML_TYPE_Q6_K: u32 = 14;
const GGML_TYPE_IQ2_XXS: u32 = 16;
const GGML_TYPE_IQ2_XS: u32 = 17;
const GGML_TYPE_IQ3_XXS: u32 = 18;
const GGML_TYPE_IQ1_S: u32 = 19;
const GGML_TYPE_IQ4_NL: u32 = 20;
const GGML_TYPE_IQ3_S: u32 = 21;
const GGML_TYPE_IQ2_S: u32 = 22;
const GGML_TYPE_IQ4_XS: u32 = 23;
const GGML_TYPE_IQ1_M: u32 = 29;

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
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    IQ2Xxs,
    IQ2Xs,
    IQ3Xxs,
    IQ1S,
    IQ4Nl,
    IQ3S,
    IQ2S,
    IQ4Xs,
    IQ1M,
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
            GGML_TYPE_Q2_K => Ok(Self::Q2K),
            GGML_TYPE_Q3_K => Ok(Self::Q3K),
            GGML_TYPE_Q4_K => Ok(Self::Q4K),
            GGML_TYPE_Q5_K => Ok(Self::Q5K),
            GGML_TYPE_Q6_K => Ok(Self::Q6K),
            GGML_TYPE_IQ2_XXS => Ok(Self::IQ2Xxs),
            GGML_TYPE_IQ2_XS => Ok(Self::IQ2Xs),
            GGML_TYPE_IQ3_XXS => Ok(Self::IQ3Xxs),
            GGML_TYPE_IQ1_S => Ok(Self::IQ1S),
            GGML_TYPE_IQ4_NL => Ok(Self::IQ4Nl),
            GGML_TYPE_IQ3_S => Ok(Self::IQ3S),
            GGML_TYPE_IQ2_S => Ok(Self::IQ2S),
            GGML_TYPE_IQ4_XS => Ok(Self::IQ4Xs),
            GGML_TYPE_IQ1_M => Ok(Self::IQ1M),
            _ => Err(GgufError::UnsupportedQuantType(v)),
        }
    }

    /// Whether this quantization type can be dequantized directly on GPU via CubeCL kernels.
    pub fn is_gpu_quant_supported(self) -> bool {
        matches!(self, Self::Q4K | Self::Q8_0)
    }

    /// Number of elements per quantization block
    pub fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K => 256,
            Self::IQ2Xxs | Self::IQ2Xs | Self::IQ3Xxs | Self::IQ1S => 256,
            Self::IQ3S | Self::IQ2S | Self::IQ4Xs | Self::IQ1M => 256,
            Self::IQ4Nl => 32,
        }
    }

    /// Size in bytes of one quantization block
    pub fn type_size(self) -> usize {
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
            Self::Q2K => 16 + 64 + 2 + 2,       // scales + quants + f16 d + f16 dmin = 84
            Self::Q3K => 32 + 64 + 12 + 2,      // hmask + qs + scales + f16 d = 110
            Self::Q6K => 128 + 64 + 16 + 2,     // ql + qh + scales + f16 d = 210
            Self::IQ2Xxs => 66,                 // 2 + uint16_t[32] = 2+64
            Self::IQ2Xs => 74,                  // 2 + uint16_t[32] + uint8_t[8] = 2+64+8
            Self::IQ3Xxs => 98,                 // 2 + uint8_t[96] = 2+96
            Self::IQ1S => 50,                   // 2 + uint8_t[32] + uint16_t[8] = 2+32+16
            Self::IQ4Nl => 18,
            Self::IQ3S => 110, // 2 + uint8_t[64] + uint8_t[8] + uint8_t[32] + uint8_t[4] = 2+64+8+32+4
            Self::IQ2S => 82,  // 2 + uint8_t[64] + uint8_t[8] + uint8_t[8] = 2+64+8+8
            Self::IQ4Xs => 136,
            Self::IQ1M => 56, // uint8_t[32] + uint8_t[16] + uint8_t[8] = 32+16+8 (no d field)
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
    mmap: Arc<memmap2::Mmap>,
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
        let mmap = Arc::new(unsafe { MmapOptions::new().map(&file)? });

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

            // GGUF stores dimensions from fastest-varying (innermost) to slowest.
            // Burn and most frameworks use the opposite convention: [slowest, ..., fastest].
            // Reverse the shape so callers get standard row-major dimension ordering.
            shape.reverse();

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
            mmap,
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

    /// Get raw tensor data bytes and info without dequantizing
    pub fn tensor_raw_data(&self, name: &str) -> Result<(&[u8], &GgufTensorInfo), GgufError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))?;

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

        Ok((data, info))
    }

    /// Estimate VRAM bytes needed under two loading strategies.
    ///
    /// `as_f16`: total bytes if all tensors are dequantized to f16 (standard load).
    /// `as_quantized`: total raw bytes if quantized tensors stay quantized (GPU quant / CPU offload).
    ///
    /// These are weight-only estimates; add ~10–20% headroom for KV cache and activations.
    pub fn estimated_vram_bytes(&self) -> (u64, u64) {
        let mut as_f16: u64 = 0;
        let mut as_quantized: u64 = 0;
        for info in self.tensors.values() {
            let n_elements: u64 = info.shape.iter().map(|&d| d as u64).product();
            let n_blocks = n_elements / info.ggml_type.block_size() as u64;
            let raw_bytes = n_blocks * info.ggml_type.type_size() as u64;
            as_quantized += raw_bytes;
            as_f16 += n_elements * 2; // f16 = 2 bytes/element
        }
        (as_f16, as_quantized)
    }

    /// Return an Arc clone of the memory-mapped file, for zero-copy sharing.
    pub fn mmap_arc(&self) -> Arc<memmap2::Mmap> {
        Arc::clone(&self.mmap)
    }

    /// Return the byte range (absolute offset, length) for a tensor without borrowing.
    pub fn tensor_byte_range(&self, name: &str) -> Result<(usize, usize), GgufError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))?;
        let n_elements: usize = info.shape.iter().product();
        let n_blocks = n_elements / info.ggml_type.block_size();
        let data_size = n_blocks * info.ggml_type.type_size();
        let abs_offset = self.tensor_data_offset + info.offset as usize;
        Ok((abs_offset, data_size))
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
pub fn dequantize(
    data: &[u8],
    ggml_type: GgmlType,
    n_elements: usize,
) -> Result<Vec<f32>, GgufError> {
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
        GgmlType::Q2K => dequantize_q2_k(data, &mut output),
        GgmlType::Q3K => dequantize_q3_k(data, &mut output),
        GgmlType::Q4K => dequantize_q4_k(data, &mut output),
        GgmlType::Q5K => dequantize_q5_k(data, &mut output),
        GgmlType::Q6K => dequantize_q6_k(data, &mut output),
        GgmlType::IQ4Nl => dequantize_iq4_nl(data, &mut output),
        GgmlType::IQ4Xs => dequantize_iq4_xs(data, &mut output),
        GgmlType::IQ2Xxs => dequantize_iq2_xxs(data, &mut output),
        GgmlType::IQ2Xs => dequantize_iq2_xs(data, &mut output),
        GgmlType::IQ3Xxs => dequantize_iq3_xxs(data, &mut output),
        GgmlType::IQ1S => dequantize_iq1_s(data, &mut output),
        GgmlType::IQ3S => dequantize_iq3_s(data, &mut output),
        GgmlType::IQ2S => dequantize_iq2_s(data, &mut output),
        GgmlType::IQ1M => dequantize_iq1_m(data, &mut output),
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

/// Q2_K: super-block of 256 values
/// Layout: 16 bytes scales (4-bit scale + 4-bit min packed) + 64 bytes quants (2-bit each) + f16 d + f16 dmin
fn dequantize_q2_k(data: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 84;

    for (block_idx, block) in data.chunks_exact(BLOCK_SIZE).enumerate() {
        let scales_buf = &block[0..16];
        let quants = &block[16..80];
        let d = read_f16(block, 80);
        let dmin = read_f16(block, 82);
        let out = &mut output[block_idx * 256..(block_idx + 1) * 256];

        for i in 0..256 {
            let ib = i / 16; // which 16-element sub-block (0..15)
            let scale_nibble = (scales_buf[ib] & 0x0F) as f32;
            let min_nibble = ((scales_buf[ib] >> 4) & 0x0F) as f32;
            let q = ((quants[i / 4] >> (2 * (i % 4))) & 0x3) as f32;
            out[i] = d * scale_nibble * q - dmin * min_nibble;
        }
    }
}

/// Q3_K: super-block of 256 values
/// Layout: 32 bytes hmask + 64 bytes qs (lower 2 bits) + 12 bytes scales (6-bit packed) + f16 d
fn dequantize_q3_k(data: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 110;

    for (block_idx, block) in data.chunks_exact(BLOCK_SIZE).enumerate() {
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let scales_raw = &block[96..108];
        let d = read_f16(block, 108);
        let out = &mut output[block_idx * 256..(block_idx + 1) * 256];

        // Decode 16 x 6-bit scales from 12 bytes using the llama.cpp packing.
        // The 12 bytes are split into two groups of 8 (indices 0..8). Each group
        // yields 8 scales via two 32-bit words: lower 6 bits from bytes 0..4, and
        // the next 6 bits (shifted >> 6) from bytes 0..4 again (masked differently).
        let utmp: [u32; 4] = [
            (scales_raw[0] as u32
                | ((scales_raw[1] as u32) << 8)
                | ((scales_raw[2] as u32) << 16)
                | ((scales_raw[3] as u32) << 24))
                & 0x3f3f3f3f,
            (scales_raw[4] as u32
                | ((scales_raw[5] as u32) << 8)
                | ((scales_raw[6] as u32) << 16)
                | ((scales_raw[7] as u32) << 24))
                & 0x3f3f3f3f,
            ((scales_raw[0] as u32
                | ((scales_raw[1] as u32) << 8)
                | ((scales_raw[2] as u32) << 16)
                | ((scales_raw[3] as u32) << 24))
                >> 6)
                & 0x3f3f3f3f,
            ((scales_raw[4] as u32
                | ((scales_raw[5] as u32) << 8)
                | ((scales_raw[6] as u32) << 16)
                | ((scales_raw[7] as u32) << 24))
                >> 6)
                & 0x3f3f3f3f,
        ];

        // Extract the 16 6-bit scales (each byte of utmp[i] holds one scale).
        let mut scales = [0i32; 16];
        for i in 0..4 {
            for byte_idx in 0..4 {
                let raw = ((utmp[i] >> (byte_idx * 8)) & 0xFF) as i32;
                scales[i * 4 + byte_idx] = raw - 32;
            }
        }

        for i in 0..256 {
            let low2 = (qs[i / 4] >> (2 * (i % 4))) & 0x3;
            let hbit = (hmask[i / 8] >> (i % 8)) & 1;
            let raw_quant = (low2 as i32 | ((hbit as i32) << 2)) - 4;
            let is = i / 16; // scale index (0..15)
            let scale = scales[is] as f32;
            out[i] = d * scale * raw_quant as f32;
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

const IQ4NL_VALUES: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// IQ4_NL: block of 32 values. Layout: f16 scale (2 bytes) + 16 bytes of 4-bit packed quants
fn dequantize_iq4_nl(data: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 18;

    for (block_idx, block) in data.chunks_exact(BLOCK_SIZE).enumerate() {
        let d = read_f16(block, 0);
        let qs = &block[2..18];
        let out = &mut output[block_idx * 32..(block_idx + 1) * 32];

        for i in 0..32 {
            let byte = qs[i / 2];
            let nibble = if i % 2 == 0 {
                byte & 0xf
            } else {
                (byte >> 4) & 0xf
            };
            out[i] = d * IQ4NL_VALUES[nibble as usize] as f32;
        }
    }
}

/// IQ4_XS: super-block of 256 values. Layout: f16 d (2 bytes) + 1 byte scales_h +
/// 4 bytes scales_l + 128 bytes 4-bit packed quants
fn dequantize_iq4_xs(data: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 136;

    for (block_idx, block) in data.chunks_exact(BLOCK_SIZE).enumerate() {
        let d = read_f16(block, 0);
        let scales_h = block[2];
        let scales_l = &block[3..7];
        let qs = &block[7..135];
        let out = &mut output[block_idx * 256..(block_idx + 1) * 256];

        for sb in 0..8usize {
            let low4 = (scales_l[sb / 2] >> (4 * (sb % 2))) & 0xf;
            let high2 = (scales_h >> (2 * sb)) & 0x3;
            let scale_raw = (high2 << 4) | low4;
            let signed_scale = (scale_raw as i32) - 32;

            for pos in 0..32usize {
                let byte = qs[sb * 16 + pos / 2];
                let nibble = if pos % 2 == 0 {
                    byte & 0xf
                } else {
                    (byte >> 4) & 0xf
                };
                out[sb * 32 + pos] =
                    d * (signed_scale as f32) * IQ4NL_VALUES[nibble as usize] as f32;
            }
        }
    }
}

// Sign mask lookup: for a 7-bit index, gives which of 8 values are negative
const KSIGNS_IQ2XS: [u8; 128] = [
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15, 144, 17, 18, 147, 20, 149,
    150, 23, 24, 153, 154, 27, 156, 29, 30, 159, 160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170,
    43, 172, 45, 46, 175, 48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207, 80, 209, 210, 83, 212,
    85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95, 96, 225, 226, 99, 228, 101, 102, 231, 232,
    105, 106, 235, 108, 237, 238, 111, 240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123,
    252, 125, 126, 255,
];

const KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

const IQ1S_DELTA: f32 = 0.125;
const IQ1M_DELTA: f32 = 0.125;

const IQ2XXS_GRID: [u64; 256] = [
    0x0808080808080808,
    0x080808080808082b,
    0x0808080808081919,
    0x0808080808082b08,
    0x0808080808082b2b,
    0x0808080808190819,
    0x0808080808191908,
    0x08080808082b0808,
    0x08080808082b082b,
    0x08080808082b2b08,
    0x08080808082b2b2b,
    0x0808080819080819,
    0x0808080819081908,
    0x0808080819190808,
    0x0808080819192b08,
    0x08080808192b0819,
    0x08080808192b1908,
    0x080808082b080808,
    0x080808082b08082b,
    0x080808082b082b2b,
    0x080808082b2b082b,
    0x0808081908080819,
    0x0808081908081908,
    0x0808081908190808,
    0x0808081908191919,
    0x0808081919080808,
    0x080808192b081908,
    0x080808192b192b08,
    0x0808082b08080808,
    0x0808082b0808082b,
    0x0808082b082b082b,
    0x0808082b2b08082b,
    0x0808190808080819,
    0x0808190808081908,
    0x0808190808190808,
    0x08081908082b0819,
    0x08081908082b1908,
    0x0808190819080808,
    0x080819081908082b,
    0x0808190819082b08,
    0x08081908192b0808,
    0x080819082b080819,
    0x080819082b081908,
    0x080819082b190808,
    0x080819082b2b1908,
    0x0808191908080808,
    0x080819190808082b,
    0x0808191908082b08,
    0x08081919082b0808,
    0x080819191908192b,
    0x08081919192b2b19,
    0x080819192b080808,
    0x080819192b190819,
    0x0808192b08082b19,
    0x0808192b08190808,
    0x0808192b19080808,
    0x0808192b2b081908,
    0x0808192b2b2b1908,
    0x08082b0808080808,
    0x08082b0808081919,
    0x08082b0808082b08,
    0x08082b0808191908,
    0x08082b08082b2b08,
    0x08082b0819080819,
    0x08082b0819081908,
    0x08082b0819190808,
    0x08082b081919082b,
    0x08082b082b082b08,
    0x08082b1908081908,
    0x08082b1919080808,
    0x08082b2b0808082b,
    0x08082b2b08191908,
    0x0819080808080819,
    0x0819080808081908,
    0x0819080808190808,
    0x08190808082b0819,
    0x0819080819080808,
    0x08190808192b0808,
    0x081908082b081908,
    0x081908082b190808,
    0x081908082b191919,
    0x0819081908080808,
    0x0819081908082b08,
    0x08190819082b0808,
    0x0819081919190808,
    0x0819081919192b2b,
    0x081908192b080808,
    0x0819082b082b1908,
    0x0819082b19081919,
    0x0819190808080808,
    0x0819190808082b08,
    0x08191908082b0808,
    0x08191908082b1919,
    0x0819190819082b19,
    0x081919082b080808,
    0x0819191908192b08,
    0x08191919192b082b,
    0x0819192b08080808,
    0x0819192b0819192b,
    0x08192b0808080819,
    0x08192b0808081908,
    0x08192b0808190808,
    0x08192b0819080808,
    0x08192b082b080819,
    0x08192b1908080808,
    0x08192b1908081919,
    0x08192b192b2b0808,
    0x08192b2b19190819,
    0x082b080808080808,
    0x082b08080808082b,
    0x082b080808082b2b,
    0x082b080819081908,
    0x082b0808192b0819,
    0x082b08082b080808,
    0x082b08082b08082b,
    0x082b0819082b2b19,
    0x082b081919082b08,
    0x082b082b08080808,
    0x082b082b0808082b,
    0x082b190808080819,
    0x082b190808081908,
    0x082b190808190808,
    0x082b190819080808,
    0x082b19081919192b,
    0x082b191908080808,
    0x082b191919080819,
    0x082b1919192b1908,
    0x082b192b2b190808,
    0x082b2b0808082b08,
    0x082b2b08082b0808,
    0x082b2b082b191908,
    0x082b2b2b19081908,
    0x1908080808080819,
    0x1908080808081908,
    0x1908080808190808,
    0x1908080808192b08,
    0x19080808082b0819,
    0x19080808082b1908,
    0x1908080819080808,
    0x1908080819082b08,
    0x190808081919192b,
    0x19080808192b0808,
    0x190808082b080819,
    0x190808082b081908,
    0x190808082b190808,
    0x1908081908080808,
    0x19080819082b0808,
    0x19080819192b0819,
    0x190808192b080808,
    0x190808192b081919,
    0x1908082b08080819,
    0x1908082b08190808,
    0x1908082b19082b08,
    0x1908082b1919192b,
    0x1908082b192b2b08,
    0x1908190808080808,
    0x1908190808082b08,
    0x19081908082b0808,
    0x190819082b080808,
    0x190819082b192b19,
    0x190819190819082b,
    0x19081919082b1908,
    0x1908192b08080808,
    0x19082b0808080819,
    0x19082b0808081908,
    0x19082b0808190808,
    0x19082b0819080808,
    0x19082b0819081919,
    0x19082b1908080808,
    0x19082b1919192b08,
    0x19082b19192b0819,
    0x19082b192b08082b,
    0x19082b2b19081919,
    0x19082b2b2b190808,
    0x1919080808080808,
    0x1919080808082b08,
    0x1919080808190819,
    0x1919080808192b19,
    0x19190808082b0808,
    0x191908082b080808,
    0x191908082b082b08,
    0x1919081908081908,
    0x191908191908082b,
    0x191908192b2b1908,
    0x1919082b2b190819,
    0x191919082b190808,
    0x191919082b19082b,
    0x1919191908082b2b,
    0x1919192b08080819,
    0x1919192b19191908,
    0x19192b0808080808,
    0x19192b0808190819,
    0x19192b0808192b19,
    0x19192b08192b1908,
    0x19192b1919080808,
    0x19192b2b08082b08,
    0x192b080808081908,
    0x192b080808190808,
    0x192b080819080808,
    0x192b0808192b2b08,
    0x192b081908080808,
    0x192b081919191919,
    0x192b082b08192b08,
    0x192b082b192b0808,
    0x192b190808080808,
    0x192b190808081919,
    0x192b191908190808,
    0x192b19190819082b,
    0x192b19192b081908,
    0x192b2b081908082b,
    0x2b08080808080808,
    0x2b0808080808082b,
    0x2b08080808082b2b,
    0x2b08080819080819,
    0x2b0808082b08082b,
    0x2b08081908081908,
    0x2b08081908192b08,
    0x2b08081919080808,
    0x2b08082b08190819,
    0x2b08190808080819,
    0x2b08190808081908,
    0x2b08190808190808,
    0x2b08190808191919,
    0x2b08190819080808,
    0x2b081908192b0808,
    0x2b08191908080808,
    0x2b0819191908192b,
    0x2b0819192b191908,
    0x2b08192b08082b19,
    0x2b08192b19080808,
    0x2b08192b192b0808,
    0x2b082b080808082b,
    0x2b082b1908081908,
    0x2b082b2b08190819,
    0x2b19080808081908,
    0x2b19080808190808,
    0x2b190808082b1908,
    0x2b19080819080808,
    0x2b1908082b2b0819,
    0x2b1908190819192b,
    0x2b1908192b080808,
    0x2b19082b19081919,
    0x2b19190808080808,
    0x2b191908082b082b,
    0x2b19190819081908,
    0x2b19191919190819,
    0x2b192b082b080819,
    0x2b192b19082b0808,
    0x2b2b08080808082b,
    0x2b2b080819190808,
    0x2b2b08082b081919,
    0x2b2b081908082b19,
    0x2b2b082b08080808,
    0x2b2b190808192b08,
    0x2b2b2b0819190808,
    0x2b2b2b1908081908,
];

const IQ2XS_GRID: [u64; 512] = [
    0x0808080808080808,
    0x080808080808082b,
    0x0808080808081919,
    0x0808080808082b08,
    0x0808080808082b2b,
    0x0808080808190819,
    0x0808080808191908,
    0x080808080819192b,
    0x0808080808192b19,
    0x08080808082b0808,
    0x08080808082b082b,
    0x08080808082b1919,
    0x08080808082b2b08,
    0x0808080819080819,
    0x0808080819081908,
    0x080808081908192b,
    0x0808080819082b19,
    0x0808080819190808,
    0x080808081919082b,
    0x0808080819191919,
    0x0808080819192b08,
    0x08080808192b0819,
    0x08080808192b1908,
    0x080808082b080808,
    0x080808082b08082b,
    0x080808082b081919,
    0x080808082b082b08,
    0x080808082b190819,
    0x080808082b191908,
    0x080808082b192b19,
    0x080808082b2b0808,
    0x0808081908080819,
    0x0808081908081908,
    0x080808190808192b,
    0x0808081908082b19,
    0x0808081908190808,
    0x080808190819082b,
    0x0808081908191919,
    0x0808081908192b08,
    0x0808081908192b2b,
    0x08080819082b0819,
    0x08080819082b1908,
    0x0808081919080808,
    0x080808191908082b,
    0x0808081919081919,
    0x0808081919082b08,
    0x0808081919190819,
    0x0808081919191908,
    0x08080819192b0808,
    0x08080819192b2b08,
    0x080808192b080819,
    0x080808192b081908,
    0x080808192b190808,
    0x0808082b08080808,
    0x0808082b0808082b,
    0x0808082b08081919,
    0x0808082b08082b08,
    0x0808082b08190819,
    0x0808082b08191908,
    0x0808082b082b0808,
    0x0808082b19080819,
    0x0808082b19081908,
    0x0808082b19190808,
    0x0808082b19191919,
    0x0808082b2b080808,
    0x0808082b2b082b2b,
    0x0808190808080819,
    0x0808190808081908,
    0x080819080808192b,
    0x0808190808082b19,
    0x0808190808190808,
    0x080819080819082b,
    0x0808190808191919,
    0x0808190808192b08,
    0x08081908082b0819,
    0x08081908082b1908,
    0x0808190819080808,
    0x080819081908082b,
    0x0808190819081919,
    0x0808190819082b08,
    0x0808190819190819,
    0x0808190819191908,
    0x080819081919192b,
    0x08081908192b0808,
    0x080819082b080819,
    0x080819082b081908,
    0x080819082b190808,
    0x0808191908080808,
    0x080819190808082b,
    0x0808191908081919,
    0x0808191908082b08,
    0x0808191908190819,
    0x0808191908191908,
    0x08081919082b0808,
    0x0808191919080819,
    0x0808191919081908,
    0x0808191919190808,
    0x08081919192b0819,
    0x080819192b080808,
    0x0808192b08080819,
    0x0808192b08081908,
    0x0808192b08190808,
    0x0808192b082b192b,
    0x0808192b19080808,
    0x0808192b1908082b,
    0x0808192b2b081908,
    0x08082b0808080808,
    0x08082b080808082b,
    0x08082b0808081919,
    0x08082b0808082b08,
    0x08082b0808082b2b,
    0x08082b0808190819,
    0x08082b0808191908,
    0x08082b08082b0808,
    0x08082b08082b1919,
    0x08082b0819080819,
    0x08082b0819081908,
    0x08082b0819190808,
    0x08082b0819192b08,
    0x08082b082b080808,
    0x08082b082b2b0808,
    0x08082b082b2b2b2b,
    0x08082b1908080819,
    0x08082b1908081908,
    0x08082b1908190808,
    0x08082b1919080808,
    0x08082b192b080819,
    0x08082b192b082b19,
    0x08082b2b08080808,
    0x08082b2b082b0808,
    0x08082b2b082b2b08,
    0x08082b2b2b19192b,
    0x08082b2b2b2b0808,
    0x0819080808080819,
    0x0819080808081908,
    0x081908080808192b,
    0x0819080808082b19,
    0x0819080808190808,
    0x081908080819082b,
    0x0819080808191919,
    0x0819080808192b08,
    0x08190808082b0819,
    0x08190808082b1908,
    0x0819080819080808,
    0x081908081908082b,
    0x0819080819081919,
    0x0819080819082b08,
    0x0819080819190819,
    0x0819080819191908,
    0x08190808192b0808,
    0x08190808192b2b2b,
    0x081908082b080819,
    0x081908082b081908,
    0x081908082b190808,
    0x0819081908080808,
    0x081908190808082b,
    0x0819081908081919,
    0x0819081908082b08,
    0x0819081908190819,
    0x0819081908191908,
    0x08190819082b0808,
    0x0819081919080819,
    0x0819081919081908,
    0x0819081919190808,
    0x081908192b080808,
    0x081908192b191908,
    0x081908192b19192b,
    0x0819082b08080819,
    0x0819082b08081908,
    0x0819082b0808192b,
    0x0819082b08190808,
    0x0819082b19080808,
    0x0819082b192b0808,
    0x0819190808080808,
    0x081919080808082b,
    0x0819190808081919,
    0x0819190808082b08,
    0x0819190808190819,
    0x0819190808191908,
    0x08191908082b0808,
    0x0819190819080819,
    0x0819190819081908,
    0x0819190819082b19,
    0x0819190819190808,
    0x08191908192b1908,
    0x081919082b080808,
    0x0819191908080819,
    0x0819191908081908,
    0x0819191908190808,
    0x0819191919080808,
    0x0819192b08080808,
    0x0819192b08191908,
    0x0819192b19082b19,
    0x08192b0808080819,
    0x08192b0808081908,
    0x08192b0808190808,
    0x08192b080819082b,
    0x08192b0819080808,
    0x08192b0819191908,
    0x08192b082b08192b,
    0x08192b1908080808,
    0x08192b1908081919,
    0x08192b19192b192b,
    0x08192b2b19190819,
    0x08192b2b2b2b2b19,
    0x082b080808080808,
    0x082b08080808082b,
    0x082b080808081919,
    0x082b080808082b08,
    0x082b080808082b2b,
    0x082b080808190819,
    0x082b080808191908,
    0x082b0808082b0808,
    0x082b080819080819,
    0x082b080819081908,
    0x082b080819190808,
    0x082b08082b080808,
    0x082b08082b2b0808,
    0x082b081908080819,
    0x082b081908081908,
    0x082b081908190808,
    0x082b081919080808,
    0x082b081919082b08,
    0x082b0819192b1919,
    0x082b082b08080808,
    0x082b082b082b082b,
    0x082b082b2b080808,
    0x082b082b2b2b2b08,
    0x082b190808080819,
    0x082b190808081908,
    0x082b190808190808,
    0x082b1908082b2b19,
    0x082b190819080808,
    0x082b191908080808,
    0x082b191919080819,
    0x082b19191919082b,
    0x082b19192b192b19,
    0x082b192b08080819,
    0x082b192b08192b2b,
    0x082b192b2b2b192b,
    0x082b2b0808080808,
    0x082b2b0808082b08,
    0x082b2b0808082b2b,
    0x082b2b08082b0808,
    0x082b2b0819191919,
    0x082b2b082b082b08,
    0x082b2b082b2b082b,
    0x082b2b19192b2b08,
    0x082b2b192b190808,
    0x082b2b2b08082b08,
    0x082b2b2b082b0808,
    0x082b2b2b2b08082b,
    0x082b2b2b2b082b08,
    0x082b2b2b2b082b2b,
    0x1908080808080819,
    0x1908080808081908,
    0x190808080808192b,
    0x1908080808082b19,
    0x1908080808190808,
    0x190808080819082b,
    0x1908080808191919,
    0x1908080808192b08,
    0x19080808082b0819,
    0x19080808082b1908,
    0x1908080819080808,
    0x190808081908082b,
    0x1908080819081919,
    0x1908080819082b08,
    0x1908080819082b2b,
    0x1908080819190819,
    0x1908080819191908,
    0x19080808192b0808,
    0x19080808192b1919,
    0x190808082b080819,
    0x190808082b081908,
    0x190808082b190808,
    0x1908081908080808,
    0x190808190808082b,
    0x1908081908081919,
    0x1908081908082b08,
    0x1908081908190819,
    0x1908081908191908,
    0x19080819082b0808,
    0x1908081919080819,
    0x1908081919081908,
    0x1908081919190808,
    0x190808192b080808,
    0x190808192b081919,
    0x190808192b2b082b,
    0x1908082b08080819,
    0x1908082b08081908,
    0x1908082b08190808,
    0x1908082b0819082b,
    0x1908082b082b2b19,
    0x1908082b19080808,
    0x1908190808080808,
    0x190819080808082b,
    0x1908190808081919,
    0x1908190808082b08,
    0x1908190808190819,
    0x1908190808191908,
    0x1908190808192b19,
    0x19081908082b0808,
    0x1908190819080819,
    0x1908190819081908,
    0x1908190819190808,
    0x190819082b080808,
    0x190819082b191908,
    0x1908191908080819,
    0x1908191908081908,
    0x1908191908190808,
    0x19081919082b1908,
    0x1908191919080808,
    0x190819192b192b2b,
    0x1908192b08080808,
    0x1908192b08082b2b,
    0x1908192b19081908,
    0x1908192b19190808,
    0x19082b0808080819,
    0x19082b0808081908,
    0x19082b0808190808,
    0x19082b0819080808,
    0x19082b0819081919,
    0x19082b0819191908,
    0x19082b08192b082b,
    0x19082b1908080808,
    0x19082b1908190819,
    0x19082b1919081908,
    0x19082b1919190808,
    0x19082b19192b2b19,
    0x19082b2b08081908,
    0x1919080808080808,
    0x191908080808082b,
    0x1919080808081919,
    0x1919080808082b08,
    0x1919080808190819,
    0x1919080808191908,
    0x19190808082b0808,
    0x19190808082b2b08,
    0x1919080819080819,
    0x1919080819081908,
    0x1919080819190808,
    0x191908082b080808,
    0x1919081908080819,
    0x1919081908081908,
    0x1919081908190808,
    0x1919081908191919,
    0x1919081919080808,
    0x191908191908082b,
    0x1919082b08080808,
    0x1919082b19081908,
    0x1919082b2b2b2b2b,
    0x1919190808080819,
    0x1919190808081908,
    0x1919190808190808,
    0x19191908082b0819,
    0x1919190819080808,
    0x19191908192b0808,
    0x191919082b080819,
    0x191919082b2b0819,
    0x1919191908080808,
    0x1919191908082b08,
    0x191919192b080808,
    0x191919192b082b08,
    0x1919192b082b0819,
    0x1919192b192b2b08,
    0x1919192b2b2b0819,
    0x19192b0808080808,
    0x19192b0808191908,
    0x19192b0819080819,
    0x19192b0819190808,
    0x19192b082b192b19,
    0x19192b1908192b2b,
    0x19192b1919080808,
    0x19192b191908082b,
    0x19192b2b2b081919,
    0x192b080808080819,
    0x192b080808081908,
    0x192b080808190808,
    0x192b080819080808,
    0x192b080819191908,
    0x192b0808192b082b,
    0x192b08082b08192b,
    0x192b08082b2b2b19,
    0x192b081908080808,
    0x192b082b082b1908,
    0x192b082b19082b2b,
    0x192b082b2b19082b,
    0x192b190808080808,
    0x192b19080819192b,
    0x192b191908190808,
    0x192b191919080808,
    0x192b191919081919,
    0x192b19192b2b1908,
    0x192b2b0808080819,
    0x192b2b08192b2b2b,
    0x192b2b19082b1919,
    0x192b2b2b0808192b,
    0x192b2b2b19191908,
    0x192b2b2b192b082b,
    0x2b08080808080808,
    0x2b0808080808082b,
    0x2b08080808081919,
    0x2b08080808082b08,
    0x2b08080808190819,
    0x2b08080808191908,
    0x2b080808082b0808,
    0x2b080808082b2b2b,
    0x2b08080819080819,
    0x2b08080819081908,
    0x2b08080819190808,
    0x2b0808082b080808,
    0x2b0808082b08082b,
    0x2b0808082b2b2b08,
    0x2b0808082b2b2b2b,
    0x2b08081908080819,
    0x2b08081908081908,
    0x2b0808190808192b,
    0x2b08081908190808,
    0x2b08081919080808,
    0x2b08081919190819,
    0x2b08081919192b19,
    0x2b08082b08080808,
    0x2b08082b082b0808,
    0x2b08082b2b080808,
    0x2b08082b2b08082b,
    0x2b08082b2b2b0808,
    0x2b08082b2b2b2b08,
    0x2b08190808080819,
    0x2b08190808081908,
    0x2b08190808190808,
    0x2b0819080819082b,
    0x2b08190808191919,
    0x2b08190819080808,
    0x2b081908192b0808,
    0x2b0819082b082b19,
    0x2b08191908080808,
    0x2b08191919081908,
    0x2b0819192b2b1919,
    0x2b08192b08192b08,
    0x2b08192b192b2b2b,
    0x2b082b0808080808,
    0x2b082b0808082b08,
    0x2b082b08082b1919,
    0x2b082b0819192b2b,
    0x2b082b082b080808,
    0x2b082b082b08082b,
    0x2b082b082b2b2b08,
    0x2b082b190808192b,
    0x2b082b2b082b082b,
    0x2b082b2b2b080808,
    0x2b082b2b2b082b08,
    0x2b082b2b2b19192b,
    0x2b082b2b2b2b2b08,
    0x2b19080808080819,
    0x2b19080808081908,
    0x2b19080808190808,
    0x2b19080819080808,
    0x2b1908081919192b,
    0x2b1908082b081908,
    0x2b19081908080808,
    0x2b190819082b082b,
    0x2b190819192b1908,
    0x2b19082b1919192b,
    0x2b19082b2b082b19,
    0x2b19190808080808,
    0x2b19190808081919,
    0x2b19190819081908,
    0x2b19190819190808,
    0x2b19190819192b08,
    0x2b191919082b2b19,
    0x2b1919192b190808,
    0x2b1919192b19082b,
    0x2b19192b19080819,
    0x2b192b0819190819,
    0x2b192b082b2b192b,
    0x2b192b1919082b19,
    0x2b192b2b08191919,
    0x2b192b2b192b0808,
    0x2b2b080808080808,
    0x2b2b08080808082b,
    0x2b2b080808082b08,
    0x2b2b080808082b2b,
    0x2b2b0808082b0808,
    0x2b2b0808082b2b2b,
    0x2b2b08082b2b0808,
    0x2b2b081919190819,
    0x2b2b081919192b19,
    0x2b2b08192b2b192b,
    0x2b2b082b08080808,
    0x2b2b082b0808082b,
    0x2b2b082b08082b08,
    0x2b2b082b082b2b2b,
    0x2b2b082b2b080808,
    0x2b2b082b2b2b0808,
    0x2b2b190819080808,
    0x2b2b19082b191919,
    0x2b2b192b192b1919,
    0x2b2b192b2b192b08,
    0x2b2b2b0808082b2b,
    0x2b2b2b08082b0808,
    0x2b2b2b08082b082b,
    0x2b2b2b08082b2b08,
    0x2b2b2b082b2b0808,
    0x2b2b2b082b2b2b08,
    0x2b2b2b1908081908,
    0x2b2b2b192b081908,
    0x2b2b2b192b08192b,
    0x2b2b2b2b082b2b08,
    0x2b2b2b2b082b2b2b,
    0x2b2b2b2b2b190819,
    0x2b2b2b2b2b2b2b2b,
];

const IQ3XXS_GRID: [u32; 256] = [
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
];

const IQ3S_GRID: [u32; 512] = [
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
];
const IQ2S_GRID: [u64; 1024] = [
    0x0808080808080808,
    0x080808080808082b,
    0x0808080808081919,
    0x0808080808082b08,
    0x0808080808082b2b,
    0x0808080808190819,
    0x0808080808191908,
    0x080808080819192b,
    0x0808080808192b19,
    0x08080808082b0808,
    0x08080808082b082b,
    0x08080808082b1919,
    0x08080808082b2b08,
    0x0808080819080819,
    0x0808080819081908,
    0x080808081908192b,
    0x0808080819082b19,
    0x0808080819190808,
    0x080808081919082b,
    0x0808080819191919,
    0x0808080819192b08,
    0x08080808192b0819,
    0x08080808192b1908,
    0x08080808192b192b,
    0x08080808192b2b19,
    0x080808082b080808,
    0x080808082b08082b,
    0x080808082b081919,
    0x080808082b082b08,
    0x080808082b190819,
    0x080808082b191908,
    0x080808082b2b0808,
    0x080808082b2b1919,
    0x080808082b2b2b2b,
    0x0808081908080819,
    0x0808081908081908,
    0x080808190808192b,
    0x0808081908082b19,
    0x0808081908190808,
    0x080808190819082b,
    0x0808081908191919,
    0x0808081908192b08,
    0x08080819082b0819,
    0x08080819082b1908,
    0x0808081919080808,
    0x080808191908082b,
    0x0808081919081919,
    0x0808081919082b08,
    0x0808081919190819,
    0x0808081919191908,
    0x080808191919192b,
    0x0808081919192b19,
    0x08080819192b0808,
    0x08080819192b1919,
    0x08080819192b2b08,
    0x080808192b080819,
    0x080808192b081908,
    0x080808192b190808,
    0x080808192b19082b,
    0x080808192b191919,
    0x080808192b2b0819,
    0x080808192b2b1908,
    0x0808082b08080808,
    0x0808082b0808082b,
    0x0808082b08081919,
    0x0808082b08082b08,
    0x0808082b08190819,
    0x0808082b08191908,
    0x0808082b082b0808,
    0x0808082b082b2b2b,
    0x0808082b19080819,
    0x0808082b19081908,
    0x0808082b1908192b,
    0x0808082b19082b19,
    0x0808082b19190808,
    0x0808082b19191919,
    0x0808082b2b080808,
    0x0808082b2b081919,
    0x0808082b2b082b2b,
    0x0808082b2b191908,
    0x0808082b2b2b082b,
    0x0808190808080819,
    0x0808190808081908,
    0x080819080808192b,
    0x0808190808082b19,
    0x0808190808190808,
    0x080819080819082b,
    0x0808190808191919,
    0x0808190808192b08,
    0x08081908082b0819,
    0x08081908082b1908,
    0x08081908082b192b,
    0x08081908082b2b19,
    0x0808190819080808,
    0x080819081908082b,
    0x0808190819081919,
    0x0808190819082b08,
    0x0808190819082b2b,
    0x0808190819190819,
    0x0808190819191908,
    0x080819081919192b,
    0x0808190819192b19,
    0x08081908192b0808,
    0x08081908192b082b,
    0x08081908192b1919,
    0x080819082b080819,
    0x080819082b081908,
    0x080819082b08192b,
    0x080819082b082b19,
    0x080819082b190808,
    0x080819082b191919,
    0x080819082b192b08,
    0x080819082b2b0819,
    0x080819082b2b1908,
    0x0808191908080808,
    0x080819190808082b,
    0x0808191908081919,
    0x0808191908082b08,
    0x0808191908082b2b,
    0x0808191908190819,
    0x0808191908191908,
    0x080819190819192b,
    0x0808191908192b19,
    0x08081919082b0808,
    0x08081919082b1919,
    0x08081919082b2b08,
    0x0808191919080819,
    0x0808191919081908,
    0x080819191908192b,
    0x0808191919082b19,
    0x0808191919190808,
    0x080819191919082b,
    0x0808191919191919,
    0x0808191919192b08,
    0x08081919192b0819,
    0x08081919192b1908,
    0x080819192b080808,
    0x080819192b08082b,
    0x080819192b081919,
    0x080819192b082b08,
    0x080819192b190819,
    0x080819192b191908,
    0x080819192b2b0808,
    0x0808192b08080819,
    0x0808192b08081908,
    0x0808192b0808192b,
    0x0808192b08082b19,
    0x0808192b08190808,
    0x0808192b08191919,
    0x0808192b19080808,
    0x0808192b19081919,
    0x0808192b19082b08,
    0x0808192b19190819,
    0x0808192b19191908,
    0x0808192b192b0808,
    0x0808192b2b080819,
    0x0808192b2b081908,
    0x0808192b2b190808,
    0x08082b0808080808,
    0x08082b080808082b,
    0x08082b0808081919,
    0x08082b0808082b08,
    0x08082b0808190819,
    0x08082b0808191908,
    0x08082b080819192b,
    0x08082b0808192b19,
    0x08082b08082b0808,
    0x08082b08082b1919,
    0x08082b08082b2b2b,
    0x08082b0819080819,
    0x08082b0819081908,
    0x08082b081908192b,
    0x08082b0819082b19,
    0x08082b0819190808,
    0x08082b081919082b,
    0x08082b0819191919,
    0x08082b0819192b08,
    0x08082b08192b0819,
    0x08082b08192b1908,
    0x08082b082b080808,
    0x08082b082b081919,
    0x08082b082b191908,
    0x08082b082b2b2b2b,
    0x08082b1908080819,
    0x08082b1908081908,
    0x08082b1908190808,
    0x08082b190819082b,
    0x08082b1908191919,
    0x08082b1908192b08,
    0x08082b19082b0819,
    0x08082b1919080808,
    0x08082b1919081919,
    0x08082b1919082b08,
    0x08082b1919190819,
    0x08082b1919191908,
    0x08082b19192b0808,
    0x08082b192b080819,
    0x08082b192b190808,
    0x08082b2b08080808,
    0x08082b2b08190819,
    0x08082b2b08191908,
    0x08082b2b082b082b,
    0x08082b2b082b2b08,
    0x08082b2b082b2b2b,
    0x08082b2b19190808,
    0x08082b2b2b192b19,
    0x0819080808080819,
    0x0819080808081908,
    0x081908080808192b,
    0x0819080808082b19,
    0x0819080808190808,
    0x081908080819082b,
    0x0819080808191919,
    0x0819080808192b08,
    0x08190808082b0819,
    0x08190808082b1908,
    0x08190808082b192b,
    0x0819080819080808,
    0x081908081908082b,
    0x0819080819081919,
    0x0819080819082b08,
    0x0819080819190819,
    0x0819080819191908,
    0x081908081919192b,
    0x0819080819192b19,
    0x08190808192b0808,
    0x08190808192b082b,
    0x08190808192b1919,
    0x08190808192b2b08,
    0x081908082b080819,
    0x081908082b081908,
    0x081908082b08192b,
    0x081908082b190808,
    0x081908082b191919,
    0x081908082b192b08,
    0x081908082b2b0819,
    0x081908082b2b1908,
    0x0819081908080808,
    0x081908190808082b,
    0x0819081908081919,
    0x0819081908082b08,
    0x0819081908082b2b,
    0x0819081908190819,
    0x0819081908191908,
    0x081908190819192b,
    0x0819081908192b19,
    0x08190819082b0808,
    0x08190819082b082b,
    0x08190819082b1919,
    0x08190819082b2b08,
    0x0819081919080819,
    0x0819081919081908,
    0x081908191908192b,
    0x0819081919082b19,
    0x0819081919190808,
    0x081908191919082b,
    0x0819081919191919,
    0x0819081919192b08,
    0x08190819192b0819,
    0x08190819192b1908,
    0x081908192b080808,
    0x081908192b08082b,
    0x081908192b081919,
    0x081908192b082b08,
    0x081908192b190819,
    0x081908192b191908,
    0x0819082b08080819,
    0x0819082b08081908,
    0x0819082b08082b19,
    0x0819082b08190808,
    0x0819082b08191919,
    0x0819082b082b0819,
    0x0819082b082b1908,
    0x0819082b19080808,
    0x0819082b19081919,
    0x0819082b19190819,
    0x0819082b19191908,
    0x0819082b2b080819,
    0x0819082b2b081908,
    0x0819082b2b190808,
    0x0819190808080808,
    0x081919080808082b,
    0x0819190808081919,
    0x0819190808082b08,
    0x0819190808190819,
    0x0819190808191908,
    0x081919080819192b,
    0x0819190808192b19,
    0x08191908082b0808,
    0x08191908082b1919,
    0x08191908082b2b08,
    0x0819190819080819,
    0x0819190819081908,
    0x081919081908192b,
    0x0819190819082b19,
    0x0819190819190808,
    0x081919081919082b,
    0x0819190819191919,
    0x0819190819192b08,
    0x08191908192b0819,
    0x08191908192b1908,
    0x081919082b080808,
    0x081919082b08082b,
    0x081919082b081919,
    0x081919082b082b08,
    0x081919082b190819,
    0x081919082b191908,
    0x081919082b2b0808,
    0x0819191908080819,
    0x0819191908081908,
    0x081919190808192b,
    0x0819191908082b19,
    0x0819191908190808,
    0x081919190819082b,
    0x0819191908191919,
    0x0819191908192b08,
    0x08191919082b0819,
    0x08191919082b1908,
    0x0819191919080808,
    0x081919191908082b,
    0x0819191919081919,
    0x0819191919082b08,
    0x0819191919190819,
    0x0819191919191908,
    0x08191919192b0808,
    0x081919192b080819,
    0x081919192b081908,
    0x081919192b190808,
    0x0819192b08080808,
    0x0819192b08081919,
    0x0819192b08082b08,
    0x0819192b08190819,
    0x0819192b08191908,
    0x0819192b082b0808,
    0x0819192b19080819,
    0x0819192b19081908,
    0x0819192b19190808,
    0x0819192b2b080808,
    0x0819192b2b2b2b2b,
    0x08192b0808080819,
    0x08192b0808081908,
    0x08192b080808192b,
    0x08192b0808082b19,
    0x08192b0808190808,
    0x08192b0808191919,
    0x08192b0808192b08,
    0x08192b08082b0819,
    0x08192b0819080808,
    0x08192b081908082b,
    0x08192b0819081919,
    0x08192b0819082b08,
    0x08192b0819190819,
    0x08192b0819191908,
    0x08192b08192b0808,
    0x08192b082b080819,
    0x08192b082b081908,
    0x08192b1908080808,
    0x08192b190808082b,
    0x08192b1908081919,
    0x08192b1908082b08,
    0x08192b1908190819,
    0x08192b1908191908,
    0x08192b19082b0808,
    0x08192b1919080819,
    0x08192b1919081908,
    0x08192b1919190808,
    0x08192b19192b2b19,
    0x08192b192b2b082b,
    0x08192b2b08081908,
    0x08192b2b08190808,
    0x08192b2b19080808,
    0x08192b2b1919192b,
    0x082b080808080808,
    0x082b08080808082b,
    0x082b080808081919,
    0x082b080808082b08,
    0x082b080808190819,
    0x082b080808191908,
    0x082b08080819192b,
    0x082b080808192b19,
    0x082b0808082b0808,
    0x082b0808082b1919,
    0x082b0808082b2b2b,
    0x082b080819080819,
    0x082b080819081908,
    0x082b080819190808,
    0x082b08081919082b,
    0x082b080819191919,
    0x082b0808192b1908,
    0x082b08082b080808,
    0x082b08082b082b2b,
    0x082b08082b191908,
    0x082b08082b2b2b2b,
    0x082b081908080819,
    0x082b081908081908,
    0x082b081908190808,
    0x082b08190819082b,
    0x082b081908191919,
    0x082b0819082b0819,
    0x082b081919080808,
    0x082b08191908082b,
    0x082b081919081919,
    0x082b081919190819,
    0x082b081919191908,
    0x082b0819192b0808,
    0x082b08192b080819,
    0x082b08192b081908,
    0x082b08192b190808,
    0x082b082b08080808,
    0x082b082b08082b2b,
    0x082b082b082b082b,
    0x082b082b082b2b08,
    0x082b082b082b2b2b,
    0x082b082b19081908,
    0x082b082b19190808,
    0x082b082b2b082b08,
    0x082b082b2b082b2b,
    0x082b082b2b2b2b08,
    0x082b190808080819,
    0x082b190808081908,
    0x082b19080808192b,
    0x082b190808082b19,
    0x082b190808190808,
    0x082b190808191919,
    0x082b190808192b08,
    0x082b1908082b0819,
    0x082b1908082b1908,
    0x082b190819080808,
    0x082b19081908082b,
    0x082b190819081919,
    0x082b190819082b08,
    0x082b190819190819,
    0x082b190819191908,
    0x082b1908192b0808,
    0x082b19082b080819,
    0x082b19082b081908,
    0x082b19082b190808,
    0x082b191908080808,
    0x082b191908081919,
    0x082b191908082b08,
    0x082b191908190819,
    0x082b191908191908,
    0x082b1919082b0808,
    0x082b191919080819,
    0x082b191919081908,
    0x082b191919190808,
    0x082b1919192b192b,
    0x082b19192b080808,
    0x082b192b08080819,
    0x082b192b08081908,
    0x082b192b08190808,
    0x082b192b19080808,
    0x082b192b19192b19,
    0x082b2b0808080808,
    0x082b2b0808081919,
    0x082b2b0808190819,
    0x082b2b0808191908,
    0x082b2b0819080819,
    0x082b2b0819081908,
    0x082b2b0819190808,
    0x082b2b082b082b2b,
    0x082b2b082b2b2b2b,
    0x082b2b1908080819,
    0x082b2b1908081908,
    0x082b2b1908190808,
    0x082b2b192b191919,
    0x082b2b2b08082b2b,
    0x082b2b2b082b082b,
    0x082b2b2b192b1908,
    0x082b2b2b2b082b08,
    0x082b2b2b2b082b2b,
    0x1908080808080819,
    0x1908080808081908,
    0x190808080808192b,
    0x1908080808082b19,
    0x1908080808190808,
    0x190808080819082b,
    0x1908080808191919,
    0x1908080808192b08,
    0x1908080808192b2b,
    0x19080808082b0819,
    0x19080808082b1908,
    0x19080808082b192b,
    0x1908080819080808,
    0x190808081908082b,
    0x1908080819081919,
    0x1908080819082b08,
    0x1908080819082b2b,
    0x1908080819190819,
    0x1908080819191908,
    0x190808081919192b,
    0x1908080819192b19,
    0x19080808192b0808,
    0x19080808192b082b,
    0x19080808192b1919,
    0x190808082b080819,
    0x190808082b081908,
    0x190808082b190808,
    0x190808082b191919,
    0x190808082b192b08,
    0x190808082b2b0819,
    0x190808082b2b1908,
    0x1908081908080808,
    0x190808190808082b,
    0x1908081908081919,
    0x1908081908082b08,
    0x1908081908190819,
    0x1908081908191908,
    0x190808190819192b,
    0x1908081908192b19,
    0x19080819082b0808,
    0x19080819082b082b,
    0x19080819082b1919,
    0x1908081919080819,
    0x1908081919081908,
    0x190808191908192b,
    0x1908081919082b19,
    0x1908081919190808,
    0x190808191919082b,
    0x1908081919191919,
    0x1908081919192b08,
    0x19080819192b0819,
    0x19080819192b1908,
    0x190808192b080808,
    0x190808192b08082b,
    0x190808192b081919,
    0x190808192b082b08,
    0x190808192b190819,
    0x190808192b191908,
    0x190808192b2b0808,
    0x1908082b08080819,
    0x1908082b08081908,
    0x1908082b08190808,
    0x1908082b0819082b,
    0x1908082b08191919,
    0x1908082b08192b08,
    0x1908082b082b1908,
    0x1908082b19080808,
    0x1908082b19081919,
    0x1908082b19082b08,
    0x1908082b19190819,
    0x1908082b19191908,
    0x1908082b192b0808,
    0x1908082b2b080819,
    0x1908082b2b081908,
    0x1908190808080808,
    0x190819080808082b,
    0x1908190808081919,
    0x1908190808082b08,
    0x1908190808082b2b,
    0x1908190808190819,
    0x1908190808191908,
    0x190819080819192b,
    0x1908190808192b19,
    0x19081908082b0808,
    0x19081908082b082b,
    0x19081908082b1919,
    0x19081908082b2b08,
    0x1908190819080819,
    0x1908190819081908,
    0x190819081908192b,
    0x1908190819082b19,
    0x1908190819190808,
    0x190819081919082b,
    0x1908190819191919,
    0x1908190819192b08,
    0x19081908192b0819,
    0x19081908192b1908,
    0x190819082b080808,
    0x190819082b08082b,
    0x190819082b081919,
    0x190819082b082b08,
    0x190819082b190819,
    0x190819082b191908,
    0x190819082b2b0808,
    0x1908191908080819,
    0x1908191908081908,
    0x190819190808192b,
    0x1908191908082b19,
    0x1908191908190808,
    0x190819190819082b,
    0x1908191908191919,
    0x1908191908192b08,
    0x19081919082b0819,
    0x19081919082b1908,
    0x1908191919080808,
    0x190819191908082b,
    0x1908191919081919,
    0x1908191919082b08,
    0x1908191919190819,
    0x1908191919191908,
    0x19081919192b0808,
    0x19081919192b2b2b,
    0x190819192b080819,
    0x190819192b081908,
    0x190819192b190808,
    0x1908192b08080808,
    0x1908192b0808082b,
    0x1908192b08081919,
    0x1908192b08082b08,
    0x1908192b08190819,
    0x1908192b08191908,
    0x1908192b082b0808,
    0x1908192b19080819,
    0x1908192b19081908,
    0x1908192b19190808,
    0x1908192b2b080808,
    0x1908192b2b2b1919,
    0x19082b0808080819,
    0x19082b0808081908,
    0x19082b0808082b19,
    0x19082b0808190808,
    0x19082b080819082b,
    0x19082b0808191919,
    0x19082b0808192b08,
    0x19082b08082b0819,
    0x19082b08082b1908,
    0x19082b0819080808,
    0x19082b081908082b,
    0x19082b0819081919,
    0x19082b0819082b08,
    0x19082b0819190819,
    0x19082b0819191908,
    0x19082b08192b0808,
    0x19082b082b081908,
    0x19082b082b190808,
    0x19082b1908080808,
    0x19082b190808082b,
    0x19082b1908081919,
    0x19082b1908082b08,
    0x19082b1908190819,
    0x19082b1908191908,
    0x19082b19082b0808,
    0x19082b1919080819,
    0x19082b1919081908,
    0x19082b1919190808,
    0x19082b192b080808,
    0x19082b192b19192b,
    0x19082b2b08080819,
    0x19082b2b08081908,
    0x19082b2b08190808,
    0x19082b2b19080808,
    0x1919080808080808,
    0x191908080808082b,
    0x1919080808081919,
    0x1919080808082b08,
    0x1919080808190819,
    0x1919080808191908,
    0x191908080819192b,
    0x1919080808192b19,
    0x19190808082b0808,
    0x19190808082b082b,
    0x19190808082b1919,
    0x19190808082b2b08,
    0x1919080819080819,
    0x1919080819081908,
    0x191908081908192b,
    0x1919080819082b19,
    0x1919080819190808,
    0x191908081919082b,
    0x1919080819191919,
    0x1919080819192b08,
    0x19190808192b0819,
    0x19190808192b1908,
    0x191908082b080808,
    0x191908082b08082b,
    0x191908082b081919,
    0x191908082b082b08,
    0x191908082b190819,
    0x191908082b191908,
    0x1919081908080819,
    0x1919081908081908,
    0x191908190808192b,
    0x1919081908082b19,
    0x1919081908190808,
    0x191908190819082b,
    0x1919081908191919,
    0x1919081908192b08,
    0x19190819082b0819,
    0x19190819082b1908,
    0x1919081919080808,
    0x191908191908082b,
    0x1919081919081919,
    0x1919081919082b08,
    0x1919081919190819,
    0x1919081919191908,
    0x19190819192b0808,
    0x191908192b080819,
    0x191908192b081908,
    0x191908192b190808,
    0x1919082b08080808,
    0x1919082b08081919,
    0x1919082b08082b08,
    0x1919082b08190819,
    0x1919082b08191908,
    0x1919082b082b0808,
    0x1919082b19080819,
    0x1919082b19081908,
    0x1919082b19190808,
    0x1919082b192b2b19,
    0x1919082b2b080808,
    0x1919190808080819,
    0x1919190808081908,
    0x191919080808192b,
    0x1919190808082b19,
    0x1919190808190808,
    0x191919080819082b,
    0x1919190808191919,
    0x1919190808192b08,
    0x19191908082b0819,
    0x19191908082b1908,
    0x1919190819080808,
    0x191919081908082b,
    0x1919190819081919,
    0x1919190819082b08,
    0x1919190819190819,
    0x1919190819191908,
    0x19191908192b0808,
    0x191919082b080819,
    0x191919082b081908,
    0x191919082b190808,
    0x1919191908080808,
    0x191919190808082b,
    0x1919191908081919,
    0x1919191908082b08,
    0x1919191908190819,
    0x1919191908191908,
    0x19191919082b0808,
    0x1919191919080819,
    0x1919191919081908,
    0x1919191919190808,
    0x191919192b080808,
    0x1919192b08080819,
    0x1919192b08081908,
    0x1919192b08190808,
    0x1919192b082b192b,
    0x1919192b19080808,
    0x19192b0808080808,
    0x19192b080808082b,
    0x19192b0808081919,
    0x19192b0808082b08,
    0x19192b0808190819,
    0x19192b0808191908,
    0x19192b08082b0808,
    0x19192b0819080819,
    0x19192b0819081908,
    0x19192b0819190808,
    0x19192b0819192b2b,
    0x19192b082b080808,
    0x19192b1908080819,
    0x19192b1908081908,
    0x19192b1908190808,
    0x19192b1919080808,
    0x19192b2b08080808,
    0x19192b2b08192b19,
    0x19192b2b2b081919,
    0x19192b2b2b2b2b08,
    0x192b080808080819,
    0x192b080808081908,
    0x192b08080808192b,
    0x192b080808190808,
    0x192b08080819082b,
    0x192b080808191919,
    0x192b080808192b08,
    0x192b0808082b0819,
    0x192b0808082b1908,
    0x192b080819080808,
    0x192b080819081919,
    0x192b080819082b08,
    0x192b080819190819,
    0x192b080819191908,
    0x192b0808192b0808,
    0x192b08082b081908,
    0x192b08082b190808,
    0x192b081908080808,
    0x192b08190808082b,
    0x192b081908081919,
    0x192b081908082b08,
    0x192b081908190819,
    0x192b081908191908,
    0x192b0819082b0808,
    0x192b081919080819,
    0x192b081919081908,
    0x192b081919190808,
    0x192b08192b080808,
    0x192b08192b192b19,
    0x192b082b08081908,
    0x192b082b08190808,
    0x192b082b19080808,
    0x192b082b1919192b,
    0x192b082b2b2b0819,
    0x192b190808080808,
    0x192b190808081919,
    0x192b190808082b08,
    0x192b190808190819,
    0x192b190808191908,
    0x192b1908082b0808,
    0x192b190819080819,
    0x192b190819081908,
    0x192b190819190808,
    0x192b19082b080808,
    0x192b191908080819,
    0x192b191908081908,
    0x192b191908190808,
    0x192b191919080808,
    0x192b191919082b2b,
    0x192b1919192b2b08,
    0x192b19192b19082b,
    0x192b192b08080808,
    0x192b192b2b191908,
    0x192b2b0808080819,
    0x192b2b0808081908,
    0x192b2b0808190808,
    0x192b2b08192b1919,
    0x192b2b082b192b08,
    0x192b2b1908080808,
    0x192b2b19082b2b2b,
    0x192b2b2b1908082b,
    0x192b2b2b2b2b0819,
    0x2b08080808080808,
    0x2b0808080808082b,
    0x2b08080808081919,
    0x2b08080808082b08,
    0x2b08080808190819,
    0x2b08080808191908,
    0x2b08080808192b19,
    0x2b080808082b0808,
    0x2b080808082b1919,
    0x2b08080819080819,
    0x2b08080819081908,
    0x2b08080819190808,
    0x2b0808081919082b,
    0x2b08080819191919,
    0x2b08080819192b08,
    0x2b080808192b0819,
    0x2b0808082b080808,
    0x2b0808082b081919,
    0x2b0808082b190819,
    0x2b0808082b191908,
    0x2b08081908080819,
    0x2b08081908081908,
    0x2b08081908082b19,
    0x2b08081908190808,
    0x2b0808190819082b,
    0x2b08081908191919,
    0x2b08081908192b08,
    0x2b080819082b0819,
    0x2b080819082b1908,
    0x2b08081919080808,
    0x2b0808191908082b,
    0x2b08081919081919,
    0x2b08081919082b08,
    0x2b08081919190819,
    0x2b08081919191908,
    0x2b0808192b080819,
    0x2b0808192b081908,
    0x2b0808192b190808,
    0x2b0808192b2b2b19,
    0x2b08082b08080808,
    0x2b08082b08081919,
    0x2b08082b08082b2b,
    0x2b08082b08190819,
    0x2b08082b08191908,
    0x2b08082b19080819,
    0x2b08082b19081908,
    0x2b08082b19190808,
    0x2b08190808080819,
    0x2b08190808081908,
    0x2b0819080808192b,
    0x2b08190808082b19,
    0x2b08190808190808,
    0x2b0819080819082b,
    0x2b08190808191919,
    0x2b08190808192b08,
    0x2b081908082b0819,
    0x2b08190819080808,
    0x2b0819081908082b,
    0x2b08190819081919,
    0x2b08190819082b08,
    0x2b08190819190819,
    0x2b08190819191908,
    0x2b081908192b0808,
    0x2b0819082b080819,
    0x2b0819082b081908,
    0x2b0819082b190808,
    0x2b08191908080808,
    0x2b0819190808082b,
    0x2b08191908081919,
    0x2b08191908082b08,
    0x2b08191908190819,
    0x2b08191908191908,
    0x2b081919082b0808,
    0x2b08191919080819,
    0x2b08191919081908,
    0x2b08191919190808,
    0x2b0819192b080808,
    0x2b0819192b082b2b,
    0x2b08192b08080819,
    0x2b08192b08081908,
    0x2b08192b08190808,
    0x2b08192b082b2b19,
    0x2b08192b19080808,
    0x2b082b0808080808,
    0x2b082b0808081919,
    0x2b082b0808190819,
    0x2b082b0808191908,
    0x2b082b0819080819,
    0x2b082b0819081908,
    0x2b082b0819190808,
    0x2b082b082b2b082b,
    0x2b082b1908080819,
    0x2b082b1908081908,
    0x2b082b1919080808,
    0x2b082b19192b1919,
    0x2b082b2b082b082b,
    0x2b082b2b19192b08,
    0x2b082b2b19192b2b,
    0x2b082b2b2b08082b,
    0x2b082b2b2b2b082b,
    0x2b19080808080819,
    0x2b19080808081908,
    0x2b19080808082b19,
    0x2b19080808190808,
    0x2b1908080819082b,
    0x2b19080808191919,
    0x2b19080808192b08,
    0x2b190808082b1908,
    0x2b19080819080808,
    0x2b1908081908082b,
    0x2b19080819081919,
    0x2b19080819082b08,
    0x2b19080819190819,
    0x2b19080819191908,
    0x2b190808192b0808,
    0x2b1908082b080819,
    0x2b1908082b081908,
    0x2b1908082b190808,
    0x2b19081908080808,
    0x2b19081908081919,
    0x2b19081908190819,
    0x2b19081908191908,
    0x2b19081919080819,
    0x2b19081919081908,
    0x2b19081919190808,
    0x2b19081919192b2b,
    0x2b19082b08080819,
    0x2b19082b08081908,
    0x2b19082b08190808,
    0x2b19082b19080808,
    0x2b19082b2b2b192b,
    0x2b19190808080808,
    0x2b1919080808082b,
    0x2b19190808081919,
    0x2b19190808082b08,
    0x2b19190808190819,
    0x2b19190808191908,
    0x2b191908082b0808,
    0x2b19190819080819,
    0x2b19190819081908,
    0x2b19190819190808,
    0x2b1919082b080808,
    0x2b1919082b19192b,
    0x2b19191908080819,
    0x2b19191908081908,
    0x2b19191908190808,
    0x2b19191919080808,
    0x2b1919192b192b08,
    0x2b1919192b2b0819,
    0x2b19192b08080808,
    0x2b19192b1908192b,
    0x2b19192b192b1908,
    0x2b192b0808080819,
    0x2b192b0808081908,
    0x2b192b0808190808,
    0x2b192b08082b192b,
    0x2b192b0819080808,
    0x2b192b082b2b2b19,
    0x2b192b1908080808,
    0x2b192b1919082b19,
    0x2b192b191919082b,
    0x2b192b2b2b190808,
    0x2b2b080808080808,
    0x2b2b080808081919,
    0x2b2b080808082b2b,
    0x2b2b080808191908,
    0x2b2b0808082b082b,
    0x2b2b0808082b2b2b,
    0x2b2b080819080819,
    0x2b2b080819081908,
    0x2b2b080819190808,
    0x2b2b08082b2b082b,
    0x2b2b08082b2b2b2b,
    0x2b2b081919080808,
    0x2b2b0819192b1919,
    0x2b2b082b0808082b,
    0x2b2b082b08082b2b,
    0x2b2b082b082b082b,
    0x2b2b082b082b2b08,
    0x2b2b082b082b2b2b,
    0x2b2b082b2b08082b,
    0x2b2b082b2b082b08,
    0x2b2b082b2b082b2b,
    0x2b2b082b2b2b2b08,
    0x2b2b190808080819,
    0x2b2b190808081908,
    0x2b2b190808190808,
    0x2b2b190819080808,
    0x2b2b19082b082b19,
    0x2b2b19082b2b1908,
    0x2b2b191908080808,
    0x2b2b191908192b19,
    0x2b2b192b19190819,
    0x2b2b2b0808082b2b,
    0x2b2b2b08082b2b08,
    0x2b2b2b082b2b082b,
    0x2b2b2b1919191908,
    0x2b2b2b192b08192b,
    0x2b2b2b2b08082b08,
    0x2b2b2b2b08082b2b,
    0x2b2b2b2b082b0808,
    0x2b2b2b2b082b082b,
    0x2b2b2b2b082b2b08,
    0x2b2b2b2b2b082b08,
    0x2b2b2b2b2b2b2b2b,
];
const IQ1S_GRID: [u64; 2048] = [
    0xffffffffffffffff,
    0xffffffffffffff01,
    0xffffffffffff0000,
    0xffffffffffff01ff,
    0xffffffffffff0101,
    0xffffffffff00ff00,
    0xffffffffff000000,
    0xffffffffff01ffff,
    0xffffffffff01ff01,
    0xffffffffff0101ff,
    0xffffffffff010101,
    0xffffffff00ff0000,
    0xffffffff0000ff00,
    0xffffffff000000ff,
    0xffffffff00000001,
    0xffffffff00010000,
    0xffffffff01ffffff,
    0xffffffff01ffff01,
    0xffffffff01ff01ff,
    0xffffffff01ff0101,
    0xffffffff01000000,
    0xffffffff0101ffff,
    0xffffffff0101ff01,
    0xffffffff010101ff,
    0xffffffff01010101,
    0xffffff00ffff00ff,
    0xffffff00ffff0000,
    0xffffff00ff00ff00,
    0xffffff00ff0000ff,
    0xffffff00ff000001,
    0xffffff00ff000100,
    0xffffff00ff000101,
    0xffffff00ff010000,
    0xffffff0000ffff00,
    0xffffff0000ff0001,
    0xffffff0000ff0100,
    0xffffff000000ff01,
    0xffffff0000000000,
    0xffffff0000000101,
    0xffffff000001ff00,
    0xffffff00000100ff,
    0xffffff0000010001,
    0xffffff00000101ff,
    0xffffff0001ff0000,
    0xffffff000100ff00,
    0xffffff00010000ff,
    0xffffff0001000001,
    0xffffff0001010000,
    0xffffff01ffffffff,
    0xffffff01ffffff01,
    0xffffff01ffff01ff,
    0xffffff01ffff0101,
    0xffffff01ff000000,
    0xffffff01ff01ffff,
    0xffffff01ff01ff01,
    0xffffff01ff0101ff,
    0xffffff01ff010101,
    0xffffff0100ff0000,
    0xffffff010000ff00,
    0xffffff0100000100,
    0xffffff01000100ff,
    0xffffff0100010100,
    0xffffff0101ffffff,
    0xffffff0101ffff01,
    0xffffff0101ff01ff,
    0xffffff0101ff0101,
    0xffffff010100ff00,
    0xffffff0101000000,
    0xffffff0101000100,
    0xffffff010101ffff,
    0xffffff010101ff01,
    0xffffff01010101ff,
    0xffffff0101010101,
    0xffff00ffff00ff00,
    0xffff00ffff0000ff,
    0xffff00ffff000001,
    0xffff00ffff010000,
    0xffff00ff00ffff00,
    0xffff00ff00ff0100,
    0xffff00ff00000000,
    0xffff00ff00000101,
    0xffff00ff000100ff,
    0xffff00ff00010000,
    0xffff00ff0100ff00,
    0xffff00ff01000100,
    0xffff00ff01010000,
    0xffff0000ffffff00,
    0xffff0000ffff00ff,
    0xffff0000ffff0000,
    0xffff0000ffff0001,
    0xffff0000ff000000,
    0xffff0000ff0001ff,
    0xffff0000ff000101,
    0xffff0000ff010100,
    0xffff000000ffffff,
    0xffff000000ff0000,
    0xffff000000ff0101,
    0xffff00000000ffff,
    0xffff00000000ff00,
    0xffff0000000000ff,
    0xffff000000000000,
    0xffff000000000001,
    0xffff000000000100,
    0xffff00000001ffff,
    0xffff00000001ff01,
    0xffff000000010000,
    0xffff0000000101ff,
    0xffff000000010101,
    0xffff000001ffff00,
    0xffff00000100ff00,
    0xffff000001000000,
    0xffff0000010001ff,
    0xffff000001000101,
    0xffff00000101ff00,
    0xffff0000010100ff,
    0xffff000001010000,
    0xffff000001010001,
    0xffff000001010100,
    0xffff0001ff0000ff,
    0xffff0001ff000100,
    0xffff000100ffff00,
    0xffff000100ff00ff,
    0xffff00010000ffff,
    0xffff00010000ff01,
    0xffff000100000000,
    0xffff0001000001ff,
    0xffff00010001ffff,
    0xffff00010001ff00,
    0xffff000100010001,
    0xffff000100010100,
    0xffff000101ff0000,
    0xffff00010100ff00,
    0xffff0001010000ff,
    0xffff000101000100,
    0xffff01ffffffffff,
    0xffff01ffffffff01,
    0xffff01ffffff01ff,
    0xffff01ffffff0101,
    0xffff01ffff000000,
    0xffff01ffff01ffff,
    0xffff01ffff01ff01,
    0xffff01ffff0101ff,
    0xffff01ffff010101,
    0xffff01ff00ff0000,
    0xffff01ff0000ff00,
    0xffff01ff00000001,
    0xffff01ff00010000,
    0xffff01ff01ffffff,
    0xffff01ff01ffff01,
    0xffff01ff01ff01ff,
    0xffff01ff01ff0101,
    0xffff01ff01000000,
    0xffff01ff0101ffff,
    0xffff01ff0101ff01,
    0xffff01ff010101ff,
    0xffff01ff01010101,
    0xffff0100ffff0000,
    0xffff0100ff00ff00,
    0xffff0100ff0000ff,
    0xffff0100ff000100,
    0xffff0100ff0100ff,
    0xffff0100ff010000,
    0xffff010000ffff00,
    0xffff01000000ffff,
    0xffff01000000ff00,
    0xffff010000000000,
    0xffff01000001ff00,
    0xffff0100000100ff,
    0xffff010000010100,
    0xffff01000100ff00,
    0xffff0100010000ff,
    0xffff010001000001,
    0xffff010001000100,
    0xffff010001010000,
    0xffff0101ffffffff,
    0xffff0101ffffff01,
    0xffff0101ffff01ff,
    0xffff0101ffff0101,
    0xffff0101ff000000,
    0xffff0101ff01ffff,
    0xffff0101ff01ff01,
    0xffff0101ff0101ff,
    0xffff0101ff010101,
    0xffff010100ff0000,
    0xffff01010000ff00,
    0xffff010100000100,
    0xffff01010001ff00,
    0xffff010100010000,
    0xffff010101ffffff,
    0xffff010101ffff01,
    0xffff010101ff0000,
    0xffff010101ff01ff,
    0xffff010101ff0101,
    0xffff010101000000,
    0xffff01010101ffff,
    0xffff01010101ff01,
    0xffff0101010101ff,
    0xffff010101010101,
    0xff00ffffff00ffff,
    0xff00ffffff00ff00,
    0xff00ffffff0000ff,
    0xff00ffffff000100,
    0xff00ffffff0100ff,
    0xff00ffffff010000,
    0xff00ffff00ffff00,
    0xff00ffff00ff00ff,
    0xff00ffff0000ffff,
    0xff00ffff00000000,
    0xff00ffff000001ff,
    0xff00ffff0001ff00,
    0xff00ffff000100ff,
    0xff00ffff00010000,
    0xff00ffff00010100,
    0xff00ffff0100ff00,
    0xff00ffff010000ff,
    0xff00ffff01000001,
    0xff00ffff0101ff00,
    0xff00ffff01010000,
    0xff00ff00ffffff00,
    0xff00ff00ffff00ff,
    0xff00ff00ffff0001,
    0xff00ff00ffff0100,
    0xff00ff00ff00ffff,
    0xff00ff00ff00ff01,
    0xff00ff00ff000000,
    0xff00ff00ff0001ff,
    0xff00ff00ff01ff00,
    0xff00ff00ff0100ff,
    0xff00ff00ff010100,
    0xff00ff0000ff0000,
    0xff00ff0000ff0101,
    0xff00ff000000ffff,
    0xff00ff000000ff00,
    0xff00ff000000ff01,
    0xff00ff00000000ff,
    0xff00ff0000000000,
    0xff00ff0000000001,
    0xff00ff0000000100,
    0xff00ff000001ffff,
    0xff00ff0000010000,
    0xff00ff0001ff00ff,
    0xff00ff000100ff01,
    0xff00ff0001000000,
    0xff00ff000101ff00,
    0xff00ff00010100ff,
    0xff00ff01ff00ff00,
    0xff00ff01ff0000ff,
    0xff00ff01ff000001,
    0xff00ff01ff010000,
    0xff00ff0100ffffff,
    0xff00ff0100ff0001,
    0xff00ff0100ff0100,
    0xff00ff010000ff01,
    0xff00ff0100000000,
    0xff00ff01000001ff,
    0xff00ff0100000101,
    0xff00ff01000100ff,
    0xff00ff0100010001,
    0xff00ff0101ff0000,
    0xff00ff010100ff00,
    0xff00ff01010000ff,
    0xff00ff0101000001,
    0xff00ff0101010000,
    0xff0000ffffffff00,
    0xff0000ffffff0001,
    0xff0000ffffff0100,
    0xff0000ffff0000ff,
    0xff0000ffff000000,
    0xff0000ffff0001ff,
    0xff0000ffff000100,
    0xff0000ffff01ff00,
    0xff0000ffff010001,
    0xff0000ff00ffff00,
    0xff0000ff00ff0000,
    0xff0000ff00ff0001,
    0xff0000ff00ff01ff,
    0xff0000ff00ff0101,
    0xff0000ff0000ff00,
    0xff0000ff000000ff,
    0xff0000ff00000000,
    0xff0000ff00000001,
    0xff0000ff00000100,
    0xff0000ff0001ff01,
    0xff0000ff00010000,
    0xff0000ff000101ff,
    0xff0000ff01ff00ff,
    0xff0000ff01ff0100,
    0xff0000ff0100ffff,
    0xff0000ff010000ff,
    0xff0000ff01000000,
    0xff0000ff010001ff,
    0xff0000ff01000100,
    0xff0000ff01000101,
    0xff0000ff0101ff00,
    0xff0000ff010100ff,
    0xff0000ff01010000,
    0xff0000ff01010100,
    0xff000000ffffff01,
    0xff000000ffff0000,
    0xff000000ffff0101,
    0xff000000ff00ff00,
    0xff000000ff0000ff,
    0xff000000ff000000,
    0xff000000ff000001,
    0xff000000ff000100,
    0xff000000ff01ffff,
    0xff000000ff01ff01,
    0xff000000ff010000,
    0xff000000ff0101ff,
    0xff000000ff010101,
    0xff00000000ffff00,
    0xff00000000ff00ff,
    0xff00000000ff0000,
    0xff00000000ff0001,
    0xff0000000000ff00,
    0xff0000000000ff01,
    0xff000000000000ff,
    0xff00000000000000,
    0xff00000000000001,
    0xff00000000000100,
    0xff00000000000101,
    0xff0000000001ff00,
    0xff000000000100ff,
    0xff00000000010000,
    0xff00000000010001,
    0xff00000000010100,
    0xff00000001ffffff,
    0xff00000001ffff01,
    0xff00000001ff00ff,
    0xff00000001ff0000,
    0xff00000001ff01ff,
    0xff00000001ff0101,
    0xff0000000100ffff,
    0xff0000000100ff00,
    0xff000000010000ff,
    0xff00000001000000,
    0xff00000001000001,
    0xff00000001000100,
    0xff00000001000101,
    0xff0000000101ffff,
    0xff0000000101ff01,
    0xff00000001010000,
    0xff000001ffffff00,
    0xff000001ffff00ff,
    0xff000001ffff0000,
    0xff000001ffff0001,
    0xff000001ff000000,
    0xff000001ff000001,
    0xff000001ff0001ff,
    0xff000001ff000101,
    0xff000001ff01ff00,
    0xff000001ff010001,
    0xff00000100ffffff,
    0xff00000100ffff01,
    0xff00000100ff00ff,
    0xff00000100ff0000,
    0xff00000100ff01ff,
    0xff00000100ff0101,
    0xff0000010000ff00,
    0xff00000100000000,
    0xff00000100000001,
    0xff000001000001ff,
    0xff00000100000100,
    0xff0000010001ff00,
    0xff000001000100ff,
    0xff00000100010000,
    0xff000001000101ff,
    0xff00000100010100,
    0xff00000100010101,
    0xff00000101ff0001,
    0xff00000101ff0101,
    0xff0000010100ff01,
    0xff00000101000000,
    0xff000001010100ff,
    0xff00000101010100,
    0xff0001ffff00ff00,
    0xff0001ffff000001,
    0xff0001ffff010000,
    0xff0001ff00ffff00,
    0xff0001ff00ff00ff,
    0xff0001ff00ff0001,
    0xff0001ff00ff0100,
    0xff0001ff0000ffff,
    0xff0001ff00000000,
    0xff0001ff000001ff,
    0xff0001ff00000101,
    0xff0001ff0001ffff,
    0xff0001ff0001ff00,
    0xff0001ff000100ff,
    0xff0001ff00010001,
    0xff0001ff00010100,
    0xff0001ff01ff0000,
    0xff0001ff0100ff00,
    0xff0001ff010000ff,
    0xff0001ff01010000,
    0xff000100ff00ffff,
    0xff000100ff00ff01,
    0xff000100ff000000,
    0xff000100ff000101,
    0xff000100ff01ff00,
    0xff000100ff010000,
    0xff00010000ffff01,
    0xff00010000ff00ff,
    0xff00010000ff0000,
    0xff00010000ff01ff,
    0xff0001000000ff00,
    0xff000100000000ff,
    0xff00010000000000,
    0xff00010000000001,
    0xff00010000000100,
    0xff00010000000101,
    0xff0001000001ffff,
    0xff00010000010000,
    0xff00010000010101,
    0xff00010001ff0100,
    0xff0001000100ff00,
    0xff0001000100ff01,
    0xff00010001000000,
    0xff000100010001ff,
    0xff0001000101ff00,
    0xff00010001010001,
    0xff00010001010100,
    0xff000101ffff0100,
    0xff000101ff000001,
    0xff000101ff0100ff,
    0xff000101ff010001,
    0xff00010100ff00ff,
    0xff00010100ff0001,
    0xff00010100ff0100,
    0xff0001010000ffff,
    0xff0001010000ff01,
    0xff00010100000000,
    0xff000101000001ff,
    0xff0001010001ff00,
    0xff00010100010001,
    0xff00010100010100,
    0xff00010101ff0000,
    0xff0001010100ff00,
    0xff00010101000001,
    0xff00010101000101,
    0xff01ffffffffffff,
    0xff01ffffffffff01,
    0xff01ffffffff01ff,
    0xff01ffffffff0101,
    0xff01ffffff000000,
    0xff01ffffff01ffff,
    0xff01ffffff01ff01,
    0xff01ffffff010000,
    0xff01ffffff0101ff,
    0xff01ffffff010101,
    0xff01ffff00ff0000,
    0xff01ffff0000ff00,
    0xff01ffff00000100,
    0xff01ffff0001ff00,
    0xff01ffff00010000,
    0xff01ffff01ffffff,
    0xff01ffff01ffff01,
    0xff01ffff01ff01ff,
    0xff01ffff01ff0101,
    0xff01ffff01000000,
    0xff01ffff0101ffff,
    0xff01ffff0101ff01,
    0xff01ffff01010000,
    0xff01ffff010101ff,
    0xff01ffff01010101,
    0xff01ff00ffff0000,
    0xff01ff00ff00ff00,
    0xff01ff00ff0000ff,
    0xff01ff00ff000100,
    0xff01ff00ff010000,
    0xff01ff0000ffff01,
    0xff01ff0000ff00ff,
    0xff01ff0000ff0100,
    0xff01ff0000000000,
    0xff01ff00000001ff,
    0xff01ff0000000101,
    0xff01ff000001ff00,
    0xff01ff00000100ff,
    0xff01ff0000010000,
    0xff01ff0000010001,
    0xff01ff0001ff0000,
    0xff01ff000100ffff,
    0xff01ff0001000001,
    0xff01ff0001000100,
    0xff01ff0001010000,
    0xff01ff01ffffff00,
    0xff01ff01ffff01ff,
    0xff01ff01ffff0101,
    0xff01ff01ff00ff00,
    0xff01ff01ff000000,
    0xff01ff01ff01ffff,
    0xff01ff01ff01ff01,
    0xff01ff01ff0101ff,
    0xff01ff01ff010101,
    0xff01ff0100ff0000,
    0xff01ff010000ff00,
    0xff01ff0100000001,
    0xff01ff0100000100,
    0xff01ff0100010000,
    0xff01ff0101ffff00,
    0xff01ff0101ff01ff,
    0xff01ff0101ff0101,
    0xff01ff010100ff00,
    0xff01ff0101000000,
    0xff01ff010101ffff,
    0xff01ff010101ff01,
    0xff01ff01010101ff,
    0xff01ff0101010101,
    0xff0100ffffff0000,
    0xff0100ffff0000ff,
    0xff0100ffff000001,
    0xff0100ffff000100,
    0xff0100ffff010000,
    0xff0100ff00ff00ff,
    0xff0100ff00ff0000,
    0xff0100ff00ff0001,
    0xff0100ff00ff0100,
    0xff0100ff0000ff01,
    0xff0100ff00000000,
    0xff0100ff000001ff,
    0xff0100ff00000101,
    0xff0100ff00010001,
    0xff0100ff01ff0000,
    0xff0100ff0100ff00,
    0xff0100ff010000ff,
    0xff0100ff01000100,
    0xff0100ff0101ff00,
    0xff0100ff01010000,
    0xff010000ffff0100,
    0xff010000ff000000,
    0xff010000ff01ff00,
    0xff010000ff010100,
    0xff01000000ffffff,
    0xff01000000ff0000,
    0xff01000000ff01ff,
    0xff0100000000ff00,
    0xff010000000000ff,
    0xff01000000000000,
    0xff01000000000100,
    0xff0100000001ff01,
    0xff01000000010000,
    0xff010000000101ff,
    0xff01000001ff0100,
    0xff0100000100ffff,
    0xff010000010000ff,
    0xff01000001000000,
    0xff010000010001ff,
    0xff01000001000101,
    0xff0100000101ff00,
    0xff010000010100ff,
    0xff01000001010001,
    0xff01000001010100,
    0xff010001ffff0000,
    0xff010001ff00ffff,
    0xff010001ff00ff01,
    0xff010001ff000100,
    0xff010001ff010000,
    0xff01000100ffff00,
    0xff01000100ff0100,
    0xff01000100000000,
    0xff0100010001ffff,
    0xff0100010001ff00,
    0xff01000100010100,
    0xff01000101ff00ff,
    0xff01000101ff0001,
    0xff0100010100ffff,
    0xff01000101000101,
    0xff0101ffffffffff,
    0xff0101ffffffff01,
    0xff0101ffffff01ff,
    0xff0101ffffff0101,
    0xff0101ffff000000,
    0xff0101ffff01ffff,
    0xff0101ffff01ff01,
    0xff0101ffff0101ff,
    0xff0101ffff010101,
    0xff0101ff00ff0000,
    0xff0101ff0000ff00,
    0xff0101ff000000ff,
    0xff0101ff00010000,
    0xff0101ff01ffffff,
    0xff0101ff01ffff01,
    0xff0101ff01ff01ff,
    0xff0101ff01ff0101,
    0xff0101ff0101ffff,
    0xff0101ff0101ff01,
    0xff0101ff010101ff,
    0xff0101ff01010101,
    0xff010100ffff0100,
    0xff010100ff00ff00,
    0xff010100ff0000ff,
    0xff010100ff000100,
    0xff010100ff010000,
    0xff01010000ff0001,
    0xff01010000ff0100,
    0xff0101000000ff01,
    0xff01010000000000,
    0xff0101000001ff00,
    0xff010100000100ff,
    0xff01010000010001,
    0xff01010000010100,
    0xff01010001ff0000,
    0xff0101000100ffff,
    0xff01010001000001,
    0xff01010001000100,
    0xff010100010100ff,
    0xff01010001010000,
    0xff010101ffffffff,
    0xff010101ffffff01,
    0xff010101ffff01ff,
    0xff010101ffff0101,
    0xff010101ff01ffff,
    0xff010101ff01ff01,
    0xff010101ff0101ff,
    0xff010101ff010101,
    0xff01010100ff0000,
    0xff0101010000ff00,
    0xff01010100000001,
    0xff01010100000100,
    0xff01010100010000,
    0xff01010101ffffff,
    0xff01010101ffff01,
    0xff01010101ff01ff,
    0xff01010101ff0101,
    0xff01010101000000,
    0xff0101010101ffff,
    0xff0101010101ff01,
    0xff010101010101ff,
    0xff01010101010101,
    0x00ffffffffff0000,
    0x00ffffffff00ff00,
    0x00ffffffff000001,
    0x00ffffffff010000,
    0x00ffffff00ff0100,
    0x00ffffff0000ff01,
    0x00ffffff00000000,
    0x00ffffff000001ff,
    0x00ffffff00000101,
    0x00ffffff0001ff00,
    0x00ffffff000100ff,
    0x00ffffff00010001,
    0x00ffffff010000ff,
    0x00ffffff01000100,
    0x00ffffff0101ff00,
    0x00ffffff01010001,
    0x00ffff00ffffffff,
    0x00ffff00ffffff00,
    0x00ffff00ffff00ff,
    0x00ffff00ffff0001,
    0x00ffff00ffff0100,
    0x00ffff00ff00ff01,
    0x00ffff00ff000000,
    0x00ffff00ff000001,
    0x00ffff00ff0001ff,
    0x00ffff00ff000101,
    0x00ffff00ff01ff00,
    0x00ffff00ff010001,
    0x00ffff00ff010100,
    0x00ffff0000ff0000,
    0x00ffff0000ff01ff,
    0x00ffff0000ff0101,
    0x00ffff000000ff00,
    0x00ffff00000000ff,
    0x00ffff0000000000,
    0x00ffff0000000001,
    0x00ffff0000000100,
    0x00ffff0000000101,
    0x00ffff0000010000,
    0x00ffff00000101ff,
    0x00ffff0000010101,
    0x00ffff0001ffff00,
    0x00ffff0001ff00ff,
    0x00ffff0001ff0001,
    0x00ffff000100ffff,
    0x00ffff000100ff01,
    0x00ffff0001000000,
    0x00ffff000101ffff,
    0x00ffff000101ff00,
    0x00ffff000101ff01,
    0x00ffff01ffff0000,
    0x00ffff01ff00ff00,
    0x00ffff01ff0000ff,
    0x00ffff01ff000001,
    0x00ffff01ff010000,
    0x00ffff0100ffff00,
    0x00ffff010000ff01,
    0x00ffff0100000000,
    0x00ffff0100000101,
    0x00ffff01000100ff,
    0x00ffff0100010100,
    0x00ffff0101ff0100,
    0x00ffff01010000ff,
    0x00ffff0101010000,
    0x00ff00ffffffff00,
    0x00ff00ffff000000,
    0x00ff00ffff000100,
    0x00ff00ffff010100,
    0x00ff00ff00ff0000,
    0x00ff00ff00ff01ff,
    0x00ff00ff00ff0101,
    0x00ff00ff0000ff00,
    0x00ff00ff000000ff,
    0x00ff00ff00000000,
    0x00ff00ff00000001,
    0x00ff00ff0001ff00,
    0x00ff00ff0001ff01,
    0x00ff00ff00010000,
    0x00ff00ff000101ff,
    0x00ff00ff00010101,
    0x00ff00ff01ffff00,
    0x00ff00ff01ff0001,
    0x00ff00ff01ff0100,
    0x00ff00ff0100ffff,
    0x00ff00ff0100ff01,
    0x00ff00ff01000000,
    0x00ff00ff0101ffff,
    0x00ff00ff0101ff00,
    0x00ff00ff01010100,
    0x00ff0000ffffff00,
    0x00ff0000ffffff01,
    0x00ff0000ffff0000,
    0x00ff0000ffff0101,
    0x00ff0000ff00ff00,
    0x00ff0000ff0000ff,
    0x00ff0000ff000000,
    0x00ff0000ff000001,
    0x00ff0000ff000100,
    0x00ff0000ff01ffff,
    0x00ff0000ff010000,
    0x00ff0000ff010101,
    0x00ff000000ffff00,
    0x00ff000000ff00ff,
    0x00ff000000ff0000,
    0x00ff000000ff0001,
    0x00ff000000ff0100,
    0x00ff00000000ffff,
    0x00ff00000000ff00,
    0x00ff0000000000ff,
    0x00ff000000000000,
    0x00ff000000000001,
    0x00ff0000000001ff,
    0x00ff000000000100,
    0x00ff00000001ff00,
    0x00ff0000000100ff,
    0x00ff000000010000,
    0x00ff000000010001,
    0x00ff000000010100,
    0x00ff000001ffff01,
    0x00ff000001ff00ff,
    0x00ff000001ff0000,
    0x00ff000001ff01ff,
    0x00ff00000100ff00,
    0x00ff0000010000ff,
    0x00ff000001000000,
    0x00ff000001000001,
    0x00ff000001000100,
    0x00ff000001000101,
    0x00ff000001010000,
    0x00ff0000010101ff,
    0x00ff000001010101,
    0x00ff0001ffffff00,
    0x00ff0001ffff0000,
    0x00ff0001ffff0100,
    0x00ff0001ff0000ff,
    0x00ff0001ff000000,
    0x00ff0001ff0001ff,
    0x00ff0001ff000101,
    0x00ff0001ff01ff00,
    0x00ff0001ff0100ff,
    0x00ff0001ff010100,
    0x00ff000100ffffff,
    0x00ff000100ffff01,
    0x00ff000100ff0000,
    0x00ff000100ff01ff,
    0x00ff00010000ffff,
    0x00ff00010000ff00,
    0x00ff00010000ff01,
    0x00ff000100000000,
    0x00ff000100000001,
    0x00ff000100000100,
    0x00ff00010001ff01,
    0x00ff000100010000,
    0x00ff0001000101ff,
    0x00ff000101ffff00,
    0x00ff000101ff0000,
    0x00ff000101ff0101,
    0x00ff0001010000ff,
    0x00ff000101000000,
    0x00ff00010101ff00,
    0x00ff0001010100ff,
    0x00ff000101010001,
    0x00ff01ffffff0000,
    0x00ff01ffff00ff00,
    0x00ff01ffff000000,
    0x00ff01ffff000101,
    0x00ff01ffff010000,
    0x00ff01ff00ffff01,
    0x00ff01ff00ff0100,
    0x00ff01ff0000ffff,
    0x00ff01ff00000000,
    0x00ff01ff000001ff,
    0x00ff01ff0001ff00,
    0x00ff01ff000100ff,
    0x00ff01ff00010001,
    0x00ff01ff00010100,
    0x00ff01ff01ff0000,
    0x00ff01ff0100ff00,
    0x00ff01ff010000ff,
    0x00ff01ff01000001,
    0x00ff01ff01000100,
    0x00ff01ff01010000,
    0x00ff0100ffffff00,
    0x00ff0100ffff0000,
    0x00ff0100ffff0001,
    0x00ff0100ffff0101,
    0x00ff0100ff00ffff,
    0x00ff0100ff0000ff,
    0x00ff0100ff000000,
    0x00ff0100ff0001ff,
    0x00ff0100ff01ff00,
    0x00ff0100ff0100ff,
    0x00ff0100ff010001,
    0x00ff010000ffffff,
    0x00ff010000ff0000,
    0x00ff010000ff0101,
    0x00ff01000000ff00,
    0x00ff01000000ff01,
    0x00ff0100000000ff,
    0x00ff010000000000,
    0x00ff010000000001,
    0x00ff010000000100,
    0x00ff01000001ffff,
    0x00ff01000001ff01,
    0x00ff010000010000,
    0x00ff010000010001,
    0x00ff010000010101,
    0x00ff010001ff0001,
    0x00ff010001ff0100,
    0x00ff01000100ff01,
    0x00ff010001000000,
    0x00ff010001000001,
    0x00ff0100010001ff,
    0x00ff01000101ff00,
    0x00ff0100010100ff,
    0x00ff010001010001,
    0x00ff010001010100,
    0x00ff0101ff000001,
    0x00ff010100ff00ff,
    0x00ff010100ff0001,
    0x00ff010100ff0100,
    0x00ff010100000000,
    0x00ff0101000001ff,
    0x00ff010100000101,
    0x00ff0101000100ff,
    0x00ff010100010100,
    0x00ff0101010000ff,
    0x00ff010101010000,
    0x0000ffffffffff00,
    0x0000ffffffff00ff,
    0x0000ffffffff0000,
    0x0000ffffffff0001,
    0x0000ffffffff0100,
    0x0000ffffff00ff01,
    0x0000ffffff000000,
    0x0000ffffff000101,
    0x0000ffffff01ff00,
    0x0000ffffff0100ff,
    0x0000ffffff010100,
    0x0000ffff00ffffff,
    0x0000ffff00ff0000,
    0x0000ffff00ff01ff,
    0x0000ffff0000ff00,
    0x0000ffff000000ff,
    0x0000ffff00000000,
    0x0000ffff00000001,
    0x0000ffff00000100,
    0x0000ffff00010000,
    0x0000ffff000101ff,
    0x0000ffff01ff0001,
    0x0000ffff01ff0100,
    0x0000ffff01000000,
    0x0000ffff010001ff,
    0x0000ffff0101ffff,
    0x0000ffff0101ff00,
    0x0000ffff01010001,
    0x0000ffff01010100,
    0x0000ff00ffff0000,
    0x0000ff00ffff01ff,
    0x0000ff00ffff0100,
    0x0000ff00ffff0101,
    0x0000ff00ff00ff00,
    0x0000ff00ff0000ff,
    0x0000ff00ff000000,
    0x0000ff00ff000001,
    0x0000ff00ff0001ff,
    0x0000ff00ff000100,
    0x0000ff00ff01ffff,
    0x0000ff00ff010000,
    0x0000ff00ff010001,
    0x0000ff00ff0101ff,
    0x0000ff00ff010101,
    0x0000ff0000ffff00,
    0x0000ff0000ff00ff,
    0x0000ff0000ff0000,
    0x0000ff0000ff0001,
    0x0000ff0000ff0100,
    0x0000ff000000ffff,
    0x0000ff000000ff00,
    0x0000ff000000ff01,
    0x0000ff00000000ff,
    0x0000ff0000000000,
    0x0000ff0000000001,
    0x0000ff00000001ff,
    0x0000ff0000000100,
    0x0000ff0000000101,
    0x0000ff000001ff00,
    0x0000ff00000100ff,
    0x0000ff0000010000,
    0x0000ff0000010001,
    0x0000ff0000010100,
    0x0000ff0001ffff01,
    0x0000ff0001ff0000,
    0x0000ff000100ff00,
    0x0000ff00010000ff,
    0x0000ff0001000000,
    0x0000ff0001000001,
    0x0000ff0001000100,
    0x0000ff000101ffff,
    0x0000ff0001010000,
    0x0000ff0001010101,
    0x0000ff01ffffff00,
    0x0000ff01ffff0001,
    0x0000ff01ff00ff01,
    0x0000ff01ff000000,
    0x0000ff01ff000101,
    0x0000ff01ff01ff00,
    0x0000ff01ff0100ff,
    0x0000ff0100ffff01,
    0x0000ff0100ff0000,
    0x0000ff0100ff0101,
    0x0000ff010000ff00,
    0x0000ff01000000ff,
    0x0000ff0100000000,
    0x0000ff0100000001,
    0x0000ff0100000100,
    0x0000ff010001ff01,
    0x0000ff0100010000,
    0x0000ff0101ff0000,
    0x0000ff010100ffff,
    0x0000ff010100ff01,
    0x0000ff0101000000,
    0x0000ff0101000100,
    0x0000ff0101000101,
    0x0000ff01010100ff,
    0x000000ffffff00ff,
    0x000000ffffff0000,
    0x000000ffff00ff00,
    0x000000ffff0000ff,
    0x000000ffff000000,
    0x000000ffff000001,
    0x000000ffff0001ff,
    0x000000ffff000100,
    0x000000ffff01ff00,
    0x000000ffff010000,
    0x000000ffff0101ff,
    0x000000ffff010101,
    0x000000ff00ffff00,
    0x000000ff00ff00ff,
    0x000000ff00ff0000,
    0x000000ff00ff0001,
    0x000000ff00ff0100,
    0x000000ff00ff0101,
    0x000000ff0000ffff,
    0x000000ff0000ff00,
    0x000000ff000000ff,
    0x000000ff00000000,
    0x000000ff00000001,
    0x000000ff000001ff,
    0x000000ff00000100,
    0x000000ff00000101,
    0x000000ff0001ff00,
    0x000000ff0001ff01,
    0x000000ff000100ff,
    0x000000ff00010000,
    0x000000ff00010001,
    0x000000ff00010100,
    0x000000ff01ffffff,
    0x000000ff01ff01ff,
    0x000000ff01ff0101,
    0x000000ff0100ff00,
    0x000000ff010000ff,
    0x000000ff01000000,
    0x000000ff01000001,
    0x000000ff01000100,
    0x000000ff0101ff00,
    0x000000ff010100ff,
    0x000000ff01010000,
    0x000000ff01010101,
    0x00000000ffffff00,
    0x00000000ffffff01,
    0x00000000ffff00ff,
    0x00000000ffff0000,
    0x00000000ffff0001,
    0x00000000ffff0100,
    0x00000000ff00ffff,
    0x00000000ff00ff00,
    0x00000000ff00ff01,
    0x00000000ff0000ff,
    0x00000000ff000000,
    0x00000000ff000001,
    0x00000000ff000100,
    0x00000000ff000101,
    0x00000000ff01ff00,
    0x00000000ff0100ff,
    0x00000000ff010000,
    0x00000000ff010001,
    0x00000000ff010100,
    0x0000000000ffffff,
    0x0000000000ffff00,
    0x0000000000ffff01,
    0x0000000000ff00ff,
    0x0000000000ff0000,
    0x0000000000ff0001,
    0x0000000000ff01ff,
    0x0000000000ff0100,
    0x000000000000ffff,
    0x000000000000ff00,
    0x000000000000ff01,
    0x00000000000000ff,
    0x0000000000000000,
    0x0000000000000001,
    0x00000000000001ff,
    0x0000000000000100,
    0x0000000000000101,
    0x000000000001ffff,
    0x000000000001ff00,
    0x00000000000100ff,
    0x0000000000010000,
    0x0000000000010001,
    0x00000000000101ff,
    0x0000000000010100,
    0x0000000000010101,
    0x0000000001ffff00,
    0x0000000001ff00ff,
    0x0000000001ff0000,
    0x0000000001ff0100,
    0x0000000001ff0101,
    0x000000000100ffff,
    0x000000000100ff00,
    0x00000000010000ff,
    0x0000000001000000,
    0x0000000001000001,
    0x00000000010001ff,
    0x0000000001000100,
    0x000000000101ff00,
    0x00000000010100ff,
    0x0000000001010000,
    0x0000000001010001,
    0x0000000001010100,
    0x00000001ffffffff,
    0x00000001ffffff00,
    0x00000001ffffff01,
    0x00000001ffff00ff,
    0x00000001ffff0001,
    0x00000001ffff01ff,
    0x00000001ffff0100,
    0x00000001ff00ff00,
    0x00000001ff0000ff,
    0x00000001ff000000,
    0x00000001ff0001ff,
    0x00000001ff000100,
    0x00000001ff01ffff,
    0x00000001ff01ff00,
    0x00000001ff01ff01,
    0x00000001ff0100ff,
    0x00000001ff010000,
    0x00000001ff010001,
    0x00000001ff0101ff,
    0x00000001ff010100,
    0x0000000100ffff00,
    0x0000000100ff0000,
    0x0000000100ff0001,
    0x0000000100ff01ff,
    0x0000000100ff0100,
    0x0000000100ff0101,
    0x000000010000ffff,
    0x000000010000ff00,
    0x000000010000ff01,
    0x00000001000000ff,
    0x0000000100000000,
    0x0000000100000001,
    0x00000001000001ff,
    0x0000000100000100,
    0x0000000100000101,
    0x000000010001ff00,
    0x00000001000100ff,
    0x0000000100010000,
    0x0000000100010100,
    0x0000000101ffff01,
    0x0000000101ff0000,
    0x0000000101ff0001,
    0x0000000101ff01ff,
    0x0000000101ff0100,
    0x0000000101ff0101,
    0x000000010100ff00,
    0x0000000101000000,
    0x0000000101000101,
    0x000000010101ff01,
    0x0000000101010000,
    0x0000000101010001,
    0x00000001010101ff,
    0x0000000101010100,
    0x000001ffffff00ff,
    0x000001ffffff0000,
    0x000001ffffff0001,
    0x000001ffffff0100,
    0x000001ffff00ffff,
    0x000001ffff000000,
    0x000001ffff0001ff,
    0x000001ffff01ff00,
    0x000001ffff010101,
    0x000001ff00ff0000,
    0x000001ff00ff01ff,
    0x000001ff00ff0101,
    0x000001ff0000ff00,
    0x000001ff000000ff,
    0x000001ff00000000,
    0x000001ff00000001,
    0x000001ff000001ff,
    0x000001ff00000100,
    0x000001ff0001ffff,
    0x000001ff0001ff01,
    0x000001ff000100ff,
    0x000001ff00010000,
    0x000001ff01ffff01,
    0x000001ff01ff0100,
    0x000001ff0100ffff,
    0x000001ff0100ff01,
    0x000001ff01000000,
    0x000001ff010001ff,
    0x000001ff0101ff00,
    0x000001ff01010100,
    0x00000100ffffff00,
    0x00000100ffffff01,
    0x00000100ffff0000,
    0x00000100ffff0101,
    0x00000100ff00ff00,
    0x00000100ff0000ff,
    0x00000100ff000000,
    0x00000100ff000001,
    0x00000100ff000100,
    0x00000100ff010000,
    0x0000010000ffff00,
    0x0000010000ff00ff,
    0x0000010000ff0000,
    0x0000010000ff0001,
    0x0000010000ff0100,
    0x000001000000ffff,
    0x000001000000ff00,
    0x000001000000ff01,
    0x00000100000000ff,
    0x0000010000000000,
    0x0000010000000001,
    0x00000100000001ff,
    0x0000010000000100,
    0x0000010000000101,
    0x000001000001ff00,
    0x00000100000100ff,
    0x0000010000010000,
    0x0000010000010001,
    0x0000010000010100,
    0x0000010001ffff00,
    0x0000010001ff0000,
    0x0000010001ff0100,
    0x000001000100ff00,
    0x00000100010000ff,
    0x0000010001000000,
    0x0000010001000001,
    0x00000100010001ff,
    0x0000010001000100,
    0x0000010001010000,
    0x00000101ffff00ff,
    0x00000101ffff01ff,
    0x00000101ff000000,
    0x00000101ff000101,
    0x00000101ff01ffff,
    0x00000101ff010000,
    0x00000101ff010001,
    0x00000101ff010100,
    0x0000010100ff0000,
    0x0000010100ff01ff,
    0x0000010100ff0100,
    0x000001010000ff00,
    0x0000010100000000,
    0x0000010100000001,
    0x00000101000001ff,
    0x0000010100000100,
    0x000001010001ff01,
    0x0000010100010000,
    0x00000101000101ff,
    0x0000010100010101,
    0x0000010101ffff00,
    0x0000010101ff0101,
    0x000001010100ff01,
    0x0000010101000000,
    0x0000010101000001,
    0x00000101010001ff,
    0x0000010101000101,
    0x000001010101ff00,
    0x0001ffffffff0000,
    0x0001ffffff0000ff,
    0x0001ffffff000001,
    0x0001ffffff000100,
    0x0001ffffff010000,
    0x0001ffff00ff00ff,
    0x0001ffff0000ffff,
    0x0001ffff00000000,
    0x0001ffff00000001,
    0x0001ffff000001ff,
    0x0001ffff00000101,
    0x0001ffff0001ff00,
    0x0001ffff000100ff,
    0x0001ffff00010001,
    0x0001ffff00010100,
    0x0001ffff01ffff00,
    0x0001ffff01000001,
    0x0001ffff01010000,
    0x0001ff00ffffff00,
    0x0001ff00ffff00ff,
    0x0001ff00ffff0001,
    0x0001ff00ffff0100,
    0x0001ff00ff00ff01,
    0x0001ff00ff000000,
    0x0001ff00ff01ff00,
    0x0001ff00ff01ff01,
    0x0001ff00ff010001,
    0x0001ff00ff010100,
    0x0001ff0000ff0000,
    0x0001ff0000ff0100,
    0x0001ff000000ff00,
    0x0001ff0000000000,
    0x0001ff0000000001,
    0x0001ff0000000100,
    0x0001ff0000010000,
    0x0001ff0000010001,
    0x0001ff0000010101,
    0x0001ff0001ff00ff,
    0x0001ff0001ff0101,
    0x0001ff000100ff01,
    0x0001ff0001000000,
    0x0001ff000101ff00,
    0x0001ff0001010001,
    0x0001ff0001010100,
    0x0001ff01ff00ff00,
    0x0001ff01ff000001,
    0x0001ff01ff000100,
    0x0001ff0100ffffff,
    0x0001ff0100ffff00,
    0x0001ff0100ff0001,
    0x0001ff0100000000,
    0x0001ff0100000001,
    0x0001ff01000001ff,
    0x0001ff010001ffff,
    0x0001ff0101ff0000,
    0x0001ff010100ff00,
    0x0001ff0101000001,
    0x0001ff0101010000,
    0x000100ffff00ff00,
    0x000100ffff00ff01,
    0x000100ffff000000,
    0x000100ffff000001,
    0x000100ffff000101,
    0x000100ffff01ff00,
    0x000100ffff010001,
    0x000100ffff010100,
    0x000100ff00ffffff,
    0x000100ff00ffff01,
    0x000100ff00ff0000,
    0x000100ff00ff01ff,
    0x000100ff00ff0101,
    0x000100ff0000ff00,
    0x000100ff000000ff,
    0x000100ff00000000,
    0x000100ff00000001,
    0x000100ff00000100,
    0x000100ff00000101,
    0x000100ff0001ffff,
    0x000100ff0001ff01,
    0x000100ff00010000,
    0x000100ff01ff00ff,
    0x000100ff01ff0000,
    0x000100ff01ff0100,
    0x000100ff0100ffff,
    0x000100ff0100ff01,
    0x000100ff010000ff,
    0x000100ff01000000,
    0x000100ff01000001,
    0x000100ff010001ff,
    0x000100ff01000101,
    0x000100ff0101ff00,
    0x000100ff010100ff,
    0x000100ff01010100,
    0x00010000ffff0000,
    0x00010000ffff01ff,
    0x00010000ffff0101,
    0x00010000ff00ff00,
    0x00010000ff000000,
    0x00010000ff000001,
    0x00010000ff000100,
    0x0001000000ff00ff,
    0x0001000000ff0000,
    0x0001000000ff0001,
    0x0001000000ff0100,
    0x000100000000ffff,
    0x000100000000ff00,
    0x00010000000000ff,
    0x0001000000000000,
    0x0001000000000001,
    0x0001000000000100,
    0x000100000001ff00,
    0x00010000000100ff,
    0x0001000000010000,
    0x0001000000010001,
    0x0001000000010100,
    0x0001000001ff0001,
    0x0001000001ff0100,
    0x0001000001ff0101,
    0x000100000100ff00,
    0x0001000001000000,
    0x0001000001000001,
    0x0001000001000100,
    0x0001000001000101,
    0x000100000101ff01,
    0x0001000001010000,
    0x0001000001010001,
    0x00010000010101ff,
    0x00010001ffffff01,
    0x00010001ffff0100,
    0x00010001ff000000,
    0x00010001ff01ffff,
    0x00010001ff010001,
    0x00010001ff0101ff,
    0x00010001ff010100,
    0x0001000100ffffff,
    0x0001000100ff0000,
    0x0001000100ff01ff,
    0x0001000100ff0101,
    0x000100010000ff00,
    0x00010001000000ff,
    0x0001000100000000,
    0x0001000100000001,
    0x00010001000001ff,
    0x0001000100000101,
    0x000100010001ffff,
    0x0001000100010000,
    0x00010001000101ff,
    0x0001000101ffffff,
    0x0001000101ffff01,
    0x0001000101ff0000,
    0x0001000101ff0101,
    0x00010001010000ff,
    0x0001000101000001,
    0x00010001010001ff,
    0x0001000101000100,
    0x000100010101ffff,
    0x00010001010100ff,
    0x0001000101010001,
    0x0001000101010101,
    0x000101ffff000001,
    0x000101ffff000100,
    0x000101ffff010000,
    0x000101ff00ffff00,
    0x000101ff0000ff01,
    0x000101ff00000000,
    0x000101ff00000101,
    0x000101ff0001ff00,
    0x000101ff00010100,
    0x000101ff01ff0000,
    0x000101ff0100ff00,
    0x000101ff010001ff,
    0x000101ff01010001,
    0x00010100ffffff00,
    0x00010100ffff00ff,
    0x00010100ff00ffff,
    0x00010100ff000000,
    0x00010100ff01ff00,
    0x00010100ff0100ff,
    0x00010100ff010001,
    0x00010100ff010100,
    0x0001010000ffffff,
    0x0001010000ffff00,
    0x0001010000ff0000,
    0x0001010000ff0001,
    0x0001010000ff01ff,
    0x000101000000ff00,
    0x00010100000000ff,
    0x0001010000000000,
    0x0001010000000001,
    0x0001010000000100,
    0x000101000001ffff,
    0x0001010000010000,
    0x0001010000010101,
    0x0001010001ffff01,
    0x0001010001ff00ff,
    0x0001010001ff0101,
    0x0001010001000000,
    0x000101000101ff00,
    0x00010100010100ff,
    0x0001010001010000,
    0x0001010001010100,
    0x00010101ff00ff00,
    0x00010101ff000001,
    0x00010101ff0001ff,
    0x0001010100ffff00,
    0x0001010100ff00ff,
    0x0001010100ff0100,
    0x000101010000ffff,
    0x0001010100000000,
    0x00010101000001ff,
    0x0001010100000101,
    0x00010101000100ff,
    0x0001010100010000,
    0x0001010100010100,
    0x0001010101ff0001,
    0x00010101010000ff,
    0x00010101010001ff,
    0x0001010101000101,
    0x0001010101010001,
    0x01ffffffffffffff,
    0x01ffffffffffff01,
    0x01ffffffffff01ff,
    0x01ffffffffff0101,
    0x01ffffffff01ffff,
    0x01ffffffff01ff01,
    0x01ffffffff0101ff,
    0x01ffffffff010101,
    0x01ffffff00ff0000,
    0x01ffffff0000ffff,
    0x01ffffff0000ff00,
    0x01ffffff000000ff,
    0x01ffffff00000001,
    0x01ffffff00000100,
    0x01ffffff00010000,
    0x01ffffff01ffffff,
    0x01ffffff01ffff01,
    0x01ffffff01ff01ff,
    0x01ffffff01ff0101,
    0x01ffffff01000000,
    0x01ffffff0101ffff,
    0x01ffffff0101ff01,
    0x01ffffff010101ff,
    0x01ffffff01010101,
    0x01ffff00ffff0000,
    0x01ffff00ff00ff00,
    0x01ffff00ff0000ff,
    0x01ffff00ff000001,
    0x01ffff00ff000100,
    0x01ffff00ff010000,
    0x01ffff0000ffff00,
    0x01ffff0000ff00ff,
    0x01ffff0000ff0100,
    0x01ffff000000ffff,
    0x01ffff000000ff01,
    0x01ffff0000000000,
    0x01ffff0000000001,
    0x01ffff00000001ff,
    0x01ffff0000000100,
    0x01ffff00000100ff,
    0x01ffff0000010001,
    0x01ffff0000010100,
    0x01ffff0001ff0000,
    0x01ffff0001ff0100,
    0x01ffff00010000ff,
    0x01ffff0001000001,
    0x01ffff0001000100,
    0x01ffff0001010000,
    0x01ffff01ffffffff,
    0x01ffff01ffffff01,
    0x01ffff01ffff01ff,
    0x01ffff01ffff0101,
    0x01ffff01ff000000,
    0x01ffff01ff01ffff,
    0x01ffff01ff01ff01,
    0x01ffff01ff0101ff,
    0x01ffff01ff010101,
    0x01ffff010000ff00,
    0x01ffff01000000ff,
    0x01ffff0100000100,
    0x01ffff0100010000,
    0x01ffff0101ffffff,
    0x01ffff0101ffff01,
    0x01ffff0101ff01ff,
    0x01ffff0101ff0101,
    0x01ffff0101000000,
    0x01ffff010101ffff,
    0x01ffff010101ff01,
    0x01ffff01010101ff,
    0x01ffff0101010101,
    0x01ff00ffff0000ff,
    0x01ff00ffff000100,
    0x01ff00ff00ffff00,
    0x01ff00ff00ff00ff,
    0x01ff00ff0000ff00,
    0x01ff00ff00000000,
    0x01ff00ff00000101,
    0x01ff00ff0001ff00,
    0x01ff00ff000100ff,
    0x01ff00ff00010100,
    0x01ff00ff010000ff,
    0x01ff00ff01000100,
    0x01ff0000ffffff00,
    0x01ff0000ffff0100,
    0x01ff0000ff00ff01,
    0x01ff0000ff000000,
    0x01ff0000ff000101,
    0x01ff0000ff010001,
    0x01ff0000ff010100,
    0x01ff000000ffffff,
    0x01ff000000ffff00,
    0x01ff000000ff0000,
    0x01ff000000ff01ff,
    0x01ff00000000ff00,
    0x01ff0000000000ff,
    0x01ff000000000000,
    0x01ff000000000001,
    0x01ff000000000100,
    0x01ff000000000101,
    0x01ff000000010000,
    0x01ff000000010001,
    0x01ff0000000101ff,
    0x01ff000000010101,
    0x01ff000001ffff00,
    0x01ff000001ff00ff,
    0x01ff000001ff0001,
    0x01ff000001ff0100,
    0x01ff00000100ffff,
    0x01ff00000100ff01,
    0x01ff000001000000,
    0x01ff0000010001ff,
    0x01ff000001010001,
    0x01ff0001ff00ff00,
    0x01ff0001ff000001,
    0x01ff0001ff000100,
    0x01ff0001ff010000,
    0x01ff000100ffff00,
    0x01ff000100ff00ff,
    0x01ff000100ff0100,
    0x01ff000100ff0101,
    0x01ff00010000ffff,
    0x01ff000100000000,
    0x01ff000100000100,
    0x01ff000100000101,
    0x01ff00010001ff00,
    0x01ff000100010001,
    0x01ff000100010101,
    0x01ff000101ff0000,
    0x01ff00010100ff00,
    0x01ff000101000101,
    0x01ff0001010100ff,
    0x01ff01ffffffffff,
    0x01ff01ffffffff01,
    0x01ff01ffffff01ff,
    0x01ff01ffffff0101,
    0x01ff01ffff000000,
    0x01ff01ffff01ffff,
    0x01ff01ffff01ff01,
    0x01ff01ffff0101ff,
    0x01ff01ffff010101,
    0x01ff01ff00ffff00,
    0x01ff01ff00ff0000,
    0x01ff01ff0000ff00,
    0x01ff01ff000000ff,
    0x01ff01ff00000100,
    0x01ff01ff00010000,
    0x01ff01ff00010100,
    0x01ff01ff01ffffff,
    0x01ff01ff01ffff01,
    0x01ff01ff01ff01ff,
    0x01ff01ff01ff0101,
    0x01ff01ff01000000,
    0x01ff01ff0101ffff,
    0x01ff01ff0101ff01,
    0x01ff01ff010101ff,
    0x01ff01ff01010101,
    0x01ff0100ffff0000,
    0x01ff0100ffff0001,
    0x01ff0100ff00ff00,
    0x01ff0100ff0000ff,
    0x01ff0100ff000001,
    0x01ff0100ff010000,
    0x01ff010000ffff00,
    0x01ff010000ff00ff,
    0x01ff010000ff0001,
    0x01ff010000ff0100,
    0x01ff01000000ffff,
    0x01ff01000000ff01,
    0x01ff010000000000,
    0x01ff010000000101,
    0x01ff01000001ff00,
    0x01ff0100000100ff,
    0x01ff010001ff0000,
    0x01ff010001000001,
    0x01ff010001000100,
    0x01ff010001010000,
    0x01ff0101ffffffff,
    0x01ff0101ffffff01,
    0x01ff0101ffff01ff,
    0x01ff0101ffff0101,
    0x01ff0101ff000000,
    0x01ff0101ff01ffff,
    0x01ff0101ff01ff01,
    0x01ff0101ff0101ff,
    0x01ff0101ff010101,
    0x01ff010100ff0000,
    0x01ff01010000ff00,
    0x01ff0101000000ff,
    0x01ff010100000001,
    0x01ff010101ffffff,
    0x01ff010101ffff01,
    0x01ff010101ff01ff,
    0x01ff010101ff0101,
    0x01ff010101000000,
    0x01ff01010101ffff,
    0x01ff01010101ff01,
    0x01ff0101010101ff,
    0x01ff010101010101,
    0x0100ffffffff0000,
    0x0100ffffff00ff00,
    0x0100ffffff000001,
    0x0100ffffff0001ff,
    0x0100ffffff000100,
    0x0100ffffff010000,
    0x0100ffff00ffff00,
    0x0100ffff00ff0001,
    0x0100ffff00ff0100,
    0x0100ffff00000000,
    0x0100ffff000001ff,
    0x0100ffff00000101,
    0x0100ffff00010100,
    0x0100ffff00010101,
    0x0100ffff01ff0000,
    0x0100ffff0100ff00,
    0x0100ffff010000ff,
    0x0100ffff01000001,
    0x0100ffff01000100,
    0x0100ffff01010000,
    0x0100ff00ffffff00,
    0x0100ff00ffff00ff,
    0x0100ff00ffff0001,
    0x0100ff00ffff0100,
    0x0100ff00ff00ffff,
    0x0100ff00ff000000,
    0x0100ff00ff0001ff,
    0x0100ff00ff000101,
    0x0100ff00ff01ff00,
    0x0100ff00ff0100ff,
    0x0100ff00ff010001,
    0x0100ff00ff010100,
    0x0100ff0000ffffff,
    0x0100ff0000ff0000,
    0x0100ff000000ffff,
    0x0100ff000000ff00,
    0x0100ff00000000ff,
    0x0100ff0000000000,
    0x0100ff0000000001,
    0x0100ff0000000100,
    0x0100ff000001ff01,
    0x0100ff0000010000,
    0x0100ff0001ff00ff,
    0x0100ff0001ff0001,
    0x0100ff000100ff01,
    0x0100ff0001000000,
    0x0100ff00010001ff,
    0x0100ff000101ff00,
    0x0100ff00010100ff,
    0x0100ff0001010001,
    0x0100ff0001010100,
    0x0100ff01ffff0000,
    0x0100ff01ff00ff00,
    0x0100ff01ff0000ff,
    0x0100ff01ff000100,
    0x0100ff01ff010000,
    0x0100ff0100ff00ff,
    0x0100ff0100ff0001,
    0x0100ff0100ff0100,
    0x0100ff010000ffff,
    0x0100ff010000ff01,
    0x0100ff0100000000,
    0x0100ff01000001ff,
    0x0100ff0100010001,
    0x0100ff0100010100,
    0x0100ff0101ff0000,
    0x0100ff01010000ff,
    0x0100ff0101000001,
    0x0100ff0101010100,
    0x010000ffffffff00,
    0x010000ffffff00ff,
    0x010000ffffff0001,
    0x010000ffff00ffff,
    0x010000ffff000000,
    0x010000ffff0001ff,
    0x010000ffff010001,
    0x010000ff00ffffff,
    0x010000ff00ff0101,
    0x010000ff0000ff00,
    0x010000ff000000ff,
    0x010000ff00000000,
    0x010000ff00000001,
    0x010000ff000001ff,
    0x010000ff00000100,
    0x010000ff0001ffff,
    0x010000ff0001ff00,
    0x010000ff0001ff01,
    0x010000ff00010000,
    0x010000ff01ff00ff,
    0x010000ff01ff0001,
    0x010000ff0100ff01,
    0x010000ff010000ff,
    0x010000ff01000000,
    0x010000ff010001ff,
    0x010000ff0101ff00,
    0x010000ff01010100,
    0x01000000ffffffff,
    0x01000000ffff0000,
    0x01000000ffff01ff,
    0x01000000ffff0101,
    0x01000000ff00ffff,
    0x01000000ff00ff00,
    0x01000000ff0000ff,
    0x01000000ff000000,
    0x01000000ff000001,
    0x01000000ff000100,
    0x01000000ff01ff00,
    0x01000000ff010000,
    0x01000000ff010100,
    0x01000000ff010101,
    0x0100000000ffff00,
    0x0100000000ff00ff,
    0x0100000000ff0000,
    0x0100000000ff0001,
    0x0100000000ff0100,
    0x010000000000ffff,
    0x010000000000ff00,
    0x010000000000ff01,
    0x01000000000000ff,
    0x0100000000000000,
    0x0100000000000001,
    0x01000000000001ff,
    0x0100000000000100,
    0x0100000000000101,
    0x010000000001ff00,
    0x01000000000100ff,
    0x0100000000010000,
    0x0100000000010001,
    0x0100000000010100,
    0x0100000001ffff00,
    0x0100000001ff0000,
    0x0100000001ff01ff,
    0x010000000100ff00,
    0x010000000100ff01,
    0x01000000010000ff,
    0x0100000001000000,
    0x0100000001000001,
    0x0100000001000100,
    0x0100000001000101,
    0x010000000101ffff,
    0x010000000101ff01,
    0x0100000001010000,
    0x01000000010101ff,
    0x0100000001010101,
    0x01000001ffffff00,
    0x01000001ffff00ff,
    0x01000001ff00ffff,
    0x01000001ff000000,
    0x01000001ff000100,
    0x01000001ff01ffff,
    0x01000001ff010001,
    0x01000001ff010100,
    0x0100000100ff0000,
    0x0100000100ff01ff,
    0x0100000100ff0100,
    0x010000010000ff00,
    0x010000010000ff01,
    0x0100000100000000,
    0x0100000100000001,
    0x0100000100000100,
    0x0100000100010000,
    0x01000001000101ff,
    0x0100000101ffff01,
    0x0100000101ff00ff,
    0x0100000101ff0100,
    0x0100000101ff0101,
    0x010000010100ff01,
    0x01000001010000ff,
    0x0100000101000000,
    0x01000001010100ff,
    0x0100000101010001,
    0x0100000101010100,
    0x010001ffffff0000,
    0x010001ffff000001,
    0x010001ffff000100,
    0x010001ffff010000,
    0x010001ff00ffff00,
    0x010001ff00ff0001,
    0x010001ff0000ffff,
    0x010001ff0000ff01,
    0x010001ff00000000,
    0x010001ff00000001,
    0x010001ff00000101,
    0x010001ff000100ff,
    0x010001ff00010000,
    0x010001ff01ff0000,
    0x010001ff0100ff00,
    0x010001ff01000001,
    0x010001ff01000100,
    0x010001ff01010000,
    0x01000100ffff00ff,
    0x01000100ffff0001,
    0x01000100ffff0100,
    0x01000100ff00ffff,
    0x01000100ff00ff01,
    0x01000100ff000000,
    0x01000100ff0001ff,
    0x01000100ff000101,
    0x01000100ff01ffff,
    0x01000100ff01ff00,
    0x01000100ff0100ff,
    0x01000100ff010001,
    0x0100010000ffffff,
    0x0100010000ffff01,
    0x0100010000ff0000,
    0x0100010000ff01ff,
    0x0100010000ff0101,
    0x010001000000ff00,
    0x01000100000000ff,
    0x0100010000000000,
    0x0100010000000001,
    0x0100010000000100,
    0x010001000001ff01,
    0x0100010000010000,
    0x0100010000010001,
    0x0100010000010101,
    0x0100010001ffff00,
    0x0100010001ff00ff,
    0x010001000100ffff,
    0x010001000100ff01,
    0x0100010001000000,
    0x0100010001000101,
    0x010001000101ff00,
    0x0100010001010001,
    0x01000101ffff0000,
    0x01000101ff000000,
    0x01000101ff010000,
    0x0100010100ff00ff,
    0x0100010100ff0001,
    0x0100010100ff0100,
    0x010001010000ffff,
    0x0100010100000000,
    0x01000101000001ff,
    0x010001010001ff00,
    0x0100010101ff0000,
    0x010001010100ff00,
    0x01000101010000ff,
    0x0100010101000000,
    0x0100010101000001,
    0x0101ffffffffffff,
    0x0101ffffffffff01,
    0x0101ffffffff01ff,
    0x0101ffffffff0101,
    0x0101ffffff000000,
    0x0101ffffff01ffff,
    0x0101ffffff01ff01,
    0x0101ffffff0101ff,
    0x0101ffffff010101,
    0x0101ffff00ff0000,
    0x0101ffff0000ff00,
    0x0101ffff000000ff,
    0x0101ffff00000001,
    0x0101ffff00000100,
    0x0101ffff01ffffff,
    0x0101ffff01ffff01,
    0x0101ffff01ff01ff,
    0x0101ffff01ff0101,
    0x0101ffff01000000,
    0x0101ffff0101ffff,
    0x0101ffff0101ff01,
    0x0101ffff010101ff,
    0x0101ffff01010101,
    0x0101ff00ffff0000,
    0x0101ff00ffff0100,
    0x0101ff00ff00ff00,
    0x0101ff00ff0000ff,
    0x0101ff00ff000001,
    0x0101ff00ff000100,
    0x0101ff00ff000101,
    0x0101ff0000ff0001,
    0x0101ff0000ff0100,
    0x0101ff000000ff00,
    0x0101ff0000000000,
    0x0101ff00000001ff,
    0x0101ff0000000101,
    0x0101ff000001ff00,
    0x0101ff00000100ff,
    0x0101ff0001ff0000,
    0x0101ff000100ffff,
    0x0101ff000100ff01,
    0x0101ff0001000001,
    0x0101ff0001000100,
    0x0101ff01ffffff01,
    0x0101ff01ffff01ff,
    0x0101ff01ffff0101,
    0x0101ff01ff00ffff,
    0x0101ff01ff000100,
    0x0101ff01ff01ff01,
    0x0101ff01ff0101ff,
    0x0101ff01ff010101,
    0x0101ff0100ff0000,
    0x0101ff010000ff00,
    0x0101ff0100000001,
    0x0101ff0100000100,
    0x0101ff0100010000,
    0x0101ff0101ffffff,
    0x0101ff0101ffff01,
    0x0101ff0101ff01ff,
    0x0101ff0101ff0101,
    0x0101ff0101000000,
    0x0101ff010101ffff,
    0x0101ff010101ff01,
    0x0101ff01010101ff,
    0x0101ff0101010101,
    0x010100ffff000100,
    0x010100ffff010000,
    0x010100ff00ffff00,
    0x010100ff00ff00ff,
    0x010100ff0000ffff,
    0x010100ff000000ff,
    0x010100ff00000000,
    0x010100ff000001ff,
    0x010100ff00000101,
    0x010100ff0001ff00,
    0x010100ff00010000,
    0x010100ff00010001,
    0x010100ff000101ff,
    0x010100ff00010100,
    0x010100ff01ff0000,
    0x01010000ffff0001,
    0x01010000ffff0100,
    0x01010000ff00ffff,
    0x01010000ff00ff01,
    0x01010000ff000000,
    0x01010000ff0001ff,
    0x01010000ff010001,
    0x01010000ff010100,
    0x0101000000ffff01,
    0x0101000000ff0000,
    0x010100000000ff00,
    0x01010000000000ff,
    0x0101000000000000,
    0x0101000000000001,
    0x0101000000000100,
    0x0101000000010000,
    0x0101000000010101,
    0x0101000001ffff00,
    0x0101000001ff00ff,
    0x0101000001ff0000,
    0x0101000001ff0001,
    0x0101000001ff0100,
    0x010100000100ff01,
    0x0101000001000000,
    0x01010000010001ff,
    0x01010001ffff0000,
    0x01010001ff00ff00,
    0x01010001ff000001,
    0x01010001ff000101,
    0x01010001ff01ff00,
    0x01010001ff010000,
    0x0101000100ff00ff,
    0x0101000100ff0001,
    0x0101000100ff0101,
    0x010100010000ff01,
    0x0101000100000000,
    0x0101000100000001,
    0x01010001000001ff,
    0x010100010001ffff,
    0x010100010001ff01,
    0x0101000101ff0001,
    0x010100010100ffff,
    0x0101000101000000,
    0x0101000101000001,
    0x0101000101000100,
    0x010100010101ff00,
    0x01010001010100ff,
    0x0101000101010001,
    0x010101ffffffffff,
    0x010101ffffffff01,
    0x010101ffffff01ff,
    0x010101ffffff0101,
    0x010101ffff01ffff,
    0x010101ffff01ff01,
    0x010101ffff0101ff,
    0x010101ffff010101,
    0x010101ff0000ff00,
    0x010101ff000000ff,
    0x010101ff00000001,
    0x010101ff00000100,
    0x010101ff01ffffff,
    0x010101ff01ffff01,
    0x010101ff01ff01ff,
    0x010101ff01ff0101,
    0x010101ff01000000,
    0x010101ff0101ffff,
    0x010101ff0101ff01,
    0x010101ff010101ff,
    0x010101ff01010101,
    0x01010100ffff0000,
    0x01010100ff0000ff,
    0x01010100ff000100,
    0x01010100ff01ff00,
    0x01010100ff010000,
    0x0101010000ffff00,
    0x010101000000ffff,
    0x0101010000000000,
    0x0101010000000101,
    0x010101000001ff00,
    0x0101010000010001,
    0x0101010000010100,
    0x010101000100ffff,
    0x0101010001000001,
    0x01010101ffffffff,
    0x01010101ffffff01,
    0x01010101ffff01ff,
    0x01010101ffff0101,
    0x01010101ff01ffff,
    0x01010101ff01ff01,
    0x01010101ff0101ff,
    0x01010101ff010101,
    0x010101010000ff00,
    0x01010101000000ff,
    0x0101010100000001,
    0x0101010101ffffff,
    0x0101010101ffff01,
    0x0101010101ff01ff,
    0x0101010101ff0101,
    0x0101010101000000,
    0x010101010101ffff,
    0x010101010101ff01,
    0x01010101010101ff,
    0x0101010101010101,
];

/// IQ2_XXS: 256 elements per block.
/// Layout: f16 d (2 bytes) + uint16_t qs[32] (64 bytes) = 66 bytes
/// qs is read as 8 groups of 4 u16 (32 bits each, stored as two consecutive u32).
/// Each 32-bit word: lower 16 bits = 8x2-bit quant indices, upper 16 bits = (4-bit scale+sign) x 4.
fn dequantize_iq2_xxs(data: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 66;

    for (block_idx, block) in data.chunks_exact(BLOCK_SIZE).enumerate() {
        let d = read_f16(block, 0);
        let qs = &block[2..66]; // 64 bytes = 32 u16 values
        let out = &mut output[block_idx * 256..(block_idx + 1) * 256];

        // 8 groups of 32 elements (each group uses 4 consecutive u16 = 8 bytes)
        for ib32 in 0..8usize {
            // Read 8 bytes as two u32
            let base = ib32 * 8;
            let aux0 = u32::from_le_bytes([qs[base], qs[base + 1], qs[base + 2], qs[base + 3]]);
            let aux1 = u32::from_le_bytes([qs[base + 4], qs[base + 5], qs[base + 6], qs[base + 7]]);

            // Scale: bits [28..31] of aux1 give a 4-bit value
            let db = d * (0.5 + ((aux1 >> 28) as f32)) * 0.25;

            // aux0 contains 16x 2-bit quant indices (8 per 16-bit half)
            // aux1 contains 4x (7-bit sign + ...) packing
            let aux8 = [
                (aux0 & 0xff) as u8,
                ((aux0 >> 8) & 0xff) as u8,
                ((aux0 >> 16) & 0xff) as u8,
                ((aux0 >> 24) & 0xff) as u8,
            ];

            for l in 0..4usize {
                let grid = IQ2XXS_GRID[aux8[l] as usize];
                let signs = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
                for j in 0..8usize {
                    let grid_val = ((grid >> (j * 8)) & 0xff) as i8 as f32;
                    let sign = if signs & KMASK_IQ2XS[j] != 0 {
                        -1.0f32
                    } else {
                        1.0f32
                    };
                    out[ib32 * 32 + l * 8 + j] = db * grid_val * sign;
                }
            }
        }
    }
}

/// IQ2_XS: 256 elements per block.
/// Layout: f16 d (2 bytes) + uint16_t qs[32] (64 bytes) + uint8_t scales[8] = 74 bytes
fn dequantize_iq2_xs(data: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 74;

    for (block_idx, block) in data.chunks_exact(BLOCK_SIZE).enumerate() {
        let d = read_f16(block, 0);
        let qs_bytes = &block[2..66]; // 64 bytes = 32 u16 values
        let scales = &block[66..74]; // 8 bytes
        let out = &mut output[block_idx * 256..(block_idx + 1) * 256];

        // 8 groups of 32 elements
        for ib32 in 0..8usize {
            let scale_byte = scales[ib32];
            let db0 = d * (0.5 + (scale_byte & 0xf) as f32) * 0.25;
            let db1 = d * (0.5 + ((scale_byte >> 4) & 0xf) as f32) * 0.25;

            for l in 0..4usize {
                let qs_idx = (ib32 * 4 + l) * 2;
                let qs_val = u16::from_le_bytes([qs_bytes[qs_idx], qs_bytes[qs_idx + 1]]) as u32;
                let grid_idx = (qs_val & 511) as usize;
                let sign_idx = ((qs_val >> 9) & 127) as usize;

                let grid = IQ2XS_GRID[grid_idx];
                let signs = KSIGNS_IQ2XS[sign_idx];
                let db = if l < 2 { db0 } else { db1 };

                for j in 0..8usize {
                    let grid_val = ((grid >> (j * 8)) & 0xff) as i8 as f32;
                    let sign = if signs & KMASK_IQ2XS[j] != 0 {
                        -1.0f32
                    } else {
                        1.0f32
                    };
                    out[ib32 * 32 + l * 8 + j] = db * grid_val * sign;
                }
            }
        }
    }
}

/// IQ2_S: 256 elements per block.
/// Layout: f16 d (2 bytes) + uint8_t qs[64] + uint8_t qh[8] + uint8_t scales[8] = 82 bytes
fn dequantize_iq2_s(data: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 82;

    for (block_idx, block) in data.chunks_exact(BLOCK_SIZE).enumerate() {
        let d = read_f16(block, 0);
        let qs = &block[2..66]; // 64 bytes
        let qh = &block[66..74]; // 8 bytes
        let scales = &block[74..82]; // 8 bytes
        let out = &mut output[block_idx * 256..(block_idx + 1) * 256];

        // signs are packed in qs[32..64] (second half)
        let signs_base = 32usize;

        for ib32 in 0..8usize {
            let scale_byte = scales[ib32];
            let db0 = d * (0.5 + (scale_byte & 0xf) as f32) * 0.25;
            let db1 = d * (0.5 + ((scale_byte >> 4) & 0xf) as f32) * 0.25;

            for l in 0..4usize {
                let qs_idx = ib32 * 4 + l;
                // 10-bit grid index: qs[qs_idx] as low 8 bits, 2 bits from qh
                let grid_idx =
                    (qs[qs_idx] as u32 | ((qh[ib32] as u32) << (8 - 2 * l) & 0x300)) as usize;
                let grid = IQ2S_GRID[grid_idx];

                let sign_idx = ib32 * 4 + l + signs_base;
                let sign_byte = if sign_idx < 64 { qs[sign_idx] } else { 0 };

                let db = if l < 2 { db0 } else { db1 };

                for j in 0..8usize {
                    let grid_val = ((grid >> (j * 8)) & 0xff) as i8 as f32;
                    let sign = if sign_byte & KMASK_IQ2XS[j] != 0 {
                        -1.0f32
                    } else {
                        1.0f32
                    };
                    out[ib32 * 32 + l * 8 + j] = db * grid_val * sign;
                }
            }
        }
    }
}

/// IQ3_XXS: 256 elements per block.
/// Layout: f16 d (2 bytes) + uint8_t qs[96] = 98 bytes
/// qs[0..64]: 2-bit lower quant indices; qs[64..96]: packed (sign,scale) info
fn dequantize_iq3_xxs(data: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 98;

    for (block_idx, block) in data.chunks_exact(BLOCK_SIZE).enumerate() {
        let d = read_f16(block, 0);
        let qs = &block[2..66]; // 64 bytes: 2-bit lower parts (256 indices, each byte has 4)
        let scales_and_signs = &block[66..98]; // 32 bytes

        let out = &mut output[block_idx * 256..(block_idx + 1) * 256];

        for ib32 in 0..8usize {
            let base = ib32 * 4;
            let aux32 = u32::from_le_bytes([
                scales_and_signs[base],
                scales_and_signs[base + 1],
                scales_and_signs[base + 2],
                scales_and_signs[base + 3],
            ]);
            let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;

            for l in 0..4usize {
                let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                let qs_offset = ib32 * 8 + l * 2;
                let grid1 = IQ3XXS_GRID[qs[qs_offset] as usize];
                let grid2 = IQ3XXS_GRID[qs[qs_offset + 1] as usize];

                for j in 0..4usize {
                    let g1 = ((grid1 >> (j * 8)) & 0xff) as i8 as f32;
                    let g2 = ((grid2 >> (j * 8)) & 0xff) as i8 as f32;
                    let s1 = if signs & KMASK_IQ2XS[j] != 0 {
                        -1.0f32
                    } else {
                        1.0f32
                    };
                    let s2 = if signs & KMASK_IQ2XS[j + 4] != 0 {
                        -1.0f32
                    } else {
                        1.0f32
                    };
                    out[ib32 * 32 + l * 8 + j] = db * g1 * s1;
                    out[ib32 * 32 + l * 8 + j + 4] = db * g2 * s2;
                }
            }
        }
    }
}

/// IQ3_S: 256 elements per block.
/// Layout: f16 d (2) + uint8_t qs[64] + uint8_t qh[8] + uint8_t signs[32] + uint8_t scales[4] = 110 bytes
fn dequantize_iq3_s(data: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 110;

    for (block_idx, block) in data.chunks_exact(BLOCK_SIZE).enumerate() {
        let d = read_f16(block, 0);
        let qs = &block[2..66]; // 64 bytes
        let qh = &block[66..74]; // 8 bytes
        let signs = &block[74..106]; // 32 bytes
        let sc = &block[106..110]; // 4 bytes (IQ3S_N_SCALE = 256/64 = 4)
        let out = &mut output[block_idx * 256..(block_idx + 1) * 256];

        let mut qs_off = 0usize;
        let mut signs_off = 0usize;
        let mut qh_off = 0usize;

        for ib32 in (0..8usize).step_by(2) {
            let db1 = d * (1.0 + 2.0 * (sc[ib32 / 2] & 0xf) as f32);
            let db2 = d * (1.0 + 2.0 * ((sc[ib32 / 2] >> 4) & 0xf) as f32);

            // First 32 of this pair (ib32)
            for l in 0..4usize {
                let hi_bit_1 = ((qh[qh_off] as u32) << (8 - 2 * l)) & 256;
                let hi_bit_2 = ((qh[qh_off] as u32) << (7 - 2 * l)) & 256;
                let grid1 = IQ3S_GRID[(qs[qs_off + 2 * l] as u32 | hi_bit_1) as usize];
                let grid2 = IQ3S_GRID[(qs[qs_off + 2 * l + 1] as u32 | hi_bit_2) as usize];
                let sign_byte = signs[signs_off + l];
                for j in 0..4usize {
                    let g1 = ((grid1 >> (j * 8)) & 0xff) as i8 as f32;
                    let g2 = ((grid2 >> (j * 8)) & 0xff) as i8 as f32;
                    let s1 = if sign_byte & KMASK_IQ2XS[j] != 0 {
                        -1.0f32
                    } else {
                        1.0f32
                    };
                    let s2 = if sign_byte & KMASK_IQ2XS[j + 4] != 0 {
                        -1.0f32
                    } else {
                        1.0f32
                    };
                    out[(ib32) * 32 + l * 8 + j] = db1 * g1 * s1;
                    out[(ib32) * 32 + l * 8 + j + 4] = db1 * g2 * s2;
                }
            }
            qs_off += 8;
            signs_off += 4;

            // Second 32 of this pair (ib32+1)
            for l in 0..4usize {
                let hi_bit_1 = ((qh[qh_off + 1] as u32) << (8 - 2 * l)) & 256;
                let hi_bit_2 = ((qh[qh_off + 1] as u32) << (7 - 2 * l)) & 256;
                let grid1 = IQ3S_GRID[(qs[qs_off + 2 * l] as u32 | hi_bit_1) as usize];
                let grid2 = IQ3S_GRID[(qs[qs_off + 2 * l + 1] as u32 | hi_bit_2) as usize];
                let sign_byte = signs[signs_off + l];
                for j in 0..4usize {
                    let g1 = ((grid1 >> (j * 8)) & 0xff) as i8 as f32;
                    let g2 = ((grid2 >> (j * 8)) & 0xff) as i8 as f32;
                    let s1 = if sign_byte & KMASK_IQ2XS[j] != 0 {
                        -1.0f32
                    } else {
                        1.0f32
                    };
                    let s2 = if sign_byte & KMASK_IQ2XS[j + 4] != 0 {
                        -1.0f32
                    } else {
                        1.0f32
                    };
                    out[(ib32 + 1) * 32 + l * 8 + j] = db2 * g1 * s1;
                    out[(ib32 + 1) * 32 + l * 8 + j + 4] = db2 * g2 * s2;
                }
            }
            qs_off += 8;
            signs_off += 4;
            qh_off += 2;
        }
    }
}

/// IQ1_S: 256 elements per block.
/// Layout: f16 d (2 bytes) + uint8_t qs[32] + uint16_t qh[8] = 50 bytes
fn dequantize_iq1_s(data: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 50;

    for (block_idx, block) in data.chunks_exact(BLOCK_SIZE).enumerate() {
        let d = read_f16(block, 0);
        let qs = &block[2..34]; // 32 bytes
        let qh_bytes = &block[34..50]; // 16 bytes = 8 x u16
        let out = &mut output[block_idx * 256..(block_idx + 1) * 256];

        for ib in 0..8usize {
            let qh = u16::from_le_bytes([qh_bytes[ib * 2], qh_bytes[ib * 2 + 1]]);
            let dl = d * (2.0 * ((qh >> 12) & 7) as f32 + 1.0);
            let delta = if qh & 0x8000 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            };

            for l in 0..4usize {
                let grid_idx =
                    (qs[ib * 4 + l] as u32 | (((qh >> (3 * l)) & 7) as u32) << 8) as usize;
                let grid = IQ1S_GRID[grid_idx];
                for j in 0..8usize {
                    let g = ((grid >> (j * 8)) & 0xff) as i8 as f32;
                    out[ib * 32 + l * 8 + j] = dl * (g + delta);
                }
            }
        }
    }
}

/// IQ1_M: 256 elements per block.
/// Layout (no d field!): uint8_t qs[32] + uint8_t qh[16] + uint8_t scales[8] = 56 bytes
/// The f16 scale is packed into the top 4 bits of each scales[i] word pair.
fn dequantize_iq1_m(data: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 56;

    for (block_idx, block) in data.chunks_exact(BLOCK_SIZE).enumerate() {
        let qs = &block[0..32];
        let qh = &block[32..48]; // 16 bytes
        let sc = &block[48..56]; // 8 bytes (scales[QK_K/32] = 8)
        let out = &mut output[block_idx * 256..(block_idx + 1) * 256];

        // Reconstruct d from the top nibbles of the scale u16 pairs
        // sc[0..8] treated as 4 x u16, and top 4 bits of each give the f16 scale
        let sc_u16: [u16; 4] = [
            u16::from_le_bytes([sc[0], sc[1]]),
            u16::from_le_bytes([sc[2], sc[3]]),
            u16::from_le_bytes([sc[4], sc[5]]),
            u16::from_le_bytes([sc[6], sc[7]]),
        ];
        let d_bits: u16 = (sc_u16[0] >> 12)
            | ((sc_u16[1] >> 8) & 0x00f0)
            | ((sc_u16[2] >> 4) & 0x0f00)
            | (sc_u16[3] & 0xf000);
        let d = half::f16::from_bits(d_bits).to_f32();

        // 8 sub-blocks of 32 elements
        for ib in 0..8usize {
            let dl1 = d * (2.0 * ((sc_u16[ib / 2] >> (6 * (ib % 2))) & 7) as f32 + 1.0);
            let dl2 = d * (2.0 * ((sc_u16[ib / 2] >> (6 * (ib % 2) + 3)) & 7) as f32 + 1.0);

            // Each ib uses 4 bytes of qs and 2 bytes of qh
            let qs_base = ib * 4;
            let qh_base = ib * 2;

            let idx: [u32; 4] = [
                qs[qs_base] as u32 | (((qh[qh_base] as u32) << 8) & 0x700),
                qs[qs_base + 1] as u32 | (((qh[qh_base] as u32) << 4) & 0x700),
                qs[qs_base + 2] as u32 | (((qh[qh_base + 1] as u32) << 8) & 0x700),
                qs[qs_base + 3] as u32 | (((qh[qh_base + 1] as u32) << 4) & 0x700),
            ];
            let delta: [f32; 4] = [
                if qh[qh_base] & 0x08 != 0 {
                    -IQ1M_DELTA
                } else {
                    IQ1M_DELTA
                },
                if qh[qh_base] & 0x80 != 0 {
                    -IQ1M_DELTA
                } else {
                    IQ1M_DELTA
                },
                if qh[qh_base + 1] & 0x08 != 0 {
                    -IQ1M_DELTA
                } else {
                    IQ1M_DELTA
                },
                if qh[qh_base + 1] & 0x80 != 0 {
                    -IQ1M_DELTA
                } else {
                    IQ1M_DELTA
                },
            ];

            for l in 0..2usize {
                let grid = IQ1S_GRID[idx[l] as usize];
                for j in 0..8usize {
                    let g = ((grid >> (j * 8)) & 0xff) as i8 as f32;
                    out[ib * 32 + l * 8 + j] = dl1 * (g + delta[l]);
                }
            }
            for l in 2..4usize {
                let grid = IQ1S_GRID[idx[l] as usize];
                for j in 0..8usize {
                    let g = ((grid >> (j * 8)) & 0xff) as i8 as f32;
                    out[ib * 32 + l * 8 + j] = dl2 * (g + delta[l]);
                }
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
