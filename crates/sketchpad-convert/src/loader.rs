//! Load tensors from .safetensors files

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use burn::prelude::*;
use half::f16;
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LoadError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Safetensors error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),

    #[error("Tensor not found: {0}")]
    TensorNotFound(String),

    #[error("Unsupported dtype: {0:?}")]
    UnsupportedDtype(Dtype),

    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
}

/// A loaded safetensors file with memory-mapped data
pub struct SafeTensorFile {
    // Keep mmap alive for the lifetime of tensors
    _mmap: memmap2::Mmap,
    // Tensor metadata (name -> (dtype, shape, data_offset, data_len))
    tensors: HashMap<String, TensorInfo>,
    // Raw pointer to mmap data for tensor access
    data_ptr: *const u8,
    #[allow(dead_code)]
    data_len: usize,
}

struct TensorInfo {
    dtype: Dtype,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

// Safety: The mmap is read-only and lives as long as SafeTensorFile
unsafe impl Send for SafeTensorFile {}
unsafe impl Sync for SafeTensorFile {}

impl SafeTensorFile {
    /// Open a safetensors file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, LoadError> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        // Parse the safetensors header to get tensor metadata
        let st = SafeTensors::deserialize(&mmap)?;

        let mut tensors = HashMap::new();
        for (name, view) in st.tensors() {
            tensors.insert(
                name.to_string(),
                TensorInfo {
                    dtype: view.dtype(),
                    shape: view.shape().to_vec(),
                    start: view.data().as_ptr() as usize - mmap.as_ptr() as usize,
                    end: view.data().as_ptr() as usize - mmap.as_ptr() as usize + view.data().len(),
                },
            );
        }

        let data_ptr = mmap.as_ptr();
        let data_len = mmap.len();

        Ok(Self {
            _mmap: mmap,
            tensors,
            data_ptr,
            data_len,
        })
    }

    /// List all tensor names
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    /// Check if a tensor exists
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Get tensor shape
    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.tensors.get(name).map(|t| t.shape.as_slice())
    }

    /// Get tensor dtype
    pub fn dtype(&self, name: &str) -> Option<Dtype> {
        self.tensors.get(name).map(|t| t.dtype)
    }

    /// Load a tensor as f32, converting from fp16/bf16 if needed
    pub fn load_f32<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &B::Device,
    ) -> Result<Tensor<B, D>, LoadError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| LoadError::TensorNotFound(name.to_string()))?;

        if info.shape.len() != D {
            return Err(LoadError::ShapeMismatch {
                expected: vec![0; D], // placeholder
                actual: info.shape.clone(),
            });
        }

        // Safety: mmap is valid for lifetime of self
        let data = unsafe {
            std::slice::from_raw_parts(self.data_ptr.add(info.start), info.end - info.start)
        };

        // Convert to f32, handling potentially unaligned mmap data
        let floats: Vec<f32> = match info.dtype {
            Dtype::F32 => {
                // Read f32 values handling potential unaligned data
                data.chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect()
            }
            Dtype::F16 => {
                // Read u16 bits and convert to f32
                data.chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        f16::from_bits(bits).to_f32()
                    })
                    .collect()
            }
            Dtype::BF16 => {
                // Read u16 bits and convert to f32
                data.chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        half::bf16::from_bits(bits).to_f32()
                    })
                    .collect()
            }
            dtype => return Err(LoadError::UnsupportedDtype(dtype)),
        };

        let shape: [usize; D] = info.shape.clone().try_into().unwrap();
        let tensor_data = TensorData::new(floats, shape);

        Ok(Tensor::from_data(tensor_data, device))
    }

    /// Load a tensor with expected shape, converting to f32
    pub fn load_f32_checked<B: Backend, const D: usize>(
        &self,
        name: &str,
        expected_shape: [usize; D],
        device: &B::Device,
    ) -> Result<Tensor<B, D>, LoadError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| LoadError::TensorNotFound(name.to_string()))?;

        if info.shape.as_slice() != expected_shape.as_slice() {
            return Err(LoadError::ShapeMismatch {
                expected: expected_shape.to_vec(),
                actual: info.shape.clone(),
            });
        }

        self.load_f32::<B, D>(name, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_error_display() {
        let err = LoadError::TensorNotFound("missing".to_string());
        assert!(err.to_string().contains("missing"));
    }
}
