//! Model serialization for save/load functionality
//!
//! This module provides utilities to save converted models to disk
//! and load them back later, avoiding the need to re-convert from
//! safetensors on each run.
//!
//! Burn's serialization works through its `Record` trait and recorders.
//! Models that derive `Module` automatically get serialization support.
//!
//! # Example
//!
//! ```ignore
//! use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
//! use burn::module::Module;
//!
//! // Save a model
//! let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
//! model.save_file("model.bin", &recorder)?;
//!
//! // Load it back
//! let model = model.load_file("model.bin", &recorder, &device)?;
//! ```
//!
//! This module re-exports common recorder types for convenience.

pub use burn::record::{
    BinBytesRecorder, BinFileRecorder, FullPrecisionSettings, HalfPrecisionSettings, Recorder,
    RecorderError,
};

/// Error type for serialization operations
#[derive(Debug, thiserror::Error)]
pub enum SerializeError {
    /// IO error during file operations
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Error from the Burn recorder
    #[error("Record error: {0}")]
    Record(#[from] RecorderError),
}

/// Create a recorder for full precision (f32) binary files
pub fn full_precision_recorder() -> BinFileRecorder<FullPrecisionSettings> {
    BinFileRecorder::new()
}

/// Create a recorder for half precision (f16) binary files
///
/// Reduces file size by approximately 50% compared to full precision.
pub fn half_precision_recorder() -> BinFileRecorder<HalfPrecisionSettings> {
    BinFileRecorder::new()
}

/// Creates a recorder for full precision binary bytes (in memory)
///
/// Useful for serializing models to memory rather than disk.
pub fn bytes_recorder() -> BinBytesRecorder<FullPrecisionSettings> {
    BinBytesRecorder::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_error_display() {
        let err = SerializeError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found",
        ));
        assert!(err.to_string().contains("IO error"));
    }

    #[test]
    fn test_recorder_creation() {
        let _fp = full_precision_recorder();
        let _hp = half_precision_recorder();
        let _bytes = bytes_recorder();
    }
}
