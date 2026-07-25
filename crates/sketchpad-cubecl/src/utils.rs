//! Utility functions for CubeCL kernels

/// Check if tensor is contiguous in memory
///
/// TODO: Use for NHWC layout handling (Phase 3 optimization)
#[allow(dead_code)]
pub fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    if shape.is_empty() {
        return true;
    }

    let mut expected_stride = 1;
    for i in (0..shape.len()).rev() {
        if strides[i] != expected_stride {
            return false;
        }
        expected_stride *= shape[i];
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_contiguous() {
        // Contiguous [2, 3, 4] with strides [12, 4, 1]
        assert!(is_contiguous(&[2, 3, 4], &[12, 4, 1]));

        // Non-contiguous (transposed)
        assert!(!is_contiguous(&[3, 2, 4], &[4, 12, 1]));

        // Empty
        assert!(is_contiguous(&[], &[]));
    }
}
