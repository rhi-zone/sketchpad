//! Memory optimization utilities for burn-models
//!
//! Provides strategies for running inference with limited VRAM:
//! - VAE tiling for encoding/decoding very large images
//! - Sequential CPU offloading (load components on-demand)

use burn::prelude::*;

/// Configuration for memory-optimized inference
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Enable VAE tiling for large images
    pub enable_vae_tiling: bool,
    /// Tile size for VAE encoding (in pixels)
    pub vae_tile_size: usize,
    /// Tile overlap (in pixels) for seamless blending
    pub vae_tile_overlap: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enable_vae_tiling: false,
            vae_tile_size: 512,
            vae_tile_overlap: 64,
        }
    }
}

impl MemoryConfig {
    /// High-memory config (no optimizations, fastest)
    pub fn high_vram() -> Self {
        Self {
            enable_vae_tiling: false,
            vae_tile_size: 512,
            vae_tile_overlap: 64,
        }
    }

    /// Low-memory config with VAE tiling
    pub fn low_vram() -> Self {
        Self {
            enable_vae_tiling: true,
            vae_tile_size: 512,
            vae_tile_overlap: 64,
        }
    }

    /// Very low memory config (smaller tiles)
    pub fn very_low_vram() -> Self {
        Self {
            enable_vae_tiling: true,
            vae_tile_size: 256,
            vae_tile_overlap: 32,
        }
    }
}

/// Tiled VAE operations for memory-efficient encoding/decoding of large images
pub struct TiledVae {
    /// Size of each tile in pixels
    tile_size: usize,
    /// Overlap between adjacent tiles for smooth blending
    overlap: usize,
}

impl TiledVae {
    /// Creates a new tiled VAE processor
    ///
    /// # Arguments
    ///
    /// * `tile_size` - Size of each tile in pixels
    /// * `overlap` - Overlap between adjacent tiles for blending
    pub fn new(tile_size: usize, overlap: usize) -> Self {
        Self { tile_size, overlap }
    }

    /// Calculates tile positions for a given dimension
    fn tile_positions(&self, size: usize) -> Vec<(usize, usize)> {
        let step = self.tile_size - self.overlap;
        let mut positions = Vec::new();
        let mut start = 0;

        while start < size {
            let end = (start + self.tile_size).min(size);
            positions.push((start, end));
            if end >= size {
                break;
            }
            start += step;
        }

        positions
    }

    /// Process an image tensor with tiling
    ///
    /// `process_fn` is called for each tile and should return the processed tile.
    /// Tiles are blended together in the overlap regions.
    pub fn process_tiled<B: Backend, F>(
        &self,
        input: Tensor<B, 4>,
        process_fn: F,
        output_scale: usize,
    ) -> Tensor<B, 4>
    where
        F: Fn(Tensor<B, 4>) -> Tensor<B, 4>,
    {
        let [batch, channels, height, width] = input.dims();
        let device = input.device();

        // Calculate output dimensions
        let out_height = height / output_scale;
        let out_width = width / output_scale;
        let out_tile_size = self.tile_size / output_scale;
        let out_overlap = self.overlap / output_scale;

        // Initialize output and blend weights
        let mut output_data = vec![0.0f32; batch * channels * out_height * out_width];
        let mut weight_data = vec![0.0f32; out_height * out_width];

        // Get tile positions
        let h_positions = self.tile_positions(height);
        let w_positions = self.tile_positions(width);

        // Process each tile
        for (h_start, h_end) in &h_positions {
            for (w_start, w_end) in &w_positions {
                // Extract tile
                let tile = input.clone().slice([
                    0..batch,
                    0..channels,
                    *h_start..*h_end,
                    *w_start..*w_end,
                ]);

                // Pad tile if needed
                let tile_h = h_end - h_start;
                let tile_w = w_end - w_start;
                let tile = if tile_h < self.tile_size || tile_w < self.tile_size {
                    Self::pad_tile(tile, self.tile_size, self.tile_size, &device)
                } else {
                    tile
                };

                // Process tile
                let processed = process_fn(tile);

                // Get processed tile data
                let tile_data = processed.into_data();
                let tile_values: Vec<f32> = tile_data.to_vec().unwrap();

                // Calculate output positions
                let out_h_start = h_start / output_scale;
                let out_h_end = (h_end / output_scale).min(out_height);
                let out_w_start = w_start / output_scale;
                let out_w_end = (w_end / output_scale).min(out_width);

                // Calculate blend weights (linear ramp at edges)
                for oh in out_h_start..out_h_end {
                    for ow in out_w_start..out_w_end {
                        let th = oh - out_h_start;
                        let tw = ow - out_w_start;

                        // Calculate edge distances for blending
                        let h_weight = Self::blend_weight(th, out_h_end - out_h_start, out_overlap);
                        let w_weight = Self::blend_weight(tw, out_w_end - out_w_start, out_overlap);
                        let weight = h_weight * w_weight;

                        // Add weighted tile contribution
                        for b in 0..batch {
                            for c in 0..channels {
                                let out_idx = b * channels * out_height * out_width
                                    + c * out_height * out_width
                                    + oh * out_width
                                    + ow;
                                let tile_idx = b * channels * out_tile_size * out_tile_size
                                    + c * out_tile_size * out_tile_size
                                    + th * out_tile_size
                                    + tw;

                                if tile_idx < tile_values.len() {
                                    output_data[out_idx] += tile_values[tile_idx] * weight;
                                }
                            }
                        }

                        let weight_idx = oh * out_width + ow;
                        weight_data[weight_idx] += weight;
                    }
                }
            }
        }

        // Normalize by blend weights
        for oh in 0..out_height {
            for ow in 0..out_width {
                let weight_idx = oh * out_width + ow;
                if weight_data[weight_idx] > 0.0 {
                    for b in 0..batch {
                        for c in 0..channels {
                            let idx = b * channels * out_height * out_width
                                + c * out_height * out_width
                                + oh * out_width
                                + ow;
                            output_data[idx] /= weight_data[weight_idx];
                        }
                    }
                }
            }
        }

        Tensor::from_data(
            TensorData::new(output_data, [batch, channels, out_height, out_width]),
            &device,
        )
    }

    /// Computes blend weight for smooth tile transitions
    fn blend_weight(pos: usize, size: usize, overlap: usize) -> f32 {
        if overlap == 0 || size <= overlap * 2 {
            return 1.0;
        }

        let overlap_f = overlap as f32;
        let pos_f = pos as f32;
        let size_f = size as f32;

        if pos_f < overlap_f {
            // Ramp up at start
            pos_f / overlap_f
        } else if pos_f >= size_f - overlap_f {
            // Ramp down at end
            (size_f - pos_f) / overlap_f
        } else {
            1.0
        }
    }

    /// Pads a tile to the target dimensions with zeros
    fn pad_tile<B: Backend>(
        tile: Tensor<B, 4>,
        target_h: usize,
        target_w: usize,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let [batch, channels, h, w] = tile.dims();

        if h >= target_h && w >= target_w {
            return tile;
        }

        // Create zero-padded output
        let mut output_data = vec![0.0f32; batch * channels * target_h * target_w];
        let tile_data = tile.into_data();
        let tile_values: Vec<f32> = tile_data.to_vec().unwrap();

        // Copy tile data into output
        for b in 0..batch {
            for c in 0..channels {
                for th in 0..h {
                    for tw in 0..w {
                        let src_idx = b * channels * h * w + c * h * w + th * w + tw;
                        let dst_idx = b * channels * target_h * target_w
                            + c * target_h * target_w
                            + th * target_w
                            + tw;
                        output_data[dst_idx] = tile_values[src_idx];
                    }
                }
            }
        }

        Tensor::from_data(
            TensorData::new(output_data, [batch, channels, target_h, target_w]),
            device,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_positions() {
        let tiler = TiledVae::new(512, 64);

        // 1024 should give 3 tiles with overlap
        let positions = tiler.tile_positions(1024);
        assert!(!positions.is_empty());
        assert_eq!(positions[0].0, 0);
        assert_eq!(positions.last().unwrap().1, 1024);
    }

    #[test]
    fn test_blend_weight() {
        let weight = TiledVae::blend_weight(0, 100, 10);
        assert!(weight < 1.0);

        let weight = TiledVae::blend_weight(50, 100, 10);
        assert!((weight - 1.0).abs() < 0.001);

        let weight = TiledVae::blend_weight(99, 100, 10);
        assert!(weight < 1.0);
    }

    #[test]
    fn test_memory_config() {
        let config = MemoryConfig::low_vram();
        assert!(config.enable_vae_tiling);
    }
}
