//! Key-Value Cache for Autoregressive Inference
//!
//! Provides efficient caching of key and value tensors during autoregressive
//! generation. Without caching, each new token would require recomputing
//! attention over the entire sequence. With KV cache, we only compute
//! attention for the new token against cached keys/values.

use std::marker::PhantomData;

use burn::prelude::*;
use half::f16;

/// Result of updating a cache layer with new K/V tensors
pub struct CacheUpdate<B: Backend> {
    /// Full key sequence including cached content: [batch, num_kv_heads, total_seq_len, head_dim]
    pub k: Tensor<B, 4>,
    /// Full value sequence including cached content: [batch, num_kv_heads, total_seq_len, head_dim]
    pub v: Tensor<B, 4>,
}

/// Trait for KV cache strategies used in attention layers
///
/// Both the simple concat-based cache and paged attention cache implement this,
/// allowing models to use either strategy without code changes.
pub trait AttentionCache<B: Backend> {
    /// Update cache with new K/V for a specific layer, return full K/V sequence
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Index of the transformer layer
    /// * `k` - New keys [batch, num_kv_heads, new_seq_len, head_dim]
    /// * `v` - New values [batch, num_kv_heads, new_seq_len, head_dim]
    fn update_layer(
        &mut self,
        layer_idx: usize,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
    ) -> CacheUpdate<B>;

    /// Current cached sequence length
    fn seq_len(&self) -> usize;

    /// Clear all cached state
    fn clear(&mut self);
}

impl<B: Backend> AttentionCache<B> for ModelKvCache<B> {
    fn update_layer(
        &mut self,
        layer_idx: usize,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
    ) -> CacheUpdate<B> {
        let (k, v) = self.layer(layer_idx).update(k, v);
        CacheUpdate { k, v }
    }

    fn seq_len(&self) -> usize {
        self.seq_len()
    }

    fn clear(&mut self) {
        self.clear();
    }
}

/// KV Cache for a single attention layer
///
/// Stores the key and value tensors from previous tokens, allowing
/// efficient incremental decoding during autoregressive generation.
#[derive(Debug, Clone)]
pub struct KvCache<B: Backend> {
    /// Cached keys: [batch, num_kv_heads, seq_len, head_dim]
    pub k: Option<Tensor<B, 4>>,
    /// Cached values: [batch, num_kv_heads, seq_len, head_dim]
    pub v: Option<Tensor<B, 4>>,
    /// Maximum sequence length this cache can hold
    pub max_seq_len: usize,
    /// Current position in the sequence
    pub current_pos: usize,
}

impl<B: Backend> KvCache<B> {
    /// Creates a new empty KV cache
    ///
    /// # Arguments
    ///
    /// * `max_seq_len` - Maximum sequence length to cache
    pub fn new(max_seq_len: usize) -> Self {
        Self {
            k: None,
            v: None,
            max_seq_len,
            current_pos: 0,
        }
    }

    /// Updates the cache with new key/value tensors
    ///
    /// # Arguments
    ///
    /// * `k` - New keys [batch, num_kv_heads, new_seq_len, head_dim]
    /// * `v` - New values [batch, num_kv_heads, new_seq_len, head_dim]
    ///
    /// # Returns
    ///
    /// Tuple of (full_keys, full_values) including cached content
    pub fn update(&mut self, k: Tensor<B, 4>, v: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [_batch, _heads, new_len, _dim] = k.dims();

        let (full_k, full_v) = match (&self.k, &self.v) {
            (Some(cached_k), Some(cached_v)) => {
                // Concatenate new tokens with cached ones
                let full_k = Tensor::cat(vec![cached_k.clone(), k], 2);
                let full_v = Tensor::cat(vec![cached_v.clone(), v], 2);
                (full_k, full_v)
            }
            _ => {
                // First tokens, no cache yet
                (k, v)
            }
        };

        // Update cache
        self.k = Some(full_k.clone());
        self.v = Some(full_v.clone());
        self.current_pos += new_len;

        (full_k, full_v)
    }

    /// Returns the current sequence length in the cache
    pub fn seq_len(&self) -> usize {
        self.current_pos
    }

    /// Clears the cache for a new generation
    pub fn clear(&mut self) {
        self.k = None;
        self.v = None;
        self.current_pos = 0;
    }

    /// Returns true if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.k.is_none()
    }
}

/// KV Cache for an entire model (all layers)
///
/// Manages KV caches for each transformer layer in the model.
#[derive(Debug)]
pub struct ModelKvCache<B: Backend> {
    /// Per-layer caches
    pub layers: Vec<KvCache<B>>,
    /// Maximum sequence length
    pub max_seq_len: usize,
}

impl<B: Backend> ModelKvCache<B> {
    /// Creates KV caches for all layers
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of transformer layers
    /// * `max_seq_len` - Maximum sequence length to cache
    pub fn new(num_layers: usize, max_seq_len: usize) -> Self {
        let layers = (0..num_layers).map(|_| KvCache::new(max_seq_len)).collect();

        Self {
            layers,
            max_seq_len,
        }
    }

    /// Gets mutable reference to a layer's cache
    pub fn layer(&mut self, idx: usize) -> &mut KvCache<B> {
        &mut self.layers[idx]
    }

    /// Returns the current sequence position (same for all layers)
    pub fn seq_len(&self) -> usize {
        self.layers.first().map(|c| c.seq_len()).unwrap_or(0)
    }

    /// Clears all layer caches
    pub fn clear(&mut self) {
        for cache in &mut self.layers {
            cache.clear();
        }
    }
}

/// Pre-allocated KV Cache for better memory efficiency
///
/// Unlike the dynamic `KvCache`, this version pre-allocates tensors
/// for the maximum sequence length, avoiding repeated allocations
/// during generation.
#[derive(Debug)]
pub struct StaticKvCache<B: Backend> {
    /// Pre-allocated keys: [batch, num_kv_heads, max_seq_len, head_dim]
    k: Tensor<B, 4>,
    /// Pre-allocated values: [batch, num_kv_heads, max_seq_len, head_dim]
    v: Tensor<B, 4>,
    /// Current position in the sequence
    current_pos: usize,
    /// Maximum sequence length
    max_seq_len: usize,
}

impl<B: Backend> StaticKvCache<B> {
    /// Creates a pre-allocated KV cache
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Batch size
    /// * `num_kv_heads` - Number of KV heads
    /// * `max_seq_len` - Maximum sequence length
    /// * `head_dim` - Dimension per head
    /// * `device` - Device to allocate on
    pub fn new(
        batch_size: usize,
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            k: Tensor::zeros([batch_size, num_kv_heads, max_seq_len, head_dim], device),
            v: Tensor::zeros([batch_size, num_kv_heads, max_seq_len, head_dim], device),
            current_pos: 0,
            max_seq_len,
        }
    }

    /// Updates the cache with new key/value tensors
    ///
    /// # Arguments
    ///
    /// * `k` - New keys [batch, num_kv_heads, new_seq_len, head_dim]
    /// * `v` - New values [batch, num_kv_heads, new_seq_len, head_dim]
    ///
    /// # Returns
    ///
    /// Slices of the full keys/values up to current position
    pub fn update(&mut self, k: Tensor<B, 4>, v: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, heads, new_len, head_dim] = k.dims();
        let start = self.current_pos;
        let end = start + new_len;

        assert!(end <= self.max_seq_len, "KV cache overflow");

        // Update the cache in place using slice assignment
        self.k = self
            .k
            .clone()
            .slice_assign([0..batch, 0..heads, start..end, 0..head_dim], k);
        self.v = self
            .v
            .clone()
            .slice_assign([0..batch, 0..heads, start..end, 0..head_dim], v);

        self.current_pos = end;

        // Return the valid portion of the cache
        let full_k = self
            .k
            .clone()
            .slice([0..batch, 0..heads, 0..end, 0..head_dim]);
        let full_v = self
            .v
            .clone()
            .slice([0..batch, 0..heads, 0..end, 0..head_dim]);

        (full_k, full_v)
    }

    /// Returns the current sequence length
    pub fn seq_len(&self) -> usize {
        self.current_pos
    }

    /// Resets the cache position without reallocating
    pub fn reset(&mut self) {
        self.current_pos = 0;
    }
}

/// Quantization method for compressed KV cache storage
pub enum KvQuantMethod {
    /// Full precision — no compression, equivalent to ModelKvCache
    Full,
    /// PolarQuant: magnitude (f16) + 4-bit direction quantization (~4x vs f16)
    PolarQuant,
}

impl KvQuantMethod {
    /// Ratio of compressed bytes to f16 bytes (used for VRAM estimation)
    pub fn compression_ratio(&self, head_dim: usize) -> f64 {
        match self {
            Self::Full => 1.0,
            Self::PolarQuant => (2.0 + head_dim as f64 / 2.0) / (head_dim as f64 * 2.0),
        }
    }
}

/// Per-layer storage for `CompressedKvCache`
struct CompressedLayerCache {
    k_bytes: Vec<u8>,
    v_bytes: Vec<u8>,
    batch: usize,
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
}

impl CompressedLayerCache {
    fn new() -> Self {
        Self {
            k_bytes: Vec::new(),
            v_bytes: Vec::new(),
            batch: 0,
            num_heads: 0,
            head_dim: 0,
            seq_len: 0,
        }
    }

    fn clear(&mut self) {
        self.k_bytes.clear();
        self.v_bytes.clear();
        self.batch = 0;
        self.num_heads = 0;
        self.seq_len = 0;
    }
}

/// KV cache that stores compressed key/value tensors on the CPU
///
/// Uses PolarQuant compression (~4x vs f16) or full f16 precision.
/// Keys and values are compressed after each step and decompressed when
/// returning the full sequence for attention computation.
pub struct CompressedKvCache<B: Backend> {
    layers: Vec<CompressedLayerCache>,
    method: KvQuantMethod,
    _phantom: PhantomData<B>,
}

impl<B: Backend> CompressedKvCache<B> {
    /// Creates a new compressed KV cache
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of transformer layers
    /// * `num_heads` - Number of KV heads (informational, derived from tensors at runtime)
    /// * `head_dim` - Dimension per head (informational, derived from tensors at runtime)
    /// * `method` - Quantization method to use
    pub fn new(
        num_layers: usize,
        _num_heads: usize,
        _head_dim: usize,
        method: KvQuantMethod,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|_| CompressedLayerCache::new())
            .collect();
        Self {
            layers,
            method,
            _phantom: PhantomData,
        }
    }
}

/// Compress a flat f32 array of vectors (row-major, each row = head_dim floats)
/// using PolarQuant encoding: f16 magnitude + 4-bit packed direction.
fn polar_quant_compress(data: &[f32], head_dim: usize) -> Vec<u8> {
    let num_vecs = data.len() / head_dim;
    // bytes per vector: 2 (f16 magnitude) + ceil(head_dim / 2) packed nibbles
    let nibble_bytes = head_dim.div_ceil(2);
    let bytes_per_vec = 2 + nibble_bytes;
    let mut out = vec![0u8; num_vecs * bytes_per_vec];

    for i in 0..num_vecs {
        let row = &data[i * head_dim..(i + 1) * head_dim];
        let base = i * bytes_per_vec;

        // Compute magnitude
        let mag: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Store magnitude as f16 little-endian
        let mag_f16 = f16::from_f32(mag);
        let mag_bits = mag_f16.to_bits().to_le_bytes();
        out[base] = mag_bits[0];
        out[base + 1] = mag_bits[1];

        if mag == 0.0 {
            // Direction bytes already zero from vec initialisation
            continue;
        }

        // Quantize direction components and pack as nibbles
        let inv_mag = 1.0 / mag;
        let nibble_base = base + 2;
        for j in (0..head_dim).step_by(2) {
            let d0 = row[j] * inv_mag;
            let q0 = (d0 * 7.0).round().clamp(-7.0, 7.0) as i8;

            let q1 = if j + 1 < head_dim {
                let d1 = row[j + 1] * inv_mag;
                (d1 * 7.0).round().clamp(-7.0, 7.0) as i8
            } else {
                0i8
            };

            // Pack: high nibble = q0, low nibble = q1 (4-bit two's complement)
            let packed = ((q0 as u8 & 0xF) << 4) | (q1 as u8 & 0xF);
            out[nibble_base + j / 2] = packed;
        }
    }

    out
}

/// Decompress bytes produced by `polar_quant_compress` back to f32 vectors.
fn polar_quant_decompress(bytes: &[u8], num_vecs: usize, head_dim: usize) -> Vec<f32> {
    let nibble_bytes = head_dim.div_ceil(2);
    let bytes_per_vec = 2 + nibble_bytes;
    let mut out = vec![0.0f32; num_vecs * head_dim];

    for i in 0..num_vecs {
        let base = i * bytes_per_vec;
        let row_base = i * head_dim;

        // Read f16 magnitude
        let mag_bits = u16::from_le_bytes([bytes[base], bytes[base + 1]]);
        let mag = f16::from_bits(mag_bits).to_f32();

        if mag == 0.0 {
            continue;
        }

        let scale = mag / 7.0;
        let nibble_base = base + 2;

        for j in (0..head_dim).step_by(2) {
            let byte = bytes[nibble_base + j / 2];

            // Unpack high nibble (q0)
            let raw0 = (byte >> 4) & 0xF;
            let n0 = if raw0 > 7 {
                raw0 as i32 - 16
            } else {
                raw0 as i32
            };
            out[row_base + j] = n0 as f32 * scale;

            // Unpack low nibble (q1)
            if j + 1 < head_dim {
                let raw1 = byte & 0xF;
                let n1 = if raw1 > 7 {
                    raw1 as i32 - 16
                } else {
                    raw1 as i32
                };
                out[row_base + j + 1] = n1 as f32 * scale;
            }
        }
    }

    out
}

/// Compress using full f16 (no direction quantization, just f32 → f16).
fn full_compress(data: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; data.len() * 2];
    for (i, &v) in data.iter().enumerate() {
        let bits = f16::from_f32(v).to_bits().to_le_bytes();
        out[i * 2] = bits[0];
        out[i * 2 + 1] = bits[1];
    }
    out
}

/// Decompress full f16 bytes back to f32.
fn full_decompress(bytes: &[u8], num_elems: usize) -> Vec<f32> {
    (0..num_elems)
        .map(|i| {
            let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
            f16::from_bits(bits).to_f32()
        })
        .collect()
}

impl<B: Backend> AttentionCache<B> for CompressedKvCache<B> {
    fn update_layer(
        &mut self,
        layer_idx: usize,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
    ) -> CacheUpdate<B> {
        let [batch, num_heads, new_seq, head_dim] = k.dims();
        let device = k.device();

        // Pull new K/V to CPU as f32
        let k_new: Vec<f32> = k.to_data().to_vec().expect("k to_vec");
        let v_new: Vec<f32> = v.to_data().to_vec().expect("v to_vec");

        let layer = &mut self.layers[layer_idx];

        // First update initialises metadata
        if layer.seq_len == 0 {
            layer.batch = batch;
            layer.num_heads = num_heads;
            layer.head_dim = head_dim;
        }

        // Compress and append
        match self.method {
            KvQuantMethod::Full => {
                layer.k_bytes.extend(full_compress(&k_new));
                layer.v_bytes.extend(full_compress(&v_new));
            }
            KvQuantMethod::PolarQuant => {
                // data is flat [batch * num_heads * new_seq * head_dim]; treat each
                // head_dim-sized chunk as one vector
                layer.k_bytes.extend(polar_quant_compress(&k_new, head_dim));
                layer.v_bytes.extend(polar_quant_compress(&v_new, head_dim));
            }
        }

        layer.seq_len += new_seq;

        // Decompress full accumulated sequence
        let total_seq = layer.seq_len;
        let num_vecs = batch * num_heads * total_seq;
        let total_elems = num_vecs * head_dim;

        let k_full_f32: Vec<f32> = match self.method {
            KvQuantMethod::Full => full_decompress(&layer.k_bytes, total_elems),
            KvQuantMethod::PolarQuant => polar_quant_decompress(&layer.k_bytes, num_vecs, head_dim),
        };
        let v_full_f32: Vec<f32> = match self.method {
            KvQuantMethod::Full => full_decompress(&layer.v_bytes, total_elems),
            KvQuantMethod::PolarQuant => polar_quant_decompress(&layer.v_bytes, num_vecs, head_dim),
        };

        let k_full = Tensor::<B, 1>::from_data(
            TensorData::new(k_full_f32, [batch * num_heads * total_seq * head_dim]),
            &device,
        )
        .reshape([batch, num_heads, total_seq, head_dim]);

        let v_full = Tensor::<B, 1>::from_data(
            TensorData::new(v_full_f32, [batch * num_heads * total_seq * head_dim]),
            &device,
        )
        .reshape([batch, num_heads, total_seq, head_dim]);

        CacheUpdate {
            k: k_full,
            v: v_full,
        }
    }

    fn seq_len(&self) -> usize {
        self.layers.first().map(|l| l.seq_len).unwrap_or(0)
    }

    fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_kv_cache_update() {
        let device = Default::default();
        let mut cache = KvCache::<TestBackend>::new(128);

        // First token
        let k1 = Tensor::ones([1, 4, 1, 32], &device);
        let v1 = Tensor::ones([1, 4, 1, 32], &device);
        let (full_k, full_v) = cache.update(k1, v1);

        assert_eq!(full_k.dims(), [1, 4, 1, 32]);
        assert_eq!(full_v.dims(), [1, 4, 1, 32]);
        assert_eq!(cache.seq_len(), 1);

        // Second token
        let k2 = Tensor::ones([1, 4, 1, 32], &device) * 2.0;
        let v2 = Tensor::ones([1, 4, 1, 32], &device) * 2.0;
        let (full_k, full_v) = cache.update(k2, v2);

        assert_eq!(full_k.dims(), [1, 4, 2, 32]);
        assert_eq!(full_v.dims(), [1, 4, 2, 32]);
        assert_eq!(cache.seq_len(), 2);
    }

    #[test]
    fn test_kv_cache_clear() {
        let device = Default::default();
        let mut cache = KvCache::<TestBackend>::new(128);

        let k = Tensor::ones([1, 4, 5, 32], &device);
        let v = Tensor::ones([1, 4, 5, 32], &device);
        cache.update(k, v);

        assert_eq!(cache.seq_len(), 5);
        assert!(!cache.is_empty());

        cache.clear();

        assert_eq!(cache.seq_len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_model_kv_cache() {
        let device = Default::default();
        let mut cache = ModelKvCache::<TestBackend>::new(32, 2048);

        assert_eq!(cache.layers.len(), 32);

        let k = Tensor::ones([1, 4, 10, 64], &device);
        let v = Tensor::ones([1, 4, 10, 64], &device);
        cache.layer(0).update(k, v);

        assert_eq!(cache.layer(0).seq_len(), 10);
        assert_eq!(cache.layer(1).seq_len(), 0);
    }

    #[test]
    fn test_static_kv_cache() {
        let device = Default::default();
        let mut cache = StaticKvCache::<TestBackend>::new(1, 4, 128, 32, &device);

        // First update
        let k1 = Tensor::ones([1, 4, 5, 32], &device);
        let v1 = Tensor::ones([1, 4, 5, 32], &device);
        let (full_k, full_v) = cache.update(k1, v1);

        assert_eq!(full_k.dims(), [1, 4, 5, 32]);
        assert_eq!(full_v.dims(), [1, 4, 5, 32]);
        assert_eq!(cache.seq_len(), 5);

        // Second update
        let k2 = Tensor::ones([1, 4, 3, 32], &device);
        let v2 = Tensor::ones([1, 4, 3, 32], &device);
        let (full_k, full_v) = cache.update(k2, v2);

        assert_eq!(full_k.dims(), [1, 4, 8, 32]);
        assert_eq!(full_v.dims(), [1, 4, 8, 32]);
        assert_eq!(cache.seq_len(), 8);
    }

    #[test]
    fn test_static_kv_cache_reset() {
        let device = Default::default();
        let mut cache = StaticKvCache::<TestBackend>::new(1, 4, 128, 32, &device);

        let k = Tensor::ones([1, 4, 10, 32], &device);
        let v = Tensor::ones([1, 4, 10, 32], &device);
        cache.update(k, v);

        assert_eq!(cache.seq_len(), 10);

        cache.reset();

        assert_eq!(cache.seq_len(), 0);
    }
}
