//! Key-Value Cache for Autoregressive Inference
//!
//! Provides efficient caching of key and value tensors during autoregressive
//! generation. Without caching, each new token would require recomputing
//! attention over the entire sequence. With KV cache, we only compute
//! attention for the new token against cached keys/values.

use burn::prelude::*;

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
