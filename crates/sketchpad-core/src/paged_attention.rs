//! Paged Attention for Efficient KV Cache Management
//!
//! Paged attention divides KV cache memory into fixed-size blocks (pages),
//! allowing efficient memory management for variable-length sequences.
//! This is the key technique used in vLLM and other production inference systems.
//!
//! # Benefits
//!
//! - Reduced memory fragmentation
//! - Efficient memory sharing between sequences (e.g., beam search)
//! - Better GPU memory utilization
//! - Support for dynamic sequence lengths without over-allocation
//!
//! # Architecture
//!
//! ```text
//! Logical KV Cache          Block Table           Physical Blocks
//! ┌─────────────┐          ┌─────────┐          ┌─────────────┐
//! │ Seq 0: pos  │          │ 0 -> 2  │          │ Block 0     │
//! │ 0,1,2,3,4,5 │  ───►    │ 1 -> 5  │  ───►    │ Block 1     │
//! └─────────────┘          └─────────┘          │ Block 2 ◄───┤
//!                                               │ ...         │
//! ┌─────────────┐          ┌─────────┐          │ Block 5 ◄───┤
//! │ Seq 1: pos  │          │ 0 -> 3  │          └─────────────┘
//! │ 0,1,2       │  ───►    └─────────┘
//! └─────────────┘
//! ```

use burn::prelude::*;
use std::collections::VecDeque;

/// Configuration for paged KV cache
#[derive(Debug, Clone)]
pub struct PagedKvCacheConfig {
    /// Number of tokens per block
    pub block_size: usize,
    /// Total number of blocks in the cache
    pub num_blocks: usize,
    /// Number of KV heads
    pub num_kv_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Number of layers
    pub num_layers: usize,
}

impl PagedKvCacheConfig {
    pub fn new(
        block_size: usize,
        num_blocks: usize,
        num_kv_heads: usize,
        head_dim: usize,
        num_layers: usize,
    ) -> Self {
        Self {
            block_size,
            num_blocks,
            num_kv_heads,
            head_dim,
            num_layers,
        }
    }

    /// Calculate config from model parameters and memory budget
    pub fn from_memory_budget(
        memory_bytes: usize,
        num_kv_heads: usize,
        head_dim: usize,
        num_layers: usize,
        block_size: usize,
    ) -> Self {
        // Each block stores: 2 (K+V) * num_layers * num_kv_heads * head_dim * block_size * 4 bytes (f32)
        let bytes_per_block = 2 * num_layers * num_kv_heads * head_dim * block_size * 4;
        let num_blocks = memory_bytes / bytes_per_block;

        Self {
            block_size,
            num_blocks,
            num_kv_heads,
            head_dim,
            num_layers,
        }
    }
}

/// Block table mapping logical positions to physical blocks for a sequence
#[derive(Debug, Clone)]
pub struct BlockTable {
    /// Physical block indices for this sequence
    block_indices: Vec<usize>,
    /// Current sequence length
    seq_len: usize,
    /// Block size
    block_size: usize,
}

impl BlockTable {
    pub fn new(block_size: usize) -> Self {
        Self {
            block_indices: Vec::new(),
            seq_len: 0,
            block_size,
        }
    }

    /// Number of blocks currently allocated
    pub fn num_blocks(&self) -> usize {
        self.block_indices.len()
    }

    /// Current sequence length
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Add a block to this sequence
    pub fn add_block(&mut self, block_idx: usize) {
        self.block_indices.push(block_idx);
    }

    /// Get physical block index for a logical position
    pub fn get_block(&self, pos: usize) -> Option<usize> {
        let block_num = pos / self.block_size;
        self.block_indices.get(block_num).copied()
    }

    /// Get slot within block for a logical position
    pub fn get_slot(&self, pos: usize) -> usize {
        pos % self.block_size
    }

    /// Check if we need a new block for the next token
    pub fn needs_new_block(&self) -> bool {
        self.seq_len > 0 && self.seq_len % self.block_size == 0
    }

    /// Increment sequence length
    pub fn increment_seq_len(&mut self, count: usize) {
        self.seq_len += count;
    }

    /// Get all block indices
    pub fn blocks(&self) -> &[usize] {
        &self.block_indices
    }
}

/// Manages physical KV cache blocks
pub struct PagedKvCache<B: Backend> {
    /// Key cache: [num_blocks, num_layers, num_kv_heads, block_size, head_dim]
    k_cache: Tensor<B, 5>,
    /// Value cache: [num_blocks, num_layers, num_kv_heads, block_size, head_dim]
    v_cache: Tensor<B, 5>,
    /// Free block indices
    free_blocks: VecDeque<usize>,
    /// Configuration
    config: PagedKvCacheConfig,
}

impl<B: Backend> PagedKvCache<B> {
    /// Create a new paged KV cache
    pub fn new(config: PagedKvCacheConfig, device: &B::Device) -> Self {
        let k_cache = Tensor::zeros(
            [
                config.num_blocks,
                config.num_layers,
                config.num_kv_heads,
                config.block_size,
                config.head_dim,
            ],
            device,
        );

        let v_cache = Tensor::zeros(
            [
                config.num_blocks,
                config.num_layers,
                config.num_kv_heads,
                config.block_size,
                config.head_dim,
            ],
            device,
        );

        let free_blocks: VecDeque<usize> = (0..config.num_blocks).collect();

        Self {
            k_cache,
            v_cache,
            free_blocks,
            config,
        }
    }

    /// Allocate a block, returns None if no blocks available
    pub fn allocate_block(&mut self) -> Option<usize> {
        self.free_blocks.pop_front()
    }

    /// Free a block
    pub fn free_block(&mut self, block_idx: usize) {
        self.free_blocks.push_back(block_idx);
    }

    /// Free all blocks in a block table
    pub fn free_sequence(&mut self, block_table: &BlockTable) {
        for &block_idx in block_table.blocks() {
            self.free_block(block_idx);
        }
    }

    /// Number of free blocks
    pub fn num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    /// Total number of blocks
    pub fn num_blocks(&self) -> usize {
        self.config.num_blocks
    }

    /// Block size (tokens per block)
    pub fn block_size(&self) -> usize {
        self.config.block_size
    }

    /// Store KV values for a layer at specific positions
    pub fn store_kv(
        &mut self,
        layer_idx: usize,
        block_table: &BlockTable,
        k: Tensor<B, 3>, // [num_kv_heads, seq_len, head_dim]
        v: Tensor<B, 3>, // [num_kv_heads, seq_len, head_dim]
        start_pos: usize,
    ) {
        let [_num_heads, seq_len, _head_dim] = k.dims();

        for i in 0..seq_len {
            let pos = start_pos + i;
            let block_idx = block_table.get_block(pos).expect("Block not allocated");
            let slot = block_table.get_slot(pos);

            // Extract single position KV
            let k_pos = k.clone().slice([
                0..self.config.num_kv_heads,
                i..i + 1,
                0..self.config.head_dim,
            ]);
            let v_pos = v.clone().slice([
                0..self.config.num_kv_heads,
                i..i + 1,
                0..self.config.head_dim,
            ]);

            // Reshape to [num_kv_heads, head_dim]
            let k_pos = k_pos.reshape([self.config.num_kv_heads, self.config.head_dim]);
            let v_pos = v_pos.reshape([self.config.num_kv_heads, self.config.head_dim]);

            // Store in cache - we need to use slice_assign or similar
            // For now, we'll rebuild the cache tensor (not optimal, but works)
            self.store_single_kv(layer_idx, block_idx, slot, k_pos, v_pos);
        }
    }

    fn store_single_kv(
        &mut self,
        layer_idx: usize,
        block_idx: usize,
        slot: usize,
        k: Tensor<B, 2>, // [num_kv_heads, head_dim]
        v: Tensor<B, 2>,
    ) {
        // Cache shape: [num_blocks, num_layers, num_kv_heads, block_size, head_dim]
        // Target slice: [block_idx, layer_idx, :, slot, :]
        // Values shape: [num_kv_heads, head_dim] -> reshape to [1, 1, num_kv_heads, 1, head_dim]
        let k_expanded = k
            .unsqueeze_dim::<3>(0) // [1, num_kv_heads, head_dim]
            .unsqueeze_dim::<4>(0) // [1, 1, num_kv_heads, head_dim]
            .unsqueeze_dim::<5>(3); // [1, 1, num_kv_heads, 1, head_dim]

        let v_expanded = v
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(0)
            .unsqueeze_dim::<5>(3);

        // Use slice_assign to write at the specific location
        self.k_cache = self.k_cache.clone().slice_assign(
            [
                block_idx..block_idx + 1,
                layer_idx..layer_idx + 1,
                0..self.config.num_kv_heads,
                slot..slot + 1,
                0..self.config.head_dim,
            ],
            k_expanded,
        );

        self.v_cache = self.v_cache.clone().slice_assign(
            [
                block_idx..block_idx + 1,
                layer_idx..layer_idx + 1,
                0..self.config.num_kv_heads,
                slot..slot + 1,
                0..self.config.head_dim,
            ],
            v_expanded,
        );
    }

    /// Gather KV values for attention computation
    pub fn gather_kv(
        &self,
        layer_idx: usize,
        block_table: &BlockTable,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        // [num_kv_heads, seq_len, head_dim]
        let seq_len = block_table.seq_len();
        let device = self.k_cache.device();

        if seq_len == 0 {
            return (
                Tensor::zeros([self.config.num_kv_heads, 0, self.config.head_dim], &device),
                Tensor::zeros([self.config.num_kv_heads, 0, self.config.head_dim], &device),
            );
        }

        // Gather from all blocks
        let mut k_parts = Vec::new();
        let mut v_parts = Vec::new();

        for (block_num, &block_idx) in block_table.blocks().iter().enumerate() {
            let start_pos = block_num * self.config.block_size;
            let end_pos = (start_pos + self.config.block_size).min(seq_len);
            let num_tokens = end_pos - start_pos;

            if num_tokens > 0 {
                // Extract block data for this layer
                let k_block = self.k_cache.clone().slice([
                    block_idx..block_idx + 1,
                    layer_idx..layer_idx + 1,
                    0..self.config.num_kv_heads,
                    0..num_tokens,
                    0..self.config.head_dim,
                ]);
                let v_block = self.v_cache.clone().slice([
                    block_idx..block_idx + 1,
                    layer_idx..layer_idx + 1,
                    0..self.config.num_kv_heads,
                    0..num_tokens,
                    0..self.config.head_dim,
                ]);

                // Reshape to [num_kv_heads, num_tokens, head_dim]
                let k_block =
                    k_block.reshape([self.config.num_kv_heads, num_tokens, self.config.head_dim]);
                let v_block =
                    v_block.reshape([self.config.num_kv_heads, num_tokens, self.config.head_dim]);

                k_parts.push(k_block);
                v_parts.push(v_block);
            }
        }

        if k_parts.is_empty() {
            return (
                Tensor::zeros([self.config.num_kv_heads, 0, self.config.head_dim], &device),
                Tensor::zeros([self.config.num_kv_heads, 0, self.config.head_dim], &device),
            );
        }

        // Concatenate along sequence dimension
        let k = Tensor::cat(k_parts, 1);
        let v = Tensor::cat(v_parts, 1);

        (k, v)
    }
}

/// Sequence state for paged attention inference
pub struct PagedSequence {
    /// Block table for this sequence
    pub block_table: BlockTable,
    /// Whether generation is complete
    pub finished: bool,
}

impl PagedSequence {
    pub fn new(block_size: usize) -> Self {
        Self {
            block_table: BlockTable::new(block_size),
            finished: false,
        }
    }

    /// Allocate blocks needed for new tokens
    pub fn allocate_for_tokens<B: Backend>(
        &mut self,
        num_tokens: usize,
        cache: &mut PagedKvCache<B>,
    ) -> Result<(), &'static str> {
        let new_len = self.block_table.seq_len() + num_tokens;
        let blocks_needed = new_len.div_ceil(cache.block_size());
        let current_blocks = self.block_table.num_blocks();

        for _ in current_blocks..blocks_needed {
            match cache.allocate_block() {
                Some(block_idx) => self.block_table.add_block(block_idx),
                None => return Err("Out of memory: no free blocks"),
            }
        }

        Ok(())
    }
}

/// Scheduler for managing multiple sequences with paged attention
pub struct PagedScheduler<B: Backend> {
    /// The paged KV cache
    pub cache: PagedKvCache<B>,
    /// Active sequences
    sequences: Vec<PagedSequence>,
}

impl<B: Backend> PagedScheduler<B> {
    pub fn new(config: PagedKvCacheConfig, device: &B::Device) -> Self {
        Self {
            cache: PagedKvCache::new(config, device),
            sequences: Vec::new(),
        }
    }

    /// Add a new sequence
    pub fn add_sequence(&mut self) -> Result<usize, &'static str> {
        let seq_id = self.sequences.len();
        self.sequences
            .push(PagedSequence::new(self.cache.block_size()));
        Ok(seq_id)
    }

    /// Get sequence by ID
    pub fn get_sequence(&self, seq_id: usize) -> Option<&PagedSequence> {
        self.sequences.get(seq_id)
    }

    /// Get mutable sequence by ID
    pub fn get_sequence_mut(&mut self, seq_id: usize) -> Option<&mut PagedSequence> {
        self.sequences.get_mut(seq_id)
    }

    /// Allocate memory for new tokens in a sequence
    pub fn allocate(&mut self, seq_id: usize, num_tokens: usize) -> Result<(), &'static str> {
        let cache = &mut self.cache;
        let seq = self
            .sequences
            .get_mut(seq_id)
            .ok_or("Invalid sequence ID")?;
        seq.allocate_for_tokens(num_tokens, cache)
    }

    /// Free a completed sequence
    pub fn free_sequence(&mut self, seq_id: usize) {
        if let Some(seq) = self.sequences.get(seq_id) {
            self.cache.free_sequence(&seq.block_table);
        }
    }

    /// Number of active sequences
    pub fn num_sequences(&self) -> usize {
        self.sequences.len()
    }

    /// Available capacity in tokens
    pub fn available_tokens(&self) -> usize {
        self.cache.num_free_blocks() * self.cache.block_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_block_table() {
        let mut table = BlockTable::new(4); // 4 tokens per block

        assert_eq!(table.seq_len(), 0);
        assert_eq!(table.num_blocks(), 0);
        assert!(!table.needs_new_block());

        // Add first block
        table.add_block(0);
        table.increment_seq_len(4);

        assert_eq!(table.seq_len(), 4);
        assert_eq!(table.num_blocks(), 1);
        assert!(table.needs_new_block()); // 4 % 4 == 0, need new block for next token

        // Check position mapping
        assert_eq!(table.get_block(0), Some(0));
        assert_eq!(table.get_block(3), Some(0));
        assert_eq!(table.get_slot(0), 0);
        assert_eq!(table.get_slot(3), 3);

        // Add second block
        table.add_block(5);
        table.increment_seq_len(2);

        assert_eq!(table.seq_len(), 6);
        assert_eq!(table.get_block(4), Some(5));
        assert_eq!(table.get_block(5), Some(5));
        assert_eq!(table.get_slot(4), 0);
        assert_eq!(table.get_slot(5), 1);
    }

    #[test]
    fn test_paged_kv_cache_allocation() {
        let device = Default::default();
        let config = PagedKvCacheConfig::new(4, 10, 8, 64, 2);
        let mut cache = PagedKvCache::<TestBackend>::new(config, &device);

        assert_eq!(cache.num_blocks(), 10);
        assert_eq!(cache.num_free_blocks(), 10);

        // Allocate some blocks
        let block0 = cache.allocate_block();
        let block1 = cache.allocate_block();

        assert!(block0.is_some());
        assert!(block1.is_some());
        assert_eq!(cache.num_free_blocks(), 8);

        // Free a block
        cache.free_block(block0.unwrap());
        assert_eq!(cache.num_free_blocks(), 9);
    }

    #[test]
    fn test_paged_scheduler() {
        let device = Default::default();
        let config = PagedKvCacheConfig::new(4, 10, 8, 64, 2);
        let mut scheduler = PagedScheduler::<TestBackend>::new(config, &device);

        // Add sequences
        let seq0 = scheduler.add_sequence().unwrap();
        let seq1 = scheduler.add_sequence().unwrap();

        assert_eq!(seq0, 0);
        assert_eq!(seq1, 1);
        assert_eq!(scheduler.num_sequences(), 2);

        // Allocate for tokens
        scheduler.allocate(seq0, 3).unwrap(); // Needs 1 block
        scheduler.allocate(seq1, 5).unwrap(); // Needs 2 blocks

        assert_eq!(scheduler.cache.num_free_blocks(), 7); // 10 - 3 = 7

        // Free sequence
        scheduler.free_sequence(seq0);
        assert_eq!(scheduler.cache.num_free_blocks(), 8); // Got 1 block back
    }

    #[test]
    fn test_config_from_memory_budget() {
        // 1GB budget
        let config = PagedKvCacheConfig::from_memory_budget(
            1024 * 1024 * 1024, // 1GB
            8,                  // num_kv_heads
            128,                // head_dim
            32,                 // num_layers
            16,                 // block_size
        );

        // Each block = 2 * 32 * 8 * 128 * 16 * 4 = 4,194,304 bytes = 4MB
        // 1GB / 4MB = 256 blocks
        assert_eq!(config.num_blocks, 256);
    }

    #[test]
    fn test_gather_kv_empty() {
        let device = Default::default();
        let config = PagedKvCacheConfig::new(4, 10, 8, 64, 2);
        let cache = PagedKvCache::<TestBackend>::new(config, &device);

        let block_table = BlockTable::new(4);
        let (k, v) = cache.gather_kv(0, &block_table);

        assert_eq!(k.dims(), [8, 0, 64]);
        assert_eq!(v.dims(), [8, 0, 64]);
    }
}
