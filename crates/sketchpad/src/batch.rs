//! Batch processing for generating multiple images
//!
//! Enables efficient generation of multiple images with different prompts
//! or seeds in a single run.

use burn::prelude::*;

/// Configuration for batch processing
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Number of images to generate per batch
    pub batch_size: usize,
    /// Maximum batch size (for memory management)
    pub max_batch_size: usize,
    /// Whether to use dynamic batching
    pub dynamic_batching: bool,
    /// Seed handling strategy
    pub seed_strategy: SeedStrategy,
}

/// How to handle seeds when batching
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeedStrategy {
    /// Same seed for all images in batch
    Same,
    /// Increment seed for each image
    Increment,
    /// Random seed for each image
    Random,
    /// Use provided list of seeds
    Explicit,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            max_batch_size: 4,
            dynamic_batching: false,
            seed_strategy: SeedStrategy::Increment,
        }
    }
}

impl BatchConfig {
    /// Create config for generating a single image
    pub fn single() -> Self {
        Self::default()
    }

    /// Create config for batch of N images
    pub fn batch(n: usize) -> Self {
        Self {
            batch_size: n,
            max_batch_size: n.max(4),
            ..Default::default()
        }
    }

    /// Enable dynamic batching
    pub fn with_dynamic_batching(mut self) -> Self {
        self.dynamic_batching = true;
        self
    }

    /// Set seed strategy
    pub fn with_seed_strategy(mut self, strategy: SeedStrategy) -> Self {
        self.seed_strategy = strategy;
        self
    }
}

/// A batch generation request
#[derive(Debug, Clone)]
pub struct BatchRequest {
    /// Prompts for each image (or single prompt for all)
    pub prompts: Vec<String>,
    /// Negative prompts
    pub negative_prompts: Vec<String>,
    /// Seeds (if using explicit strategy)
    pub seeds: Vec<u64>,
    /// Additional per-image parameters
    pub params: Vec<BatchParams>,
}

/// Per-image parameters in a batch
#[derive(Debug, Clone, Default)]
pub struct BatchParams {
    /// Guidance scale override
    pub guidance_scale: Option<f64>,
    /// Steps override
    pub steps: Option<usize>,
    /// Image-specific LoRA scale
    pub lora_scale: Option<f32>,
}

impl BatchRequest {
    /// Create a batch request with a single prompt
    pub fn single(prompt: impl Into<String>) -> Self {
        Self {
            prompts: vec![prompt.into()],
            negative_prompts: vec![String::new()],
            seeds: vec![],
            params: vec![BatchParams::default()],
        }
    }

    /// Create a batch request with multiple prompts
    pub fn with_prompts(prompts: Vec<String>) -> Self {
        let n = prompts.len();
        Self {
            prompts,
            negative_prompts: vec![String::new(); n],
            seeds: vec![],
            params: vec![BatchParams::default(); n],
        }
    }

    /// Add negative prompts
    pub fn with_negative_prompts(mut self, negatives: Vec<String>) -> Self {
        self.negative_prompts = negatives;
        self
    }

    /// Add explicit seeds
    pub fn with_seeds(mut self, seeds: Vec<u64>) -> Self {
        self.seeds = seeds;
        self
    }

    /// Get the number of images in this batch
    pub fn len(&self) -> usize {
        self.prompts.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.prompts.is_empty()
    }

    /// Expand prompts to match batch size (repeating if needed)
    pub fn expand_prompts(&self, batch_size: usize) -> Vec<String> {
        if self.prompts.len() >= batch_size {
            self.prompts[..batch_size].to_vec()
        } else {
            let mut expanded = self.prompts.clone();
            while expanded.len() < batch_size {
                expanded.push(self.prompts[expanded.len() % self.prompts.len()].clone());
            }
            expanded
        }
    }

    /// Generate seeds based on strategy
    pub fn generate_seeds(&self, config: &BatchConfig, base_seed: u64) -> Vec<u64> {
        let n = self.prompts.len();
        match config.seed_strategy {
            SeedStrategy::Same => vec![base_seed; n],
            SeedStrategy::Increment => (0..n).map(|i| base_seed.wrapping_add(i as u64)).collect(),
            SeedStrategy::Random => {
                use std::collections::hash_map::RandomState;
                use std::hash::{BuildHasher, Hasher};
                let state = RandomState::new();
                (0..n)
                    .map(|i| {
                        let mut hasher = state.build_hasher();
                        hasher.write_u64(base_seed.wrapping_add(i as u64));
                        hasher.finish()
                    })
                    .collect()
            }
            SeedStrategy::Explicit => {
                if self.seeds.len() >= n {
                    self.seeds[..n].to_vec()
                } else {
                    // Pad with incremented seeds
                    let mut seeds = self.seeds.clone();
                    let last = *seeds.last().unwrap_or(&base_seed);
                    while seeds.len() < n {
                        seeds.push(last.wrapping_add(seeds.len() as u64));
                    }
                    seeds
                }
            }
        }
    }
}

/// Result of batch generation
#[derive(Debug)]
pub struct BatchResult<B: Backend> {
    /// Generated latents for each image
    pub latents: Vec<Tensor<B, 4>>,
    /// Seeds used for each image
    pub seeds: Vec<u64>,
    /// Generation time per image (if tracked)
    pub times_ms: Vec<u64>,
    /// Any errors that occurred
    pub errors: Vec<Option<String>>,
}

impl<B: Backend> BatchResult<B> {
    /// Create a new empty result
    pub fn new() -> Self {
        Self {
            latents: vec![],
            seeds: vec![],
            times_ms: vec![],
            errors: vec![],
        }
    }

    /// Add a successful result
    pub fn add_success(&mut self, latent: Tensor<B, 4>, seed: u64, time_ms: u64) {
        self.latents.push(latent);
        self.seeds.push(seed);
        self.times_ms.push(time_ms);
        self.errors.push(None);
    }

    /// Add a failed result
    pub fn add_error(&mut self, seed: u64, error: String) {
        self.seeds.push(seed);
        self.times_ms.push(0);
        self.errors.push(Some(error));
    }

    /// Get the number of successful generations
    pub fn num_success(&self) -> usize {
        self.latents.len()
    }

    /// Get the number of failed generations
    pub fn num_failed(&self) -> usize {
        self.errors.iter().filter(|e| e.is_some()).count()
    }

    /// Check if all generations succeeded
    pub fn all_success(&self) -> bool {
        self.errors.iter().all(|e| e.is_none())
    }

    /// Get total generation time
    pub fn total_time_ms(&self) -> u64 {
        self.times_ms.iter().sum()
    }

    /// Get average time per image
    pub fn avg_time_ms(&self) -> u64 {
        if self.times_ms.is_empty() {
            0
        } else {
            self.total_time_ms() / self.times_ms.len() as u64
        }
    }
}

impl<B: Backend> Default for BatchResult<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// Split a batch into smaller sub-batches
pub fn split_batch(total: usize, max_batch_size: usize) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0;

    while start < total {
        let end = (start + max_batch_size).min(total);
        ranges.push(start..end);
        start = end;
    }

    ranges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.batch_size, 1);
    }

    #[test]
    fn test_batch_request_single() {
        let req = BatchRequest::single("a cat");
        assert_eq!(req.len(), 1);
        assert_eq!(req.prompts[0], "a cat");
    }

    #[test]
    fn test_seed_generation_increment() {
        let req = BatchRequest::with_prompts(vec!["a".into(), "b".into(), "c".into()]);
        let config = BatchConfig::default();
        let seeds = req.generate_seeds(&config, 42);
        assert_eq!(seeds, vec![42, 43, 44]);
    }

    #[test]
    fn test_seed_generation_same() {
        let req = BatchRequest::with_prompts(vec!["a".into(), "b".into()]);
        let config = BatchConfig::default().with_seed_strategy(SeedStrategy::Same);
        let seeds = req.generate_seeds(&config, 42);
        assert_eq!(seeds, vec![42, 42]);
    }

    #[test]
    fn test_split_batch() {
        let ranges = split_batch(10, 3);
        assert_eq!(ranges.len(), 4);
        assert_eq!(ranges[0], 0..3);
        assert_eq!(ranges[3], 9..10);
    }

    #[test]
    fn test_expand_prompts() {
        let req = BatchRequest::with_prompts(vec!["a".into(), "b".into()]);
        let expanded = req.expand_prompts(5);
        assert_eq!(expanded.len(), 5);
        assert_eq!(expanded, vec!["a", "b", "a", "b", "a"]);
    }
}
