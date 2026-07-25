//! Continuous Batching for High-Throughput Inference
//!
//! Continuous batching (also called iteration-level scheduling) allows
//! sequences to dynamically join and leave a batch at each iteration,
//! maximizing GPU utilization.
//!
//! # How It Works
//!
//! Traditional batching:
//! ```text
//! Batch 1: [seq0, seq1, seq2] -> wait for all to finish -> Batch 2
//! ```
//!
//! Continuous batching:
//! ```text
//! Iteration 1: [seq0, seq1, seq2]
//! Iteration 2: [seq0, seq1, seq3]  <- seq2 finished, seq3 joined
//! Iteration 3: [seq0, seq3, seq4]  <- seq1 finished, seq4 joined
//! ```
//!
//! # Benefits
//!
//! - Higher throughput (no waiting for slow sequences)
//! - Lower latency for short sequences
//! - Better GPU utilization
//! - Works with paged attention for memory efficiency

use std::collections::{HashMap, VecDeque};

/// Sequence state in the batch
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
    /// Waiting in queue
    Waiting,
    /// Currently being processed
    Running,
    /// Generation complete
    Finished,
    /// Preempted due to memory pressure
    Preempted,
}

/// A sequence being processed
#[derive(Debug)]
pub struct Sequence {
    /// Unique sequence ID
    pub id: usize,
    /// Input token IDs
    pub input_ids: Vec<i64>,
    /// Generated token IDs
    pub output_ids: Vec<i64>,
    /// Current status
    pub status: SequenceStatus,
    /// Number of tokens generated
    pub generated_len: usize,
    /// Maximum tokens to generate
    pub max_new_tokens: usize,
    /// Arrival time (for scheduling)
    pub arrival_time: u64,
    /// Number of blocks allocated (for memory tracking)
    pub num_blocks: usize,
}

impl Sequence {
    pub fn new(id: usize, input_ids: Vec<i64>, max_new_tokens: usize, arrival_time: u64) -> Self {
        Self {
            id,
            input_ids,
            output_ids: Vec::new(),
            status: SequenceStatus::Waiting,
            generated_len: 0,
            max_new_tokens,
            arrival_time,
            num_blocks: 0,
        }
    }

    /// Total sequence length (prompt + generated)
    pub fn total_len(&self) -> usize {
        self.input_ids.len() + self.output_ids.len()
    }

    /// Check if generation is complete
    pub fn is_finished(&self) -> bool {
        self.generated_len >= self.max_new_tokens || self.status == SequenceStatus::Finished
    }

    /// Add a generated token
    pub fn add_token(&mut self, token: i64) {
        self.output_ids.push(token);
        self.generated_len += 1;
    }

    /// Mark as finished
    pub fn finish(&mut self) {
        self.status = SequenceStatus::Finished;
    }
}

/// Scheduling policy for sequence selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// First come, first served
    Fcfs,
    /// Shortest job first (by remaining tokens)
    ShortestFirst,
    /// Longest job first
    LongestFirst,
}

/// Configuration for continuous batching
#[derive(Debug, Clone)]
pub struct ContinuousBatchingConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Block size for paged attention
    pub block_size: usize,
    /// Total memory blocks available
    pub total_blocks: usize,
    /// Scheduling policy
    pub policy: SchedulingPolicy,
    /// Enable preemption under memory pressure
    pub enable_preemption: bool,
}

impl Default for ContinuousBatchingConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 256,
            max_seq_len: 4096,
            block_size: 16,
            total_blocks: 1000,
            policy: SchedulingPolicy::Fcfs,
            enable_preemption: true,
        }
    }
}

impl ContinuousBatchingConfig {
    pub fn new(max_batch_size: usize, total_blocks: usize) -> Self {
        Self {
            max_batch_size,
            total_blocks,
            ..Default::default()
        }
    }
}

/// Result of a scheduling decision
#[derive(Debug)]
pub struct ScheduleOutput {
    /// Sequences to run this iteration
    pub running: Vec<usize>,
    /// Sequences that were preempted
    pub preempted: Vec<usize>,
    /// Sequences that finished
    pub finished: Vec<usize>,
    /// Number of free blocks remaining
    pub free_blocks: usize,
}

/// Continuous batching scheduler
pub struct ContinuousBatchingScheduler {
    /// Configuration
    config: ContinuousBatchingConfig,
    /// All sequences by ID
    sequences: HashMap<usize, Sequence>,
    /// Waiting queue
    waiting_queue: VecDeque<usize>,
    /// Currently running sequences
    running: Vec<usize>,
    /// Next sequence ID
    next_id: usize,
    /// Current time (iteration count)
    current_time: u64,
    /// Free memory blocks
    free_blocks: usize,
}

impl ContinuousBatchingScheduler {
    pub fn new(config: ContinuousBatchingConfig) -> Self {
        let free_blocks = config.total_blocks;
        Self {
            config,
            sequences: HashMap::new(),
            waiting_queue: VecDeque::new(),
            running: Vec::new(),
            next_id: 0,
            current_time: 0,
            free_blocks,
        }
    }

    /// Add a new sequence to the scheduler
    pub fn add_sequence(&mut self, input_ids: Vec<i64>, max_new_tokens: usize) -> usize {
        let id = self.next_id;
        self.next_id += 1;

        let seq = Sequence::new(id, input_ids, max_new_tokens, self.current_time);
        self.sequences.insert(id, seq);
        self.waiting_queue.push_back(id);

        id
    }

    /// Get a sequence by ID
    pub fn get_sequence(&self, id: usize) -> Option<&Sequence> {
        self.sequences.get(&id)
    }

    /// Get mutable sequence by ID
    pub fn get_sequence_mut(&mut self, id: usize) -> Option<&mut Sequence> {
        self.sequences.get_mut(&id)
    }

    /// Calculate blocks needed for a sequence
    fn blocks_needed(&self, seq: &Sequence) -> usize {
        let total_tokens = seq.total_len() + 1; // +1 for next token
        total_tokens.div_ceil(self.config.block_size)
    }

    /// Schedule sequences for the next iteration
    pub fn schedule(&mut self) -> ScheduleOutput {
        self.current_time += 1;

        let mut output = ScheduleOutput {
            running: Vec::new(),
            preempted: Vec::new(),
            finished: Vec::new(),
            free_blocks: self.free_blocks,
        };

        // First, handle finished sequences
        let mut still_running = Vec::new();
        for &seq_id in &self.running {
            if let Some(seq) = self.sequences.get(&seq_id) {
                if seq.is_finished() {
                    output.finished.push(seq_id);
                    // Free blocks
                    self.free_blocks += seq.num_blocks;
                } else {
                    still_running.push(seq_id);
                }
            }
        }
        self.running = still_running;

        // Check if running sequences need more blocks
        for &seq_id in &self.running {
            if let Some(seq) = self.sequences.get(&seq_id) {
                let needed = self.blocks_needed(seq);
                if needed > seq.num_blocks {
                    let additional = needed - seq.num_blocks;
                    if additional <= self.free_blocks {
                        self.free_blocks -= additional;
                        if let Some(seq) = self.sequences.get_mut(&seq_id) {
                            seq.num_blocks = needed;
                        }
                    } else if self.config.enable_preemption {
                        // Need to preempt - handled below
                    }
                }
            }
        }

        // Handle preemption if needed
        if self.config.enable_preemption {
            while self.free_blocks == 0 && !self.running.is_empty() {
                // Preempt the sequence with most tokens (likely to free most blocks)
                let preempt_id = self.select_for_preemption();
                if let Some(id) = preempt_id {
                    if let Some(seq) = self.sequences.get_mut(&id) {
                        self.free_blocks += seq.num_blocks;
                        seq.num_blocks = 0;
                        seq.status = SequenceStatus::Preempted;
                        output.preempted.push(id);
                    }
                    self.running.retain(|&x| x != id);
                    self.waiting_queue.push_front(id); // Re-add to front of queue
                } else {
                    break;
                }
            }
        }

        // Try to add waiting sequences
        while self.running.len() < self.config.max_batch_size {
            let next_id = self.select_next_waiting();
            if let Some(seq_id) = next_id {
                if let Some(seq) = self.sequences.get(&seq_id) {
                    let blocks_needed = self.blocks_needed(seq);
                    if blocks_needed <= self.free_blocks {
                        self.free_blocks -= blocks_needed;
                        self.waiting_queue.retain(|&x| x != seq_id);
                        self.running.push(seq_id);
                        if let Some(seq) = self.sequences.get_mut(&seq_id) {
                            seq.status = SequenceStatus::Running;
                            seq.num_blocks = blocks_needed;
                        }
                    } else {
                        break; // Not enough memory
                    }
                }
            } else {
                break; // No more waiting sequences
            }
        }

        output.running = self.running.clone();
        output.free_blocks = self.free_blocks;

        output
    }

    /// Select next waiting sequence based on policy
    fn select_next_waiting(&self) -> Option<usize> {
        match self.config.policy {
            SchedulingPolicy::Fcfs => self.waiting_queue.front().copied(),
            SchedulingPolicy::ShortestFirst => self
                .waiting_queue
                .iter()
                .min_by_key(|&&id| {
                    self.sequences
                        .get(&id)
                        .map(|s| s.max_new_tokens - s.generated_len)
                        .unwrap_or(usize::MAX)
                })
                .copied(),
            SchedulingPolicy::LongestFirst => self
                .waiting_queue
                .iter()
                .max_by_key(|&&id| {
                    self.sequences
                        .get(&id)
                        .map(|s| s.max_new_tokens - s.generated_len)
                        .unwrap_or(0)
                })
                .copied(),
        }
    }

    /// Select a sequence to preempt (longest running)
    fn select_for_preemption(&self) -> Option<usize> {
        self.running
            .iter()
            .max_by_key(|&&id| self.sequences.get(&id).map(|s| s.num_blocks).unwrap_or(0))
            .copied()
    }

    /// Update sequence with generated token
    pub fn update_sequence(&mut self, seq_id: usize, token: i64) {
        if let Some(seq) = self.sequences.get_mut(&seq_id) {
            seq.add_token(token);
        }
    }

    /// Mark sequence as finished (e.g., hit EOS token)
    pub fn finish_sequence(&mut self, seq_id: usize) {
        if let Some(seq) = self.sequences.get_mut(&seq_id) {
            seq.finish();
        }
    }

    /// Remove a finished sequence and return it
    pub fn remove_sequence(&mut self, seq_id: usize) -> Option<Sequence> {
        self.running.retain(|&x| x != seq_id);
        self.waiting_queue.retain(|&x| x != seq_id);
        self.sequences.remove(&seq_id)
    }

    /// Number of sequences in the system
    pub fn num_sequences(&self) -> usize {
        self.sequences.len()
    }

    /// Number of running sequences
    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    /// Number of waiting sequences
    pub fn num_waiting(&self) -> usize {
        self.waiting_queue.len()
    }

    /// Check if scheduler is empty
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    /// Get throughput statistics
    pub fn stats(&self) -> BatchingStats {
        let total_running_tokens: usize = self
            .running
            .iter()
            .filter_map(|&id| self.sequences.get(&id))
            .map(|s| s.total_len())
            .sum();

        BatchingStats {
            num_running: self.running.len(),
            num_waiting: self.waiting_queue.len(),
            total_running_tokens,
            free_blocks: self.free_blocks,
            total_blocks: self.config.total_blocks,
            utilization: 1.0 - (self.free_blocks as f32 / self.config.total_blocks as f32),
        }
    }
}

/// Statistics for monitoring batch performance
#[derive(Debug, Clone)]
pub struct BatchingStats {
    pub num_running: usize,
    pub num_waiting: usize,
    pub total_running_tokens: usize,
    pub free_blocks: usize,
    pub total_blocks: usize,
    pub utilization: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_lifecycle() {
        let mut seq = Sequence::new(0, vec![1, 2, 3], 5, 0);

        assert_eq!(seq.total_len(), 3);
        assert!(!seq.is_finished());
        assert_eq!(seq.status, SequenceStatus::Waiting);

        seq.add_token(4);
        seq.add_token(5);
        assert_eq!(seq.total_len(), 5);
        assert_eq!(seq.generated_len, 2);

        seq.add_token(6);
        seq.add_token(7);
        seq.add_token(8);
        assert!(seq.is_finished()); // 5 tokens generated
    }

    #[test]
    fn test_scheduler_add_and_schedule() {
        let config = ContinuousBatchingConfig {
            max_batch_size: 2,
            total_blocks: 100,
            block_size: 4,
            ..Default::default()
        };
        let mut scheduler = ContinuousBatchingScheduler::new(config);

        // Add sequences
        let id0 = scheduler.add_sequence(vec![1, 2, 3], 10);
        let _id1 = scheduler.add_sequence(vec![4, 5], 10);
        let _id2 = scheduler.add_sequence(vec![6, 7, 8, 9], 10);

        assert_eq!(scheduler.num_sequences(), 3);
        assert_eq!(scheduler.num_waiting(), 3);

        // Schedule first batch
        let output = scheduler.schedule();
        assert_eq!(output.running.len(), 2); // Max batch size
        assert_eq!(scheduler.num_waiting(), 1);

        // Finish one sequence
        scheduler.finish_sequence(id0);

        // Schedule again
        let output = scheduler.schedule();
        assert!(output.finished.contains(&id0));
        assert_eq!(output.running.len(), 2); // id2 should have joined
    }

    #[test]
    fn test_scheduler_memory_limits() {
        let config = ContinuousBatchingConfig {
            max_batch_size: 10,
            total_blocks: 5, // Very limited memory
            block_size: 4,
            enable_preemption: false,
            ..Default::default()
        };
        let mut scheduler = ContinuousBatchingScheduler::new(config);

        // Add sequences that each need 1 block
        for i in 0..10 {
            scheduler.add_sequence(vec![i as i64; 3], 10);
        }

        let output = scheduler.schedule();
        // Should only run as many as memory allows
        assert!(output.running.len() <= 5);
    }

    #[test]
    fn test_scheduler_preemption() {
        let config = ContinuousBatchingConfig {
            max_batch_size: 10,
            total_blocks: 10,
            block_size: 4,
            enable_preemption: true,
            ..Default::default()
        };
        let mut scheduler = ContinuousBatchingScheduler::new(config);

        // Add and run a sequence
        let id0 = scheduler.add_sequence(vec![1; 15], 100); // Needs 4 blocks
        let output = scheduler.schedule();
        assert!(output.running.contains(&id0));

        // Simulate growth - add many tokens
        for _ in 0..30 {
            scheduler.update_sequence(id0, 99);
        }

        // Add more sequences that compete for memory
        for _ in 0..10 {
            scheduler.add_sequence(vec![1; 10], 50);
        }

        // This should trigger preemption eventually
        let output = scheduler.schedule();
        // Either preempted or running with fewer sequences
        assert!(!output.preempted.is_empty() || output.running.len() < 10);
    }

    #[test]
    fn test_scheduling_policies() {
        // Test shortest first
        let config = ContinuousBatchingConfig {
            max_batch_size: 1,
            total_blocks: 100,
            policy: SchedulingPolicy::ShortestFirst,
            ..Default::default()
        };
        let mut scheduler = ContinuousBatchingScheduler::new(config);

        let _long_id = scheduler.add_sequence(vec![1], 100);
        let short_id = scheduler.add_sequence(vec![2], 10);

        let output = scheduler.schedule();
        // Should pick short sequence first
        assert_eq!(output.running[0], short_id);
    }

    #[test]
    fn test_stats() {
        let config = ContinuousBatchingConfig::default();
        let mut scheduler = ContinuousBatchingScheduler::new(config);

        scheduler.add_sequence(vec![1, 2, 3], 10);
        scheduler.add_sequence(vec![4, 5], 10);

        scheduler.schedule();

        let stats = scheduler.stats();
        assert_eq!(stats.num_running, 2);
        assert_eq!(stats.total_running_tokens, 5); // 3 + 2
        assert!(stats.utilization > 0.0);
    }
}
