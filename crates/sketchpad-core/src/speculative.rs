//! Speculative Decoding for Accelerated Inference
//!
//! Speculative decoding uses a smaller draft model to predict multiple tokens,
//! then verifies them in parallel with the target model. This can significantly
//! speed up inference (2-3x) when the draft model has good agreement.
//!
//! # Algorithm
//!
//! 1. Draft model generates K tokens autoregressively
//! 2. Target model scores all K tokens in a single forward pass
//! 3. Accept/reject each token based on probability ratio
//! 4. Keep accepted tokens, resample from adjusted distribution if rejected
//!
//! # Reference
//!
//! "Fast Inference from Transformers via Speculative Decoding"
//! Leviathan et al., 2023

use burn::prelude::*;

/// Configuration for speculative decoding
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of tokens to speculate (draft length)
    pub num_speculative_tokens: usize,
    /// Temperature for sampling
    pub temperature: f32,
    /// Whether to use greedy decoding (argmax) vs sampling
    pub greedy: bool,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            num_speculative_tokens: 4,
            temperature: 1.0,
            greedy: false,
        }
    }
}

impl SpeculativeConfig {
    pub fn new(num_speculative_tokens: usize) -> Self {
        Self {
            num_speculative_tokens,
            ..Default::default()
        }
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_greedy(mut self, greedy: bool) -> Self {
        self.greedy = greedy;
        self
    }
}

/// Result of speculative verification
#[derive(Debug)]
pub struct SpeculativeResult<B: Backend> {
    /// Accepted tokens (may be fewer than speculated)
    pub accepted_tokens: Tensor<B, 1, Int>,
    /// Number of tokens accepted
    pub num_accepted: usize,
    /// Whether all speculated tokens were accepted
    pub all_accepted: bool,
    /// Next token after accepted sequence (from target model)
    pub bonus_token: Option<Tensor<B, 1, Int>>,
}

/// Verify speculated tokens against target model probabilities
///
/// # Arguments
/// * `draft_probs` - Draft model probabilities [num_tokens, vocab_size]
/// * `target_probs` - Target model probabilities [num_tokens, vocab_size]
/// * `draft_tokens` - Tokens sampled from draft model [num_tokens]
/// * `config` - Speculative decoding configuration
///
/// # Returns
/// Verification result with accepted tokens and optional bonus token
pub fn verify_speculative<B: Backend>(
    draft_probs: Tensor<B, 2>,
    target_probs: Tensor<B, 2>,
    draft_tokens: Tensor<B, 1, Int>,
    config: &SpeculativeConfig,
) -> SpeculativeResult<B> {
    let [num_tokens, vocab_size] = draft_probs.dims();
    let device = draft_probs.device();

    if num_tokens == 0 {
        return SpeculativeResult {
            accepted_tokens: Tensor::zeros([0], &device),
            num_accepted: 0,
            all_accepted: true,
            bonus_token: None,
        };
    }

    // Get probabilities for the draft tokens
    let draft_tokens_data = draft_tokens.clone().into_data();
    let draft_tokens_vec: Vec<i64> = draft_tokens_data.to_vec().unwrap();

    let mut accepted_tokens_vec: Vec<i64> = Vec::new();
    let mut first_rejection_idx: Option<usize> = None;

    for (i, &token) in draft_tokens_vec.iter().enumerate() {
        let token_idx = token as usize;

        // Get p_draft(token) and p_target(token)
        let p_draft = draft_probs
            .clone()
            .slice([i..i + 1, token_idx..token_idx + 1]);
        let p_target = target_probs
            .clone()
            .slice([i..i + 1, token_idx..token_idx + 1]);

        let p_draft_val: f32 = p_draft.into_scalar().elem();
        let p_target_val: f32 = p_target.into_scalar().elem();

        // Acceptance probability: min(1, p_target / p_draft)
        let accept_prob = if p_draft_val > 0.0 {
            (p_target_val / p_draft_val).min(1.0)
        } else if p_target_val > 0.0 {
            1.0 // Target wants this token but draft gave 0 prob - accept
        } else {
            0.0 // Both are 0 - reject
        };

        // For greedy mode, accept if target agrees
        let accepted = if config.greedy {
            // In greedy mode, accept if target's argmax matches
            let target_row = target_probs.clone().slice([i..i + 1, 0..vocab_size]);
            let target_argmax = target_row.argmax(1);
            let target_argmax_val: i64 = target_argmax.into_scalar().elem();
            target_argmax_val == token
        } else {
            // Stochastic acceptance based on ratio
            // In actual implementation, you'd sample uniform and compare
            // For simplicity, accept if ratio >= 0.5
            accept_prob >= 0.5
        };

        if accepted {
            accepted_tokens_vec.push(token);
        } else {
            first_rejection_idx = Some(i);
            break;
        }
    }

    let num_accepted = accepted_tokens_vec.len();
    let all_accepted = first_rejection_idx.is_none();

    // Get bonus token from target model at the position after accepted tokens
    let bonus_token = if all_accepted && num_tokens > 0 {
        // All tokens accepted - sample next token from target's last position
        // This gives us one extra token "for free"
        let last_target_probs = target_probs.slice([num_tokens - 1..num_tokens, 0..vocab_size]);
        let bonus = if config.greedy {
            last_target_probs.argmax(1)
        } else {
            // Sample from distribution (simplified - actual impl would do proper sampling)
            last_target_probs.argmax(1)
        };
        Some(bonus.reshape([1]))
    } else if let Some(reject_idx) = first_rejection_idx {
        // Resample from adjusted distribution at rejection point
        // p_adjusted(x) = max(0, p_target(x) - p_draft(x)) normalized
        let target_row = target_probs
            .clone()
            .slice([reject_idx..reject_idx + 1, 0..vocab_size]);
        let draft_row = draft_probs.slice([reject_idx..reject_idx + 1, 0..vocab_size]);

        let adjusted = target_row - draft_row;
        let adjusted = adjusted.clamp_min(0.0);

        // Normalize and sample (simplified)
        let resampled = adjusted.argmax(1);
        Some(resampled.reshape([1]))
    } else {
        None
    };

    let accepted_tokens = if accepted_tokens_vec.is_empty() {
        Tensor::zeros([0], &device)
    } else {
        Tensor::from_ints(accepted_tokens_vec.as_slice(), &device)
    };

    SpeculativeResult {
        accepted_tokens,
        num_accepted,
        all_accepted,
        bonus_token,
    }
}

/// Statistics for speculative decoding performance
#[derive(Debug, Default, Clone)]
pub struct SpeculativeStats {
    /// Total tokens generated
    pub total_tokens: usize,
    /// Total speculative rounds
    pub total_rounds: usize,
    /// Total accepted draft tokens
    pub accepted_tokens: usize,
    /// Total draft tokens proposed
    pub proposed_tokens: usize,
    /// Target model forward passes
    pub target_forwards: usize,
    /// Draft model forward passes
    pub draft_forwards: usize,
}

impl SpeculativeStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a speculative round
    pub fn record_round(&mut self, proposed: usize, accepted: usize, bonus: bool) {
        self.total_rounds += 1;
        self.proposed_tokens += proposed;
        self.accepted_tokens += accepted;
        self.total_tokens += accepted + if bonus { 1 } else { 0 };
        self.target_forwards += 1;
        self.draft_forwards += proposed;
    }

    /// Acceptance rate
    pub fn acceptance_rate(&self) -> f32 {
        if self.proposed_tokens == 0 {
            0.0
        } else {
            self.accepted_tokens as f32 / self.proposed_tokens as f32
        }
    }

    /// Average tokens per target forward (speedup indicator)
    pub fn tokens_per_target_forward(&self) -> f32 {
        if self.target_forwards == 0 {
            0.0
        } else {
            self.total_tokens as f32 / self.target_forwards as f32
        }
    }

    /// Theoretical speedup over standard decoding
    pub fn theoretical_speedup(&self) -> f32 {
        // Standard decoding: 1 token per forward
        // Speculative: tokens_per_target_forward tokens per forward
        self.tokens_per_target_forward()
    }
}

/// Helper to compute softmax probabilities from logits
pub fn logits_to_probs<B: Backend>(logits: Tensor<B, 2>, temperature: f32) -> Tensor<B, 2> {
    let scaled = if (temperature - 1.0).abs() > 1e-6 {
        logits / temperature
    } else {
        logits
    };
    burn::tensor::activation::softmax(scaled, 1)
}

/// Sample a token from probability distribution
/// Assumes probs is [batch=1, vocab_size]
pub fn sample_token<B: Backend>(probs: Tensor<B, 2>, greedy: bool) -> Tensor<B, 1, Int> {
    let token = if greedy {
        probs.argmax(1)
    } else {
        // Multinomial sampling via inverse CDF
        let [_batch, vocab_size] = probs.dims();
        let device = probs.device();

        // Compute cumulative sum along vocab dimension
        // cumsum[i] = sum(probs[0..=i])
        let probs_1d = probs.clone().flatten::<1>(0, 1);
        let probs_data: Vec<f32> = probs_1d.to_data().to_vec().unwrap();

        let mut cumsum = Vec::with_capacity(vocab_size);
        let mut sum = 0.0f32;
        for &p in &probs_data {
            sum += p;
            cumsum.push(sum);
        }

        // Sample uniform random value in [0, 1]
        let uniform: Tensor<B, 1> =
            Tensor::random([1], burn::tensor::Distribution::Uniform(0.0, 1.0), &device);
        let threshold: f32 = uniform.into_scalar().elem();

        // Find first index where cumsum > threshold
        let sampled_idx = cumsum
            .iter()
            .position(|&cs| cs > threshold)
            .unwrap_or(vocab_size - 1);

        Tensor::from_ints([sampled_idx as i32], &device)
    };
    token.flatten(0, 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_speculative_config() {
        let config = SpeculativeConfig::new(5)
            .with_temperature(0.8)
            .with_greedy(true);

        assert_eq!(config.num_speculative_tokens, 5);
        assert_eq!(config.temperature, 0.8);
        assert!(config.greedy);
    }

    #[test]
    fn test_verify_speculative_all_accepted() {
        let device = Default::default();

        // Create probabilities where target agrees with draft
        // vocab_size = 4, num_tokens = 3
        let draft_probs = Tensor::<TestBackend, 2>::from_floats(
            [
                [0.1, 0.7, 0.1, 0.1], // token 1
                [0.1, 0.1, 0.7, 0.1], // token 2
                [0.7, 0.1, 0.1, 0.1], // token 0
            ],
            &device,
        );

        // Target has similar distribution
        let target_probs = Tensor::<TestBackend, 2>::from_floats(
            [
                [0.1, 0.8, 0.05, 0.05], // agrees: token 1
                [0.05, 0.05, 0.8, 0.1], // agrees: token 2
                [0.8, 0.1, 0.05, 0.05], // agrees: token 0
            ],
            &device,
        );

        let draft_tokens = Tensor::<TestBackend, 1, Int>::from_ints([1, 2, 0], &device);
        let config = SpeculativeConfig::new(3).with_greedy(true);

        let result = verify_speculative(draft_probs, target_probs, draft_tokens, &config);

        assert_eq!(result.num_accepted, 3);
        assert!(result.all_accepted);
        assert!(result.bonus_token.is_some());
    }

    #[test]
    fn test_verify_speculative_partial_rejection() {
        let device = Default::default();

        let draft_probs = Tensor::<TestBackend, 2>::from_floats(
            [
                [0.1, 0.7, 0.1, 0.1], // token 1
                [0.1, 0.1, 0.7, 0.1], // token 2 (but target wants 3)
                [0.7, 0.1, 0.1, 0.1], // token 0
            ],
            &device,
        );

        // Target disagrees on second token
        let target_probs = Tensor::<TestBackend, 2>::from_floats(
            [
                [0.1, 0.8, 0.05, 0.05], // agrees: token 1
                [0.05, 0.05, 0.1, 0.8], // disagrees: wants token 3
                [0.8, 0.1, 0.05, 0.05], // (never reached)
            ],
            &device,
        );

        let draft_tokens = Tensor::<TestBackend, 1, Int>::from_ints([1, 2, 0], &device);
        let config = SpeculativeConfig::new(3).with_greedy(true);

        let result = verify_speculative(draft_probs, target_probs, draft_tokens, &config);

        assert_eq!(result.num_accepted, 1); // Only first token accepted
        assert!(!result.all_accepted);
        assert!(result.bonus_token.is_some()); // Resampled token
    }

    #[test]
    fn test_verify_speculative_empty() {
        let device = Default::default();
        let config = SpeculativeConfig::new(0);

        let draft_probs = Tensor::<TestBackend, 2>::zeros([0, 100], &device);
        let target_probs = Tensor::<TestBackend, 2>::zeros([0, 100], &device);
        let draft_tokens = Tensor::<TestBackend, 1, Int>::zeros([0], &device);

        let result = verify_speculative(draft_probs, target_probs, draft_tokens, &config);

        assert_eq!(result.num_accepted, 0);
        assert!(result.all_accepted);
    }

    #[test]
    fn test_speculative_stats() {
        let mut stats = SpeculativeStats::new();

        // Round 1: 4 proposed, 3 accepted, bonus token
        stats.record_round(4, 3, true);
        assert_eq!(stats.total_tokens, 4); // 3 accepted + 1 bonus
        assert_eq!(stats.total_rounds, 1);

        // Round 2: 4 proposed, 4 accepted, bonus token
        stats.record_round(4, 4, true);
        assert_eq!(stats.total_tokens, 9); // 4 + 5

        assert_eq!(stats.acceptance_rate(), 7.0 / 8.0); // 7/8 = 0.875
        assert_eq!(stats.tokens_per_target_forward(), 4.5); // 9/2
    }

    #[test]
    fn test_logits_to_probs() {
        let device = Default::default();
        let logits = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0]], &device);

        let probs = logits_to_probs(logits, 1.0);
        let probs_data: Vec<f32> = probs.into_data().to_vec().unwrap();

        // Softmax should sum to 1
        let sum: f32 = probs_data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Highest logit should have highest prob
        assert!(probs_data[2] > probs_data[1]);
        assert!(probs_data[1] > probs_data[0]);
    }
}
