//! LLM Token Sampling
//!
//! Implements a composable sampler pipeline for LLM text generation.
//! Samplers are applied in order: penalties → temperature → filtering → selection.
//!
//! # Sampler chain (recommended order)
//!
//! 1. DRY (Don't Repeat Yourself) — penalize repeated n-gram patterns
//! 2. XTC (eXclude Top Choices) — probabilistically remove top tokens
//! 3. Repetition penalty — penalize recently used tokens
//! 4. Temperature — scale logit distribution
//! 5. Top-k — keep only k highest-probability tokens
//! 6. Top-p (nucleus) — keep smallest set summing to p probability mass
//! 7. Min-p — keep tokens with probability >= min_p * max_probability
//! 8. Sample — weighted random selection from remaining candidates

use burn::prelude::*;

/// Configuration for the sampler pipeline
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// Temperature scaling (1.0 = no change, <1 = sharper, >1 = flatter)
    pub temperature: f32,
    /// Top-k: keep only the k most probable tokens (0 = disabled)
    pub top_k: usize,
    /// Top-p (nucleus): keep smallest set with cumulative probability >= p (1.0 = disabled)
    pub top_p: f32,
    /// Min-p: keep tokens with probability >= min_p * max_prob (0.0 = disabled)
    pub min_p: f32,
    /// Repetition penalty multiplier (1.0 = disabled)
    pub repetition_penalty: f32,
    /// How many recent tokens to consider for repetition penalty
    pub repetition_penalty_window: usize,
    /// DRY: multiplier for n-gram repetition penalty (0.0 = disabled)
    pub dry_multiplier: f32,
    /// DRY: base value for exponential penalty scaling
    pub dry_base: f32,
    /// DRY: minimum n-gram length to penalize
    pub dry_allowed_length: usize,
    /// DRY: how many recent tokens to scan for patterns
    pub dry_penalty_window: usize,
    /// XTC: probability threshold above which tokens may be excluded (1.0 = disabled)
    pub xtc_threshold: f32,
    /// XTC: probability of actually excluding an eligible token
    pub xtc_probability: f32,
    /// Random seed for sampling (None = use system randomness)
    pub seed: Option<u64>,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.05,
            repetition_penalty: 1.0,
            repetition_penalty_window: 64,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_penalty_window: 256,
            xtc_threshold: 1.0,
            xtc_probability: 0.0,
            seed: None,
        }
    }
}

impl SamplerConfig {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            min_p: 0.0,
            ..Default::default()
        }
    }

    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    pub fn with_top_p(mut self, p: f32) -> Self {
        self.top_p = p;
        self
    }

    pub fn with_min_p(mut self, p: f32) -> Self {
        self.min_p = p;
        self
    }

    pub fn with_repetition_penalty(mut self, penalty: f32, window: usize) -> Self {
        self.repetition_penalty = penalty;
        self.repetition_penalty_window = window;
        self
    }

    pub fn with_dry(
        mut self,
        multiplier: f32,
        base: f32,
        allowed_length: usize,
        window: usize,
    ) -> Self {
        self.dry_multiplier = multiplier;
        self.dry_base = base;
        self.dry_allowed_length = allowed_length;
        self.dry_penalty_window = window;
        self
    }

    pub fn with_xtc(mut self, threshold: f32, probability: f32) -> Self {
        self.xtc_threshold = threshold;
        self.xtc_probability = probability;
        self
    }
}

/// Simple xorshift64 PRNG for reproducible sampling without external deps
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0xDEAD_BEEF_CAFE_BABE
            } else {
                seed
            },
        }
    }

    fn from_time() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42);
        Self::new(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Returns a float in [0, 1)
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Sample a token from logits using the configured sampler pipeline
///
/// # Arguments
/// * `logits` - Raw logits for the vocabulary [vocab_size]
/// * `context_tokens` - Previously generated token IDs (for repetition/DRY penalties)
/// * `config` - Sampler configuration
///
/// # Returns
/// Selected token ID
pub fn sample_token(logits: &[f32], context_tokens: &[u32], config: &SamplerConfig) -> u32 {
    let vocab_size = logits.len();
    let mut scores: Vec<f32> = logits.to_vec();

    let mut rng = match config.seed {
        Some(seed) => Rng::new(seed),
        None => Rng::from_time(),
    };

    // 1. DRY penalty
    if config.dry_multiplier > 0.0 && !context_tokens.is_empty() {
        apply_dry_penalty(
            &mut scores,
            context_tokens,
            config.dry_multiplier,
            config.dry_base,
            config.dry_allowed_length,
            config.dry_penalty_window,
        );
    }

    // 2. Repetition penalty
    if config.repetition_penalty != 1.0 && !context_tokens.is_empty() {
        apply_repetition_penalty(
            &mut scores,
            context_tokens,
            config.repetition_penalty,
            config.repetition_penalty_window,
        );
    }

    // 3. Temperature
    if config.temperature <= 0.0 || config.top_k == 1 {
        // Greedy: return argmax
        return scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }

    if (config.temperature - 1.0).abs() > 1e-6 {
        let inv_temp = 1.0 / config.temperature;
        for s in &mut scores {
            *s *= inv_temp;
        }
    }

    // Build sorted index for filtering
    let mut indices: Vec<u32> = (0..vocab_size as u32).collect();
    indices.sort_unstable_by(|&a, &b| {
        scores[b as usize]
            .partial_cmp(&scores[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Convert to probabilities via softmax on sorted scores
    let max_score = scores[indices[0] as usize];
    let mut probs: Vec<f32> = indices
        .iter()
        .map(|&i| (scores[i as usize] - max_score).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for p in &mut probs {
            *p *= inv_sum;
        }
    }

    // 4. XTC (eXclude Top Choices)
    // Remove tokens whose probability exceeds threshold, with xtc_probability chance
    let mut keep = vec![true; indices.len()];
    if config.xtc_threshold < 1.0 && config.xtc_probability > 0.0 {
        let mut num_kept = indices.len();
        for (rank, &prob) in probs.iter().enumerate() {
            if prob < config.xtc_threshold {
                break; // Sorted descending, no more candidates
            }
            // Always keep at least one token
            if num_kept <= 1 {
                break;
            }
            if rng.next_f32() < config.xtc_probability {
                keep[rank] = false;
                num_kept -= 1;
            }
        }
    }

    // 5. Top-k
    if config.top_k > 0 && config.top_k < indices.len() {
        for k in keep.iter_mut().skip(config.top_k) {
            *k = false;
        }
    }

    // 6. Min-p
    if config.min_p > 0.0 {
        let threshold = config.min_p * probs[0]; // probs[0] is max prob (sorted)
        for (i, &p) in probs.iter().enumerate() {
            if p < threshold {
                keep[i] = false;
            }
        }
    }

    // 7. Top-p (nucleus)
    if config.top_p < 1.0 {
        let mut cumulative = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            if !keep[i] {
                continue;
            }
            cumulative += p;
            if cumulative > config.top_p {
                // Keep this one but exclude the rest
                for k in keep.iter_mut().skip(i + 1) {
                    *k = false;
                }
                break;
            }
        }
    }

    // Collect surviving candidates
    let mut candidates: Vec<(u32, f32)> = indices
        .iter()
        .zip(probs.iter())
        .zip(keep.iter())
        .filter(|&(_, &k)| k)
        .map(|((&idx, &prob), _)| (idx, prob))
        .collect();

    if candidates.is_empty() {
        // Shouldn't happen, but fall back to argmax
        return indices[0];
    }

    // Renormalize
    let total: f32 = candidates.iter().map(|(_, p)| p).sum();
    if total > 0.0 {
        let inv_total = 1.0 / total;
        for (_, p) in &mut candidates {
            *p *= inv_total;
        }
    }

    // 8. Weighted random selection
    let r = rng.next_f32();
    let mut cumulative = 0.0;
    for &(token, prob) in &candidates {
        cumulative += prob;
        if r < cumulative {
            return token;
        }
    }

    // Float precision fallback
    candidates.last().map(|&(t, _)| t).unwrap_or(0)
}

/// DRY (Don't Repeat Yourself) penalty
///
/// Scans recent context for repeated n-gram patterns ending at the current position.
/// For each token that would continue a repeated pattern, applies an exponential
/// penalty based on how many times the pattern has repeated.
fn apply_dry_penalty(
    scores: &mut [f32],
    context: &[u32],
    multiplier: f32,
    base: f32,
    allowed_length: usize,
    window: usize,
) {
    let ctx_len = context.len();
    if ctx_len < 2 {
        return;
    }

    let start = ctx_len.saturating_sub(window);
    let recent = &context[start..];

    // For each position in the recent context, check if the sequence ending there
    // matches a sequence ending at the current position (end of context)
    let last_token = *context.last().unwrap();

    // Find positions where the last token appeared before
    for scan_end in (0..recent.len().saturating_sub(1)).rev() {
        if recent[scan_end] != last_token {
            continue;
        }

        // Count how far back the match extends
        let mut match_len = 1;
        while match_len <= scan_end
            && match_len < recent.len() - scan_end - 1
            && recent[scan_end - match_len] == recent[recent.len() - 1 - match_len]
        {
            match_len += 1;
        }

        if match_len <= allowed_length {
            continue;
        }

        // The token that followed this pattern last time would be penalized
        let next_pos = scan_end + 1;
        if next_pos < recent.len() {
            let next_token = recent[next_pos] as usize;
            if next_token < scores.len() {
                // Exponential penalty: multiplier * base^(match_len - allowed_length)
                let exponent = (match_len - allowed_length) as f32;
                let penalty = multiplier * base.powf(exponent);
                scores[next_token] -= penalty;
            }
        }
    }
}

/// Standard repetition penalty: scale logits for tokens that appear in context
fn apply_repetition_penalty(scores: &mut [f32], context: &[u32], penalty: f32, window: usize) {
    let start = context.len().saturating_sub(window);
    let recent = &context[start..];

    // Collect unique tokens in window
    let mut seen = std::collections::HashSet::new();
    for &token in recent {
        seen.insert(token as usize);
    }

    for token_id in seen {
        if token_id < scores.len() {
            if scores[token_id] > 0.0 {
                scores[token_id] /= penalty;
            } else {
                scores[token_id] *= penalty;
            }
        }
    }
}

/// Convenience: sample from a Burn logits tensor
///
/// Extracts the last position's logits, converts to f32 vec, and samples.
pub fn sample_from_logits<B: Backend>(
    logits: Tensor<B, 2>, // [batch, vocab_size] — only batch=1 supported
    context_tokens: &[u32],
    config: &SamplerConfig,
) -> u32 {
    let [_batch, _vocab] = logits.dims();
    let logits_vec: Vec<f32> = logits
        .slice([0..1, 0.._vocab])
        .reshape([_vocab])
        .to_data()
        .to_vec()
        .unwrap();
    sample_token(&logits_vec, context_tokens, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_sampling() {
        let logits = vec![1.0, 5.0, 2.0, 3.0];
        let config = SamplerConfig::greedy();
        let token = sample_token(&logits, &[], &config);
        assert_eq!(token, 1); // Index of max value
    }

    #[test]
    fn test_temperature_zero_is_greedy() {
        let logits = vec![1.0, 2.0, 10.0, 3.0];
        let config = SamplerConfig {
            temperature: 0.0,
            ..Default::default()
        };
        let token = sample_token(&logits, &[], &config);
        assert_eq!(token, 2);
    }

    #[test]
    fn test_top_k_limits_candidates() {
        // With top_k=1, should always pick the max
        let logits = vec![1.0, 5.0, 2.0, 3.0];
        let config = SamplerConfig {
            top_k: 1,
            temperature: 1.0,
            ..Default::default()
        };
        for _ in 0..10 {
            let token = sample_token(&logits, &[], &config);
            assert_eq!(token, 1);
        }
    }

    #[test]
    fn test_repetition_penalty() {
        // Token 1 has highest logit but was recently used — penalty should reduce it
        let logits = vec![1.0, 5.0, 4.9, 3.0];
        let context = vec![1u32; 10]; // Token 1 repeated
        let mut scores = logits.clone();
        apply_repetition_penalty(&mut scores, &context, 2.0, 64);
        // Token 1's score should be halved (positive logit / penalty)
        assert!((scores[1] - 2.5).abs() < 1e-6);
        // Token 0 should be halved too (it appears as token "1" which is index 1... wait)
        // Actually context is [1,1,1,...] so token_id=1 is penalized
        assert!((scores[0] - 1.0).abs() < 1e-6); // Token 0 untouched
        assert!((scores[2] - 4.9).abs() < 1e-6); // Token 2 untouched
    }

    #[test]
    fn test_dry_penalty_penalizes_repeated_ngram() {
        // Context: "A B C A B" — the next token continuing the "A B C" pattern would be C
        let context = vec![10, 20, 30, 10, 20];
        let mut scores = vec![0.0; 100];
        scores[30] = 5.0; // Token 30 (C) has high score

        apply_dry_penalty(&mut scores, &context, 1.0, 1.75, 1, 256);

        // Token 30 should be penalized because "A B" repeated and C followed last time
        assert!(
            scores[30] < 5.0,
            "DRY should penalize token 30, got {}",
            scores[30]
        );
    }

    #[test]
    fn test_dry_penalty_ignores_short_patterns() {
        // With allowed_length=2, single-token matches should not be penalized
        let context = vec![10, 20, 30, 10];
        let mut scores = vec![0.0; 100];
        scores[20] = 5.0;

        apply_dry_penalty(&mut scores, &context, 1.0, 1.75, 2, 256);

        // Match length is only 1 (just token 10), which is <= allowed_length=2
        assert!(
            (scores[20] - 5.0).abs() < 1e-6,
            "Short pattern should not be penalized"
        );
    }

    #[test]
    fn test_min_p_filtering() {
        // With a very dominant token, min_p should filter out low-prob tokens
        let logits = vec![10.0, 0.0, 0.0, 0.0]; // Token 0 is dominant
        let config = SamplerConfig {
            temperature: 1.0,
            min_p: 0.1,
            top_k: 0,
            top_p: 1.0,
            seed: Some(42),
            ..Default::default()
        };
        // Should almost always pick token 0 since others are far below min_p threshold
        let token = sample_token(&logits, &[], &config);
        assert_eq!(token, 0);
    }

    #[test]
    fn test_xtc_can_exclude_top() {
        // With xtc_probability=1.0 and low threshold, all tokens above threshold are excluded
        // (keeping at least one). Logits [10, 9.9, 0, 0] → probs ~[0.52, 0.48, ~0, ~0].
        // Both tokens 0 and 1 exceed 0.01, so both get excluded (leaving tokens 2 and 3).
        // XTC must keep at least 1, so the last eligible one survives.
        let logits = vec![10.0, 9.9, 0.0, 0.0];
        let config = SamplerConfig {
            temperature: 1.0,
            xtc_threshold: 0.01,
            xtc_probability: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: Some(42),
            ..Default::default()
        };
        let token = sample_token(&logits, &[], &config);
        // Top tokens (0, 1) should be excluded — result should NOT be 0
        assert!(token != 0, "XTC should have excluded token 0, got {token}");
    }

    #[test]
    fn test_rng_deterministic() {
        let mut rng1 = Rng::new(12345);
        let mut rng2 = Rng::new(12345);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_seeded_sampling_deterministic() {
        let logits = vec![1.0, 2.0, 3.0, 2.5, 1.5];
        let config = SamplerConfig {
            temperature: 1.0,
            seed: Some(42),
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            ..Default::default()
        };
        let t1 = sample_token(&logits, &[], &config);
        let t2 = sample_token(&logits, &[], &config);
        assert_eq!(t1, t2, "Same seed should produce same result");
    }
}
