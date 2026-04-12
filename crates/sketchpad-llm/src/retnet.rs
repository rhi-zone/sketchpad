//! RetNet: Retentive Network
//!
//! Microsoft's architecture from "Retentive Network: A Successor to Transformer for
//! Large Language Models" (Sun et al., 2023). Key innovation: retention mechanism
//! supporting three equivalent computation modes — parallel (training), recurrent
//! (O(1) per step inference), and chunkwise (balance).
//!
//! This implementation uses the **recurrent mode** for inference. Each head maintains
//! a state matrix S ∈ ℝ^{d_k × d_v} that is updated as:
//!
//! ```text
//! S_t = γ * S_{t-1} + k ⊗ v   (outer product update)
//! y   = q @ S_t
//! ```
//!
//! where γ = 1 - 2^{-5 - h} differs per head h.
//!
//! # Architecture
//!
//! - Embedding → N layers → LayerNorm → LM head
//! - Each layer: LayerNorm → MultiScaleRetention → residual; LayerNorm → FFN → residual
//! - FFN: SwiGLU gated FFN: `down(SiLU(gate(x)) * up(x))`
//! - GroupNorm after retention output (across heads)

use burn::nn::{
    Embedding, EmbeddingConfig, GroupNorm, GroupNormConfig, LayerNorm, LayerNormConfig, Linear,
    LinearConfig,
};
use burn::prelude::*;
use burn::tensor::activation::silu;

/// RetNet model configuration
#[derive(Debug, Clone)]
pub struct RetNetConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Number of transformer-like layers
    pub num_layers: usize,
    /// Model (embedding) dimension
    pub d_model: usize,
    /// Number of retention heads
    pub num_heads: usize,
    /// FFN intermediate dimension (typically 4 * d_model)
    pub d_ffn: usize,
    /// Per-head key/query dimension (d_model / num_heads)
    pub qk_dim: usize,
    /// Per-head value dimension
    pub v_dim: usize,
}

impl RetNetConfig {
    /// Small configuration for testing (~1.3B-class proportions but tiny)
    pub fn tiny() -> Self {
        Self {
            vocab_size: 1000,
            num_layers: 2,
            d_model: 64,
            num_heads: 4,
            d_ffn: 256,
            qk_dim: 16, // d_model / num_heads
            v_dim: 16,
        }
    }

    /// RetNet-1.3B approximate configuration
    pub fn retnet_1_3b() -> Self {
        Self {
            vocab_size: 32000,
            num_layers: 24,
            d_model: 2048,
            num_heads: 8,
            d_ffn: 8192,
            qk_dim: 256, // 2048 / 8
            v_dim: 256,
        }
    }

    /// RetNet-6.7B approximate configuration
    pub fn retnet_6_7b() -> Self {
        Self {
            vocab_size: 32000,
            num_layers: 32,
            d_model: 4096,
            num_heads: 16,
            d_ffn: 16384,
            qk_dim: 256,
            v_dim: 256,
        }
    }

    /// Initialize the model and runtime
    pub fn init<B: Backend>(&self, device: &B::Device) -> (RetNet<B>, RetNetRuntime<B>) {
        let layers: Vec<RetNetLayer<B>> = (0..self.num_layers)
            .map(|_| RetNetLayer::new(self, device))
            .collect();

        let model = RetNet {
            embed_tokens: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            layers,
            ln_out: LayerNormConfig::new(self.d_model).init(device),
            lm_head: LinearConfig::new(self.d_model, self.vocab_size)
                .with_bias(false)
                .init(device),
        };

        let runtime = RetNetRuntime {
            config: self.clone(),
            _marker: std::marker::PhantomData,
        };

        (model, runtime)
    }
}

/// Multi-Scale Retention (MSR) module
///
/// Q/K/V projections with per-head decay scalars γ_h = 1 - 2^{-5-h}.
/// GroupNorm is applied after concatenating all heads' outputs.
#[derive(Module, Debug)]
pub struct MultiScaleRetention<B: Backend> {
    /// Q projection: d_model → num_heads * qk_dim
    pub q_proj: Linear<B>,
    /// K projection: d_model → num_heads * qk_dim
    pub k_proj: Linear<B>,
    /// V projection: d_model → num_heads * v_dim
    pub v_proj: Linear<B>,
    /// Output projection: num_heads * v_dim → d_model
    pub out_proj: Linear<B>,
    /// Group norm across heads (num_heads groups, each of size v_dim)
    pub group_norm: GroupNorm<B>,
    #[module(skip)]
    pub num_heads: usize,
    #[module(skip)]
    pub qk_dim: usize,
    #[module(skip)]
    pub v_dim: usize,
}

impl<B: Backend> MultiScaleRetention<B> {
    pub fn new(config: &RetNetConfig, device: &B::Device) -> Self {
        let inner_dim = config.num_heads * config.v_dim;

        Self {
            q_proj: LinearConfig::new(config.d_model, config.num_heads * config.qk_dim)
                .with_bias(false)
                .init(device),
            k_proj: LinearConfig::new(config.d_model, config.num_heads * config.qk_dim)
                .with_bias(false)
                .init(device),
            v_proj: LinearConfig::new(config.d_model, inner_dim)
                .with_bias(false)
                .init(device),
            out_proj: LinearConfig::new(inner_dim, config.d_model)
                .with_bias(false)
                .init(device),
            group_norm: GroupNormConfig::new(config.num_heads, inner_dim).init(device),
            num_heads: config.num_heads,
            qk_dim: config.qk_dim,
            v_dim: config.v_dim,
        }
    }

    /// Forward pass — recurrent mode.
    ///
    /// `state`: per-head retention state `[batch, num_heads, qk_dim, v_dim]`.
    /// When `state` is `Some`, it is updated in-place.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        state: Option<&mut RetNetLayerState<B>>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _] = x.dims();
        let device = x.device();

        // Project to Q, K, V
        // q, k: [batch, seq, num_heads * qk_dim]
        // v:    [batch, seq, num_heads * v_dim]
        let scale = (self.qk_dim as f64).powf(-0.5) as f32;
        let q = self.q_proj.forward(x.clone()) * scale;
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        // Reshape to per-head: [batch, seq, num_heads, dim]
        let q = q.reshape([batch, seq_len, self.num_heads, self.qk_dim]);
        let k = k.reshape([batch, seq_len, self.num_heads, self.qk_dim]);
        let v = v.reshape([batch, seq_len, self.num_heads, self.v_dim]);

        // Decay scalars γ_h = 1 - 2^{-5 - h} for h in 0..num_heads
        let gammas: Vec<f32> = (0..self.num_heads)
            .map(|h| 1.0 - 2.0_f32.powf(-5.0 - h as f32))
            .collect();

        // Initialize or retrieve retention state: [batch, num_heads, qk_dim, v_dim]
        let mut s = match &state {
            Some(ls) => ls.retention_state.clone(),
            None => Tensor::zeros([batch, self.num_heads, self.qk_dim, self.v_dim], &device),
        };

        // Process sequence step by step (recurrent mode)
        let mut outputs: Vec<Tensor<B, 4>> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            // Extract time step: [batch, num_heads, dim]
            let q_t = q
                .clone()
                .slice([0..batch, t..t + 1, 0..self.num_heads, 0..self.qk_dim])
                .squeeze_dim::<3>(1);
            let k_t = k
                .clone()
                .slice([0..batch, t..t + 1, 0..self.num_heads, 0..self.qk_dim])
                .squeeze_dim::<3>(1);
            let v_t = v
                .clone()
                .slice([0..batch, t..t + 1, 0..self.num_heads, 0..self.v_dim])
                .squeeze_dim::<3>(1);

            // Outer product: k_t ⊗ v_t → [batch, num_heads, qk_dim, v_dim]
            // k_t: [batch, num_heads, qk_dim] → unsqueeze → [batch, num_heads, qk_dim, 1]
            // v_t: [batch, num_heads, v_dim]  → unsqueeze → [batch, num_heads, 1, v_dim]
            let kv = k_t.unsqueeze_dim::<4>(3).matmul(v_t.unsqueeze_dim::<4>(2));

            // Apply per-head decay: γ_h * S_{t-1} + kv
            // Build gamma tensor [1, num_heads, 1, 1] for broadcasting
            let gamma_data: Vec<f32> = gammas.clone();
            let gamma_tensor: Tensor<B, 1> = Tensor::from_floats(gamma_data.as_slice(), &device);
            let gamma_tensor = gamma_tensor.reshape([1, self.num_heads, 1, 1]).expand([
                batch,
                self.num_heads,
                self.qk_dim,
                self.v_dim,
            ]);

            s = s * gamma_tensor + kv;

            // Compute output: q_t @ S_t → [batch, num_heads, v_dim]
            // q_t: [batch, num_heads, qk_dim] → unsqueeze → [batch, num_heads, 1, qk_dim]
            // s:   [batch, num_heads, qk_dim, v_dim]
            // result: [batch, num_heads, 1, v_dim] → squeeze → [batch, num_heads, v_dim]
            let y_t = q_t
                .unsqueeze_dim::<4>(2)
                .matmul(s.clone())
                .squeeze_dim::<3>(2);

            outputs.push(y_t.unsqueeze_dim::<4>(1));
        }

        // Update state
        if let Some(ls) = state {
            ls.retention_state = s;
        }

        // Concatenate along sequence dimension: [batch, seq, num_heads, v_dim]
        let output = Tensor::cat(outputs, 1);

        // Reshape to [batch, seq, num_heads * v_dim]
        let inner_dim = self.num_heads * self.v_dim;
        let output = output.reshape([batch, seq_len, inner_dim]);

        // GroupNorm expects [batch, channels, spatial...] format
        // Transpose: [batch, seq, channels] → [batch, channels, seq]
        let output = output.swap_dims(1, 2);
        let output = self.group_norm.forward(output);
        let output = output.swap_dims(1, 2); // Back to [batch, seq, channels]

        // Output projection
        self.out_proj.forward(output)
    }
}

/// SwiGLU Feed-Forward Network for RetNet
///
/// Computes: `down(SiLU(gate(x)) * up(x))`
#[derive(Module, Debug)]
pub struct RetNetFFN<B: Backend> {
    /// Gate projection: d_model → d_ffn
    pub gate_proj: Linear<B>,
    /// Up projection: d_model → d_ffn
    pub up_proj: Linear<B>,
    /// Down projection: d_ffn → d_model
    pub down_proj: Linear<B>,
}

impl<B: Backend> RetNetFFN<B> {
    pub fn new(config: &RetNetConfig, device: &B::Device) -> Self {
        Self {
            gate_proj: LinearConfig::new(config.d_model, config.d_ffn)
                .with_bias(false)
                .init(device),
            up_proj: LinearConfig::new(config.d_model, config.d_ffn)
                .with_bias(false)
                .init(device),
            down_proj: LinearConfig::new(config.d_ffn, config.d_model)
                .with_bias(false)
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}

/// Single RetNet layer
///
/// pre-norm → MSR → residual; pre-norm → FFN → residual
#[derive(Module, Debug)]
pub struct RetNetLayer<B: Backend> {
    /// Layer norm before retention
    pub ln1: LayerNorm<B>,
    /// Multi-scale retention
    pub retention: MultiScaleRetention<B>,
    /// Layer norm before FFN
    pub ln2: LayerNorm<B>,
    /// Feed-forward network
    pub ffn: RetNetFFN<B>,
}

impl<B: Backend> RetNetLayer<B> {
    pub fn new(config: &RetNetConfig, device: &B::Device) -> Self {
        Self {
            ln1: LayerNormConfig::new(config.d_model).init(device),
            retention: MultiScaleRetention::new(config, device),
            ln2: LayerNormConfig::new(config.d_model).init(device),
            ffn: RetNetFFN::new(config, device),
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        state: Option<&mut RetNetLayerState<B>>,
    ) -> Tensor<B, 3> {
        // MSR sub-layer
        let residual = x.clone();
        let x_norm = self.ln1.forward(x);
        let ret_out = self.retention.forward(x_norm, state);
        let x = residual + ret_out;

        // FFN sub-layer
        let residual = x.clone();
        let x_norm = self.ln2.forward(x);
        let ffn_out = self.ffn.forward(x_norm);
        residual + ffn_out
    }
}

/// RetNet model
#[derive(Module, Debug)]
pub struct RetNet<B: Backend> {
    /// Token embeddings
    pub embed_tokens: Embedding<B>,
    /// RetNet layers
    pub layers: Vec<RetNetLayer<B>>,
    /// Final layer norm
    pub ln_out: LayerNorm<B>,
    /// Language model head
    pub lm_head: Linear<B>,
}

/// Runtime configuration for RetNet (non-Module data)
pub struct RetNetRuntime<B: Backend> {
    pub config: RetNetConfig,
    pub _marker: std::marker::PhantomData<B>,
}

/// Recurrent state for a single RetNet layer
///
/// Holds the retention state matrix for each head.
#[derive(Clone, Debug)]
pub struct RetNetLayerState<B: Backend> {
    /// Retention state: [batch, num_heads, qk_dim, v_dim]
    pub retention_state: Tensor<B, 4>,
}

impl<B: Backend> RetNetLayerState<B> {
    pub fn new(config: &RetNetConfig, batch: usize, device: &B::Device) -> Self {
        Self {
            retention_state: Tensor::zeros(
                [batch, config.num_heads, config.qk_dim, config.v_dim],
                device,
            ),
        }
    }
}

/// Output from a RetNet forward pass
pub struct RetNetOutput<B: Backend> {
    /// Logits over vocabulary: [batch, seq_len, vocab_size]
    pub logits: Tensor<B, 3>,
    /// Final hidden states: [batch, seq_len, d_model]
    pub hidden_states: Tensor<B, 3>,
}

impl<B: Backend> RetNet<B> {
    /// Forward pass through the model
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        mut states: Option<&mut Vec<RetNetLayerState<B>>>,
    ) -> RetNetOutput<B> {
        let mut hidden_states = self.embed_tokens.forward(input_ids);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let state = states.as_mut().map(|s| &mut s[layer_idx]);
            hidden_states = layer.forward(hidden_states, state);
        }

        hidden_states = self.ln_out.forward(hidden_states);
        let logits = self.lm_head.forward(hidden_states.clone());

        RetNetOutput {
            logits,
            hidden_states,
        }
    }

    /// Initialize fresh recurrent states for inference
    pub fn init_states(
        &self,
        runtime: &RetNetRuntime<B>,
        batch: usize,
        device: &B::Device,
    ) -> Vec<RetNetLayerState<B>> {
        (0..runtime.config.num_layers)
            .map(|_| RetNetLayerState::new(&runtime.config, batch, device))
            .collect()
    }

    /// Generate text autoregressively using recurrent mode
    pub fn generate(
        &self,
        input_ids: Tensor<B, 2, Int>,
        runtime: &RetNetRuntime<B>,
        max_new_tokens: usize,
        sampler: &crate::sampling::SamplerConfig,
    ) -> Tensor<B, 2, Int> {
        let [batch, _] = input_ids.dims();
        let device = input_ids.device();

        // Track context tokens for repetition/DRY penalties
        let input_data: Vec<i64> = input_ids.to_data().to_vec().unwrap();
        let mut context_tokens: Vec<u32> = input_data.iter().map(|&id| id as u32).collect();

        // Initialize recurrent states
        let mut states = self.init_states(runtime, batch, &device);

        // Process prompt — updates states in-place
        let output = self.forward(input_ids.clone(), Some(&mut states));

        // Extract last-position logits for next-token prediction
        let [_, seq_len, vocab_size] = output.logits.dims();
        let mut last_logits = output
            .logits
            .slice([0..batch, seq_len - 1..seq_len, 0..vocab_size])
            .squeeze_dim::<2>(1);

        let mut all_tokens = input_ids;

        for _ in 0..max_new_tokens {
            let token_id =
                crate::sampling::sample_from_logits(last_logits, &context_tokens, sampler);
            context_tokens.push(token_id);

            let next_token = Tensor::<B, 2, Int>::from_ints([[token_id as i32]], &device);
            all_tokens = Tensor::cat(vec![all_tokens, next_token.clone()], 1);

            // Single-token forward pass with state
            let output = self.forward(next_token, Some(&mut states));
            last_logits = output.logits.squeeze_dim::<2>(1);
        }

        all_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Int;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_retnet_config() {
        let _ = RetNetConfig::tiny();
        let _ = RetNetConfig::retnet_1_3b();
        let _ = RetNetConfig::retnet_6_7b();
    }

    #[test]
    fn test_retnet_tiny_forward() {
        let device = Default::default();
        let config = RetNetConfig::tiny();
        let (model, _runtime) = config.init::<TestBackend>(&device);

        let input_ids = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device);
        let output = model.forward(input_ids, None);

        assert_eq!(output.logits.dims(), [1, 4, 1000]);
        assert_eq!(output.hidden_states.dims(), [1, 4, 64]);
    }

    #[test]
    fn test_retnet_with_state() {
        let device = Default::default();
        let config = RetNetConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        let mut states = model.init_states(&runtime, 1, &device);

        // First forward pass
        let input1 = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device);
        let _ = model.forward(input1, Some(&mut states));

        // Incremental single-token forward
        let input2 = Tensor::<TestBackend, 2, Int>::from_ints([[3]], &device);
        let output = model.forward(input2, Some(&mut states));

        assert_eq!(output.logits.dims(), [1, 1, 1000]);
    }

    #[test]
    fn test_retnet_generate() {
        let device = Default::default();
        let config = RetNetConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        let prompt = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device);
        let generated = model.generate(
            prompt,
            &runtime,
            3,
            &crate::sampling::SamplerConfig::greedy(),
        );

        assert_eq!(generated.dims(), [1, 5]);
    }

    #[test]
    fn test_multi_scale_retention() {
        let device = Default::default();
        let config = RetNetConfig::tiny();
        let msr = MultiScaleRetention::new(&config, &device);

        let x = Tensor::<TestBackend, 3>::zeros([1, 4, 64], &device);
        let output = msr.forward(x, None);

        assert_eq!(output.dims(), [1, 4, 64]);
    }

    #[test]
    fn test_retnet_ffn() {
        let device = Default::default();
        let config = RetNetConfig::tiny();
        let ffn = RetNetFFN::new(&config, &device);

        let x = Tensor::<TestBackend, 3>::zeros([1, 4, 64], &device);
        let output = ffn.forward(x);

        assert_eq!(output.dims(), [1, 4, 64]);
    }

    #[test]
    fn test_retnet_layer() {
        let device = Default::default();
        let config = RetNetConfig::tiny();
        let layer = RetNetLayer::new(&config, &device);

        let x = Tensor::<TestBackend, 3>::zeros([1, 4, 64], &device);
        let output = layer.forward(x, None);

        assert_eq!(output.dims(), [1, 4, 64]);
    }

    #[test]
    fn test_retnet_state_persistence() {
        let device = Default::default();
        let config = RetNetConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        // Check that the state is non-zero after a forward pass
        let mut states = model.init_states(&runtime, 1, &device);

        let input = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3]], &device);
        let _ = model.forward(input, Some(&mut states));

        // State should now be non-zero (retention memory is active)
        let state_sum: f32 = states[0]
            .retention_state
            .clone()
            .abs()
            .sum()
            .into_scalar()
            .elem();
        assert!(
            state_sum > 0.0,
            "state should be non-zero after forward pass"
        );
    }
}
