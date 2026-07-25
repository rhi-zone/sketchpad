//! Mixture of Experts (MoE) routing
//!
//! Provides sparse MoE layers as used in models like Mixtral, DeepSeek,
//! and other large-scale models. MoE allows scaling model capacity without
//! proportionally scaling compute by activating only a subset of experts
//! for each token.

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

use crate::glu::{SwiGluFfn, SwiGluFfnConfig};

/// MoE Router that selects experts for each token
///
/// Uses a learned linear projection followed by softmax to produce
/// routing weights, then selects top-k experts per token.
#[derive(Module, Debug)]
pub struct MoeRouter<B: Backend> {
    pub gate: Linear<B>,
    pub num_experts: usize,
    pub top_k: usize,
}

/// Configuration for MoeRouter
pub struct MoeRouterConfig {
    /// Hidden dimension
    pub hidden_size: usize,
    /// Total number of experts
    pub num_experts: usize,
    /// Number of experts to activate per token
    pub top_k: usize,
}

impl MoeRouterConfig {
    /// Creates a new router config
    pub fn new(hidden_size: usize, num_experts: usize, top_k: usize) -> Self {
        Self {
            hidden_size,
            num_experts,
            top_k,
        }
    }

    /// Initializes the router
    pub fn init<B: Backend>(&self, device: &B::Device) -> MoeRouter<B> {
        MoeRouter {
            gate: LinearConfig::new(self.hidden_size, self.num_experts)
                .with_bias(false)
                .init(device),
            num_experts: self.num_experts,
            top_k: self.top_k,
        }
    }
}

/// Routing decisions for a batch of tokens
#[derive(Debug, Clone)]
pub struct RoutingOutput<B: Backend> {
    /// Selected expert indices for each token: [batch * seq_len, top_k]
    pub expert_indices: Tensor<B, 2, Int>,
    /// Routing weights for selected experts: [batch * seq_len, top_k]
    pub routing_weights: Tensor<B, 2>,
}

impl<B: Backend> MoeRouter<B> {
    /// Computes routing decisions for input tokens
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor [batch, seq_len, hidden_size]
    ///
    /// # Returns
    ///
    /// Routing output containing expert indices and weights
    pub fn forward(&self, x: Tensor<B, 3>) -> RoutingOutput<B> {
        let [batch, seq_len, hidden_size] = x.dims();
        let num_tokens = batch * seq_len;

        // Flatten to [batch * seq_len, hidden_size]
        let x_flat = x.reshape([num_tokens, hidden_size]);

        // Compute router logits: [num_tokens, num_experts]
        let logits = self.gate.forward(x_flat);

        // Softmax over experts
        let probs = burn::tensor::activation::softmax(logits, 1);

        // Get top-k experts and their weights
        let (top_weights, top_indices) = probs.topk_with_indices(self.top_k, 1);

        // Normalize weights to sum to 1
        let weight_sum = top_weights.clone().sum_dim(1);
        let routing_weights = top_weights / weight_sum;

        RoutingOutput {
            expert_indices: top_indices,
            routing_weights,
        }
    }
}

/// Sparse Mixture of Experts FFN layer
///
/// Each token is processed by top-k experts, with outputs combined
/// according to router weights. This provides model capacity scaling
/// without proportional compute scaling.
///
/// # Architecture
///
/// ```text
/// 1. Router computes expert assignments for each token
/// 2. Each expert (SwiGLU FFN) processes assigned tokens
/// 3. Outputs are combined weighted by routing probabilities
/// ```
///
/// # References
///
/// - [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
/// - [Switch Transformers](https://arxiv.org/abs/2101.03961)
#[derive(Module, Debug)]
pub struct SparseMoeFfn<B: Backend> {
    pub router: MoeRouter<B>,
    pub experts: Vec<SwiGluFfn<B>>,
    pub num_experts: usize,
    pub top_k: usize,
}

/// Configuration for SparseMoeFfn
pub struct SparseMoeFfnConfig {
    /// Hidden dimension
    pub hidden_size: usize,
    /// Intermediate FFN dimension per expert
    pub intermediate_size: usize,
    /// Total number of experts
    pub num_experts: usize,
    /// Number of experts to activate per token
    pub top_k: usize,
}

impl SparseMoeFfnConfig {
    /// Creates a new config (Mixtral-style: 8 experts, top-2)
    pub fn mixtral(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            num_experts: 8,
            top_k: 2,
        }
    }

    /// Creates a custom config
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        num_experts: usize,
        top_k: usize,
    ) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            num_experts,
            top_k,
        }
    }

    /// Initializes the MoE layer
    pub fn init<B: Backend>(&self, device: &B::Device) -> SparseMoeFfn<B> {
        let router =
            MoeRouterConfig::new(self.hidden_size, self.num_experts, self.top_k).init(device);

        let experts = (0..self.num_experts)
            .map(|_| SwiGluFfnConfig::new(self.hidden_size, self.intermediate_size).init(device))
            .collect();

        SparseMoeFfn {
            router,
            experts,
            num_experts: self.num_experts,
            top_k: self.top_k,
        }
    }
}

impl<B: Backend> SparseMoeFfn<B> {
    /// Forward pass through the MoE layer
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor [batch, seq_len, hidden_size]
    ///
    /// # Returns
    ///
    /// Output tensor [batch, seq_len, hidden_size]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, hidden_size] = x.dims();
        let num_tokens = batch * seq_len;
        let device = x.device();

        // Get routing decisions
        let routing = self.router.forward(x.clone());

        // Flatten input for processing
        let x_flat = x.reshape([num_tokens, hidden_size]);

        // Initialize output accumulator
        let mut output = Tensor::zeros([num_tokens, hidden_size], &device);

        // Process each expert
        for expert_idx in 0..self.num_experts {
            // Find tokens assigned to this expert
            let expert_mask =
                self.compute_expert_mask(&routing.expert_indices, expert_idx, num_tokens, &device);

            // Skip if no tokens assigned
            if !self.has_assigned_tokens(&expert_mask) {
                continue;
            }

            // Get weights for this expert
            let expert_weights = self.get_expert_weights(
                &routing.expert_indices,
                &routing.routing_weights,
                expert_idx,
                num_tokens,
                &device,
            );

            // Process all tokens through expert (masked by weight)
            let expert_input = x_flat.clone().reshape([num_tokens, 1, hidden_size]);
            let expert_out = self.experts[expert_idx].forward(expert_input);
            let expert_out = expert_out.reshape([num_tokens, hidden_size]);

            // Weight and accumulate
            let weighted_out = expert_out * expert_weights.unsqueeze_dim(1);
            output = output + weighted_out;
        }

        output.reshape([batch, seq_len, hidden_size])
    }

    /// Computes a mask indicating which tokens are assigned to an expert
    fn compute_expert_mask(
        &self,
        expert_indices: &Tensor<B, 2, Int>,
        expert_idx: usize,
        num_tokens: usize,
        device: &B::Device,
    ) -> Tensor<B, 1, Bool> {
        // Check if expert_idx appears in any of the top-k slots
        let expert_tensor = Tensor::<B, 2, Int>::from_ints([[expert_idx as i32; 1]; 1], device)
            .repeat_dim(0, num_tokens)
            .repeat_dim(1, self.top_k);

        let matches = expert_indices.clone().equal(expert_tensor);
        matches.any_dim(1).squeeze_dim::<1>(1)
    }

    /// Checks if any tokens are assigned to an expert
    fn has_assigned_tokens(&self, mask: &Tensor<B, 1, Bool>) -> bool {
        // Convert to int and sum - if > 0, there are assigned tokens
        let count: i32 = mask.clone().int().sum().into_scalar().elem();
        count > 0
    }

    /// Gets routing weights for a specific expert
    fn get_expert_weights(
        &self,
        expert_indices: &Tensor<B, 2, Int>,
        routing_weights: &Tensor<B, 2>,
        expert_idx: usize,
        num_tokens: usize,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let expert_tensor = Tensor::<B, 2, Int>::from_ints([[expert_idx as i32; 1]; 1], device)
            .repeat_dim(0, num_tokens)
            .repeat_dim(1, self.top_k);

        // Create mask where expert matches
        let matches = expert_indices.clone().equal(expert_tensor);
        let matches_float = matches.float();

        // Extract weights where expert matches (sum across top_k dimension)
        (routing_weights.clone() * matches_float)
            .sum_dim(1)
            .squeeze_dim::<1>(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_router_output_shape() {
        let device = Default::default();
        let config = MoeRouterConfig::new(256, 8, 2);
        let router = config.init::<TestBackend>(&device);

        let x = Tensor::zeros([2, 16, 256], &device);
        let routing = router.forward(x);

        assert_eq!(routing.expert_indices.dims(), [32, 2]); // batch * seq_len, top_k
        assert_eq!(routing.routing_weights.dims(), [32, 2]);
    }

    #[test]
    fn test_routing_weights_sum_to_one() {
        let device = Default::default();
        let config = MoeRouterConfig::new(128, 4, 2);
        let router = config.init::<TestBackend>(&device);

        let x = Tensor::ones([1, 8, 128], &device);
        let routing = router.forward(x);

        let sums: Vec<f32> = routing
            .routing_weights
            .sum_dim(1)
            .squeeze_dim::<1>(1)
            .into_data()
            .to_vec()
            .unwrap();

        for sum in sums {
            assert!((sum - 1.0).abs() < 1e-5, "Routing weights should sum to 1");
        }
    }

    #[test]
    fn test_sparse_moe_shape() {
        let device = Default::default();
        let config = SparseMoeFfnConfig::new(128, 256, 4, 2);
        let moe = config.init::<TestBackend>(&device);

        let x = Tensor::zeros([2, 8, 128], &device);
        let y = moe.forward(x);

        assert_eq!(y.dims(), [2, 8, 128]);
    }

    #[test]
    fn test_mixtral_config() {
        let device = Default::default();
        let config = SparseMoeFfnConfig::mixtral(4096, 14336);
        let moe = config.init::<TestBackend>(&device);

        assert_eq!(moe.num_experts, 8);
        assert_eq!(moe.top_k, 2);
    }
}
