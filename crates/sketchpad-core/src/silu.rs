use burn::prelude::*;
use burn::tensor::activation::sigmoid;

/// SiLU (Sigmoid Linear Unit) activation: x * sigmoid(x)
pub fn silu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    x.clone() * sigmoid(x)
}
