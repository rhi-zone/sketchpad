//! DeepSeek Model Implementation
//!
//! DeepSeek is a family of large language models from DeepSeek AI.
//!
//! # Architecture
//!
//! DeepSeek V2/V3 features Multi-head Latent Attention (MLA):
//! - Low-rank compression for KV cache
//! - Reduces memory usage during inference
//! - RoPE applied to compressed keys
//!
//! DeepSeek V1 uses standard transformer attention similar to LLaMA.
//!
//! # Supported Variants
//!
//! - DeepSeek V1 (7B, 67B)
//! - DeepSeek V2 (16B, 236B) with MLA
//! - DeepSeek V2.5
//! - DeepSeek V3 (671B) with MLA + MoE

use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;

use sketchpad_core::glu::{SwiGluFfn, SwiGluFfnConfig};
use sketchpad_core::kv_cache::{AttentionCache, CompressedKvCache, KvCacheConfig, ModelKvCache};
use sketchpad_core::rmsnorm::RmsNorm;
use sketchpad_core::rope::RotaryEmbedding;
use sketchpad_core::transformer::causal_mask;

/// DeepSeek Model Configuration
#[derive(Debug, Clone)]
pub struct DeepSeekConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Intermediate (FFN) dimension
    pub intermediate_size: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of KV heads (for GQA)
    pub num_kv_heads: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// RMSNorm epsilon
    pub norm_eps: f64,
    /// RoPE base frequency
    pub rope_base: f32,
    /// Use Multi-head Latent Attention (V2/V3)
    pub use_mla: bool,
    /// Q LoRA rank for MLA (0 = no compression)
    pub q_lora_rank: usize,
    /// KV LoRA rank for MLA (latent dimension)
    pub kv_lora_rank: usize,
    /// Nope head dimension (non-RoPE portion of attention)
    pub qk_nope_head_dim: usize,
    /// RoPE head dimension
    pub qk_rope_head_dim: usize,
}

impl DeepSeekConfig {
    /// DeepSeek V1 7B configuration (standard attention)
    pub fn deepseek_v1_7b() -> Self {
        Self {
            vocab_size: 102400,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_layers: 30,
            num_heads: 32,
            num_kv_heads: 32,
            max_seq_len: 4096,
            norm_eps: 1e-6,
            rope_base: 10000.0,
            use_mla: false,
            q_lora_rank: 0,
            kv_lora_rank: 0,
            qk_nope_head_dim: 0,
            qk_rope_head_dim: 0,
        }
    }

    /// DeepSeek V2 Lite 16B configuration (MLA)
    pub fn deepseek_v2_lite() -> Self {
        Self {
            vocab_size: 102400,
            hidden_size: 2048,
            intermediate_size: 10944,
            num_layers: 27,
            num_heads: 16,
            num_kv_heads: 16,
            max_seq_len: 32768,
            norm_eps: 1e-6,
            rope_base: 10000.0,
            use_mla: true,
            q_lora_rank: 1536,
            kv_lora_rank: 512,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
        }
    }

    /// DeepSeek V2 236B configuration (MLA)
    pub fn deepseek_v2() -> Self {
        Self {
            vocab_size: 102400,
            hidden_size: 5120,
            intermediate_size: 12288,
            num_layers: 60,
            num_heads: 128,
            num_kv_heads: 128,
            max_seq_len: 32768,
            norm_eps: 1e-6,
            rope_base: 10000.0,
            use_mla: true,
            q_lora_rank: 1536,
            kv_lora_rank: 512,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
        }
    }

    /// Creates a tiny model for testing
    pub fn tiny() -> Self {
        Self {
            vocab_size: 1000,
            hidden_size: 128,
            intermediate_size: 256,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            max_seq_len: 512,
            norm_eps: 1e-6,
            rope_base: 10000.0,
            use_mla: false,
            q_lora_rank: 0,
            kv_lora_rank: 0,
            qk_nope_head_dim: 0,
            qk_rope_head_dim: 0,
        }
    }

    /// Creates a tiny MLA model for testing
    pub fn tiny_mla() -> Self {
        Self {
            vocab_size: 1000,
            hidden_size: 128,
            intermediate_size: 256,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            max_seq_len: 512,
            norm_eps: 1e-6,
            rope_base: 10000.0,
            use_mla: true,
            q_lora_rank: 64,
            kv_lora_rank: 32,
            qk_nope_head_dim: 16,
            qk_rope_head_dim: 16,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> (DeepSeek<B>, DeepSeekRuntime<B>) {
        let head_dim = if self.use_mla {
            self.qk_nope_head_dim + self.qk_rope_head_dim
        } else {
            self.hidden_size / self.num_heads
        };

        let layers: Vec<DeepSeekLayer<B>> = (0..self.num_layers)
            .map(|_| self.init_layer(device))
            .collect();

        let model = DeepSeek {
            embed_tokens: EmbeddingConfig::new(self.vocab_size, self.hidden_size).init(device),
            layers,
            norm: RmsNorm::with_eps(self.hidden_size, self.norm_eps, device),
            lm_head: LinearConfig::new(self.hidden_size, self.vocab_size)
                .with_bias(false)
                .init(device),
        };

        let rope_dim = if self.use_mla {
            self.qk_rope_head_dim
        } else {
            head_dim
        };

        let runtime = DeepSeekRuntime {
            rope: RotaryEmbedding::with_base(rope_dim, self.max_seq_len, self.rope_base, device),
            config: self.clone(),
        };

        (model, runtime)
    }

    fn init_layer<B: Backend>(&self, device: &B::Device) -> DeepSeekLayer<B> {
        let attention = if self.use_mla {
            self.init_mla_attention(device)
        } else {
            self.init_standard_attention(device)
        };

        DeepSeekLayer {
            attention,
            ffn: SwiGluFfnConfig::new(self.hidden_size, self.intermediate_size).init(device),
            input_norm: RmsNorm::with_eps(self.hidden_size, self.norm_eps, device),
            post_attention_norm: RmsNorm::with_eps(self.hidden_size, self.norm_eps, device),
        }
    }

    fn init_standard_attention<B: Backend>(&self, device: &B::Device) -> DeepSeekAttention<B> {
        let head_dim = self.hidden_size / self.num_heads;
        let kv_dim = head_dim * self.num_kv_heads;

        DeepSeekAttention {
            q_proj: LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(false)
                .init(device),
            k_proj: LinearConfig::new(self.hidden_size, kv_dim)
                .with_bias(false)
                .init(device),
            v_proj: LinearConfig::new(self.hidden_size, kv_dim)
                .with_bias(false)
                .init(device),
            o_proj: LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(false)
                .init(device),
            q_a_proj: None,
            q_b_proj: None,
            kv_a_proj_with_mqa: None,
            kv_b_proj: None,
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim,
            qk_nope_head_dim: 0,
            qk_rope_head_dim: head_dim,
            use_mla: false,
            kv_lora_rank: 0,
        }
    }

    fn init_mla_attention<B: Backend>(&self, device: &B::Device) -> DeepSeekAttention<B> {
        let head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim;
        let q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim;
        let v_head_dim = self.qk_nope_head_dim; // V uses only nope dim in MLA

        // For MLA:
        // q_a_proj: hidden -> q_lora_rank (compression)
        // q_b_proj: q_lora_rank -> num_heads * q_head_dim (expansion)
        // kv_a_proj_with_mqa: hidden -> kv_lora_rank + qk_rope_head_dim (compressed KV + rope keys)
        // kv_b_proj: kv_lora_rank -> num_heads * (qk_nope_head_dim + v_head_dim) (expansion)

        DeepSeekAttention {
            q_proj: LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(false)
                .init(device), // Placeholder, not used with MLA
            k_proj: LinearConfig::new(
                self.hidden_size,
                self.hidden_size / self.num_heads * self.num_kv_heads,
            )
            .with_bias(false)
            .init(device), // Placeholder
            v_proj: LinearConfig::new(
                self.hidden_size,
                self.hidden_size / self.num_heads * self.num_kv_heads,
            )
            .with_bias(false)
            .init(device), // Placeholder
            o_proj: LinearConfig::new(self.num_heads * v_head_dim, self.hidden_size)
                .with_bias(false)
                .init(device),
            q_a_proj: Some(
                LinearConfig::new(self.hidden_size, self.q_lora_rank)
                    .with_bias(false)
                    .init(device),
            ),
            q_b_proj: Some(
                LinearConfig::new(self.q_lora_rank, self.num_heads * q_head_dim)
                    .with_bias(false)
                    .init(device),
            ),
            kv_a_proj_with_mqa: Some(
                LinearConfig::new(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim)
                    .with_bias(false)
                    .init(device),
            ),
            kv_b_proj: Some(
                LinearConfig::new(
                    self.kv_lora_rank,
                    self.num_heads * (self.qk_nope_head_dim + v_head_dim),
                )
                .with_bias(false)
                .init(device),
            ),
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim,
            qk_nope_head_dim: self.qk_nope_head_dim,
            qk_rope_head_dim: self.qk_rope_head_dim,
            use_mla: true,
            kv_lora_rank: self.kv_lora_rank,
        }
    }
}

/// DeepSeek attention (supports both standard and MLA)
#[derive(Module, Debug)]
pub struct DeepSeekAttention<B: Backend> {
    // Standard attention projections
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    // MLA projections (optional)
    pub q_a_proj: Option<Linear<B>>,
    pub q_b_proj: Option<Linear<B>>,
    pub kv_a_proj_with_mqa: Option<Linear<B>>,
    pub kv_b_proj: Option<Linear<B>>,
    // Config
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    #[module(skip)]
    pub use_mla: bool,
    #[module(skip)]
    pub kv_lora_rank: usize,
}

impl<B: Backend> DeepSeekAttention<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        start_pos: usize,
        mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        if self.use_mla {
            self.forward_mla(x, rope, start_pos, mask)
        } else {
            self.forward_standard(x, rope, start_pos, mask)
        }
    }

    pub fn forward_cached(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        start_pos: usize,
        mask: Option<Tensor<B, 2>>,
        cache: &mut dyn AttentionCache<B>,
        layer_idx: usize,
    ) -> Tensor<B, 3> {
        if self.use_mla {
            self.forward_mla_cached(x, rope, start_pos, mask, cache, layer_idx)
        } else {
            self.forward_standard_cached(x, rope, start_pos, mask, cache, layer_idx)
        }
    }

    fn forward_standard(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        start_pos: usize,
        mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden] = x.dims();

        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        let q = q
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        let (q, k) = rope.forward(q, k, start_pos);

        let k = self.repeat_kv(k);
        let v = self.repeat_kv(v);

        let scale = (self.head_dim as f64).powf(-0.5);
        let attn = q.matmul(k.transpose()) * scale;

        let attn = match mask {
            Some(m) => attn + m.unsqueeze::<3>().unsqueeze(),
            None => attn,
        };

        let attn = burn::tensor::activation::softmax(attn, 3);
        let out = attn.matmul(v);

        let out = out
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * self.head_dim]);
        self.o_proj.forward(out)
    }

    fn forward_standard_cached(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        start_pos: usize,
        mask: Option<Tensor<B, 2>>,
        cache: &mut dyn AttentionCache<B>,
        layer_idx: usize,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden] = x.dims();

        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        let q = q
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        let (q, k) = rope.forward(q, k, start_pos);

        let update = cache.update_layer(layer_idx, k, v);
        let k = self.repeat_kv(update.k);
        let v = self.repeat_kv(update.v);

        let total_len = k.dims()[2];
        let scale = (self.head_dim as f64).powf(-0.5);
        let attn = q.matmul(k.transpose()) * scale;

        let attn = match mask {
            Some(m) => {
                // mask shape is [seq_len, total_len]; unsqueeze for [1, 1, seq_len, total_len]
                let _ = total_len;
                attn + m.unsqueeze::<3>().unsqueeze()
            }
            None => attn,
        };

        let attn = burn::tensor::activation::softmax(attn, 3);
        let out = attn.matmul(v);

        let out = out
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * self.head_dim]);
        self.o_proj.forward(out)
    }

    fn forward_mla(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        start_pos: usize,
        mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden] = x.dims();
        let q_a = self.q_a_proj.as_ref().unwrap();
        let q_b = self.q_b_proj.as_ref().unwrap();
        let kv_a = self.kv_a_proj_with_mqa.as_ref().unwrap();
        let kv_b = self.kv_b_proj.as_ref().unwrap();

        // Q: compress -> expand
        let q_compressed = q_a.forward(x.clone());
        let q = q_b.forward(q_compressed);

        // KV: compress with rope keys, then expand
        let kv_compressed = kv_a.forward(x);

        // Split compressed KV into latent part and rope keys
        let latent = kv_compressed
            .clone()
            .slice([0..batch, 0..seq_len, 0..self.kv_lora_rank]);
        let k_rope = kv_compressed.slice([
            0..batch,
            0..seq_len,
            self.kv_lora_rank..self.kv_lora_rank + self.qk_rope_head_dim,
        ]);

        // Expand latent to get K (nope) and V
        let kv_expanded = kv_b.forward(latent);

        // Split Q into nope and rope parts
        let q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim;
        let q = q.reshape([batch, seq_len, self.num_heads, q_head_dim]);
        let q_nope = q.clone().slice([
            0..batch,
            0..seq_len,
            0..self.num_heads,
            0..self.qk_nope_head_dim,
        ]);
        let q_rope = q.slice([
            0..batch,
            0..seq_len,
            0..self.num_heads,
            self.qk_nope_head_dim..q_head_dim,
        ]);

        // Split expanded KV
        let v_head_dim = self.qk_nope_head_dim;
        let kv_per_head = self.qk_nope_head_dim + v_head_dim;
        let kv_expanded = kv_expanded.reshape([batch, seq_len, self.num_heads, kv_per_head]);
        let k_nope = kv_expanded.clone().slice([
            0..batch,
            0..seq_len,
            0..self.num_heads,
            0..self.qk_nope_head_dim,
        ]);
        let v = kv_expanded.slice([
            0..batch,
            0..seq_len,
            0..self.num_heads,
            self.qk_nope_head_dim..kv_per_head,
        ]);

        // Apply RoPE to rope parts
        // Reshape for RoPE: [batch, seq, heads, dim] -> [batch, heads, seq, dim]
        let q_rope = q_rope.swap_dims(1, 2);
        let k_rope = k_rope
            .unsqueeze_dim::<4>(2)
            .repeat_dim(2, self.num_heads)
            .swap_dims(1, 2);

        let (q_rope, k_rope) = rope.forward(q_rope, k_rope, start_pos);

        // Reshape back and concatenate
        let q_rope = q_rope.swap_dims(1, 2);
        let k_rope = k_rope.swap_dims(1, 2);

        let q = Tensor::cat(vec![q_nope, q_rope], 3);
        let k = Tensor::cat(vec![k_nope, k_rope], 3);

        // Standard attention
        let q = q.swap_dims(1, 2); // [batch, heads, seq, head_dim]
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        let scale = (self.head_dim as f64).powf(-0.5);
        let attn = q.matmul(k.transpose()) * scale;

        let attn = match mask {
            Some(m) => attn + m.unsqueeze::<3>().unsqueeze(),
            None => attn,
        };

        let attn = burn::tensor::activation::softmax(attn, 3);
        let out = attn.matmul(v);

        let out = out
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * v_head_dim]);
        self.o_proj.forward(out)
    }

    fn forward_mla_cached(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        start_pos: usize,
        mask: Option<Tensor<B, 2>>,
        cache: &mut dyn AttentionCache<B>,
        layer_idx: usize,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden] = x.dims();
        let q_a = self.q_a_proj.as_ref().unwrap();
        let q_b = self.q_b_proj.as_ref().unwrap();
        let kv_a = self.kv_a_proj_with_mqa.as_ref().unwrap();
        let kv_b = self.kv_b_proj.as_ref().unwrap();

        // Q: compress -> expand
        let q_compressed = q_a.forward(x.clone());
        let q = q_b.forward(q_compressed);

        // KV: compress with rope keys, then expand
        let kv_compressed = kv_a.forward(x);

        // Split compressed KV into latent part and rope keys
        let latent = kv_compressed
            .clone()
            .slice([0..batch, 0..seq_len, 0..self.kv_lora_rank]);
        let k_rope = kv_compressed.slice([
            0..batch,
            0..seq_len,
            self.kv_lora_rank..self.kv_lora_rank + self.qk_rope_head_dim,
        ]);

        // Expand latent to get K (nope) and V
        let kv_expanded = kv_b.forward(latent);

        // Split Q into nope and rope parts
        let q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim;
        let q = q.reshape([batch, seq_len, self.num_heads, q_head_dim]);
        let q_nope = q.clone().slice([
            0..batch,
            0..seq_len,
            0..self.num_heads,
            0..self.qk_nope_head_dim,
        ]);
        let q_rope = q.slice([
            0..batch,
            0..seq_len,
            0..self.num_heads,
            self.qk_nope_head_dim..q_head_dim,
        ]);

        // Split expanded KV
        let v_head_dim = self.qk_nope_head_dim;
        let kv_per_head = self.qk_nope_head_dim + v_head_dim;
        let kv_expanded = kv_expanded.reshape([batch, seq_len, self.num_heads, kv_per_head]);
        let k_nope = kv_expanded.clone().slice([
            0..batch,
            0..seq_len,
            0..self.num_heads,
            0..self.qk_nope_head_dim,
        ]);
        let v = kv_expanded.slice([
            0..batch,
            0..seq_len,
            0..self.num_heads,
            self.qk_nope_head_dim..kv_per_head,
        ]);

        // Apply RoPE to rope parts
        let q_rope = q_rope.swap_dims(1, 2);
        let k_rope = k_rope
            .unsqueeze_dim::<4>(2)
            .repeat_dim(2, self.num_heads)
            .swap_dims(1, 2);

        let (q_rope, k_rope) = rope.forward(q_rope, k_rope, start_pos);

        let q_rope = q_rope.swap_dims(1, 2);
        let k_rope = k_rope.swap_dims(1, 2);

        let q = Tensor::cat(vec![q_nope, q_rope], 3);
        let k_new = Tensor::cat(vec![k_nope, k_rope], 3);

        // Reshape to [batch, heads, seq_len, head_dim] for cache update
        let k_new = k_new.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        let update = cache.update_layer(layer_idx, k_new, v);
        let k = update.k;
        let v = update.v;

        // Standard attention
        let q = q.swap_dims(1, 2); // [batch, heads, seq_len, head_dim]

        let scale = (self.head_dim as f64).powf(-0.5);
        let attn = q.matmul(k.transpose()) * scale;

        let attn = match mask {
            Some(m) => attn + m.unsqueeze::<3>().unsqueeze(),
            None => attn,
        };

        let attn = burn::tensor::activation::softmax(attn, 3);
        let out = attn.matmul(v);

        let out = out
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * v_head_dim]);
        self.o_proj.forward(out)
    }

    fn repeat_kv(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        if self.num_kv_heads == self.num_heads {
            return x;
        }

        let [batch, kv_heads, seq_len, head_dim] = x.dims();
        let n_rep = self.num_heads / self.num_kv_heads;

        x.unsqueeze_dim::<5>(2).repeat_dim(2, n_rep).reshape([
            batch,
            kv_heads * n_rep,
            seq_len,
            head_dim,
        ])
    }
}

/// DeepSeek transformer layer
#[derive(Module, Debug)]
pub struct DeepSeekLayer<B: Backend> {
    pub attention: DeepSeekAttention<B>,
    pub ffn: SwiGluFfn<B>,
    pub input_norm: RmsNorm<B>,
    pub post_attention_norm: RmsNorm<B>,
}

impl<B: Backend> DeepSeekLayer<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        start_pos: usize,
        mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        // Pre-norm attention
        let normed = self.input_norm.forward(x.clone());
        let h = x + self.attention.forward(normed, rope, start_pos, mask);

        // Pre-FFN norm
        let normed = self.post_attention_norm.forward(h.clone());
        h + self.ffn.forward(normed)
    }

    pub fn forward_cached(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        start_pos: usize,
        mask: Option<Tensor<B, 2>>,
        cache: &mut dyn AttentionCache<B>,
        layer_idx: usize,
    ) -> Tensor<B, 3> {
        // Pre-norm attention with cache
        let normed = self.input_norm.forward(x.clone());
        let h = x + self
            .attention
            .forward_cached(normed, rope, start_pos, mask, cache, layer_idx);

        // Pre-FFN norm
        let normed = self.post_attention_norm.forward(h.clone());
        h + self.ffn.forward(normed)
    }
}

/// DeepSeek Model
#[derive(Module, Debug)]
pub struct DeepSeek<B: Backend> {
    pub embed_tokens: Embedding<B>,
    pub layers: Vec<DeepSeekLayer<B>>,
    pub norm: RmsNorm<B>,
    pub lm_head: Linear<B>,
}

/// Runtime state for DeepSeek
pub struct DeepSeekRuntime<B: Backend> {
    pub rope: RotaryEmbedding<B>,
    pub config: DeepSeekConfig,
}

impl<B: Backend> DeepSeekRuntime<B> {
    /// Create a KV cache according to the given config
    pub fn create_kv_cache(&self, config: &KvCacheConfig) -> Box<dyn AttentionCache<B>> {
        let head_dim = self.config.hidden_size / self.config.num_heads;
        match config {
            KvCacheConfig::Standard => Box::new(ModelKvCache::<B>::new(
                self.config.num_layers,
                self.config.max_seq_len,
            )),
            KvCacheConfig::Compressed { method } => Box::new(CompressedKvCache::<B>::new(
                self.config.num_layers,
                self.config.num_kv_heads,
                head_dim,
                method.clone(),
            )),
        }
    }
}

/// Output from the DeepSeek model
pub struct DeepSeekOutput<B: Backend> {
    pub logits: Tensor<B, 3>,
    pub hidden_states: Tensor<B, 3>,
}

impl<B: Backend> DeepSeek<B> {
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        runtime: &DeepSeekRuntime<B>,
        cache: Option<&mut dyn AttentionCache<B>>,
    ) -> DeepSeekOutput<B> {
        let [_batch, seq_len] = input_ids.dims();
        let device = input_ids.device();

        let mut hidden_states = self.embed_tokens.forward(input_ids);

        match cache {
            Some(cache) => {
                let start_pos = cache.seq_len();

                let mask = if seq_len > 1 {
                    let total_len = start_pos + seq_len;
                    Some(prefill_causal_mask::<B>(seq_len, total_len, &device))
                } else {
                    None
                };

                for (layer_idx, layer) in self.layers.iter().enumerate() {
                    hidden_states = layer.forward_cached(
                        hidden_states,
                        &runtime.rope,
                        start_pos,
                        mask.clone(),
                        cache,
                        layer_idx,
                    );
                }
            }
            None => {
                let mask = if seq_len > 1 {
                    Some(causal_mask::<B>(seq_len, &device))
                } else {
                    None
                };

                for layer in &self.layers {
                    hidden_states = layer.forward(hidden_states, &runtime.rope, 0, mask.clone());
                }
            }
        }

        hidden_states = self.norm.forward(hidden_states);

        let logits = self.lm_head.forward(hidden_states.clone());

        DeepSeekOutput {
            logits,
            hidden_states,
        }
    }

    /// Generate text autoregressively with KV caching
    ///
    /// Uses incremental generation: the prompt is processed in one pass (prefill),
    /// then each subsequent token is generated by only computing the new token
    /// against cached K/V from previous steps.
    pub fn generate(
        &self,
        input_ids: Tensor<B, 2, Int>,
        runtime: &DeepSeekRuntime<B>,
        max_new_tokens: usize,
        cache: &mut dyn AttentionCache<B>,
        sampler: &crate::sampling::SamplerConfig,
    ) -> Tensor<B, 2, Int> {
        let [batch, _prompt_len] = input_ids.dims();

        // Track generated token IDs for repetition/DRY penalties
        let input_data: Vec<i64> = input_ids.to_data().to_vec().unwrap();
        let mut context_tokens: Vec<u32> = input_data.iter().map(|&id| id as u32).collect();

        // Prefill: process the entire prompt at once
        let output = self.forward(input_ids.clone(), runtime, Some(cache));

        let seq_len = input_ids.dims()[1];
        let last_logits = output.logits.slice([
            0..batch,
            (seq_len - 1)..seq_len,
            0..runtime.config.vocab_size,
        ]);
        let last_logits = last_logits.reshape([batch, runtime.config.vocab_size]);
        let token_id = crate::sampling::sample_from_logits(last_logits, &context_tokens, sampler);
        context_tokens.push(token_id);

        let device = input_ids.device();
        let mut next_token = Tensor::<B, 2, Int>::from_ints([[token_id as i32]], &device);
        let mut all_tokens = Tensor::cat(vec![input_ids, next_token.clone()], 1);

        // Decode: generate one token at a time using the cache
        for _ in 1..max_new_tokens {
            let output = self.forward(next_token, runtime, Some(cache));

            let last_logits = output
                .logits
                .slice([0..batch, 0..1, 0..runtime.config.vocab_size]);
            let last_logits = last_logits.reshape([batch, runtime.config.vocab_size]);
            let token_id =
                crate::sampling::sample_from_logits(last_logits, &context_tokens, sampler);
            context_tokens.push(token_id);

            next_token = Tensor::<B, 2, Int>::from_ints([[token_id as i32]], &device);
            all_tokens = Tensor::cat(vec![all_tokens, next_token.clone()], 1);
        }

        all_tokens
    }
}

/// Creates a causal mask for prefill with cached positions
///
/// When there are already `start_pos` cached tokens, new tokens at positions
/// `[start_pos, start_pos + new_len)` should be able to attend to all positions
/// `[0, start_pos + new_len)` but not to future positions within the new chunk.
///
/// Returns a mask of shape [new_len, total_len] where total_len = start_pos + new_len.
fn prefill_causal_mask<B: Backend>(
    new_len: usize,
    total_len: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let start_pos = total_len - new_len;
    let mut mask_data = vec![0.0f32; new_len * total_len];
    for i in 0..new_len {
        let global_pos = start_pos + i;
        for j in (global_pos + 1)..total_len {
            mask_data[i * total_len + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::<B, 1>::from_floats(mask_data.as_slice(), device).reshape([new_len, total_len])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_deepseek_tiny_forward() {
        let device = Default::default();
        let config = DeepSeekConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        let input_ids = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device);
        let output = model.forward(input_ids, &runtime, None);

        assert_eq!(output.logits.dims(), [1, 4, 1000]);
        assert_eq!(output.hidden_states.dims(), [1, 4, 128]);
    }

    #[test]
    fn test_deepseek_generate() {
        use sketchpad_core::kv_cache::KvCacheConfig;

        let device = Default::default();
        let config = DeepSeekConfig::tiny();
        let (model, runtime) = config.init::<TestBackend>(&device);

        let prompt = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device);
        let mut cache = runtime.create_kv_cache(&KvCacheConfig::Standard);
        let generated = model.generate(
            prompt,
            &runtime,
            3,
            cache.as_mut(),
            &crate::sampling::SamplerConfig::greedy(),
        );

        assert_eq!(generated.dims(), [1, 5]);
    }

    #[test]
    fn test_deepseek_configs() {
        let _ = DeepSeekConfig::deepseek_v1_7b();
        let _ = DeepSeekConfig::deepseek_v2_lite();
        let _ = DeepSeekConfig::deepseek_v2();
    }

    #[test]
    fn test_deepseek_mla_tiny_forward() {
        let device = Default::default();
        let config = DeepSeekConfig::tiny_mla();
        let (model, runtime) = config.init::<TestBackend>(&device);

        let input_ids = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device);
        let output = model.forward(input_ids, &runtime, None);

        assert_eq!(output.logits.dims(), [1, 4, 1000]);
        assert_eq!(output.hidden_states.dims(), [1, 4, 128]);
    }

    #[test]
    fn test_deepseek_mla_attention() {
        let device = Default::default();
        let config = DeepSeekConfig::tiny_mla();
        let (model, _runtime) = config.init::<TestBackend>(&device);

        // Verify MLA is enabled
        assert!(model.layers[0].attention.use_mla);
        assert!(model.layers[0].attention.q_a_proj.is_some());
        assert!(model.layers[0].attention.kv_a_proj_with_mqa.is_some());
    }
}
