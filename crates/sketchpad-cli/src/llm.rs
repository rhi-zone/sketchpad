//! LLM CLI Commands
//!
//! Thin CLI wrapper over sketchpad-llm library APIs.

use std::io::{self, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use burn::prelude::*;
use clap::{Subcommand, ValueEnum};

use sketchpad_convert::gguf::{GgufFile, MetadataValue};
use sketchpad_llm::gemma_gguf_loader::load_gemma_gguf;
use sketchpad_llm::gemma4_gguf_loader::{load_gemma4_gguf, load_gemma4_gguf_offloaded};
use sketchpad_llm::gguf_tokenizer::{self, ChatMessage};
use sketchpad_llm::sampling::SamplerConfig;
use sketchpad_llm::{ChatSession, GenerationConfig, LlmInstance, ModelType};

/// LLM model type selection
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum LlmModelType {
    /// LLaMA 2/3
    Llama,
    /// Mistral 7B
    Mistral,
    /// Mixtral MoE
    Mixtral,
    /// Gemma 2
    Gemma,
    /// Phi-2/3
    Phi,
    /// Qwen 1.5/2
    Qwen,
    /// DeepSeek
    DeepSeek,
    /// RWKV-7
    Rwkv,
    /// Mamba SSM
    Mamba,
    /// Jamba hybrid
    Jamba,
}

impl From<LlmModelType> for ModelType {
    fn from(t: LlmModelType) -> Self {
        match t {
            LlmModelType::Llama => ModelType::Llama,
            LlmModelType::Mistral => ModelType::Mistral,
            LlmModelType::Mixtral => ModelType::Mixtral,
            LlmModelType::Gemma => ModelType::Gemma,
            LlmModelType::Phi => ModelType::Phi,
            LlmModelType::Qwen => ModelType::Qwen,
            LlmModelType::DeepSeek => ModelType::DeepSeek,
            LlmModelType::Rwkv => ModelType::Rwkv,
            LlmModelType::Mamba => ModelType::Mamba,
            LlmModelType::Jamba => ModelType::Jamba,
        }
    }
}

/// LLM subcommands
#[derive(Subcommand)]
pub enum LlmCommands {
    /// Generate text from a prompt
    Generate {
        /// Model type
        #[arg(short, long, value_enum)]
        model: LlmModelType,

        /// Path to model weights directory
        #[arg(short, long)]
        weights: PathBuf,

        /// Input prompt
        #[arg(short, long)]
        prompt: String,

        /// Maximum tokens to generate
        #[arg(long, default_value = "256")]
        max_tokens: usize,

        /// Sampling temperature
        #[arg(long, default_value = "0.7")]
        temperature: f32,

        /// Top-p sampling threshold
        #[arg(long, default_value = "0.9")]
        top_p: f32,
    },

    /// Interactive chat session
    Chat {
        /// Model type
        #[arg(short, long, value_enum)]
        model: LlmModelType,

        /// Path to model weights directory
        #[arg(short, long)]
        weights: PathBuf,

        /// System prompt
        #[arg(short, long, default_value = "You are a helpful assistant.")]
        system: String,

        /// Maximum tokens per response
        #[arg(long, default_value = "256")]
        max_tokens: usize,

        /// Sampling temperature
        #[arg(long, default_value = "0.7")]
        temperature: f32,
    },

    /// Generate text from a GGUF model file (auto-detects architecture)
    ///
    /// Supports gemma2 and gemma4 architectures. Tokenizer and chat template
    /// are loaded directly from the GGUF — no separate tokenizer.json needed.
    ///
    /// Inference strategy is selected automatically:
    /// - Q4K/Q8_0 GGUF + GPU backend → GPU-side quantized matmul (best VRAM efficiency)
    /// - Q4K/Q8_0 GGUF + CPU backend → CPU-offloaded dequant (RAM-resident quantized weights)
    /// - F16/F32 GGUF → standard load (all weights to device)
    Gguf {
        /// Path to the GGUF model file
        #[arg(short, long)]
        weights: PathBuf,

        /// Input prompt (chat template is applied automatically for instruct models)
        #[arg(short, long)]
        prompt: String,

        /// Maximum tokens to generate
        #[arg(long, default_value = "256")]
        max_tokens: usize,

        /// Sampling temperature
        #[arg(long, default_value = "0.7")]
        temperature: f32,

        /// Top-p sampling threshold
        #[arg(long, default_value = "0.9")]
        top_p: f32,

        /// Float precision for model weights and activations
        /// f16 halves VRAM vs f32; bf16 has the same range as f32 but less precision.
        /// On CPU (ndarray), f32 is always used regardless.
        #[arg(long, value_enum, default_value = "bf16")]
        precision: crate::Precision,

        /// Available VRAM in gigabytes. Used to select the best loading strategy.
        /// Quantized bytes fit in budget: GPU-quantized (bytes stay in VRAM, dequant per forward pass).
        /// Neither fits: CPU-offloaded (quantized bytes in RAM, dequant on CPU).
        /// Without this flag, quantized GGUFs default to GPU-quantized on GPU backends.
        #[arg(long, value_name = "GB")]
        vram: Option<f32>,
    },

    /// Start an OpenAI-compatible HTTP server
    #[cfg(feature = "llm-serve")]
    Serve {
        /// Model type
        #[arg(short, long, value_enum)]
        model: LlmModelType,

        /// Path to model weights directory
        #[arg(short, long)]
        weights: PathBuf,

        /// Host address to bind
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port to bind
        #[arg(long, default_value = "8080")]
        port: u16,
    },
}

/// Run the generate command
pub fn run_generate<B: Backend>(
    model_type: LlmModelType,
    weights: PathBuf,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    device: &B::Device,
) -> Result<()> {
    println!("Loading model...");
    let llm = LlmInstance::<B>::load(model_type.into(), &weights, device)
        .context("Failed to load model")?;

    println!("Generating...\n");

    let config = GenerationConfig::new(max_tokens)
        .with_temperature(temperature)
        .with_top_p(top_p);

    let output = llm
        .generate(&prompt, &config)
        .context("Generation failed")?;

    println!("{}", output);

    Ok(())
}

/// Run the chat command
pub fn run_chat<B: Backend>(
    model_type: LlmModelType,
    weights: PathBuf,
    system: String,
    max_tokens: usize,
    temperature: f32,
    device: &B::Device,
) -> Result<()> {
    println!("Loading model...");
    let llm = LlmInstance::<B>::load(model_type.into(), &weights, device)
        .context("Failed to load model")?;

    let mut session = ChatSession::new(llm, Some(&system));

    let config = GenerationConfig::new(max_tokens).with_temperature(temperature);

    println!("\nChat session started. Type 'quit' or 'exit' to end.\n");

    loop {
        // Print prompt
        print!("You: ");
        io::stdout().flush()?;

        // Read user input
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        // Check for exit
        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
            println!("\nGoodbye!");
            break;
        }

        if input.is_empty() {
            continue;
        }

        // Generate response
        match session.send(input, &config) {
            Ok(response) => {
                println!("\nAssistant: {}\n", response);
            }
            Err(e) => {
                eprintln!("\nError: {}\n", e);
            }
        }
    }

    Ok(())
}

type TokenDecoder = Box<dyn Fn(&[u32]) -> String>;

/// Shared inference parameters for GGUF runs.
pub struct GgufRunParams {
    pub weights: PathBuf,
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub vram_gb: Option<f32>,
}

/// Loading strategy selected based on model size and available VRAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufLoadStrategy {
    /// Dequantize all weights to f16/bf16 at load time (fastest inference, most VRAM).
    Standard,
    /// Keep quantized bytes in VRAM, dequantize per forward pass (GPU quant supported types only).
    GpuQuant,
    /// Keep quantized bytes in RAM, dequantize on CPU per forward pass (least VRAM).
    CpuOffload,
}

struct GgufInput {
    arch: String,
    strategy: GgufLoadStrategy,
    /// Decodes a slice of token IDs to a string (wraps tokenizer::decode).
    decode: TokenDecoder,
    seq_len: usize,
    token_ids: Vec<i32>,
}

fn load_gguf_input(
    path: &str,
    prompt: &str,
    vram_gb: Option<f32>,
    gpu_quant_available: bool,
) -> Result<GgufInput> {
    let file = GgufFile::open(path).context("Failed to open GGUF")?;

    let arch = match file.metadata().get("general.architecture") {
        Some(MetadataValue::String(s)) => s.clone(),
        _ => "gemma4".to_string(),
    };

    let vram_budget = vram_gb.map(|gb| (gb * 0.85 * 1024.0 * 1024.0 * 1024.0) as u64);
    let (f16_bytes, quantized_bytes) = file.estimated_vram_bytes();

    if let Some(budget) = vram_budget {
        eprintln!(
            "Model size: {:.1} GB f16, {:.1} GB quantized  (budget: {:.1} GB)",
            f16_bytes as f64 / 1e9,
            quantized_bytes as f64 / 1e9,
            budget as f64 / 1e9,
        );
    }

    // TODO: auto-detect VRAM via wgpu adapter when vram_budget is None
    let strategy = match vram_budget {
        Some(budget) => {
            if quantized_bytes <= budget {
                // Whole model fits as quantized bytes — use GPU quant (best of both worlds)
                if gpu_quant_available {
                    GgufLoadStrategy::GpuQuant
                } else {
                    GgufLoadStrategy::Standard
                }
            } else {
                GgufLoadStrategy::CpuOffload
            }
        }
        None => {
            // No VRAM hint — heuristic: GPU quant if supported, else Standard
            if gpu_quant_available {
                GgufLoadStrategy::GpuQuant
            } else {
                GgufLoadStrategy::Standard
            }
        }
    };

    let formatted = if gguf_tokenizer::raw_chat_template(&file).is_some() {
        let msgs = [ChatMessage::user(prompt)];
        gguf_tokenizer::apply_chat_template(&file, &msgs, true)
            .context("Failed to apply chat template")?
    } else {
        prompt.to_string()
    };

    let tokenizer = gguf_tokenizer::load_tokenizer(&file).context("Failed to load tokenizer")?;
    let encoding = tokenizer
        .encode(formatted.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Tokenize error: {e}"))?;
    let token_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
    let seq_len = token_ids.len();

    let decode = Box::new(move |ids: &[u32]| {
        tokenizer
            .decode(ids, true)
            .unwrap_or_else(|e| format!("<decode error: {e}>"))
    });

    Ok(GgufInput {
        arch,
        strategy,
        decode,
        seq_len,
        token_ids,
    })
}

/// Run inference on a GGUF model file (standard load or CPU-offload path).
/// GPU-quant strategy is dispatched separately via `run_gguf_gpu_quant`.
pub fn run_gguf<B: Backend>(params: GgufRunParams, device: &B::Device) -> Result<()> {
    let GgufRunParams {
        weights,
        prompt,
        max_tokens,
        temperature,
        top_p,
        vram_gb,
    } = params;
    let path = weights.to_str().context("Invalid path")?;
    let input = load_gguf_input(path, &prompt, vram_gb, false)?;
    eprintln!(
        "Architecture: {}  Prompt: {} tokens",
        input.arch, input.seq_len
    );

    let data = TensorData::new(input.token_ids.clone(), [1, input.seq_len]);
    let input_ids = Tensor::<B, 2, Int>::from_data(data, device);

    let all_ids: Vec<i32> = match input.arch.as_str() {
        "gemma2" => {
            eprintln!("Loading Gemma 2...");
            let start = std::time::Instant::now();
            let (model, runtime, _) =
                load_gemma_gguf::<B, _>(path, device).context("Failed to load Gemma 2")?;
            eprintln!("Loaded in {:.1}s", start.elapsed().as_secs_f64());

            let t = std::time::Instant::now();
            let out = model.generate(
                input_ids,
                &runtime,
                max_tokens,
                &SamplerConfig {
                    temperature,
                    ..SamplerConfig::greedy()
                },
            );
            let elapsed = t.elapsed().as_secs_f64();
            let all: Vec<i32> = out.to_data().to_vec().unwrap();
            let n = all.len() - input.seq_len;
            eprintln!(
                "{n} tokens in {elapsed:.1}s ({:.1} tok/s)",
                n as f64 / elapsed
            );
            all
        }
        _ => {
            let offload = input.strategy == GgufLoadStrategy::CpuOffload;
            let label = if offload { " (CPU offload)" } else { "" };
            eprintln!("Loading Gemma 4{label}...");
            let start = std::time::Instant::now();
            let (model, runtime, _) = if offload {
                load_gemma4_gguf_offloaded::<B, _>(path, device)
                    .context("Failed to load Gemma 4 (offloaded)")?
            } else {
                load_gemma4_gguf::<B, _>(path, device).context("Failed to load Gemma 4")?
            };
            eprintln!("Loaded in {:.1}s", start.elapsed().as_secs_f64());

            let sampler = SamplerConfig {
                temperature,
                top_p,
                ..SamplerConfig::greedy()
            };
            let t = std::time::Instant::now();
            let out = model.generate(input_ids, &runtime, max_tokens, &sampler);
            let elapsed = t.elapsed().as_secs_f64();
            let all: Vec<i32> = out.to_data().to_vec().unwrap();
            let n = all.len() - input.seq_len;
            eprintln!(
                "{n} tokens in {elapsed:.1}s ({:.1} tok/s)",
                n as f64 / elapsed
            );
            all
        }
    };

    let new_ids: Vec<u32> = all_ids[input.seq_len..]
        .iter()
        .map(|&id| id as u32)
        .collect();
    println!("{}", (input.decode)(&new_ids));
    Ok(())
}

/// Run GPU-quantized inference on a GGUF model file.
///
/// Projection weights are uploaded to VRAM as quantized bytes (Q4K/Q8_0) and
/// dequantized in GPU kernels per forward pass. Achieves 4–8× VRAM savings vs f16
/// at the cost of per-token dequantization on the GPU.
#[cfg(feature = "cubecl")]
pub fn run_gguf_gpu_quant<R, F, I, BT>(
    params: GgufRunParams,
    dtype: burn::tensor::DType,
    device: &R::Device,
) -> Result<()>
where
    R: burn_cubecl::CubeRuntime + Send + Sync + 'static,
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::BoolElement,
{
    use burn_cubecl::CubeBackend;
    use sketchpad_llm::gemma4_gguf_loader::load_gemma4_gguf_gpu_quant;

    let path = params.weights.to_str().context("Invalid path")?;
    let input = load_gguf_input(path, &params.prompt, params.vram_gb, true)?;

    // If VRAM hint says standard load or CPU offload is better, delegate to run_gguf
    if input.strategy != GgufLoadStrategy::GpuQuant {
        return run_gguf::<CubeBackend<R, F, I, BT>>(params, device);
    }
    eprintln!(
        "Architecture: {}  Prompt: {} tokens",
        input.arch, input.seq_len
    );

    type B<R, F, I, BT> = CubeBackend<R, F, I, BT>;

    let data = TensorData::new(input.token_ids.clone(), [1, input.seq_len]);
    let input_ids = Tensor::<B<R, F, I, BT>, 2, Int>::from_data(data, device);

    eprintln!("Loading Gemma 4 (GPU quant)...");
    let start = std::time::Instant::now();
    let (model, runtime, _) = load_gemma4_gguf_gpu_quant::<R, F, I, BT, _>(path, device, dtype)
        .context("Failed to load Gemma 4 (GPU quant)")?;
    eprintln!("Loaded in {:.1}s", start.elapsed().as_secs_f64());

    let sampler = SamplerConfig {
        temperature: params.temperature,
        top_p: params.top_p,
        ..SamplerConfig::greedy()
    };
    let t = std::time::Instant::now();
    let out = model.generate(input_ids, &runtime, params.max_tokens, &sampler);
    let elapsed = t.elapsed().as_secs_f64();
    let all: Vec<i32> = out.to_data().to_vec().unwrap();
    let n = all.len() - input.seq_len;
    eprintln!(
        "{n} tokens in {elapsed:.1}s ({:.1} tok/s)",
        n as f64 / elapsed
    );

    let new_ids: Vec<u32> = all[input.seq_len..].iter().map(|&id| id as u32).collect();
    println!("{}", (input.decode)(&new_ids));
    Ok(())
}

/// Run the serve command
#[cfg(feature = "llm-serve")]
pub async fn run_serve<B: Backend + 'static>(
    model_type: LlmModelType,
    weights: PathBuf,
    host: String,
    port: u16,
    device: &B::Device,
) -> Result<()> {
    println!("Loading model...");
    let llm = LlmInstance::<B>::load(model_type.into(), &weights, device)
        .context("Failed to load model")?;

    println!("Starting server on {}:{}...", host, port);
    sketchpad_llm::serve::run_server(llm, &host, port)
        .await
        .context("Server error")?;

    Ok(())
}
