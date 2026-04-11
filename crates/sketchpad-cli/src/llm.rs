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
use sketchpad_llm::gemma4_gguf_loader::load_gemma4_gguf;
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
        #[arg(long, value_enum, default_value = "f16")]
        precision: crate::Precision,
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

/// Run inference on a GGUF model file, auto-detecting architecture
pub fn run_gguf<B: Backend>(
    weights: PathBuf,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    device: &B::Device,
) -> Result<()> {
    let path = weights.to_str().context("Invalid path")?;

    // Read arch, tokenizer, and chat template from a single GGUF open
    let (arch, tokenizer, seq_len, token_ids) = {
        let file = GgufFile::open(path).context("Failed to open GGUF")?;

        let arch = match file.metadata().get("general.architecture") {
            Some(MetadataValue::String(s)) => s.clone(),
            _ => "gemma4".to_string(),
        };

        let formatted = if gguf_tokenizer::raw_chat_template(&file).is_some() {
            let msgs = [ChatMessage::user(prompt.as_str())];
            gguf_tokenizer::apply_chat_template(&file, &msgs, true)
                .context("Failed to apply chat template")?
        } else {
            prompt.clone()
        };

        let tokenizer =
            gguf_tokenizer::load_tokenizer(&file).context("Failed to load tokenizer")?;
        let encoding = tokenizer
            .encode(formatted.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Tokenize error: {e}"))?;
        let ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
        let len = ids.len();

        (arch, tokenizer, len, ids)
    };

    eprintln!("Architecture: {arch}  Prompt: {seq_len} tokens");

    let data = TensorData::new(token_ids, [1, seq_len]);
    let input_ids = Tensor::<B, 2, Int>::from_data(data, device);

    let new_ids: Vec<u32> = match arch.as_str() {
        "gemma2" => {
            eprintln!("Loading Gemma 2...");
            let start = std::time::Instant::now();
            let (model, runtime, _) =
                load_gemma_gguf::<B, _>(path, device).context("Failed to load Gemma 2")?;
            eprintln!("Loaded in {:.1}s", start.elapsed().as_secs_f64());

            let t = std::time::Instant::now();
            let out = model.generate(input_ids, &runtime, max_tokens, temperature);
            let elapsed = t.elapsed().as_secs_f64();
            let all: Vec<i32> = out.to_data().to_vec().unwrap();
            let n = all.len() - seq_len;
            eprintln!(
                "{n} tokens in {elapsed:.1}s ({:.1} tok/s)",
                n as f64 / elapsed
            );
            all[seq_len..].iter().map(|&id| id as u32).collect()
        }
        _ => {
            eprintln!("Loading Gemma 4...");
            let start = std::time::Instant::now();
            let (model, runtime, _) =
                load_gemma4_gguf::<B, _>(path, device).context("Failed to load Gemma 4")?;
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
            let n = all.len() - seq_len;
            eprintln!(
                "{n} tokens in {elapsed:.1}s ({:.1} tok/s)",
                n as f64 / elapsed
            );
            all[seq_len..].iter().map(|&id| id as u32).collect()
        }
    };

    let generated = tokenizer
        .decode(&new_ids, true)
        .unwrap_or_else(|e| format!("<decode error: {e}>"));
    println!("{generated}");

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
