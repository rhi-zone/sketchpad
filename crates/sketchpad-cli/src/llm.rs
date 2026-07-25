//! LLM CLI Commands
//!
//! Thin CLI wrapper over sketchpad-llm library APIs.

use std::io::{self, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use burn::prelude::*;
use clap::{Subcommand, ValueEnum};

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
