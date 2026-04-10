//! Quick smoke test for GGUF Gemma 4 loading
//!
//! Usage: cargo run --example test_gguf -- <path-to-gguf> <path-to-tokenizer.json>

use burn_ndarray::NdArray;
use sketchpad_llm::inference::{GenerationConfig, LlmInstance};

type Backend = NdArray<f32>;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <gguf-path> <tokenizer-path>", args[0]);
        std::process::exit(1);
    }

    let gguf_path = &args[1];
    let tokenizer_path = &args[2];
    let device = Default::default();

    eprintln!("Loading model from {gguf_path}...");
    let start = std::time::Instant::now();

    let llm = LlmInstance::<Backend>::load_gguf(gguf_path, tokenizer_path, &device)
        .expect("Failed to load model");

    eprintln!("Model loaded in {:.1}s", start.elapsed().as_secs_f64());

    let config = GenerationConfig::new(32).with_temperature(0.7);
    let prompt = "Hello! I am a language model and";

    eprintln!("Generating from: \"{prompt}\"");
    let start = std::time::Instant::now();

    match llm.generate(prompt, &config) {
        Ok(output) => {
            eprintln!("Generated in {:.1}s", start.elapsed().as_secs_f64());
            println!("{prompt}{output}");
        }
        Err(e) => {
            eprintln!("Generation failed: {e}");
            std::process::exit(1);
        }
    }
}
