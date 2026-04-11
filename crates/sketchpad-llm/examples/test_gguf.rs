//! Quick smoke test for GGUF Gemma 4 loading
//!
//! Usage: cargo run --example test_gguf -- <path-to-gguf> <path-to-tokenizer.json> [prompt]

use burn::prelude::*;
use burn_ndarray::NdArray;
use sketchpad_llm::gemma4_gguf_loader::load_gemma4_gguf;
use tokenizers::Tokenizer;

type B = NdArray<f32>;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <gguf-path> <tokenizer-path> [prompt]", args[0]);
        std::process::exit(1);
    }

    let gguf_path = &args[1];
    let tokenizer_path = &args[2];
    let prompt = if args.len() > 3 {
        args[3].as_str()
    } else {
        "Hello! I am a language model and"
    };
    let device = <B as Backend>::Device::default();

    eprintln!("Loading model from {gguf_path}...");
    let start = std::time::Instant::now();

    let (model, runtime, config) =
        load_gemma4_gguf::<B, _>(gguf_path, &device).expect("Failed to load model");

    eprintln!("Model loaded in {:.1}s", start.elapsed().as_secs_f64());

    let tokenizer = Tokenizer::from_file(tokenizer_path).expect("Failed to load tokenizer");
    let encoding = tokenizer.encode(prompt, false).expect("Failed to encode");
    let token_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
    eprintln!("Prompt: {:?}", prompt);
    eprintln!("Token IDs: {:?}", token_ids);

    let seq_len = token_ids.len();
    let data = TensorData::new(token_ids, [1, seq_len]);
    let input_ids = Tensor::<B, 2, Int>::from_data(data, &device);

    eprintln!("Running forward pass...");
    let fwd_start = std::time::Instant::now();

    let output = model.forward(input_ids, &runtime, None);

    eprintln!("Forward pass in {:.1}s", fwd_start.elapsed().as_secs_f64());

    // Print top-5 tokens at the last position
    let logit_vals: Vec<f32> = output
        .logits
        .reshape([seq_len, config.vocab_size])
        .to_data()
        .to_vec()
        .unwrap();
    let last: Vec<f32> = logit_vals[(seq_len - 1) * config.vocab_size..].to_vec();
    let mut indexed: Vec<(usize, f32)> = last.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!("Top-5 predictions at last position:");
    for (token_id, logit) in indexed.iter().take(5) {
        let text = tokenizer
            .decode(&[*token_id as u32], false)
            .unwrap_or_else(|_| format!("<{token_id}>"));
        eprintln!("  {token_id:6}  {logit:8.3}  {:?}", text);
    }
}
