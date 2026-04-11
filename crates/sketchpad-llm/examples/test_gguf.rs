//! Gemma 4 GGUF text generation
//!
//! Tokenizer and chat template are loaded directly from the GGUF file —
//! no separate tokenizer.json needed.
//!
//! If the GGUF contains a chat template (instruct models), the prompt is
//! automatically formatted with it. Base models receive the prompt as-is.
//!
//! Usage: cargo run --example test_gguf -- <gguf-path> <prompt> [max-tokens]

use burn::prelude::*;
use burn_ndarray::NdArray;
use sketchpad_llm::gemma4_gguf_loader::load_gemma4_gguf;
use sketchpad_llm::gguf_tokenizer::{self, ChatMessage};
use sketchpad_llm::sampling::SamplerConfig;

type B = NdArray<f32>;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <gguf-path> <prompt> [max-tokens]", args[0]);
        std::process::exit(1);
    }

    let gguf_path = &args[1];
    let prompt = args[2].as_str();
    let max_tokens: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(256);

    let device = <B as Backend>::Device::default();

    eprintln!("Loading model from {gguf_path}...");
    let start = std::time::Instant::now();
    let (model, runtime, _config) =
        load_gemma4_gguf::<B, _>(gguf_path, &device).expect("Failed to load model");
    eprintln!("Model loaded in {:.1}s", start.elapsed().as_secs_f64());

    // Load tokenizer from the GGUF file directly — no external tokenizer.json needed
    let file = sketchpad_convert::gguf::GgufFile::open(gguf_path).expect("Failed to reopen GGUF");
    let tokenizer = gguf_tokenizer::load_tokenizer(&file).expect("Failed to load tokenizer");

    // Apply chat template if the model has one (instruct models), else use prompt as-is
    let formatted = if gguf_tokenizer::raw_chat_template(&file).is_some() {
        let msgs = [ChatMessage::user(prompt)];
        gguf_tokenizer::apply_chat_template(&file, &msgs, true)
            .expect("Failed to apply chat template")
    } else {
        prompt.to_string()
    };

    let encoding = tokenizer
        .encode(formatted.as_str(), false)
        .expect("Failed to encode");
    let token_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
    let seq_len = token_ids.len();

    eprintln!("Prompt ({seq_len} tokens)");

    let data = TensorData::new(token_ids, [1, seq_len]);
    let input_ids = Tensor::<B, 2, Int>::from_data(data, &device);

    eprintln!("Generating up to {max_tokens} tokens...");
    let gen_start = std::time::Instant::now();

    let sampler = SamplerConfig {
        temperature: 0.7,
        top_p: 0.9,
        ..SamplerConfig::greedy()
    };
    let output_ids = model.generate(input_ids, &runtime, max_tokens, &sampler);
    eprintln!("Generated in {:.1}s", gen_start.elapsed().as_secs_f64());

    // Decode the newly generated tokens (skip the prompt)
    let all_ids: Vec<i32> = output_ids.to_data().to_vec().unwrap();
    let new_ids: Vec<u32> = all_ids[seq_len..].iter().map(|&id| id as u32).collect();
    let generated = tokenizer
        .decode(&new_ids, true)
        .unwrap_or_else(|e| format!("<decode error: {e}>"));

    println!("{generated}");
}
