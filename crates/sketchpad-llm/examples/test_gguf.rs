//! GGUF text generation
//!
//! Auto-detects model architecture from GGUF metadata (`general.architecture`).
//! Supported: `gemma2`, `gemma4`.
//!
//! Tokenizer and chat template are loaded directly from the GGUF file —
//! no separate tokenizer.json needed. Chat template is applied automatically
//! for instruct models.
//!
//! Running on CPU: this example uses the NdArray backend (pure CPU).
//! For GPU (ROCm via Vulkan or CUDA), use the `burn-wgpu` or `burn-cuda`
//! backend instead of `NdArray` and set the appropriate device.
//!
//! RAM: non-expert weights are dequantized to f32 on load. For large models
//! (Gemma 4 27B+) this can use >30 GB. Expert weights stay quantized.
//!
//! Usage: cargo run --example test_gguf -- <gguf-path> <prompt> [max-tokens]

use burn::prelude::*;
use burn_ndarray::NdArray;
use sketchpad_convert::gguf::{GgufFile, MetadataValue};
use sketchpad_core::kv_cache::KvCacheConfig;
use sketchpad_llm::gemma_gguf_loader::load_gemma_gguf;
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

    // Read architecture and tokenize from one GGUF open
    let (arch, formatted, tokenizer, seq_len, token_ids) = {
        let file = GgufFile::open(gguf_path).expect("Failed to open GGUF");

        let arch = match file.metadata().get("general.architecture") {
            Some(MetadataValue::String(s)) => s.clone(),
            _ => "gemma4".to_string(),
        };

        let formatted = if gguf_tokenizer::raw_chat_template(&file).is_some() {
            let msgs = [ChatMessage::user(prompt)];
            gguf_tokenizer::apply_chat_template(&file, &msgs, true)
                .expect("Failed to apply chat template")
        } else {
            prompt.to_string()
        };

        let tokenizer = gguf_tokenizer::load_tokenizer(&file).expect("Failed to load tokenizer");
        let encoding = tokenizer
            .encode(formatted.as_str(), false)
            .expect("Failed to encode");
        let ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
        let len = ids.len();

        (arch, formatted, tokenizer, len, ids)
    };

    eprintln!("Architecture: {arch}");
    eprintln!("Prompt ({seq_len} tokens)");
    let _ = formatted; // consumed above

    let data = TensorData::new(token_ids, [1, seq_len]);
    let input_ids = Tensor::<B, 2, Int>::from_data(data, &device);

    eprintln!("Generating up to {max_tokens} tokens...");

    let new_ids: Vec<u32> = match arch.as_str() {
        "gemma2" => {
            eprintln!("Loading Gemma 2 model...");
            let start = std::time::Instant::now();
            let (model, runtime, _config) =
                load_gemma_gguf::<B, _>(gguf_path, &device).expect("Failed to load model");
            eprintln!("Model loaded in {:.1}s", start.elapsed().as_secs_f64());

            if let Some(rss) = peak_rss_mb() {
                eprintln!("RAM after load: {rss} MB");
            }

            let gen_start = std::time::Instant::now();
            let mut cache = runtime.create_kv_cache(&KvCacheConfig::Standard);
            let output_ids = model.generate(
                input_ids,
                &runtime,
                max_tokens,
                cache.as_mut(),
                &sketchpad_llm::sampling::SamplerConfig {
                    temperature: 0.7,
                    ..sketchpad_llm::sampling::SamplerConfig::greedy()
                },
            );
            let elapsed = gen_start.elapsed().as_secs_f64();

            let all_ids: Vec<i32> = output_ids.to_data().to_vec().unwrap();
            let new_count = all_ids.len() - seq_len;
            eprintln!(
                "Generated {new_count} tokens in {elapsed:.1}s ({:.1} tok/s)",
                new_count as f64 / elapsed
            );

            all_ids[seq_len..].iter().map(|&id| id as u32).collect()
        }
        _ => {
            // gemma4 (and any unknown architecture defaults to gemma4)
            eprintln!("Loading Gemma 4 model...");
            let start = std::time::Instant::now();
            let (model, runtime, _config) =
                load_gemma4_gguf::<B, _>(gguf_path, &device).expect("Failed to load model");
            eprintln!("Model loaded in {:.1}s", start.elapsed().as_secs_f64());

            if let Some(rss) = peak_rss_mb() {
                eprintln!("RAM after load: {rss} MB");
            }

            let sampler = SamplerConfig {
                temperature: 0.7,
                top_p: 0.9,
                ..SamplerConfig::greedy()
            };

            let gen_start = std::time::Instant::now();
            let mut cache = runtime.create_kv_cache(&KvCacheConfig::Standard);
            let output_ids =
                model.generate(input_ids, &runtime, max_tokens, cache.as_mut(), &sampler);
            let elapsed = gen_start.elapsed().as_secs_f64();

            let all_ids: Vec<i32> = output_ids.to_data().to_vec().unwrap();
            let new_count = all_ids.len() - seq_len;
            eprintln!(
                "Generated {new_count} tokens in {elapsed:.1}s ({:.1} tok/s)",
                new_count as f64 / elapsed
            );

            all_ids[seq_len..].iter().map(|&id| id as u32).collect()
        }
    };

    if let Some(rss) = peak_rss_mb() {
        eprintln!("Peak RAM: {rss} MB");
    }

    let generated = tokenizer
        .decode(&new_ids, true)
        .unwrap_or_else(|e| format!("<decode error: {e}>"));

    println!("{generated}");
}

/// Read peak virtual memory size from /proc/self/status (Linux only).
/// Returns None on non-Linux or if the file is unreadable.
fn peak_rss_mb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if line.starts_with("VmPeak:") {
            let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
            return Some(kb / 1024);
        }
    }
    None
}
