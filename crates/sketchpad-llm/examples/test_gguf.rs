//! Quick smoke test for GGUF Gemma 4 loading
//!
//! Usage: cargo run --example test_gguf -- <path-to-gguf> <path-to-tokenizer.json> [prompt]

use burn::prelude::*;
use burn_ndarray::NdArray;
use sketchpad_llm::gemma4_gguf_loader::load_gemma4_gguf;
use tokenizers::Tokenizer;

type B = NdArray<f32>;

fn vec_stats(vals: &[f32], label: &str) {
    let n = vals.len() as f32;
    let mean = vals.iter().sum::<f32>() / n;
    let rms = (vals.iter().map(|x| x * x).sum::<f32>() / n).sqrt();
    let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
    let nans = vals.iter().filter(|x| x.is_nan()).count();
    let infs = vals.iter().filter(|x| x.is_infinite()).count();
    eprintln!(
        "  {label}: rms={rms:.4} mean={mean:.4} min={min:.4} max={max:.4} nan={nans} inf={infs}"
    );
}

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

    let ropes = &runtime.ropes;
    let sliding_mask = Some(sketchpad_core::transformer::sliding_window_mask::<B>(
        seq_len,
        config.sliding_window,
        &device,
    ));
    let global_mask = Some(sketchpad_core::transformer::causal_mask::<B>(
        seq_len, &device,
    ));

    let hidden_scale = (config.hidden_size as f32).sqrt();
    let mut hs: Tensor<B, 3> = model.embed_tokens.forward(input_ids) * hidden_scale;

    for (i, layer) in model.layers.iter().enumerate() {
        hs = layer.forward(
            hs,
            ropes,
            0,
            sliding_mask.clone(),
            global_mask.clone(),
            config.attn_logit_softcap,
        );
        if i < 3 || i == config.num_layers - 1 {
            let v: Vec<f32> = hs
                .clone()
                .reshape([seq_len * config.hidden_size])
                .to_data()
                .to_vec()
                .unwrap();
            vec_stats(&v, &format!("after layer {i}"));
        }
    }

    hs = model.norm.forward(hs);

    let weight = model.embed_tokens.weight.val().transpose();
    let flat_hidden = hs.reshape([seq_len, config.hidden_size]);
    let logits = flat_hidden.matmul(weight);

    let logit_vals: Vec<f32> = logits.to_data().to_vec().unwrap();
    vec_stats(&logit_vals, "logits (post-softcap)");

    eprintln!("Forward pass in {:.1}s", fwd_start.elapsed().as_secs_f64());

    // Top-5 predictions at last position
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
