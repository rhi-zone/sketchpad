//! Dump GGUF tensor names and shapes
//!
//! Usage: cargo run --example dump_tensors -- <gguf-path> [filter]

use burn::prelude::*;
use burn_ndarray::NdArray;
use sketchpad_convert::gguf::GgufFile;

type B = NdArray<f32>;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <gguf-path> [filter]", args[0]);
        std::process::exit(1);
    }

    let filter = args.get(2).map(|s| s.as_str()).unwrap_or("");

    let file = GgufFile::open(&args[1]).expect("Failed to open GGUF");
    let mut names: Vec<String> = file.tensor_names().map(|s| s.to_string()).collect();
    names.sort();
    let device = <B as Backend>::Device::default();

    for name in &names {
        if filter.is_empty() || name.contains(filter) {
            let info = file.tensor_info(name).unwrap();
            let n_elems: usize = info.shape.iter().product();
            if n_elems <= 8 {
                if let Ok(t) = file.load_f32::<B, 1>(name, &device) {
                    let vals: Vec<f32> = t.to_data().to_vec().unwrap();
                    println!(
                        "{}: {:?} ({:?}) = {:?}",
                        name, info.shape, info.ggml_type, vals
                    );
                } else {
                    println!("{}: {:?} ({:?})", name, info.shape, info.ggml_type);
                }
            } else if info.shape.len() == 1
                && info.ggml_type == sketchpad_convert::gguf::GgmlType::F32
            {
                if let Ok(t) = file.load_f32::<B, 1>(name, &device) {
                    let vals: Vec<f32> = t.to_data().to_vec().unwrap();
                    let n = vals.len() as f32;
                    let mean = vals.iter().sum::<f32>() / n;
                    let rms = (vals.iter().map(|x| x * x).sum::<f32>() / n).sqrt();
                    let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
                    println!(
                        "{}: {:?} ({:?}) rms={:.4} mean={:.4} min={:.4} max={:.4}",
                        name, info.shape, info.ggml_type, rms, mean, min, max
                    );
                } else {
                    println!("{}: {:?} ({:?})", name, info.shape, info.ggml_type);
                }
            } else {
                println!("{}: {:?} ({:?})", name, info.shape, info.ggml_type);
            }
        }
    }
}
