//! Dump tensor names from a GGUF file, filtered by prefix.
//!
//! Usage: cargo run -p sketchpad-convert --example dump_tensors -- <gguf-path> [prefix...]

use sketchpad_convert::gguf::GgufFile;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <gguf-path> [prefix...]", args[0]);
        std::process::exit(1);
    }

    let path = &args[1];
    let prefixes: Vec<&str> = if args.len() > 2 {
        args[2..].iter().map(|s| s.as_str()).collect()
    } else {
        vec![]
    };

    eprintln!("Opening {path}...");
    let file = GgufFile::open(path).expect("Failed to open GGUF file");

    let mut names: Vec<&str> = file.tensor_names().collect();
    names.sort();

    if prefixes.is_empty() {
        for name in &names {
            let info = file.tensor_info(name).unwrap();
            println!("{name}  {:?}  {:?}", info.shape, info.ggml_type);
        }
    } else {
        for prefix in &prefixes {
            println!("=== Tensors matching prefix \"{prefix}\" ===");
            let mut count = 0;
            for name in &names {
                if name.starts_with(prefix) {
                    let info = file.tensor_info(name).unwrap();
                    println!("  {name}  {:?}  {:?}", info.shape, info.ggml_type);
                    count += 1;
                }
            }
            println!("  ({count} tensors)\n");
        }
    }
}
