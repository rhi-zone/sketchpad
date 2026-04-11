//! Dump GGUF metadata keys
//!
//! Usage: cargo run --example dump_meta -- <gguf-path>

use sketchpad_convert::gguf::GgufFile;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <gguf-path>", args[0]);
        std::process::exit(1);
    }

    let file = GgufFile::open(&args[1]).expect("Failed to open GGUF");
    let mut keys: Vec<&str> = file.metadata().keys().map(|s| s.as_str()).collect();
    keys.sort();
    for k in &keys {
        println!("{}: {:?}", k, file.metadata().get(*k).unwrap());
    }
}
