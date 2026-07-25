//! Safetensors inspection tool
//!
//! Prints tensor names, shapes, and dtypes from safetensors files.

use std::fs::File;
use std::path::PathBuf;

use clap::Parser;
use memmap2::MmapOptions;
use safetensors::SafeTensors;

#[derive(Parser)]
#[command(name = "inspect-safetensors")]
#[command(about = "Inspect safetensors files")]
struct Args {
    /// Path to safetensors file
    path: PathBuf,

    /// Filter tensor names (substring match)
    #[arg(short, long)]
    filter: Option<String>,

    /// Show only shapes, not dtypes
    #[arg(long)]
    shapes_only: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let file = File::open(&args.path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap)?;

    let mut names = tensors.names();
    names.sort();

    println!("File: {}", args.path.display());
    println!("Total tensors: {}\n", names.len());

    for name in names {
        if let Some(ref filter) = args.filter {
            if !name.contains(filter) {
                continue;
            }
        }

        let view = tensors.tensor(name)?;
        let shape: Vec<_> = view.shape().iter().collect();

        if args.shapes_only {
            println!("{}: {:?}", name, shape);
        } else {
            println!("{}: {:?} ({:?})", name, shape, view.dtype());
        }
    }

    Ok(())
}
