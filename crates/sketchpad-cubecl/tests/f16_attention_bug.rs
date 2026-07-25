//! Minimal reproduction of cubek-attention half-precision alignment bug
//!
//! Run with:
//!   cargo test -p burn-models-cubecl --features cuda --test f16_attention_bug --release -- --nocapture --ignored
//!
//! Results (cubek-attention 0.1.0-pre.1):
//! - f32:  PASS ✓
//! - bf16: FAIL - "assertion failed: unit_tile.layout.num_cols % line_size == 0"
//! - f16:  FAIL - "assertion failed: unit_tile.layout.num_cols % line_size == 0"
//!
//! The bug is in cubek-attention's tile inference for half-precision types.
//! The internal tile dimensions don't align with the line_size for f16/bf16.

#![cfg(feature = "cuda")]

use burn::backend::cuda::CudaDevice;
use burn::tensor::{DType, Shape};
use burn_cubecl::ops::numeric::empty_device_dtype;
use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use cubek::attention::{
    definition::{AccumulatorPrecision, AttentionGlobalTypes, AttentionOptions},
    launch::{BlueprintStrategy, Strategy},
};

/// Run flash attention with given dtype, returns Ok(()) on success or Err with message
fn run_attention(
    device: &CudaDevice,
    dtype: DType,
    batch: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<(), String> {
    let client = CudaRuntime::client(device);
    let shape = Shape::new([batch, heads, seq_len, head_dim]);

    // Create Q, K, V, Out tensors
    let q = empty_device_dtype::<CudaRuntime>(client.clone(), device.clone(), shape.clone(), dtype);
    let k = empty_device_dtype::<CudaRuntime>(client.clone(), device.clone(), shape.clone(), dtype);
    let v = empty_device_dtype::<CudaRuntime>(client.clone(), device.clone(), shape.clone(), dtype);
    let out =
        empty_device_dtype::<CudaRuntime>(client.clone(), device.clone(), shape.clone(), dtype);

    let dtypes = AttentionGlobalTypes {
        query: dtype.into(),
        key: dtype.into(),
        value: dtype.into(),
        mask: DType::U8.into(),
        out: dtype.into(),
    };

    // Use the same strategy as burn-models-cubecl/src/attention.rs
    let strategy = Strategy::Unit(BlueprintStrategy::Inferred(()));

    // Use catch_unwind since the bug causes a panic, not an error return
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cubek::attention::launch::launch_ref::<CudaRuntime>(
            strategy,
            &client,
            &q.as_handle_ref(),
            &k.as_handle_ref(),
            &v.as_handle_ref(),
            &None,
            &out.as_handle_ref(),
            &dtypes,
            AttentionOptions {
                causal: false,
                accumulator_precision: AccumulatorPrecision::Strict(
                    cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(
                        cubecl::ir::FloatKind::F32,
                    )),
                ),
            },
        )
    }));

    match result {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(e)) => Err(format!("AttentionSetupError: {:?}", e)),
        Err(panic) => {
            if let Some(s) = panic.downcast_ref::<&str>() {
                Err(format!("Panic: {}", s))
            } else if let Some(s) = panic.downcast_ref::<String>() {
                Err(format!("Panic: {}", s))
            } else {
                Err("Panic: <unknown>".to_string())
            }
        }
    }
}

/// Test with SD 1.x dimensions: batch=1, heads=8, seq=4096 (64x64), head_dim=64 (padded from 40)
#[test]
#[ignore = "requires CUDA GPU - demonstrates cubek-attention half-precision bug"]
fn test_half_precision_attention_bug() {
    let device = CudaDevice::default();

    let batch = 1;
    let heads = 8;
    let seq_len = 4096; // 64x64 spatial positions
    let head_dim = 64; // Padded from 40 to power-of-2

    println!("\n=== cubek-attention half-precision alignment bug reproduction ===");
    println!(
        "Dimensions: batch={}, heads={}, seq_len={}, head_dim={}",
        batch, heads, seq_len, head_dim
    );
    println!("cubek-attention version: 0.1.0-pre.1");
    println!();

    // Test f32 - should work
    print!("f32:  ");
    match run_attention(&device, DType::F32, batch, heads, seq_len, head_dim) {
        Ok(_) => println!("PASS ✓"),
        Err(e) => println!("FAIL: {}", e),
    }

    // Test bf16 - PANICS
    print!("bf16: ");
    match run_attention(&device, DType::BF16, batch, heads, seq_len, head_dim) {
        Ok(_) => println!("PASS ✓"),
        Err(e) => println!("FAIL: {}", e),
    }

    // Test f16 - PANICS
    print!("f16:  ");
    match run_attention(&device, DType::F16, batch, heads, seq_len, head_dim) {
        Ok(_) => println!("PASS ✓"),
        Err(e) => println!("FAIL: {}", e),
    }
}

/// Test different head_dim values to find what works for half precision
#[test]
#[ignore = "requires CUDA GPU - exploring padding workarounds"]
fn test_head_dim_padding() {
    let device = CudaDevice::default();

    println!("\n=== Testing different head_dim values for half precision ===\n");

    let batch = 1;
    let heads = 8;
    let seq_len = 256; // Smaller for faster testing

    // Try various head_dim values
    let head_dims = [32, 64, 128, 256];

    for head_dim in head_dims {
        println!("head_dim={}", head_dim);

        print!("  f32:  ");
        match run_attention(&device, DType::F32, batch, heads, seq_len, head_dim) {
            Ok(_) => println!("PASS"),
            Err(e) => println!("FAIL: {}", e),
        }

        print!("  bf16: ");
        match run_attention(&device, DType::BF16, batch, heads, seq_len, head_dim) {
            Ok(_) => println!("PASS"),
            Err(e) => println!("FAIL: {}", e),
        }

        print!("  f16:  ");
        match run_attention(&device, DType::F16, batch, heads, seq_len, head_dim) {
            Ok(_) => println!("PASS"),
            Err(e) => println!("FAIL: {}", e),
        }

        println!();
    }
}

/// Test different seq_len values
#[test]
#[ignore = "requires CUDA GPU - exploring padding workarounds"]
fn test_seq_len_padding() {
    let device = CudaDevice::default();

    println!("\n=== Testing different seq_len values for half precision ===\n");

    let batch = 1;
    let heads = 8;
    let head_dim = 128; // Try larger head_dim

    // Try various seq_len values
    let seq_lens = [64, 128, 256, 512, 1024];

    for seq_len in seq_lens {
        println!("seq_len={}", seq_len);

        print!("  f32:  ");
        match run_attention(&device, DType::F32, batch, heads, seq_len, head_dim) {
            Ok(_) => println!("PASS"),
            Err(e) => println!("FAIL: {}", e),
        }

        print!("  bf16: ");
        match run_attention(&device, DType::BF16, batch, heads, seq_len, head_dim) {
            Ok(_) => println!("PASS"),
            Err(e) => println!("FAIL: {}", e),
        }

        print!("  f16:  ");
        match run_attention(&device, DType::F16, batch, heads, seq_len, head_dim) {
            Ok(_) => println!("PASS"),
            Err(e) => println!("FAIL: {}", e),
        }

        println!();
    }
}

/// Test BlackboxAccelerated strategy (uses TensorCores)
#[test]
#[ignore = "requires CUDA GPU - testing BlackboxAccelerated strategy"]
fn test_blackbox_accelerated_strategy() {
    use cubek::attention::launch::Strategy;

    let device = CudaDevice::default();
    let client = CudaRuntime::client(&device);

    let batch = 1;
    let heads = 8;
    let seq_len = 256;
    let head_dim = 64;

    println!("\n=== Testing BlackboxAccelerated strategy ===\n");

    for dtype in [DType::F32, DType::BF16, DType::F16] {
        let shape = Shape::new([batch, heads, seq_len, head_dim]);
        let q =
            empty_device_dtype::<CudaRuntime>(client.clone(), device.clone(), shape.clone(), dtype);
        let k =
            empty_device_dtype::<CudaRuntime>(client.clone(), device.clone(), shape.clone(), dtype);
        let v =
            empty_device_dtype::<CudaRuntime>(client.clone(), device.clone(), shape.clone(), dtype);
        let out =
            empty_device_dtype::<CudaRuntime>(client.clone(), device.clone(), shape.clone(), dtype);

        let dtypes = AttentionGlobalTypes {
            query: dtype.into(),
            key: dtype.into(),
            value: dtype.into(),
            mask: DType::U8.into(),
            out: dtype.into(),
        };

        // Try BlackboxAccelerated instead of Unit
        let strategy = Strategy::BlackboxAccelerated(BlueprintStrategy::Inferred(()));

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cubek::attention::launch::launch_ref::<CudaRuntime>(
                strategy,
                &client,
                &q.as_handle_ref(),
                &k.as_handle_ref(),
                &v.as_handle_ref(),
                &None,
                &out.as_handle_ref(),
                &dtypes,
                AttentionOptions {
                    causal: false,
                    accumulator_precision: AccumulatorPrecision::Strict(
                        cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(
                            cubecl::ir::FloatKind::F32,
                        )),
                    ),
                },
            )
        }));

        let dtype_name = match dtype {
            DType::F32 => "f32",
            DType::BF16 => "bf16",
            DType::F16 => "f16",
            _ => "other",
        };

        match result {
            Ok(Ok(_)) => println!("{}: PASS ✓ (BlackboxAccelerated)", dtype_name),
            Ok(Err(e)) => println!("{}: FAIL (error): {:?}", dtype_name, e),
            Err(_) => println!("{}: FAIL (panic)", dtype_name),
        }
    }
}

/// Test combinations to find working dimensions
#[test]
#[ignore = "requires CUDA GPU - exploring padding workarounds"]
fn test_dimension_combinations() {
    let device = CudaDevice::default();

    println!("\n=== Testing dimension combinations for bf16 ===\n");
    println!("Looking for combinations where bf16 works...\n");

    let batch = 1;
    let heads_options = [1, 8];
    let seq_lens = [64, 128, 256];
    let head_dims = [32, 64, 128, 256];

    for &heads in &heads_options {
        for &seq_len in &seq_lens {
            for &head_dim in &head_dims {
                let result = run_attention(&device, DType::BF16, batch, heads, seq_len, head_dim);
                if result.is_ok() {
                    println!(
                        "✓ bf16 WORKS: heads={}, seq_len={}, head_dim={}",
                        heads, seq_len, head_dim
                    );
                }
            }
        }
    }

    println!("\nIf no lines printed above, bf16 fails for all tested combinations.");
}
