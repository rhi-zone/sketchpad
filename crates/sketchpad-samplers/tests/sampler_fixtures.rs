//! Integration tests using JSON fixtures generated from k-diffusion reference.
//!
//! To regenerate fixtures: `python scripts/gen_sampler_fixtures.py`

use serde::Deserialize;
use std::fs;
use std::path::PathBuf;

/// Tolerance for floating point comparisons
const EPSILON: f32 = 1e-3;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn assert_approx_eq(expected: f32, actual: f32, name: &str) {
    let diff = (expected - actual).abs();
    assert!(
        diff < EPSILON,
        "{}: expected {}, got {} (diff: {})",
        name,
        expected,
        actual,
        diff
    );
}

// ============================================================================
// Sigma Schedule Tests
// ============================================================================

#[derive(Debug, Deserialize)]
struct SigmaScheduleFixture {
    karras: KarrasSchedules,
    sigma_min: f32,
    sigma_max: f32,
    rho: f32,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct KarrasSchedules {
    #[serde(rename = "5_steps")]
    five_steps: Vec<f32>,
    #[serde(rename = "10_steps")]
    ten_steps: Vec<f32>,
    #[serde(rename = "20_steps")]
    twenty_steps: Vec<f32>,
}

#[test]
fn test_karras_sigmas_match_reference() {
    let fixture_path = fixtures_dir().join("sigma_schedules.json");
    let content = fs::read_to_string(&fixture_path).expect("Failed to read fixture");
    let fixture: SigmaScheduleFixture =
        serde_json::from_str(&content).expect("Failed to parse fixture");

    // Verify Karras schedule computation
    let computed = compute_karras_sigmas(5, fixture.sigma_min, fixture.sigma_max, fixture.rho);

    eprintln!("Computed 5-step Karras: {:?}", computed);

    assert_eq!(
        computed.len(),
        fixture.karras.five_steps.len(),
        "Length mismatch"
    );

    for (i, (expected, actual)) in fixture.karras.five_steps.iter().zip(&computed).enumerate() {
        assert_approx_eq(*expected, *actual, &format!("karras_5_step[{}]", i));
    }
}

/// Reference implementation of Karras sigma schedule for testing
fn compute_karras_sigmas(n_steps: usize, sigma_min: f32, sigma_max: f32, rho: f32) -> Vec<f32> {
    let mut sigmas = Vec::with_capacity(n_steps + 1);
    let min_inv_rho = sigma_min.powf(1.0 / rho);
    let max_inv_rho = sigma_max.powf(1.0 / rho);

    for i in 0..=n_steps {
        let ramp = i as f32 / n_steps as f32;
        let sigma = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)).powf(rho);
        sigmas.push(sigma);
    }
    sigmas[n_steps] = 0.0; // Final sigma is 0
    sigmas
}

/// Run this test to regenerate fixture values from Rust
/// cargo test -p burn-models-samplers --test sampler_fixtures -- --nocapture generate_fixtures --ignored
#[test]
#[ignore]
fn generate_fixtures() {
    let sigma_min = 0.0292_f32;
    let sigma_max = 14.6146_f32;
    let rho = 7.0_f32;

    eprintln!("=== Sigma Schedule Fixtures ===");
    for n in [5, 10, 20] {
        let sigmas = compute_karras_sigmas(n, sigma_min, sigma_max, rho);
        eprintln!("\"{}_steps\": {:?},", n, sigmas);
    }

    eprintln!("\n=== DPM++ 2M Intermediate Values ===");
    let sigmas = compute_karras_sigmas(5, sigma_min, sigma_max, rho);

    for i in 0..sigmas.len() - 1 {
        let sigma = sigmas[i];
        let sigma_next = sigmas[i + 1];
        let t = -(sigma.ln());
        let t_next = if sigma_next > 0.0 {
            -(sigma_next.ln())
        } else {
            f32::INFINITY
        };
        let h = t_next - t;

        eprintln!(
            "Step {}: sigma={:.6}, sigma_next={:.6}",
            i, sigma, sigma_next
        );
        eprintln!("  t={:.6}, t_next={:.6}, h={:.6}", t, t_next, h);

        if i > 0 {
            let old_sigma = sigmas[i - 1];
            let t_prev = -(old_sigma.ln());
            let h_prev = t - t_prev;
            let r = h_prev / h;
            let coeff = 1.0 / (2.0 * r);
            eprintln!(
                "  t_prev={:.6}, h_prev={:.6}, r={:.6}, coeff={:.6}",
                t_prev, h_prev, r, coeff
            );
        }
        eprintln!();
    }
}

// ============================================================================
// DPM++ 2M Tests
// ============================================================================

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct Dpm2mReference {
    formulas: Dpm2mFormulas,
    test_cases: Vec<Dpm2mTestCase>,
    coefficient_verification: CoeffVerification,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct Dpm2mFormulas {
    first_order: String,
    second_order: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct Dpm2mTestCase {
    name: String,
    sigma: f32,
    sigma_next: f32,
    #[serde(default)]
    old_sigma: Option<f32>,
    expected: Dpm2mExpected,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct Dpm2mExpected {
    t: f32,
    t_next: f32,
    h: f32,
    #[serde(default)]
    sigma_ratio: Option<f32>,
    #[serde(default)]
    exp_neg_h: Option<f32>,
    #[serde(default)]
    t_prev: Option<f32>,
    #[serde(default)]
    h_prev: Option<f32>,
    #[serde(default)]
    r: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct CoeffVerification {
    examples: Vec<CoeffExample>,
}

#[derive(Debug, Deserialize)]
struct CoeffExample {
    r: f32,
    coeff: f32,
    d_weight: f32,
    old_d_weight: f32,
}

#[test]
fn test_dpm_2m_intermediate_values() {
    let fixture_path = fixtures_dir().join("dpm_2m_reference.json");
    let content = fs::read_to_string(&fixture_path).expect("Failed to read fixture");
    let fixture: Dpm2mReference = serde_json::from_str(&content).expect("Failed to parse fixture");

    for case in &fixture.test_cases {
        let t = -(case.sigma.ln());
        let t_next = -(case.sigma_next.ln());
        let h = t_next - t;

        assert_approx_eq(case.expected.t, t, &format!("{}: t", case.name));
        assert_approx_eq(
            case.expected.t_next,
            t_next,
            &format!("{}: t_next", case.name),
        );
        assert_approx_eq(case.expected.h, h, &format!("{}: h", case.name));

        if let Some(old_sigma) = case.old_sigma {
            let t_prev = -(old_sigma.ln());
            let h_prev = t - t_prev;
            let r = h_prev / h;

            if let Some(expected_t_prev) = case.expected.t_prev {
                assert_approx_eq(expected_t_prev, t_prev, &format!("{}: t_prev", case.name));
            }
            if let Some(expected_h_prev) = case.expected.h_prev {
                assert_approx_eq(expected_h_prev, h_prev, &format!("{}: h_prev", case.name));
            }
            if let Some(expected_r) = case.expected.r {
                assert_approx_eq(expected_r, r, &format!("{}: r", case.name));
            }
        }
    }
}

#[test]
fn test_dpm_2m_second_order_coefficients() {
    let fixture_path = fixtures_dir().join("dpm_2m_reference.json");
    let content = fs::read_to_string(&fixture_path).expect("Failed to read fixture");
    let fixture: Dpm2mReference = serde_json::from_str(&content).expect("Failed to parse fixture");

    for example in &fixture.coefficient_verification.examples {
        let r = example.r;
        let coeff = 1.0 / (2.0 * r);
        let d_weight = 1.0 + coeff;
        let old_d_weight = -coeff;

        assert_approx_eq(example.coeff, coeff, &format!("r={}: coeff", r));
        assert_approx_eq(example.d_weight, d_weight, &format!("r={}: d_weight", r));
        assert_approx_eq(
            example.old_d_weight,
            old_d_weight,
            &format!("r={}: old_d_weight", r),
        );
    }
}

// ============================================================================
// Sampler Property Tests (no fixtures needed)
// ============================================================================

#[test]
fn test_karras_sigmas_decreasing() {
    let sigmas = compute_karras_sigmas(20, 0.0292, 14.6146, 7.0);

    for i in 1..sigmas.len() {
        assert!(
            sigmas[i] <= sigmas[i - 1],
            "Sigmas not decreasing at index {}: {} > {}",
            i,
            sigmas[i],
            sigmas[i - 1]
        );
    }
}

#[test]
fn test_karras_sigmas_bounds() {
    let sigmas = compute_karras_sigmas(20, 0.0292, 14.6146, 7.0);

    assert_approx_eq(14.6146, sigmas[0], "sigma_max");
    assert_approx_eq(0.0, sigmas[sigmas.len() - 1], "sigma_final");
}

#[test]
fn test_log_snr_space_is_increasing() {
    // In log-SNR space (t = -log(sigma)), t should be increasing as sigma decreases
    let sigmas = compute_karras_sigmas(10, 0.0292, 14.6146, 7.0);

    let mut prev_t = f32::NEG_INFINITY;
    for sigma in &sigmas[..sigmas.len() - 1] {
        // Skip final sigma=0
        let t = -(sigma.ln());
        assert!(
            t > prev_t,
            "log-SNR not increasing: prev_t={}, t={}, sigma={}",
            prev_t,
            t,
            sigma
        );
        prev_t = t;
    }
}
