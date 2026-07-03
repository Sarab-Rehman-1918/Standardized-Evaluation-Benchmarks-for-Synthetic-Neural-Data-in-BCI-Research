"""
run_benchmark.py

Orchestrates the full benchmark pipeline — runs all three generators
through all four metrics and the safety protocol, then collects
results into a structured leaderboard table.

This is the single entry point for reproducing the full benchmark.
Output is saved to data/results/benchmark_results.json for use
by generate_report.py.
"""

import sys
import os
import json
import random
import numpy as np
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from synth_eeg_bench.preprocessing.load_epilepsy_chbmit import load_subject

from synth_eeg_bench.metrics.spectral_fidelity import compute_spectral_fidelity
from synth_eeg_bench.metrics.phase_amplitude_coupling import compare_pac_fidelity
from synth_eeg_bench.metrics.subject_fingerprint import compute_subject_fingerprint_score
from synth_eeg_bench.metrics.cross_session_stability import compute_cross_session_stability

from synth_eeg_bench.generators.vae_baseline import (
    train_vae,
    generate_synthetic_segments as vae_generate,
)
from synth_eeg_bench.generators.gan_baseline import (
    train_gan,
    generate_synthetic_segments as gan_generate,
)
from synth_eeg_bench.generators.diffusion_baseline import (
    train_diffusion,
    generate_synthetic_segments as diffusion_generate,
)

from synth_eeg_bench.safety_protocol.train_downstream_classifier import run_safety_protocol
from synth_eeg_bench.safety_protocol.evaluate_safety import check_safety_thresholds


# ----------------------------------------------------------------------
# Dataset paths
# ----------------------------------------------------------------------

CHB01_DIR = "data/raw/epilepsy/chb_mit/chb01"
CHB01_SUMMARY = "data/raw/epilepsy/chb_mit/chb01/chb01-summary.txt"
CHB03_DIR = "data/raw/epilepsy/chb_mit/chb03"
CHB03_SUMMARY = "data/raw/epilepsy/chb_mit/chb03/chb03-summary.txt"

RESULTS_PATH = "data/results/benchmark_results.json"


# ----------------------------------------------------------------------
# 1. Load and prepare real data
# ----------------------------------------------------------------------

def load_real_data():
    print("=" * 60)
    print("STEP 1: Loading real CHB-MIT data")
    print("=" * 60)

    chb01_segments = load_subject(CHB01_DIR, CHB01_SUMMARY, segment_length_sec=10)
    chb03_segments = load_subject(CHB03_DIR, CHB03_SUMMARY, segment_length_sec=10)

    all_segments = chb01_segments + chb03_segments
    sfreq = all_segments[0]["sfreq"]

    real_interictal = [s["data"] for s in all_segments if s["label"] == "interictal"]
    real_ictal = [s["data"] for s in all_segments if s["label"] == "ictal"]

    print(f"Interictal segments: {len(real_interictal)}")
    print(f"Ictal segments:      {len(real_ictal)}")

    return real_interictal, real_ictal, sfreq


# ----------------------------------------------------------------------
# 2. Run all metrics for one generator's synthetic output
# ----------------------------------------------------------------------

def run_metrics(
    generator_name,
    synthetic_segments,
    real_interictal,
    real_ictal,
    sfreq,
):
    print(f"\n  Running metrics for {generator_name}...")
    metrics = {}

    # sample real data to match synthetic count for fair comparison
    random.seed(42)
    real_sample = random.sample(real_interictal, min(len(synthetic_segments), len(real_interictal)))

    # ------------------------------------------------------------------
    # Metric 1: Spectral Fidelity
    # ------------------------------------------------------------------
    print("    [1/4] Spectral Fidelity...")
    spectral_scores = compute_spectral_fidelity(real_sample, synthetic_segments, sfreq)
    metrics["spectral_fidelity"] = {
        band: float(score) for band, score in spectral_scores.items()
    }
    metrics["spectral_fidelity_mean"] = float(np.mean(list(spectral_scores.values())))

    # ------------------------------------------------------------------
    # Metric 2: Phase-Amplitude Coupling
    # ------------------------------------------------------------------
    print("    [2/4] Phase-Amplitude Coupling...")
    pac_result = compare_pac_fidelity(
        real_sample[:10],       # subsample heavily — PAC is slow
        synthetic_segments[:10],
        sfreq,
    )
    metrics["pac"] = {
        "real_mean_mi": float(pac_result["real_mean_mi"]),
        "synthetic_mean_mi": float(pac_result["synthetic_mean_mi"]),
        "absolute_difference": float(pac_result["absolute_difference"]),
    }

    # ------------------------------------------------------------------
    # Metric 3: Subject Fingerprint
    # ------------------------------------------------------------------
    print("    [3/4] Subject Fingerprint...")
    half = len(synthetic_segments) // 2
    segments_by_subject = {
        "real": real_sample[:half],
        "synthetic": synthetic_segments[:half],
    }
    fingerprint_result = compute_subject_fingerprint_score(segments_by_subject, sfreq)
    metrics["subject_fingerprint"] = {
        "mean_accuracy": float(fingerprint_result["mean_accuracy"]),
        "std_accuracy": float(fingerprint_result["std_accuracy"]),
    }

    # ------------------------------------------------------------------
    # Metric 4: Cross-Session Stability
    # ------------------------------------------------------------------
    print("    [4/4] Cross-Session Stability...")
    n = len(synthetic_segments) // 3
    segments_by_subject_session = {
        "real": {
            "session_1": real_sample[:n],
            "session_2": real_sample[n:2*n],
        },
        "synthetic": {
            "session_1": synthetic_segments[:n],
            "session_2": synthetic_segments[n:2*n],
        },
    }
    stability_result = compute_cross_session_stability(
        segments_by_subject_session, sfreq
    )
    metrics["cross_session_stability"] = {
        "within_subject_distance": float(stability_result["mean_within_subject_distance"]),
        "between_subject_distance": float(stability_result["mean_between_subject_distance"]),
        "stability_ratio": float(stability_result["stability_ratio"]),
    }

    return metrics


# ----------------------------------------------------------------------
# 3. Run safety protocol for one generator
# ----------------------------------------------------------------------

def run_safety(
    generator_name,
    synthetic_segments,
    real_interictal,
    real_ictal,
    sfreq,
):
    print(f"\n  Running safety protocol for {generator_name}...")

    random.seed(42)
    real_sample = random.sample(real_interictal, min(100, len(real_interictal)))
    real_segments = real_sample + real_ictal
    real_labels = [0] * len(real_sample) + [1] * len(real_ictal)

    synth_labels = [0] * len(synthetic_segments)

    results = run_safety_protocol(
        real_segments, real_labels,
        synthetic_segments, synth_labels,
        sfreq,
    )

    threshold_check = check_safety_thresholds(results["synthetic_only"])

    return {
        "real_only": {
            "accuracy": float(results["real_only"]["accuracy"]),
            "f1_score": float(results["real_only"]["f1_score"]),
            "false_positive_rate": float(results["real_only"]["false_positive_rate"]),
        },
        "synthetic_only": {
            "accuracy": float(results["synthetic_only"]["accuracy"]),
            "f1_score": float(results["synthetic_only"]["f1_score"]),
            "false_positive_rate": float(results["synthetic_only"]["false_positive_rate"]),
        },
        "mixed": {
            "accuracy": float(results["mixed"]["accuracy"]),
            "f1_score": float(results["mixed"]["f1_score"]),
            "false_positive_rate": float(results["mixed"]["false_positive_rate"]),
        },
        "certified_safe": bool(threshold_check["overall_passed"]),
    }


# ----------------------------------------------------------------------
# 4. Full benchmark runner
# ----------------------------------------------------------------------

def run_full_benchmark():
    random.seed(42)
    np.random.seed(42)

    real_interictal, real_ictal, sfreq = load_real_data()

    # subsample training pool for speed
    train_pool = random.sample(real_interictal, min(150, len(real_interictal)))

    leaderboard = {
        "benchmark_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "CHB-MIT (chb01 + chb03)",
        "sfreq": sfreq,
        "n_real_interictal": len(real_interictal),
        "n_real_ictal": len(real_ictal),
        "generators": {},
    }

    # ==================================================================
    # Generator 1: VAE
    # ==================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Training and evaluating VAE")
    print("=" * 60)

    vae_model, vae_mean, vae_std, vae_shape = train_vae(
        train_pool, latent_dim=64, epochs=20, batch_size=16, learning_rate=1e-4
    )
    vae_synthetic = vae_generate(
        vae_model, vae_mean, vae_std, vae_shape, n_samples=60
    )

    leaderboard["generators"]["VAE"] = {
        "metrics": run_metrics("VAE", vae_synthetic, real_interictal, real_ictal, sfreq),
        "safety": run_safety("VAE", vae_synthetic, real_interictal, real_ictal, sfreq),
    }

    # ==================================================================
    # Generator 2: GAN
    # ==================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Training and evaluating GAN")
    print("=" * 60)

    gan_gen, gan_mean, gan_std, gan_shape = train_gan(
        train_pool, latent_dim=64, epochs=30, batch_size=16, learning_rate=2e-4
    )
    gan_synthetic = gan_generate(
        gan_gen, gan_mean, gan_std, gan_shape, n_samples=60
    )

    leaderboard["generators"]["GAN"] = {
        "metrics": run_metrics("GAN", gan_synthetic, real_interictal, real_ictal, sfreq),
        "safety": run_safety("GAN", gan_synthetic, real_interictal, real_ictal, sfreq),
    }

    # ==================================================================
    # Generator 3: Diffusion
    # ==================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Training and evaluating Diffusion Model")
    print("=" * 60)

    diff_model, diff_schedule, diff_mean, diff_std, diff_shape = train_diffusion(
        train_pool, timesteps=100, epochs=20, batch_size=16, learning_rate=1e-3
    )
    diff_synthetic = diffusion_generate(
        diff_model, diff_schedule, diff_mean, diff_std, diff_shape,
        n_samples=60, timesteps=100
    )

    leaderboard["generators"]["Diffusion"] = {
        "metrics": run_metrics("Diffusion", diff_synthetic, real_interictal, real_ictal, sfreq),
        "safety": run_safety("Diffusion", diff_synthetic, real_interictal, real_ictal, sfreq),
    }

    # ==================================================================
    # Save results
    # ==================================================================
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(leaderboard, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Benchmark complete. Results saved to {RESULTS_PATH}")
    print(f"{'=' * 60}")

    return leaderboard


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    run_full_benchmark()