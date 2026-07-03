"""
train_vae_real_data.py

Trains the VAE baseline on real CHB-MIT interictal EEG segments
and generates synthetic samples, then runs spectral fidelity
to do a quick quality check on the generated data.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from synth_eeg_bench.preprocessing.load_epilepsy_chbmit import load_subject
from synth_eeg_bench.generators.vae_baseline import (
    train_vae, generate_synthetic_segments, save_vae
)
from synth_eeg_bench.metrics.spectral_fidelity import compute_spectral_fidelity


CHB01_DIR = "data/raw/epilepsy/chb_mit/chb01"
CHB01_SUMMARY = "data/raw/epilepsy/chb_mit/chb01/chb01-summary.txt"
CHB03_DIR = "data/raw/epilepsy/chb_mit/chb03"
CHB03_SUMMARY = "data/raw/epilepsy/chb_mit/chb03/chb03-summary.txt"


if __name__ == "__main__":
    print("Loading real CHB-MIT segments...")
    chb01_segments = load_subject(CHB01_DIR, CHB01_SUMMARY, segment_length_sec=10)
    chb03_segments = load_subject(CHB03_DIR, CHB03_SUMMARY, segment_length_sec=10)

    all_segments = chb01_segments + chb03_segments

    # use only interictal segments for training
    # (enough data, consistent baseline brain activity)
    interictal = [s["data"] for s in all_segments if s["label"] == "interictal"]
    sfreq = all_segments[0]["sfreq"]

    print(f"Total interictal segments for training: {len(interictal)}")

    # subsample to keep training time reasonable on CPU
    import random
    random.seed(42)
    train_segments = random.sample(interictal, min(200, len(interictal)))
    print(f"Using {len(train_segments)} segments for training (subsampled for speed)")

    # train VAE
    model, mean, std, segment_shape = train_vae(
        train_segments,
        latent_dim=64,
        epochs=30,
        batch_size=16,
        learning_rate=1e-4,
    )

    # save model
    save_vae(model, path="data/synthetic/vae_model.pt")

    # generate synthetic segments
    print("\nGenerating synthetic segments...")
    synthetic_segments = generate_synthetic_segments(
        model, mean, std, segment_shape, n_samples=50
    )
    print(f"Generated {len(synthetic_segments)} synthetic segments")
    print(f"Shape: {synthetic_segments[0].shape}")

    # quick quality check: spectral fidelity between real and synthetic
    print("\nRunning spectral fidelity check (real vs synthetic)...")
    real_sample = random.sample(interictal, 50)
    scores = compute_spectral_fidelity(real_sample, synthetic_segments, sfreq)

    print("\nSpectral Fidelity Scores (lower = more similar to real):")
    for band, score in scores.items():
        print(f"  {band:8s}: {score:.6f}")
        