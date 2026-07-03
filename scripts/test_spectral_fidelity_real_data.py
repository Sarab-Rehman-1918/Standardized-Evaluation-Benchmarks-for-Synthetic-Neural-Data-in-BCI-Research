"""
test_spectral_fidelity_real_data.py

Validates spectral_fidelity.py using real CHB-MIT data instead of random noise.
Compares ictal vs interictal segments — since seizures have known spectral
signatures, a working metric should show a clear difference between them.
"""

import sys
import os

# allow imports from src/
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from synth_eeg_bench.preprocessing.load_epilepsy_chbmit import load_subject
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

    ictal_data = [s["data"] for s in all_segments if s["label"] == "ictal"]
    interictal_data = [s["data"] for s in all_segments if s["label"] == "interictal"]

    sfreq = all_segments[0]["sfreq"]

    print(f"\nIctal segments: {len(ictal_data)}")
    print(f"Interictal segments: {len(interictal_data)}")

    print("\n--- Comparing Ictal vs Interictal ---")
    scores_real_comparison = compute_spectral_fidelity(interictal_data, ictal_data, sfreq)
    for band, score in scores_real_comparison.items():
        print(f"  {band:8s}: {score:.6f}")

    # sanity baseline: split interictal data in half and compare to itself
    half = len(interictal_data) // 2
    interictal_a = interictal_data[:half]
    interictal_b = interictal_data[half:]

    print("\n--- Sanity Baseline: Interictal vs Interictal (should be near zero) ---")
    scores_baseline = compute_spectral_fidelity(interictal_a, interictal_b, sfreq)
    for band, score in scores_baseline.items():
        print(f"  {band:8s}: {score:.6f}")