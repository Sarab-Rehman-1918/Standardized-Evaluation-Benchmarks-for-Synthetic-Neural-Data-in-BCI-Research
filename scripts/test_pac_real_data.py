"""
test_pac_real_data.py

Validates phase_amplitude_coupling.py using real CHB-MIT data.
Compares ictal vs interictal segments — seizures typically show
stronger phase-amplitude coupling than baseline brain activity,
so a working PAC metric should reflect that difference.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from synth_eeg_bench.preprocessing.load_epilepsy_chbmit import load_subject
from synth_eeg_bench.metrics.phase_amplitude_coupling import compare_pac_fidelity


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

    # NOTE: PAC is computationally heavy (filtering + Hilbert transform per channel).
    # With ~2490 interictal segments this could take a while, so we subsample.
    import random
    random.seed(42)
    interictal_sample = random.sample(interictal_data, min(30, len(interictal_data)))

    print(f"\nUsing {len(ictal_data)} ictal segments and {len(interictal_sample)} interictal segments (subsampled for speed)")

    print("\n--- Comparing Ictal vs Interictal PAC ---")
    result = compare_pac_fidelity(interictal_sample, ictal_data, sfreq)
    print(f"Interictal mean MI: {result['real_mean_mi']:.6f}")
    print(f"Ictal mean MI:      {result['synthetic_mean_mi']:.6f}")
    print(f"Absolute difference: {result['absolute_difference']:.6f}")

    # sanity baseline: split interictal sample in half
    half = len(interictal_sample) // 2
    interictal_a = interictal_sample[:half]
    interictal_b = interictal_sample[half:]

    print("\n--- Sanity Baseline: Interictal vs Interictal (should be near zero) ---")
    baseline_result = compare_pac_fidelity(interictal_a, interictal_b, sfreq)
    print(f"Group A mean MI: {baseline_result['real_mean_mi']:.6f}")
    print(f"Group B mean MI: {baseline_result['synthetic_mean_mi']:.6f}")
    print(f"Absolute difference: {baseline_result['absolute_difference']:.6f}")