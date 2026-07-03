"""
test_subject_fingerprint_real_data.py

Validates subject_fingerprint.py using real CHB-MIT data.
Tests whether chb01 and chb03 (two real, distinct patients) can be
correctly distinguished from their EEG band-power fingerprints.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from synth_eeg_bench.preprocessing.load_epilepsy_chbmit import load_subject
from synth_eeg_bench.metrics.subject_fingerprint import compute_subject_fingerprint_score


CHB01_DIR = "data/raw/epilepsy/chb_mit/chb01"
CHB01_SUMMARY = "data/raw/epilepsy/chb_mit/chb01/chb01-summary.txt"

CHB03_DIR = "data/raw/epilepsy/chb_mit/chb03"
CHB03_SUMMARY = "data/raw/epilepsy/chb_mit/chb03/chb03-summary.txt"


if __name__ == "__main__":
    print("Loading real CHB-MIT segments...")
    chb01_segments = load_subject(CHB01_DIR, CHB01_SUMMARY, segment_length_sec=10)
    chb03_segments = load_subject(CHB03_DIR, CHB03_SUMMARY, segment_length_sec=10)

    # use only interictal segments for a fair "baseline brain activity" fingerprint test
    chb01_interictal = [s["data"] for s in chb01_segments if s["label"] == "interictal"]
    chb03_interictal = [s["data"] for s in chb03_segments if s["label"] == "interictal"]

    sfreq = chb01_segments[0]["sfreq"]

    # subsample for speed (fingerprinting runs a classifier, not too heavy, but keep it light)
    import random
    random.seed(42)
    chb01_sample = random.sample(chb01_interictal, min(50, len(chb01_interictal)))
    chb03_sample = random.sample(chb03_interictal, min(50, len(chb03_interictal)))

    segments_by_subject = {
        "chb01": chb01_sample,
        "chb03": chb03_sample,
    }

    print(f"\nchb01 segments used: {len(chb01_sample)}")
    print(f"chb03 segments used: {len(chb03_sample)}")

    print("\nComputing subject fingerprint score on REAL data...")
    result = compute_subject_fingerprint_score(segments_by_subject, sfreq)

    print(f"\nMean classification accuracy: {result['mean_accuracy']:.4f}")
    print(f"Std deviation: {result['std_accuracy']:.4f}")
    print(f"Number of subjects: {result['n_subjects']}")
    print(f"Number of segments: {result['n_segments']}")
    print("\n(Accuracy near 0.5 = subjects indistinguishable; near 1.0 = strong fingerprint)")