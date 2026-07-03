"""
subject_fingerprint.py

Implements subject-specific fingerprint retention metrics for evaluating
synthetic EEG quality. Real EEG carries individual neural signatures —
good synthetic data should preserve these subject-specific patterns
rather than producing generic, averaged-out signals.

Method: build a feature vector per segment (band power profile across
channels), then use those features to test whether segments can be
correctly matched back to their subject of origin (classification accuracy)
and/or measure how separable subjects are in feature space.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from synth_eeg_bench.metrics.spectral_fidelity import compute_band_power_profile, EEG_BANDS


# ----------------------------------------------------------------------
# 1. Build a feature vector for one segment
# ----------------------------------------------------------------------

def extract_fingerprint_features(segment_data, sfreq):
    """
    Extracts a feature vector representing a segment's "fingerprint":
    band power per channel, flattened into one vector.

    Parameters:
        segment_data (np.ndarray): shape (channels, time)
        sfreq (float): sampling frequency

    Returns:
        np.ndarray: 1D feature vector, shape (channels * n_bands,)
    """
    profile = compute_band_power_profile(segment_data, sfreq)
    # profile is {band_name: array of shape (channels,)}
    feature_vector = np.concatenate([profile[band] for band in EEG_BANDS])
    return feature_vector


# ----------------------------------------------------------------------
# 2. Build a dataset of features + subject labels
# ----------------------------------------------------------------------

def build_fingerprint_dataset(segments_by_subject, sfreq):
    """
    Builds a feature matrix and subject label array from segments grouped by subject.

    Parameters:
        segments_by_subject (dict): {subject_id: list of segment_data arrays}
        sfreq (float): sampling frequency

    Returns:
        X (np.ndarray): shape (n_segments, n_features)
        y (np.ndarray): shape (n_segments,) — subject_id labels
    """
    X = []
    y = []

    for subject_id, segments in segments_by_subject.items():
        for segment in segments:
            features = extract_fingerprint_features(segment, sfreq)
            X.append(features)
            y.append(subject_id)

    return np.array(X), np.array(y)


# ----------------------------------------------------------------------
# 3. Compute subject fingerprint retention score
# ----------------------------------------------------------------------

def compute_subject_fingerprint_score(segments_by_subject, sfreq, cv_folds=5):
    """
    Measures how well a subject can be identified from their EEG segments
    using a simple classifier on band-power fingerprint features.

    Higher accuracy = stronger subject-specific fingerprint retention.
    This can be run on real data (to confirm subjects ARE distinguishable)
    or on synthetic data (to check if the generator preserves this property).

    Parameters:
        segments_by_subject (dict): {subject_id: list of segment_data arrays}
        sfreq (float): sampling frequency
        cv_folds (int): number of cross-validation folds

    Returns:
        dict: {
            "mean_accuracy": float,
            "std_accuracy": float,
            "n_subjects": int,
            "n_segments": int
        }
    """
    X, y = build_fingerprint_dataset(segments_by_subject, sfreq)

    if len(np.unique(y)) < 2:
        raise ValueError("Need at least 2 subjects to compute fingerprint score.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000)

    # adjust cv folds if a class has too few samples
    min_class_count = min(np.bincount(np.unique(y, return_inverse=True)[1]))
    actual_folds = min(cv_folds, min_class_count)

    scores = cross_val_score(clf, X_scaled, y, cv=actual_folds)

    return {
        "mean_accuracy": scores.mean(),
        "std_accuracy": scores.std(),
        "n_subjects": len(np.unique(y)),
        "n_segments": len(y),
    }


# ----------------------------------------------------------------------
# 4. Quick test / entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    sfreq = 256
    n_channels = 23
    n_time = 2560

    # simulate 2 "subjects" with slightly different signal characteristics
    subject_a_segments = [np.random.randn(n_channels, n_time) * 1.0 for _ in range(15)]
    subject_b_segments = [np.random.randn(n_channels, n_time) * 1.5 for _ in range(15)]

    segments_by_subject = {
        "subject_a": subject_a_segments,
        "subject_b": subject_b_segments,
    }

    print("Computing subject fingerprint score on simulated data...")
    result = compute_subject_fingerprint_score(segments_by_subject, sfreq)

    print(f"\nMean classification accuracy: {result['mean_accuracy']:.4f}")
    print(f"Std deviation: {result['std_accuracy']:.4f}")
    print(f"Number of subjects: {result['n_subjects']}")
    print(f"Number of segments: {result['n_segments']}")
    print("\n(Accuracy near 0.5 = subjects indistinguishable; near 1.0 = strong fingerprint)")