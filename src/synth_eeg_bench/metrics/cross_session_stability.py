"""
cross_session_stability.py

Implements cross-session stability metrics for evaluating synthetic EEG quality.
Real EEG should show a degree of consistency in a subject's characteristic
patterns (e.g. band power fingerprint) across different recording sessions,
even though some natural variability exists. Synthetic data that fails to
preserve this stability is a sign of unrealistic generation.

Method: compute fingerprint feature vectors per session (grouped by file/session),
then measure within-subject variability (across sessions) vs between-subject
variability. A stable, realistic signal should have lower within-subject
variability than between-subject variability.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from synth_eeg_bench.metrics.subject_fingerprint import extract_fingerprint_features


# ----------------------------------------------------------------------
# 1. Compute average fingerprint per session
# ----------------------------------------------------------------------

def compute_session_fingerprint(segments, sfreq):
    """
    Computes an averaged fingerprint feature vector for one session
    (averaging across all segments belonging to that session).

    Parameters:
        segments (list of np.ndarray): each shape (channels, time)
        sfreq (float): sampling frequency

    Returns:
        np.ndarray: averaged feature vector for this session
    """
    features = [extract_fingerprint_features(seg, sfreq) for seg in segments]
    return np.mean(features, axis=0)


# ----------------------------------------------------------------------
# 2. Build session-level fingerprints grouped by subject
# ----------------------------------------------------------------------

def build_session_fingerprints(segments_by_subject_and_session, sfreq):
    """
    Computes one fingerprint vector per (subject, session) pair.

    Parameters:
        segments_by_subject_and_session (dict):
            {subject_id: {session_id: list of segment_data arrays}}
        sfreq (float): sampling frequency

    Returns:
        dict: {subject_id: {session_id: feature_vector}}
    """
    fingerprints = {}

    for subject_id, sessions in segments_by_subject_and_session.items():
        fingerprints[subject_id] = {}
        for session_id, segments in sessions.items():
            if len(segments) == 0:
                continue
            fingerprints[subject_id][session_id] = compute_session_fingerprint(segments, sfreq)

    return fingerprints


# ----------------------------------------------------------------------
# 3. Compute within-subject vs between-subject variability
# ----------------------------------------------------------------------

def compute_cross_session_stability(segments_by_subject_and_session, sfreq):
    """
    Measures cross-session stability: compares within-subject variability
    (across that subject's sessions) to between-subject variability.

    A physiologically valid result: within-subject distance < between-subject distance,
    meaning a subject's sessions are more similar to each other than to other subjects.

    Parameters:
        segments_by_subject_and_session (dict):
            {subject_id: {session_id: list of segment_data arrays}}
        sfreq (float): sampling frequency

    Returns:
        dict: {
            "mean_within_subject_distance": float,
            "mean_between_subject_distance": float,
            "stability_ratio": float  # between / within; >1 means good stability
        }
    """
    fingerprints = build_session_fingerprints(segments_by_subject_and_session, sfreq)

    within_distances = []
    between_distances = []

    subject_ids = list(fingerprints.keys())

    # within-subject: distances between sessions of the SAME subject
    for subject_id in subject_ids:
        sessions = list(fingerprints[subject_id].values())
        for i in range(len(sessions)):
            for j in range(i + 1, len(sessions)):
                dist = np.linalg.norm(sessions[i] - sessions[j])
                within_distances.append(dist)

    # between-subject: distances between sessions of DIFFERENT subjects
    for i in range(len(subject_ids)):
        for j in range(i + 1, len(subject_ids)):
            sessions_a = list(fingerprints[subject_ids[i]].values())
            sessions_b = list(fingerprints[subject_ids[j]].values())
            for sa in sessions_a:
                for sb in sessions_b:
                    dist = np.linalg.norm(sa - sb)
                    between_distances.append(dist)

    mean_within = np.mean(within_distances) if within_distances else float("nan")
    mean_between = np.mean(between_distances) if between_distances else float("nan")

    stability_ratio = mean_between / mean_within if mean_within > 0 else float("inf")

    return {
        "mean_within_subject_distance": mean_within,
        "mean_between_subject_distance": mean_between,
        "stability_ratio": stability_ratio,
    }


# ----------------------------------------------------------------------
# 4. Quick test / entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    sfreq = 256
    n_channels = 23
    n_time = 2560

    # simulate 2 subjects, each with 3 "sessions" (slightly varying noise per session)
    def make_session(scale, n_segments=5):
        return [np.random.randn(n_channels, n_time) * scale for _ in range(n_segments)]

    segments_by_subject_and_session = {
        "subject_a": {
            "session_1": make_session(1.0),
            "session_2": make_session(1.05),
            "session_3": make_session(0.95),
        },
        "subject_b": {
            "session_1": make_session(2.0),
            "session_2": make_session(2.05),
            "session_3": make_session(1.95),
        },
    }

    print("Computing cross-session stability on simulated data...")
    result = compute_cross_session_stability(segments_by_subject_and_session, sfreq)

    print(f"\nMean within-subject distance:  {result['mean_within_subject_distance']:.4f}")
    print(f"Mean between-subject distance: {result['mean_between_subject_distance']:.4f}")
    print(f"Stability ratio (between/within): {result['stability_ratio']:.4f}")
    print("\n(Ratio > 1 = good: subject is more consistent within itself than vs others)")