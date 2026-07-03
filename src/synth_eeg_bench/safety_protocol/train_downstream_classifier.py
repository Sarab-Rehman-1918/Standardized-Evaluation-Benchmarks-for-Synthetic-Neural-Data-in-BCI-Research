"""
train_downstream_classifier.py

Implements the downstream classifier safety protocol.
Trains an EEG pathology detector (seizure vs non-seizure) on different
training data conditions and evaluates performance on real held-out data.

This directly addresses Objective 3 from the proposal:
"measuring the impact of synthetic training data on clinical decision
model accuracy and false-positive rates in pathology detection tasks."

Training conditions tested:
    1. Real data only        (baseline/gold standard)
    2. Synthetic data only   (worst case — no real data)
    3. Mixed real+synthetic  (augmentation scenario)
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from synth_eeg_bench.metrics.spectral_fidelity import compute_band_power_profile, EEG_BANDS


# ----------------------------------------------------------------------
# 1. Feature Extraction for Classifier
# ----------------------------------------------------------------------

def extract_classifier_features(segment_data, sfreq):
    """
    Extracts a feature vector from one EEG segment for the downstream classifier.
    Uses band power profile (same as fingerprinting) as features.

    Parameters:
        segment_data (np.ndarray): shape (channels, time)
        sfreq (float): sampling frequency

    Returns:
        np.ndarray: 1D feature vector shape (channels * n_bands,)
    """
    profile = compute_band_power_profile(segment_data, sfreq)
    feature_vector = np.concatenate([profile[band] for band in EEG_BANDS])
    return feature_vector


def build_feature_matrix(segments, labels, sfreq):
    """
    Builds feature matrix X and label array y from a list of segments.

    Parameters:
        segments (list of np.ndarray): each shape (channels, time)
        labels (list of int): 0 = interictal, 1 = ictal
        sfreq (float): sampling frequency

    Returns:
        X (np.ndarray): shape (n_segments, n_features)
        y (np.ndarray): shape (n_segments,)
    """
    X = np.array([extract_classifier_features(seg, sfreq) for seg in segments])
    y = np.array(labels)
    return X, y


# ----------------------------------------------------------------------
# 2. Train Downstream Classifier
# ----------------------------------------------------------------------

def train_classifier(X_train, y_train):
    """
    Trains a Random Forest classifier on the given feature matrix.
    Random Forest is chosen because:
        - handles class imbalance reasonably well
        - robust to feature scale differences
        - interpretable feature importances

    Parameters:
        X_train (np.ndarray): training features
        y_train (np.ndarray): training labels

    Returns:
        clf: trained classifier
        scaler: fitted StandardScaler
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    clf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",  # handles ictal/interictal imbalance
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_scaled, y_train)

    return clf, scaler


# ----------------------------------------------------------------------
# 3. Evaluate Classifier
# ----------------------------------------------------------------------

def evaluate_classifier(clf, scaler, X_test, y_test):
    """
    Evaluates a trained classifier on real held-out test data.

    Parameters:
        clf: trained classifier
        scaler: fitted StandardScaler from training
        X_test (np.ndarray): test features
        y_test (np.ndarray): test labels

    Returns:
        dict: evaluation metrics
    """
    X_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_scaled)

    cm = confusion_matrix(y_test, y_pred)

    # extract false positive rate from confusion matrix
    # FPR = FP / (FP + TN)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "false_positive_rate": fpr,
        "confusion_matrix": cm,
        "classification_report": classification_report(
            y_test, y_pred,
            target_names=["interictal", "ictal"],
            zero_division=0
        ),
    }


# ----------------------------------------------------------------------
# 4. Run All Training Conditions
# ----------------------------------------------------------------------

def run_safety_protocol(
    real_segments, real_labels,
    synthetic_segments, synthetic_labels,
    sfreq,
    test_size=0.2,
):
    """
    Runs all three training conditions and evaluates each on real held-out data.

    Parameters:
        real_segments (list of np.ndarray): real EEG segments
        real_labels (list of int): labels for real segments
        synthetic_segments (list of np.ndarray): synthetic EEG segments
        synthetic_labels (list of int): labels for synthetic segments
        sfreq (float): sampling frequency
        test_size (float): fraction of real data held out for testing

    Returns:
        dict: results for each training condition
    """
    print("Extracting features from real segments...")
    X_real, y_real = build_feature_matrix(real_segments, real_labels, sfreq)

    print("Extracting features from synthetic segments...")
    X_synth, y_synth = build_feature_matrix(synthetic_segments, synthetic_labels, sfreq)

    # split real data into train/test
    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        X_real, y_real, test_size=test_size, random_state=42, stratify=y_real
    )

    results = {}

    # ------------------------------------------------------------------
    # Condition 1: Train on REAL data only (gold standard baseline)
    # ------------------------------------------------------------------
    print("\nCondition 1: Training on real data only...")
    clf_real, scaler_real = train_classifier(X_real_train, y_real_train)
    results["real_only"] = evaluate_classifier(clf_real, scaler_real, X_real_test, y_real_test)

    # ------------------------------------------------------------------
    # Condition 2: Train on SYNTHETIC data only
    # ------------------------------------------------------------------
    print("Condition 2: Training on synthetic data only...")
    clf_synth, scaler_synth = train_classifier(X_synth, y_synth)
    results["synthetic_only"] = evaluate_classifier(clf_synth, scaler_synth, X_real_test, y_real_test)

    # ------------------------------------------------------------------
    # Condition 3: Train on MIXED real + synthetic data
    # ------------------------------------------------------------------
    print("Condition 3: Training on mixed real + synthetic data...")
    X_mixed = np.concatenate([X_real_train, X_synth])
    y_mixed = np.concatenate([y_real_train, y_synth])
    clf_mixed, scaler_mixed = train_classifier(X_mixed, y_mixed)
    results["mixed"] = evaluate_classifier(clf_mixed, scaler_mixed, X_real_test, y_real_test)

    return results


# ----------------------------------------------------------------------
# 5. Quick test / entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    sfreq = 256
    n_channels = 23
    n_time = 2560

    print("Testing downstream classifier with simulated data...")

    # simulate real segments: 200 interictal + 20 ictal
    real_interictal = [np.random.randn(n_channels, n_time) * 1.0
                       for _ in range(200)]
    real_ictal = [np.random.randn(n_channels, n_time) * 2.0
                  for _ in range(20)]

    real_segments = real_interictal + real_ictal
    real_labels = [0] * 200 + [1] * 20

    # simulate synthetic segments: 150 interictal + 15 ictal
    synth_interictal = [np.random.randn(n_channels, n_time) * 1.1
                        for _ in range(150)]
    synth_ictal = [np.random.randn(n_channels, n_time) * 1.9
                   for _ in range(15)]

    synthetic_segments = synth_interictal + synth_ictal
    synthetic_labels = [0] * 150 + [1] * 15

    results = run_safety_protocol(
        real_segments, real_labels,
        synthetic_segments, synthetic_labels,
        sfreq,
    )

    print("\n" + "=" * 50)
    print("DOWNSTREAM CLASSIFIER SAFETY RESULTS")
    print("=" * 50)

    for condition, metrics in results.items():
        print(f"\n--- {condition.upper()} ---")
        print(f"  Accuracy:            {metrics['accuracy']:.4f}")
        print(f"  F1 Score:            {metrics['f1_score']:.4f}")
        print(f"  False Positive Rate: {metrics['false_positive_rate']:.4f}")