"""
spectral_fidelity.py

Implements spectral band fidelity metrics for evaluating EEG signal quality.
Computes Power Spectral Density (PSD) across standard EEG frequency bands
and compares distributions between two sets of EEG segments
(e.g. real vs synthetic, or ictal vs interictal).

Standard EEG bands used:
    Delta : 0.5 - 4 Hz
    Theta : 4   - 8 Hz
    Alpha : 8   - 13 Hz
    Beta  : 13  - 30 Hz
    Gamma : 30  - 45 Hz
"""

import numpy as np
from scipy.signal import welch
from scipy.stats import wasserstein_distance


# ----------------------------------------------------------------------
# 1. Define standard EEG frequency bands
# ----------------------------------------------------------------------

EEG_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


# ----------------------------------------------------------------------
# 2. Compute PSD for a single segment
# ----------------------------------------------------------------------

def compute_psd(segment_data, sfreq, nperseg=256):
    """
    Computes Power Spectral Density (PSD) for a single EEG segment using Welch's method.

    Parameters:
        segment_data (np.ndarray): shape (channels, time)
        sfreq (float): sampling frequency in Hz
        nperseg (int): length of each FFT window

    Returns:
        freqs (np.ndarray): frequency bins
        psd (np.ndarray): shape (channels, freq_bins) — PSD per channel
    """
    freqs, psd = welch(segment_data, fs=sfreq, nperseg=min(nperseg, segment_data.shape[1]), axis=1)
    return freqs, psd


# ----------------------------------------------------------------------
# 3. Compute band power from PSD
# ----------------------------------------------------------------------

def compute_band_power(freqs, psd, band):
    """
    Computes average power within a given frequency band.

    Parameters:
        freqs (np.ndarray): frequency bins from compute_psd
        psd (np.ndarray): shape (channels, freq_bins)
        band (tuple): (low_freq, high_freq)

    Returns:
        np.ndarray: shape (channels,) — average power per channel in this band
    """
    low, high = band
    band_mask = (freqs >= low) & (freqs <= high)
    band_power = psd[:, band_mask].mean(axis=1)
    return band_power


# ----------------------------------------------------------------------
# 4. Compute full band power profile for one segment
# ----------------------------------------------------------------------

def compute_band_power_profile(segment_data, sfreq):
    """
    Computes average power across all standard EEG bands for one segment.

    Parameters:
        segment_data (np.ndarray): shape (channels, time)
        sfreq (float): sampling frequency in Hz

    Returns:
        dict: {band_name: np.ndarray of shape (channels,)}
    """
    freqs, psd = compute_psd(segment_data, sfreq)

    profile = {}
    for band_name, band_range in EEG_BANDS.items():
        profile[band_name] = compute_band_power(freqs, psd, band_range)

    return profile


# ----------------------------------------------------------------------
# 5. Compare band power distributions between two groups of segments
# ----------------------------------------------------------------------

def compute_spectral_fidelity(segments_real, segments_synthetic, sfreq):
    """
    Compares band power distributions between real and synthetic EEG segments
    using Wasserstein distance (Earth Mover's Distance) per band.

    Lower distance = more physiologically faithful synthetic data.

    Parameters:
        segments_real (list of np.ndarray): each shape (channels, time)
        segments_synthetic (list of np.ndarray): each shape (channels, time)
        sfreq (float): sampling frequency in Hz

    Returns:
        dict: {band_name: float distance score}
    """
    # collect band power values across all segments, averaged across channels
    real_band_values = {band: [] for band in EEG_BANDS}
    synthetic_band_values = {band: [] for band in EEG_BANDS}

    for segment in segments_real:
        profile = compute_band_power_profile(segment, sfreq)
        for band_name, power in profile.items():
            real_band_values[band_name].append(power.mean())

    for segment in segments_synthetic:
        profile = compute_band_power_profile(segment, sfreq)
        for band_name, power in profile.items():
            synthetic_band_values[band_name].append(power.mean())

    # compute distance per band
    fidelity_scores = {}
    for band_name in EEG_BANDS:
        real_vals = np.array(real_band_values[band_name])
        synth_vals = np.array(synthetic_band_values[band_name])
        distance = wasserstein_distance(real_vals, synth_vals)
        fidelity_scores[band_name] = distance

    return fidelity_scores


# ----------------------------------------------------------------------
# 6. Quick test / entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Quick sanity test using random data shaped like real EEG segments
    # (channels=23, time=2560 → matches your loader's 10-sec, 256Hz segments)

    sfreq = 256
    n_channels = 23
    n_time = 2560

    # simulate "real" segments
    real_segments = [np.random.randn(n_channels, n_time) for _ in range(20)]

    # simulate "synthetic" segments (slightly shifted distribution)
    synthetic_segments = [np.random.randn(n_channels, n_time) * 1.2 for _ in range(20)]

    scores = compute_spectral_fidelity(real_segments, synthetic_segments, sfreq)

    print("Spectral Fidelity Scores (lower = more similar):")
    for band, score in scores.items():
        print(f"  {band:8s}: {score:.6f}")