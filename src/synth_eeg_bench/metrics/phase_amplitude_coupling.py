"""
phase_amplitude_coupling.py

Implements Phase-Amplitude Coupling (PAC) metrics for evaluating EEG signal quality.
PAC measures how the phase of a low-frequency band modulates the amplitude
of a high-frequency band — a key neurophysiological property real EEG exhibits.
Synthetic EEG that lacks this coupling structure is a sign of unrealistic
"hallucinated" signal generation, even if it passes simpler spectral checks.

Method used: Modulation Index (MI) based on Tort et al. (2010).
"""

import numpy as np
from scipy.signal import hilbert, butter, filtfilt


# ----------------------------------------------------------------------
# 1. Bandpass filter helper
# ----------------------------------------------------------------------

def bandpass_filter(signal, sfreq, low, high, order=4):
    """
    Applies a Butterworth bandpass filter to a 1D signal.

    Parameters:
        signal (np.ndarray): shape (time,)
        sfreq (float): sampling frequency in Hz
        low (float): low cutoff frequency
        high (float): high cutoff frequency
        order (int): filter order

    Returns:
        np.ndarray: filtered signal, same shape as input
    """
    nyquist = sfreq / 2
    b, a = butter(order, [low / nyquist, high / nyquist], btype="band")
    filtered = filtfilt(b, a, signal)
    return filtered


# ----------------------------------------------------------------------
# 2. Extract phase and amplitude using Hilbert transform
# ----------------------------------------------------------------------

def get_phase(signal, sfreq, low, high):
    """Extracts instantaneous phase of a signal within a frequency band."""
    filtered = bandpass_filter(signal, sfreq, low, high)
    analytic_signal = hilbert(filtered)
    phase = np.angle(analytic_signal)
    return phase


def get_amplitude(signal, sfreq, low, high):
    """Extracts instantaneous amplitude envelope of a signal within a frequency band."""
    filtered = bandpass_filter(signal, sfreq, low, high)
    analytic_signal = hilbert(filtered)
    amplitude = np.abs(analytic_signal)
    return amplitude


# ----------------------------------------------------------------------
# 3. Compute Modulation Index (Tort et al., 2010)
# ----------------------------------------------------------------------

def compute_modulation_index(phase_signal, amplitude_signal, n_bins=18):
    """
    Computes the Modulation Index (MI) — a measure of phase-amplitude coupling strength.
    Higher MI = stronger coupling between phase and amplitude.

    Parameters:
        phase_signal (np.ndarray): instantaneous phase, shape (time,)
        amplitude_signal (np.ndarray): instantaneous amplitude, shape (time,)
        n_bins (int): number of phase bins (default 18, standard in literature)

    Returns:
        float: Modulation Index (0 = no coupling, higher = stronger coupling)
    """
    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    mean_amplitude = np.zeros(n_bins)

    for i in range(n_bins):
        bin_mask = (phase_signal >= phase_bins[i]) & (phase_signal < phase_bins[i + 1])
        if np.sum(bin_mask) > 0:
            mean_amplitude[i] = amplitude_signal[bin_mask].mean()
        else:
            mean_amplitude[i] = 0

    # normalize to get a probability distribution
    if mean_amplitude.sum() == 0:
        return 0.0

    p = mean_amplitude / mean_amplitude.sum()

    # avoid log(0)
    p_safe = np.where(p > 0, p, 1e-12)

    # Shannon entropy of the distribution
    entropy = -np.sum(p_safe * np.log(p_safe))
    max_entropy = np.log(n_bins)

    # Modulation Index = normalized deviation from uniform distribution
    mi = (max_entropy - entropy) / max_entropy
    return mi


# ----------------------------------------------------------------------
# 4. Compute PAC for one EEG segment (single channel)
# ----------------------------------------------------------------------

def compute_pac_single_channel(channel_data, sfreq, phase_band=(4, 8), amplitude_band=(30, 45)):
    """
    Computes PAC (Modulation Index) for a single EEG channel.

    Parameters:
        channel_data (np.ndarray): shape (time,)
        sfreq (float): sampling frequency
        phase_band (tuple): low-frequency band for phase (default theta: 4-8 Hz)
        amplitude_band (tuple): high-frequency band for amplitude (default gamma: 30-45 Hz)

    Returns:
        float: Modulation Index for this channel
    """
    phase = get_phase(channel_data, sfreq, *phase_band)
    amplitude = get_amplitude(channel_data, sfreq, *amplitude_band)
    mi = compute_modulation_index(phase, amplitude)
    return mi


# ----------------------------------------------------------------------
# 5. Compute PAC across all channels in a segment
# ----------------------------------------------------------------------

def compute_pac_segment(segment_data, sfreq, phase_band=(4, 8), amplitude_band=(30, 45)):
    """
    Computes PAC for every channel in a multi-channel EEG segment.

    Parameters:
        segment_data (np.ndarray): shape (channels, time)
        sfreq (float): sampling frequency
        phase_band (tuple): phase frequency band
        amplitude_band (tuple): amplitude frequency band

    Returns:
        np.ndarray: shape (channels,) — MI per channel
    """
    n_channels = segment_data.shape[0]
    mi_values = np.zeros(n_channels)

    for ch in range(n_channels):
        mi_values[ch] = compute_pac_single_channel(
            segment_data[ch], sfreq, phase_band, amplitude_band
        )

    return mi_values


# ----------------------------------------------------------------------
# 6. Compare PAC distributions between two groups (e.g. real vs synthetic)
# ----------------------------------------------------------------------

def compare_pac_fidelity(segments_real, segments_synthetic, sfreq,
                          phase_band=(4, 8), amplitude_band=(30, 45)):
    """
    Compares average PAC strength between two groups of EEG segments.

    Parameters:
        segments_real (list of np.ndarray): each shape (channels, time)
        segments_synthetic (list of np.ndarray): each shape (channels, time)
        sfreq (float): sampling frequency
        phase_band (tuple): phase frequency band
        amplitude_band (tuple): amplitude frequency band

    Returns:
        dict: {
            "real_mean_mi": float,
            "synthetic_mean_mi": float,
            "absolute_difference": float
        }
    """
    real_mi_values = []
    for segment in segments_real:
        mi = compute_pac_segment(segment, sfreq, phase_band, amplitude_band)
        real_mi_values.append(mi.mean())

    synthetic_mi_values = []
    for segment in segments_synthetic:
        mi = compute_pac_segment(segment, sfreq, phase_band, amplitude_band)
        synthetic_mi_values.append(mi.mean())

    real_mean = np.mean(real_mi_values)
    synthetic_mean = np.mean(synthetic_mi_values)

    return {
        "real_mean_mi": real_mean,
        "synthetic_mean_mi": synthetic_mean,
        "absolute_difference": abs(real_mean - synthetic_mean),
    }


# ----------------------------------------------------------------------
# 7. Quick test / entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    sfreq = 256
    n_channels = 23
    n_time = 2560

    # simulate test segments
    real_segments = [np.random.randn(n_channels, n_time) for _ in range(5)]
    synthetic_segments = [np.random.randn(n_channels, n_time) for _ in range(5)]

    print("Testing single segment PAC computation...")
    mi_values = compute_pac_segment(real_segments[0], sfreq)
    print(f"MI per channel (first segment): {mi_values}")
    print(f"Mean MI: {mi_values.mean():.6f}")

    print("\nComparing real vs synthetic PAC fidelity...")
    result = compare_pac_fidelity(real_segments, synthetic_segments, sfreq)
    print(result)