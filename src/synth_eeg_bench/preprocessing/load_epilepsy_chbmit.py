"""
load_epilepsy_chbmit.py

Loader for the CHB-MIT Scalp EEG Database (epilepsy cohort).
Parses the per-patient summary.txt files to find seizure annotations,
loads .edf files with MNE, and slices each recording into labeled
ictal (seizure) and interictal (non-seizure) segments.

Standardized output format:
    {
        "subject_id": str,          # e.g. "chb01"
        "file_name": str,           # e.g. "chb01_03.edf"
        "data": np.ndarray,         # shape (channels, time)
        "sfreq": float,             # sampling frequency in Hz
        "label": str,               # "ictal" or "interictal"
        "start_sec": float,         # segment start time within file
        "end_sec": float,           # segment end time within file
    }
"""

import os
import re
import numpy as np
import mne


# ----------------------------------------------------------------------
# 1. Parse summary.txt to extract seizure annotations per file
# ----------------------------------------------------------------------

def parse_summary_file(summary_path):
    """
    Parses a CHB-MIT chbXX-summary.txt file.

    Returns a dict:
        {
            "chb01_03.edf": [(2996, 3036)],   # list of (seizure_start, seizure_end) in seconds
            "chb01_01.edf": [],               # empty list if no seizures
            ...
        }
    """
    file_seizures = {}
    current_file = None

    with open(summary_path, "r") as f:
        lines = f.readlines()

    seizure_starts = []
    seizure_ends = []

    for line in lines:
        line = line.strip()

        if line.startswith("File Name:"):
            # save previous file's seizures before moving to next file
            if current_file is not None:
                file_seizures[current_file] = list(zip(seizure_starts, seizure_ends))

            current_file = line.split("File Name:")[1].strip()
            seizure_starts = []
            seizure_ends = []

        elif line.startswith("Seizure Start Time:"):
            sec = int(re.search(r"\d+", line).group())
            seizure_starts.append(sec)

        elif line.startswith("Seizure End Time:"):
            sec = int(re.search(r"\d+", line).group())
            seizure_ends.append(sec)

    # save the last file in the loop
    if current_file is not None:
        file_seizures[current_file] = list(zip(seizure_starts, seizure_ends))

    return file_seizures


# ----------------------------------------------------------------------
# 2. Load a single .edf file and slice into labeled segments
# ----------------------------------------------------------------------

def load_edf_segments(edf_path, seizure_windows, segment_length_sec=10):
    """
    Loads one .edf file and splits it into fixed-length labeled segments.

    Parameters:
        edf_path (str): path to the .edf file
        seizure_windows (list of tuples): [(start_sec, end_sec), ...] for this file
        segment_length_sec (int): length of each output segment in seconds

    Returns:
        list of dicts (see standardized format above)
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    sfreq = raw.info["sfreq"]
    data = raw.get_data()  # shape: (channels, total_time_samples)

    total_duration_sec = data.shape[1] / sfreq
    segment_samples = int(segment_length_sec * sfreq)

    segments = []
    file_name = os.path.basename(edf_path)
    subject_id = file_name.split("_")[0]

    start_sec = 0
    while start_sec + segment_length_sec <= total_duration_sec:
        end_sec = start_sec + segment_length_sec
        start_sample = int(start_sec * sfreq)
        end_sample = start_sample + segment_samples

        segment_data = data[:, start_sample:end_sample]

        # determine label: ictal if this segment overlaps any seizure window
        label = "interictal"
        for (sz_start, sz_end) in seizure_windows:
            if start_sec < sz_end and end_sec > sz_start:
                label = "ictal"
                break

        segments.append({
            "subject_id": subject_id,
            "file_name": file_name,
            "data": segment_data,
            "sfreq": sfreq,
            "label": label,
            "start_sec": start_sec,
            "end_sec": end_sec,
        })

        start_sec = end_sec

    return segments


# ----------------------------------------------------------------------
# 3. Load all available files for a given subject folder
# ----------------------------------------------------------------------

def load_subject(subject_dir, summary_path, segment_length_sec=10):
    """
    Loads all .edf files present in subject_dir, using summary_path
    to determine seizure labeling.

    Parameters:
        subject_dir (str): folder containing .edf files for one subject
        summary_path (str): path to that subject's summary.txt
        segment_length_sec (int): segment length for slicing

    Returns:
        list of segment dicts across all files found in subject_dir
    """
    file_seizures = parse_summary_file(summary_path)
    all_segments = []

    for file_name in os.listdir(subject_dir):
        if not file_name.endswith(".edf"):
            continue

        edf_path = os.path.join(subject_dir, file_name)
        seizure_windows = file_seizures.get(file_name, [])

        print(f"Loading {file_name} | seizures: {seizure_windows}")
        segments = load_edf_segments(edf_path, seizure_windows, segment_length_sec)
        all_segments.extend(segments)

    return all_segments


# ----------------------------------------------------------------------
# 4. Quick test / entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Adjust these paths to match your folder structure
    CHB01_DIR = "data/raw/epilepsy/chb_mit/chb01"
    CHB01_SUMMARY = "data/raw/epilepsy/chb_mit/chb01/chb01-summary.txt"

    CHB03_DIR = "data/raw/epilepsy/chb_mit/chb03"
    CHB03_SUMMARY = "data/raw/epilepsy/chb_mit/chb03/chb03-summary.txt"

    chb01_segments = load_subject(CHB01_DIR, CHB01_SUMMARY, segment_length_sec=10)
    chb03_segments = load_subject(CHB03_DIR, CHB03_SUMMARY, segment_length_sec=10)

    all_segments = chb01_segments + chb03_segments

    n_ictal = sum(1 for s in all_segments if s["label"] == "ictal")
    n_interictal = sum(1 for s in all_segments if s["label"] == "interictal")

    print(f"\nTotal segments: {len(all_segments)}")
    print(f"Ictal segments: {n_ictal}")
    print(f"Interictal segments: {n_interictal}")
    print(f"Example segment shape: {all_segments[0]['data'].shape}")