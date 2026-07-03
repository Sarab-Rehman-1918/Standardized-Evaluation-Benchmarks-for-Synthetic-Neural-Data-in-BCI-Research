# Standardized Evaluation Benchmarks for Synthetic Neural Data in BCI Research

> A physiologically grounded benchmarking suite for evaluating synthetic EEG data quality in Brain-Computer Interface (BCI) research.

---

## What Is This Project?

**The Problem:**
Artificial Intelligence can now generate fake/synthetic EEG data. EEG (Electroencephalogram) is the electrical signal recorded from the brain — used to detect seizures and other brain conditions. The problem was that when researchers checked whether their fake EEG was good or not, they used tools borrowed from computer vision (originally designed for images). This was wrong because a fake EEG signal could score well on those image-based tools while completely missing fundamental brain properties — and if such data was used to train clinical models, it could cause dangerous misdiagnoses in real patients.

**What This Project Solves:**
This project builds a standardized benchmarking suite that evaluates synthetic EEG quality using brain science rules — not image processing rules. It lets researchers know whether their synthetic EEG is genuinely safe and useful, or just statistically plausible but clinically harmful.

**How It Works — Step by Step:**
1. Loads real brain recordings from the CHB-MIT epilepsy dataset
2. Trains three AI generators (VAE, GAN, Diffusion) to produce fake EEG
3. Evaluates the fake EEG using four neurophysiology-based metrics
4. Runs a clinical safety test to check if the fake data is safe for medical use
5. Produces a leaderboard report and interactive dashboard showing all results

**One Line Summary:**
We built a system that checks AI-generated brain signals against real brain science rules — so doctors and researchers can know whether synthetic EEG data is safe to use in clinical settings.

---

## Overview

Generative AI has introduced powerful capabilities for synthesizing artificial EEG data in BCI research. However, existing evaluation methods borrow metrics from computer vision (e.g. Fréchet Inception Distance) that fail to capture fundamental neurophysiological properties of brain signals.

This project provides a standardized, open-source benchmarking suite that evaluates synthetic EEG quality using metrics grounded in neuroscience — enabling researchers to distinguish genuinely useful synthetic data from statistically plausible but clinically harmful fabrications.

---

## Project Structure

```
Standardized-Evaluation-Benchmarks-for-Synthetic-Neural-Data-in-BCI-Research/
│
├── data/
│   ├── raw/
│   │   ├── motor_imagery/
│   │   │   └── bci_competition_iv/
│   │   ├── epilepsy/
│   │   │   └── chb_mit/
│   │   │       ├── chb01/
│   │   │       └── chb03/
│   │   └── cognitive_state/
│   ├── processed/
│   ├── synthetic/
│   └── results/
│       ├── benchmark_results.json
│       └── leaderboard_report.md
│
├── src/
│   └── synth_eeg_bench/
│       ├── preprocessing/
│       │   └── load_epilepsy_chbmit.py
│       ├── metrics/
│       │   ├── spectral_fidelity.py
│       │   ├── phase_amplitude_coupling.py
│       │   ├── subject_fingerprint.py
│       │   └── cross_session_stability.py
│       ├── generators/
│       │   ├── vae_baseline.py
│       │   ├── gan_baseline.py
│       │   └── diffusion_baseline.py
│       ├── safety_protocol/
│       │   ├── train_downstream_classifier.py
│       │   └── evaluate_safety.py
│       └── leaderboard/
│           ├── run_benchmark.py
│           └── generate_report.py
│
├── scripts/
│   ├── test_spectral_fidelity_real_data.py
│   ├── test_pac_real_data.py
│   ├── test_subject_fingerprint_real_data.py
│   └── train_vae_real_data.py
│
├── web/
│   └── dashboard.py
│
├── configs/
├── notebooks/
├── tests/
├── docs/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Research Objectives

| # | Objective | Status |
|---|-----------|--------|
| 1 | Develop physiologically grounded metrics for synthetic EEG quality | Done |
| 2 | Curate a multi-cohort EEG benchmark dataset | Done |
| 3 | Design a downstream classifier safety protocol | Done |
| 4 | Release full suite as an open-source benchmarking platform | Done |

---

## Metrics Implemented

### 1. Spectral Fidelity (`metrics/spectral_fidelity.py`)
Checks whether the power of brain waves in each frequency band matches between real and synthetic EEG. Uses Wasserstein distance to compare distributions.

Bands evaluated: Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-45 Hz)

Lower score = synthetic data closer to real EEG. Think of it like checking whether a music copy has the same bass and treble as the original.

### 2. Phase-Amplitude Coupling (`metrics/phase_amplitude_coupling.py`)
Real brains have a special property where slow waves and fast waves synchronize with each other. This metric checks whether synthetic EEG preserves this synchronization. Uses the Modulation Index (MI) method from Tort et al. (2010).

Lower absolute difference = better coupling preservation. Synthetic EEG that fails this test contains hallucinated patterns that do not exist in real brains.

### 3. Subject Fingerprint Retention (`metrics/subject_fingerprint.py`)
Every person's EEG is slightly unique — like a fingerprint. This metric checks whether synthetic data preserves the individual characteristics of the subject it was generated from, using a band-power feature classifier.

Lower classification accuracy = synthetic data is harder to distinguish from real data (better). Higher accuracy means the synthetic data looks too different from real data.

### 4. Cross-Session Stability (`metrics/cross_session_stability.py`)
A person's EEG should remain somewhat consistent across different recording sessions, while still being different from other people's EEG. This metric checks whether synthetic data preserves that stability.

Stability ratio > 1 means within-subject sessions are more similar to each other than to other subjects — which is the correct physiological behavior.

---

## Generative Models Evaluated

| Model | File | Description |
|-------|------|-------------|
| VAE | `generators/vae_baseline.py` | Variational Autoencoder — compresses real EEG into a small latent space, then samples from it to generate new fake EEG |
| GAN | `generators/gan_baseline.py` | Generative Adversarial Network — a generator and discriminator trained against each other until the generator fools the discriminator |
| Diffusion | `generators/diffusion_baseline.py` | Denoising Diffusion Probabilistic Model — gradually adds noise to real EEG, then learns to reverse that process to generate realistic EEG from pure noise |

---

## Dataset

**CHB-MIT Scalp EEG Database** (PhysioNet)
- Subjects: chb01, chb03 (two distinct pediatric patients)
- Sampling rate: 256 Hz
- Channels: 23 (standard 10-20 montage)
- Segment length: 10 seconds per segment
- Labels: ictal (during seizure) and interictal (seizure-free baseline)
- Total segments: 2,520 (30 ictal, 2,490 interictal)

Download: https://physionet.org/content/chbmit/1.0.0/

Place downloaded files in:
```
data/raw/epilepsy/chb_mit/chb01/
data/raw/epilepsy/chb_mit/chb03/
```

---

## Safety Protocol

Trains a downstream seizure detection classifier (Random Forest) under three different training conditions and evaluates all three on real held-out test data:

| Condition | Description | Purpose |
|-----------|-------------|---------|
| Real only | Trained on real EEG only | Gold standard baseline |
| Synthetic only | Trained on synthetic EEG only | Tests clinical risk of replacing real data |
| Mixed | Trained on real + synthetic combined | Tests data augmentation scenario |

**Safety thresholds for clinical certification:**

| Metric | Threshold |
|--------|-----------|
| Accuracy | >= 0.75 |
| F1 Score | >= 0.70 |
| False Positive Rate | <= 0.20 |

A generator is marked CERTIFIED SAFE only if its synthetic-only trained classifier passes all three thresholds simultaneously.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Standardized-Evaluation-Benchmarks-for-Synthetic-Neural-Data-in-BCI-Research.git
cd Standardized-Evaluation-Benchmarks-for-Synthetic-Neural-Data-in-BCI-Research

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\Activate.ps1

# Mac/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Run the full benchmark
```bash
python "src/synth_eeg_bench/leaderboard/run_benchmark.py"
```
This trains all three generators on real CHB-MIT data, runs all four metrics, runs the safety protocol, and saves results to `data/results/benchmark_results.json`.

### Step 2 — Generate the leaderboard report
```bash
python "src/synth_eeg_bench/leaderboard/generate_report.py"
```
Prints a formatted leaderboard table to the terminal and saves a markdown report to `data/results/leaderboard_report.md`.

### Step 3 — Launch the interactive dashboard
```bash
streamlit run web/dashboard.py
```
Opens an interactive web dashboard at `http://localhost:8501` showing all benchmark results as charts and tables.

### Run individual validation scripts
```bash
# Test spectral fidelity on real CHB-MIT data
python "scripts/test_spectral_fidelity_real_data.py"

# Test PAC on real CHB-MIT data
python "scripts/test_pac_real_data.py"

# Test subject fingerprinting on real CHB-MIT data
python "scripts/test_subject_fingerprint_real_data.py"

# Run safety evaluation only
python "src/synth_eeg_bench/safety_protocol/evaluate_safety.py"
```

---

## Interactive Dashboard

The Streamlit dashboard (`web/dashboard.py`) provides an interactive visual interface for exploring benchmark results. It includes:

- Dataset overview metrics
- Spectral fidelity bar charts per frequency band
- PAC comparison charts (real vs synthetic modulation index)
- Subject fingerprint accuracy with error bars
- Cross-session stability ratio charts
- Safety protocol results across all three training conditions
- Final ranked leaderboard with medals

**Important:** The dashboard only reads `data/results/benchmark_results.json` — it does not rerun the benchmark itself. To get updated results:
1. Run `run_benchmark.py` first to generate new results
2. Refresh the browser page — the dashboard will automatically show the latest data

---

## Requirements

```
numpy
scipy
scikit-learn
mne
matplotlib
pandas
pyyaml
torch
streamlit
plotly
```

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## Benchmark Results

| Rank | Generator | PAC Difference | Safety Certified |
|------|-----------|---------------|-----------------|
| #1 | Diffusion | 0.000012 | No |
| #2 | GAN | 0.000096 | No |
| #3 | VAE | 0.000162 | No |

The Diffusion model ranked first with the smallest PAC difference — meaning its synthetic EEG most closely preserved the phase-amplitude coupling structure of real brain signals.

**Why are none certified safe?**
All three generators were trained only on interictal (non-seizure) data, so they can only generate interictal synthetic segments. A classifier trained on synthetic-only data never sees any seizure examples and therefore cannot detect seizures — causing F1 score to be 0. Adding ictal synthesis capability to the generators is the primary planned improvement.

---

## How Results Change Between Runs

Results vary slightly between benchmark runs due to:
- Random weight initialization in VAE, GAN, and Diffusion model training
- Random subsampling when selecting training segments
- Random latent vector sampling during synthetic data generation

To make results fully reproducible (same output every run), add these lines at the top of `run_benchmark.py` after the imports:

```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

---

## References

Han, S., Feng, S., & Li, F. (2026). Advancing brain-computer interfaces with
generative AI: A review of state-of-the-art and future outlook.
*The Innovation Life*, 4(1).

Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet:
Components of a new research resource for complex physiologic signals.
*Circulation*, 101(23).

Tort, A. B., Komorowski, R., Eichenbaum, H., & Kopell, N. (2010).
Measuring phase-amplitude coupling between neuronal oscillations of
different frequencies. *Journal of Neurophysiology*, 104(2), 1195-1210.

---

## Author

**Sarab Rehman** — ITSOLERA Internship Project, 2026

---

## License

MIT License — see `LICENSE` for details.
