# Standardized Evaluation Benchmarks for Synthetic Neural Data in BCI Research

> A physiologically grounded benchmarking suite for evaluating synthetic EEG data quality in Brain-Computer Interface (BCI) research.

---

## What Is This Project? (Simple Explanation in Urdu)

**Problem kya tha?**
Aaj kal AI se fake/synthetic EEG data banaya ja sakta hai. EEG woh signal hota hai jo brain se aata hai — jaise seizure detect karna ya koi aur brain condition. Lekin problem yeh thi ke jab log check karte the ke yeh fake EEG achha hai ya nahi, toh woh computer vision ke tools use karte the — jo actually images ke liye bane hain, EEG ke liye nahi. Iska matlab tha ke ek fake EEG "achha" lag sakta tha lekin actually usme brain ki asli properties nahi hoti thin — aur agar aisa data hospital mein use ho jata toh galat diagnosis ho sakti thi, jo patient ke liye khatra tha.

**Is project ne kya solve kiya?**
Humne ek aisa system banaya jo fake EEG ko properly check karta hai — brain science ke hisaab se, na computer vision ke hisaab se.

**Kaise kaam karta hai?**
- Asli brain recordings (CHB-MIT dataset) load karta hai
- 3 AI models (VAE, GAN, Diffusion) se fake EEG banata hai
- 4 brain-science metrics se fake EEG ko check karta hai
- Safety test karta hai ke yeh fake data clinic mein safe hai ya nahi
- Sab results ek leaderboard mein dikhata hai

**Summary ek line mein:**
Humne ek aisa system banaya jo AI se bani fake brain signals ko brain science ke proper rules se check karta hai — taake doctors aur researchers ko pata chale ke yeh fake data clinic mein safely use ho sakta hai ya nahi.

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
Compares power spectral density (PSD) distributions across standard EEG frequency bands between real and synthetic data using Wasserstein distance.

Bands evaluated: Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-45 Hz)

Lower score = synthetic data closer to real EEG spectral distribution.

### 2. Phase-Amplitude Coupling (`metrics/phase_amplitude_coupling.py`)
Measures how well synthetic EEG preserves the coupling between low-frequency phase (theta) and high-frequency amplitude (gamma) — a fundamental neurophysiological property of real brain signals.

Uses Modulation Index (MI) based on Tort et al. (2010).

Lower absolute difference = better PAC preservation.

### 3. Subject Fingerprint Retention (`metrics/subject_fingerprint.py`)
Tests whether synthetic data preserves subject-specific neural signatures using a band-power feature classifier.

Lower classification accuracy = synthetic data is less distinguishable from real (better generalization).

### 4. Cross-Session Stability (`metrics/cross_session_stability.py`)
Measures within-subject vs between-subject variability across sessions.

Stability ratio > 1 = subject is more consistent within itself than vs other subjects (physiologically expected).

---

## Generative Models Evaluated

| Model | File | Description |
|-------|------|-------------|
| VAE | `generators/vae_baseline.py` | Variational Autoencoder — compresses real EEG into latent space then generates new samples |
| GAN | `generators/gan_baseline.py` | Generative Adversarial Network — generator and discriminator trained adversarially |
| Diffusion | `generators/diffusion_baseline.py` | DDPM — learns to reverse a gradual noising process to generate realistic EEG |

---

## Dataset

**CHB-MIT Scalp EEG Database** (PhysioNet)
- Subjects: chb01, chb03 (two distinct pediatric patients)
- Sampling rate: 256 Hz
- Channels: 23 (standard 10-20 montage)
- Segment length: 10 seconds
- Labels: ictal (seizure) and interictal (non-seizure)
- Total segments: 2,520 (30 ictal, 2,490 interictal)

Download: https://physionet.org/content/chbmit/1.0.0/

Place data in:
```
data/raw/epilepsy/chb_mit/chb01/
data/raw/epilepsy/chb_mit/chb03/
```

---

## Safety Protocol

Trains a downstream pathology detection classifier (Random Forest) under three conditions:

| Condition | Description |
|-----------|-------------|
| Real only | Trained on real EEG only — gold standard baseline |
| Synthetic only | Trained on synthetic EEG only — tests clinical risk |
| Mixed | Trained on real + synthetic — tests augmentation safety |

**Safety thresholds for certification:**
- Accuracy >= 0.75
- F1 Score >= 0.70
- False Positive Rate <= 0.20

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

### Step 2 — Generate the leaderboard report
```bash
python "src/synth_eeg_bench/leaderboard/generate_report.py"
```

### Step 3 — Launch the interactive dashboard
```bash
streamlit run web/dashboard.py
```

Opens at `http://localhost:8501` in your browser automatically.

Results are saved to:
- `data/results/benchmark_results.json`
- `data/results/leaderboard_report.md`

### Run individual modules
```bash
# Test spectral fidelity on real data
python "scripts/test_spectral_fidelity_real_data.py"

# Test PAC on real data
python "scripts/test_pac_real_data.py"

# Test subject fingerprinting on real data
python "scripts/test_subject_fingerprint_real_data.py"

# Run safety evaluation only
python "src/synth_eeg_bench/safety_protocol/evaluate_safety.py"
```

---

## Dashboard

The interactive Streamlit dashboard (`web/dashboard.py`) visualizes all benchmark results with charts and tables. It reads from `data/results/benchmark_results.json` — so results update only when you rerun `run_benchmark.py`.

**Note:** The dashboard never reruns the benchmark itself. To get fresh results:
1. Run `run_benchmark.py` first
2. Then refresh the dashboard in your browser

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

Install with:
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

> Note: Safety certification requires the synthetic_only classifier to meet
> all three clinical thresholds simultaneously. Current generators only
> synthesize interictal segments — adding ictal synthesis is a planned
> future improvement.

---

## How Results Change

Results vary between benchmark runs due to:
- Random weight initialization in VAE, GAN, and Diffusion training
- Random subsampling of training data
- Random latent vector sampling during generation

To make results fully reproducible, add these seeds at the top of `run_benchmark.py`:
```python
import random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

---

## References

Han, S., Feng, S., & Li, F. (2026). Advancing brain-computer interfaces with
generative AI: A review of state-of-the-art and future outlook.
*The Innovation Life*, 4(1).

Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
*Circulation*, 101(23).

Tort, A. B., et al. (2010). Measuring phase-amplitude coupling between neuronal
oscillations of different frequencies. *Journal of Neurophysiology*, 104(2).

---

## Author

**Sarab Rehman** — ITSOLERA Internship Project, 2026

---

## License

MIT License — see `LICENSE` for details.