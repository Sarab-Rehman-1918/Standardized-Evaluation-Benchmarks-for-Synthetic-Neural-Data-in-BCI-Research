# Synthetic EEG Benchmark Leaderboard

**Dataset:** CHB-MIT (chb01 + chb03)  
**Date:** 2026-07-01 08:12:28  
**Sampling Frequency:** 256.0 Hz  
**Real Segments:** 2490 interictal, 30 ictal  

## Metric 1: Spectral Fidelity

> Lower score = synthetic distribution closer to real EEG

| Generator | Delta | Theta | Alpha | Beta | Gamma | Mean |
|-----------|-------|-------|-------|------|-------|------|
| VAE | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| GAN | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Diffusion | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Metric 2: Phase-Amplitude Coupling

> Lower absolute difference = better PAC preservation

| Generator | Real MI | Synthetic MI | Difference |
|-----------|---------|--------------|------------|
| VAE | 0.000570 | 0.000407 | 0.000162 |
| GAN | 0.000570 | 0.000666 | 0.000096 |
| Diffusion | 0.000570 | 0.000557 | 0.000012 |

## Metric 3: Subject Fingerprint Retention

> Lower accuracy = synthetic data less distinguishable from real (better)

| Generator | Mean Accuracy | Std |
|-----------|--------------|-----|
| VAE | 0.9667 | 0.0408 |
| GAN | 1.0000 | 0.0000 |
| Diffusion | 0.9833 | 0.0333 |

## Metric 4: Cross-Session Stability

> Ratio > 1 = between-subject variance exceeds within-subject variance (better)

| Generator | Within Distance | Between Distance | Ratio |
|-----------|----------------|-----------------|-------|
| VAE | 0.0000 | 0.0000 | 3.6129 |
| GAN | 0.0000 | 0.0000 | 3.3792 |
| Diffusion | 0.0000 | 0.0000 | 3.0178 |

## Safety Protocol: Downstream Classifier

| Generator | Condition | Accuracy | F1 | FPR | Certified |
|-----------|-----------|----------|----|-----|-----------|
| VAE | real_only | 0.9615 | 0.9091 | 0.0000 |  |
| VAE | synthetic_only | 0.7692 | 0.0000 | 0.0000 | NO |
| VAE | mixed | 0.9231 | 0.8333 | 0.0500 |  |
| GAN | real_only | 0.9615 | 0.9091 | 0.0000 |  |
| GAN | synthetic_only | 0.7692 | 0.0000 | 0.0000 | NO |
| GAN | mixed | 0.9231 | 0.8333 | 0.0500 |  |
| Diffusion | real_only | 0.9615 | 0.9091 | 0.0000 |  |
| Diffusion | synthetic_only | 0.7692 | 0.0000 | 0.0000 | NO |
| Diffusion | mixed | 0.9231 | 0.8333 | 0.0500 |  |

## Final Ranking

> Ranked by PAC absolute difference (lower = better)

| Rank | Generator | PAC Difference | Safety |
|------|-----------|---------------|--------|
| #1 | Diffusion | 0.000012 | Not Certified |
| #2 | GAN | 0.000096 | Not Certified |
| #3 | VAE | 0.000162 | Not Certified |