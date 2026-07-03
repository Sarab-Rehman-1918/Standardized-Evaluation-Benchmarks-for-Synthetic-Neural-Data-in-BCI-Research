"""
generate_report.py

Reads benchmark_results.json produced by run_benchmark.py and generates
a formatted leaderboard report — both printed to terminal and saved
as a markdown file for documentation/publication.

This is the final deliverable output of the benchmarking suite,
satisfying Objective 4 of the proposal (open-source leaderboard).
"""

import json
import os
import sys
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

RESULTS_PATH = "data/results/benchmark_results.json"
REPORT_PATH = "data/results/leaderboard_report.md"


# ----------------------------------------------------------------------
# 1. Load benchmark results
# ----------------------------------------------------------------------

def load_results(path=RESULTS_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No benchmark results found at {path}.\n"
            f"Run run_benchmark.py first to generate results."
        )
    with open(path, "r") as f:
        return json.load(f)


# ----------------------------------------------------------------------
# 2. Print terminal summary
# ----------------------------------------------------------------------

def print_terminal_report(data):
    generators = data["generators"]

    print("\n" + "=" * 70)
    print("  SYNTHETIC EEG BENCHMARK LEADERBOARD")
    print(f"  Dataset : {data['dataset']}")
    print(f"  Date    : {data['benchmark_date']}")
    print(f"  Sfreq   : {data['sfreq']} Hz")
    print(f"  Real interictal: {data['n_real_interictal']} | "
          f"Real ictal: {data['n_real_ictal']}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Spectral Fidelity Table
    # ------------------------------------------------------------------
    print("\n--- METRIC 1: Spectral Fidelity (lower = better) ---")
    print(f"{'Generator':<15} {'Delta':>10} {'Theta':>10} {'Alpha':>10} "
          f"{'Beta':>10} {'Gamma':>10} {'Mean':>10}")
    print("-" * 70)

    for gen_name, gen_data in generators.items():
        sf = gen_data["metrics"]["spectral_fidelity"]
        mean = gen_data["metrics"]["spectral_fidelity_mean"]
        print(f"{gen_name:<15} "
              f"{sf['delta']:>10.6f} "
              f"{sf['theta']:>10.6f} "
              f"{sf['alpha']:>10.6f} "
              f"{sf['beta']:>10.6f} "
              f"{sf['gamma']:>10.6f} "
              f"{mean:>10.6f}")

    # ------------------------------------------------------------------
    # PAC Table
    # ------------------------------------------------------------------
    print("\n--- METRIC 2: Phase-Amplitude Coupling (lower diff = better) ---")
    print(f"{'Generator':<15} {'Real MI':>12} {'Synth MI':>12} {'Difference':>12}")
    print("-" * 55)

    for gen_name, gen_data in generators.items():
        pac = gen_data["metrics"]["pac"]
        print(f"{gen_name:<15} "
              f"{pac['real_mean_mi']:>12.6f} "
              f"{pac['synthetic_mean_mi']:>12.6f} "
              f"{pac['absolute_difference']:>12.6f}")

    # ------------------------------------------------------------------
    # Subject Fingerprint Table
    # ------------------------------------------------------------------
    print("\n--- METRIC 3: Subject Fingerprint (lower accuracy = better) ---")
    print(f"{'Generator':<15} {'Mean Accuracy':>15} {'Std':>10}")
    print("-" * 45)

    for gen_name, gen_data in generators.items():
        fp = gen_data["metrics"]["subject_fingerprint"]
        print(f"{gen_name:<15} "
              f"{fp['mean_accuracy']:>15.4f} "
              f"{fp['std_accuracy']:>10.4f}")

    # ------------------------------------------------------------------
    # Cross-Session Stability Table
    # ------------------------------------------------------------------
    print("\n--- METRIC 4: Cross-Session Stability (ratio > 1 = better) ---")
    print(f"{'Generator':<15} {'Within':>12} {'Between':>12} {'Ratio':>10}")
    print("-" * 55)

    for gen_name, gen_data in generators.items():
        cs = gen_data["metrics"]["cross_session_stability"]
        print(f"{gen_name:<15} "
              f"{cs['within_subject_distance']:>12.4f} "
              f"{cs['between_subject_distance']:>12.4f} "
              f"{cs['stability_ratio']:>10.4f}")

    # ------------------------------------------------------------------
    # Safety Protocol Table
    # ------------------------------------------------------------------
    print("\n--- SAFETY PROTOCOL: Downstream Classifier Performance ---")
    print(f"{'Generator':<15} {'Condition':<18} {'Accuracy':>10} "
          f"{'F1':>8} {'FPR':>8} {'Certified':>10}")
    print("-" * 75)

    for gen_name, gen_data in generators.items():
        safety = gen_data["safety"]
        certified = "YES" if safety["certified_safe"] else "NO"

        for condition in ["real_only", "synthetic_only", "mixed"]:
            m = safety[condition]
            cert_col = certified if condition == "synthetic_only" else ""
            print(f"{gen_name:<15} "
                  f"{condition:<18} "
                  f"{m['accuracy']:>10.4f} "
                  f"{m['f1_score']:>8.4f} "
                  f"{m['false_positive_rate']:>8.4f} "
                  f"{cert_col:>10}")

    # ------------------------------------------------------------------
    # Final ranking by PAC difference
    # ------------------------------------------------------------------
    print("\n--- FINAL RANKING (by PAC difference, lower = better) ---")
    print("-" * 55)

    ranked = sorted(
        generators.items(),
        key=lambda x: x[1]["metrics"]["pac"]["absolute_difference"]
    )

    for rank, (gen_name, gen_data) in enumerate(ranked, 1):
        pac_diff = gen_data["metrics"]["pac"]["absolute_difference"]
        certified = "CERTIFIED" if gen_data["safety"]["certified_safe"] else "NOT CERTIFIED"
        print(f"  #{rank}  {gen_name:<15} "
              f"PAC Diff: {pac_diff:.6f}  |  Safety: {certified}")

    print("=" * 70)


# ----------------------------------------------------------------------
# 3. Generate markdown report
# ----------------------------------------------------------------------

def generate_markdown_report(data):
    generators = data["generators"]
    lines = []

    lines.append("# Synthetic EEG Benchmark Leaderboard\n")
    lines.append(f"**Dataset:** {data['dataset']}  ")
    lines.append(f"**Date:** {data['benchmark_date']}  ")
    lines.append(f"**Sampling Frequency:** {data['sfreq']} Hz  ")
    lines.append(f"**Real Segments:** {data['n_real_interictal']} interictal, "
                 f"{data['n_real_ictal']} ictal  \n")

    # Spectral Fidelity
    lines.append("## Metric 1: Spectral Fidelity\n")
    lines.append("> Lower score = synthetic distribution closer to real EEG\n")
    lines.append("| Generator | Delta | Theta | Alpha | Beta | Gamma | Mean |")
    lines.append("|-----------|-------|-------|-------|------|-------|------|")
    for gen_name, gen_data in generators.items():
        sf = gen_data["metrics"]["spectral_fidelity"]
        mean = gen_data["metrics"]["spectral_fidelity_mean"]
        lines.append(
            f"| {gen_name} "
            f"| {sf['delta']:.6f} "
            f"| {sf['theta']:.6f} "
            f"| {sf['alpha']:.6f} "
            f"| {sf['beta']:.6f} "
            f"| {sf['gamma']:.6f} "
            f"| {mean:.6f} |"
        )

    # PAC
    lines.append("\n## Metric 2: Phase-Amplitude Coupling\n")
    lines.append("> Lower absolute difference = better PAC preservation\n")
    lines.append("| Generator | Real MI | Synthetic MI | Difference |")
    lines.append("|-----------|---------|--------------|------------|")
    for gen_name, gen_data in generators.items():
        pac = gen_data["metrics"]["pac"]
        lines.append(
            f"| {gen_name} "
            f"| {pac['real_mean_mi']:.6f} "
            f"| {pac['synthetic_mean_mi']:.6f} "
            f"| {pac['absolute_difference']:.6f} |"
        )

    # Subject Fingerprint
    lines.append("\n## Metric 3: Subject Fingerprint Retention\n")
    lines.append("> Lower accuracy = synthetic data less distinguishable from real (better)\n")
    lines.append("| Generator | Mean Accuracy | Std |")
    lines.append("|-----------|--------------|-----|")
    for gen_name, gen_data in generators.items():
        fp = gen_data["metrics"]["subject_fingerprint"]
        lines.append(
            f"| {gen_name} "
            f"| {fp['mean_accuracy']:.4f} "
            f"| {fp['std_accuracy']:.4f} |"
        )

    # Cross-Session Stability
    lines.append("\n## Metric 4: Cross-Session Stability\n")
    lines.append("> Ratio > 1 = between-subject variance exceeds within-subject variance (better)\n")
    lines.append("| Generator | Within Distance | Between Distance | Ratio |")
    lines.append("|-----------|----------------|-----------------|-------|")
    for gen_name, gen_data in generators.items():
        cs = gen_data["metrics"]["cross_session_stability"]
        lines.append(
            f"| {gen_name} "
            f"| {cs['within_subject_distance']:.4f} "
            f"| {cs['between_subject_distance']:.4f} "
            f"| {cs['stability_ratio']:.4f} |"
        )

    # Safety Protocol
    lines.append("\n## Safety Protocol: Downstream Classifier\n")
    lines.append("| Generator | Condition | Accuracy | F1 | FPR | Certified |")
    lines.append("|-----------|-----------|----------|----|-----|-----------|")
    for gen_name, gen_data in generators.items():
        safety = gen_data["safety"]
        certified = "YES" if safety["certified_safe"] else "NO"
        for condition in ["real_only", "synthetic_only", "mixed"]:
            m = safety[condition]
            cert_col = certified if condition == "synthetic_only" else ""
            lines.append(
                f"| {gen_name} "
                f"| {condition} "
                f"| {m['accuracy']:.4f} "
                f"| {m['f1_score']:.4f} "
                f"| {m['false_positive_rate']:.4f} "
                f"| {cert_col} |"
            )

    # Final Ranking by PAC difference
    lines.append("\n## Final Ranking\n")
    lines.append("> Ranked by PAC absolute difference (lower = better)\n")
    lines.append("| Rank | Generator | PAC Difference | Safety |")
    lines.append("|------|-----------|---------------|--------|")

    ranked = sorted(
        generators.items(),
        key=lambda x: x[1]["metrics"]["pac"]["absolute_difference"]
    )
    for rank, (gen_name, gen_data) in enumerate(ranked, 1):
        pac_diff = gen_data["metrics"]["pac"]["absolute_difference"]
        certified = "Certified" if gen_data["safety"]["certified_safe"] else "Not Certified"
        lines.append(
            f"| #{rank} | {gen_name} | {pac_diff:.6f} | {certified} |"
        )

    return "\n".join(lines)


# ----------------------------------------------------------------------
# 4. Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading benchmark results...")
    data = load_results()

    print_terminal_report(data)

    print(f"\nGenerating markdown report...")
    markdown = generate_markdown_report(data)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"Markdown report saved to {REPORT_PATH}")