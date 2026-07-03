"""
dashboard.py

Streamlit leaderboard dashboard for the Synthetic EEG Benchmark Suite.
Reads data/results/benchmark_results.json produced by run_benchmark.py.
No changes required to any other project files.

Run with:
    streamlit run web/dashboard.py
"""

import json
import os
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

RESULTS_PATH = "data/results/benchmark_results.json"

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="Synthetic EEG Benchmark Leaderboard",
    page_icon="🧠",
    layout="wide",
)

# ----------------------------------------------------------------------
# Load results
# ----------------------------------------------------------------------

def load_results():
    if not os.path.exists(RESULTS_PATH):
        return None
    with open(RESULTS_PATH, "r") as f:
        return json.load(f)


data = load_results()

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------

st.title("🧠 Synthetic EEG Benchmark Leaderboard")
st.markdown(
    "**Standardized Evaluation Benchmarks for Synthetic Neural Data in BCI Research**"
)
st.markdown("---")

if data is None:
    st.error(
        "No benchmark results found. "
        "Please run `src/synth_eeg_bench/leaderboard/run_benchmark.py` first."
    )
    st.stop()

# ----------------------------------------------------------------------
# Dataset info
# ----------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)
col1.metric("Dataset", data["dataset"])
col2.metric("Sampling Rate", f"{data['sfreq']} Hz")
col3.metric("Interictal Segments", data["n_real_interictal"])
col4.metric("Ictal Segments", data["n_real_ictal"])

st.caption(f"Benchmark run: {data['benchmark_date']}")
st.markdown("---")

generators = data["generators"]
gen_names = list(generators.keys())

# ----------------------------------------------------------------------
# Metric 1: Spectral Fidelity
# ----------------------------------------------------------------------

st.subheader("Metric 1: Spectral Fidelity")
st.caption("Lower = synthetic distribution closer to real EEG (better)")

bands = ["delta", "theta", "alpha", "beta", "gamma"]
sf_data = []
for gen_name in gen_names:
    sf = generators[gen_name]["metrics"]["spectral_fidelity"]
    for band in bands:
        sf_data.append({
            "Generator": gen_name,
            "Band": band.capitalize(),
            "Wasserstein Distance": sf[band],
        })

sf_df = pd.DataFrame(sf_data)
fig_sf = px.bar(
    sf_df,
    x="Band",
    y="Wasserstein Distance",
    color="Generator",
    barmode="group",
    color_discrete_sequence=["#4C72B0", "#DD8452", "#55A868"],
)
fig_sf.update_layout(
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font_color="white",
    legend=dict(bgcolor="#0e1117"),
)
st.plotly_chart(fig_sf, use_container_width=True)

# ----------------------------------------------------------------------
# Metric 2: Phase-Amplitude Coupling
# ----------------------------------------------------------------------

st.markdown("---")
st.subheader("Metric 2: Phase-Amplitude Coupling (PAC)")
st.caption("Lower absolute difference = better PAC preservation (better)")

pac_rows = []
for gen_name in gen_names:
    pac = generators[gen_name]["metrics"]["pac"]
    pac_rows.append({
        "Generator": gen_name,
        "Real MI": pac["real_mean_mi"],
        "Synthetic MI": pac["synthetic_mean_mi"],
        "Absolute Difference": pac["absolute_difference"],
    })

pac_df = pd.DataFrame(pac_rows)

col1, col2 = st.columns(2)

with col1:
    fig_pac_bar = go.Figure()
    fig_pac_bar.add_trace(go.Bar(
        name="Real MI",
        x=pac_df["Generator"],
        y=pac_df["Real MI"],
        marker_color="#4C72B0",
    ))
    fig_pac_bar.add_trace(go.Bar(
        name="Synthetic MI",
        x=pac_df["Generator"],
        y=pac_df["Synthetic MI"],
        marker_color="#DD8452",
    ))
    fig_pac_bar.update_layout(
        barmode="group",
        title="Real vs Synthetic Modulation Index",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        legend=dict(bgcolor="#0e1117"),
    )
    st.plotly_chart(fig_pac_bar, use_container_width=True)

with col2:
    fig_pac_diff = px.bar(
        pac_df,
        x="Generator",
        y="Absolute Difference",
        color="Generator",
        title="PAC Absolute Difference (lower = better)",
        color_discrete_sequence=["#4C72B0", "#DD8452", "#55A868"],
    )
    fig_pac_diff.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        showlegend=False,
    )
    st.plotly_chart(fig_pac_diff, use_container_width=True)

# ----------------------------------------------------------------------
# Metric 3: Subject Fingerprint
# ----------------------------------------------------------------------

st.markdown("---")
st.subheader("Metric 3: Subject Fingerprint Retention")
st.caption("Lower accuracy = synthetic less distinguishable from real (better)")

fp_rows = []
for gen_name in gen_names:
    fp = generators[gen_name]["metrics"]["subject_fingerprint"]
    fp_rows.append({
        "Generator": gen_name,
        "Mean Accuracy": fp["mean_accuracy"],
        "Std": fp["std_accuracy"],
    })

fp_df = pd.DataFrame(fp_rows)

fig_fp = go.Figure()
fig_fp.add_trace(go.Bar(
    x=fp_df["Generator"],
    y=fp_df["Mean Accuracy"],
    error_y=dict(type="data", array=fp_df["Std"].tolist()),
    marker_color=["#4C72B0", "#DD8452", "#55A868"],
))
fig_fp.update_layout(
    title="Subject Classification Accuracy (with std dev)",
    yaxis=dict(range=[0, 1.1], title="Accuracy"),
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font_color="white",
)
st.plotly_chart(fig_fp, use_container_width=True)

# ----------------------------------------------------------------------
# Metric 4: Cross-Session Stability
# ----------------------------------------------------------------------

st.markdown("---")
st.subheader("Metric 4: Cross-Session Stability")
st.caption("Stability ratio > 1 = within-subject more stable than between-subject (better)")

cs_rows = []
for gen_name in gen_names:
    cs = generators[gen_name]["metrics"]["cross_session_stability"]
    cs_rows.append({
        "Generator": gen_name,
        "Within-Subject Distance": cs["within_subject_distance"],
        "Between-Subject Distance": cs["between_subject_distance"],
        "Stability Ratio": cs["stability_ratio"],
    })

cs_df = pd.DataFrame(cs_rows)

col1, col2 = st.columns(2)

with col1:
    fig_cs_dist = go.Figure()
    fig_cs_dist.add_trace(go.Bar(
        name="Within-Subject",
        x=cs_df["Generator"],
        y=cs_df["Within-Subject Distance"],
        marker_color="#4C72B0",
    ))
    fig_cs_dist.add_trace(go.Bar(
        name="Between-Subject",
        x=cs_df["Generator"],
        y=cs_df["Between-Subject Distance"],
        marker_color="#DD8452",
    ))
    fig_cs_dist.update_layout(
        barmode="group",
        title="Within vs Between Subject Distance",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        legend=dict(bgcolor="#0e1117"),
    )
    st.plotly_chart(fig_cs_dist, use_container_width=True)

with col2:
    fig_cs_ratio = px.bar(
        cs_df,
        x="Generator",
        y="Stability Ratio",
        color="Generator",
        title="Stability Ratio (higher = better)",
        color_discrete_sequence=["#4C72B0", "#DD8452", "#55A868"],
    )
    fig_cs_ratio.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="white",
        annotation_text="Threshold (ratio = 1)",
    )
    fig_cs_ratio.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        showlegend=False,
    )
    st.plotly_chart(fig_cs_ratio, use_container_width=True)

# ----------------------------------------------------------------------
# Safety Protocol
# ----------------------------------------------------------------------

st.markdown("---")
st.subheader("Safety Protocol: Downstream Classifier")
st.caption(
    "Safety thresholds: Accuracy >= 0.75 | F1 >= 0.70 | FPR <= 0.20"
)

safety_rows = []
for gen_name in gen_names:
    safety = generators[gen_name]["safety"]
    for condition in ["real_only", "synthetic_only", "mixed"]:
        m = safety[condition]
        safety_rows.append({
            "Generator": gen_name,
            "Condition": condition.replace("_", " ").title(),
            "Accuracy": m["accuracy"],
            "F1 Score": m["f1_score"],
            "False Positive Rate": m["false_positive_rate"],
            "Certified": "Yes" if safety["certified_safe"] else "No",
        })

safety_df = pd.DataFrame(safety_rows)

col1, col2, col3 = st.columns(3)

with col1:
    fig_acc = px.bar(
        safety_df,
        x="Generator",
        y="Accuracy",
        color="Condition",
        barmode="group",
        title="Accuracy by Condition",
        color_discrete_sequence=["#4C72B0", "#DD8452", "#55A868"],
    )
    fig_acc.add_hline(y=0.75, line_dash="dash", line_color="red",
                       annotation_text="Threshold")
    fig_acc.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        legend=dict(bgcolor="#0e1117"),
        yaxis=dict(range=[0, 1.1]),
    )
    st.plotly_chart(fig_acc, use_container_width=True)

with col2:
    fig_f1 = px.bar(
        safety_df,
        x="Generator",
        y="F1 Score",
        color="Condition",
        barmode="group",
        title="F1 Score by Condition",
        color_discrete_sequence=["#4C72B0", "#DD8452", "#55A868"],
    )
    fig_f1.add_hline(y=0.70, line_dash="dash", line_color="red",
                      annotation_text="Threshold")
    fig_f1.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        legend=dict(bgcolor="#0e1117"),
        yaxis=dict(range=[0, 1.1]),
    )
    st.plotly_chart(fig_f1, use_container_width=True)

with col3:
    fig_fpr = px.bar(
        safety_df,
        x="Generator",
        y="False Positive Rate",
        color="Condition",
        barmode="group",
        title="False Positive Rate by Condition",
        color_discrete_sequence=["#4C72B0", "#DD8452", "#55A868"],
    )
    fig_fpr.add_hline(y=0.20, line_dash="dash", line_color="red",
                       annotation_text="Threshold")
    fig_fpr.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        legend=dict(bgcolor="#0e1117"),
        yaxis=dict(range=[0, 1.1]),
    )
    st.plotly_chart(fig_fpr, use_container_width=True)

# ----------------------------------------------------------------------
# Final Ranking
# ----------------------------------------------------------------------

st.markdown("---")
st.subheader("Final Ranking")
st.caption("Ranked by PAC absolute difference (lower = better)")

ranked = sorted(
    gen_names,
    key=lambda g: generators[g]["metrics"]["pac"]["absolute_difference"]
)

for rank, gen_name in enumerate(ranked, 1):
    pac_diff = generators[gen_name]["metrics"]["pac"]["absolute_difference"]
    sf_mean = generators[gen_name]["metrics"]["spectral_fidelity_mean"]
    stability = generators[gen_name]["metrics"]["cross_session_stability"]["stability_ratio"]
    certified = generators[gen_name]["safety"]["certified_safe"]

    medal = ["🥇", "🥈", "🥉"][rank - 1]
    cert_badge = "✅ Certified" if certified else "❌ Not Certified"

    with st.container():
        col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 2])
        col1.markdown(f"### {medal} #{rank}")
        col2.metric("Generator", gen_name)
        col3.metric("PAC Difference", f"{pac_diff:.6f}")
        col4.metric("Stability Ratio", f"{stability:.4f}")
        col5.metric("Safety", cert_badge)
        st.markdown("---")

# ----------------------------------------------------------------------
# Raw data table
# ----------------------------------------------------------------------

with st.expander("View Raw Results Table"):
    rows = []
    for gen_name in gen_names:
        g = generators[gen_name]
        rows.append({
            "Generator": gen_name,
            "Spectral Mean": g["metrics"]["spectral_fidelity_mean"],
            "PAC Diff": g["metrics"]["pac"]["absolute_difference"],
            "Fingerprint Acc": g["metrics"]["subject_fingerprint"]["mean_accuracy"],
            "Stability Ratio": g["metrics"]["cross_session_stability"]["stability_ratio"],
            "Safety Accuracy": g["safety"]["synthetic_only"]["accuracy"],
            "Safety F1": g["safety"]["synthetic_only"]["f1_score"],
            "Safety FPR": g["safety"]["synthetic_only"]["false_positive_rate"],
            "Certified": g["safety"]["certified_safe"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.markdown("---")
st.caption(
    "Standardized Evaluation Benchmarks for Synthetic Neural Data in BCI Research | "
    "ITSOLERA Internship Project 2026"
)