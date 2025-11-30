"""
AI Assistance Notice:
This application was developed with significant assistance from ChatGPT.
All analytical decisions, dataset preparation, simulation logic,
and final implementation were reviewed, tested, and validated by me.
"""

# =============================================================
# 💶 Instant Payments Readiness Simulator — Streamlit App
# =============================================================

import sys, os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.sim_core import (
    generate_synth,
    simulate_vop,
    scan_vop,
    simulate_fraud,
    scan_fraud
)

st.set_page_config(page_title="Instant Payments Readiness Simulator", layout="wide")
plt.rcParams["figure.figsize"] = (7,3.2)
plt.rcParams["axes.grid"] = True


# =============================================================
# NORMALIZATION + FULL KPI SCORE
# =============================================================
# Recources for KPI numbers - https://www.researchgate.net/publication/396290877_Machine_learning_for_fraud_detection_in_digital_banking_a_systematic_literature_review_REVIEW
def normalize(value, min_val, max_val):
    """Normalize a single KPI value to 0–1."""
    if max_val - min_val == 0:
        return 0
    return (value - min_val) / (max_val - min_val)

def compute_full_kpi(conv, latency, manual, risk,
                     conv_min, conv_max,
                     lat_min, lat_max,
                     man_min, man_max,
                     risk_min, risk_max):
    """
    Full KPI: weighted balance of 4 KPIs
    w_c = 0.40, w_r = 0.35, w_m = 0.15, w_l = 0.10
    """
    W_C = 0.40
    W_R = 0.35
    W_M = 0.15
    W_L = 0.10

    conv_n = normalize(conv, conv_min, conv_max)
    lat_n = normalize(latency, lat_min, lat_max)
    man_n = normalize(manual, man_min, man_max)
    risk_n = normalize(risk, risk_min, risk_max)

    return (
        W_C * conv_n -
        W_L * lat_n -
        W_M * man_n -
        W_R * risk_n
    )


# =============================================================
# HEADER
# =============================================================
st.title("💶 Instant Payments Readiness & Impact Simulator")

st.markdown("""
Analyze how **Verification of Payee (VoP)** and **Fraud Filter** thresholds impact:
- Conversion Rate  
- Latency (p95)  
- Manual Review Rate  
- Risk Exposure  
""")


# =============================================================
# SIDEBAR
# =============================================================
st.sidebar.header("⚙️ Simulation Settings")

n_rows = st.sidebar.select_slider(
    "Synthetic transactions",
    options=[5000, 20000, 50000, 100000],
    value=20000
)
seed = st.sidebar.number_input("Random seed", 0, 999999, value=42)

vop_thr = st.sidebar.slider("VoP threshold", 0.50, 0.95, value=0.80, step=0.05)
fraud_thr = st.sidebar.slider("Fraud threshold", 0.20, 0.90, value=0.50, step=0.10)

vop_grid = np.arange(0.50, 0.95, 0.05)
fraud_grid = np.arange(0.20, 0.90, 0.10)


# =============================================================
# DATA LOADING
# =============================================================
@st.cache_data(show_spinner=False)
def load_data(n, seed):
    df = generate_synth(n=n, seed=seed)
    df["is_true_fraud"] = (df["fraud_probability"] > 0.90).astype(int)
    return df

df = load_data(n_rows, seed)


# =============================================================
# CALCULATE KPIs
# =============================================================
vop_res = simulate_vop(df, threshold=vop_thr)
fraud_res = simulate_fraud(df, threshold=fraud_thr)

vop_curves = scan_vop(df, vop_grid)
fraud_curves = scan_fraud(df, fraud_grid)


# =============================================================
# KPI SNAPSHOT
# =============================================================
st.subheader("📈 Current KPI Snapshot")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Conversion Rate (%)", f"{vop_res['conversion_rate']:.1f}")
col2.metric("Latency p95 (s)", f"{vop_res['latency_p95']:.2f}")
col3.metric("Manual Review Rate (%)", f"{fraud_res['manual_review_rate']:.1f}")
col4.metric("Risk Exposure (€)", f"{fraud_res['risk_exposure_eur']:,.0f}")

st.caption(f"VoP={vop_thr:.2f} | Fraud={fraud_thr:.2f} | N={n_rows:,}")


# =============================================================
# TABS
# =============================================================
tab_h1, tab_h2, tab_h3, tab_h4, tab_heatmap = st.tabs([
    "📊 H1 — Instant vs Paper",
    "📈 H2 — Infrastructure Load",
    "🎛 H3 — VoP Simulation",
    "🔐 H4 — Fraud Simulation",
    "🌈 Full KPI Heatmap"
])


# ========== TAB H1 ==========
with tab_h1:
    st.subheader("📊 H1 — Instant vs Paper")
    st.image(os.path.join(FIG_DIR, "H1_stacked_sct_vs_paper.png"))


# ========== TAB H2 ==========
with tab_h2:
    st.subheader("📈 H2 — System Load")
    st.image(os.path.join(FIG_DIR, "H2_total_domestic.png"))
    st.image(os.path.join(FIG_DIR, "H2_total_domestic_values.png"))


# ========== TAB H3 ==========
with tab_h3:
    st.subheader("🎛 H3 — VoP Simulation")
    fig1, ax1 = plt.subplots()
    ax1.plot(vop_curves["vop_threshold"], vop_curves["conversion_rate"], marker="o")
    ax1.set_title("VoP → Conversion")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(vop_curves["vop_threshold"], vop_curves["latency_p95"], marker="o")
    ax2.set_title("VoP → Latency p95")
    st.pyplot(fig2)


# ========== TAB H4 ==========
with tab_h4:
    st.subheader("🔐 H4 — Fraud Simulation")
    st.image(os.path.join(FIG_DIR, "H4_line_fraud_risk_exposure.png"))


# =============================================================
# TAB — FINAL FULL KPI HEATMAP
# =============================================================
with tab_heatmap:
    st.subheader("🌈 Full KPI Heatmap (4-Metric Score)")

    vop_grid = np.round(np.arange(0.50, 0.95, 0.05), 2)
    fraud_grid = np.round(np.arange(0.20, 0.90, 0.10), 2)

    vop_curves["vop_threshold"] = vop_curves["vop_threshold"].round(2)
    fraud_curves["fraud_threshold"] = fraud_curves["fraud_threshold"].round(2)
    
    combined = []

    # Determine min/max for normalization
    conv_min, conv_max = vop_curves["conversion_rate"].min(), vop_curves["conversion_rate"].max()
    lat_min, lat_max = vop_curves["latency_p95"].min(), vop_curves["latency_p95"].max()
    man_min, man_max = fraud_curves["manual_review_rate"].min(), fraud_curves["manual_review_rate"].max()
    risk_min, risk_max = fraud_curves["risk_exposure_eur"].min(), fraud_curves["risk_exposure_eur"].max()

    for v in vop_grid:
        # match vop values
        conv_val = vop_curves.loc[vop_curves["vop_threshold"] == v, "conversion_rate"].values[0]
        lat_val = vop_curves.loc[vop_curves["vop_threshold"] == v, "latency_p95"].values[0]

        for f in fraud_grid:
            man_val = fraud_curves.loc[fraud_curves["fraud_threshold"] == f, "manual_review_rate"].values[0]
            risk_val = fraud_curves.loc[fraud_curves["fraud_threshold"] == f, "risk_exposure_eur"].values[0]

            full_score = compute_full_kpi(
                conv_val, lat_val, man_val, risk_val,
                conv_min, conv_max,
                lat_min, lat_max,
                man_min, man_max,
                risk_min, risk_max
            )

            combined.append([v, f, full_score])

    heat_df = pd.DataFrame(combined, columns=["VoP", "Fraud", "Full_KPI"])

    # pivot
    pivot = heat_df.pivot(index="Fraud", columns="VoP", values="Full_KPI")

    plt.figure(figsize=(9, 6))
    sns.set(style="white")
    ax = sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=0.5)
    plt.title("Full KPI Heatmap (Balanced Across 4 KPIs)")
    plt.xlabel("VoP Threshold")
    plt.ylabel("Fraud Threshold")
    st.pyplot(plt)

    
    st.success("""
### Recommended Region:
**VoP ≈ 0.80**  
**Fraud ≈ 0.50**  

This point balances:
- High Conversion  
- Low Latency  
- Low Fraud Exposure  
- Acceptable Manual Review Load
""")
    
# -----------------------------
# Optional: Download helpers (for demo completeness)
# -----------------------------
# Provide compact CSV exports of curves for easy sharing
dl_c1, dl_c2 = st.columns(2)
with dl_c1:
    st.download_button(
        label="⬇️ Download VoP curves (CSV)",
        data=vop_curves.to_csv(index=False),
        file_name="vop_curves.csv",
        mime="text/csv"
    )
with dl_c2:
    st.download_button(
        label="⬇️ Download Fraud curves (CSV)",
        data=fraud_curves.to_csv(index=False),
        file_name="fraud_curves.csv",
        mime="text/csv"
    )


# FOOTER
st.caption("© Instant Payments Readiness Simulator — Real data + simulation + full KPI scoring")
