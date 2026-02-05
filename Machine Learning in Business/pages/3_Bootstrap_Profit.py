import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.modeling import train_and_evaluate
from utils.bootstrap import bootstrap_profit
from utils.visuals import plot_profit_distribution
from utils.defaults import DEFAULT_PARAMS

if "params" not in st.session_state:
    st.session_state["params"] = DEFAULT_PARAMS.copy()

sns.set(style="whitegrid")

FEATURE_COLUMNS = ["f0", "f1", "f2"]
TARGET_COLUMN = "product"

st.set_page_config(
    page_title="Bootstrap Profit Simulation",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("💰 Bootstrap Profit Simulation")

st.markdown("""
This section simulates **1,000+ exploration campaigns** per region using bootstrap sampling.

Each simulation:
- Samples wells with replacement  
- Selects the top predicted wells  
- Computes profit (true reserves × revenue − budget)  

This produces a **profit distribution** that reveals:
- Expected mean profit  
- 95% confidence interval  
- Loss risk (percentage of simulations below zero)  
""")

st.divider()

# -----------------------------
# Load cleaned data
# -----------------------------
if "cleaned_data" not in st.session_state:
    st.error("Cleaned dataset not found. Please complete Page 2 first.")
    st.stop()

df = st.session_state["cleaned_data"]
regions = sorted(df["region"].unique())

params = st.session_state["params"]

# -----------------------------
# Region selection
# -----------------------------
st.header("📍 Select Region for Simulation")

region = st.selectbox("Choose a region:", regions)

df_region = df[df["region"] == region].reset_index(drop=True)

st.write(f"Rows in region **{region}**: {len(df_region)}")

st.divider()

# -----------------------------
# Train model on full region
# -----------------------------
st.header("🧠 Training Model for Bootstrap Simulation")

results = train_and_evaluate(
    df_region,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    test_size=0.25,
    random_state=42
)

model = results["model"]

st.metric("Validation RMSE", f"{results['rmse']:.2f}")
st.metric("Mean Predicted Reserves (Validation)", f"{results['mean_pred']:.2f}")

# Predict on full region
X_full = df_region[FEATURE_COLUMNS]
y_true_full = df_region[TARGET_COLUMN]
y_pred_full = pd.Series(model.predict(X_full), index=df_region.index)

st.divider()

# -----------------------------
# Bootstrap Simulation
# -----------------------------
st.header("🎲 Running Bootstrap Profit Simulation")

profits = bootstrap_profit(
    y_true=y_true_full,
    y_pred=y_pred_full,
    n_bootstrap=params["BOOTSTRAP_ITER"],
    study_size=params["NUM_POINTS_STUDY"],
    num_select=params["NUM_WELLS_SELECTED"],
    budget=params["BUDGET"],
    revenue_per_unit=params["REVENUE_PER_UNIT"],
    random_state=42
)

mean_profit = float(np.mean(profits))
ci_low, ci_high = np.percentile(profits, [2.5, 97.5])
loss_risk = float((profits < 0).mean() * 100)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Mean Profit", f"${mean_profit:,.0f}")

with col2:
    st.metric("95% CI Low", f"${ci_low:,.0f}")

with col3:
    st.metric("Loss Risk", f"{loss_risk:.2f}%")

st.divider()

# -----------------------------
# Profit Distribution Plot
# -----------------------------
st.header("📉 Profit Distribution")

fig = plot_profit_distribution(profits)
st.pyplot(fig)

st.divider()

# -----------------------------
# Summary Table
# -----------------------------
st.header("📊 Summary Table")

summary_df = pd.DataFrame({
    "Region": [region],
    "Mean Profit": [mean_profit],
    "CI Low": [ci_low],
    "CI High": [ci_high],
    "Loss Risk (%)": [loss_risk],
    "Bootstrap Iterations": [params["BOOTSTRAP_ITER"]],
    "Study Size": [params["NUM_POINTS_STUDY"]],
    "Wells Selected": [params["NUM_WELLS_SELECTED"]],
})

st.table(summary_df)

# Store results for Page 6
st.session_state.setdefault("bootstrap_results", {})
st.session_state["bootstrap_results"][region] = summary_df.iloc[0].to_dict()

st.success("Bootstrap simulation complete.")

st.divider()
st.caption("NovaVoro Interactive — Profit Simulation Engine")