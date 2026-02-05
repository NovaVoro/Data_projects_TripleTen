import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.modeling import train_and_evaluate
from utils.bootstrap import bootstrap_profit, compute_profit
from utils.visuals import plot_profit_distribution
from utils.defaults import DEFAULT_PARAMS

if "params" not in st.session_state:
    st.session_state["params"] = DEFAULT_PARAMS.copy()

sns.set(style="whitegrid")

FEATURE_COLUMNS = ["f0", "f1", "f2"]
TARGET_COLUMN = "product"

st.set_page_config(
    page_title="A/B Campaign Evaluation",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("🧪 A/B Campaign Evaluation")

st.markdown("""
This section validates model stability using **A/B campaign testing**.

For each region:

1. Randomly sample **1000 wells**  
2. Split into:
   - **Campaign A** → training + bootstrap simulation  
   - **Campaign B** → independent profit evaluation  
3. Compare:
   - RMSE on A  
   - Bootstrap mean profit on A  
   - Independent profit on B  
   - Loss risk  
4. Assess generalization and volatility  

This mirrors real-world exploration validation.
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
st.header("📍 Select Region for A/B Evaluation")

region = st.selectbox("Choose a region:", regions)

df_region = df[df["region"] == region].reset_index(drop=True)

if len(df_region) < 1000:
    st.error(f"Region {region} has only {len(df_region)} rows — requires at least 1000.")
    st.stop()

st.write(f"Rows in region **{region}**: {len(df_region)}")

st.divider()

# -----------------------------
# Create A/B Campaigns
# -----------------------------
st.header("🎯 Creating A/B Campaigns")

df_sampled = df_region.sample(n=1000, random_state=42).reset_index(drop=True)

campaign_A = df_sampled.iloc[:500].reset_index(drop=True)
campaign_B = df_sampled.iloc[500:].reset_index(drop=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Campaign A")
    st.write(f"Rows: {len(campaign_A)}")
    st.dataframe(campaign_A.head(5), use_container_width=True)

with col2:
    st.subheader("Campaign B")
    st.write(f"Rows: {len(campaign_B)}")
    st.dataframe(campaign_B.head(5), use_container_width=True)

st.divider()

# -----------------------------
# Train on Campaign A
# -----------------------------
st.header("🧠 Training Model on Campaign A")

res_A = train_and_evaluate(
    campaign_A,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    test_size=0.25,
    random_state=42
)

st.metric("Campaign A RMSE", f"{res_A['rmse']:.2f}")
st.metric("Mean Predicted Reserves (A Validation)", f"{res_A['mean_pred']:.2f}")

# Predict full region using A's model
X_full = df_region[FEATURE_COLUMNS]
y_true_full = df_region[TARGET_COLUMN]
y_pred_full = pd.Series(res_A["model"].predict(X_full), index=df_region.index)

st.divider()

# -----------------------------
# Bootstrap Simulation (Campaign A)
# -----------------------------
st.header("🎲 Bootstrap Profit Simulation (Campaign A Model)")

profits_A = bootstrap_profit(
    y_true=y_true_full,
    y_pred=y_pred_full,
    n_bootstrap=params["BOOTSTRAP_ITER"],
    study_size=params["NUM_POINTS_STUDY"],
    num_select=params["NUM_WELLS_SELECTED"],
    budget=params["BUDGET"],
    revenue_per_unit=params["REVENUE_PER_UNIT"],
    random_state=42
)

mean_profit_A = float(np.mean(profits_A))
ci_low_A, ci_high_A = np.percentile(profits_A, [2.5, 97.5])
loss_risk_A = float((profits_A < 0).mean() * 100)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Bootstrap Mean Profit (A)", f"${mean_profit_A:,.0f}")

with col2:
    st.metric("95% CI Low (A)", f"${ci_low_A:,.0f}")

with col3:
    st.metric("Loss Risk (A)", f"{loss_risk_A:.2f}%")

st.subheader("Profit Distribution (Campaign A Model)")
fig = plot_profit_distribution(profits_A)
st.pyplot(fig)

st.divider()

# -----------------------------
# Independent Test on Campaign B
# -----------------------------
st.header("🧪 Independent Profit Evaluation (Campaign B)")

X_B = campaign_B[FEATURE_COLUMNS]
y_B = campaign_B[TARGET_COLUMN]

y_pred_B = pd.Series(res_A["model"].predict(X_B), index=y_B.index)

profit_B = compute_profit(
    y_true=y_B,
    y_pred=y_pred_B,
    num_select=params["NUM_WELLS_SELECTED"],
    budget=params["BUDGET"],
    revenue_per_unit=params["REVENUE_PER_UNIT"]
)

st.metric("Campaign B Independent Profit", f"${profit_B:,.0f}")

st.divider()

# -----------------------------
# Summary Table
# -----------------------------
st.header("📊 A/B Summary Table")

summary_df = pd.DataFrame({
    "Region": [region],
    "RMSE (A)": [res_A["rmse"]],
    "Mean Predicted (A)": [res_A["mean_pred"]],
    "Bootstrap Mean Profit (A)": [mean_profit_A],
    "CI Low (A)": [ci_low_A],
    "CI High (A)": [ci_high_A],
    "Loss Risk (A %)": [loss_risk_A],
    "Independent Profit (B)": [profit_B],
})

st.table(summary_df)

# Save for Page 6
st.session_state.setdefault("ab_results", {})
st.session_state["ab_results"][region] = summary_df.iloc[0].to_dict()

st.success("A/B evaluation complete.")

st.divider()
st.caption("NovaVoro Interactive — Model Stability Validation")