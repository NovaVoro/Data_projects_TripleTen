import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.modeling import train_and_evaluate, break_even_threshold
from utils.defaults import DEFAULT_PARAMS

if "params" not in st.session_state:
    st.session_state["params"] = DEFAULT_PARAMS.copy()

sns.set(style="whitegrid")

FEATURE_COLUMNS = ["f0", "f1", "f2"]
TARGET_COLUMN = "product"

st.set_page_config(
    page_title="Model Training & Evaluation",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("🤖 Model Training & Evaluation")

st.markdown("""
This section trains a **Linear Regression model** for each region and evaluates:

- Validation RMSE  
- Mean predicted reserves  
- Predicted vs. actual scatterplot  
- Break‑even threshold comparison  
- Region‑level summary table  

All modeling uses the cleaned dataset from Page 2.
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

# -----------------------------
# Region selection
# -----------------------------
st.header("📍 Select Region for Model Training")

region = st.selectbox("Choose a region:", regions)

df_region = df[df["region"] == region].reset_index(drop=True)

st.write(f"Rows in region **{region}**: {len(df_region)}")

st.dataframe(df_region.head(10), use_container_width=True)

st.divider()

# -----------------------------
# Train model
# -----------------------------
st.header("🧠 Training Linear Regression Model")

results = train_and_evaluate(
    df_region,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    test_size=0.25,
    random_state=42
)

rmse = results["rmse"]
mean_pred = results["mean_pred"]

col1, col2 = st.columns(2)

with col1:
    st.metric("Validation RMSE", f"{rmse:.2f}")

with col2:
    st.metric("Mean Predicted Reserves (Validation)", f"{mean_pred:.2f} thousand barrels")

# -----------------------------
# Scatterplot: Predicted vs Actual
# -----------------------------
st.subheader("📉 Predicted vs. Actual (Validation Set)")

fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(
    x=results["y_valid"],
    y=results["y_pred"],
    ax=ax,
    color="steelblue"
)
ax.set_xlabel("Actual Product (thousand barrels)")
ax.set_ylabel("Predicted Product")
ax.set_title(f"Predicted vs Actual — {region}")
ax.plot([results["y_valid"].min(), results["y_valid"].max()],
        [results["y_valid"].min(), results["y_valid"].max()],
        color="red", linestyle="--")
st.pyplot(fig)

st.divider()

# -----------------------------
# Break-even threshold
# -----------------------------
st.header("💵 Break-Even Threshold Comparison")

params = st.session_state["params"]

t_break_even = break_even_threshold(
    params["BUDGET"],
    params["REVENUE_PER_UNIT"],
    params["NUM_WELLS_SELECTED"]
)

avg_product = df_region[TARGET_COLUMN].mean()

col1, col2 = st.columns(2)

with col1:
    st.metric("Break-Even Product per Well", f"{t_break_even:.2f}")

with col2:
    st.metric("Average Product (Region)", f"{avg_product:.2f}")

if avg_product < t_break_even:
    st.warning("Average reserves are below break-even. Profitability depends on accurate well selection.")
else:
    st.success("Average reserves exceed break-even. Region shows strong baseline potential.")

st.divider()

# -----------------------------
# Region Summary Table
# -----------------------------
st.header("📊 Region Summary")

summary_df = pd.DataFrame({
    "Region": [region],
    "Rows": [len(df_region)],
    "Avg Product": [avg_product],
    "RMSE": [rmse],
    "Mean Predicted Reserves": [mean_pred],
    "Break-Even Threshold": [t_break_even]
})

st.table(summary_df)

st.divider()
st.caption("NovaVoro Interactive — Predictive Modeling Pipeline")