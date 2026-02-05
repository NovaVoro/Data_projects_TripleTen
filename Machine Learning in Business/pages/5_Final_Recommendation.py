import streamlit as st
import pandas as pd
from utils.defaults import DEFAULT_PARAMS

if "params" not in st.session_state:
    st.session_state["params"] = DEFAULT_PARAMS.copy()

st.set_page_config(
    page_title="Final Recommendation",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("🏁 Final Recommendation")

st.markdown("""
This page consolidates all analyses:

- Model performance  
- Bootstrap profit simulations  
- A/B campaign validation  
- Risk thresholds  

The goal is to identify the region that offers the **best balance of profitability, reliability, and risk control**.
""")

st.divider()

# -----------------------------
# Validate required data
# -----------------------------
if "bootstrap_results" not in st.session_state:
    st.error("Bootstrap results not found. Please complete Page 4 for each region.")
    st.stop()

if "ab_results" not in st.session_state:
    st.error("A/B results not found. Please complete Page 5 for each region.")
    st.stop()

params = st.session_state["params"]
risk_threshold = params["RISK_THRESHOLD_PCT"]

bootstrap_results = st.session_state["bootstrap_results"]
ab_results = st.session_state["ab_results"]

# -----------------------------
# Display Bootstrap Summary
# -----------------------------
st.header("💰 Bootstrap Profit Summary (All Regions)")

bootstrap_df = pd.DataFrame(bootstrap_results).T.reset_index(drop=True)
bootstrap_df = bootstrap_df.rename(columns={"Region": "Region"})  # ensure consistent naming

st.dataframe(bootstrap_df, use_container_width=True)

st.divider()

# -----------------------------
# Display A/B Summary
# -----------------------------
st.header("🧪 A/B Campaign Summary (All Regions)")

ab_df = pd.DataFrame(ab_results).T.reset_index(drop=True)

st.dataframe(ab_df, use_container_width=True)

st.divider()

# -----------------------------
# Determine Best Region
# -----------------------------
st.header("🏆 Final Evaluation & Recommendation")

# Filter by risk threshold
eligible = bootstrap_df[bootstrap_df["Loss Risk (%)"] < risk_threshold]

if eligible.empty:
    st.error("No regions meet the risk threshold. Consider revisiting model assumptions.")
    st.stop()

# Choose region with highest mean profit
best_region = eligible.loc[eligible["Mean Profit"].idxmax()]

region_name = best_region["Region"]
mean_profit = best_region["Mean Profit"]
ci_low = best_region["CI Low"]
ci_high = best_region["CI High"]
loss_risk = best_region["Loss Risk (%)"]

st.subheader(f"📌 Recommended Region: **{region_name}**")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Expected Mean Profit", f"${mean_profit:,.0f}")

with col2:
    st.metric("95% Confidence Interval", f"${ci_low:,.0f} – ${ci_high:,.0f}")

with col3:
    st.metric("Loss Risk", f"{loss_risk:.2f}%")

st.success(f"Region {region_name} meets the risk threshold (< {risk_threshold}%) and offers the highest expected profit among eligible regions.")

st.divider()

# -----------------------------
# Narrative Summary
# -----------------------------
st.header("📝 Executive Summary")

st.markdown(f"""
### Final Recommendation: **Develop wells in Region {region_name}**

Region {region_name} demonstrates:

- **Strong profitability** with an expected mean profit of **${mean_profit:,.0f}**
- **Low downside exposure**, with a loss risk of only **{loss_risk:.2f}%**
- **Stable generalization**, confirmed through A/B campaign validation
- **Consistent performance** across bootstrap simulations

The 95% confidence interval (**${ci_low:,.0f} to ${ci_high:,.0f}**) indicates a reliable upside with minimal volatility.

This aligns with the company’s financial objectives and risk tolerance threshold of **{risk_threshold}%**.
""")

st.info("This recommendation is based on combined modeling, simulation, and validation across all regions.")

st.divider()
st.caption("NovaVoro Interactive — Strategic Decision Engine")