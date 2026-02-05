import streamlit as st
import pandas as pd

from utils.data_loader import load_raw_region_data
from utils.defaults import DEFAULT_PARAMS

if "params" not in st.session_state:
    st.session_state["params"] = DEFAULT_PARAMS.copy()

st.set_page_config(
    page_title="Oil Region Analysis — Overview",
    layout="wide"
)

# -----------------------------
# Title & Description
# -----------------------------
st.title("📊 Oil Region Profitability Analysis — Overview")

st.markdown("""
This dashboard evaluates three oil regions using machine learning, 
profit simulations, and risk analysis.  
The workflow mirrors a real-world exploration decision pipeline:

- Data cleaning & deduplication  
- Outlier inspection and capping  
- Linear regression modeling per region  
- Break-even analysis  
- Bootstrap profit simulation  
- A/B campaign validation  
- Final recommendation for development  

Use the sidebar to adjust project parameters and explore each stage.
""")

st.divider()

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Project Parameters")

NUM_POINTS_STUDY = st.sidebar.number_input(
    "Wells per exploration campaign",
    min_value=100,
    max_value=2000,
    value=500,
    step=50
)

NUM_WELLS_SELECTED = st.sidebar.number_input(
    "Top wells selected for development",
    min_value=50,
    max_value=500,
    value=200,
    step=10
)

BUDGET = st.sidebar.number_input(
    "Budget (USD)",
    min_value=10_000_000,
    max_value=500_000_000,
    value=100_000_000,
    step=5_000_000,
    format="%i"
)

REVENUE_PER_UNIT = st.sidebar.number_input(
    "Revenue per thousand barrels (USD)",
    min_value=1000,
    max_value=10000,
    value=4500,
    step=250
)

RISK_THRESHOLD_PCT = st.sidebar.slider(
    "Maximum acceptable loss risk (%)",
    min_value=0.0,
    max_value=10.0,
    value=2.5,
    step=0.1
)

BOOTSTRAP_ITER = st.sidebar.number_input(
    "Bootstrap iterations",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100
)

st.sidebar.success("Parameters updated")

# Store parameters in session_state for other pages
st.session_state["params"] = {
    "NUM_POINTS_STUDY": NUM_POINTS_STUDY,
    "NUM_WELLS_SELECTED": NUM_WELLS_SELECTED,
    "BUDGET": BUDGET,
    "REVENUE_PER_UNIT": REVENUE_PER_UNIT,
    "RISK_THRESHOLD_PCT": RISK_THRESHOLD_PCT,
    "BOOTSTRAP_ITER": BOOTSTRAP_ITER,
}

# -----------------------------
# Load Raw Data
# -----------------------------
st.header("📁 Loaded Region Datasets")

regions = load_raw_region_data()

col1, col2, col3 = st.columns(3)

for i, (region, df) in enumerate(regions.items()):
    with [col1, col2, col3][i]:
        st.subheader(region)
        st.write(f"Rows: **{len(df)}**")
        st.dataframe(df.head(5), use_container_width=True)

st.info("Data is loaded automatically from the `datasets/` folder.")

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption("NovaVoro Interactive — Machine Learning in Business • Streamlit Edition")