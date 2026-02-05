import streamlit as st
import pandas as pd

from utils.data_loader import load_raw_region_data
from utils.cleaning import (
    deduplicate_file,
    review_duplicates,
    apply_outlier_capping
)
from utils.defaults import DEFAULT_PARAMS

if "params" not in st.session_state:
    st.session_state["params"] = DEFAULT_PARAMS.copy()

FEATURE_COLUMNS = ["f0", "f1", "f2"]
TARGET_COLUMN = "product"

st.set_page_config(
    page_title="Data Cleaning & Outlier Handling",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("🧹 Data Cleaning & Outlier Handling")

st.markdown("""
This section performs the core data‑quality steps required before modeling:

- Deduplication of well IDs  
- Detection of within‑region and cross‑region duplicates  
- Outlier inspection using IQR  
- Outlier capping for feature columns  
- Region‑level summaries and previews  

All transformations are performed **in memory** and used by later pages.
""")

st.divider()

# -----------------------------
# Load raw data
# -----------------------------
raw_regions = load_raw_region_data()

st.header("📁 Raw Region Data (Before Cleaning)")

col1, col2, col3 = st.columns(3)
for i, (region, df) in enumerate(raw_regions.items()):
    with [col1, col2, col3][i]:
        st.subheader(region)
        st.write(f"Rows: **{len(df)}**")
        st.dataframe(df.head(5), use_container_width=True)

st.divider()

# -----------------------------
# Deduplication
# -----------------------------
st.header("🔍 Deduplication")

cleaned = []
for region, df in raw_regions.items():
    cleaned_df = deduplicate_file(df, region)
    cleaned.append(cleaned_df)

combined = pd.concat(cleaned, ignore_index=True)

st.success(f"Combined cleaned dataset: {combined.shape[0]} rows")

# -----------------------------
# Duplicate Review
# -----------------------------
st.subheader("Duplicate ID Review")

within, cross = review_duplicates(combined)

with st.expander("Within‑Region Duplicates"):
    if within.empty:
        st.info("No within‑region duplicates found.")
    else:
        st.dataframe(within, use_container_width=True)

with st.expander("Cross‑Region Duplicates"):
    if cross.empty:
        st.info("No cross‑region duplicates found.")
    else:
        st.dataframe(cross, use_container_width=True)

st.divider()

# -----------------------------
# Outlier Detection
# -----------------------------
st.header("📈 Outlier Detection (IQR Method)")

def count_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return ((df[col] < lower) | (df[col] > upper)).sum()

outlier_counts_before = {
    col: count_outliers(combined, col)
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]
}

st.subheader("Outliers Before Capping")
st.table(pd.DataFrame.from_dict(outlier_counts_before, orient="index", columns=["Count"]))

# -----------------------------
# Apply Outlier Capping
# -----------------------------
st.header("✂️ Applying Outlier Capping (Features Only)")

capped = apply_outlier_capping(combined, FEATURE_COLUMNS)

outlier_counts_after = {
    col: count_outliers(capped, col)
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]
}

st.subheader("Outliers After Capping")
st.table(pd.DataFrame.from_dict(outlier_counts_after, orient="index", columns=["Count"]))

st.info("Target column (`product`) is not capped to preserve modeling integrity.")

st.divider()

# -----------------------------
# Region Summaries
# -----------------------------
st.header("📊 Region Summaries After Cleaning")

region_counts = capped["region"].value_counts()
st.write("### Row Counts per Region")
st.table(region_counts)

st.write("### Sample Rows per Region")
sample = (
    capped.groupby("region", group_keys=False)
    .apply(lambda x: x.sample(n=min(3, len(x)), random_state=42))
)
st.dataframe(sample, use_container_width=True)

st.write("### Average Product per Region")
avg_product = capped.groupby("region")[TARGET_COLUMN].mean()
st.table(avg_product)

# -----------------------------
# Save cleaned data to session_state
# -----------------------------
st.session_state["cleaned_data"] = capped

st.success("Cleaned dataset stored for use in later pages.")

st.divider()
st.caption("NovaVoro Interactive — Data Quality Pipeline")