import streamlit as st
from utils.data_loader import load_datasets

st.title("🔍 Feature Availability Analysis")

train_df, test_df, full_df = load_datasets()

train_cols = set(train_df.columns)
test_cols = set(test_df.columns)

missing = sorted(list(train_cols - test_cols))

st.subheader("Features Missing in Test Set")
st.write(len(missing))
st.dataframe(missing)

types = {col: str(full_df[col].dtype) if col in full_df else "unknown" for col in missing}

st.subheader("Missing Feature Types")
st.json(types)