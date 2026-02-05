import streamlit as st
import pandas as pd
from utils.data_loader import load_datasets
from utils.analysis import summarize_stage_concentrations

st.title("📊 Data Overview")

train_df, test_df, full_df = load_datasets()

st.subheader("Dataset Shapes")
st.write({
    "Train": train_df.shape,
    "Test": test_df.shape,
    "Full": full_df.shape
})

st.subheader("Train Sample")
st.dataframe(train_df.head())

st.subheader("Stage Concentration Summary")
summary = summarize_stage_concentrations(train_df)
st.dataframe(summary)