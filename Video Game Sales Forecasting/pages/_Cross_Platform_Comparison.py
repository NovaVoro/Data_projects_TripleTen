import streamlit as st
from utils.data_loader import load_data

df = load_data()

st.title("🔀 Cross‑Platform Comparison")

multi = df.groupby("name")["platform"].nunique()
multi = df[df["name"].isin(multi[multi > 1].index)]

st.subheader("Multi‑Platform Games")
st.dataframe(multi.head())