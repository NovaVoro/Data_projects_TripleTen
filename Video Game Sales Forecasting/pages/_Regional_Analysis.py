import streamlit as st
import matplotlib.pyplot as plt
from utils.data_loader import load_data

df = load_data()

st.title("🌍 Regional Market Analysis")

regional = df.groupby("platform")[["na_sales", "eu_sales", "jp_sales", "other_sales"]].sum()

fig, ax = plt.subplots(figsize=(12, 6))
regional.plot(kind="bar", stacked=True, ax=ax)
st.pyplot(fig)