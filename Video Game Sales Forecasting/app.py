import streamlit as st
from utils.data_loader import load_data

st.set_page_config(
    page_title="Video Game Sales Dashboard",
    layout="wide",
)

st.title("🎮 Video Game Sales Analysis Dashboard")

st.markdown("""
Welcome to the multi‑page analytics dashboard exploring global video game sales, 
platform trends, genre performance, regional markets, review score impact, 
and statistical hypothesis testing.

Use the sidebar to navigate through the analysis.
""")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())