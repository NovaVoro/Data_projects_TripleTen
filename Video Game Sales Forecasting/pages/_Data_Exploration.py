import streamlit as st
from utils.data_loader import load_data

df = load_data()

st.title("📊 Data Exploration")

st.subheader("Dataset Overview")
st.dataframe(df.head())

st.subheader("Summary Statistics")
st.write(df.describe())

st.subheader("Missing Values (%)")
st.write((df.isnull().sum() * 100 / len(df)).round(2))