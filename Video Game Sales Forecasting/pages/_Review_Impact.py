import streamlit as st
from utils.data_loader import load_data
from utils.charts import plot_scatter

df = load_data()

st.title("⭐ Review Score Impact")

platform = st.selectbox("Choose a platform:", df["platform"].unique())
dfp = df[df["platform"] == platform]

st.subheader("Critic Score vs Sales")
st.pyplot(plot_scatter(dfp, "critic_score", "total_sales", "Critic Score vs Sales"))

st.subheader("User Score vs Sales")
st.pyplot(plot_scatter(dfp, "user_score", "total_sales", "User Score vs Sales"))