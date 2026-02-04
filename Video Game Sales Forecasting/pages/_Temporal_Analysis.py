import streamlit as st
from utils.data_loader import load_data
from utils.charts import plot_bar

df = load_data()

st.title("⏳ Temporal Analysis")

releases = df.groupby("year_of_release")["name"].count()

fig = plot_bar(
    x=releases.index,
    y=releases.values,
    title="Game Releases by Year"
)
st.pyplot(fig)