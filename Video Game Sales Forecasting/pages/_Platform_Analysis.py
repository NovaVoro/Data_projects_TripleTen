import streamlit as st
from utils.data_loader import load_data
from utils.charts import plot_heatmap

df = load_data()

st.title("🕹️ Platform Sales Analysis")

platform_sales = df.pivot_table(
    index="year_of_release",
    columns="platform",
    values="total_sales",
    aggfunc="sum",
    fill_value=0,
)

fig = plot_heatmap(platform_sales, "Platform Sales Over Time")
st.pyplot(fig)