import streamlit as st
from utils.data_loader import load_data
from utils.charts import plot_bar

df = load_data()

st.title("📚 Genre Analysis")

genre_sales = df.groupby("genre")["total_sales"].sum().sort_values(ascending=False)

fig = plot_bar(
    x=genre_sales.index,
    y=genre_sales.values,
    title="Total Sales by Genre"
)
st.pyplot(fig)