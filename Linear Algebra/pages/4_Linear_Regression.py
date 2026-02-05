import streamlit as st
import numpy as np

from utils.data_loader import load_data
from utils.modeling import run_linear_regression


@st.cache_data
def get_data():
    return load_data("data/insurance_us.csv")


st.title("📐 Custom Linear Regression")

df, df_scaled, feature_names = get_data()

st.subheader("Model setup")
st.write(
    "Custom linear regression is trained to predict `insurance_benefits` "
    "from `gender`, `age`, `income`, and `family_members`."
)

results = run_linear_regression(df, df_scaled, feature_names)

st.subheader("Unscaled data results")
st.write(f"RMSE: {results['unscaled']['rmse']:.2f}")
st.write(f"R²: {results['unscaled']['r2']:.2f}")
st.write("Weights (bias first):")
st.write(np.round(results["unscaled"]["weights"], 4))

st.subheader("Scaled data results")
st.write(f"RMSE: {results['scaled']['rmse']:.2f}")
st.write(f"R²: {results['scaled']['r2']:.2f}")
st.write("Weights (bias first):")
st.write(np.round(results["scaled"]["weights"], 4))