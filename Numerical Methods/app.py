import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.data_loader import load_data
from utils.cleaning import clean_data
from utils.features import engineer_features
from utils.eda import eda_summary
from utils.visuals import (
    plot_histograms,
    plot_correlations,
    plot_boxplots
)
from utils.preprocessors import (
    build_linear_preprocessor,
    build_tree_preprocessor
)
from utils.modeling import (
    train_and_evaluate,
    build_models
)

# ---------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Car Price Prediction Dashboard")
st.write("A complete end‑to‑end ML workflow converted into a Streamlit app.")

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------
DATA_PATH = "data/car_data.csv"

@st.cache_data
def load_sample():
    return load_data(DATA_PATH)

df_raw = load_sample()

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
tabs = [
    "Home",
    "Data Preview",
    "Cleaning & Features",
    "Visualizations",
    "Model Training",
    "Model Comparison"
]

page = st.sidebar.radio("Navigation", tabs)

# ---------------------------------------------------------
# HOME
# ---------------------------------------------------------
if page == "Home":
    st.header("Project Overview")
    st.write("""
    This dashboard demonstrates a full machine learning workflow for predicting car prices.
    It includes:
    - Data loading  
    - Cleaning  
    - Feature engineering  
    - EDA  
    - Visualizations  
    - Model training  
    - RMSE comparison  
    """)

    st.subheader("Dataset")
    st.write("The app uses a built‑in sample dataset located at `data/sample_car_data.csv`.")

# ---------------------------------------------------------
# DATA PREVIEW
# ---------------------------------------------------------
elif page == "Data Preview":
    st.header("📄 Raw Dataset Preview")

    st.write("### Shape")
    st.write(df_raw.shape)

    st.write("### Columns")
    st.write(df_raw.columns.tolist())

    st.write("### Head")
    st.dataframe(df_raw.head())

    st.write("### EDA Summary")
    eda_summary(df_raw)

# ---------------------------------------------------------
# CLEANING & FEATURES
# ---------------------------------------------------------
elif page == "Cleaning & Features":
    st.header("🧹 Cleaning & Feature Engineering")

    df_clean = clean_data(df_raw)
    st.write("### After Cleaning")
    st.write(df_clean.shape)
    st.dataframe(df_clean.head())

    df_feat = engineer_features(df_clean)
    st.write("### After Feature Engineering")
    st.write(df_feat.shape)
    st.dataframe(df_feat.head())

    st.write("### Missingness After Cleaning")
    st.write(df_clean.isna().mean().sort_values(ascending=False))

# ---------------------------------------------------------
# VISUALIZATIONS
# ---------------------------------------------------------
elif page == "Visualizations":
    st.header("📊 Visualizations")

    df_clean = clean_data(df_raw)
    df_feat = engineer_features(df_clean)

    numeric_cols = df_feat.select_dtypes(exclude=["object"]).columns.tolist()
    categorical_cols = df_feat.select_dtypes(include=["object"]).columns.tolist()

    st.subheader("Histograms")
    plot_histograms(df_feat, numeric_cols)

    st.subheader("Correlation Heatmap")
    plot_correlations(df_feat, numeric_cols)

    st.subheader("Boxplots")
    plot_boxplots(df_feat, numeric_cols)

# ---------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------
elif page == "Model Training":
    st.header("🤖 Model Training")

    df_clean = clean_data(df_raw)
    df_feat = engineer_features(df_clean)

    TARGET = "LogPrice"
    df_feat = df_feat.dropna(subset=[TARGET])

    drop_cols = ["DateCrawled", "DateCreated", "LastSeen", "Price"]
    df_feat = df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns])

    X = df_feat.drop(columns=[TARGET])
    y = df_feat[TARGET]

    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    linear_pre = build_linear_preprocessor(numeric_cols, categorical_cols)
    tree_pre = build_tree_preprocessor(numeric_cols, categorical_cols)

    models = build_models()

    selected_model = st.selectbox(
        "Choose a model",
        list(models.keys())
    )

    if st.button("Train Model"):
        pipe, rmse = train_and_evaluate(
            selected_model,
            models[selected_model],
            linear_pre if selected_model == "Linear Regression" else tree_pre,
            X, y
        )

        st.success(f"RMSE: {rmse:.2f}")

# ---------------------------------------------------------
# MODEL COMPARISON
# ---------------------------------------------------------
elif page == "Model Comparison":
    st.header("📈 Model Comparison")

    df_clean = clean_data(df_raw)
    df_feat = engineer_features(df_clean)

    TARGET = "LogPrice"
    df_feat = df_feat.dropna(subset=[TARGET])

    drop_cols = ["DateCrawled", "DateCreated", "LastSeen", "Price"]
    df_feat = df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns])

    X = df_feat.drop(columns=[TARGET])
    y = df_feat[TARGET]

    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    linear_pre = build_linear_preprocessor(numeric_cols, categorical_cols)
    tree_pre = build_tree_preprocessor(numeric_cols, categorical_cols)

    models = build_models()

    results = []

    for name, model in models.items():
        pipe, rmse = train_and_evaluate(
            name,
            model,
            linear_pre if name == "Linear Regression" else tree_pre,
            X, y,
            silent=True
        )
        results.append([name, rmse])

    results_df = pd.DataFrame(results, columns=["Model", "RMSE"]).sort_values("RMSE")

    st.write("### RMSE Comparison")
    st.dataframe(results_df)