import streamlit as st
from utils.data_loader import load_datasets
from utils.preprocessing import prepare_training_data
from utils.modeling import train_models
from utils.metrics import smape

st.title("🧪 Model Training & Selection")

train_df, test_df, _ = load_datasets()

X_train, y_r, y_f, X_test, preprocessor = prepare_training_data(train_df, test_df)

with st.spinner("Training models..."):
    results = train_models(X_train, y_r, y_f, preprocessor)

st.subheader("Model Comparison")
st.write(results["comparison"])

st.subheader("Selected Model")
st.write(results["winner"])