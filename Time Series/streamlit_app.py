# streamlit_app.py
import streamlit as st
from model_core import train_all_models, predict_next_hour

CSV_PATH = "taxi.csv"

@st.cache_resource
def load_and_train():
    return train_all_models(CSV_PATH)

st.title("Sweet Lift Taxi - Hourly Demand Forecast")

st.write("Predict the number of taxi orders for the next hour using historical data.")

with st.spinner("Training models..."):
    results = load_and_train()

best_name = results["best_model_name"]
best_rmse = results["best_test_rmse"]
all_results = results["all_results"]
model = results["best_model"]
df_features = results["df_features"]
df_hourly = results["df_hourly"]

st.subheader("Model performance (Test RMSE)")
for name, res in all_results.items():
    st.write(f"**{name}** - Test RMSE: {res['test_rmse']:.3f}")

st.success(f"Best model: **{best_name}** with Test RMSE = {best_rmse:.3f}")

if st.button("Predict next hour demand"):
    pred = predict_next_hour(model, df_features, df_hourly)
    st.subheader("Next hour prediction")
    st.write(f"Last timestamp in data: `{pred['last_time']}`")
    st.write(f"Next hour: `{pred['next_time']}`")
    st.metric("Predicted number of orders", f"{pred['prediction']:.0f}")