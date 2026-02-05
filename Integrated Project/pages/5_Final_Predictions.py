import streamlit as st
from utils.data_loader import load_datasets
from utils.preprocessing import prepare_training_data
from utils.modeling import train_models

st.title("📈 Final Predictions")

train_df, test_df, _ = load_datasets()

X_train, y_r, y_f, X_test, preprocessor = prepare_training_data(train_df, test_df)
results = train_models(X_train, y_r, y_f, preprocessor)

rougher_pred = results["best_rougher"].predict(X_test)
final_pred = results["best_final"].predict(X_test)

st.subheader("Prediction Summary")
st.write({
    "Rougher Mean": float(rougher_pred.mean()),
    "Final Mean": float(final_pred.mean())
})

st.subheader("Preview")
st.dataframe({
    "rougher.output.recovery": rougher_pred,
    "final.output.recovery": final_pred
})