import streamlit as st
import pandas as pd

from utils.data_loader import load_data
from utils.modeling import (
    rnd_model_predict,
    eval_classifier,
    get_knn,
    knn_classification_experiment,
)


@st.cache_data
def get_data():
    return load_data("data/insurance_us.csv")


st.title("🤖 kNN Classification")

df, df_scaled, feature_names = get_data()

st.subheader("Class imbalance")
st.write(df["insurance_benefits_received"].value_counts())
st.write(df["insurance_benefits_received"].value_counts(normalize=True))

st.subheader("Random baseline models")
P_values = [0, df["insurance_benefits_received"].sum() / len(df), 0.5, 1]
baseline_rows = []
for P in P_values:
    y_pred_rnd = rnd_model_predict(P, size=len(df))
    f1, cm = eval_classifier(df["insurance_benefits_received"], y_pred_rnd)
    baseline_rows.append(
        {
            "P": round(P, 2),
            "F1": f1,
            "Confusion matrix (normalized)": cm,
        }
    )

st.write("Random model performance:")
for row in baseline_rows:
    st.markdown(f"**P = {row['P']}** — F1 = {row['F1']:.2f}")
    st.write(row["Confusion matrix (normalized)"])

st.subheader("kNN classification: unscaled vs scaled")
results_unscaled, results_scaled = knn_classification_experiment(
    df, df_scaled, feature_names
)

st.markdown("**Unscaled data (Euclidean distance)**")
rows_unscaled = []
for res in results_unscaled:
    rows_unscaled.append({"k": res["k"], "F1": res["f1"]})
st.dataframe(pd.DataFrame(rows_unscaled))

st.markdown("**Scaled data (Euclidean distance)**")
rows_scaled = []
for res in results_scaled:
    rows_scaled.append({"k": res["k"], "F1": res["f1"]})
st.dataframe(pd.DataFrame(rows_scaled))

st.subheader("Nearest neighbors explorer")

index = st.number_input(
    "Select row index for neighbor search",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1,
)
k = st.slider("Number of neighbors (k)", min_value=1, max_value=10, value=5)
metric = st.selectbox("Distance metric", ["euclidean", "manhattan"])
use_scaled = st.checkbox("Use scaled data", value=False)

if st.button("Find neighbors"):
    source_df = df_scaled if use_scaled else df
    neighbors = get_knn(source_df, feature_names, n=index, k=k, metric=metric)
    st.dataframe(neighbors)