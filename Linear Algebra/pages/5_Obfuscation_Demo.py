import streamlit as st
import numpy as np

from utils.data_loader import load_data
from utils.obfuscation import (
    generate_random_invertible_matrix,
    transform_features,
    recover_features,
    test_lr_with_obfuscation,
)


@st.cache_data
def get_data():
    return load_data("data/insurance_us.csv")


st.title("🔒 Obfuscation Demo")

df, df_scaled, feature_names = get_data()

st.subheader("Feature matrix for obfuscation")
personal_info_column_list = ["gender", "age", "income", "family_members"]
X = df[personal_info_column_list].to_numpy()
y = df["insurance_benefits"].to_numpy()

st.write("First 5 rows of original feature matrix:")
st.write(X[:5])

st.subheader("Random transformation matrix P")
P, det_P, cond_number, invertible = generate_random_invertible_matrix(
    X.shape[1],
    seed=42,
)
st.write(f"Determinant of P: {det_P}")
st.write(f"Condition number of P: {cond_number:.2f}")
st.write(f"Is P invertible? {invertible}")

X_prime = transform_features(X, P)
X_recovered = recover_features(X_prime, P)

st.subheader("Transformed vs recovered data (first 5 rows)")
st.markdown("**Transformed (X')**")
st.write(np.round(X_prime[:5], 4))

st.markdown("**Recovered (X' @ P⁻¹)**")
st.write(np.round(X_recovered[:5], 4))

st.markdown("**Original (X)**")
st.write(X[:5])

st.subheader("Linear regression with obfuscation")
results = test_lr_with_obfuscation(X, y, P)

st.markdown("**Original data**")
st.write(f"RMSE: {results['original']['rmse']:.2f}")
st.write(f"R²: {results['original']['r2']:.2f}")

st.markdown("**Obfuscated data**")
st.write(f"RMSE: {results['obfuscated']['rmse']:.2f}")
st.write(f"R²: {results['obfuscated']['r2']:.2f}")