import streamlit as st
from utils.data_loader import load_data

st.set_page_config(
    page_title="Insurance Benefits Analysis",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def get_data():
    return load_data("data/insurance_us.csv")

df, df_scaled, feature_names = get_data()

st.title("Insurance Benefits Analysis & Modeling Dashboard")

st.markdown(
    """
This dashboard explores an insurance dataset and demonstrates:

- Exploratory data analysis (EDA)
- Feature engineering and scaling
- k-Nearest Neighbors (kNN) classification
- Custom linear regression
- Data obfuscation with recoverable transformations
"""
)

st.subheader("Dataset preview")
st.dataframe(df.head())

st.subheader("Target variable")
st.write("`insurance_benefits` — number of benefits received.")
st.write("Binary target `insurance_benefits_received` is 1 if benefits > 0, else 0.")

col1, col2 = st.columns(2)
with col1:
    st.metric("Total records", len(df))
with col2:
    st.metric("Share with benefits",
              f"{df['insurance_benefits_received'].mean():.1%}")