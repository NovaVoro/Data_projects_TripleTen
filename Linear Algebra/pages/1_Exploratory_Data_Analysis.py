import streamlit as st
import pandas as pd

from utils.data_loader import load_data
from utils.plots import (
    plot_target_distribution,
    plot_binary_target_distribution,
    plot_correlation_heatmap,
    plot_boxplots_by_benefits,
    plot_histograms,
    plot_gender_boxplots,
)


@st.cache_data
def get_data():
    return load_data("data/insurance_us.csv")


st.title("📊 Exploratory Data Analysis")

df, df_scaled, feature_names = get_data()

st.subheader("Dataset overview")
st.dataframe(df.head())

st.subheader("Descriptive statistics")
st.dataframe(df.describe())

st.subheader("Missing values")
missing = df.isnull().sum()
st.dataframe(pd.DataFrame({"missing": missing}))

st.subheader("Target distribution")
st.write("Normalized value counts for `insurance_benefits`:")
st.write(df["insurance_benefits"].value_counts(normalize=True))

fig = plot_target_distribution(df)
st.pyplot(fig)

st.subheader("Binary target distribution")
st.write(df["insurance_benefits_received"].value_counts())
st.write(df["insurance_benefits_received"].value_counts(normalize=True))
st.pyplot(plot_binary_target_distribution(df))

st.subheader("Correlation matrix")
st.pyplot(plot_correlation_heatmap(df))

st.subheader("Boxplots by insurance benefits")
box_feats = ["age", "income", "family_members"]
for fig_box in plot_boxplots_by_benefits(df, box_feats):
    st.pyplot(fig_box)

st.subheader("Feature distributions")
for fig_hist in plot_histograms(df, ["age", "income"]):
    st.pyplot(fig_hist)

st.subheader("Gender-based comparisons")
for fig_gender in plot_gender_boxplots(df, ["income", "age"]):
    st.pyplot(fig_gender)

st.subheader("Group-based aggregations")
grouped = df.groupby("insurance_benefits")[["age", "income", "family_members"]].agg(
    ["mean", "median", "count"]
)
st.dataframe(grouped)