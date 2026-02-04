import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from utils.data_loader import load_data

df = load_data()

st.title("🧹 Data Preparation")

st.subheader("Missing Value Correlation")
missing_corr = df.isnull().astype(int).corr()

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(missing_corr, cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.subheader("Imputed Critic Scores (Median)")
imputer = SimpleImputer(strategy="median")
df["critic_score_median"] = imputer.fit_transform(df[["critic_score"]])

fig, ax = plt.subplots(figsize=(10, 5))
df.groupby("year_of_release")["critic_score_median"].mean().plot(ax=ax)
st.pyplot(fig)