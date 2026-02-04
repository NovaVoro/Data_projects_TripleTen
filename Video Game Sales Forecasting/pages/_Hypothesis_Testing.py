import streamlit as st
from scipy.stats import ttest_ind
from utils.data_loader import load_data

df = load_data()

st.title("📐 Hypothesis Testing")

st.subheader("Xbox One vs PC — User Scores")
xbox = df[df["platform"] == "XOne"]["user_score"].dropna()
pc = df[df["platform"] == "PC"]["user_score"].dropna()

t1, p1 = ttest_ind(xbox, pc, equal_var=False)
st.write(f"T-statistic: {t1:.3f}, P-value: {p1:.3f}")

st.subheader("Action vs Sports — User Scores")
action = df[df["genre"] == "Action"]["user_score"].dropna()
sports = df[df["genre"] == "Sports"]["user_score"].dropna()

t2, p2 = ttest_ind(action, sports, equal_var=False)
st.write(f"T-statistic: {t2:.3f}, P-value: {p2:.3f}")