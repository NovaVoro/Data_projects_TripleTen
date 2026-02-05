import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def plot_histograms(df, numeric_cols):
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(5, 3))
        df[col].hist(bins=40, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

def plot_correlations(df, numeric_cols):
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

def plot_boxplots(df, numeric_cols):
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f"Boxplot of {col}")
        st.pyplot(fig)