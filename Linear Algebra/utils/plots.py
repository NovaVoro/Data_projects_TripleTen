import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_target_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x="insurance_benefits", data=df, palette="Blues", ax=ax)
    ax.set_title("Distribution of Insurance Benefits")
    ax.set_xlabel("Number of Benefits")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_binary_target_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="insurance_benefits_received", data=df, palette="pastel", ax=ax)
    ax.set_title("Binary Target: Insurance Benefits Received")
    ax.set_xlabel("Received Benefits (1 = Yes, 0 = No)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    corr_matrix = df.corr(numeric_only=True)
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix")
    fig.tight_layout()
    return fig


def plot_boxplots_by_benefits(df: pd.DataFrame, features):
    figs = []
    for feature in features:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(
            x="insurance_benefits",
            y=feature,
            data=df,
            palette="Set2",
            ax=ax,
        )
        ax.set_title(f"{feature.capitalize()} by Insurance Benefits")
        ax.set_xlabel("Insurance Benefits")
        ax.set_ylabel(feature.capitalize())
        fig.tight_layout()
        figs.append(fig)
    return figs


def plot_histograms(df: pd.DataFrame, features):
    figs = []
    for feature in features:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[feature], kde=True, bins=30, color="teal", ax=ax)
        ax.set_title(f"Distribution of {feature.capitalize()}")
        ax.set_xlabel(feature.capitalize())
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        figs.append(fig)
    return figs


def plot_gender_boxplots(df: pd.DataFrame, features):
    figs = []
    for feature in features:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x="gender", y=feature, data=df, palette="cool", ax=ax)
        ax.set_title(f"{feature.capitalize()} by Gender")
        ax.set_xlabel("Gender (0 = Female, 1 = Male)")
        ax.set_ylabel(feature.capitalize())
        fig.tight_layout()
        figs.append(fig)
    return figs


def plot_income_per_member_box(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        x="insurance_benefits",
        y="income_per_member",
        data=df,
        palette="Purples",
        ax=ax,
    )
    ax.set_title("Income per Family Member by Insurance Benefits")
    ax.set_xlabel("Insurance Benefits")
    ax.set_ylabel("Income per Member")
    fig.tight_layout()
    return fig


def plot_scaling_preview(df_original: pd.DataFrame, df_scaled: pd.DataFrame, features):
    fig, axes = plt.subplots(2, len(features), figsize=(5 * len(features), 8))
    for i, feature in enumerate(features):
        sns.histplot(
            df_original[feature],
            ax=axes[0, i],
            kde=True,
            color="coral",
        )
        axes[0, i].set_title(f"Original {feature}")

        sns.histplot(
            df_scaled[feature],
            ax=axes[1, i],
            kde=True,
            color="skyblue",
        )
        axes[1, i].set_title(f"Scaled {feature}")

    fig.tight_layout()
    return fig