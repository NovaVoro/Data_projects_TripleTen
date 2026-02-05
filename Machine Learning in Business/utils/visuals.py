import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid")

def plot_profit_distribution(profits):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(profits, bins=40, kde=True, color="steelblue", ax=ax)
    ci_low, ci_high = np.percentile(profits, [2.5, 97.5])
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.axvline(ci_low, color="orange", linestyle="--", linewidth=1)
    ax.axvline(ci_high, color="green", linestyle="--", linewidth=1)
    ax.set_title("Profit Distribution (Bootstrap)")
    ax.set_xlabel("Profit (USD)")
    return fig