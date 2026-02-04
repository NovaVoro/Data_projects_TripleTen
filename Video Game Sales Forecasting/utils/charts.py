import matplotlib.pyplot as plt
import seaborn as sns

def plot_bar(x, y, title, rotation=45):
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x=x, y=y, ax=ax)
    plt.xticks(rotation=rotation)
    plt.title(title)
    return fig

def plot_heatmap(data, title):
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(data, cmap="YlGnBu", ax=ax)
    plt.title(title)
    return fig

def plot_scatter(df, x, y, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=df, x=x, y=y, ax=ax)
    plt.title(title)
    return fig