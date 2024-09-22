# src/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_data(file_path):
    return pd.read_csv(file_path)


def basic_statistics(df):
    return df.describe()


def create_visualizations(df):
    # Histogram
    df.hist(figsize=(12, 10))
    plt.tight_layout()
    plt.savefig("results/histograms.png")
    plt.close()

    # Box plots
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="campaign", y="conversion", data=df)
    plt.title("Conversion by Campaign")
    plt.savefig("results/boxplot_conversion.png")
    plt.close()

    # Scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="age", y="conversion", hue="campaign", data=df)
    plt.title("Conversion vs Age by Campaign")
    plt.savefig("results/scatterplot_age_conversion.png")
    plt.close()


def analyze_correlations(df):
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.savefig("results/correlation_matrix.png")
    plt.close()
    return corr_matrix


if __name__ == "__main__":
    df = load_data("data/marketing_ab.csv")
    print(basic_statistics(df))
    create_visualizations(df)
    corr_matrix = analyze_correlations(df)
    print("Correlation matrix:")
    print(corr_matrix)
