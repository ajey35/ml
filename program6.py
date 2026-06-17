import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load dataset
df = fetch_california_housing(as_frame=True).frame

print(df.head())

# Plot histograms
def plot_histograms(df):
    df.hist(figsize=(12, 10), bins=30)
    plt.suptitle("Histograms")
    plt.show()

# Plot box plots
def plot_boxplots(df):
    print("Hiii BoxPlottt")
    plt.figure(figsize=(12, 10))

    for i, col in enumerate(df.columns, 1):
        plt.subplot(3, 3, i)
        sns.boxplot(y=df[col])
        plt.title(col)

    plt.tight_layout()
    plt.show()

# Analyze features and detect outliers
def analyze_features(df):
    for col in df.columns:
        print("\nFeature:", col)
        print("Mean   :", round(df[col].mean(), 2))
        print("Median :", round(df[col].median(), 2))
        print("Std Dev:", round(df[col].std(), 2))

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower) | (df[col] > upper)]

        print("Outliers:", len(outliers))

# Function calls
plot_histograms(df)
plot_boxplots(df)
analyze_features(df)
