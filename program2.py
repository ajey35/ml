import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Load dataset
df = fetch_california_housing(as_frame=True).frame

print(df.head())

corr = df.corr()
print("\nCorrelation Matrix:")
print(corr)


# Plot heatmap
def plot_heatmap(corr):
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr,
                annot=True,
                fmt=".2f",
                cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

# Plot pair plot
def plot_pairplot(df):
    sns.pairplot(df)
    plt.show()

# Function calls
plot_heatmap(corr)
plot_pairplot(df)
