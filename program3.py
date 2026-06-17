import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
def load_data():
    iris = load_iris()
    return iris.data, iris.target, iris.target_names

# Standardize data
def standardize_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

# Apply PCA
def apply_pca(X):
    pca = PCA(n_components=2)
    return pca.fit_transform(X)

# Plot PCA result
def plot_pca(X_pca, y, target_names):
    plt.figure(figsize=(8, 6))

    colors = ['r', 'g', 'b']

    for target, color, label in zip(range(3), colors, target_names):
        plt.scatter(
            X_pca[y == target, 0],
            X_pca[y == target, 1],
            color=color,
            label=label
        )

    plt.title("PCA of Iris Dataset")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.show()

# Main
X, y, target_names = load_data()
X_std = standardize_data(X)
X_pca = apply_pca(X_std)
plot_pca(X_pca, y, target_names)
