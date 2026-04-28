import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# 1. Load and preprocess data
def load_data():
    data = load_breast_cancer()
    X = data.data

    # Scale features (important for K-Means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled


# 2. Apply K-Means
def run_kmeans(X, k=2):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X)
    return model.cluster_centers_, model.labels_


# 3. Visualize clusters
def plot_clusters(X, labels, centers):
    plt.figure()

    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=labels)

    # Plot centroids
    plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=200)

    plt.title("K-Means Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


# 4. Main function
def main():
    X = load_data()

    centers, labels = run_kmeans(X, k=2)

    plot_clusters(X, labels, centers)


# Run program
main()