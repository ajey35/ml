import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
def load_data():
    data = load_breast_cancer()

    X = data.data
    y = data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

# Apply K-Means
def apply_kmeans(X):
    kmeans = KMeans(
        n_clusters=2,
        random_state=42
    )

    kmeans.fit(X)

    return kmeans.cluster_centers_, kmeans.labels_

# Visualize clusters
def show_clusters(X, labels, centers):

    plt.figure(figsize=(8, 6))

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=labels,
        cmap='viridis'
    )

    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        c='red',
        marker='x',
        s=200,
        label='Centroids'
    )

    plt.title("K-Means Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

# Main
X, y = load_data()

centers, labels = apply_kmeans(X)

show_clusters(X, labels, centers)
