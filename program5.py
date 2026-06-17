import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Generate dataset
def generate_data():
    np.random.seed(42)
    x = np.random.rand(100)
    return x

# Prepare training and test data
def prepare_data(x):
    y_train = np.where(x[:50] <= 0.5, 1, 2)

    X_train = x[:50].reshape(-1, 1)
    X_test = x[50:].reshape(-1, 1)

    return X_train, y_train, X_test

# Apply KNN for different k values
def apply_knn(X_train, y_train, X_test):
    k_values = [1, 2, 3, 4, 5, 20, 30]

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)

        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)

        print(f"\nk = {k}")
        print("Predicted Classes:", y_pred)

# Main
x = generate_data()

X_train, y_train, X_test = prepare_data(x)

apply_knn(X_train, y_train, X_test)
