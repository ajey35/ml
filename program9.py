import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# 1. Load dataset
def load_data():
    data = fetch_olivetti_faces(shuffle=True, random_state=42)
    return data.data, data.target


# 2. Split data
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


# 3. Train model
def train_model(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


# 4. Predict and evaluate
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return y_pred, acc


# 5. Visualize predictions
def show_predictions(X_test, y_pred, n=5):
    plt.figure(figsize=(10, 4))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(X_test[i].reshape(64, 64), cmap='gray')
        plt.title(f"Pred: {y_pred[i]}")
        plt.axis("off")
    plt.show()


# 6. Main function
def main():
    # Load data
    X, y = load_data()

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train
    model = train_model(X_train, y_train)

    # Evaluate
    y_pred, acc = evaluate_model(model, X_test, y_test)
    print("Accuracy:", acc * 100, "%")

    # Show results
    show_predictions(X_test, y_pred)


# Run program
main()