import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report


# 1. Load dataset
def load_data():
    data = load_breast_cancer()
    return data.data, data.target, data.feature_names, data.target_names


# 2. Train model
def train_model(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


# 3. Plot decision tree
def visualize_tree(model, feature_names):
    plt.figure(figsize=(12, 8))
    plot_tree(model,
              feature_names=feature_names,
              class_names=["Malignant", "Benign"],
              filled=True)
    plt.title("Decision Tree")
    plt.show()


# 4. Predict new sample
def predict_sample(model, sample):
    sample = np.array(sample).reshape(1, -1)
    return model.predict(sample)


# 5. Main function
def main():
    # Load data
    X, y, feature_names, target_names = load_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    model = train_model(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluation
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nReport:\n", classification_report(y_test, y_pred, target_names=target_names))

    # Visualize tree
    visualize_tree(model, feature_names)

    # Test on one sample
    sample = X_test[0]
    result = predict_sample(model, sample)

    print("\nNew Sample Prediction:", "Benign" if result == 1 else "Malignant")


# Run program
main()