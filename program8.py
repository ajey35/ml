import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
def load_data():
    data = load_breast_cancer()
    return data.data, data.target, data.feature_names, data.target_names

# Train model
def train_model(X_train, y_train):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf

# Plot tree
def show_tree(clf, features):
    plt.figure(figsize=(12, 8))

    plot_tree(clf,
              feature_names=features,
              class_names=["Malignant", "Benign"],
              filled=True)

    plt.show()

# Predict new sample
def predict_sample(clf, sample):
    sample = np.array(sample).reshape(1, -1)
    return clf.predict(sample)

# Main
X, y, features, targets = load_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = train_model(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                            target_names=targets))

show_tree(clf, features)

sample = X_test[0]
prediction = predict_sample(clf, sample)

print("\nPrediction:",
      "Benign" if prediction[0] == 1 else "Malignant")
