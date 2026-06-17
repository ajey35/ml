import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load dataset
def load_data():
    data = datasets.fetch_olivetti_faces(
        shuffle=True,
        random_state=42
    )

    return data.data, data.target

# Split dataset
def split_data(X, y):
    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

# Train model
def train_model(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

# Predict and evaluate
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc * 100, "%")

    return y_pred

# Display predictions
def show_predictions(X_test, y_pred):
    plt.figure(figsize=(10, 5))

    for i in range(5):
        plt.subplot(1, 5, i + 1)

        plt.imshow(
            X_test[i].reshape(64, 64),
            cmap='gray'
        )

        plt.title(f"Pred: {y_pred[i]}")
        plt.axis('off')

    plt.show()

# Main
X, y = load_data()

X_train, X_test, y_train, y_test = split_data(X, y)

model = train_model(X_train, y_train)

y_pred = evaluate_model(model, X_test, y_test)

show_predictions(X_test, y_pred)
