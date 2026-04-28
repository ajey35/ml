import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# -------------------------------
# 1. Linear Regression (Boston)
# -------------------------------
def linear_regression_boston():

    # Load dataset
    data = fetch_openml(name="boston", version=1, as_frame=True)
    X = data.data.to_numpy()
    y = data.target.to_numpy()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluation
    print("\n--- Linear Regression (Boston) ---")
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

    # Plot
    plt.figure()
    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test), max(y_test)],
             [min(y_test), max(y_test)])
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Linear Regression - Boston Housing")
    plt.grid()
    plt.show()


# -------------------------------------
# 2. Polynomial Regression (Auto MPG)
# -------------------------------------
def polynomial_regression_auto_mpg():

    # Load dataset
    data = fetch_openml(name="autoMpg", version=1, as_frame=True)
    df = data.data
    y = data.target.astype(float)

    # Clean data (remove missing horsepower)
    df = df.dropna(subset=["horsepower"])
    y = y.loc[df.index]

    # Feature and target
    X = df[["horsepower"]].astype(float)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create polynomial features
    poly = PolynomialFeatures(degree=3)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Train model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Predict
    y_pred = model.predict(X_test_poly)

    # Evaluation
    print("\n--- Polynomial Regression (Auto MPG) ---")
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

    # Sort values for smooth curve
    sorted_X, sorted_y = zip(*sorted(zip(X_test.values.flatten(), y_test)))
    sorted_X = np.array(sorted_X).reshape(-1, 1)
    smooth_pred = model.predict(poly.transform(sorted_X))

    # Plot
    plt.figure()
    plt.scatter(X_test, y_test, label="Actual")
    plt.plot(sorted_X, smooth_pred, label="Polynomial Fit")
    plt.xlabel("Horsepower")
    plt.ylabel("MPG")
    plt.title("Polynomial Regression - Auto MPG")
    plt.legend()
    plt.grid()
    plt.show()


# -------------------------------
# Run both models
# -------------------------------
def main():
    linear_regression_boston()
    polynomial_regression_auto_mpg()


main()