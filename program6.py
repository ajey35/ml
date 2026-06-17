import numpy as np
import matplotlib.pyplot as plt

# Generate data
def generate_data():
    np.random.seed(42)
    X = np.linspace(0, 10, 100)
    y = np.sin(X) + np.random.normal(0, 0.2, 100)

    return X.reshape(-1, 1), y

# LWR for one query point
def lwr(X, y, x_query, tau):

    # Compute weights using Gaussian kernel
    weights = np.exp(-(X - x_query)**2 / (2 * tau**2)).flatten()

    # Create diagonal weight matrix
    W = np.diag(weights)

    # Add bias term
    X_b = np.hstack([np.ones_like(X), X])

    # Compute theta using weighted normal equation
    theta = np.linalg.inv(X_b.T @ W @ X_b) @ X_b.T @ W @ y

    # Predict output for query point
    x_query_b = np.array([1, x_query])

    return x_query_b @ theta

# Predict for all points
def predict(X, y, tau):

    y_pred = []

    for x in X:
        y_pred.append(lwr(X, y, x[0], tau))

    return np.array(y_pred)

# Plot graph
def plot_graph(X, y, y_pred, tau):

    plt.figure(figsize=(10, 6))

    plt.scatter(X, y, color='blue',
                label='Data Points')

    plt.plot(X, y_pred, color='red',
             linewidth=2,
             label=f'LWR (tau={tau})')

    plt.title('Locally Weighted Regression')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.legend()
    plt.grid(True)

    plt.show()

# Main
X, y = generate_data()

tau = 0.5

y_pred = predict(X, y, tau)

plot_graph(X, y, y_pred, tau)
