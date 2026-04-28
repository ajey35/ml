import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create dataset
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = np.sin(X) + np.random.normal(0, 0.2, 100)

# Convert shape
X = X.reshape(-1, 1)

# Step 2: LWR function
def lwr(X, y, x_query, tau):
    # Compute weights
    weights = np.exp(-(X - x_query)**2 / (2 * tau**2)).flatten()
    
    # Create diagonal weight matrix
    W = np.diag(weights)
    
    # Add bias term
    X_b = np.hstack([np.ones_like(X), X])
    
    # Compute theta
    theta = np.linalg.inv(X_b.T @ W @ X_b) @ X_b.T @ W @ y
    
    # Prediction
    x_query_b = np.array([1, x_query])
    return x_query_b @ theta

# Step 3: Predict all points
def predict(X, y, tau):
    return np.array([lwr(X, y, x, tau) for x in X.flatten()])

# Step 4: Run model
tau = 0.5
y_pred = predict(X, y, tau)

# Step 5: Plot
plt.scatter(X, y, s=10, label="Data")
plt.plot(X, y_pred, linewidth=2, label="LWR")
plt.legend()
plt.show()