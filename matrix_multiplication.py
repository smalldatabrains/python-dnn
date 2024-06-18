import numpy as np

# Define the input, weights, and bias
x = np.array([1, 2, 3])  # Input vector (1D array)
W = np.array([[0.2, 0.8, -0.5], [0.5, 0.1, 0.2]])  # Weight matrix (2x3)
b = np.array([0.1, -0.1])  # Bias vector (1D array)

# Compute the forward propagation using np.dot
z = np.dot(W, x) + b
print("Forward Propagation Output using np.dot:")
print(z)


# Compute the forward propagation using np.matmul or @
z = np.matmul(W, x) + b  # Equivalent to np.matmul(W, x) + b
print("Forward Propagation Output using np.matmul or @:")
print(z)
