import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Activation functions and their derivatives
def normalize(X):
    return X / np.amax(X)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def softmax(z):
    return np.exp(z - np.max(z)) / np.sum(np.exp(z - np.max(z)), axis=1, keepdims=True)

def dsigmoid(z):
    return np.exp(-z) / (1 + np.exp(-z)) ** 2

def dtanh(z):
    return 1 - tanh(z) ** 2

# Functions to initialize weights and biases
def Weights(k, l):
    return np.random.randn(k, l)

def bias(l):
    return np.random.rand(l)

# Function to create the neural network
def create_network(input, n_hidden, n_neurons, output):
    W = dict()
    b = dict()
    for i in range(n_hidden + 1):
        if i == 0:
            W[i] = Weights(input.shape[1], n_neurons[i])
            b[i] = bias(n_neurons[i])
        elif i == n_hidden:
            W[i] = Weights(n_neurons[i - 1], 2)
            b[i] = bias(2)
        else:
            W[i] = Weights(n_neurons[i - 1], n_neurons[i])
            b[i] = bias(n_neurons[i])
    return W, b

# Forward propagation function
def forward(input, W, b):
    for layer in range(len(W)):
        if layer == 0:
            Yth = np.dot(input, W[layer]) + b[layer]
            Yth = sigmoid(Yth)
        elif layer == len(W) - 1:
            Yth = np.dot(Yth, W[layer]) + b[layer]
            Yth = softmax(Yth)
        else:
            Yth = np.dot(Yth, W[layer]) + b[layer]
            Yth = sigmoid(Yth)
    return Yth

# Cost function
def cost(Yth, Yreal):
    J = np.sum((Yth - Yreal) ** 2) / (2 * len(Yreal))
    return J

# Gradient computation function
def gradient(Yth, Yreal, X, W):
    gradient_W = dict()
    gradient_b = dict()
    delta = dict()
    Z = dict()
    a = dict()

    for layer in range(len(W)):
        if layer == 0:
            Z[layer] = np.dot(X, W[layer]) + b[layer]
            a[layer] = sigmoid(Z[layer])
        else:
            Z[layer] = np.dot(a[layer - 1], W[layer]) + b[layer]
            a[layer] = sigmoid(Z[layer])

    for layer in reversed(range(len(W))):
        if layer == len(W) - 1:
            delta[layer] = np.multiply((Yth - Yreal), dsigmoid(Z[layer]))
            gradient_W[layer] = np.dot(a[layer - 1].transpose(), delta[layer])
            gradient_b[layer] = np.sum(delta[layer], axis=0, keepdims=True)
        else:
            if layer == 0:
                delta[layer] = np.dot(delta[layer + 1], W[layer + 1].transpose()) * dsigmoid(Z[layer])
                gradient_W[layer] = np.dot(X.transpose(), delta[layer])
                gradient_b[layer] = np.sum(delta[layer], axis=0, keepdims=True)
            else:
                delta[layer] = np.dot(delta[layer + 1], W[layer + 1].transpose()) * dsigmoid(Z[layer])
                gradient_W[layer] = np.dot(a[layer - 1].transpose(), delta[layer])
                gradient_b[layer] = np.sum(delta[layer], axis=0, keepdims=True)

    return gradient_W, gradient_b, a

# Gradient descent function
def gradient_descent(gradient_W, gradient_b, W, b, learning_rate):
    for layer in range(len(W)):
        W[layer] = W[layer] - learning_rate * gradient_W[layer]
        b[layer] = b[layer] - learning_rate * gradient_b[layer].squeeze()
    return W, b

# Data retrieving
data = pd.read_csv("data.csv", sep=";")

# m: number of examples, p: number of features
m = data.shape[0]
p = data.shape[1]

# Separating inputs and outputs
X = data.iloc[:, 0:p-1]
y = data.iloc[:, p-1]
Yreal = np.zeros((len(y), 2))
for i in range(len(y)):
    if y[i] == 1:
        Yreal[i] = [1, 0]
    elif y[i] == 0:
        Yreal[i] = [0, 1]

# Parameters
n_neurons = [4]
n_hidden = len(n_neurons)

# Normalization of the inputs and network initialization
X = normalize(X)
W, b = create_network(X, n_hidden, n_neurons, y)

# Calculate 1st forward propagation and initial cost
Yth = forward(X, W, b)
J = cost(Yth, Yreal)
learning_rate = 0.0005

# Training the network
for epoch in range(3000):
    Yth = forward(X, W, b)
    J = cost(Yth, Yreal)
    print('Cost is:', J)
    gradient_W, gradient_b, a = gradient(Yth, Yreal, X, W)
    W, b = gradient_descent(gradient_W, gradient_b, W, b, learning_rate)

# Plotting the activation of the first hidden layer neurons
inter = np.reshape(a[0][1], [2, 2])
plt.matshow(inter)
plt.show()

# Prediction on the test set
Yth = forward(X, W, b)
print(Yth)
