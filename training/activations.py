import numpy as np

def heaviside(x):
    return (x >= 0).astype(int)

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative_relu(x):
    return (x > 0).astype(int)

def derivative_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)