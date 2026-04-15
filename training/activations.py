import numpy as np

def heaviside(x):
    return (x >= 0).astype(int)

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu_derivative(x):
    return (x > 0).astype(int)

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True)) # On fait ça pour éviter les problèmes de débordement numérique
    return e / np.sum(e, axis=-1, keepdims=True) # axis=-1 pour normaliser sur la dernière dimension (sinon il prend le max par colonne, ce qu'on aimerait éviter) et keepdims=True pour garder la même forme de tableau (broadcasting)

if __name__ == "__main__":
    x = np.array([-2, -1, 0, 1, 2])
    print("Heaviside:", heaviside(x))
    print("ReLU:", relu(x))
    print("Sigmoid:", sigmoid(x))
    print("Derivative ReLU:", relu_derivative(x))
    print("Derivative Sigmoid:", sigmoid_derivative(x))
    print("Softmax:", softmax(x))
    print("Somme:", softmax(x).sum())