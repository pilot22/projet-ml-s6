import numpy as np
from training.activations import softmax, relu
from training.loss import cross_entropy_gradient
from training.loss import to_one_hot
from training.activations import relu_derivative


class MLPModel:
    def __init__(self, hidden_dims):
        self.hidden_dims = hidden_dims
        self.A = []
        self.b = []

        self.sizes = [784] + self.hidden_dims + [10]

        for i in range(len(self.sizes) - 1):
            self.b.append(np.zeros(self.sizes[i+1]))
            self.A.append(np.random.randn(self.sizes[i+1], self.sizes[i]) * np.sqrt(2 / self.sizes[i]))
        
    
    def forward(self, X):
        self.X = X
        self.z = [X]  # z[0] = entrée, z[1] = sortie couche 1, etc.
        self.o = []   # scores avant activation

        current = X
        for i in range(len(self.sizes) - 2):
            o = current @ self.A[i].T + self.b[i]
            self.o.append(o)
            current = relu(o)
            self.z.append(current)

        # Dernière couche : softmax au lieu de relu
        o = current @ self.A[-1].T + self.b[-1]
        self.o.append(o)
        self.P = softmax(o)

        return self.P
               

    def backward(self, y, learning_rate):
    
        n = len(y)
        Y = to_one_hot(y)

        # Gradient de la dernière couche
        dO = (self.P - Y) / n

        # Stocker les gradients pour mise à jour
        dA = [None] * len(self.A)
        db = [None] * len(self.b)

        # Dernière couche
        dA[-1] = dO.T @ self.z[-1]
        db[-1] = dO.sum(axis=0)

        # Remonter couche par couche
        for i in range(len(self.A) - 2, -1, -1):
            dz = dO @ self.A[i + 1]
            dO = dz * relu_derivative(self.o[i])
            dA[i] = dO.T @ self.z[i]
            db[i] = dO.sum(axis=0)

        # Mise à jour des poids
        for i in range(len(self.A)):
            self.A[i] -= learning_rate * dA[i]
            self.b[i] -= learning_rate * db[i]    
