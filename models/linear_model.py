import numpy as np
from training.activations import softmax
from training.loss import cross_entropy_gradient


class LinearModel:
    def __init__(self):
        self.A = np.random.randn(10, 784) * 0.01 # on prend 0.01 pour avoir des valeurs proches de zéro mais pas nulles, pour que le softmax donne des probabilités proches de 1/10 au début (incertitude uniforme) et que les gradients soient assez forts pour faire avancer l'apprentissage
        self.b = np.zeros(10)

    def forward(self, X):
        # o = Ax + b pour toutes les images du batch
        self.X = X
        self.scores = X @ self.A.T + self.b
        self.P = softmax(self.scores)
        return self.P
    
    def backward(self, y, learning_rate):
        dO = cross_entropy_gradient(self.P, y)  # shape (n, 10)
        dA = dO.T @ self.X                       # shape (10, 784)
        db = dO.sum(axis=0)                      # shape (10,)
        self.A -= learning_rate * dA
        self.b -= learning_rate * db