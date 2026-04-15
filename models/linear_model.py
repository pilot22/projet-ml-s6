import numpy as np

class LinearModel:
    def __init__(self):
        self.A = np.random.randn(10, 784) * 0.01 # on prend 0.01 pour avoir des valeurs proches de zéro mais pas nulles, pour que le softmax donne des probabilités proches de 1/10 au début (incertitude uniforme) et que les gradients soient assez forts pour faire avancer l'apprentissage
        self.b = np.zeros(10)

    def forward(self, X):
        # o = Ax + b pour toutes les images du batch
        self.scores = X @ self.A.T + self.b
        return self.scores