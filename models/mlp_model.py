import numpy as np
from training.activations import softmax
from training.loss import cross_entropy_gradient


class MLPModel:
    def __init__(self, hidden_dims):
        self.hidden_dims = hidden_dims
        self.A = []
        self.b = []

        sizes = [784] + self.hidden_dims + [10]

        for i in range(len(sizes) - 1):
            self.b.append(np.zeros(sizes[i+1]))
            self.A.append(np.random.randn(sizes[i+1], sizes[i]) * 0.01)
        
               

    def forward():
        pass       
