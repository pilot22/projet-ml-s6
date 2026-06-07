import numpy as np
from training.activations import softmax
from training.loss import cross_entropy_gradient
from training.trainer import train
from evaluation.metrics import accuracy
from utils.config import EPOCHS, BATCH_SIZE, LEARNING_RATE


class LinearModel:
    def __init__(self):
        self.A = None
        self.b = None

    def _initialize_parameters(self, input_dim):
        self.A = np.random.randn(10, input_dim) * 0.01
        self.b = np.zeros(10)

    def forward(self, X):
        if self.A is None or self.b is None:
            self._initialize_parameters(X.shape[1])

        self.X = X
        self.scores = X @ self.A.T + self.b
        self.P = softmax(self.scores)
        return self.P

    def backward(self, y, learning_rate):
        dO = cross_entropy_gradient(self.P, y)
        dA = dO.T @ self.X
        db = dO.sum(axis=0)
        self.A -= learning_rate * dA
        self.b -= learning_rate * db

    def save(self, path):
        np.savez(path, A=self.A, b=self.b)

    def load(self, path):
        data = np.load(path)
        self.A = data['A']
        self.b = data['b']

    def get_class_heatmaps(self):
        """Retourne les heatmaps d'importance des pixels pour chaque classe (0-9)."""
        heatmaps = {}
        for digit in range(10):
            heatmap = self.A[digit].reshape(28, 28)
            h_min, h_max = heatmap.min(), heatmap.max()
            if h_max > h_min:
                heatmap = (heatmap - h_min) / (h_max - h_min)
            else:
                heatmap = np.zeros_like(heatmap)
            heatmaps[digit] = heatmap
        return heatmaps


def predict_linear(X, A, b):
    return softmax(X @ A.T + b)


def train_linear_model(
    X_train,
    y_train,
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    verbose=True,
):
    model = LinearModel()
    losses = train(
        model, X_train, y_train,
        epochs=epochs, lr=lr, batch_size=batch_size, verbose=verbose,
    )
    return model.A, model.b, losses


def accuracy_linear(X, y, A, b):
    return accuracy(predict_linear(X, A, b), y)
