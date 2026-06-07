import numpy as np
from training.activations import softmax, relu
from training.loss import cross_entropy_gradient
from training.loss import to_one_hot
from training.activations import relu_derivative
from training.trainer import train
from evaluation.metrics import accuracy
from utils.config import EPOCHS, BATCH_SIZE, LEARNING_RATE


class MLPModel:
    def __init__(self, hidden_dims):
        self.hidden_dims = hidden_dims
        self.A = []
        self.b = []
        self.sizes = None
        self._initialized = False

    def _initialize_parameters(self, input_dim):
        self.sizes = [input_dim] + self.hidden_dims + [10]

        for i in range(len(self.sizes) - 1):
            self.b.append(np.zeros(self.sizes[i + 1]))
            self.A.append(
                np.random.randn(self.sizes[i + 1], self.sizes[i]) * np.sqrt(2 / self.sizes[i])
            )

        self._initialized = True

    def forward(self, X):
        if not self._initialized:
            self._initialize_parameters(X.shape[1])

        self.X = X
        self.z = [X]
        self.o = []

        current = X
        for i in range(len(self.sizes) - 2):
            o = current @ self.A[i].T + self.b[i]
            self.o.append(o)
            current = relu(o)
            self.z.append(current)

        o = current @ self.A[-1].T + self.b[-1]
        self.o.append(o)
        self.P = softmax(o)

        return self.P

    def backward(self, y, learning_rate):
        n = len(y)
        Y = to_one_hot(y)

        dO = (self.P - Y) / n

        dA = [None] * len(self.A)
        db = [None] * len(self.b)

        dA[-1] = dO.T @ self.z[-1]
        db[-1] = dO.sum(axis=0)

        for i in range(len(self.A) - 2, -1, -1):
            dz = dO @ self.A[i + 1]
            dO = dz * relu_derivative(self.o[i])
            dA[i] = dO.T @ self.z[i]
            db[i] = dO.sum(axis=0)

        for i in range(len(self.A)):
            self.A[i] -= learning_rate * dA[i]
            self.b[i] -= learning_rate * db[i]

    def save(self, path):
        data = {}
        for i in range(len(self.A)):
            data[f'A_{i}'] = self.A[i]
            data[f'b_{i}'] = self.b[i]
        np.savez(path, **data)

    def load(self, path):
        data = np.load(path)
        self.A = []
        self.b = []
        i = 0
        while f'A_{i}' in data:
            self.A.append(data[f'A_{i}'])
            self.b.append(data[f'b_{i}'])
            i += 1


def predict_mlp(X, A, b):
    current = X
    for i in range(len(A) - 1):
        current = relu(current @ A[i].T + b[i])
    return softmax(current @ A[-1].T + b[-1])


def train_mlp(
    X_train,
    y_train,
    hidden_dims,
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    verbose=True,
):
    model = MLPModel(hidden_dims)
    losses = train(
        model, X_train, y_train,
        epochs=epochs, lr=lr, batch_size=batch_size, verbose=verbose,
    )
    return model.A, model.b, losses


def accuracy_mlp(X, y, A, b):
    return accuracy(predict_mlp(X, A, b), y)
