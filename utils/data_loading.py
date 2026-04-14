import numpy as np
from sklearn.datasets import fetch_openml

try:
    from utils.config import SEED
except ImportError:
    from config import SEED


def load_mnist_data():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data / 255.0, mnist.target.astype(int)

    # Split officiel MNIST : 60k train, 10k test
    X_train, y_train = X[:60000], y[:60000]
    X_test, y_test = X[60000:], y[60000:]

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist_data()
    print(f"Train : {X_train.shape}, Test : {X_test.shape}")
    print(f"Pixels normalisés : [{X_train.min():.1f}, {X_train.max():.1f}]")
    print(f"Labels : {np.unique(y_train)}")