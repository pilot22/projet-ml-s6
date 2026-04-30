import numpy as np
import os
from sklearn.datasets import fetch_openml

try:
    from utils.config import SEED
except ImportError:
    from config import SEED

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "datasets")
X_PATH = os.path.join(DATA_DIR, "X.npy")
Y_PATH = os.path.join(DATA_DIR, "y.npy")


def load_mnist_data():
    if os.path.exists(X_PATH) and os.path.exists(Y_PATH):
        print("Chargement depuis le cache local...")
        X = np.load(X_PATH).astype(np.float64) / 255.0
        y = np.load(Y_PATH).astype(int)
    else:
        print("Telechargement de MNIST...")

        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
        X_raw, y_raw = mnist.data, mnist.target.astype(int)

        os.makedirs(DATA_DIR, exist_ok=True)
        np.save(X_PATH, X_raw)
        np.save(Y_PATH, y_raw)
        print(f"Sauvegarde dans {DATA_DIR}")

        X = X_raw.astype(np.float64) / 255.0
        y = y_raw.astype(int)

    X_train, y_train = X[:60000], y[:60000]
    X_test, y_test = X[60000:], y[60000:]

    return X_train, y_train, X_test, y_test