import os
import numpy as np
from PIL import Image

from utils.config import MNIST_DIR, SEED



def load_mnist_data(test_ratio=0.2):
    images = []
    labels = []

    for filename in sorted(os.listdir(MNIST_DIR)):
        if not filename.endswith(".png"):
            continue

        # Label extrait du filename : mnist_<label>_<index>.png
        label = int(filename.split("-")[1])

        # Chargement en niveaux de gris, aplatissement en vecteur de 784
        img = Image.open(os.path.join(MNIST_DIR, filename)).convert("L")
        pixels = np.array(img, dtype=np.float64).flatten() / 255.0

        images.append(pixels)
        labels.append(label)

    X = np.array(images)
    y = np.array(labels)

    # Shuffle puis split train/test
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(X))
    X, y = X[indices], y[indices]

    split = int(len(X) * (1 - test_ratio))
    return X[:split], y[:split], X[split:], y[split:]


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist_data()
    print(f"Train : {X_train.shape}, Test : {X_test.shape}")
    print(f"Pixels normalisés : [{X_train.min():.1f}, {X_train.max():.1f}]")
    print(f"Labels : {np.unique(y_train)}")