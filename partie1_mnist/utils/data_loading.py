import numpy as np
import os
from PIL import Image
from sklearn.datasets import fetch_openml

try:
    from utils.config import SEED
except ImportError:
    from config import SEED

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "datasets")
X_PATH = os.path.join(DATA_DIR, "X.npy")
Y_PATH = os.path.join(DATA_DIR, "y.npy")

ROTATION_AUGMENT_MIN = -30
ROTATION_AUGMENT_MAX = 30


def _rotate_flat_image(img_flat, angle):
    """Rotation d'une image MNIST aplatie (28x28), angle en degrés (sens trigo)."""
    img = (img_flat.reshape(28, 28) * 255).astype(np.uint8)
    rotated = Image.fromarray(img).rotate(
        angle, resample=Image.BILINEAR, fillcolor=0, expand=False
    )
    return np.array(rotated, dtype=np.float64).flatten() / 255.0


def augment_with_rotations(X, y, min_angle=ROTATION_AUGMENT_MIN, max_angle=ROTATION_AUGMENT_MAX):
    """Duplique le jeu de données avec une rotation aleatoire uniforme par image."""
    rng = np.random.default_rng(SEED)
    angles = rng.uniform(min_angle, max_angle, size=len(X))
    print(
        f"  Augmentation : rotation aleatoire uniforme "
        f"[{min_angle:+d}°, {max_angle:+d}°] sur {len(X)} images..."
    )
    X_rot = np.array([_rotate_flat_image(x, angle) for x, angle in zip(X, angles)])
    return np.vstack([X, X_rot]), np.concatenate([y, y])


def load_mnist_data(augment=True):
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

    if augment:
        print("Augmentation par rotations...")
        X_train, y_train = augment_with_rotations(X_train, y_train)

    return X_train, y_train, X_test, y_test