import numpy as np
from tensorflow.keras.datasets import cifar10

def load_cifar_data(normalize=True):
    """
    Charge CIFAR-10 depuis le cache Keras.
    """

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    if normalize:
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0

    return X_train, y_train, X_test, y_test


def convert_to_grayscale(X):
    """
    Convertit des images RGB CIFAR-10 en niveaux de gris.
    """

    return (
        0.299 * X[:, :, :, 0]
        + 0.587 * X[:, :, :, 1]
        + 0.114 * X[:, :, :, 2]
    )

def flatten_images(X):
    """Aplatit les images en matrice (n_samples, n_features)."""
    return X.reshape(len(X), -1)


def prepare_cifar_features(X_train, X_test):
    """
    Prépare les variantes grayscale et RGB aplaties pour l'entraînement.
    Retourne (X_gray_train, X_gray_test, X_rgb_train, X_rgb_test).
    """
    X_gray_train = flatten_images(convert_to_grayscale(X_train))
    X_gray_test = flatten_images(convert_to_grayscale(X_test))
    X_rgb_train = flatten_images(X_train)
    X_rgb_test = flatten_images(X_test)
    return X_gray_train, X_gray_test, X_rgb_train, X_rgb_test