import numpy as np


def accuracy(P, y):
    """Proportion de prédictions correctes."""
    y_pred = np.argmax(P, axis=1)
    return np.mean(y_pred == y)


def error_rate(P, y):
    """Proportion de prédictions incorrectes."""
    return 1 - accuracy(P, y)


def confusion_matrix(P, y, num_classes=10):
    """Matrice de confusion : ligne = vrai label, colonne = prédiction."""
    y_pred = np.argmax(P, axis=1)
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y, y_pred):
        matrix[true, pred] += 1
    return matrix