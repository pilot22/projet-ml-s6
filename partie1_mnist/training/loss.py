import numpy as np

def to_one_hot(y, num_classes=10):
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

def cross_entropy(P, y):
    Y = to_one_hot(y)
    loss = -np.mean(np.sum(Y * np.log(P + 1e-12), axis=1))
    return loss

def cross_entropy_gradient(P, y):
    n = len(y)
    Y = to_one_hot(y)
    grad = (P - Y) / n
    return grad

