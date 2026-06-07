import numpy as np
from tensorflow.keras.datasets import cifar10


K1 = np.ones((3, 3)) / 9

K2 = np.array([
  [0, -1, 0],
  [-1, 5, -1],
  [0, -1, 0]
])

K3 = np.array([
  [-1, 2, -1],
  [-1, 2, -1],
  [-1, 2, -1]
])

K4 = np.array([
  [-1, 0, 1],
  [-1, 0, 1],
  [-1, 0, 1]
])

K5 = np.array([
    [-1,  0,  1],
    [-2,  0,  2],
    [-1,  0,  1]
])

K6 = np.array([
    [-2, -1,  0],
    [-1,  1,  1],
    [ 0,  1,  2]
])

def convolution(image, kernels, biases, padding=1):
  H, W, C = image.shape
  num_filters = kernels.shape[0]

  # Padding (pour garder 32x32)
  padded = np.pad(image, ((padding, padding), (padding, padding), (0, 0)))

  # Output
  output = np.zeros((H, W, num_filters))

  # Convolution
  for f in range(num_filters):  # pour chaque filtre
    K = kernels[f]
    b = biases[f]

    for i in range(H):
      for j in range(W):
        output[i, j, f] = np.sum(padded[i:i + 3, j:j + 3, :] * K) + b

  return output

def convolution_grayscale(image, kernel):
    # On transforme l'image RGB en grayscale à la volée
    if image.ndim == 3:
        image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]

    H, W = image.shape
    K = kernel.shape[0]
    pad = K // 2

    # Padding de zéros autour de l'image pour filtrer les bords
    padded = np.zeros((H + 2 * pad, W + 2 * pad))
    padded[pad:pad + H, pad:pad + W] = image

    output = np.zeros((H, W))

    # Convolution
    for i in range(H):
        for j in range(W):
            region = padded[i:i + K, j:j + K]
            output[i, j] = np.sum(region * kernel)

    return output