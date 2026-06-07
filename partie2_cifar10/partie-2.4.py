from convolution.filters import K1, K2, K3, K4, K5, K6, convolution_grayscale
from convolution.cnn import entrainer_cnn
from utils.data_loading import load_cifar_data
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch

X_train, y_train, X_test, y_test = load_cifar_data()

X_train = torch.tensor(
    np.transpose(X_train, (0, 3, 1, 2)),
    dtype=torch.float32
)

X_test = torch.tensor(
    np.transpose(X_test, (0, 3, 1, 2)),
    dtype=torch.float32
)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

entrainer_cnn(
    X_train, y_train, X_test, y_test,
    save_loss_path="loss_cnn_cifar.png",
)