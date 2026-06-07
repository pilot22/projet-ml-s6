import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "partie1_mnist"))

from utils.data_loading import load_cifar_data, prepare_cifar_features
from models.linear_model import train_linear_model, accuracy_linear
from models.mlp_model import train_mlp, accuracy_mlp
from utils.config import HIDDEN_DIMS_H1, HIDDEN_DIMS_H2, EPOCHS, LEARNING_RATE

X_train, y_train, X_test, y_test = load_cifar_data()
print(f"Train : {X_train.shape}, Test : {X_test.shape}, Classes : {np.unique(y_train)}")

X_gray_train, X_gray_test, X_rgb_train, X_rgb_test = prepare_cifar_features(X_train, X_test)

print("\n=== Modèle linéaire ===")
print("Grayscale")
A_lin_gray, b_lin_gray, _ = train_linear_model(X_gray_train, y_train, epochs=EPOCHS, lr=LEARNING_RATE)
print("RGB")
A_lin_rgb, b_lin_rgb, _ = train_linear_model(X_rgb_train, y_train, epochs=EPOCHS, lr=LEARNING_RATE)

acc_gray_train = accuracy_linear(X_gray_train, y_train, A_lin_gray, b_lin_gray)
acc_rgb_train = accuracy_linear(X_rgb_train, y_train, A_lin_rgb, b_lin_rgb)

acc_gray = accuracy_linear(X_gray_test, y_test, A_lin_gray, b_lin_gray)
acc_rgb = accuracy_linear(X_rgb_test, y_test, A_lin_rgb, b_lin_rgb)

print(f"Accuracy train (gray): {acc_gray_train:.4f}")
print(f"Accuracy train (RGB) : {acc_rgb_train:.4f}")
print(f"Accuracy test (gray) : {acc_gray:.4f}")
print(f"Accuracy test (RGB)  : {acc_rgb:.4f}")

print(f"\n=== MLP H=1 {HIDDEN_DIMS_H1} ===")
print("Grayscale")
A_mlp1_gray, b_mlp1_gray, _ = train_mlp(X_gray_train, y_train, HIDDEN_DIMS_H1, epochs=EPOCHS, lr=LEARNING_RATE)
print("RGB")
A_mlp1_rgb, b_mlp1_rgb, _ = train_mlp(X_rgb_train, y_train, HIDDEN_DIMS_H1, epochs=EPOCHS, lr=LEARNING_RATE)

acc_gray_train = accuracy_mlp(X_gray_train, y_train, A_mlp1_gray, b_mlp1_gray)
acc_rgb_train = accuracy_mlp(X_rgb_train, y_train, A_mlp1_rgb, b_mlp1_rgb)

acc_gray = accuracy_mlp(X_gray_test, y_test, A_mlp1_gray, b_mlp1_gray)
acc_rgb = accuracy_mlp(X_rgb_test, y_test, A_mlp1_rgb, b_mlp1_rgb)

print(f"Accuracy train (gray): {acc_gray_train:.4f}")
print(f"Accuracy train (RGB) : {acc_rgb_train:.4f}")
print(f"Accuracy test (gray) : {acc_gray:.4f}")
print(f"Accuracy test (RGB)  : {acc_rgb:.4f}")

print(f"\n=== MLP H=2 {HIDDEN_DIMS_H2} ===")
print("Grayscale")
A_mlp2_gray, b_mlp2_gray, _ = train_mlp(X_gray_train, y_train, HIDDEN_DIMS_H2, epochs=EPOCHS, lr=LEARNING_RATE)
print("RGB")
A_mlp2_rgb, b_mlp2_rgb, _ = train_mlp(X_rgb_train, y_train, HIDDEN_DIMS_H2, epochs=EPOCHS, lr=LEARNING_RATE)

acc_gray_train = accuracy_mlp(X_gray_train, y_train, A_mlp2_gray, b_mlp2_gray)
acc_rgb_train = accuracy_mlp(X_rgb_train, y_train, A_mlp2_rgb, b_mlp2_rgb)

acc_gray = accuracy_mlp(X_gray_test, y_test, A_mlp2_gray, b_mlp2_gray)
acc_rgb = accuracy_mlp(X_rgb_test, y_test, A_mlp2_rgb, b_mlp2_rgb)

print(f"Accuracy train (gray): {acc_gray_train:.4f}")
print(f"Accuracy train (RGB) : {acc_rgb_train:.4f}")
print(f"Accuracy test (gray) : {acc_gray:.4f}")
print(f"Accuracy test (RGB)  : {acc_rgb:.4f}")