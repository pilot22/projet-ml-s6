import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def predict_image(model, image_path):
    img = Image.open(image_path).convert("L").resize((28, 28))
    x = np.array(img, dtype=np.float64).flatten() / 255.0
    x = 1.0 - x  # inversion : chiffre blanc sur fond noir comme MNIST
    x = x.reshape(1, 784)

    P = model.forward(x)
    prediction = np.argmax(P, axis=1)[0]
    confidence = P[0, prediction] * 100

    print(f"Prediction : {prediction} (confiance : {confidence:.1f}%)")
    return prediction


def plot_loss_curves(histories, labels):
    plt.figure(figsize=(10, 6))
    for history, label in zip(histories, labels):
        plt.plot(range(1, len(history) + 1), history, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Courbes de loss par epoch")
    plt.legend()
    plt.grid(True)
    plt.show()