import numpy as np
from PIL import Image

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