import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw


class DrawPredict:
    def __init__(self, models):
        self.models = models  # dict {"nom": model}
        self.window = tk.Tk()
        self.window.title("Prediction de chiffre")

        # Canvas pour dessiner (280x280, on redimensionnera en 28x28)
        self.canvas = tk.Canvas(self.window, width=280, height=280, bg="black")
        self.canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.draw)

        # Image interne pour capturer le dessin
        self.image = Image.new("L", (280, 280), 0)
        self.image_draw = ImageDraw.Draw(self.image)

        # Sélection du modèle
        self.selected_model = tk.StringVar(value=list(models.keys())[0])
        for i, name in enumerate(models.keys()):
            tk.Radiobutton(self.window, text=name, variable=self.selected_model, value=name).grid(row=1, column=i)

        # Boutons
        tk.Button(self.window, text="Predire", command=self.predict).grid(row=2, column=0, pady=5)
        tk.Button(self.window, text="Comparer tous", command=self.compare_all).grid(row=2, column=1, pady=5)
        tk.Button(self.window, text="Effacer", command=self.clear).grid(row=2, column=2, pady=5)

        # Zone de résultats
        self.result_label = tk.Label(self.window, text="Dessine un chiffre", font=("Arial", 14))
        self.result_label.grid(row=3, column=0, columnspan=3, pady=10)

        self.window.mainloop()

    def draw(self, event):
        x, y = event.x, event.y
        r = 8  # épaisseur du trait
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.image_draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 0)
        self.image_draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Dessine un chiffre")

    def preprocess(self):
        img = np.array(self.image)

        rows = np.any(img > 50, axis=1)
        cols = np.any(img > 50, axis=0)
        if not rows.any() or not cols.any():
            return None
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        img = img[rmin:rmax + 1, cmin:cmax + 1]

        # Rendre l'image carrée en ajoutant du padding
        h, w = img.shape
        size = max(h, w)
        square = np.zeros((size, size), dtype=np.uint8)
        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        square[y_offset:y_offset + h, x_offset:x_offset + w] = img

        # Resize 20x20 + padding 4px pour faire 28x28
        square = Image.fromarray(square).resize((20, 20))
        result = Image.new("L", (28, 28), 0)
        result.paste(square, (4, 4))

        x = np.array(result, dtype=np.float64).flatten() / 255.0
        return x.reshape(1, 784)

    def predict(self):
        x = self.preprocess()
        if x is None:
            self.result_label.config(text="Dessine quelque chose d'abord !")
            return

        name = self.selected_model.get()
        model = self.models[name]
        P = model.forward(x)
        pred = np.argmax(P, axis=1)[0]
        conf = P[0, pred] * 100

        self.result_label.config(text=f"{name} : {pred} (confiance : {conf:.1f}%)")

    def compare_all(self):
        x = self.preprocess()
        if x is None:
            self.result_label.config(text="Dessine quelque chose d'abord !")
            return

        results = []
        for name, model in self.models.items():
            P = model.forward(x)
            pred = np.argmax(P, axis=1)[0]
            conf = P[0, pred] * 100
            results.append(f"{name} : {pred} ({conf:.1f}%)")

        self.result_label.config(text="\n".join(results))