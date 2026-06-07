import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from PIL import Image


class DrawPredict:
    CANVAS_SIZE = 280
    BRUSH_RADIUS = 8

    def __init__(self, models):
        self.models = models
        self.model_names = list(models.keys())
        self.selected_model = self.model_names[0]
        self.drawing = False
        self.image = np.zeros((self.CANVAS_SIZE, self.CANVAS_SIZE), dtype=np.uint8)

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(5, 7))
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.42)
        self.ax.set_title("Prediction de chiffre")
        self.ax.axis("off")
        self.im = self.ax.imshow(self.image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        rax = self.fig.add_axes([0.1, 0.28, 0.8, 0.1])
        self.radio = RadioButtons(rax, self.model_names)
        self.radio.on_clicked(self._on_model_select)

        ax_predict = self.fig.add_axes([0.1, 0.16, 0.22, 0.08])
        ax_compare = self.fig.add_axes([0.39, 0.16, 0.22, 0.08])
        ax_clear = self.fig.add_axes([0.68, 0.16, 0.22, 0.08])

        # Garder des references aux widgets (sinon garbage-collectes -> boutons morts).
        self.btn_predict = Button(ax_predict, "Predire")
        self.btn_compare = Button(ax_compare, "Comparer")
        self.btn_clear = Button(ax_clear, "Effacer")
        self.btn_predict.on_clicked(self.predict)
        self.btn_compare.on_clicked(self.compare_all)
        self.btn_clear.on_clicked(self.clear)

        self.widget_axes = {rax, ax_predict, ax_compare, ax_clear}
        self.result_text = self.fig.text(
            0.5, 0.06, "Dessine un chiffre (P=predire, C=comparer, E=effacer)",
            ha="center", fontsize=10,
        )
        plt.show(block=True)

    def _on_model_select(self, label):
        self.selected_model = label

    def _on_key(self, event):
        if event.key in ("p", "P"):
            self.predict()
        elif event.key in ("c", "C"):
            self.compare_all()
        elif event.key in ("e", "E"):
            self.clear()

    def _on_press(self, event):
        if event.inaxes in self.widget_axes or event.inaxes != self.ax or event.button != 1:
            return
        self.drawing = True
        self._draw_at(event.xdata, event.ydata)

    def _on_release(self, event):
        self.drawing = False

    def _on_motion(self, event):
        if self.drawing and event.inaxes == self.ax and event.xdata is not None:
            self._draw_at(event.xdata, event.ydata)

    def _draw_at(self, x, y):
        xi, yi = int(x), int(y)
        r = self.BRUSH_RADIUS
        y0, y1 = max(0, yi - r), min(self.CANVAS_SIZE, yi + r + 1)
        x0, x1 = max(0, xi - r), min(self.CANVAS_SIZE, xi + r + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - xi) ** 2 + (yy - yi) ** 2 <= r ** 2
        self.image[y0:y1, x0:x1][mask] = 255
        self.im.set_data(self.image)
        self.fig.canvas.draw_idle()

    def clear(self, event=None):
        self.image.fill(0)
        self.im.set_data(self.image)
        self.result_text.set_text("Dessine un chiffre (P=predire, C=comparer, E=effacer)")
        self.fig.canvas.draw_idle()

    def preprocess(self):
        img = self.image.copy()

        rows = np.any(img > 50, axis=1)
        cols = np.any(img > 50, axis=0)
        if not rows.any() or not cols.any():
            return None
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        img = img[rmin : rmax + 1, cmin : cmax + 1]

        h, w = img.shape
        size = max(h, w)
        square = np.zeros((size, size), dtype=np.uint8)
        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        square[y_offset : y_offset + h, x_offset : x_offset + w] = img

        square = Image.fromarray(square).resize((20, 20))
        result = Image.new("L", (28, 28), 0)
        result.paste(square, (4, 4))

        x = np.array(result, dtype=np.float64).flatten() / 255.0
        return x.reshape(1, 784)

    def predict(self, event=None):
        x = self.preprocess()
        if x is None:
            self.result_text.set_text("Dessine quelque chose d'abord !")
            self.fig.canvas.draw_idle()
            return

        model = self.models[self.selected_model]
        P = model.forward(x)
        pred = np.argmax(P, axis=1)[0]
        conf = P[0, pred] * 100
        self.result_text.set_text(f"{self.selected_model} : {pred} (confiance : {conf:.1f}%)")
        self.fig.canvas.draw_idle()

    def compare_all(self, event=None):
        x = self.preprocess()
        if x is None:
            self.result_text.set_text("Dessine quelque chose d'abord !")
            self.fig.canvas.draw_idle()
            return

        results = []
        for name, model in self.models.items():
            P = model.forward(x)
            pred = np.argmax(P, axis=1)[0]
            conf = P[0, pred] * 100
            results.append(f"{name} : {pred} ({conf:.1f}%)")

        self.result_text.set_text("\n".join(results))
        self.fig.canvas.draw_idle()
