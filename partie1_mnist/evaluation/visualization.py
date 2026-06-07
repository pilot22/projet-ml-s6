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

import numpy as np
import matplotlib.pyplot as plt


def plot_misclassified(X, y, P, n=16):
    y_pred = np.argmax(P, axis=1)
    wrong = np.where(y_pred != y)[0]

    # On prend n exemples au hasard parmi les erreurs
    indices = np.random.choice(wrong, size=min(n, len(wrong)), replace=False)

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))

    for i, ax in enumerate(axes.flat):
        if i < len(indices):
            idx = indices[i]
            ax.imshow(X[idx].reshape(28, 28), cmap='gray')
            ax.set_title(f"Vrai: {y[idx]}  Predit: {y_pred[idx]}", color='red')
        ax.axis('off')

    plt.suptitle("Exemples de chiffres mal classes", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_digit_heatmaps(model, model_name="Modèle linéaire"):
    """Affiche les heatmaps d'importance des pixels par classe (0-9)."""
    heatmaps = model.get_class_heatmaps()

    stacked = np.stack([heatmaps[d] for d in range(10)], axis=0)
    if np.min(stacked) < 0:
        vmax = np.percentile(np.abs(stacked), 99)
        vmin = -vmax
        cmap = "seismic"
    else:
        vmin = np.percentile(stacked, 1)
        vmax = np.percentile(stacked, 99)
        cmap = "hot"

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    axes = axes.flatten()

    for digit in range(10):
        ax = axes[digit]
        im = ax.imshow(heatmaps[digit], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Digit {digit}", fontsize=12)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{model_name} - Importance des pixels par classe", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(X, y, P):
    from evaluation.metrics import confusion_matrix

    y_pred = np.argmax(P, axis=1)
    cm = confusion_matrix(P, y)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues')

    # Afficher les nombres dans chaque case
    for i in range(10):
        for j in range(10):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=10)

    ax.set_xlabel("Prediction")
    ax.set_ylabel("Vrai label")
    ax.set_title("Matrice de confusion")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()

def plot_2d_projection(model, X, y, method='pca', source='activations'):
    """Projection 2D des données ou des activations internes du réseau.

    source='activations' : dernière couche cachée (128D ou 64D) → bassin appris
    source='input'       : pixels bruts (784D) → structure des données MNIST
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    model.forward(X)

    if source == 'activations':
        features = model.z[-1]
        feature_label = f"activations couche cachée ({features.shape[1]}D)"
    else:
        features = X
        feature_label = f"pixels d'entrée ({features.shape[1]}D)"

    if method == 'pca':
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(features)
        var = reducer.explained_variance_ratio_.sum() * 100
        title = f"PCA — {feature_label} → 2D (variance expliquée : {var:.1f}%)"
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        coords = reducer.fit_transform(features)
        title = f"t-SNE — {feature_label} → 2D"

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for digit in range(10):
        mask = y == digit
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[colors[digit]], label=str(digit), alpha=0.5, s=5)

    ax.set_title(title)
    ax.legend(title="Classe", markerscale=5)
    ax.set_xlabel("Composante 1")
    ax.set_ylabel("Composante 2")
    plt.tight_layout()
    plt.show()