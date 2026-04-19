# Projet ML S6 : Classification de chiffres manuscrits (MNIST)
## Partie 1

Projet de Mathématiques pour le Machine Learning - EFREI Paris, S6 2025-2026.

### Structure du projet

```bash
.
├── utils/
│   ├── config.py            # Hyperparamètres et constantes
│   └── data_loading.py      # Chargement et normalisation MNIST
├── models/
│   ├── linear_model.py      # Modèle linéaire multi-classe
│   └── mlp_model.py         # Perceptron multi-couches
├── training/
│   ├── activations.py       # Softmax, ReLU, sigmoid et dérivées
│   ├── loss.py              # Cross-entropy et gradient
│   └── trainer.py           # Boucle d'entraînement
├── evaluation/
│   ├── metrics.py           # Accuracy, taux d'erreur, matrice de confusion
│   └── visualization.py     # Prédiction sur image personnelle
├── main.py                  # Point d'entrée
├── requirements.txt
└── README.md
```

### Dataset

Le dataset MNIST (70 000 images 28×28 en niveaux de gris) est chargé via `sklearn.datasets.fetch_openml`.
Ce choix permet d'obtenir directement les images en vecteurs de 784 pixels avec les labels associés et le split officiel 60k train / 10k test, sans prétraitement manuel de fichiers PNG (ce qui n'est pas le but du projet).
Le téléchargement se fait une seule fois, sklearn met les données en cache ensuite.

### Visualiser une image du dataset

On peut visualiser une image du dataset avec ce code. N'importe où dans le code, il suffit d'avoir un vecteur de 784 valeurs.

```python
import matplotlib.pyplot as plt
from utils.data_loading import load_mnist_data

X_train, y_train, X_test, y_test = load_mnist_data()

index = 42  # changer l'index pour voir une autre image
img = X_train[index].reshape(28, 28)
plt.imshow(img, cmap='gray')
plt.title(f"Label : {y_train[index]}")
plt.show()
```

### Installation

Python 3.12 requis.

```bash
py -3.12 -m pip install -r requirements.txt
```

### Lancement

```bash
py -3.12 main.py
```

### Résultats (modèle linéaire)

| Métrique | Train | Test |
|---|---|---|
| Taux d'erreur | 9.69% | 9.09% |

Config : `lr=0.01`, `batch_size=64`, `epochs=20`.