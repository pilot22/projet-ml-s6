# Projet ML S6 : Classification de chiffres manuscrits (MNIST)
## Partie 1

Projet de Mathématiques pour le Machine Learning - EFREI Paris, S6 2025-2026.

### Structure du projet

```bash
.
├── partie1_mnist/
│   ├── utils/
│   │   ├── config.py            # Hyperparamètres et constantes
│   │   └── data_loading.py      # Chargement et normalisation MNIST
│   ├── models/
│   │   ├── linear_model.py      # Modèle linéaire multi-classe
│   │   └── mlp_model.py         # Perceptron multi-couches
│   ├── training/
│   │   ├── activations.py       # Softmax, ReLU, sigmoid et dérivées
│   │   ├── loss.py              # Cross-entropy et gradient
│   │   └── trainer.py           # Boucle d'entraînement
│   ├── evaluation/
│   │   ├── metrics.py           # Accuracy, taux d'erreur, matrice de confusion
│   │   └── visualization.py     # Courbes, matrice de confusion, projection 2D
│   └── main.py                  # Point d'entrée avec menu interactif
├── partie2_cifar10/
│   └── main.py
├── partie3_mammographies/
│   └── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

### Dataset

Le dataset MNIST (70 000 images 28×28 en niveaux de gris) est chargé via `sklearn.datasets.fetch_openml`

Le premier lancement télécharge automatiquement le dataset (~50 Mo) et le sauvegarde dans `datasets/` au format `.npy`. Les lancements suivants utilisent ce cache local, sans connexion réseau.

### Visualiser une image du dataset

N'importe où dans le code, il suffit d'avoir un vecteur de 784 valeurs :

```python
import matplotlib.pyplot as plt
from utils.data_loading import load_mnist_data

X_train, y_train, X_test, y_test = load_mnist_data()

index = 42
img = X_train[index].reshape(28, 28)
plt.imshow(img, cmap='gray')
plt.title(f"Label : {y_train[index]}")
plt.show()
```

### Installation

Python 3.11 ou 3.12 requis.

```bash
pip install -r requirements.txt
```

### Lancement

```bash
cd partie1_mnist
python main.py
```

Un menu interactif permet de choisir le modèle à entraîner et les visualisations à afficher.

### Résultats

| Modèle | Erreur train (10 ep.) | Erreur test (10 ep.) | Erreur train (30 ep.) | Erreur test (30 ep.) |
|---|---|---|---|---|
| Linéaire | 9.67% | 9.07% | 8.35% | 8.17% |
| MLP (H=1, [128]) | 5.91% | 6.00% | 3.40% | 3.90% |
| MLP (H=2, [128, 64]) | 4.38% | 4.60% | 1.89% | 2.81% |

Config : `lr=0.01`, `batch_size=64`, initialisation de He pour le MLP.