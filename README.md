# De la classification de chiffres manuscrits à la détection de cancers du sein
**Projet SM604 — Mathématiques pour le Machine Learning · EFREI Paris S6 2025-2026**

Implémentation from scratch de modèles de classification d'images (régression logistique, MLP, CNN) appliqués successivement à MNIST, CIFAR-10 et des mammographies médicales (CBIS-DDSM). La rétropropagation des parties 1 et 2 est entièrement manuelle (sans autograd).

---

## Installation

```bash
pip install -r requirements.txt  # Python 3.11 / 3.12
```

---

## Partie 1 — MNIST

**Dataset** : 70 000 images 28×28 px, 60k train / 10k test, vecteur $\vec{x} \in \mathbb{R}^{784}$.

```bash
cd partie1_mnist && python main.py
```

**Modèles :**

- **Linéaire** : $o = A\vec{x} + b$, softmax + cross-entropy, SGD mini-batch
- **MLP** : couches cachées ReLU, init. de He, backprop from scratch
  - `H1 = [128]` · `H2 = [128, 64]`

**Hyperparamètres** : `lr=0.01` · `batch=64` · `epochs=30` · `seed=42`

**Résultats :**

| Modèle | Erreur train | Erreur test |
|---|---|---|
| Linéaire | 8.35% | 8.17% |
| MLP H1 [128] | 3.40% | 3.90% |
| MLP H2 [128, 64] | 1.89% | 2.81% |

---

## Partie 2 — CIFAR-10

**Dataset** : 60 000 images couleur 32×32 px, 10 classes.

### 2.2 — Modèles linéaire et MLP

```bash
cd partie2_cifar10 && python partie-2.2.py
```

Deux modes d'entrée : **grayscale** ($\vec{x} \in \mathbb{R}^{1024}$) et **RGB** ($\vec{x} \in \mathbb{R}^{3072}$).

| Modèle | Accuracy (gray) | Accuracy (RGB) |
|---|---|---|
| Linéaire | 28.87% | 38.85% |
| MLP H1 [128] | 39.89% | 48.17% |
| MLP H2 [128, 64] | 39.19% | 49.51% |

### 2.3 — Filtres de convolution (from scratch)

```bash
python partie-2.3.py
```

Convolution 2D NumPy avec zero-padding. 6 filtres appliqués sur `photo_chat.png` → `outputs/f1-f6.png` : moyenneur, Laplacien, Sobel vertical/horizontal, gradient horizontal, emboss diagonal.

### 2.4 — CNN PyTorch

```bash
python partie-2.4.py
```

Architecture : `Conv(3→64) → Conv(64→64) → MaxPool → Conv(64→64) → MaxPool → Conv(64→64) → Flatten → FC(4096→10)`

Options : BatchNorm, Dropout, Data Augmentation, Early Stopping, class weights. Optimiseur : Adam.

---

## Partie 3 — Mammographies (CBIS-DDSM)

**Dataset** : [CBIS-DDSM sur Kaggle](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) — classification binaire bénin (0) / malin (1). Images redimensionnées en 128×128 px.

```bash
cd partie3_mammographies && python partie-3.py
```

Structure attendue :
```
datasets/cbis-ddsm/
├── csv/   # mass_case_description_*.csv + dicom_info.csv
└── jpeg/
```

**Config CNN** : `in_channels=1` · `filters=(16,32,32,32)` · `BN=True` · `dropout=0.5` · `augment=True` · `threshold=0.35` · `epochs=50` · `lr=0.001` · `patience=10`

Le déséquilibre de classes est compensé par `class_weights_from_labels`. L'évaluation porte particulièrement sur la **sensibilité** et la **spécificité** pour minimiser les faux négatifs.

---

## Stack

`numpy 2.4.3` · `torch ≥2.0` · `scikit-learn 1.8.0` · `matplotlib 3.10.8` · `pandas 3.0.2` · `Pillow 12.1.1` · `tensorflow ≥2.15` · `pydicom ≥2.4`
