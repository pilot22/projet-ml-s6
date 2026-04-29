import os

# Chemins
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "datasets")
MNIST_DIR = os.path.join(DATA_DIR, "mnist-0-1000")

# Reproductibilité
SEED = 42

# Hyperparamètres
LEARNING_RATE = 0.01
BATCH_SIZE = 64
EPOCHS = 10

# Architecture MLP (Perceptron Multicouche) à renseigner pour le modele MLP
HIDDEN_DIMS_H1 = [128]  # [nb de neurones couche cachée 1]
HIDDEN_DIMS_H2 = [128, 64]  # [nb de neurones 1ere couche, nb de neurones 2e couche]
