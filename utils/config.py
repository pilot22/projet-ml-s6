import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "datasets")
MNIST_DIR = os.path.join(DATA_DIR, "mnist-0-1000")