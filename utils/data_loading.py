import os
from . import MNIST_DIR

def retrieve_files(path):
  if not os.path.exists(path):
    return []
  
  return [
    os.path.join(path, f) 
    for f in os.listdir(path) 
  ]

def load_mnist_data():
  files = retrieve_files(MNIST_DIR)

  for f in files:
    print(f)