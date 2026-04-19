import numpy as np
from utils.config import EPOCHS, BATCH_SIZE, LEARNING_RATE, SEED
from training.loss import cross_entropy


def train(model, X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    n = len(X_train) # Nombre d'images d'entraînement
    history = []
    rng = np.random.default_rng(SEED)

    for epoch in range(epochs):
        # Mélanger les données à chaque epoch car sinon le modele apprend l'ordre des données.
        indices = rng.permutation(n)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        epoch_loss = 0
        n_batches = 0

        # Boucle sur les batches 
        for start in range(0, n, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Propagation, loss et rétropropagation
            P = model.forward(X_batch)
            loss = cross_entropy(P, y_batch)
            model.backward(y_batch, lr)

            epoch_loss += loss
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs} - Loss : {avg_loss:.4f}")

    return history