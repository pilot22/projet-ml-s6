from utils.data_loading import load_mnist_data
from models.linear_model import LinearModel
from training.trainer import train
from training.loss import cross_entropy
from evaluation.metrics import error_rate

def run_linear_model():
    # 1. Charger les données
    print("Chargement de MNIST...")
    X_train, y_train, X_test, y_test = load_mnist_data()
    print(f"Train : {X_train.shape}, Test : {X_test.shape}\n")

    # 2. Créer le modèle linéaire
    model = LinearModel()

    # 3. Entraîner
    print("Entrainement du modele lineaire...")
    history = train(model, X_train, y_train, lr=0.01)

    # 4. Évaluer sur train et test
    P_train = model.forward(X_train)
    P_test = model.forward(X_test)

    print(f"\nTaux d'erreur train : {error_rate(P_train, y_train) * 100:.2f}%")
    print(f"Taux d'erreur test  : {error_rate(P_test, y_test) * 100:.2f}%")


if __name__ == "__main__":
    run_linear_model()
    