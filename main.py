from utils.data_loading import load_mnist_data
from models.linear_model import LinearModel
from models.mlp_model import MLPModel
from training.trainer import train
from evaluation.metrics import error_rate
from utils.config import HIDDEN_DIMS_H1, HIDDEN_DIMS_H2
from evaluation.visualization import plot_loss_curves


def run_model(X_train, y_train, X_test, y_test, model, name):
    print(f"\nEntrainement du {name}...")
    history = train(model, X_train, y_train)

    P_train = model.forward(X_train)
    P_test = model.forward(X_test)

    print(f"\nTaux d'erreur train : {error_rate(P_train, y_train) * 100:.2f}%")
    print(f"Taux d'erreur test  : {error_rate(P_test, y_test) * 100:.2f}%")

    return history


def main():
    print("Chargement de MNIST...")
    X_train, y_train, X_test, y_test = load_mnist_data()
    print(f"Train : {X_train.shape}, Test : {X_test.shape}")

    print("\n=== Menu ===")
    print("1. Modele lineaire")
    print("2. MLP (H=1)")
    print("3. MLP (H=2)")
    print("4. Tous les modeles")

    choix = input("\nChoix : ")

    models = {
        "1": (LinearModel(), "modele lineaire"),
        "2": (MLPModel(HIDDEN_DIMS_H1), "MLP (H=1)"),
        "3": (MLPModel(HIDDEN_DIMS_H2), "MLP (H=2)"),
    }

    if choix == "4":
        histories = []
        labels = []
        for model, name in models.values():
            h = run_model(X_train, y_train, X_test, y_test, model, name)
            histories.append(h)
            labels.append(name)
        plot_loss_curves(histories, labels)
    else:
        print("Choix invalide.")


if __name__ == "__main__":
    main()