from utils.data_loading import load_mnist_data
from models.linear_model import LinearModel
from models.mlp_model import MLPModel
from training.trainer import train
from evaluation.metrics import error_rate
from utils.config import HIDDEN_DIMS_H1, HIDDEN_DIMS_H2
from evaluation.visualization import plot_loss_curves, plot_misclassified, plot_confusion_matrix, plot_2d_projection


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

    histories = {}
    trained_models = {}

    while True:
        print("\n=== Menu ===")
        print("1. Modele lineaire")
        print("2. MLP (H=1)")
        print("3. MLP (H=2)")
        print("4. Tous les modeles")
        print("5. Afficher les courbes de loss")
        print("6. Afficher les chiffres mal classes")
        print("7. Afficher la matrice de confusion")
        print("8. Afficher la projection 2D")
        print("9. Quitter")

        choix = input("\nChoix : ")

        models_config = {
            "1": (LinearModel(), "modele lineaire"),
            "2": (MLPModel(HIDDEN_DIMS_H1), "MLP (H=1)"),
            "3": (MLPModel(HIDDEN_DIMS_H2), "MLP (H=2)"),
        }

        match choix:
            case "1" | "2" | "3":
                model, name = models_config[choix]
                h = run_model(X_train, y_train, X_test, y_test, model, name)
                histories[name] = h
                trained_models[name] = model

            case "4":
                for key, (model, name) in models_config.items():
                    h = run_model(X_train, y_train, X_test, y_test, model, name)
                    histories[name] = h
                    trained_models[name] = model

            case "5":
                if histories:
                    plot_loss_curves(list(histories.values()), list(histories.keys()))
                else:
                    print("Aucun modele entraine.")

            case "6" | "7" | "8":
                if not trained_models:
                    print("Aucun modele entraine.")
                    continue

                # Choisir quel modèle utiliser
                print("\nModeles disponibles :")
                for i, name in enumerate(trained_models.keys(), 1):
                    print(f"  {i}. {name}")
                model_choix = input("Choix : ")

                names = list(trained_models.keys())
                idx = int(model_choix) - 1
                if idx < 0 or idx >= len(names):
                    print("Choix invalide.")
                    continue

                model = trained_models[names[idx]]
                P_test = model.forward(X_test)

                match choix:
                    case "6":
                        plot_misclassified(X_test, y_test, P_test)
                    case "7":
                        plot_confusion_matrix(X_test, y_test, P_test)
                    case "8":
                        if names[idx] == "modele lineaire":
                            print("La projection 2D n'est disponible que pour les MLP.")
                        else:
                            plot_2d_projection(model, X_test, y_test)

            case "9":
                print("Au revoir !")
                break

            case _:
                print("Choix invalide.")


if __name__ == "__main__":
    main()