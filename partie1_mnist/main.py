import os
from utils.data_loading import load_mnist_data
from models.linear_model import LinearModel
from models.mlp_model import MLPModel
from training.trainer import train
from evaluation.metrics import error_rate
from utils.config import HIDDEN_DIMS_H1, HIDDEN_DIMS_H2
from evaluation.visualization import (
    plot_loss_curves,
    plot_misclassified,
    plot_confusion_matrix,
    plot_2d_projection,
    plot_digit_heatmaps,
)
from evaluation.draw_predict import DrawPredict

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
SAVE_PATHS = {
    "modele lineaire": os.path.join(SAVE_DIR, "linear.npz"),
    "MLP (H=1)": os.path.join(SAVE_DIR, "mlp_h1.npz"),
    "MLP (H=2)": os.path.join(SAVE_DIR, "mlp_h2.npz"),
}


def run_model(X_train, y_train, X_test, y_test, model, name):
    print(f"\nEntrainement du {name}...")
    history = train(model, X_train, y_train)

    P_train = model.forward(X_train)
    P_test = model.forward(X_test)

    print(f"\nTaux d'erreur train : {error_rate(P_train, y_train) * 100:.2f}%")
    print(f"Taux d'erreur test  : {error_rate(P_test, y_test) * 100:.2f}%")

    # Sauvegarde automatique après entraînement
    os.makedirs(SAVE_DIR, exist_ok=True)
    if name in SAVE_PATHS:
        model.save(SAVE_PATHS[name])
        print(f"Modele sauvegarde dans {SAVE_PATHS[name]}")

    return history


def main():
    print("Chargement de MNIST...")
    X_train, y_train, X_test, y_test = load_mnist_data()
    print(f"Train : {X_train.shape}, Test : {X_test.shape}")

    histories = {}
    trained_models = {}

    while True:
        print("\n=== Menu ===")
        print("Entrainer un modele")
        print("  1. Modele lineaire")
        print("  2. MLP (H=1)")
        print("  3. MLP (H=2)")
        print("  4. Tous les modeles")
        print("Charger un modele sauvegarde")
        print("  5. Charger un modele")
        print("Visualisations")
        print("  6. Afficher les courbes de loss")
        print("  7. Afficher les chiffres mal classes")
        print("  8. Afficher la matrice de confusion")
        print("  9. Afficher la projection 2D")
        print(" 10. Afficher les heatmaps (modele lineaire)")
        print(" 11. Dessiner un chiffre")
        print("12. Quitter")

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
                load_config = {
                    "1": (LinearModel(), "modele lineaire"),
                    "2": (MLPModel(HIDDEN_DIMS_H1), "MLP (H=1)"),
                    "3": (MLPModel(HIDDEN_DIMS_H2), "MLP (H=2)"),
                }
                print("\n1. Lineaire\n2. MLP (H=1)\n3. MLP (H=2)")
                c = input("Charger : ")
                if c in load_config:
                    model, name = load_config[c]
                    path = SAVE_PATHS[name]
                    if os.path.exists(path):
                        model.load(path)
                        trained_models[name] = model
                        print(f"{name} charge depuis {path}")
                    else:
                        print(f"Fichier {path} introuvable. Entraine d'abord.")
                else:
                    print("Choix invalide.")

            case "6":
                if histories:
                    plot_loss_curves(list(histories.values()), list(histories.keys()))
                else:
                    print("Aucun modele entraine.")

            case "7" | "8" | "9":
                if not trained_models:
                    print("Aucun modele entraine.")
                    continue

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
                    case "7":
                        plot_misclassified(X_test, y_test, P_test)
                    case "8":
                        plot_confusion_matrix(X_test, y_test, P_test)
                    case "9":
                        if names[idx] == "modele lineaire":
                            print("La projection 2D n'est disponible que pour les MLP.")
                        else:
                            print("\n1. PCA activations (bassin appris)")
                            print("2. PCA pixels bruts (donnees MNIST)")
                            print("3. t-SNE activations")
                            proj = input("Choix : ")
                            match proj:
                                case "1":
                                    plot_2d_projection(model, X_test, y_test, method='pca', source='activations')
                                case "2":
                                    plot_2d_projection(model, X_test, y_test, method='pca', source='input')
                                case "3":
                                    plot_2d_projection(model, X_test, y_test, method='tsne', source='activations')
                                case _:
                                    print("Choix invalide.")

            case "10":
                if "modele lineaire" not in trained_models:
                    print("Entraine ou charge d'abord le modele lineaire.")
                else:
                    plot_digit_heatmaps(trained_models["modele lineaire"])

            case "11":
                if not trained_models:
                    print("Aucun modele entraine.")
                else:
                    DrawPredict(trained_models)

            case "12":
                print("Au revoir !")
                break

            case _:
                print("Choix invalide.")


if __name__ == "__main__":
    main()