import sys
from pathlib import Path

PARTIE3_DIR = Path(__file__).resolve().parent
ROOT = PARTIE3_DIR.parent

# Partie 3 en premier pour que utils/ = utils de cette partie
sys.path.insert(0, str(PARTIE3_DIR))
sys.path.append(str(ROOT / "partie1_mnist"))
sys.path.append(str(ROOT / "partie2_cifar10"))

import numpy as np
import torch

from utils.config import IMAGE_SIZE, EPOCHS, BATCH_SIZE, LEARNING_RATE, CLASS_WEIGHTS, LOG_DIR
from utils.data_loading import load_mammography_data, print_class_distribution, class_weights_from_labels
from utils.progress import setup_logger, StepTimer
from convolution.cnn import entrainer_cnn
from evaluation.metrics import confusion_matrix_from_preds

logger = setup_logger("partie3", log_dir=LOG_DIR)

with StepTimer("Chargement CBIS-DDSM", logger):
    X_train, y_train, X_test, y_test = load_mammography_data(logger=logger)

logger.info(f"Train : {X_train.shape}, Test : {X_test.shape}")

logger.info("=== Distribution des classes (déséquilibre) ===")
print_class_distribution(y_train, y_test, logger)

X_train_t = torch.tensor(X_train[:, np.newaxis, :, :], dtype=torch.float32)
X_test_t = torch.tensor(X_test[:, np.newaxis, :, :], dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)

class_weights = CLASS_WEIGHTS if CLASS_WEIGHTS is not None else class_weights_from_labels(y_train)
print("Class weights:", class_weights)
logger.info("=== CNN binaire (architecture partie 2) ===")
with StepTimer("Entraînement CNN", logger):
    _, _, y_pred_test, _ = entrainer_cnn(
        X_train_t,
        y_train_t,
        X_test_t,
        y_test_t,
        in_channels=1,
        num_classes=2,
        input_size=IMAGE_SIZE,
        filters=(16, 32, 32, 32),
        use_bn=True,
        patience=10,
        augment=True,
        dropout=0.5,
        threshold=0.35,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        class_weights=class_weights,
        title="CNN mammographies",
        save_loss_path="loss_cnn_mammography.png",
    )

logger.info("=== Matrice de confusion (test) ===")
logger.info("Lignes = vrai label, Colonnes = prédiction — [bénin, malin]")
cm = confusion_matrix_from_preds(y_test, y_pred_test, num_classes=2)
logger.info(f"\n{cm}")

tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
logger.info(f"Vrais négatifs  (bénin → bénin)   : {tn}")
logger.info(f"Faux positifs   (bénin → malin)   : {fp}")
logger.info(f"Faux négatifs   (malin → bénin)   : {fn}  ← critique en diagnostic")
logger.info(f"Vrais positifs  (malin → malin)   : {tp}")

if fn + tp > 0:
    logger.info(f"Sensibilité (rappel malins) : {tp / (tp + fn):.4f}")
if tn + fp > 0:
    logger.info(f"Spécificité (vrais bénins) : {tn / (tn + fp):.4f}")
