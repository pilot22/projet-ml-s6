import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "datasets", "cbis-ddsm")

# Layout Kaggle (awsaf49/cbis-ddsm-breast-cancer-image-dataset) après unzip :
#   datasets/cbis-ddsm/
#     csv/mass_case_description_train_set.csv
#     csv/mass_case_description_test_set.csv
#     jpeg/...   (arborescence d'images .jpg)
TRAIN_CSV = os.path.join(DATA_DIR, "csv", "mass_case_description_train_set.csv")
TEST_CSV = os.path.join(DATA_DIR, "csv", "mass_case_description_test_set.csv")
DICOM_INFO_CSV = os.path.join(DATA_DIR, "csv", "dicom_info.csv")
IMAGES_ROOT = os.path.join(DATA_DIR, "jpeg")

CACHE_DIR = os.path.join(DATA_DIR, "cache")
LOG_DIR = os.path.join(BASE_DIR, "logs")
PROGRESS_EVERY = 50  # log tous les N images lors du chargement

PATHOLOGY_COL = "pathology"
IMAGE_PATH_COL = "cropped image file path"

BENIGN_LABELS = {"BENIGN", "BENIGN_WITHOUT_CALLBACK"}
MALIGNANT_LABEL = "MALIGNANT"

IMAGE_SIZE = 128
MAX_SAMPLES = None  # ex. 200 pour un test rapide

EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
CLASS_WEIGHTS = None
