import os

import numpy as np
import pandas as pd
from PIL import Image

from utils.config import (
    DATA_DIR,
    TRAIN_CSV,
    TEST_CSV,
    DICOM_INFO_CSV,
    IMAGES_ROOT,
    CACHE_DIR,
    PATHOLOGY_COL,
    IMAGE_PATH_COL,
    BENIGN_LABELS,
    MALIGNANT_LABEL,
    IMAGE_SIZE,
    MAX_SAMPLES,
    PROGRESS_EVERY,
)
from utils.progress import StepTimer, ProgressTracker

_JPEG_LOOKUP = None


def _build_jpeg_lookup(logger=None):
    """
    Le pack Kaggle stocke les .jpg sous jpeg/<UID>/.
    dicom_info.csv fait le lien entre le chemin .dcm du CSV mass et le .jpg local.
    """
    global _JPEG_LOOKUP
    if _JPEG_LOOKUP is not None:
        return _JPEG_LOOKUP

    if not os.path.isfile(DICOM_INFO_CSV):
        raise FileNotFoundError(
            f"dicom_info.csv introuvable : {DICOM_INFO_CSV}\n"
            f"Vérifie que le dossier csv/ du dataset Kaggle est complet."
        )

    if logger:
        logger.info("Construction de l'index CSV → jpeg (dicom_info.csv)...")

    dicom = pd.read_csv(DICOM_INFO_CSV)
    cropped = dicom[dicom["SeriesDescription"] == "cropped images"]

    lookup = {}
    for _, row in cropped.iterrows():
        file_path = str(row["file_path"]).replace("\\", "/")
        series_uid = file_path.rstrip("/").split("/")[-2]
        image_path = str(row["image_path"]).replace("\\", "/")
        jpeg_rel = image_path.split("jpeg/")[-1]
        lookup[series_uid] = os.path.join(IMAGES_ROOT, jpeg_rel)

    _JPEG_LOOKUP = lookup
    if logger:
        logger.info(f"Index jpeg : {len(lookup)} images recensées")
    return lookup


def pathology_to_label(pathology):
    """0 = bénin, 1 = malin."""
    value = str(pathology).strip().upper()
    if value in BENIGN_LABELS:
        return 0
    if value == MALIGNANT_LABEL:
        return 1
    raise ValueError(f"Pathology inconnue : {pathology}")


def _resolve_image_path(relative_path, jpeg_lookup):
    """
    Les CSV mass pointent vers des .dcm (TCIA).
    Le pack Kaggle fournit des .jpg indexés via dicom_info.csv.
    """
    rel = str(relative_path).replace("\\", "/").strip()

    if rel.lower().endswith(".dcm"):
        series_uid = rel.split("/")[-2]
        if series_uid in jpeg_lookup:
            return jpeg_lookup[series_uid]

    rel_jpg = rel[:-4] + ".jpg" if rel.lower().endswith(".dcm") else rel
    candidates = [
        os.path.join(IMAGES_ROOT, rel_jpg),
        os.path.join(IMAGES_ROOT, rel),
        os.path.join(DATA_DIR, rel_jpg),
        os.path.join(IMAGES_ROOT, os.path.basename(rel_jpg)),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0]


def _load_dicom_grayscale(path):
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError(
            "Installe pydicom pour lire les mammographies DICOM : pip install pydicom"
        ) from exc

    ds = pydicom.dcmread(path)
    pixels = ds.pixel_array.astype(np.float32)

    pixels -= pixels.min()
    if pixels.max() > 0:
        pixels /= pixels.max()
    return pixels


def _load_image_grayscale(path):
    if path.lower().endswith(".dcm"):
        return _load_dicom_grayscale(path)

    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def _load_csv_split(csv_path, max_samples=None, logger=None):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"CSV introuvable : {csv_path}\n"
            f"Après téléchargement Kaggle, place les CSV dans :\n"
            f"  {os.path.join(DATA_DIR, 'csv')}/"
        )

    df = pd.read_csv(csv_path)
    if max_samples is not None:
        df = df.head(max_samples)

    jpeg_lookup = _build_jpeg_lookup(logger)
    split_name = os.path.basename(csv_path)
    progress = ProgressTracker(
        len(df),
        label=f"Images {split_name}",
        logger=logger,
        every=PROGRESS_EVERY,
    )

    images = []
    labels = []
    missing = 0

    for _, row in df.iterrows():
        img_path = _resolve_image_path(row[IMAGE_PATH_COL], jpeg_lookup)
        if not os.path.isfile(img_path):
            missing += 1
            progress.update()
            continue

        img = _load_image_grayscale(img_path)
        img = Image.fromarray((img * 255).astype(np.uint8))
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        img = np.asarray(img, dtype=np.float32) / 255.0

        images.append(img)
        labels.append(pathology_to_label(row[PATHOLOGY_COL]))
        progress.update()

    progress.finish()

    if missing:
        msg = f"Attention : {missing} images introuvables dans {csv_path}"
        if logger:
            logger.warning(msg)
        else:
            print(msg)

    if not images:
        raise FileNotFoundError(
            f"Aucune image chargée depuis {csv_path}.\n"
            f"Vérifie que le dossier jpeg/ est bien dans {DATA_DIR}/\n"
            f"IMAGES_ROOT = {IMAGES_ROOT}"
        )

    X = np.stack(images)
    y = np.array(labels, dtype=np.int64)
    return X, y


def class_weights_from_labels(y):
    """Poids inversement proportionnels à la fréquence des classes."""
    counts = np.bincount(y, minlength=2)
    weights = len(y) / (len(counts) * counts)
    return weights.astype(np.float32)


def load_mammography_data(use_cache=True, max_samples=MAX_SAMPLES, logger=None):
    """
    Charge CBIS-DDSM (masses), redimensionne et normalise en niveaux de gris.
    Retourne (X_train, y_train, X_test, y_test) avec X de shape (N, H, W).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_suffix = f"{IMAGE_SIZE}_{max_samples}" if max_samples else f"{IMAGE_SIZE}_all"
    cache_train_x = os.path.join(CACHE_DIR, f"X_train_{cache_suffix}.npy")
    cache_train_y = os.path.join(CACHE_DIR, f"y_train_{cache_suffix}.npy")
    cache_test_x = os.path.join(CACHE_DIR, f"X_test_{cache_suffix}.npy")
    cache_test_y = os.path.join(CACHE_DIR, f"y_test_{cache_suffix}.npy")

    if use_cache and all(os.path.exists(p) for p in (cache_train_x, cache_train_y, cache_test_x, cache_test_y)):
        with StepTimer("Chargement depuis le cache .npy", logger):
            return (
                np.load(cache_train_x),
                np.load(cache_train_y),
                np.load(cache_test_x),
                np.load(cache_test_y),
            )

    with StepTimer("Chargement train (CSV + images)", logger):
        X_train, y_train = _load_csv_split(TRAIN_CSV, max_samples, logger)

    with StepTimer("Chargement test (CSV + images)", logger):
        X_test, y_test = _load_csv_split(TEST_CSV, max_samples, logger)

    with StepTimer("Sauvegarde cache .npy", logger):
        np.save(cache_train_x, X_train)
        np.save(cache_train_y, y_train)
        np.save(cache_test_x, X_test)
        np.save(cache_test_y, y_test)

    if logger:
        logger.info(f"Cache sauvegardé dans {CACHE_DIR}")

    return X_train, y_train, X_test, y_test


def print_class_distribution(y_train, y_test, logger=None):
    for name, y in [("Train", y_train), ("Test", y_test)]:
        n_benign = np.sum(y == 0)
        n_malignant = np.sum(y == 1)
        msg = (
            f"{name} — bénin : {n_benign} ({100 * n_benign / len(y):.1f}%) | "
            f"malin : {n_malignant} ({100 * n_malignant / len(y):.1f}%)"
        )
        if logger:
            logger.info(msg)
        else:
            print(msg)
