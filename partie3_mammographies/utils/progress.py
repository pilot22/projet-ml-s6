import logging
import sys
import time
from pathlib import Path


def setup_logger(name="partie3", log_dir=None):
    """Logger console + fichier optionnel, avec horodatage."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_file = Path(log_dir) / f"{name}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.info(f"Journal : {log_file}")

    return logger


class StepTimer:
    """Chronomètre pour une étape (ex. chargement train, entraînement CNN)."""

    def __init__(self, label, logger=None):
        self.label = label
        self.logger = logger or logging.getLogger("partie3")
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()
        self.logger.info(f"▶ {self.label}")
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        self.logger.info(f"✓ {self.label} — {elapsed:.1f}s")
        return False


class ProgressTracker:
    """Suivi d'avancement avec % et ETA."""

    def __init__(self, total, label, logger=None, every=50):
        self.total = max(total, 1)
        self.label = label
        self.logger = logger or logging.getLogger("partie3")
        self.every = every
        self.done = 0
        self.start = time.perf_counter()
        self.logger.info(f"{label} — 0/{self.total} (0.0%)")

    def update(self, n=1):
        self.done += n
        if self.done % self.every != 0 and self.done != self.total:
            return

        elapsed = time.perf_counter() - self.start
        pct = 100 * self.done / self.total
        rate = self.done / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.done) / rate if rate > 0 else 0
        self.logger.info(
            f"{self.label} — {self.done}/{self.total} ({pct:.1f}%) "
            f"| {self._fmt(elapsed)} écoulé | ~{self._fmt(remaining)} restant"
        )

    def finish(self):
        elapsed = time.perf_counter() - self.start
        self.logger.info(f"{self.label} — terminé : {self.done}/{self.total} en {self._fmt(elapsed)}")

    @staticmethod
    def _fmt(seconds):
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes, secs = divmod(int(seconds), 60)
        return f"{minutes}m{secs:02d}s"
