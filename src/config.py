"""Central config: classes, label mapping, default hyperparameters."""
from pathlib import Path

# ────────────────────────────────────────────────────────────────────
# Classes
# ────────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "photo",
    "mrz",
    "name",
    "birth_date",
    "expiry_date",
    "document_number",
]
NUM_CLASSES = len(CLASS_NAMES)

# Mapping VIA label → class id. MIDV-2020 has multiple naming conventions
# depending on document type; we normalize here.
CLASS_MAPPING = {
    # photo
    "photo": 0, "portrait": 0, "face": 0, "photograph": 0,
    # mrz (all lines merged into one class)
    "mrz": 1, "mrz_line_1": 1, "mrz_line_2": 1, "mrz_line_3": 1,
    "machine_readable_zone": 1,
    # name
    "surname": 2, "name": 2, "given_names": 2, "given_name": 2,
    "first_name": 2, "last_name": 2,
    # birth date
    "date_of_birth": 3, "birth_date": 3, "dob": 3, "birthdate": 3,
    # expiry date
    "date_of_expiry": 4, "expiry_date": 4, "expiration": 4,
    "date_of_expiration": 4,
    # document number
    "document_number": 5, "doc_number": 5, "passport_number": 5,
    "id_number": 5, "card_number": 5, "number": 5,
}

# ────────────────────────────────────────────────────────────────────
# Training defaults
# ────────────────────────────────────────────────────────────────────

DEFAULT_TRAIN_KWARGS = dict(
    epochs=50,
    imgsz=640,
    batch=16,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=3,
    patience=15,
    close_mosaic=10,
    # Domain-specific augmentations — disable horizontal flip
    # (dates and MRZ are not symmetric, flipping creates harmful artifacts)
    fliplr=0.0,
    flipud=0.0,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    mixup=0.0,
    save=True,
    plots=True,
    verbose=True,
)

# ────────────────────────────────────────────────────────────────────
# KYC validation thresholds
# ────────────────────────────────────────────────────────────────────

PHOTO_MIN_CONF = 0.70           # Photo must be detected with at least this conf
MEAN_CONF_THRESHOLD = 0.70      # If mean(confs) < this → flag for review
MAX_MISSING_FIELDS = 3          # If more than this many fields missing → review

# ────────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
