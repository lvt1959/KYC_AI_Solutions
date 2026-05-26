"""Central config: classes, aliases, default hyperparameters.

Aligned with the GenMRP+MIDV-2020 dataset (Roboflow, 33 classes).
"""
from pathlib import Path

# ────────────────────────────────────────────────────────────────────
# Classes — 33 classes from GenMRP+MIDV-2020 (Maastricht University)
# ────────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "MRZ_line_1",
    "MRZ_line_2",
    "date_of_birth",
    "date_of_birth_caption",
    "date_of_expiry",
    "date_of_expiry_caption",
    "date_of_issue",
    "date_of_issue_caption",
    "document",
    "document_code",
    "document_code_caption",
    "document_number",
    "document_number_caption",
    "face_image",
    "issue_authority",
    "issue_authority_caption",
    "issuing_state_code",
    "issuing_state_code_caption",
    "issuing_state_full",
    "nationality",
    "nationality_caption",
    "personal_number",
    "personal_number_caption",
    "place_of_birth",
    "place_of_birth_caption",
    "primary_identifier",
    "primary_identifier_caption",
    "secondary_identifier",
    "secondary_identifier_caption",
    "sex",
    "sex_caption",
    "signature",
    "signature_caption",
]
NUM_CLASSES = len(CLASS_NAMES)

# ────────────────────────────────────────────────────────────────────
# KYC-critical class aliases
# Different datasets may name "photo" differently. We normalize here.
# ────────────────────────────────────────────────────────────────────

PHOTO_ALIASES = {"photo", "face", "portrait", "photograph", "face_photo", "face_image"}
EXPIRY_ALIASES = {"date_of_expiry", "expiry_date", "expiration", "date_of_expiration", "expiry"}

# Resolved class names for this dataset
PHOTO_CLASS = next((c for c in CLASS_NAMES if c.lower() in PHOTO_ALIASES), None)
EXPIRY_CLASS = next((c for c in CLASS_NAMES if c.lower() in EXPIRY_ALIASES), None)

# ────────────────────────────────────────────────────────────────────
# Training defaults — YOLO11n on GenMRP+MIDV-2020
# ────────────────────────────────────────────────────────────────────

DEFAULT_TRAIN_KWARGS = dict(
    epochs=40,
    imgsz=640,
    batch=32,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=3,
    patience=12,
    close_mosaic=10,
    # Domain-specific augmentations — disable flips
    # (text, dates, MRZ are not symmetric; flipping creates harmful artifacts)
    fliplr=0.0,
    flipud=0.0,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    hsv_h=0.015,
    hsv_s=0.6,
    hsv_v=0.3,
    mixup=0.0,
    save=True,
    plots=True,
    verbose=True,
)

# ────────────────────────────────────────────────────────────────────
# KYC validation thresholds
# ────────────────────────────────────────────────────────────────────

PHOTO_MIN_CONF = 0.60           # Photo must be detected with at least this conf
MEAN_CONF_THRESHOLD = 0.65      # If mean(confs) < this -> flag for review

# ────────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
