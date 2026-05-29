"""Step 1 — Document classification via TensorFlow/Keras CNN.

Model: kyc_classifier_best.keras (binary: id / passport)
Input: 224x224 RGB image
Output: {type_document, confiance}
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Class names matching the training folder structure (alphabetical order)
CLASS_NAMES = ["id", "passport"]
IMG_SIZE = (224, 224)


def load_classifier(model_path: str | Path):
    """Load the Keras .keras model from disk."""
    import tensorflow as tf

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return tf.keras.models.load_model(str(model_path))


def classify_document(img_bgr: np.ndarray, model) -> dict:
    """Run classification on a BGR image.

    Returns
    -------
    dict with:
      - type_document: str ("Carte Nationale d'Identite" | "Passeport")
      - confiance: float [0, 1]
      - classe_brute: str ("id" | "passport")
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_batch = np.expand_dims(img_resized, axis=0).astype(np.float32)
    # Note: the model includes a Rescaling(1/255) layer — no manual normalization needed

    preds = model.predict(img_batch, verbose=0)
    class_idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][class_idx])

    label_map = {
        "id": "Carte Nationale d'Identite",
        "passport": "Passeport",
    }

    return {
        "type_document": label_map.get(CLASS_NAMES[class_idx], CLASS_NAMES[class_idx]),
        "confiance": round(confidence, 4),
        "classe_brute": CLASS_NAMES[class_idx],
    }
