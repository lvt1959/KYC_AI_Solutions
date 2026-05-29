"""Step 2.2 — Face matching: selfie vs document photo.

Models (auto-downloaded, no local weights needed):
  - YOLOv8-Face (HuggingFace: arnabdhar/YOLOv8-Face-Detection)
  - ArcFace buffalo_l (insightface, ResNet50, 512-dim embeddings)

Pipeline: detect face -> extract embedding -> cosine distance -> decision
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass  # HEIC support optional

# ── Thresholds ──
SEUIL_DECISION = 0.40       # Strict KYC threshold (prefer rejecting real person over accepting impostor)
SEUIL_CONFIANCE_YOLO = 0.5  # Minimum YOLO face detection confidence
TAILLE_MIN_VISAGE = 60      # Minimum face size in pixels
TAILLE_MAX = 1920            # Max image dimension before resize


def load_face_models() -> tuple:
    """Load YOLO-Face and ArcFace models.

    Returns (yolo_model, arcface_app).
    Both are downloaded automatically on first run (~300 MB total).
    """
    from ultralytics import YOLO
    from huggingface_hub import hf_hub_download
    from insightface.app import FaceAnalysis

    # YOLO-Face from HuggingFace
    chemin_yolo = hf_hub_download(
        repo_id="arnabdhar/YOLOv8-Face-Detection",
        filename="model.pt",
    )
    yolo_model = YOLO(chemin_yolo)

    # ArcFace buffalo_l (ResNet50, 99.8% on LFW)
    arcface_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
    )
    arcface_app.prepare(ctx_id=-1, det_size=(640, 640))

    return yolo_model, arcface_app


def pretraiter_image(image_input) -> np.ndarray:
    """Normalize an image for YOLO/ArcFace. Returns BGR numpy array.

    Parameters
    ----------
    image_input : str, Path, or PIL.Image
    """
    if isinstance(image_input, (str,)):
        from pathlib import Path
        image_input = Path(image_input)

    if hasattr(image_input, "exists"):
        # It's a Path
        img_pil = Image.open(str(image_input))
    elif isinstance(image_input, np.ndarray):
        # Already a numpy array (BGR)
        return image_input
    else:
        # Assume PIL Image or file-like
        img_pil = Image.open(image_input)

    img_pil = ImageOps.exif_transpose(img_pil)

    if img_pil.mode == "RGBA":
        bg = Image.new("RGB", img_pil.size, (255, 255, 255))
        bg.paste(img_pil, mask=img_pil.split()[3])
        img_pil = bg
    else:
        img_pil = img_pil.convert("RGB")

    if max(img_pil.size) > TAILLE_MAX:
        img_pil.thumbnail((TAILLE_MAX, TAILLE_MAX), Image.LANCZOS)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def detecter_visage_yolo(img_bgr: np.ndarray, yolo_model) -> np.ndarray:
    """Detect the largest face in an image using YOLO-Face. Returns cropped face (BGR)."""
    resultats = yolo_model(img_bgr, conf=SEUIL_CONFIANCE_YOLO, verbose=False, device="cpu")
    boites = resultats[0].boxes

    if boites is None or len(boites) == 0:
        raise ValueError("Aucun visage detecte. Verifiez que le visage est bien visible et centre.")

    boites_xyxy = boites.xyxy.cpu().numpy()
    boites_conf = boites.conf.cpu().numpy()

    # Select the largest face
    surfaces = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boites_xyxy]
    idx = int(np.argmax(surfaces))
    x1, y1, x2, y2 = [int(v) for v in boites_xyxy[idx]]
    confiance = float(boites_conf[idx])

    largeur = x2 - x1
    hauteur = y2 - y1
    if largeur < TAILLE_MIN_VISAGE or hauteur < TAILLE_MIN_VISAGE:
        raise ValueError(f"Visage trop petit ({largeur}x{hauteur}px). Rapprochez la camera.")

    # 20% margin around the face
    h_img, w_img = img_bgr.shape[:2]
    x1 = max(0, x1 - int(largeur * 0.20))
    y1 = max(0, y1 - int(hauteur * 0.20))
    x2 = min(w_img, x2 + int(largeur * 0.20))
    y2 = min(h_img, y2 + int(hauteur * 0.20))

    return img_bgr[y1:y2, x1:x2]


def extraire_embedding(img_bgr: np.ndarray, arcface_app) -> np.ndarray:
    """Extract ArcFace 512-dim L2-normalized embedding from a face image."""
    faces = arcface_app.get(img_bgr)
    if not faces:
        raise ValueError("Aucun visage detecte pour l'extraction d'embedding.")
    best = max(faces, key=lambda f: f.det_score)
    return best.normed_embedding


def comparer_visages(emb1: np.ndarray, emb2: np.ndarray, seuil: float | None = None) -> dict:
    """Compare two ArcFace embeddings and return KYC decision.

    Returns
    -------
    dict with: est_match, score (0-100), distance, confiance (HAUTE/MOYENNE/FAIBLE), seuil
    """
    seuil = seuil or SEUIL_DECISION
    similarite = float(np.dot(emb1, emb2))
    distance = 1.0 - similarite
    score = float(np.clip(similarite * 100, 0, 100))
    est_match = distance < seuil

    marge = abs(distance - seuil)
    if marge >= 0.15:
        confiance = "HAUTE"
    elif marge >= 0.05:
        confiance = "MOYENNE"
    else:
        confiance = "FAIBLE"

    return {
        "est_match": est_match,
        "score": round(score, 1),
        "distance": round(distance, 4),
        "confiance": confiance,
        "seuil": seuil,
    }


def run_face_match(
    photo_id_bgr: np.ndarray,
    selfie_input,
    yolo_model,
    arcface_app,
) -> dict:
    """Full face match pipeline: preprocess -> detect -> embed -> compare.

    Parameters
    ----------
    photo_id_bgr : BGR crop of the face from the ID document
    selfie_input : file-like object, path, or BGR array of the selfie
    yolo_model : pre-loaded YOLO-Face model
    arcface_app : pre-loaded ArcFace FaceAnalysis

    Returns
    -------
    dict with: result (comparer_visages output), crop_selfie (BGR), crop_id (BGR)
    """
    # Process selfie
    if isinstance(selfie_input, np.ndarray):
        img_selfie = selfie_input
    else:
        img_selfie = pretraiter_image(selfie_input)

    crop_selfie = detecter_visage_yolo(img_selfie, yolo_model)

    # Process ID photo (already a crop from step 2.1, but may need face detection)
    img_id = photo_id_bgr
    try:
        crop_id = detecter_visage_yolo(img_id, yolo_model)
    except ValueError:
        # If YOLO can't detect a face on the ID crop, use the full crop
        crop_id = img_id

    # Extract embeddings
    emb_selfie = extraire_embedding(img_selfie, arcface_app)
    emb_id = extraire_embedding(img_id, arcface_app)

    result = comparer_visages(emb_selfie, emb_id, seuil=SEUIL_DECISION)

    return {
        "result": result,
        "crop_selfie": crop_selfie,
        "crop_id": crop_id,
    }
