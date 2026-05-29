from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass  # Support HEIC optionnel

# ── Seuils et Configurations ──
SEUIL_DECISION = 0.60  # Seuil KYC (distance cosinus)
TAILLE_MAX = 1920  # Dimension max avant redimensionnement


def load_face_model():
    """Initialise uniquement InsightFace (détection SCRFD + embedding ArcFace).

    Téléchargement automatique au premier lancement (~260 MB).
    """
    from insightface.app import FaceAnalysis

    # Initialisation avec le provider d'exécution (CPU ici)
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
    )

    # CORRECTION : ctx_id=-1 pour l'API, det_size standard, et on baisse le seuil (det_thresh)
    # pour être plus tolérant sur les visages de documents (hologrammes, etc.)
    app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.25)

    return app


def pretraiter_image(image_input) -> np.ndarray:
    """Normalise l'image en entrée. Retourne un tableau numpy au format BGR."""
    if isinstance(image_input, (str,)):
        from pathlib import Path
        image_input = Path(image_input)

    if hasattr(image_input, "exists"):
        img_pil = Image.open(str(image_input))
    elif isinstance(image_input, np.ndarray):
        return image_input
    else:
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


def extraire_donnees_visage(img_bgr: np.ndarray, arcface_app) -> tuple[np.ndarray, np.ndarray]:
    """Détecte, aligne le visage et extrait l'embedding.

    Retourne l'embedding normalisé et le crop BGR du visage détecté.
    """
    # .get() effectue la détection, l'alignement et l'extraction d'embedding
    faces = arcface_app.get(img_bgr)

    # 💡 Sécurité supplémentaire : Si aucun visage n'est détecté
    # (souvent à cause d'un crop très serré issu de YOLO sur la pièce d'identité)
    # on force le seuil à un niveau très bas temporairement.
    if not faces:
        seuil_original = arcface_app.det_model.det_thresh
        arcface_app.det_model.det_thresh = 0.05
        faces = arcface_app.get(img_bgr)
        # On restaure le seuil normal pour la suite
        arcface_app.det_model.det_thresh = seuil_original

    if not faces:
        raise ValueError("Aucun visage détecté sur l'image (même après abaissement du seuil). Vérifiez la netteté.")

    # On sélectionne le visage qui a le plus grand score de confiance de détection
    meilleur_visage = max(faces, key=lambda f: f.det_score)

    # 1. Extraction de l'embedding 512-dim normalisé
    embedding = meilleur_visage.normed_embedding

    # 2. Récupération de la bounding box pour recréer le comportement de "crop" visuel
    bbox = meilleur_visage.bbox.astype(int)
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

    # Ajustement des bordures de sécurité
    h_img, w_img = img_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)

    crop_visage = img_bgr[y1:y2, x1:x2]

    return embedding, crop_visage


def comparer_visages(emb1: np.ndarray, emb2: np.ndarray, seuil: float | None = None) -> dict:
    """Compare deux embeddings ArcFace via la distance cosinus."""
    seuil = seuil or SEUIL_DECISION

    # Produit scalaire sur vecteurs normalisés L2 = Similarité cosinus
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
        arcface_app,
) -> dict:
    """Pipeline KYC simplifié : charge, extrait via InsightFace et compare."""
    # Traitement du selfie
    if isinstance(selfie_input, np.ndarray):
        img_selfie = selfie_input
    else:
        img_selfie = pretraiter_image(selfie_input)

    # Traitement de l'image de la pièce d'identité
    if isinstance(photo_id_bgr, np.ndarray):
        img_id = photo_id_bgr
    else:
        img_id = pretraiter_image(photo_id_bgr)

    # Extraction des embeddings et des crops en une seule étape par image
    emb_selfie, crop_selfie = extraire_donnees_visage(img_selfie, arcface_app)
    emb_id, crop_id = extraire_donnees_visage(img_id, arcface_app)

    # Comparaison des visages (calcul de la distance)
    result = comparer_visages(emb_selfie, emb_id, seuil=SEUIL_DECISION)

    return {
        "result": result,
        "crop_selfie": crop_selfie,
        "crop_id": crop_id,
    }