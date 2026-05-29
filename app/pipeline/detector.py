"""Step 2.1 — Document field detection via YOLO11n.

Model: best.pt (33 classes from GenMRP+MIDV-2020)
Input: document image
Output: detections dict + KYC validation
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ── 33 classes from GenMRP+MIDV-2020 ──
CLASS_NAMES = [
    "MRZ_line_1", "MRZ_line_2", "date_of_birth", "date_of_birth_caption",
    "date_of_expiry", "date_of_expiry_caption", "date_of_issue",
    "date_of_issue_caption", "document", "document_code",
    "document_code_caption", "document_number", "document_number_caption",
    "face_image", "issue_authority", "issue_authority_caption",
    "issuing_state_code", "issuing_state_code_caption", "issuing_state_full",
    "nationality", "nationality_caption", "personal_number",
    "personal_number_caption", "place_of_birth", "place_of_birth_caption",
    "primary_identifier", "primary_identifier_caption", "secondary_identifier",
    "secondary_identifier_caption", "sex", "sex_caption", "signature",
    "signature_caption",
]

PHOTO_CLASS = "face_image"
EXPIRY_CLASS = "date_of_expiry"
PHOTO_MIN_CONF = 0.60
MEAN_CONF_THRESHOLD = 0.65


def load_detector(model_path: str | Path):
    """Load YOLO .pt model."""
    from ultralytics import YOLO

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return YOLO(str(model_path))


def predict_and_crop(
    img_bgr: np.ndarray,
    model,
    output_dir: str | Path = "/tmp/kyc_crops",
    conf_threshold: float = 0.25,
    image_name: str = "document",
) -> dict[str, Any]:
    """Detect all KYC fields and save crops.

    Parameters
    ----------
    img_bgr : BGR numpy array of the document
    model : pre-loaded YOLO model
    output_dir : where to save crop images
    conf_threshold : YOLO confidence cutoff
    image_name : stem for naming the crop subfolder

    Returns
    -------
    dict with detections, kyc_critical, missing_fields, annotated_image
    """
    out_dir = Path(output_dir) / image_name
    out_dir.mkdir(parents=True, exist_ok=True)

    h, w = img_bgr.shape[:2]

    result: dict[str, Any] = {
        "detections": {},
        "kyc_critical": {"photo": None, "date_of_expiry": None},
        "missing_fields": [],
        "annotated_image": None,
    }

    # Save temp file for YOLO (it expects a path)
    tmp_path = out_dir / "_input.jpg"
    cv2.imwrite(str(tmp_path), img_bgr)

    preds = model.predict(str(tmp_path), conf=conf_threshold, verbose=False)[0]

    # Build annotated image with bounding boxes
    annotated = img_bgr.copy()

    for box in preds.boxes:
        cls_id = int(box.cls.item())
        if cls_id >= len(CLASS_NAMES):
            continue
        cls_name = CLASS_NAMES[cls_id]
        conf = float(box.conf.item())
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Keep highest confidence per class
        if cls_name in result["detections"] and result["detections"][cls_name]["conf"] >= conf:
            continue

        # Save crop
        crop = img_bgr[y1:y2, x1:x2]
        crop_path = out_dir / f"{cls_name}.jpg"
        if crop.size > 0:
            cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

        det = {
            "bbox": [x1, y1, x2, y2],
            "conf": round(conf, 4),
            "crop_path": str(crop_path),
        }
        result["detections"][cls_name] = det

        if cls_name == PHOTO_CLASS:
            result["kyc_critical"]["photo"] = det
        if cls_name == EXPIRY_CLASS:
            result["kyc_critical"]["date_of_expiry"] = det

        # Draw on annotated image
        color = (0, 200, 0) if cls_name in (PHOTO_CLASS, EXPIRY_CLASS) else (255, 180, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    result["annotated_image"] = annotated

    expected = [c for c in ("document", PHOTO_CLASS, EXPIRY_CLASS) if c]
    result["missing_fields"] = [c for c in expected if c not in result["detections"]]

    # Cleanup temp
    tmp_path.unlink(missing_ok=True)

    return result


def validate_for_kyc(prediction_result: dict[str, Any]) -> dict[str, Any]:
    """Apply KYC business rules. Returns {status, reasons}."""
    reasons: list[str] = []
    status = "OK"

    det = prediction_result.get("detections", {})
    crit = prediction_result.get("kyc_critical", {})

    if not det:
        return {"status": "REJECTED", "reasons": ["Aucun champ detecte — image non conforme"]}

    if not crit.get("photo"):
        return {"status": "REJECTED", "reasons": ["Photo du porteur non detectee — face-match impossible"]}

    if crit["photo"]["conf"] < PHOTO_MIN_CONF:
        reasons.append(f"Photo detectee avec faible confiance ({crit['photo']['conf']:.2f})")
        status = "REVIEW"

    if not crit.get("date_of_expiry"):
        reasons.append("Date d'expiration non detectee — rapport incomplet")
        if status == "OK":
            status = "REVIEW"

    if det:
        mean_conf = sum(d["conf"] for d in det.values()) / len(det)
        if mean_conf < MEAN_CONF_THRESHOLD:
            reasons.append(f"Confiance moyenne faible ({mean_conf:.2f}) — image probablement floue")
            if status == "OK":
                status = "REVIEW"

    return {"status": status, "reasons": reasons}
