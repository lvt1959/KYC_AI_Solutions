"""Inference + KYC validation rules.

Provides:
- predict_and_crop(image_path, ...): returns structured detections + saved crops
- validate_for_kyc(prediction_result): returns status (OK/REVIEW/REJECTED) + reasons

Usage as CLI:
    python -m src.inference --model runs/train/kyc_2_1_midv2020/weights/best.pt --image doc.jpg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.config import (
    CLASS_NAMES,
    EXPIRY_CLASS,
    MEAN_CONF_THRESHOLD,
    PHOTO_CLASS,
    PHOTO_MIN_CONF,
)


def predict_and_crop(
    image_path: str | Path,
    model_path: str | Path | None = None,
    output_dir: str | Path = "crops",
    conf_threshold: float = 0.25,
    model_obj: Any = None,
) -> dict[str, Any]:
    """Detect all KYC fields on a doc image and save crops.

    Parameters
    ----------
    image_path : path to the document image
    model_path : path to .pt weights (used if model_obj is None)
    output_dir : base directory for crop output
    conf_threshold : YOLO confidence threshold
    model_obj : pre-loaded YOLO model (takes priority over model_path)

    Returns
    -------
    dict with:
      - image_path: original input path
      - detections: {class_name: {bbox, conf, crop_path}}
      - kyc_critical: {photo: {...} | None, date_of_expiry: {...} | None}
      - missing_fields: list[str]
    """
    import cv2
    from ultralytics import YOLO

    image_path = Path(image_path)
    out_dir = Path(output_dir) / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "image_path": str(image_path),
        "detections": {},
        "kyc_critical": {"photo": None, "date_of_expiry": None},
        "missing_fields": [],
    }

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        result["error"] = f"Cannot read image: {image_path}"
        return result
    h, w = img_bgr.shape[:2]

    # Load model
    if model_obj is not None:
        model = model_obj
    elif model_path is not None:
        model = YOLO(str(model_path))
    else:
        raise ValueError("Either model_path or model_obj must be provided")

    preds = model.predict(str(image_path), conf=conf_threshold, verbose=False)[0]

    for box in preds.boxes:
        cls_id = int(box.cls.item())
        cls_name = CLASS_NAMES[cls_id]
        conf = float(box.conf.item())
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # If a class appears multiple times, keep highest confidence
        if cls_name in result["detections"] and result["detections"][cls_name]["conf"] >= conf:
            continue

        crop = img_bgr[y1:y2, x1:x2]
        crop_path = out_dir / f"{cls_name}.jpg"
        cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

        det = {
            "bbox": [x1, y1, x2, y2],
            "conf": round(conf, 4),
            "crop_path": str(crop_path),
        }
        result["detections"][cls_name] = det

        # Flag KYC-critical fields
        if cls_name == PHOTO_CLASS:
            result["kyc_critical"]["photo"] = det
        if cls_name == EXPIRY_CLASS:
            result["kyc_critical"]["date_of_expiry"] = det

    # Expected fields that were not detected
    expected = [c for c in ("document", PHOTO_CLASS, EXPIRY_CLASS) if c]
    result["missing_fields"] = [c for c in expected if c not in result["detections"]]

    return result


def validate_for_kyc(prediction_result: dict[str, Any]) -> dict[str, Any]:
    """Apply KYC business rules to a prediction result.

    Returns {"status": "OK"|"REVIEW"|"REJECTED", "reasons": list[str]}.
    """
    reasons: list[str] = []
    status = "OK"

    det = prediction_result.get("detections", {})
    crit = prediction_result.get("kyc_critical", {})

    if not det:
        return {"status": "REJECTED",
                "reasons": ["Aucun champ detecte - image non conforme"]}

    # Photo is mandatory (input for step 2.2 face-match)
    if not crit.get("photo"):
        return {
            "status": "REJECTED",
            "reasons": ["CRITIQUE : photo du porteur non detectee - face-match impossible"],
        }

    # Photo detected with low confidence
    if crit["photo"]["conf"] < PHOTO_MIN_CONF:
        reasons.append(
            f"Photo detectee avec faible confiance ({crit['photo']['conf']:.2f})"
        )
        status = "REVIEW"

    # Expiry date missing -> report will be incomplete
    if not crit.get("date_of_expiry"):
        reasons.append("Date d'expiration non detectee - rapport incomplet")
        if status == "OK":
            status = "REVIEW"

    # Average confidence across all detections
    if det:
        mean_conf = sum(d["conf"] for d in det.values()) / len(det)
        if mean_conf < MEAN_CONF_THRESHOLD:
            reasons.append(
                f"Confiance moyenne faible ({mean_conf:.2f}) - image probablement floue"
            )
            if status == "OK":
                status = "REVIEW"

    return {"status": status, "reasons": reasons}


def main() -> None:
    p = argparse.ArgumentParser(description="KYC document detection inference")
    p.add_argument("--model", type=Path, required=True, help="Path to .pt model weights")
    p.add_argument("--image", type=Path, required=True, help="Input document image")
    p.add_argument("--output", type=Path, default=Path("crops"),
                   help="Output directory for crops")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Confidence threshold")
    args = p.parse_args()

    result = predict_and_crop(args.image, model_path=args.model, output_dir=args.output,
                              conf_threshold=args.conf)
    quality = validate_for_kyc(result)

    output = {**result, "quality": quality}
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
