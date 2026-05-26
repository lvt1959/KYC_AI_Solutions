"""Inference + KYC validation rules.

Provides:
- predict_and_crop(image_path, model_path): returns structured detections + saved crops
- validate_detection_quality(result):       returns status (OK/REVIEW/REJECTED) + warnings

Usage as CLI:
    python -m src.inference --model runs/train/kyc_2_1/weights/best.pt --image doc.jpg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.config import (
    CLASS_NAMES,
    MAX_MISSING_FIELDS,
    MEAN_CONF_THRESHOLD,
    PHOTO_MIN_CONF,
)


def predict_and_crop(
    image_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path = "crops",
    conf_threshold: float = 0.25,
) -> dict[str, Any]:
    """Detect all KYC fields on a doc image and save crops.

    Returns
    -------
    dict with:
      - image_path: original input path
      - detections: {class_name: {bbox, conf, crop_path}}
      - missing_fields: list[str]
      - is_valid_kyc_input: bool
    """
    import cv2
    from ultralytics import YOLO

    image_path = Path(image_path)
    output_dir = Path(output_dir) / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return {"error": f"Cannot read image: {image_path}", "image_path": str(image_path)}
    h, w = img_bgr.shape[:2]

    model = YOLO(str(model_path))
    results = model.predict(str(image_path), conf=conf_threshold, verbose=False)[0]

    detections: dict[str, Any] = {}
    for box in results.boxes:
        cls_id = int(box.cls.item())
        cls_name = CLASS_NAMES[cls_id]
        conf = float(box.conf.item())
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # If a class appears multiple times, keep highest confidence
        if cls_name in detections and detections[cls_name]["conf"] >= conf:
            continue

        crop = img_bgr[y1:y2, x1:x2]
        crop_path = output_dir / f"{cls_name}.png"
        cv2.imwrite(str(crop_path), crop)

        detections[cls_name] = {
            "bbox": [x1, y1, x2, y2],
            "conf": round(conf, 4),
            "crop_path": str(crop_path),
        }

    missing = [c for c in CLASS_NAMES if c not in detections]
    is_valid = "photo" in detections and detections["photo"]["conf"] > PHOTO_MIN_CONF

    return {
        "image_path": str(image_path),
        "detections": detections,
        "missing_fields": missing,
        "is_valid_kyc_input": is_valid,
    }


def validate_detection_quality(result: dict[str, Any]) -> dict[str, Any]:
    """Apply KYC business rules to a prediction result.

    Returns {"status": "OK"|"REVIEW"|"REJECTED", "warnings": list[str]}.
    """
    warnings: list[str] = []
    status = "OK"

    det = result.get("detections", {})
    if not det:
        return {"status": "REJECTED",
                "warnings": ["No detections — invalid image"]}

    # Photo is mandatory
    if "photo" not in det:
        warnings.append("CRITICAL: cardholder photo not detected")
        status = "REJECTED"
    elif det["photo"]["conf"] < PHOTO_MIN_CONF:
        warnings.append(
            f"Photo detected with low confidence ({det['photo']['conf']:.2f})"
        )
        status = "REVIEW"

    # Average confidence
    mean_conf = sum(d["conf"] for d in det.values()) / len(det)
    if mean_conf < MEAN_CONF_THRESHOLD:
        warnings.append(
            f"Low mean confidence ({mean_conf:.2f}) — document may be blurry"
        )
        if status == "OK":
            status = "REVIEW"

    # Missing fields
    n_missing = len(result.get("missing_fields", []))
    if n_missing > MAX_MISSING_FIELDS:
        warnings.append(
            f"{n_missing} fields missing — document may be truncated"
        )
        if status == "OK":
            status = "REVIEW"

    return {"status": status, "warnings": warnings}


def main() -> None:
    p = argparse.ArgumentParser(description="KYC document detection inference")
    p.add_argument("--model", type=Path, required=True, help="Path to .pt model weights")
    p.add_argument("--image", type=Path, required=True, help="Input document image")
    p.add_argument("--output", type=Path, default=Path("crops"),
                   help="Output directory for crops")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Confidence threshold")
    args = p.parse_args()

    result = predict_and_crop(args.image, args.model, args.output, args.conf)
    quality = validate_detection_quality(result)

    output = {**result, "quality": quality}
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
