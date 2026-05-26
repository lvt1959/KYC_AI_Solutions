"""Convert MIDV-2020 VIA annotations to YOLO format, then train/val/test split.

Usage:
    python -m src.data_prep --input data/raw --output data/yolo
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple

from PIL import Image

from src.config import CLASS_MAPPING, CLASS_NAMES, NUM_CLASSES


# ────────────────────────────────────────────────────────────────────
# Core conversion helpers
# ────────────────────────────────────────────────────────────────────

def polygon_to_yolo_bbox(
    points_x: Iterable[float],
    points_y: Iterable[float],
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float]:
    """Convert a polygon (xs, ys) to a normalized YOLO bbox (cx, cy, w, h)."""
    px = list(points_x)
    py = list(points_y)
    x_min, x_max = min(px), max(px)
    y_min, y_max = min(py), max(py)
    cx = (x_min + x_max) / 2 / img_w
    cy = (y_min + y_max) / 2 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return (
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, w)),
        max(0.0, min(1.0, h)),
    )


def extract_label(region_attrs: dict) -> str | None:
    """Get a normalized label from a VIA region's attributes dict."""
    for key in ("field_name", "class", "label", "type"):
        if key in region_attrs:
            return str(region_attrs[key]).lower().strip()
    return None


def via_region_to_polygon(region: dict) -> Tuple[list, list] | None:
    """Extract (xs, ys) from any supported VIA shape; return None if unsupported."""
    shape = region.get("shape_attributes", {})
    name = shape.get("name")
    if name in ("polygon", "polyline"):
        return shape.get("all_points_x", []), shape.get("all_points_y", [])
    if name == "rect":
        x, y, w, h = shape["x"], shape["y"], shape["width"], shape["height"]
        return [x, x + w, x + w, x], [y, y, y + h, y + h]
    return None


def convert_via_json(
    via_json_path: Path,
    images_root: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    mapping: Dict[str, int] = CLASS_MAPPING,
) -> Tuple[int, int, int]:
    """Convert one VIA JSON file.

    Returns (n_imgs_ok, n_imgs_skipped, n_boxes).
    """
    with open(via_json_path) as f:
        via = json.load(f)
    metadata = via.get("_via_img_metadata", via)

    n_ok = n_skip = n_boxes = 0
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    for img_key, img_data in metadata.items():
        filename = img_data.get("filename")
        if not filename:
            continue
        src_img = images_root / filename
        if not src_img.exists():
            matches = list(images_root.rglob(filename))
            if not matches:
                n_skip += 1
                continue
            src_img = matches[0]

        try:
            with Image.open(src_img) as im:
                img_w, img_h = im.size
        except Exception:
            n_skip += 1
            continue

        lines = []
        for region in img_data.get("regions", []):
            label = extract_label(region.get("region_attributes", {}))
            if label not in mapping:
                continue
            poly = via_region_to_polygon(region)
            if poly is None:
                continue
            px, py = poly
            if len(px) < 2 or len(py) < 2:
                continue
            cx, cy, bw, bh = polygon_to_yolo_bbox(px, py, img_w, img_h)
            if bw <= 0 or bh <= 0:
                continue
            class_id = mapping[label]
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            n_boxes += 1

        if not lines:
            n_skip += 1
            continue

        unique_name = (
            src_img.stem + "_" + str(hash(str(src_img)) & 0xFFFFFF) + src_img.suffix
        )
        shutil.copy(src_img, out_images_dir / unique_name)
        (out_labels_dir / (Path(unique_name).stem + ".txt")).write_text("\n".join(lines))
        n_ok += 1

    return n_ok, n_skip, n_boxes


# ────────────────────────────────────────────────────────────────────
# Split
# ────────────────────────────────────────────────────────────────────

def split_dataset(
    tmp_images: Path,
    tmp_labels: Path,
    yolo_root: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, int]:
    """Move images+labels into yolo_root/{train,val,test}/{images,labels}."""
    random.seed(seed)
    image_exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
    all_imgs = sorted(p.name for p in tmp_images.iterdir() if p.suffix.lower() in image_exts)
    random.shuffle(all_imgs)

    n = len(all_imgs)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    splits = {
        "train": all_imgs[:n_train],
        "val": all_imgs[n_train:n_train + n_val],
        "test": all_imgs[n_train + n_val:],
    }

    for split, names in splits.items():
        (yolo_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for name in names:
            src_img = tmp_images / name
            src_lbl = tmp_labels / (Path(name).stem + ".txt")
            shutil.move(src_img, yolo_root / "images" / split / name)
            if src_lbl.exists():
                shutil.move(src_lbl, yolo_root / "labels" / split / src_lbl.name)

    return {k: len(v) for k, v in splits.items()}


def write_data_yaml(yolo_root: Path) -> Path:
    """Write Ultralytics data.yaml at yolo_root/data.yaml."""
    yaml_path = yolo_root / "data.yaml"
    lines = [
        "# YOLOv11 dataset config — KYC step 2.1",
        f"path: {yolo_root.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {NUM_CLASSES}",
        "names:",
    ]
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"  {i}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n")
    return yaml_path


# ────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MIDV-2020 VIA → YOLO converter")
    parser.add_argument("--input", type=Path, required=True,
                        help="Folder containing extracted MIDV-2020 (with VIA JSONs)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output YOLO dataset root")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    tmp_imgs = args.output / "_tmp" / "images"
    tmp_lbls = args.output / "_tmp" / "labels"

    ann_files = list(args.input.rglob("*.json"))
    print(f"Found {len(ann_files)} VIA JSON files in {args.input}")

    label_counter: Counter = Counter()
    total_ok = total_skip = total_boxes = 0
    for ann in ann_files:
        ok, skip, boxes = convert_via_json(
            ann, ann.parent, tmp_imgs, tmp_lbls
        )
        total_ok += ok
        total_skip += skip
        total_boxes += boxes

    print(f"\nConversion complete:")
    print(f"  {total_ok} images converted")
    print(f"  {total_skip} skipped")
    print(f"  {total_boxes} bounding boxes total")

    print("\nSplitting…")
    counts = split_dataset(tmp_imgs, tmp_lbls, args.output,
                           args.train_ratio, args.val_ratio, args.seed)
    for split, n in counts.items():
        print(f"  {split:6s}: {n}")

    # Clean up tmp dirs
    shutil.rmtree(args.output / "_tmp", ignore_errors=True)

    yaml_path = write_data_yaml(args.output)
    print(f"\ndata.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
