"""Train YOLO11n on GenMRP+MIDV-2020 dataset.

Usage:
    python -m src.train --data data/dataset/data.yaml --epochs 40 --model yolo11n
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import DEFAULT_TRAIN_KWARGS, RUNS_DIR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO11 for KYC document detection")
    p.add_argument("--data", type=Path, required=True,
                   help="Path to data.yaml")
    p.add_argument("--model", default="yolo11n",
                   choices=["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"],
                   help="YOLO11 variant (default: yolo11n)")
    p.add_argument("--epochs", type=int, default=DEFAULT_TRAIN_KWARGS["epochs"])
    p.add_argument("--imgsz", type=int, default=DEFAULT_TRAIN_KWARGS["imgsz"])
    p.add_argument("--batch", type=int, default=DEFAULT_TRAIN_KWARGS["batch"])
    p.add_argument("--name", default="kyc_2_1_midv2020",
                   help="Run name (subfolder under runs/)")
    p.add_argument("--project", type=Path, default=RUNS_DIR / "train")
    p.add_argument("--resume", action="store_true",
                   help="Resume from last checkpoint")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Lazy import to keep --help fast
    from ultralytics import YOLO

    weights = f"{args.model}.pt"
    print(f"Loading {weights} (COCO-pretrained)...")
    model = YOLO(weights)

    train_kwargs = {**DEFAULT_TRAIN_KWARGS,
                    "epochs": args.epochs,
                    "imgsz": args.imgsz,
                    "batch": args.batch,
                    "project": str(args.project),
                    "name": args.name,
                    "exist_ok": True,
                    "resume": args.resume,
                    "data": str(args.data)}

    print(f"\nTraining on {args.data}")
    print(f"  epochs={args.epochs} imgsz={args.imgsz} batch={args.batch}")
    print(f"  -> {args.project / args.name}\n")

    results = model.train(**train_kwargs)

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best}")

    # Final val on test split
    print("\nEvaluating on test split...")
    test_metrics = model.val(data=str(args.data), split="test", plots=True)
    print(f"\nTest mAP@50    : {test_metrics.box.map50:.4f}")
    print(f"Test mAP@50-95 : {test_metrics.box.map:.4f}")
    print(f"Test precision : {test_metrics.box.mp:.4f}")
    print(f"Test recall    : {test_metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
