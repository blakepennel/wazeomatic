"""Fine-tune YOLO11n on the merged hazard dataset.

    python tools/train.py                       # defaults: 416 px, 60 epochs, batch 32
    python tools/train.py --imgsz 320 --epochs 40 --name hazard320

Expects the dataset produced by tools/prepare_datasets.py (data yaml path below).
Checkpoints land in runs/train/<name>/weights/{best,last}.pt; export them with
tools/export_onnx.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path("F:/dev/datasets/wazeomatic_hazards/data.yaml")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--model", default="yolo11n.pt", help="starting weights (COCO pretrained)")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--name", default="hazard")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(ROOT / "runs" / "train"),
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        resume=args.resume,
        # dashcam-friendly augmentation: no vertical flips (potholes do not hang from the sky),
        # mild colour jitter for exposure differences, mosaic off for the last epochs
        fliplr=0.5, flipud=0.0, degrees=2.0, translate=0.1, scale=0.4, shear=1.0,
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.4, mosaic=1.0, close_mosaic=10, mixup=0.05,
        cos_lr=True, lr0=0.01, warmup_epochs=3, plots=True, verbose=True,
    )
    print("best weights:", ROOT / "runs" / "train" / args.name / "weights" / "best.pt")


if __name__ == "__main__":
    main()
