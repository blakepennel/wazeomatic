"""Export a YOLO checkpoint to a static-shape ONNX file for ONNX Runtime.

Examples
    python tools/export_onnx.py --coco                       # stock yolo11n, 80 classes
    python tools/export_onnx.py --weights runs/train/hazard/weights/best.pt --imgsz 416
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", help="path to .pt checkpoint")
    ap.add_argument("--coco", action="store_true", help="export the stock COCO yolo11n instead")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--out", help="output .onnx path (default models/<name>.onnx)")
    ap.add_argument("--half", action="store_true", help="fp16 weights (CPU EPs usually prefer fp32)")
    args = ap.parse_args()

    from ultralytics import YOLO

    if args.coco:
        weights = "yolo11n.pt"
        name = "coco_yolo11n"
        labels = COCO_LABELS
    else:
        if not args.weights:
            ap.error("--weights or --coco required")
        weights = args.weights
        name = "hazard_yolo11n"
        labels = None

    model = YOLO(weights)
    if labels is None:
        labels = [model.names[i] for i in range(len(model.names))]

    exported = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, simplify=True,
                            dynamic=False, half=args.half, nms=False)
    MODELS.mkdir(exist_ok=True)
    out = Path(args.out) if args.out else MODELS / f"{name}_{args.imgsz}.onnx"
    shutil.move(str(exported), str(out))
    (out.with_suffix(".labels.txt")).write_text("\n".join(labels) + "\n", encoding="utf-8")
    # convenience copies with stable names that the Android app and configs refer to
    stable = MODELS / f"{name}.onnx"
    shutil.copyfile(out, stable)
    shutil.copyfile(out.with_suffix(".labels.txt"), stable.with_suffix(".labels.txt"))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) and {stable}")
    print(f"labels: {len(labels)} -> {stable.with_suffix('.labels.txt')}")


if __name__ == "__main__":
    main()
