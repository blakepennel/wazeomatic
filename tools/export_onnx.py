"""Export a YOLO checkpoint to a static-shape ONNX file for ONNX Runtime.

Examples
    python tools/export_onnx.py --coco                       # stock yolo11n, 80 classes
    python tools/export_onnx.py --weights runs/train/hazard_v3n/weights/best.pt --name hazard_yolo11n_v3
    python tools/export_onnx.py --weights runs/train/hazard_v3n/weights/best.pt --name hazard_yolo11n_v3 --ship

Without --ship only models/<name>.onnx is written. --ship also copies it to models/hazard_yolo11n.onnx,
the stable name the desktop pipeline and the Android build load, and refuses to do that for a model
that is not a nano unless --force is given.
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


SHIPPED_HAZARD = "hazard_yolo11n.onnx"


def model_arch(model) -> str:
    """'yolo11n', 'yolo11s', ... from the checkpoint's own config, 'yolo' if it cannot be told."""
    yaml_cfg = getattr(getattr(model, "model", None), "yaml", None) or {}
    scale = str(yaml_cfg.get("scale") or "")
    stem = Path(str(yaml_cfg.get("yaml_file") or "")).stem      # e.g. yolo11s or yolo11
    if stem and scale and not stem.endswith(scale):
        return stem + scale
    return stem or "yolo"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", help="path to .pt checkpoint")
    ap.add_argument("--coco", action="store_true", help="export the stock COCO yolo11n instead")
    ap.add_argument("--imgsz", type=int, help="input size (default: 640 for a hazard model, 416 for --coco)")
    ap.add_argument("--name", help="output base name, e.g. hazard_yolo11s_v3 (default: hazard_<arch>_<imgsz>)")
    ap.add_argument("--ship", action="store_true",
                    help="also make this the shipped model (models/hazard_yolo11n.onnx, bundled by the Android build)")
    ap.add_argument("--force", action="store_true", help="with --ship: allow a model that is not a YOLO11n")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--out", help="output .onnx path (default models/<name>.onnx)")
    ap.add_argument("--half", action="store_true", help="fp16 weights (CPU EPs usually prefer fp32)")
    ap.add_argument("--no-stable", action="store_true",
                    help="--coco only: do not refresh models/coco_yolo11n.onnx")
    args = ap.parse_args()

    from ultralytics import YOLO

    if args.coco:
        weights = "yolo11n.pt"
        labels = COCO_LABELS
    else:
        if not args.weights:
            ap.error("--weights or --coco required")
        weights = args.weights
        labels = None
    imgsz = args.imgsz or (416 if args.coco else 640)

    model = YOLO(weights)
    if labels is None:
        labels = [model.names[i] for i in range(len(model.names))]
    arch = model_arch(model)

    if args.ship and not args.coco and arch != "yolo11n" and not args.force:
        raise SystemExit(f"--ship would replace the shipped nano model with a {arch}; the phone build bundles that "
                         f"file. Pass --force if that is what you want.")
    exported = model.export(format="onnx", imgsz=imgsz, opset=args.opset, simplify=True,
                            dynamic=False, half=args.half, nms=False)
    MODELS.mkdir(exist_ok=True)
    if args.out:
        out = Path(args.out)
    elif args.coco:
        out = MODELS / f"coco_yolo11n_{imgsz}.onnx"
    else:
        out = MODELS / f"{args.name or f'hazard_{arch}_{imgsz}'}.onnx"
    shutil.move(str(exported), str(out))
    (out.with_suffix(".labels.txt")).write_text("\n".join(labels) + "\n", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), {len(labels)} labels, {arch}, {imgsz} px")
    stable = None
    if args.coco and not args.no_stable:
        stable = MODELS / "coco_yolo11n.onnx"          # the stock detector the pipeline and the app load
    elif args.ship and not args.coco:
        stable = MODELS / SHIPPED_HAZARD
    if stable is not None and stable.resolve() != out.resolve():
        shutil.copyfile(out, stable)
        shutil.copyfile(out.with_suffix(".labels.txt"), stable.with_suffix(".labels.txt"))
        print(f"refreshed {stable}")


if __name__ == "__main__":
    main()
