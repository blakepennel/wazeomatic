"""Train an Ultralytics ``-cls`` model on an image-folder dataset and export it to ONNX for the pipeline.

    python tools/train_classifier.py vehicle      # <datasets>/vehicle_crops  -> models/vehicle_cls.onnx
    python tools/train_classifier.py weather      # <datasets>/scene_weather  -> models/weather_cls.onnx
    python tools/train_classifier.py surface      # <datasets>/scene_surface  -> models/surface_cls.onnx
    python tools/train_classifier.py vehicle --model yolo11s-cls.pt --imgsz 160 --epochs 40 --name vehicle_s

The ONNX file takes ``[N, 3, imgsz, imgsz]`` RGB in [0, 1] (dynamic batch) and returns class probabilities;
the class names go to ``<model>.labels.txt`` in output order (wazeomatic/classifiers.py reads them). Without
``--ship`` the export is written as ``models/<name>.onnx`` and the model the pipeline loads is left alone.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wazeomatic.paths import datasets_root  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PRESETS = {
    "vehicle": {"data": "vehicle_crops", "imgsz": 160, "ship": "vehicle_cls"},
    "weather": {"data": "scene_weather", "imgsz": 224, "ship": "weather_cls"},
    "surface": {"data": "scene_surface", "imgsz": 160, "ship": "surface_cls"},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=list(PRESETS))
    ap.add_argument("--data", help="image-folder root with train/ and val/ (default from the preset)")
    ap.add_argument("--model", default="yolo11n-cls.pt")
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", help="run and export name (default <task>_<model stem>)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--ship", action="store_true", help="also copy the export to the model the pipeline loads")
    ap.add_argument("--export-only", help="skip training, export this .pt")
    args = ap.parse_args()
    pre = PRESETS[args.task]
    data = Path(args.data) if args.data else datasets_root() / pre["data"]
    imgsz = args.imgsz or pre["imgsz"]
    name = args.name or f"{args.task}_{Path(args.model).stem.replace('-cls', '')}"
    from ultralytics import YOLO

    if args.export_only:
        best = Path(args.export_only)
    else:
        model = YOLO(args.model)
        model.train(data=str(data), imgsz=imgsz, epochs=args.epochs, batch=args.batch, workers=args.workers,
                    device=args.device, project=str(ROOT / "runs" / "classify"), name=name, exist_ok=True,
                    patience=10, cos_lr=True, fliplr=0.5, hsv_h=0.01, hsv_s=0.4, hsv_v=0.4, erasing=0.2,
                    plots=False, verbose=False)
        best = ROOT / "runs" / "classify" / name / "weights" / "best.pt"
    m = YOLO(str(best))
    onnx = Path(m.export(format="onnx", imgsz=imgsz, dynamic=True, simplify=True, opset=17))
    names = [m.names[i] for i in sorted(m.names)]
    dest = ROOT / "models" / f"{name}.onnx"
    shutil.copy2(onnx, dest)
    dest.with_suffix(".labels.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"exported {dest} ({names}, {imgsz}px)")
    if args.ship:
        ship = ROOT / "models" / f"{pre['ship']}.onnx"
        shutil.copy2(dest, ship)
        shutil.copy2(dest.with_suffix(".labels.txt"), ship.with_suffix(".labels.txt"))
        print(f"shipped as {ship}")


if __name__ == "__main__":
    main()
