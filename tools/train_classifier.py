"""Train an Ultralytics ``-cls`` model on an image-folder dataset and export it to ONNX for the pipeline.

    python tools/train_classifier.py vehicle      # <datasets>/vehicle_crops  -> models/vehicle_cls.onnx
    python tools/train_classifier.py weather      # <datasets>/scene_weather  -> models/weather_cls.onnx
    python tools/train_classifier.py surface      # <datasets>/scene_surface  -> models/surface_cls.onnx
    python tools/train_classifier.py vehicle --model yolo11s-cls.pt --imgsz 160 --epochs 40 --name vehicle_s
    python tools/train_classifier.py vehicle --res-jitter 0.5 --name vehicle_v7    # half the crops also soft
    python tools/train_classifier.py vehicle --occlude 0.5 --name vehicle_v9       # half the crops also partly hidden

``--res-jitter P`` trains on a copy of the set (``<data>_rj``, hard links to the originals) where a share P of
the training crops gets a second, softened version: scaled down to 20-70 % and back up, then saved at JPEG
quality 40-90. A dashcam shows the same police car as a sharp 300 px box close by and as a soft 40 px box
far away; the classifier should not depend on which (docs/waze-coverage.md, the missed police cars).

``--occlude P`` does the same with partly hidden copies (``<data>_oc``, see ``occlude``): a police car parked
behind a hedge, a barrier or another car shows only part of its livery, and a classifier trained on whole cars
calls almost none of those police (docs/waze-coverage.md, partly hidden police).

The ONNX file takes ``[N, 3, imgsz, imgsz]`` RGB in [0, 1] (dynamic batch) and returns class probabilities;
the class names go to ``<model>.labels.txt`` in output order (wazeomatic/classifiers.py reads them). Without
``--ship`` the export is written as ``models/<name>.onnx`` and the model the pipeline loads is left alone.
"""
from __future__ import annotations

import argparse
import os
import random
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


def res_jitter_copy(data: Path, share: float, seed: int = 0) -> Path:
    """``<data>_rj``: train/ and val/ hard-linked from ``data``, plus a softened copy of ``share`` of the
    training images (``*_rj.jpg``). Made once; delete the folder to rebuild it."""
    import cv2

    out = data.with_name(data.name + "_rj")
    if (out / "done.txt").exists():
        return out
    rng = random.Random(seed)
    n = 0
    for split in ("train", "val"):
        for src in sorted((data / split).rglob("*.jpg")):
            dst = out / src.relative_to(data)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
            if split == "train" and rng.random() < share:
                im = cv2.imread(str(src))
                if im is None:
                    continue
                h, w = im.shape[:2]
                s = rng.uniform(0.2, 0.7)
                small = cv2.resize(im, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
                soft = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(str(dst.with_name(dst.stem + "_rj.jpg")), soft,
                            [cv2.IMWRITE_JPEG_QUALITY, rng.randint(40, 90)])
                n += 1
    (out / "done.txt").write_text(f"{n} softened copies, share {share}, seed {seed}\n", encoding="utf-8")
    print(f"res-jitter set {out}: {n} softened copies")
    return out


def _content_box(im, fill: int = 114, tol: float = 3.0) -> tuple[int, int, int, int]:
    """x1, y1, x2, y2 of the part of a letterboxed crop (square_crop) that is not the grey fill."""
    import numpy as np

    d = np.abs(im.astype(np.int16) - fill).mean(axis=2) > tol
    rows, cols = np.where(d.any(axis=1))[0], np.where(d.any(axis=0))[0]
    if len(rows) < 8 or len(cols) < 8:
        return 0, 0, im.shape[1], im.shape[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def _letterbox(part, size: int, fill: int = 114):
    import cv2
    import numpy as np

    h, w = part.shape[:2]
    s = size / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    r = cv2.resize(part, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    out = np.full((size, size, 3), fill, np.uint8)
    y, x = (size - nh) // 2, (size - nw) // 2
    out[y:y + nh, x:x + nw] = r
    return out


def occlude(im, rng: random.Random, cars: list, scenes: list):
    """A copy of a letterboxed vehicle crop with part of the vehicle hidden, one of three ways:

    * ``car``: a band (25-50 % of the width, left or right) covered by part of another crop's vehicle, as when
      the car stands behind one in the next lane;
    * ``scene``: an irregular patch (blobs rising from the bottom or one side, 20-45 % of the vehicle) filled with
      road-side texture cut from a road-scene frame, as behind a hedge, a barrier or a parked trailer;
    * ``cut``: only the top 50-75 % or one side 55-80 % of the vehicle, letterboxed again: what the detector's
      box holds when it boxes only the visible part.
    """
    import cv2
    import numpy as np

    size = im.shape[0]
    x1, y1, x2, y2 = _content_box(im)
    w, h = x2 - x1, y2 - y1
    kind = rng.choice(("car", "scene", "cut"))
    out = im.copy()
    if kind == "cut":
        if rng.random() < 0.5:
            part = im[y1:y1 + max(4, int(h * rng.uniform(0.5, 0.75))), x1:x2]
        else:
            keep = max(4, int(w * rng.uniform(0.55, 0.8)))
            part = im[y1:y2, x1:x1 + keep] if rng.random() < 0.5 else im[y1:y2, x2 - keep:x2]
        return _letterbox(part, size), kind
    if kind == "car" and cars:
        o = cars[rng.randrange(len(cars))]
        ox1, oy1, ox2, oy2 = _content_box(o)
        band = max(4, int(w * rng.uniform(0.25, 0.5)))
        patch = cv2.resize(o[oy1:oy2, ox1:ox2], (band, h), interpolation=cv2.INTER_AREA)
        if rng.random() < 0.5:
            out[y1:y2, x1:x1 + band] = patch
        else:
            out[y1:y2, x2 - band:x2] = patch
        return out, kind
    if not scenes:
        return out, "none"
    sc = scenes[rng.randrange(len(scenes))]
    sh, sw = sc.shape[:2]
    pw, ph = min(sw, max(w, 16) * rng.randint(1, 3)), min(sh, max(h, 16) * rng.randint(1, 3))
    px, py = rng.randrange(0, sw - pw + 1), rng.randrange(sh // 3, sh - ph + 1) if sh - ph + 1 > sh // 3 else 0
    tex = cv2.resize(sc[py:py + ph, px:px + pw], (w, h), interpolation=cv2.INTER_AREA)
    mask = np.zeros((h, w), np.uint8)
    frac = rng.uniform(0.2, 0.45)
    side = rng.choice(("bottom", "left", "right"))
    for _ in range(rng.randint(8, 24)):
        rx, ry = max(2, int(w * rng.uniform(0.06, 0.2))), max(2, int(h * rng.uniform(0.08, 0.25)))
        if side == "bottom":
            cx, cy = rng.randrange(w), int(h * (1 - rng.uniform(0, frac * 1.2)))
        else:
            d = int(w * rng.uniform(0, frac * 1.2))
            cx, cy = (d if side == "left" else w - 1 - d), rng.randrange(h)
        cv2.ellipse(mask, (cx, cy), (rx, ry), rng.uniform(0, 180), 0, 360, 255, -1)
    if side == "bottom":
        mask[int(h * (1 - frac * 0.7)):] = 255
    elif side == "left":
        mask[:, :int(w * frac * 0.7)] = 255
    else:
        mask[:, w - int(w * frac * 0.7):] = 255
    region = out[y1:y2, x1:x2]
    region[mask > 0] = tex[mask > 0]
    return out, kind


def occlude_copy(data: Path, share: float, scenes_dir: Path, seed: int = 0) -> Path:
    """``<data>_oc``: train/ and val/ hard-linked from ``data``, plus a partly hidden copy (``*_oc.jpg``,
    ``occlude``) of ``share`` of the training crops of every class. The texture of the ``scene`` kind comes from
    the first 2,000 frames of ``scenes_dir`` (road scenes); the ``car`` kind takes its cover from the
    civilian training crops. Made once; delete the folder to rebuild it."""
    import cv2

    out = data.with_name(data.name + "_oc")
    if (out / "done.txt").exists():
        return out
    rng = random.Random(seed)
    scenes = []
    for e in os.scandir(scenes_dir):
        if e.name.lower().endswith((".jpg", ".png")):
            im = cv2.imread(e.path)
            if im is not None:
                scenes.append(cv2.resize(im, (640, 360), interpolation=cv2.INTER_AREA))
            if len(scenes) >= 2000:
                break
    civ = [e.path for e in os.scandir(data / "train" / "civilian") if e.name.endswith(".jpg")]
    rng.shuffle(civ)
    cars = [im for im in (cv2.imread(p) for p in civ[:3000]) if im is not None]
    counts: dict[str, int] = {}
    for split in ("train", "val"):
        for cls_dir in sorted(p for p in (data / split).iterdir() if p.is_dir()):
            for e in sorted(os.scandir(cls_dir), key=lambda e: e.name):
                if not e.name.endswith(".jpg"):
                    continue
                dst = out / split / cls_dir.name / e.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    try:
                        os.link(e.path, dst)
                    except OSError:
                        shutil.copy2(e.path, dst)
                if split == "train" and rng.random() < share:
                    im = cv2.imread(e.path)
                    if im is None or im.shape[0] != im.shape[1]:
                        continue
                    occ, kind = occlude(im, rng, cars, scenes)
                    cv2.imwrite(str(dst.with_name(dst.stem + "_oc.jpg")), occ, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    counts[kind] = counts.get(kind, 0) + 1
    (out / "done.txt").write_text(f"partly hidden copies {counts}, share {share}, seed {seed}, "
                                  f"{len(scenes)} scene frames from {scenes_dir}\n", encoding="utf-8")
    print(f"occlusion set {out}: {counts}")
    return out


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
    ap.add_argument("--res-jitter", type=float, default=0.0,
                    help="add a softened copy of this share of the training crops (see the module docstring)")
    ap.add_argument("--occlude", type=float, default=0.0,
                    help="add a partly hidden copy of this share of the training crops (see occlude)")
    ap.add_argument("--occlude-scenes", help="folder of road-scene frames for the hedge / barrier texture "
                    "(default <datasets>/scene_weather/train/clear)")
    args = ap.parse_args()
    pre = PRESETS[args.task]
    data = Path(args.data) if args.data else datasets_root() / pre["data"]
    imgsz = args.imgsz or pre["imgsz"]
    if args.res_jitter > 0 and not args.export_only:
        data = res_jitter_copy(data, args.res_jitter)
    if args.occlude > 0 and not args.export_only:
        scenes = Path(args.occlude_scenes) if args.occlude_scenes else datasets_root() / "scene_weather" / "train" / "clear"
        data = occlude_copy(data, args.occlude, scenes)
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
