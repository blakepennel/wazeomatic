"""Build the image-folder dataset of the vehicle classifier: civilian / police / ambulance / fire.

    python tools/build_vehicle_crops.py                     # -> <datasets>/vehicle_crops/{train,val}/<class>/
    python tools/build_vehicle_crops.py --negatives 40000 --detector yolo11l.pt

Sources (all under the datasets root):

* ``commons_vehicles/{police,ambulance,fire}`` (tools/fetch_commons_vehicles.py): one photo per vehicle;
  ``civilian_hard`` and ``civilian_taxi`` are look-alikes (patrol-car models, tow trucks, taxis) as civilian. A
  large detector finds the vehicles; the photo is used only when one car, truck or bus clearly dominates it
  (largest box at least ``--dominant`` times the next one), so a line-up of cars cannot put a civilian car
  into the police folder.
* ``openimages/Ambulance`` (tools/fetch_openimages.py): the annotated boxes themselves.
* ``openimages/{Car,Taxi,Van}``: hand-held photos of ordinary vehicles as civilian, so that photo style alone
  says nothing about the class.
* ``bdd100k_weather`` (US dashcam frames): every vehicle the detector finds, as civilian negatives. A few of
  them are police cars; ``--exclude`` takes a list of crop names checked by hand and moved out of civilian
  (``tools/mine_vehicle_crops.py`` writes it).
* ``mined/<class>`` (optional): crops from dashcam frames confirmed by hand, the in-domain positives.
* ``mined_tracks/<class>/<track>/`` (optional, tools/mine_vehicle_tracks.py): every frame of a vehicle track
  confirmed by hand, near and far, front and rear. A track stays on one side of the split (crops ``mt_*``).

Crops are cut with ``wazeomatic.classifiers.square_crop`` (box grown 12 % per side, letterboxed), the rule the
pipeline uses at run time. Photos of a vehicle filling the frame do not look like a car 40 m ahead, so every
train positive also gets ``--degrade`` copies shrunk to 20-70 px and blown back up with blur and JPEG
artefacts. The split is by source file (a photo and its copies stay on one side).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wazeomatic.classifiers import square_crop  # noqa: E402
from wazeomatic.paths import datasets_root  # noqa: E402

VEHICLE_COCO = {2: "car", 5: "bus", 7: "truck"}
CLASSES = ("civilian", "police", "ambulance", "fire")
SOURCES = ("police", "ambulance", "fire", "civilian_hard", "civilian_taxi", "openimages", "oi_civilian", "bdd",
           "mined", "tracks")


def split_of(key: str, val_frac: float) -> str:
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val_frac else "train"


def degrade(crop: np.ndarray, rng: random.Random) -> np.ndarray:
    size = crop.shape[0]
    small = rng.randint(20, 70)
    img = cv2.resize(crop, (small, small), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        k = rng.choice([3, 5])
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k                       # horizontal motion blur
        img = cv2.filter2D(img, -1, kernel)
    q = rng.randint(25, 75)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5:                                # exposure drift
        img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.6, 1.3), beta=rng.uniform(-25, 25))
    return img


def to_xywhn(x1, y1, x2, y2, w, h):
    return ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)


class Writer:
    def __init__(self, out: Path, size: int):
        self.out, self.size, self.counts = out, size, {}

    def put(self, split: str, cls: str, name: str, img: np.ndarray) -> None:
        d = self.out / split / cls
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        self.counts[(split, cls)] = self.counts.get((split, cls), 0) + 1


def detect_vehicles(model, paths: list[Path], batch: int, conf: float):
    """Yields (path, image, [(x1, y1, x2, y2, conf)]) with vehicle boxes in pixels."""
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        imgs = [cv2.imread(str(p)) for p in chunk]
        keep = [(p, im) for p, im in zip(chunk, imgs) if im is not None]
        if not keep:
            continue
        res = model.predict([im for _, im in keep], conf=conf, classes=list(VEHICLE_COCO), verbose=False,
                            imgsz=640, half=True)
        for (p, im), r in zip(keep, res):
            b = r.boxes
            boxes = [(*xy.tolist(), float(c)) for xy, c in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy())]
            yield p, im, boxes
        if (i // batch) % 50 == 0:
            print(f"  {min(i + batch, len(paths))}/{len(paths)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", help="datasets root (default: wazeomatic/paths.py)")
    ap.add_argument("--out", help="default <datasets>/vehicle_crops")
    ap.add_argument("--detector", default="yolo11l.pt", help="Ultralytics COCO detector for the crop boxes")
    ap.add_argument("--size", type=int, default=192, help="stored crop size (training resizes)")
    ap.add_argument("--dominant", type=float, default=2.0)
    ap.add_argument("--min-frac", type=float, default=0.08, help="dominant vehicle covers at least this much")
    ap.add_argument("--negatives", type=int, default=40000, help="civilian crops from dashcam frames")
    ap.add_argument("--neg-min-px", type=int, default=28)
    ap.add_argument("--degrade", type=int, default=2, help="degraded copies per train positive")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--exclude", help="text file of civilian crop names to leave out (hand-checked police)")
    ap.add_argument("--mined-repeat", type=int, default=6, help="extra copies of each hand-confirmed train crop")
    ap.add_argument("--track-repeat", type=int, default=1, help="extra degraded copies of each train track frame")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES),
                    help="build only these sources (adds to what is already in --out)")
    args = ap.parse_args()
    root = Path(args.datasets) if args.datasets else datasets_root()
    out = Path(args.out) if args.out else root / "vehicle_crops"
    rng = random.Random(args.seed)
    from ultralytics import YOLO

    model = YOLO(args.detector)
    wr = Writer(out, args.size)
    manifest = []

    # ------------------------------------------------------------------ Commons photos
    for src in [s for s in ("police", "ambulance", "fire", "civilian_hard", "civilian_taxi") if s in args.sources]:
        cls = "civilian" if src.startswith("civilian") else src
        tag = "tx" if src == "civilian_taxi" else "cm"           # tx: the eval reports taxis on their own
        folder = root / "commons_vehicles" / src
        paths = sorted(p for p in folder.glob("*.jpg")) if folder.exists() else []
        used = 0
        print(f"[commons/{cls}] {len(paths)} photos")
        for p, im, boxes in detect_vehicles(model, paths, args.batch, 0.35):
            h, w = im.shape[:2]
            boxes.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
            if not boxes:
                continue
            a0 = (boxes[0][2] - boxes[0][0]) * (boxes[0][3] - boxes[0][1])
            a1 = (boxes[1][2] - boxes[1][0]) * (boxes[1][3] - boxes[1][1]) if len(boxes) > 1 else 0.0
            if a0 < args.min_frac * w * h or (a1 > 0 and a0 < args.dominant * a1):
                continue
            bb = to_xywhn(*boxes[0][:4], w, h)
            crop = square_crop(im, bb, size=args.size)
            split = split_of(p.stem, args.val_frac)
            wr.put(split, cls, f"{tag}_{p.stem[:80]}", crop)
            if split == "train":
                for k in range(args.degrade):
                    wr.put(split, cls, f"{tag}_{p.stem[:80]}_d{k}", degrade(crop, rng))
            manifest.append({"src": str(p), "cls": cls, "split": split})
            used += 1
        print(f"[commons/{cls}] {used} used")

    # ------------------------------------------------------------------ Open Images ambulances
    for split_dir in sorted((root / "openimages" / "Ambulance").glob("*")) if "openimages" in args.sources else []:
        bj = split_dir / "boxes.jsonl"
        if not bj.exists():
            continue
        n = 0
        for line in bj.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            im = cv2.imread(str(split_dir / rec["image"]))
            if im is None:
                continue
            for k, b in enumerate(rec["boxes"]):
                if b["group"] or b["depiction"]:
                    continue
                x1, y1, x2, y2 = b["box"]
                if (x2 - x1) * (y2 - y1) < 0.01:
                    continue
                crop = square_crop(im, ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), size=args.size)
                split = split_of(rec["image"], args.val_frac)
                name = f"oi_{Path(rec['image']).stem}_{k}"
                wr.put(split, "ambulance", name, crop)
                if split == "train":
                    for j in range(args.degrade):
                        wr.put(split, "ambulance", f"{name}_d{j}", degrade(crop, rng))
                n += 1
        print(f"[openimages/{split_dir.name}] {n} ambulance boxes")

    # ------------------------------------------------------------------ Open Images civilian photos
    # Hand-held photos of ordinary cars, taxis and vans. Without them nearly every photo-style crop is an
    # emergency vehicle and every dashcam crop civilian, and the classifier learns the camera, not the livery.
    for oi_cls in ("Car", "Taxi", "Van") if "oi_civilian" in args.sources else ():
        for split_dir in sorted((root / "openimages" / oi_cls).glob("*")):
            bj = split_dir / "boxes.jsonl"
            if not bj.exists():
                continue
            n = 0
            for line in bj.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                big = [b for b in rec["boxes"] if not (b["group"] or b["depiction"])
                       and (b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1]) >= 0.02]
                if not big:
                    continue
                im = cv2.imread(str(split_dir / rec["image"]))
                if im is None:
                    continue
                big.sort(key=lambda b: -(b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1]))
                split = split_of(rec["image"], args.val_frac)
                for k, b in enumerate(big[:2]):
                    x1, y1, x2, y2 = b["box"]
                    crop = square_crop(im, ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), size=args.size)
                    name = f"oc_{Path(rec['image']).stem}_{k}"
                    wr.put(split, "civilian", name, crop)
                    if split == "train" and k == 0:
                        wr.put(split, "civilian", f"{name}_d0", degrade(crop, rng))
                    n += 1
            print(f"[openimages/{oi_cls}/{split_dir.name}] {n} civilian boxes")

    # ------------------------------------------------------------------ dashcam negatives
    # exclude.txt lists hand-confirmed crops as bdd_<frame>_<k>; the k of the miner is not the k used here (the
    # boxes are shuffled), so a whole frame with a confirmed police car / ambulance gives no civilian crops
    exclude = set()
    if args.exclude and Path(args.exclude).exists():
        exclude = {ln.strip().rsplit("_", 1)[0] for ln in Path(args.exclude).read_text(encoding="utf-8").splitlines()
                   if ln.strip()}
    bdd = sorted((root / "bdd100k_weather" / "data" / "images").rglob("*.jpg")) if "bdd" in args.sources else []
    rng.shuffle(bdd)
    taken = 0
    print(f"[bdd] {len(bdd)} frames available")
    per_frame = 4
    for p, im, boxes in detect_vehicles(model, bdd, args.batch, 0.4):
        if taken >= args.negatives:
            break
        if f"bdd_{p.stem}" in exclude:
            continue
        h, w = im.shape[:2]
        boxes = [b for b in boxes if min(b[2] - b[0], b[3] - b[1]) >= args.neg_min_px]
        rng.shuffle(boxes)
        for k, b in enumerate(boxes[:per_frame]):
            name = f"bdd_{p.stem}_{k}"
            crop = square_crop(im, to_xywhn(*b[:4], w, h), size=args.size)
            wr.put(split_of(p.stem, args.val_frac), "civilian", name, crop)
            taken += 1
    print(f"[bdd] {taken} civilian crops")

    # ------------------------------------------------------------------ hand-confirmed dashcam crops
    mined = root / "mined_vehicles"
    for cls in CLASSES if "mined" in args.sources else ():
        for p in sorted((mined / cls).glob("*.jpg")) if (mined / cls).exists() else []:
            im = cv2.imread(str(p))
            if im is None:
                continue
            im = cv2.resize(im, (args.size, args.size))
            split = split_of(p.stem, args.val_frac)
            wr.put(split, cls, f"mn_{p.stem}", im)
            if split == "train":
                # few but in-domain: repeat them (degraded) so they weigh against thousands of photos
                for k in range(args.mined_repeat):
                    wr.put(split, cls, f"mn_{p.stem}_d{k}", degrade(im, rng) if k % 2 else cv2.flip(im, 1))
    # ------------------------------------------------------------------ hand-confirmed dashcam tracks
    tracks = root / "mined_tracks"
    for cls in CLASSES if "tracks" in args.sources else ():
        for d in sorted(p for p in (tracks / cls).iterdir() if p.is_dir()) if (tracks / cls).exists() else []:
            split = split_of(d.name, args.val_frac)
            for p in sorted(d.glob("*.jpg")):
                im = cv2.imread(str(p))
                if im is None:
                    continue
                im = cv2.resize(im, (args.size, args.size))
                wr.put(split, cls, f"mt_{d.name}_{p.stem}", im)
                if split == "train":
                    for k in range(args.track_repeat):
                        wr.put(split, cls, f"mt_{d.name}_{p.stem}_d{k}", degrade(im, rng))
    if "mined" in args.sources and exclude:
        # BDD civilian crops already written from frames whose vehicle was confirmed by hand as police etc.
        gone = 0
        for split in ("train", "val"):
            for f in (out / split / "civilian").glob("bdd_*.jpg"):
                if f.stem.rsplit("_", 1)[0] in exclude:
                    f.unlink()
                    gone += 1
        print(f"[mined] removed {gone} civilian crops from hand-confirmed frames")
    counts = {f"{s}/{c}": len(list((out / s / c).glob("*.jpg"))) for s in ("train", "val") for c in CLASSES}
    (out / "counts.json").write_text(json.dumps(counts, indent=1), encoding="utf-8")
    print(json.dumps({f"{s}/{c}": n for (s, c), n in sorted(wr.counts.items())}, indent=1))


if __name__ == "__main__":
    main()
