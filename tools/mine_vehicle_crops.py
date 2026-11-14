"""Find police cars, ambulances and fire engines in dashcam frames for review by hand.

    python tools/mine_vehicle_crops.py scan --model models/vehicle_n.onnx        # score every BDD100K vehicle
    python tools/mine_vehicle_crops.py rescore --model models/vehicle_v2.onnx      # re-rank with a newer model
    python tools/mine_vehicle_crops.py sheets --cls police --top 600              # numbered contact sheets
    python tools/mine_vehicle_crops.py label --cls police --yes 3,17,40-44 --sheet-range 0-599
    python tools/mine_vehicle_crops.py scan --clips <datasets>/argoverse2/clips --clip-half 0 --work <datasets>/mining_argo

The first classifier is trained on photos; the frames it will see are dashcam frames. ``scan`` runs the
detector and the classifier over the BDD100K frames and keeps the best-scoring crops per class. ``sheets``
draws them as numbered grids. A person looks at the grids and ``label`` files the confirmed ones under
``<datasets>/mined_vehicles/<cls>`` (in-domain positives) and the rejected ones under
``mined_vehicles/civilian`` (hard negatives: taxis, security cars, white SUVs with roof racks). It also writes
``exclude.txt`` with the confirmed crops so ``build_vehicle_crops.py --exclude`` keeps them out of civilian.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wazeomatic.classifiers import OnnxClassifier, square_crop  # noqa: E402
from wazeomatic.paths import datasets_root  # noqa: E402

VEHICLE_COCO = {2: "car", 5: "bus", 7: "truck"}


def parse_ranges(spec: str) -> set[int]:
    out: set[int] = set()
    for part in filter(None, (p.strip() for p in spec.split(","))):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def clip_half(stem: str) -> int:
    """0 or 1 by a hash of the clip name: mine one half of a clip folder, evaluate on the other."""
    return int(hashlib.md5(stem.encode("utf-8")).hexdigest()[:8], 16) % 2


def clip_frames(folder: Path, every: int, half: int) -> list[str]:
    """Frame references ``<clip path>#<index>`` for every ``every``-th frame of the clips in one half."""
    out = []
    for clip in sorted(folder.glob("*.mp4")):
        if clip_half(clip.stem) != half:
            continue
        n = int(cv2.VideoCapture(str(clip)).get(cv2.CAP_PROP_FRAME_COUNT))
        out += [f"{clip}#{i}" for i in range(0, n, every)]
    return out


def load_frame(ref) -> np.ndarray | None:
    ref = str(ref)
    if "#" not in ref:
        return cv2.imread(ref)
    path, idx = ref.rsplit("#", 1)
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, im = cap.read()
    cap.release()
    return im if ok else None


def scan(args, root: Path, work: Path) -> None:
    from ultralytics import YOLO

    det = YOLO(args.detector)
    clf = OnnxClassifier(args.model, threads=4)
    if args.clips:
        frames = clip_frames(Path(args.clips), args.every, args.clip_half)
    else:
        frames = sorted((root / "bdd100k_weather" / "data" / "images").rglob("*.jpg"))
    if args.limit:
        frames = frames[: args.limit]
    keep: dict[str, list] = {c: [] for c in clf.labels if c != "civilian"}
    crops_dir = work / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(frames), args.batch):
        chunk = frames[i:i + args.batch]
        imgs = [load_frame(p) for p in chunk]
        res = det.predict([im for im in imgs if im is not None], conf=0.4, classes=list(VEHICLE_COCO),
                          verbose=False, imgsz=640, half=True)
        for p, im, r in zip([p for p, im in zip(chunk, imgs) if im is not None], [im for im in imgs if im is not None], res):
            h, w = im.shape[:2]
            boxes = [b for b in r.boxes.xyxy.cpu().numpy() if min(b[2] - b[0], b[3] - b[1]) >= args.min_px]
            if not boxes:
                continue
            crops = [square_crop(im, ((b[0] + b[2]) / 2 / w, (b[1] + b[3]) / 2 / h, (b[2] - b[0]) / w,
                                      (b[3] - b[1]) / h), size=clf.size) for b in boxes]
            probs = clf.predict(crops)
            for k, (b, pr) in enumerate(zip(boxes, probs)):
                for c in keep:
                    s = float(pr[clf.labels.index(c)])
                    if s >= args.min_score:
                        ref = str(p)
                        stem = (Path(ref.rsplit("#", 1)[0]).stem + "_f" + ref.rsplit("#", 1)[1]) if "#" in ref                             else Path(ref).stem
                        name = f"{'av' if args.clips else 'bdd'}_{stem}_{k}"
                        big = square_crop(im, ((b[0] + b[2]) / 2 / w, (b[1] + b[3]) / 2 / h, (b[2] - b[0]) / w,
                                               (b[3] - b[1]) / h), size=192)
                        cv2.imwrite(str(crops_dir / f"{name}.jpg"), big)
                        keep[c].append({"name": name, "score": s, "frame": str(p),
                                        "box": [float(v) for v in b]})
        if (i // args.batch) % 100 == 0:
            print(f"  {i + len(chunk)}/{len(frames)} " + " ".join(f"{c}:{len(v)}" for c, v in keep.items()), flush=True)
    for c, v in keep.items():
        v.sort(key=lambda x: -x["score"])
        (work / f"{c}.json").write_text(json.dumps(v, indent=0), encoding="utf-8")
        print(f"[{c}] {len(v)} crops at or above {args.min_score}")


def rescore(args, work: Path) -> None:
    """Score the saved crops again with a newer classifier and re-sort every class list by it."""
    clf = OnnxClassifier(args.model, threads=8)
    for c in [c for c in clf.labels if c != "civilian"]:
        f = work / f"{c}.json"
        if not f.exists():
            continue
        items = json.loads(f.read_text(encoding="utf-8"))
        for i in range(0, len(items), 128):
            chunk = items[i:i + 128]
            ims = [cv2.imread(str(work / "crops" / f"{it['name']}.jpg")) for it in chunk]
            ok = [(it, im) for it, im in zip(chunk, ims) if im is not None]
            probs = clf.predict([cv2.resize(im, (clf.size, clf.size), interpolation=cv2.INTER_AREA) for _, im in ok])
            for (it, _), pr in zip(ok, probs):
                it["score"] = float(pr[clf.labels.index(c)])
        items.sort(key=lambda x: -x["score"])
        f.write_text(json.dumps(items, indent=0), encoding="utf-8")
        print(f"[{c}] {len(items)} crops rescored; {sum(it['score'] >= 0.5 for it in items)} at or above 0.5")


def sheets(args, work: Path) -> None:
    items = json.loads((work / f"{args.cls}.json").read_text(encoding="utf-8"))[: args.top]
    cols, rows, cell = 8, 6, 160
    out = work / "sheets"
    out.mkdir(exist_ok=True)
    for s in range(0, len(items), cols * rows):
        sheet = np.full((rows * (cell + 18), cols * cell, 3), 255, np.uint8)
        for j, it in enumerate(items[s:s + cols * rows]):
            im = cv2.imread(str(work / "crops" / f"{it['name']}.jpg"))
            if im is None:
                continue
            r, c = divmod(j, cols)
            y, x = r * (cell + 18), c * cell
            sheet[y:y + cell, x:x + cell] = cv2.resize(im, (cell, cell))
            cv2.putText(sheet, f"{s + j} {it['score']:.2f}", (x + 3, y + cell + 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 0), 1)
        path = out / f"{args.cls}_{s:04d}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(path)


def label(args, root: Path, work: Path) -> None:
    items = json.loads((work / f"{args.cls}.json").read_text(encoding="utf-8"))
    yes = parse_ranges(args.yes or "")
    skip = parse_ranges(args.skip or "")
    lo, hi = (int(v) for v in args.sheet_range.split("-"))
    dst_yes = root / "mined_vehicles" / args.cls
    dst_no = root / "mined_vehicles" / "civilian"
    dst_yes.mkdir(parents=True, exist_ok=True)
    dst_no.mkdir(parents=True, exist_ok=True)
    excl = root / "mined_vehicles" / "exclude.txt"
    names = set(excl.read_text(encoding="utf-8").split()) if excl.exists() else set()
    ny = nn = 0
    for i in range(lo, min(hi, len(items) - 1) + 1):
        it = items[i]
        src = work / "crops" / f"{it['name']}.jpg"
        if not src.exists() or i in skip:
            continue
        if i in yes:
            (dst_yes / src.name).write_bytes(src.read_bytes())
            names.add(it["name"])
            ny += 1
        elif not args.only_yes:
            (dst_no / src.name).write_bytes(src.read_bytes())
            nn += 1
    excl.write_text("\n".join(sorted(names)) + "\n", encoding="utf-8")
    print(f"{ny} filed as {args.cls}, {nn} as civilian hard negatives; exclude list has {len(names)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["scan", "rescore", "sheets", "label"])
    ap.add_argument("--datasets")
    ap.add_argument("--work", help="default <datasets>/mining")
    ap.add_argument("--model", default="models/vehicle_cls.onnx")
    ap.add_argument("--detector", default="yolo11l.pt")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--clips", help="scan: a folder of .mp4 clips instead of the BDD100K frames")
    ap.add_argument("--every", type=int, default=5, help="scan --clips: every n-th frame")
    ap.add_argument("--clip-half", type=int, default=0, choices=[0, 1],
                    help="scan --clips: which half of the clips (by name hash) to mine; keep the other for testing")
    ap.add_argument("--min-px", type=int, default=28)
    ap.add_argument("--min-score", type=float, default=0.15)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--cls", default="police")
    ap.add_argument("--top", type=int, default=480)
    ap.add_argument("--yes", help="indices that ARE the class, e.g. 3,17,40-44")
    ap.add_argument("--sheet-range", default="0-479", help="indices reviewed; the rest of them become negatives")
    ap.add_argument("--only-yes", action="store_true", help="file only the confirmed ones")
    ap.add_argument("--skip", help="indices too unclear to file either way")
    args = ap.parse_args()
    root = Path(args.datasets) if args.datasets else datasets_root()
    work = Path(args.work) if args.work else root / "mining"
    work.mkdir(parents=True, exist_ok=True)
    {"scan": lambda: scan(args, root, work), "rescore": lambda: rescore(args, work),
     "sheets": lambda: sheets(args, work),
     "label": lambda: label(args, root, work)}[args.cmd]()


if __name__ == "__main__":
    main()
