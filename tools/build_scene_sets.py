"""Build the image-folder datasets of the two scene classifiers.

    python tools/build_scene_sets.py weather        # <datasets>/scene_weather/{train,val}/{clear,rain,snow,fog}
    python tools/build_scene_sets.py surface        # <datasets>/scene_surface/{train,val}/{dry,wet,water,slush,snow,ice}
    python tools/build_scene_sets.py topview        # <datasets>/scene_surface_topview/{train,val}/{dry,damp,wet,snow,slush}

**weather** - BDD100K frames (US dashcam) by their weather attribute: clear, partly cloudy and overcast are
``clear``; rainy, snowy and foggy are ``rain``, ``snow`` and ``fog``. Real fog is rare in BDD100K (about 140
frames), so the train split also gets synthetic fog: clear daytime frames blended towards a grey airlight with
a transmission that falls off towards the horizon (the usual single-scattering model with the image row as a
depth proxy). The val split has only real fog, so the reported fog recall is on real frames. Frames are
squashed to a square, the way ``SceneClassifier`` feeds the whole frame.

**surface** - RSCD road-surface images grouped into dry, wet, water (standing water), slush (melted snow),
snow (fresh snow) and ice. At run time ``RoadSurface`` (wazeomatic/classifiers.py) classifies the patch of
road just above the hood. These are close-ups from above, and the model does not carry over to a dashcam
(docs/waze-coverage.md, "Road surface").

**topview** - RoadSaW and RoadSC (VISCODA, CC BY-NC-SA 4.0; the zips under ``<datasets>/roadsaw``): road
patches seen from a car and warped to a bird's-eye view, 7.5-30 m ahead. Their classes fold into dry, damp,
wet (wet and very wet), snow (fresh and fully packed) and slush (partially covered); their own train /
validation split is kept. At run time ``RoadSurface`` with ``topview`` warps the 3.6 m square 15 m ahead
with the flat-road camera model (``--camera-geometry``), so the patch looks like these.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wazeomatic.paths import datasets_root  # noqa: E402

WEATHER_MAP = {"clear": "clear", "partly cloudy": "clear", "overcast": "clear", "rainy": "rain",
               "snowy": "snow", "foggy": "fog"}


def split_of(key: str, val_frac: float) -> str:
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val_frac else "train"


def synth_fog(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    horizon = rng.uniform(0.35, 0.5)
    beta = rng.uniform(1.6, 3.6)
    airlight = rng.uniform(0.72, 0.95) * 255
    ys = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    # distance proxy: 1 at and above the horizon, falling to ~0.1 at the bottom row
    dist = np.clip(1.0 - (ys - horizon) / (1.0 - horizon), 0.1, 1.0)
    t = np.exp(-beta * dist)                                   # transmission
    t = np.repeat(t, w, axis=1)[..., None]
    f = img.astype(np.float32) * t + airlight * (1 - t)
    f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.5, 1.5))
    f += np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, 3, f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


def build_weather(root: Path, out: Path, size: int, cap: dict, fog_synth: int, val_frac: float) -> Counter:
    rng = random.Random(0)
    period = {}
    for m in (root / "bdd100k_period").rglob("metadata.jsonl"):
        for ln in m.read_text(encoding="utf-8").splitlines():
            r = json.loads(ln)
            period[r["file_name"]] = r.get("label", "")
    items = []
    for m in (root / "bdd100k_weather" / "data" / "images").rglob("metadata.jsonl"):
        for ln in m.read_text(encoding="utf-8").splitlines():
            r = json.loads(ln)
            cls = WEATHER_MAP.get(r.get("label", ""))
            p = m.parent / r["file_name"]
            if cls and p.exists():
                items.append((p, cls))
    rng.shuffle(items)
    counts: Counter = Counter()
    clear_day = []
    for p, cls in items:
        split = split_of(p.stem, val_frac)
        if counts[(split, cls)] >= cap.get(cls, 10 ** 9) * (val_frac if split == "val" else 1.0):
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        sq = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        d = out / split / cls
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{p.stem}.jpg"), sq, [cv2.IMWRITE_JPEG_QUALITY, 90])
        counts[(split, cls)] += 1
        if cls == "clear" and split == "train" and period.get(p.name, "daytime") == "daytime":
            clear_day.append(p)
    rng.shuffle(clear_day)
    d = out / "train" / "fog"
    d.mkdir(parents=True, exist_ok=True)
    for p in clear_day[:fog_synth]:
        img = cv2.imread(str(p))
        sq = cv2.resize(synth_fog(img, rng), (size, size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(d / f"synth_{p.stem}.jpg"), sq, [cv2.IMWRITE_JPEG_QUALITY, 90])
        counts[("train", "fog_synth")] += 1
    return counts


SURFACE = [(re.compile(r"fresh_snow|fresh-snow"), "snow"), (re.compile(r"melted_snow|melted-snow"), "slush"),
           (re.compile(r"(^|[-_/])ice"), "ice"), (re.compile(r"(^|[-_/])water"), "water"),
           (re.compile(r"(^|[-_/])wet"), "wet"), (re.compile(r"(^|[-_/])dry"), "dry")]


def surface_class(path: Path) -> str | None:
    key = f"{path.parent.name}/{path.stem}".lower()
    for rx, cls in SURFACE:
        if rx.search(key):
            return cls
    return None


def build_surface(root: Path, out: Path, size: int, cap: int, val_frac: float) -> Counter:
    rng = random.Random(0)
    paths = sorted((root / "rscd").rglob("*.jpg"))
    rng.shuffle(paths)
    counts: Counter = Counter()
    for p in paths:
        cls = surface_class(p)
        if cls is None:
            continue
        split = split_of(p.stem, val_frac)
        if counts[(split, cls)] >= cap * (val_frac if split == "val" else 1.0):
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        sq = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        d = out / split / cls
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{p.stem}.jpg"), sq, [cv2.IMWRITE_JPEG_QUALITY, 90])
        counts[(split, cls)] += 1
    return counts


# RoadSaW wetness levels and RoadSC snow cover -> the classes of the top-view surface model. RoadSaW's "wet" and
# "very wet" both mean a water film a tyre notices; "damp" stays apart so the model can say "not quite".
TOPVIEW = {"dry": "dry", "damp": "damp", "wet": "wet", "verywet": "wet",
           "fresh_snow": "snow", "fully_packed": "snow", "partially_covered": "slush"}


def build_topview(root: Path, out: Path, size: int) -> Counter:
    """RoadSaW (150_l) and RoadSC (balanced to RoadSaW, 150_l) straight from their zip files: bird's-eye road
    patches 3.6 m square, 15 m ahead of a car or truck camera. Their own train / validation splits are kept
    (validation becomes val, test is left out for a final check)."""
    import zipfile
    counts: Counter = Counter()
    zips = [root / "roadsaw" / "RoadSaW-150_l.zip", root / "roadsaw" / "RoadSC-balanced_to_RoadSaW12-150_l.zip"]
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                parts = name.split("/")
                if len(parts) < 4 or not parts[-1].lower().endswith((".png", ".jpg")):
                    continue
                split = {"train": "train", "validation": "val"}.get(parts[1])
                folder = parts[2]
                cond = folder.split("_", 1)[1] if folder.split("_", 1)[0] in ("asphalt", "cobble", "concrete")                     else folder
                cls = TOPVIEW.get(cond)
                if split is None or cls is None:
                    continue
                img = cv2.imdecode(np.frombuffer(zf.read(name), np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                d = out / split / cls
                d.mkdir(parents=True, exist_ok=True)
                stem = f"{folder}_{Path(parts[-1]).stem}"
                cv2.imwrite(str(d / f"{stem}.jpg"), cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA),
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                counts[(split, cls)] += 1
        print(f"[{z.name}] {sum(counts.values())} patches so far", flush=True)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=["weather", "surface", "topview"])
    ap.add_argument("--datasets")
    ap.add_argument("--out")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--clear-cap", type=int, default=12000)
    ap.add_argument("--fog-synth", type=int, default=4000)
    ap.add_argument("--surface-cap", type=int, default=6000)
    args = ap.parse_args()
    root = Path(args.datasets) if args.datasets else datasets_root()
    if args.which == "weather":
        out = Path(args.out) if args.out else root / "scene_weather"
        c = build_weather(root, out, args.size, {"clear": args.clear_cap}, args.fog_synth, args.val_frac)
    elif args.which == "topview":
        out = Path(args.out) if args.out else root / "scene_surface_topview"
        c = build_topview(root, out, args.size)
    else:
        out = Path(args.out) if args.out else root / "scene_surface"
        c = build_surface(root, out, args.size, args.surface_cap, args.val_frac)
    summary = {f"{s}/{k}": n for (s, k), n in sorted(c.items())}
    (out / "counts.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
