"""Check the weather and road-surface classifiers on held-out dashcam frames.

    python tools/eval_scene.py                                   # models/weather_cls.onnx + models/surface_cls.onnx
    python tools/eval_scene.py --weather models/weather_yolo11n.onnx --surface models/surface_yolo11n.onnx --limit 3000

The frames are the BDD100K frames that ``tools/build_scene_sets.py weather`` put in its val split (same md5
rule, so none of them was trained on), at full resolution. For each one it runs the two classifiers the way
the pipeline does (``SceneClassifier`` on the squashed frame, ``RoadSurface`` on the road patch) and reports,
per BDD weather label:

* the weather confusion (clear / rain / snow / fog), and how often a single frame meets the fog rule's
  threshold (fog above 0.70) and the rain half of the slippery rule (rain above 0.65);
* how often each single-frame surface condition of the scene rules would be met: water above 0.85, ice above
  0.80, fresh snow above 0.75, wet-or-water-or-slush above 0.65. The rules also need the condition to hold for
  8 to 60 s, so these per-frame rates are an upper bound on what reaches Waze. On clear frames every one of them
  is a false alarm; the RSCD training pictures are close-ups of the road, the patch is the road 10-30 m ahead,
  so this is where a domain gap shows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wazeomatic.classifiers import RoadSurface, SceneClassifier  # noqa: E402
from wazeomatic.paths import datasets_root  # noqa: E402

WEATHER_MAP = {"clear": "clear", "partly cloudy": "clear", "overcast": "clear", "rainy": "rain",
               "snowy": "snow", "foggy": "fog"}
SURFACE_RULES = {"flood": (("surface_water",), 0.85), "ice": (("surface_ice",), 0.80),
                 "unplowed": (("surface_snow",), 0.75),
                 "wet_any": (("surface_wet", "surface_water", "surface_slush"), 0.65)}
WEATHER_RULES = {"fog": ("fog", 0.70), "rain": ("rain", 0.65)}


def is_val(stem: str, val_frac: float = 0.15) -> bool:
    return int(hashlib.md5(stem.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF < val_frac


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weather", default="models/weather_cls.onnx")
    ap.add_argument("--surface", default="models/surface_cls.onnx")
    ap.add_argument("--datasets")
    ap.add_argument("--limit", type=int, default=4000, help="frames per weather label at most")
    ap.add_argument("--out", help="write the JSON result here")
    args = ap.parse_args()
    root = Path(args.datasets) if args.datasets else datasets_root()
    items = defaultdict(list)
    for m in sorted((root / "bdd100k_weather" / "data" / "images").rglob("metadata.jsonl")):
        for ln in m.read_text(encoding="utf-8").splitlines():
            r = json.loads(ln)
            cls = WEATHER_MAP.get(r.get("label", ""))
            p = m.parent / r["file_name"]
            if cls and is_val(p.stem) and len(items[cls]) < args.limit:
                items[cls].append(p)
    wc = SceneClassifier(args.weather, every_s=0.0, threads=4) if Path(args.weather).exists() else None
    sc = RoadSurface(args.surface, every_s=0.0, threads=4) if Path(args.surface).exists() else None
    if wc is None and sc is None:
        raise SystemExit("neither classifier exists")
    res: dict = {"frames": {k: len(v) for k, v in items.items()}}
    conf = {k: Counter() for k in items}
    rule_hits = {k: Counter() for k in items}
    weather_hits = {k: Counter() for k in items}
    surf_top = {k: Counter() for k in items}
    tick = 0.0                                   # a running clock: the classifiers rate-limit on it
    for cls, paths in items.items():
        for p in paths:
            tick += 1.0
            img = cv2.imread(str(p))
            if img is None:
                continue
            if wc is not None:
                w = wc(img, tick)
                conf[cls][max(w, key=w.get)] += 1
                for name, (key, thr) in WEATHER_RULES.items():
                    weather_hits[cls][name] += w.get(key, 0.0) > thr
            if sc is not None:
                s = sc(img, tick)
                surf_top[cls][max(s, key=s.get)] += 1
                for name, (keys, thr) in SURFACE_RULES.items():
                    rule_hits[cls][name] += max(s.get(k, 0.0) for k in keys) > thr
        print(f"[{cls}] {len(paths)} frames", flush=True)
    n = {k: max(1, len(v)) for k, v in items.items()}
    if wc is not None:
        res["weather_confusion"] = {k: dict(v) for k, v in conf.items()}
        res["weather_recall"] = {k: round(conf[k][k] / n[k], 3) for k in items}
        res["weather_rule_rate"] = {k: {r: round(weather_hits[k][r] / n[k], 4) for r in WEATHER_RULES}
                                    for k in items}
    if sc is not None:
        res["surface_argmax"] = {k: dict(v) for k, v in surf_top.items()}
        res["surface_rule_rate"] = {k: {r: round(rule_hits[k][r] / n[k], 4) for r in SURFACE_RULES} for k in items}
    text = json.dumps(res, indent=1)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
