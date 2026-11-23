"""Score Car on shoulder and the Other side rules against the Argoverse 2 3D tracks and lane maps.

    python tools/fetch_argoverse_hazards.py --stage maps
    python tools/label_vehicle_motion_argoverse.py --cache runs/cache/argo_h1.jsonl --out runs/cache/argo_h1_motion.jsonl
    python tools/eval_road_side.py --cache runs/cache/argo_h1.jsonl --motion runs/cache/argo_h1_motion.jsonl \
        --out runs/eval/road_side_h1.json

Truth per detector box (``tools/label_vehicle_motion_argoverse.py``): its ground speed, whether its centre lies
in a lane segment of the log's vector map (``in_lane``), and which way the nearest lane runs against ours
(``side``: ``ours``, ``other``, ``cross``).

**Car on shoulder.** ``ShoulderCars`` runs over every clip with the fast-road condition off (Argoverse is city
driving, almost never above 50 km/h, so that condition would keep it silent): what is left is the geometry and
the standing test. A box it marks is right when the vehicle stands (under 0.5 m/s) outside every lane on our
side of the road: parked at the kerb, which in town is what a car on a shoulder looks like. The report lists
what the marked boxes are, and how many of the truly parked vehicles in the rule's range it marks at least once.
It also says how long the camera-speed fast-road condition would have held on these clips, where the true
speed is far below 50 km/h nearly all the time: every such second would let the rule run in town.

**Other side.** For every vehicle box 8-60 m ahead with a lane ``ours`` or ``other``, the share each rule gets
right: the old one (the whole box in the left 22 % of the frame) and the ground-point offset to the left of
the camera beyond a distance.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wazeomatic.paths import datasets_root  # noqa: E402
from wazeomatic.scene import EgoMotion  # noqa: E402
from wazeomatic.schema import Detection  # noqa: E402
from wazeomatic.shoulder import ShoulderCars, ShoulderConfig  # noqa: E402

GEOMETRY = (47.2, 1.44, 0.53)       # Argoverse 2 front camera in the clip crop: hfov, mount height, horizon


def key(t: float, bbox) -> tuple:
    return round(t, 1), tuple(round(v, 4) for v in bbox)


def what(lab: dict) -> str:
    sp = lab["speed"]
    if sp is None or np.isnan(sp) or 0.5 <= sp <= 2.0:
        return "unclear speed"
    if sp > 2.0:
        return "moving"
    if lab.get("in_lane") is None:
        return "standing, no map"
    if lab["in_lane"]:
        return "standing in a lane"
    return {"ours": "parked, our side", "other": "parked, other side", "cross": "parked, side street"}.get(
        lab.get("side"), "parked, no lane near")


def run_clip(path: Path, rows: list[dict], truth: dict, cfg: ShoulderConfig) -> dict:
    cap = cv2.VideoCapture(str(path))
    ego = EgoMotion()
    sc = ShoulderCars(cfg, horizon_y=GEOMETRY[2], hfov_deg=GEOMETRY[0], cam_height_m=GEOMETRY[1])
    marked, cands, speeds = [], [], []
    for r in sorted(rows, key=lambda r: r["t"]):
        ok, img = cap.read()
        if not ok:
            break
        dets = [Detection(d[0], d[1], tuple(d[2:6]), model="coco") for d in r["dets"]]
        ego.update(r["t"], img, dets)
        out = sc.update(r["t"], img, dets, ego.state(r["t"]))
        for d in sc.last_candidates:
            lab = truth.get(key(r["t"], d.bbox))
            if lab is not None:
                cands.append(lab)
        for d in out:
            lab = truth.get(key(r["t"], d.bbox))
            marked.append(lab)
        speeds.append((r["t"], sc.speed, sc.highway))
    cap.release()
    return {"marked": marked, "candidates": cands, "speeds": speeds}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--set", action="append", default=[], help="ShoulderConfig field=value")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root) if args.root else datasets_root() / "argoverse2"
    frames = defaultdict(list)
    with open(args.cache, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            frames[r["clip"]].append(r)
    truth = defaultdict(dict)
    with open(args.motion, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            truth[r["clip"]][key(r["t"], r["bbox"])] = r
    clips = sorted(c for c in frames if c in truth)
    if args.max_clips:
        clips = clips[: args.max_clips]

    # ------------------------------------------------------------------ other side (boxes only, no video)
    sc0 = ShoulderCars(horizon_y=GEOMETRY[2], hfov_deg=GEOMETRY[0], cam_height_m=GEOMETRY[1])
    rules = {"old: box in the left 22 %": lambda b, g: b[0] + b[2] / 2 < 0.22}
    for m in (3.0, 4.0, 5.0, 6.0, 8.0):
        rules[f"ground point {m:g} m or more to the left"] = lambda b, g, m=m: g is not None and -g[1] >= m
    side_counts = {k: Counter() for k in rules}
    for clip in clips:
        for lab in truth[clip].values():
            if lab.get("side") not in ("ours", "other"):
                continue
            b = lab["bbox"]
            g = sc0.ground_point(b, 1280 / 1098)
            if g is None or not 8 <= g[0] <= 60:
                continue
            for name, rule in rules.items():
                said = rule(b, g)
                side_counts[name][(lab["side"], said)] += 1
    result = {"other_side": {}}
    print("Other side (vehicle boxes 8-60 m ahead, truth from the lane map):")
    for name, c in side_counts.items():
        tp, fp = c[("other", True)], c[("ours", True)]
        fn = c[("other", False)]
        prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
        result["other_side"][name] = {"precision": round(prec, 3), "recall": round(rec, 3), "tp": tp, "fp": fp, "fn": fn}
        print(f"  {name:40s} precision {prec:.2f} recall {rec:.2f} (other side called {tp}, ours called other {fp})")

    # ------------------------------------------------------------------ car on shoulder (video)
    cfg = ShoulderConfig(require_highway=False)
    for s in args.set:
        k, v = s.split("=", 1)
        cur = getattr(cfg, k)
        setattr(cfg, k, (v.lower() in ("1", "true", "yes")) if isinstance(cur, bool) else type(cur)(v))
    marked, cands = [], []
    gate_open, gate_n, true_fast = 0, 0, 0
    for i, clip in enumerate(clips):
        r = run_clip(root / "clips" / clip, frames[clip], truth[clip], cfg)
        marked += r["marked"]
        cands += r["candidates"]
        ego = {round(x["t"], 1): x["ego_speed"] for x in truth[clip].values()}
        for t, _v, hw in r["speeds"]:
            gate_n += 1
            gate_open += bool(hw)
            true_fast += (ego.get(round(t, 1)) or 0) >= 13.9
        print(f"[{i + 1}/{len(clips)}] {clip}: {len(marked)} marked boxes so far", flush=True)
    kinds = Counter("no 3D match" if m is None else what(m) for m in marked)
    # the fast-road condition keeps the rule to driving; the same boxes split by our own speed at the time
    by_speed = {}
    for lo, hi in ((0.0, 2.0), (2.0, 8.0), (8.0, 99.0)):
        sel = [m for m in marked if m is not None and m.get("ego_speed") is not None and lo <= m["ego_speed"] < hi]
        by_speed[f"{lo:g}-{hi:g} m/s"] = dict(Counter(what(m) for m in sel))
    tracks_marked = {(m["clip"], m["track"]) for m in marked if m is not None}
    parked = {(c["clip"], c["track"]) for c in cands if what(c) == "parked, our side"}
    result["shoulder"] = {"marked_boxes": dict(kinds), "parked_our_side_tracks_in_range": len(parked),
                          "of_them_marked": len(parked & tracks_marked),
                          "camera_fast_road_share": round(gate_open / max(1, gate_n), 4),
                          "true_fast_share": round(true_fast / max(1, gate_n), 4), "by_our_speed": by_speed,
                          "config": vars(cfg)}
    print("Car on shoulder, fast-road condition off; what the marked boxes are:")
    for k, n in kinds.most_common():
        print(f"  {k:24s} {n} ({n / max(1, len(marked)):.0%})")
    for k, c in by_speed.items():
        n = sum(c.values())
        print(f"  while we drove {k}: {n} boxes, " + ", ".join(f"{a} {b / max(1, n):.0%}" for a, b in
                                                             sorted(c.items(), key=lambda x: -x[1])))
    print(f"  parked on our side, in range: {len(parked)} tracks, marked {len(parked & tracks_marked)}")
    print(f"  camera fast-road condition held {gate_open / max(1, gate_n):.1%} of frames; "
          f"true speed 50 km/h or more on {true_fast / max(1, gate_n):.1%}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
