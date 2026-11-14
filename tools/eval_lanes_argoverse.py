"""Score the blocked-lane rule against Argoverse 2 3D labels.

    python tools/eval_vehicle.py clips --hazard --clips <datasets>/argoverse2/clips --step 1 \\
        --out runs/eval/scene_argoverse.json -- --no-vehicle --no-weather --camera-geometry 47.2,1.44,0.53
    python tools/eval_lanes_argoverse.py --pred runs/eval/scene_argoverse.json
    python tools/cache_detections.py --clips <datasets>/argoverse2/clips --out runs/cache/argoverse.jsonl
    python tools/eval_lanes_argoverse.py --cache runs/cache/argoverse.jsonl --sweep    # replay the rule, tune it

The 3D cuboids of every CONSTRUCTION_CONE and CONSTRUCTION_BARREL give the truth: where each one stands on the
road relative to the car, in metres. A moment is a blocked lane when at least three of them stand 3-40 m ahead
within one lane width of our centre line and at least two of them inside our lane (centre less than half a lane
width from our centre line, 1.8 m), for 1.5 s; the side is where their median lies (left / right beyond 0.6 of
a half-lane, center between). The definition is fixed here (TRUTH), independent of the rule's parameters, so
the rule can be tuned against it. Only objects inside the front camera's field of view and with lidar points
on them count.

Reports per clip whether the truth and the pipeline agree on any blocked lane and on the side, and the totals:
clip-level precision and recall of BLOCKED_LANE events. The clips are 5 fps crops of ring_front_center (rows
307-1637 of the portrait 1550 x 2048 image), which is where ``--camera-geometry 47.2,1.44,0.53`` comes from:
fx 1775.5 px over a 1550 px width, the camera 1.44 m above the road, the principal point at row 0.530 of the crop.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wazeomatic.paths import datasets_root  # noqa: E402
from wazeomatic.scene import SceneAnalyzer, SceneConfig  # noqa: E402
from wazeomatic.schema import Detection  # noqa: E402

pd = None

OBJECTS = {"CONSTRUCTION_CONE", "CONSTRUCTION_BARREL"}
CAM_X = 1.63             # ring_front_center sits 1.63 m ahead of the ego origin


TRUTH = SceneConfig(lane_max_offset=2.0, lane_reach=1.0, lane_side_u=0.6, lane_min_objects=3, lane_hold_s=1.5)


def truth_sides(ann, cfg: SceneConfig, half_fov_deg: float, hold_s: float) -> set[str]:
    rows = ann[ann.category.isin(OBJECTS) & (ann.num_interior_pts > 0)]
    by_t = defaultdict(list)
    tan = math.tan(math.radians(half_fov_deg))
    for ts, x, y in zip(rows.timestamp_ns, rows.tx_m, rows.ty_m):
        xc = x - CAM_X
        if not cfg.lane_min_dist_m <= xc <= cfg.lane_max_dist_m or abs(y) > tan * xc:
            continue
        u = -y / (cfg.lane_width_m / 2)             # Argoverse y points left; u is positive to the right
        if abs(u) <= cfg.lane_max_offset:
            by_t[ts].append(u)
    sides, cur, since = set(), None, 0.0
    for ts in sorted(set(ann.timestamp_ns)):
        t = ts / 1e9
        us = sorted(by_t.get(ts, []))
        side = None
        if len(us) >= cfg.lane_min_objects and sum(1 for u in us if abs(u) <= cfg.lane_reach) >= 2:
            mid = us[len(us) // 2]
            side = "left" if mid < -cfg.lane_side_u else "right" if mid > cfg.lane_side_u else "center"
        if side != cur:
            cur, since = side, t
        if side is not None and t - since >= hold_s:
            sides.add(side)
    return sides


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", help="JSON written by tools/eval_vehicle.py clips --out")
    ap.add_argument("--cache", help="detections from tools/cache_detections.py: replay the lane rule on them")
    ap.add_argument("--camera-geometry", default="47.2,1.44,0.53", help="HFOV,HEIGHT,HORIZON of the clips")
    ap.add_argument("--sweep", action="store_true", help="with --cache: try a grid of rule parameters")
    ap.add_argument("--truth-hold", type=float, default=TRUTH.lane_hold_s,
                    help="seconds the truth condition must hold without a break (cars pass a taper in 1-2 s)")
    ap.add_argument("--datasets")
    ap.add_argument("--half-fov", type=float, default=23.6, help="half the camera's horizontal field of view")
    ap.add_argument("--out")
    args = ap.parse_args()
    global pd
    import pandas as pd

    root = Path(args.datasets) if args.datasets else datasets_root()
    ann_dir = root / "argoverse2" / "annotations"
    files = {p.name[:8]: p for p in ann_dir.rglob("*.feather")}
    truth_cache: dict[str, set[str]] = {}

    def truth(clip: str) -> set[str] | None:
        key = Path(clip).stem[:8]
        if key not in files:
            return None
        if key not in truth_cache:
            truth_cache[key] = truth_sides(pd.read_feather(files[key]), TRUTH, args.half_fov, args.truth_hold)
        return truth_cache[key]

    if args.cache:
        frames = defaultdict(list)
        for ln in Path(args.cache).read_text(encoding="utf-8").splitlines():
            r = json.loads(ln)
            frames[r["clip"]].append(r)
        hfov, height, horizon = (float(v) for v in args.camera_geometry.split(","))
        base = dict(hfov_deg=hfov, cam_height_m=height, horizon_y=horizon, enabled=("lanes",))
        grid = [{}]
        if args.sweep:
            grid = [{"lane_reach": r, "lane_max_offset": m, "lane_min_objects": n}
                    for r in (0.8, 0.9, 1.0, 1.1, 1.3) for m in (1.6, 2.0) for n in (3, 4)]
        rows = []
        for g in grid:
            cfg = SceneConfig(**base, **g)
            pred = {}
            for clip, frs in frames.items():
                sa = SceneAnalyzer(cfg)
                got = set()
                for r in frs:
                    ds = [Detection(d[0], d[1], tuple(d[2:6])) for d in r["dets"]]
                    got |= {o.label.split("_")[-1] for o in sa._lanes(r["t"], ds, r["w"] / r["h"])}
                pred[clip] = {"events": [f"BLOCKED_LANE_{s.upper()}" for s in got]}
            res = score(pred, truth)
            rows.append({**g, **{k: v for k, v in res.items() if k != "per_clip"}})
            print(json.dumps(rows[-1]), flush=True)
        if args.out:
            Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
        return
    pred = json.loads(Path(args.pred).read_text(encoding="utf-8"))["per_clip"]
    res = score(pred, truth)
    print(json.dumps({k: v for k, v in res.items() if k != "per_clip"}))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


def score(pred: dict, truth) -> dict:
    tp = fp = fn = tn = side_ok = 0
    per = {}
    for clip, r in sorted(pred.items()):
        tr = truth(clip)
        if tr is None or "events" not in r:
            continue
        got = {e.split("_")[-1].lower() for e in r["events"] if e.startswith("BLOCKED_LANE_")}
        if tr and got:
            tp += 1
            side_ok += bool(tr & got)
        elif got:
            fp += 1
        elif tr:
            fn += 1
        else:
            tn += 1
        per[clip] = {"truth": sorted(tr), "pipeline": sorted(got)}
    return {"clips": len(per), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(tp / max(1, tp + fp), 3), "recall": round(tp / max(1, tp + fn), 3),
            "side_agrees_on_tp": round(side_ok / max(1, tp), 3), "per_clip": per}


if __name__ == "__main__":
    main()
