"""Score Construction reports against the Argoverse 2 3D labels.

    python tools/eval_vehicle.py clips --hazard --clips <datasets>/argoverse2/clips --clip-half 1 --step 1 \\
        --out runs/eval/argo_h1_hazard_all.json
    python tools/eval_construction_argoverse.py --pred runs/eval/argo_h1_hazard_all.json

Truth for a clip: a work-zone object (CONSTRUCTION_CONE, CONSTRUCTION_BARREL, MESSAGE_BOARD_TRAILER,
TRAFFIC_LIGHT_TRAILER, OFFICIAL_SIGNALER) with lidar points on it stands 3-40 m ahead of the front camera,
inside its field of view and at most 10 m to the side, for at least ``--hold`` seconds. A clip whose objects
only ever stand further out, or only for a moment, is "unclear" and is left out of precision and recall: the
camera may or may not have had a fair look at them. Argoverse has no label for a construction sign or a
road worker as such (they are SIGN and PEDESTRIAN), so a Construction report on a sign alone counts as false
here; the list of false clips is printed so they can be checked by eye.
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

OBJECTS = {"CONSTRUCTION_CONE", "CONSTRUCTION_BARREL", "MESSAGE_BOARD_TRAILER", "TRAFFIC_LIGHT_TRAILER",
           "OFFICIAL_SIGNALER"}
CAM_X = 1.63             # ring_front_center sits 1.63 m ahead of the ego origin


def truth_state(ann, half_fov_deg: float, near: tuple[float, float], side_m: float, hold_s: float,
                far_m: float) -> str:
    """"yes" (a work zone in clear view for hold_s), "no" (no work-zone object within far_m ahead at all)
    or "unclear"."""
    rows = ann[ann.category.isin(OBJECTS) & (ann.num_interior_pts > 0)]
    tan = math.tan(math.radians(half_fov_deg))
    seen, anywhere = set(), False
    for ts, x, y in zip(rows.timestamp_ns, rows.tx_m, rows.ty_m):
        xc = x - CAM_X
        if 0 < xc <= far_m and abs(y) <= tan * xc + side_m:
            anywhere = True
        if near[0] <= xc <= near[1] and abs(y) <= min(tan * xc, side_m):
            seen.add(int(ts))
    stamps = sorted(set(int(t) for t in ann.timestamp_ns))
    run, start = False, 0.0
    for ts in stamps:
        t = ts / 1e9
        if ts in seen:
            if not run:
                run, start = True, t
            if t - start >= hold_s:
                return "yes"
        else:
            run = False
    return "unclear" if anywhere else "no"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="JSON written by tools/eval_vehicle.py clips --hazard --out")
    ap.add_argument("--datasets")
    ap.add_argument("--half-fov", type=float, default=23.6, help="half the camera's horizontal field of view")
    ap.add_argument("--hold", type=float, default=1.0, help="seconds a work-zone object must stay in view")
    ap.add_argument("--far", type=float, default=80.0, help="a clip is 'no' only without any object this close")
    ap.add_argument("--out")
    args = ap.parse_args()
    import pandas as pd

    root = Path(args.datasets) if args.datasets else datasets_root()
    files = {p.name[:8]: p for p in (root / "argoverse2" / "annotations").rglob("*.feather")}
    pred = json.loads(Path(args.pred).read_text(encoding="utf-8"))["per_clip"]
    counts = defaultdict(int)
    lists = defaultdict(list)
    for clip, r in sorted(pred.items()):
        key = Path(clip).stem[:8]
        if key not in files:
            counts["no_labels"] += 1
            continue
        state = truth_state(pd.read_feather(files[key]), args.half_fov, (3.0, 40.0), 10.0, args.hold, args.far)
        fired = "HAZARD_ON_ROAD_CONSTRUCTION" in r.get("events", [])
        counts[f"truth_{state}"] += 1
        if state == "yes":
            counts["tp" if fired else "fn"] += 1
            if not fired:
                lists["missed"].append(key)
        elif state == "no":
            counts["fp" if fired else "tn"] += 1
            if fired:
                lists["false"].append(key)
        else:
            counts["unclear_fired"] += fired
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    res = {**counts, "precision": round(tp / max(1, tp + fp), 3), "recall": round(tp / max(1, tp + fn), 3),
           **lists}
    print(json.dumps(res))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
