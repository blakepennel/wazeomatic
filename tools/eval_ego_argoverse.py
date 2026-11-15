"""How well does the camera-only ego-motion estimate tell stopped, slow and moving apart? Scored on Argoverse 2.

    python tools/cache_detections.py --clips <datasets>/argoverse2/clips --clip-half 1 --out runs/cache/argo_h1.jsonl
    python tools/eval_ego_argoverse.py --cache runs/cache/argo_h1.jsonl --out runs/eval/ego_h1.json

The Traffic reports (``scene_traffic_standstill``, ``scene_traffic_heavy``) rest on ``EgoMotion``: without GPS
speed (``use_gps_speed`` is off by default on the phone) it is the only thing that says the car is not moving.
Argoverse logs carry the true pose of the car (``city_SE3_egovehicle.feather``), so each clip frame gets a true
speed: the distance travelled between 0.5 s before and 0.5 s after the frame's camera timestamp. Frame ``k`` of
``<clip>.mp4`` is the ``k``-th front-camera image of the log (``tools/fetch_argoverse_hazards.py --stage clips``),
whose file name is its timestamp. The estimate is run over the clip with the cached boxes masked, exactly as in the
pipeline, and the first second of each clip (no history yet) is left out.

Truth states use the estimate's own speed cut-offs: stationary below 0.8 m/s, slow below 6 m/s, moving above.
Speeds within 0.3 m/s of a cut-off are not scored, since 0.5 s of pose noise or a creeping start can sit either
side. The report is the confusion matrix and, for the Traffic rules, the share of truly stopped frames called
stationary and of moving frames called stationary.
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

STATES = ("stationary", "slow", "moving")


def true_speeds(log: Path, stamps: list[int]) -> list[float]:
    import pandas as pd
    p = pd.read_feather(log / "city_SE3_egovehicle.feather").sort_values("timestamp_ns")
    ts = p.timestamp_ns.to_numpy().astype(np.float64) / 1e9
    x, y = p.tx_m.to_numpy(), p.ty_m.to_numpy()
    out = []
    for s in stamps:
        t = s / 1e9
        a, b = max(ts[0], t - 0.5), min(ts[-1], t + 0.5)
        if b - a < 0.5:
            out.append(float("nan"))
            continue
        dx = np.interp(b, ts, x) - np.interp(a, ts, x)
        dy = np.interp(b, ts, y) - np.interp(a, ts, y)
        out.append(float(np.hypot(dx, dy) / (b - a)))
    return out


def truth_state(v: float, ego: EgoMotion, margin: float) -> str | None:
    if np.isnan(v):
        return None
    for cut in (ego.stationary_mps, ego.slow_mps):
        if abs(v - cut) < margin:
            return None
    return "stationary" if v < ego.stationary_mps else "slow" if v < ego.slow_mps else "moving"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="jsonl from tools/cache_detections.py")
    ap.add_argument("--root", default=None, help="argoverse2 folder (default <datasets>/argoverse2)")
    ap.add_argument("--clips", default="clips", help="clip folder under the root")
    ap.add_argument("--margin", type=float, default=0.3, help="m/s around each cut-off left unscored")
    ap.add_argument("--warmup", type=float, default=1.0, help="seconds at the start of a clip left unscored")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root) if args.root else datasets_root() / "argoverse2"
    logs = {p.name[:8]: p for split in ("train", "val") for p in (root / "logs" / split).glob("*") if p.is_dir()}
    frames = defaultdict(list)
    with open(args.cache, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            frames[r["clip"]].append(r)

    conf = Counter()
    per_clip = {}
    skipped = 0
    for clip, rows in sorted(frames.items()):
        log = logs.get(clip[:8])
        jpgs = sorted((log / "sensors" / "cameras" / "ring_front_center").glob("*.jpg")) if log else []
        cap = cv2.VideoCapture(str(root / args.clips / clip))
        if not jpgs or not cap.isOpened():
            skipped += 1
            continue
        rows.sort(key=lambda r: r["t"])
        speeds = true_speeds(log, [int(p.stem) for p in jpgs])
        ego = EgoMotion()
        c = Counter()
        k = 0
        for r in rows:
            ok, img = cap.read()
            if not ok or k >= len(speeds):
                break
            dets = [Detection(label=d[0], confidence=d[1], bbox=tuple(d[2:6])) for d in r["dets"]]
            ego.update(r["t"], img, dets)
            truth = truth_state(speeds[k], ego, args.margin)
            if truth and r["t"] >= args.warmup:
                c[(truth, ego.state(r["t"]))] += 1
            k += 1
        cap.release()
        conf.update(c)
        per_clip[clip] = {f"{a}->{b}": n for (a, b), n in c.items()}

    print(f"{len(per_clip)} clips scored, {skipped} without a log or video")
    print(f"{'truth \\ estimate':>18} " + " ".join(f"{s:>11}" for s in STATES) + "   frames")
    summary = {}
    for a in STATES:
        n = sum(conf[(a, b)] for b in STATES)
        cells = " ".join(f"{conf[(a, b)] / n:>10.1%} " if n else f"{'-':>11}" for b in STATES)
        print(f"{a:>18} {cells}  {n}")
        summary[a] = {b: conf[(a, b)] for b in STATES}
    stop_n = sum(summary["stationary"].values())
    move_n = sum(summary["moving"].values())
    res = {
        "clips": len(per_clip), "confusion": summary,
        "stopped_called_stationary": summary["stationary"]["stationary"] / stop_n if stop_n else None,
        "moving_called_stationary": summary["moving"]["stationary"] / move_n if move_n else None,
        "per_clip": per_clip,
    }
    print(f"stopped frames called stationary: {res['stopped_called_stationary']}")
    print(f"moving frames called stationary: {res['moving_called_stationary']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
