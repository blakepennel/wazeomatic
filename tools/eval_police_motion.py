"""How well does ``PoliceMotion`` tell a standing car from a driving one? Scored on Argoverse 2 against the 3D tracks.

    python tools/label_vehicle_motion_argoverse.py --cache runs/cache/argo_h1.jsonl --out runs/cache/argo_h1_motion.jsonl
    python tools/eval_police_motion.py --cache runs/cache/argo_h1.jsonl --motion runs/cache/argo_h1_motion.jsonl \
        --out runs/eval/police_motion_h1.json

Police cars are too few to score a motion rule on, so every vehicle stands in for one: each cached car, truck
or bus box (the COCO detector's, conf >= 0.35, at least 28 px) is handed to ``PoliceMotion`` as a police box,
with the camera-only ego state of ``EgoMotion`` as in the pipeline. ``tools/label_vehicle_motion_argoverse.py``
gives the matched boxes their track and true ground speed. A truth track is *stopped* when its speed stays
under 0.5 m/s, *moving* when it stays over 2 m/s; tracks in between are left out. The verdict for a track is the
module's state on the track's last scored frame.

The report, by distance band (median cuboid depth): the share of stopped tracks that pass (state stopped), the
share of moving tracks that pass (these would become false Police reports), and the share still pending.
``pass`` is the verdict on a track's last frame; ``hit2`` is the share of tracks read standing on at least two
frames, as the pipeline needs two hits to report (a car read standing that starts a new track as it leaves the
frame counts as pending under ``pass`` but did report).
Run with ``--set name=value`` to try MotionConfig values.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wazeomatic.motion import MOVING, PENDING, POLICE, MotionConfig, PoliceMotion  # noqa: E402
from wazeomatic.paths import datasets_root  # noqa: E402
from wazeomatic.scene import EgoMotion  # noqa: E402
from wazeomatic.schema import Detection  # noqa: E402

BANDS = ((0, 30), (30, 60), (60, 1000))


def _truth(v) -> str | None:
    sp = [x[1] for x in v if not np.isnan(x[1])]
    return None if not sp else "stopped" if max(sp) < 0.5 else "moving" if min(sp) > 2.0 else None


def run_clip(path: Path, rows: list[dict], truth: dict, cfg: MotionConfig, speedup: int = 1) -> dict:
    """Track key -> list of (verdict, speed, depth) per scored frame. ``speedup`` k keeps every k-th frame and
    stamps it as if the frames still came at the clip's rate: static objects then move as they would from a car
    driving k times as fast (the truth is looked up at the frame's real time)."""
    cap = cv2.VideoCapture(str(path))
    ego, pm = EgoMotion(), PoliceMotion(cfg)
    out = defaultdict(list)
    for n, r in enumerate(sorted(rows, key=lambda r: r["t"])):
        ok, img = cap.read()
        if not ok:
            break
        if n % speedup:
            continue
        t = r["t"] / speedup
        h, w = img.shape[:2]
        dets = [Detection(d[0], d[1], tuple(d[2:6]), model="coco") for d in r["dets"]]
        ego.update(t, img, dets)
        police = [Detection(POLICE, 0.9, d.bbox, model="vehicle") for d in dets
                  if d.label in ("car", "truck", "bus") and d.confidence >= 0.35
                  and min(d.bbox[2] * w, d.bbox[3] * h) >= 28]
        res = pm.update(t, img, dets + police, ego.state(t))
        for d, tr in zip(res[len(dets):], pm.last_tracks):
            lab = truth.get((round(r["t"], 1), tuple(round(v, 4) for v in d.bbox)))
            if lab is None:
                continue
            verdict = "stopped" if d.label == POLICE else "moving" if d.label == MOVING else "pending"
            regime = "moving" if len(tr.resid) >= len(tr.still) else "standing"
            out[(path.name, lab["track"])].append((verdict, lab["speed"], lab["depth_m"], regime, lab["ego_speed"]))
    cap.release()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--root", default=None, help="argoverse2 folder (default <datasets>/argoverse2)")
    ap.add_argument("--set", action="append", default=[], help="MotionConfig field=value")
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--speedup", type=int, default=1, help="keep every k-th frame, stamped at the clip's rate "
                    "(static objects move as seen from a car k times as fast)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = MotionConfig()
    for s in args.set:
        k, v = s.split("=", 1)
        cur = getattr(cfg, k)
        setattr(cfg, k, tuple(int(x) for x in v.split(",")) if isinstance(cur, tuple) else type(cur)(v))
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
            truth[r["clip"]][(round(r["t"], 1), tuple(round(v, 4) for v in r["bbox"]))] = r
    clips = sorted(c for c in frames if c in truth)
    if args.max_clips:
        clips = clips[: args.max_clips]
    tracks = {}
    for i, clip in enumerate(clips):
        tracks.update(run_clip(root / "clips" / clip, frames[clip], truth[clip], cfg, args.speedup))
        print(f"[{i + 1}/{len(clips)}] {clip}: {len(tracks)} tracks", flush=True)

    result = {"config": vars(cfg), "speedup": args.speedup, "bands": {}}
    for lo, hi in BANDS:
        c = {k: {"n": 0, "stopped": 0, "moving": 0, "pending": 0, "hit2": 0} for k in ("stopped", "moving")}
        for v in tracks.values():
            sp = [x[1] for x in v if not np.isnan(x[1])]
            if not sp:
                continue
            truth_state = _truth(v)
            if truth_state is None or not lo <= float(np.median([x[2] for x in v])) < hi:
                continue
            c[truth_state]["n"] += 1
            c[truth_state][v[-1][0]] += 1
            c[truth_state]["hit2"] += sum(x[0] == "stopped" for x in v) >= 2
        result["bands"][f"{lo}-{hi} m"] = c
        line = []
        for k in ("stopped", "moving"):
            n = max(1, c[k]["n"])
            line.append(f"{k} {c[k]['n']}: pass {c[k]['stopped'] / n:.2f} pending {c[k]['pending'] / n:.2f} "
                        f"hit2 {c[k]['hit2'] / n:.2f}")
        print(f"{lo}-{hi} m  " + " | ".join(line))
        if lo == 0:      # where the decided verdicts come from, near band
            for k in ("stopped", "moving"):
                for reg in ("moving", "standing"):
                    vs = [v[-1][0] for v in tracks.values() if v[-1][0] != "pending" and v[-1][3] == reg
                          and _truth(v) == k and lo <= float(np.median([x[2] for x in v])) < hi]
                    if vs:
                        print(f"    truth {k}, decided while we were {reg}: {len(vs)}, pass {vs.count('stopped') / len(vs):.2f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
