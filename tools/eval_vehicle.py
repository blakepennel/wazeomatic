"""Evaluate the vehicle classifier: on held-out crops, and as reports on whole clips.

    python tools/eval_vehicle.py crops --model models/vehicle_cls.onnx
    python tools/eval_vehicle.py clips --model models/vehicle_cls.onnx --clips F:/dev/datasets/commons_videos
    python tools/eval_vehicle.py clips --clips assets/videos --expect none          # false reports on clean drives

**crops** - precision / recall per class on ``<datasets>/vehicle_crops/val``, and for police the score
threshold that keeps the civilian false-positive rate at 1 %, 0.3 % and 0.1 % of crops, with the police recall
at each. A dashcam sees thousands of civilian cars an hour, so the low-FPR end is the operating point.

**clips** - runs the real pipeline (COCO detector, vehicle classifier, flashing-light detector, scene rules,
report manager with the shipped thresholds) over every video in a folder and lists, per clip, the events it
would send, the best police score and how many frames had flashing lights. ``--expect none`` turns the run into
a check: any POLICE / CRASH event is a false report and the exit code is 1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wazeomatic.classifiers import OnnxClassifier  # noqa: E402
from wazeomatic.paths import datasets_root  # noqa: E402

VIDEO_EXT = {".mp4", ".webm", ".ogv", ".ogg", ".mkv", ".mov", ".avi"}


def eval_crops(args) -> dict:
    root = Path(args.data) if args.data else datasets_root() / "vehicle_crops" / "val"
    clf = OnnxClassifier(args.model, threads=8)
    names = clf.labels
    ys, ps = [], []
    for ci, cls in enumerate(names):
        files = sorted((root / cls).glob("*.jpg"))
        for i in range(0, len(files), 64):
            batch = [cv2.resize(cv2.imread(str(f)), (clf.size, clf.size), interpolation=cv2.INTER_AREA)
                     for f in files[i:i + 64]]
            ps.append(clf.predict(batch))
            ys += [ci] * len(batch)
    y, p = np.array(ys), np.concatenate(ps)
    pred = p.argmax(1)
    out = {"n": int(len(y)), "top1": round(float((pred == y).mean()), 4), "per_class": {}}
    for ci, cls in enumerate(names):
        tp = int(((pred == ci) & (y == ci)).sum())
        out["per_class"][cls] = {"n": int((y == ci).sum()), "precision": round(tp / max(1, int((pred == ci).sum())), 4),
                                 "recall": round(tp / max(1, int((y == ci).sum())), 4)}
    if "police" in names and "civilian" in names:
        pi, ci = names.index("police"), names.index("civilian")
        civ = np.sort(p[y == ci, pi])[::-1]
        pol = p[y == pi, pi]
        ops = {}
        for fpr in (0.01, 0.003, 0.001):
            k = max(0, int(np.ceil(fpr * len(civ))) - 1)
            thr = float(civ[k]) if len(civ) else 1.0
            ops[f"fpr_{fpr}"] = {"threshold": round(thr, 4), "police_recall": round(float((pol > thr).mean()), 4)}
        out["police_operating_points"] = ops
        out["confusion"] = {names[a]: {names[b]: int(((y == a) & (pred == b)).sum()) for b in range(len(names))}
                            for a in range(len(names))}
    return out


def eval_clips(args) -> tuple[dict, int]:
    from wazeomatic.build import DetectorSpec, add_detector_args, add_scene_args, build_detectors, build_second_stage
    from wazeomatic.pipeline import Pipeline
    from wazeomatic.report_manager import ReportConfig, ReportManager
    from wazeomatic.schema import load_class_to_hazard
    from wazeomatic.sources import VideoSource

    ap = argparse.ArgumentParser()
    add_detector_args(ap)
    add_scene_args(ap)
    dargs = ap.parse_args(["--no-hazard"] + (["--vehicle-model", args.model] if args.model else []) + args.extra)
    spec = DetectorSpec.from_args(dargs)
    folder = Path(args.clips)
    clips = sorted(p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXT)
    results, bad = {}, 0
    for clip in clips:
        dets = build_detectors(spec)
        vehicle, weather, scene = build_second_stage(dargs)
        mgr = ReportManager(ReportConfig(class_to_hazard=load_class_to_hazard(), global_cooldown_s=0.0))
        stats = {"police_max": 0.0, "light_frames": 0, "police_frames": 0}

        def on_frame(_vis, ds, _evs):
            pol = max((d.confidence for d in ds if d.label == "police_vehicle"), default=0.0)
            stats["police_max"] = max(stats["police_max"], pol)
            stats["police_frames"] += pol >= 0.5
            stats["light_frames"] += any(d.label == "emergency_lights" for d in ds)

        try:
            src = VideoSource(str(clip), step=args.step)
        except Exception as e:  # noqa: BLE001
            results[clip.name] = {"error": str(e)}
            continue
        pipe = Pipeline(dets, mgr, [], on_frame=on_frame, vehicle_classifier=vehicle, scene_classifier=weather,
                        scene=scene)
        events = []
        mgr_update = mgr.update

        def capture(*a, **k):
            evs = mgr_update(*a, **k)
            events.extend(evs)
            return evs

        mgr.update = capture
        st = pipe.run(src)
        src.release()
        types = [e.hazard_type for e in events]
        flagged = [t for t in types if t.startswith(("POLICE", "CRASH"))]
        if args.expect == "none" and flagged:
            bad += 1
        if args.expect == "any" and not flagged:
            bad += 1
        results[clip.name] = {"seconds": round(st.frames * args.step / max(1.0, src.fps), 1), "events": types,
                              "police_max": round(stats["police_max"], 3), "police_frames": stats["police_frames"],
                              "light_frames": stats["light_frames"], "ended": st.ended}
        print(clip.name, json.dumps(results[clip.name]), flush=True)
    n = len([r for r in results.values() if "error" not in r])
    hit = sum(1 for r in results.values() if any(t.startswith(("POLICE", "CRASH")) for t in r.get("events", [])))
    return {"clips": n, "clips_with_police_or_crash": hit, "per_clip": results}, bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["crops", "clips"])
    ap.add_argument("--model", help="vehicle classifier ONNX (default models/vehicle_cls.onnx)")
    ap.add_argument("--data", help="crops: val folder")
    ap.add_argument("--clips", help="clips: folder of videos (searched recursively)")
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--expect", choices=["none", "any"], help="clips: fail on any / on no police-crash event")
    ap.add_argument("--out", help="write the JSON result here")
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="clips: extra pipeline flags after --")
    args = ap.parse_args()
    args.extra = [a for a in args.extra if a != "--"]
    if args.mode == "crops":
        args.model = args.model or "models/vehicle_cls.onnx"
        res, bad = eval_crops(args), 0
    else:
        res, bad = eval_clips(args)
    text = json.dumps(res, indent=1)
    print(text if args.mode == "crops" else json.dumps({k: v for k, v in res.items() if k != "per_clip"}))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
