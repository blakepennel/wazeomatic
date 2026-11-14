"""Evaluate the vehicle classifier: on held-out crops, and as reports on whole clips.

    python tools/eval_vehicle.py crops --model models/vehicle_cls.onnx
    python tools/eval_vehicle.py clips --model models/vehicle_cls.onnx --clips F:/dev/datasets/commons_videos
    python tools/eval_vehicle.py clips --clips assets/videos --expect none          # false reports on clean drives
    python tools/eval_vehicle.py clips --hazard --clips F:/dev/datasets/argoverse2/clips   # every event type
    python tools/eval_vehicle.py clips --clips F:/dev/datasets/argoverse2/clips --clip-half 1 --step 1 \
        --police-truth configs/police_truth_argoverse.txt

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
    # a comma list averages the models per class, as VehicleClassifier does with several models
    clfs = [OnnxClassifier(m, threads=8) for m in args.model.split(",")]
    names = clfs[0].labels
    ys, ps, srcs = [], [], []
    for ci, cls in enumerate(names):
        files = sorted((root / cls).glob("*.jpg"))
        srcs += [f.name.split("_", 1)[0] for f in files]
        for i in range(0, len(files), 64):
            ims = [cv2.imread(str(f)) for f in files[i:i + 64]]
            acc = np.zeros((len(ims), len(names)), np.float32)
            for clf in clfs:
                batch = [cv2.resize(im, (clf.size, clf.size), interpolation=cv2.INTER_AREA) for im in ims]
                pr = clf.predict(batch)
                acc += pr[:, [clf.labels.index(n) for n in names]]
            ps.append(acc / len(clfs))
            ys += [ci] * len(ims)
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
        # at fixed thresholds, the shipped floor among them: police recall, how often a civilian vehicle is called
        # police by where the crop comes from (bdd = dashcam frames, oc = Open Images photos, cm = Commons
        # look-alikes, mn = hand-checked dashcam crops), and how often an ambulance or fire engine is
        src = np.array(srcs)
        at = {}
        for thr in (0.8, 0.85, 0.9):
            row = {"police_recall": round(float((pol >= thr).mean()), 4)}
            for prefix in sorted(set(src[y == ci])):
                m = (y == ci) & (src == prefix)
                row[f"civilian_{prefix}_fpr"] = round(float((p[m, pi] >= thr).mean()), 5)
                row[f"civilian_{prefix}_n"] = int(m.sum())
            for other in ("ambulance", "fire"):
                if other in names:
                    m = y == names.index(other)
                    row[f"{other}_as_police"] = round(float((p[m, pi] >= thr).mean()), 4)
            at[str(thr)] = row
        out["at_threshold"] = at
        out["confusion"] = {names[a]: {names[b]: int(((y == a) & (pred == b)).sum()) for b in range(len(names))}
                            for a in range(len(names))}
    return out


def eval_clips(args) -> tuple[dict, int]:
    from wazeomatic.build import DetectorSpec, add_detector_args, add_scene_args, build_detectors, build_second_stage
    from wazeomatic.hood import HoodEstimator
    from wazeomatic.pipeline import Pipeline
    from wazeomatic.report_manager import ReportConfig, ReportManager
    from wazeomatic.schema import load_class_to_hazard
    from wazeomatic.sources import VideoSource

    ap = argparse.ArgumentParser()
    add_detector_args(ap)
    add_scene_args(ap)
    dargs = ap.parse_args(([] if args.hazard else ["--no-hazard"]) +
                          (["--vehicle-model", args.model] if args.model else []) + args.extra)
    spec = DetectorSpec.from_args(dargs)
    folder = Path(args.clips)
    clips = sorted(p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXT)
    if args.clip_half is not None:
        # the half that tools/mine_vehicle_crops.py --clip-half did not mine
        from mine_vehicle_crops import clip_half
        clips = [p for p in clips if clip_half(p.stem) == args.clip_half]
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
        # the learnt hood zone, as in run_video.py and the app: without it a dashboard vent can pass as a pothole
        pipe = Pipeline(dets, mgr, [], on_frame=on_frame, vehicle_classifier=vehicle, scene_classifier=weather,
                        scene=scene, hood=HoodEstimator(min_frac=mgr.cfg.hood_frac))
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
    by_type: dict[str, int] = {}
    for r in results.values():
        for t in r.get("events", []):
            by_type[t] = by_type.get(t, 0) + 1
    minutes = round(sum(r.get("seconds", 0.0) for r in results.values()) / 60.0, 1)
    out = {"clips": n, "minutes": minutes, "clips_with_police_or_crash": hit, "events_by_type": by_type}
    if args.police_truth:
        # clip stems checked by eye to show a marked police vehicle; every other clip has none
        truth = {ln.split("#")[0].strip() for ln in Path(args.police_truth).read_text(encoding="utf-8").splitlines()}
        truth.discard("")
        fired = {Path(k).stem for k, r in results.items() if "POLICE" in r.get("events", [])}
        stems = {Path(k).stem for k, r in results.items() if "error" not in r}
        pos = truth & stems
        out["police_truth"] = {"clips_with_police": len(pos), "found": len(fired & pos), "missed": sorted(pos - fired),
                               "clips_without_police": len(stems - pos), "false_clips": sorted(fired - pos)}
    out["per_clip"] = results
    return out, bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["crops", "clips"])
    ap.add_argument("--model", help="vehicle classifier ONNX (default models/vehicle_cls.onnx); crops: a comma "
                                    "list averages several")
    ap.add_argument("--data", help="crops: val folder")
    ap.add_argument("--clips", help="clips: folder of videos (searched recursively)")
    ap.add_argument("--clip-half", type=int, choices=[0, 1],
                    help="clips: only this half of the folder (split as tools/mine_vehicle_crops.py does)")
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--hazard", action="store_true", help="clips: run the hazard model too (all scene rules)")
    ap.add_argument("--expect", choices=["none", "any"], help="clips: fail on any / on no police-crash event")
    ap.add_argument("--police-truth", help="clips: text file of clip stems that show a marked police vehicle "
                                           "(one per line); scores POLICE events against it")
    ap.add_argument("--out", help="write the JSON result here")
    argv = sys.argv[1:]
    # pipeline flags go after a bare "--"; a REMAINDER positional would swallow --model and the rest too
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    args = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    args.extra = extra
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
