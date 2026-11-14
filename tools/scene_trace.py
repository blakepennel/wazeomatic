"""Trace the scene layer over a clip: ego motion, vehicles, signals and the scene labels, second by second.

    python tools/scene_trace.py assets/videos/dashcam_urban_1080p.mp4 --out runs/scene/urban.csv
    python tools/scene_trace.py clip.mp4 --vehicle-model models/vehicle_cls.onnx --weather-model models/weather_cls.onnx

Used to tune ``SceneConfig`` thresholds: the CSV has one row per inference frame (t, flow, ego state, vehicle
count, red signal seen, emergency lights, the weather probabilities, every scene label raised) and the summary
lists how long each label was up. A clean clip should raise nothing for long.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wazeomatic.build import DetectorSpec, add_detector_args, add_scene_args, build_detectors, build_second_stage  # noqa: E402
from wazeomatic.sources import VideoSource  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    add_detector_args(ap)
    add_scene_args(ap)
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--out", help="CSV path (default runs/scene/<clip>.csv)")
    args = ap.parse_args()
    spec = DetectorSpec.from_args(args)
    dets_models = build_detectors(spec)
    vehicle, weather, scene = build_second_stage(args)
    if scene is None:
        raise SystemExit("--no-scene leaves nothing to trace")
    src = VideoSource(args.video, step=args.step, max_frames=args.max_frames)
    out = Path(args.out) if args.out else Path("runs/scene") / f"{Path(args.video).stem}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    up: Counter = Counter()
    police_frames = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["t", "flow", "ego", "vehicles", "lights", "police_max", "weather", "labels"])
        last_t = None
        for fr in src:
            dets = []
            for m in dets_models:
                dets += m(fr.image)
            if vehicle is not None:
                dets += vehicle(fr.image, dets)
            wprobs = weather(fr.image, fr.ts) if weather is not None else None
            extra = scene.update(fr.ts, fr.image, dets, wprobs)
            labels = sorted({d.label for d in extra if d.model == "scene"})
            lights = sum(1 for d in extra if d.label == "emergency_lights")
            pol = max((d.confidence for d in dets if d.label == "police_vehicle"), default=0.0)
            police_frames += pol >= 0.5
            dt = 0.0 if last_t is None else fr.ts - last_t
            last_t = fr.ts
            for lab in labels:
                up[lab] += dt
            wr.writerow([f"{fr.ts:.2f}", f"{scene.ego.flow:.4f}", scene.state.get("ego"), scene.state.get("vehicles"),
                         lights, f"{pol:.2f}", json.dumps({k: round(v, 2) for k, v in scene.weather.items()}),
                         " ".join(labels)])
    src.release()
    print(json.dumps({"clip": args.video, "csv": str(out), "seconds_up": {k: round(v, 1) for k, v in up.items()},
                      "frames_police_ge_0.5": police_frames}, indent=1))


if __name__ == "__main__":
    main()
