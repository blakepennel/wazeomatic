"""Desktop simulator: run the full detection pipeline on a video (or webcam).

Examples
    python -m wazeomatic.run_video assets/videos/dashcam_urban_1080p.mp4
    python -m wazeomatic.run_video clip.mp4 --camera-sim --show --record runs/sim/out.mp4
    python -m wazeomatic.run_video clip.mp4 --sink tcp:192.168.1.20:8765        # to the phone app
    python -m wazeomatic.run_video clip.mp4 --sink adb --adb-dry-run            # straight to emulator
    python -m wazeomatic.run_video 0 --camera                                    # webcam / Pi camera
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .detector import OnnxDetector
from .pipeline import Pipeline
from .report_manager import ReportConfig, ReportManager
from .schema import DEFAULT_CLASS_TO_HAZARD
from .sinks import build_sinks
from .sources import CameraSim, CameraSource, VideoSource

ROOT = Path(__file__).resolve().parents[1]


def _labels_for(model_path: Path, explicit: str | None) -> list[str]:
    if explicit:
        return [ln.strip() for ln in Path(explicit).read_text(encoding="utf-8").splitlines() if ln.strip()]
    cand = model_path.with_suffix(".labels.txt")
    if cand.exists():
        return [ln.strip() for ln in cand.read_text(encoding="utf-8").splitlines() if ln.strip()]
    raise SystemExit(f"no labels file for {model_path} (expected {cand})")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="video path, or camera index with --camera")
    ap.add_argument("--camera", action="store_true")
    ap.add_argument("--hazard-model", default=str(ROOT / "models" / "hazard_yolo11n.onnx"))
    ap.add_argument("--hazard-labels")
    ap.add_argument("--coco-model", default=str(ROOT / "models" / "coco_yolo11n.onnx"))
    ap.add_argument("--coco-labels")
    ap.add_argument("--no-coco", action="store_true", help="skip the stock COCO detector")
    ap.add_argument("--no-hazard", action="store_true", help="skip the custom hazard detector")
    ap.add_argument("--classes", default=str(ROOT / "configs" / "classes.yaml"))
    ap.add_argument("--conf", type=float, default=0.35, help="detector confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=4, help="ORT intra-op threads (phone-like: 4)")
    ap.add_argument("--step", type=int, default=3, help="use every Nth frame (30 fps / 3 = 10 inferences/s)")
    ap.add_argument("--start", type=float, default=0.0, help="start offset in seconds")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--camera-sim", action="store_true", help="degrade frames like a phone camera")
    ap.add_argument("--sim-res", default="1280x720")
    # report manager
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--report-conf", type=float, default=0.45)
    ap.add_argument("--cooldown", type=float, default=60.0)
    ap.add_argument("--global-cooldown", type=float, default=5.0)
    ap.add_argument("--stopped-vehicle", action="store_true", help="enable experimental shoulder heuristic")
    ap.add_argument("--event-source", default="sim", choices=["sim", "pi", "phone"])
    # sinks
    ap.add_argument("--sink", action="append", default=[],
                    help="stdout | jsonl:<path> | tcp:<host>[:port] | adb[:<serial>] | http:<url> (repeatable)")
    ap.add_argument("--adb", default="adb")
    ap.add_argument("--adb-package", help="explicit target package for the broadcast, e.g. com.arlosoft.macrodroid")
    ap.add_argument("--adb-dry-run", action="store_true", help="set dry_run=true on broadcast events")
    # output
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--record", help="write annotated video here")
    ap.add_argument("--stats-json", help="write run statistics here")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    cls_path = Path(args.classes)
    cls_cfg = yaml.safe_load(cls_path.read_text(encoding="utf-8")) if cls_path.exists() else {}
    class_to_hazard = dict(DEFAULT_CLASS_TO_HAZARD)
    class_to_hazard.update((cls_cfg or {}).get("hazard_map", {}))

    detectors = []
    hz = Path(args.hazard_model)
    if not args.no_hazard and hz.exists():
        detectors.append(OnnxDetector(hz, _labels_for(hz, args.hazard_labels), "hazard", args.conf, args.iou,
                                      threads=args.threads))
    elif not args.no_hazard:
        print(f"[warn] hazard model not found at {hz}; running without it", file=sys.stderr)
    cc = Path(args.coco_model)
    if not args.no_coco and cc.exists():
        detectors.append(OnnxDetector(cc, _labels_for(cc, args.coco_labels), "coco", args.conf, args.iou,
                                      threads=args.threads))
    elif not args.no_coco:
        print(f"[warn] coco model not found at {cc}; running without it", file=sys.stderr)
    if not detectors:
        raise SystemExit("no models available; run tools/export_onnx.py first")

    cfg = ReportConfig(window=args.window, min_hits=args.min_hits, min_confidence=args.report_conf,
                       cooldown_s=args.cooldown, global_cooldown_s=args.global_cooldown,
                       class_to_hazard=class_to_hazard, stopped_vehicle_enabled=args.stopped_vehicle)
    manager = ReportManager(cfg, source=args.event_source)
    sinks = build_sinks(args.sink or ["stdout"], adb=args.adb, package=args.adb_package, dry_run=args.adb_dry_run)

    sim = None
    if args.camera_sim:
        w, h = (int(v) for v in args.sim_res.lower().split("x"))
        sim = CameraSim(width=w, height=h)
    if args.camera:
        source = CameraSource(int(args.source), step=args.step, sim=sim)
    else:
        source = VideoSource(args.source, step=args.step, loop=args.loop, start_s=args.start,
                             max_frames=args.max_frames, sim=sim)
        print(f"[info] {args.source}: {source.width}x{source.height} @ {source.fps:.1f} fps, "
              f"{source.frame_count} frames, inferring every {args.step} -> {source.effective_fps:.1f}/s",
              file=sys.stderr)
    for d in detectors:
        print(f"[info] model {d.name}: input {d.size}, {len(d.labels)} classes", file=sys.stderr)

    pipe = Pipeline(detectors, manager, sinks, show=args.show, record_path=args.record,
                    record_fps=source.effective_fps)
    stats = pipe.run(source)
    source.release()
    summary = stats.summary()
    print("[stats] " + json.dumps(summary), file=sys.stderr)
    if args.stats_json:
        Path(args.stats_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
