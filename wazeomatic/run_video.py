"""Desktop simulator: run the full detection pipeline on a video (or webcam).

Examples
    python -m wazeomatic.run_video assets/videos/dashcam_urban_1080p.mp4
    python -m wazeomatic.run_video clip.mp4 --camera-sim --show --record runs/sim/out.mp4
    python -m wazeomatic.run_video clip.mp4 --sink tcp:192.168.1.20:8765        # to the phone app
    python -m wazeomatic.run_video clip.mp4 --sink adb                          # straight to emulator (dry_run=true)
    python -m wazeomatic.run_video 0 --camera                                    # webcam / Pi camera
    python -m wazeomatic.run_video clip.mp4 --ensemble-model models/hazard_yolo11n_416.onnx --zoom-crop \
        --track-iou 0.3 --class-conf cone=0.5                                    # fused + tracked
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


from .build import DetectorSpec, add_detector_args, add_scene_args, build_detectors, build_second_stage
from .fusion import parse_class_thresholds
from .hood import HoodEstimator
from .pipeline import Pipeline
from .report_manager import ReportConfig, ReportManager, default_label_floor_spec
from .schema import load_class_to_hazard
from .sinks import build_sinks
from .sources import CameraSim, CameraSource, Prefetch, VideoSource

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="video path, or camera index with --camera")
    ap.add_argument("--camera", action="store_true")
    add_detector_args(ap)
    add_scene_args(ap)
    ap.add_argument("--classes", default=str(ROOT / "configs" / "classes.yaml"))
    ap.add_argument("--prefetch", type=int, default=2, help="frames decoded ahead on a background thread (0 = off)")
    ap.add_argument("--step", type=int, default=3, help="use every Nth frame (30 fps / 3 = 10 inferences/s)")
    ap.add_argument("--start", type=float, default=0.0, help="start offset in seconds")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--camera-sim", action="store_true", help="degrade frames like a phone camera")
    ap.add_argument("--sim-res", default="1280x720")
    # report manager
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--report-conf", type=float, default=0.40)
    ap.add_argument("--cooldown", type=float, default=60.0)
    ap.add_argument("--global-cooldown", type=float, default=60.0,
                    help="seconds between any two events (the macro files at most one report a minute)")
    ap.add_argument("--report-class-conf", default=default_label_floor_spec(),
                    help="per-label hit floors for the report manager, e.g. cone=0.55 "
                         "(default: the shipped floors; pass an empty string for none)")
    ap.add_argument("--track-iou", type=float, default=0.0,
                    help="> 0: hits must come from boxes overlapping frame to frame (spatial tracking)")
    ap.add_argument("--track-grow", type=float, default=1.5, help="box growth factor for the tracking IoU test")
    ap.add_argument("--hood-frac", type=float, default=0.10, help="minimum hood/dashboard zone (fraction of height)")
    ap.add_argument("--no-auto-hood", action="store_true",
                    help="do not grow the hood zone from the static bottom rows of the video (wazeomatic/hood.py)")
    ap.add_argument("--stopped-vehicle", action="store_true", help="enable experimental shoulder heuristic")
    ap.add_argument("--event-source", default="sim", choices=["sim", "pi", "phone"])
    # sinks
    ap.add_argument("--sink", action="append", default=[],
                    help="stdout | jsonl:<path> | tcp:<host>[:port] | adb[:<serial>] | http:<url> (repeatable)")
    ap.add_argument("--adb", default="adb")
    ap.add_argument("--adb-package", help="explicit target package for the broadcast, e.g. com.arlosoft.macrodroid")
    ap.add_argument("--adb-live", action="store_true",
                    help="send dry_run=false on adb broadcasts (the macro may then file a real report if "
                         "its own DRY_RUN is off too); without it every broadcast carries dry_run=true")
    ap.add_argument("--adb-dry-run", action="store_true",
                    help="accepted for old command lines; dry run is already the default")
    # read-only Waze screen
    ap.add_argument("--waze-reader", nargs="?", const="", default=None, metavar="SERIAL",
                    help="read the Waze (or mock Waze) screen of an adb device, read only, and do not send a "
                         "hazard Waze already shows nearby; optional device serial")
    ap.add_argument("--waze-interval", type=float, default=2.0, help="seconds between screen reads")
    # output
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--record", help="write annotated video here")
    ap.add_argument("--stats-json", help="write run statistics here")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    class_to_hazard = load_class_to_hazard(args.classes)

    if args.zoom_crop is None and not args.no_zoom:
        args.zoom_crop = "0.25,0.35,0.75,0.75"  # the measured operating point: zoom pass on
    spec = DetectorSpec.from_args(args)
    detectors = build_detectors(spec)
    if not detectors:
        raise SystemExit("no models available; see models/README.md for how to get or build them")
    print(f"[info] detectors: {spec.describe()}", file=sys.stderr)

    cfg = ReportConfig(window=args.window, min_hits=args.min_hits, min_confidence=args.report_conf,
                       min_confidence_by_label=parse_class_thresholds(args.report_class_conf),
                       track_iou=args.track_iou, track_grow=args.track_grow,
                       cooldown_s=args.cooldown, global_cooldown_s=args.global_cooldown, hood_frac=args.hood_frac,
                       class_to_hazard=class_to_hazard, stopped_vehicle_enabled=args.stopped_vehicle)
    if getattr(args, "camera_geometry", None):
        cfg.road_top = float(args.camera_geometry.split(",")[2]) - 0.15     # debris must lie below the horizon
    waze = reader = None
    if args.waze_reader is not None:
        from .waze_reader import AdbWazeReader, WazeAlerts

        waze = WazeAlerts()
        reader = AdbWazeReader(waze, serial=args.waze_reader or None, adb=args.adb, interval_s=args.waze_interval,
                               on_new=lambda a: print("[waze] " + json.dumps(a.to_dict()), file=sys.stderr)).start()
    manager = ReportManager(cfg, source=args.event_source, waze=waze)
    vehicle_clf, weather_clf, scene = build_second_stage(args)
    for name, part in (("vehicle classifier", vehicle_clf), ("weather classifier", weather_clf)):
        if part is not None:
            print(f"[info] {name}: {part.clf.path.name}, {part.clf.labels}", file=sys.stderr)
    hood = None if args.no_auto_hood else HoodEstimator(min_frac=args.hood_frac)
    if args.adb_live and args.adb_dry_run:
        raise SystemExit("--adb-live and --adb-dry-run contradict each other")
    sinks = build_sinks(args.sink or ["stdout"], adb=args.adb, package=args.adb_package, dry_run=not args.adb_live)

    sim = None
    if args.camera_sim:
        w, h = (int(v) for v in args.sim_res.lower().split("x"))
        sim = CameraSim(width=w, height=h)
    if args.camera:
        source = CameraSource(int(args.source), step=args.step, sim=sim, max_frames=args.max_frames)
    else:
        source = VideoSource(args.source, step=args.step, loop=args.loop, start_s=args.start,
                             max_frames=args.max_frames, sim=sim)
        print(f"[info] {args.source}: {source.width}x{source.height} @ {source.fps:.1f} fps, "
              f"{source.frame_count} frames, inferring every {args.step} -> {source.effective_fps:.1f}/s",
              file=sys.stderr)
    for d in detectors:
        print(f"[info] model {d.name}: input {d.size}, {len(d.labels)} classes", file=sys.stderr)
    if args.prefetch > 0:
        source = Prefetch(source, depth=args.prefetch)

    pipe = Pipeline(detectors, manager, sinks, show=args.show, record_path=args.record,
                    record_fps=source.effective_fps, parallel=not args.no_parallel, hood=hood,
                    vehicle_classifier=vehicle_clf, scene_classifier=weather_clf, scene=scene)
    stats = pipe.run(source)
    if reader is not None:
        reader.stop()
    for c in manager.confirmations:
        print("[confirmed] " + json.dumps(c), file=sys.stderr)
    source.release()
    summary = stats.summary()
    print("[stats] " + json.dumps(summary), file=sys.stderr)
    if args.stats_json:
        Path(args.stats_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if stats.error:
        # a dead camera or a decoder error must not look like the end of the clip
        print(f"[error] run ended early: {stats.error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
