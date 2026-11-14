"""Build the detector stack from command-line style options.

Shared by ``run_video.py`` (live pipeline) and ``tools/eval_events.py`` (evaluation) so both
run exactly the same models, fusion and thresholds::

    hazard models  -> [OnnxDetector] -> (ZoomDetector each) -> EnsembleDetector (WBF) -> ClassFilter
    coco model     -> OnnxDetector   -> (ZoomDetector)

A single hazard model without zoom stays a plain ``OnnxDetector`` (no fusion overhead).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .detector import ClassFilter, EnsembleDetector, OnnxDetector, ZoomDetector
from .fusion import parse_class_thresholds

ROOT = Path(__file__).resolve().parents[1]


def labels_for(model_path: Path, explicit: str | None = None) -> list[str]:
    if explicit:
        return [ln.strip() for ln in Path(explicit).read_text(encoding="utf-8").splitlines() if ln.strip()]
    cand = model_path.with_suffix(".labels.txt")
    if cand.exists():
        return [ln.strip() for ln in cand.read_text(encoding="utf-8").splitlines() if ln.strip()]
    raise SystemExit(f"no labels file for {model_path} (expected {cand})")


def parse_crop(spec: str | None) -> tuple[float, float, float, float] | None:
    if not spec:
        return None
    crop = tuple(float(v) for v in spec.split(","))
    if len(crop) != 4:
        raise SystemExit("--zoom-crop wants x1,y1,x2,y2 as fractions, e.g. 0.25,0.35,0.75,0.75")
    return crop  # type: ignore[return-value]


@dataclass
class DetectorSpec:
    hazard_models: list[Path] = field(default_factory=list)   # one or more; >1 = ensemble
    hazard_labels: str | None = None
    coco_model: Path | None = None
    coco_labels: str | None = None
    conf: float = 0.35
    iou: float = 0.5
    threads: int = 4
    zoom_crop: tuple[float, float, float, float] | None = None
    zoom_coco: bool = False          # also zoom the COCO model (costly, rarely useful)
    fusion: str = "wbf"              # wbf | nms
    weights: list[float] | None = None
    conf_type: str = "avg"
    fusion_iou: float = 0.55
    class_conf: dict[str, float] = field(default_factory=dict)
    parallel: bool = True

    @classmethod
    def from_args(cls, args) -> "DetectorSpec":
        """Read the shared CLI flags added by ``add_detector_args``."""
        hz: list[Path] = []
        if not getattr(args, "no_hazard", False):
            hz.append(Path(args.hazard_model))
            hz += [Path(p) for p in (getattr(args, "ensemble_model", None) or [])]
        weights = None
        if getattr(args, "ensemble_weights", None):
            weights = [float(v) for v in args.ensemble_weights.split(",")]
        return cls(
            hazard_models=hz,
            hazard_labels=getattr(args, "hazard_labels", None),
            coco_model=None if getattr(args, "no_coco", False) else Path(args.coco_model),
            coco_labels=getattr(args, "coco_labels", None),
            conf=args.conf, iou=args.iou, threads=args.threads,
            zoom_crop=None if getattr(args, "no_zoom", False) else parse_crop(getattr(args, "zoom_crop", None)),
            zoom_coco=bool(getattr(args, "zoom_coco", False)),
            fusion=getattr(args, "fusion", "wbf"), weights=weights,
            conf_type=getattr(args, "conf_type", "avg"), fusion_iou=getattr(args, "fusion_iou", 0.55),
            class_conf=parse_class_thresholds(getattr(args, "class_conf", None)),
            parallel=not getattr(args, "no_parallel", False),
        )

    def describe(self) -> str:
        parts = [f"hazard={[p.name for p in self.hazard_models]}"]
        if len(self.hazard_models) > 1:
            parts.append(f"fusion={self.fusion}/{self.conf_type} weights={self.weights or 'equal'}")
        if self.zoom_crop:
            parts.append(f"zoom={self.zoom_crop}")
        if self.class_conf:
            parts.append(f"class_conf={self.class_conf}")
        if self.coco_model:
            parts.append(f"coco={self.coco_model.name}")
        return " ".join(parts)


def add_detector_args(ap, default_hazard: Path | None = None, default_coco: Path | None = None) -> None:
    """Attach the shared detector flags to an argparse parser."""
    ap.add_argument("--hazard-model", default=str(default_hazard or ROOT / "models" / "hazard_yolo11n.onnx"))
    ap.add_argument("--hazard-labels")
    ap.add_argument("--ensemble-model", action="append", default=[],
                    help="extra hazard ONNX model(s) fused with --hazard-model (repeatable)")
    ap.add_argument("--ensemble-weights", help="comma list, one weight per hazard model (default equal)")
    ap.add_argument("--fusion", default="wbf", choices=["wbf", "nms"], help="how ensemble boxes are merged")
    ap.add_argument("--conf-type", default="avg", choices=["avg", "max", "box_avg"],
                    help="WBF confidence: avg penalises boxes missing from a member (default)")
    ap.add_argument("--fusion-iou", type=float, default=0.55)
    ap.add_argument("--class-conf", help="per-label confidence floors after fusion, e.g. pothole=0.5,cone=0.4")
    ap.add_argument("--coco-model", default=str(default_coco or ROOT / "models" / "coco_yolo11n.onnx"))
    ap.add_argument("--coco-labels")
    ap.add_argument("--no-coco", action="store_true", help="skip the stock COCO detector")
    ap.add_argument("--no-hazard", action="store_true", help="skip the custom hazard detector")
    ap.add_argument("--conf", type=float, default=0.35, help="detector confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=4, help="ORT intra-op threads per model (phone-like: 4)")
    ap.add_argument("--zoom-crop", nargs="?", const="0.25,0.35,0.75,0.75", default=None,
                    help="also run the hazard model(s) on a zoomed crop of the road ahead (x1,y1,x2,y2 "
                         "fractions); one extra inference per model, finds small distant hazards. "
                         "run_video.py turns it on by default (pass --no-zoom to disable); eval_events.py "
                         "leaves it off unless asked, so old comparisons stay valid")
    ap.add_argument("--no-zoom", action="store_true", help="disable the zoom pass where it is on by default")
    ap.add_argument("--zoom-coco", action="store_true", help="apply the zoom pass to the COCO model too")
    ap.add_argument("--no-parallel", action="store_true", help="run models one after another (debugging)")


def build_hazard_detector(spec: DetectorSpec):
    """Hazard side only: plain detector, zoomed, ensemble, with class floors. None if no model."""
    members = []
    for p in spec.hazard_models:
        if not p.exists():
            # running on with the COCO model alone looks like clean footage: none of the hazard
            # classes can fire. Asked for a hazard model and it is not there -> stop.
            raise SystemExit(f"hazard model not found: {p}\n"
                             f"  models are not committed; see models/README.md for how to build them, "
                             f"or pass --no-hazard to run the stock COCO detector alone")
        det = OnnxDetector(p, labels_for(p, spec.hazard_labels), f"hazard:{p.stem}", spec.conf, spec.iou,
                           threads=spec.threads)
        if spec.zoom_crop:
            det = ZoomDetector(det, spec.zoom_crop, spec.iou)
        members.append(det)
    if not members:
        return None
    if len(members) == 1:
        det = members[0]
        det.name = "hazard"
        return ClassFilter(det, spec.class_conf) if spec.class_conf else det
    return EnsembleDetector(members, spec.weights, spec.fusion, spec.fusion_iou, spec.conf_type,
                            parallel=spec.parallel, class_conf=spec.class_conf, name="hazard")


def build_detectors(spec: DetectorSpec) -> list:
    """Everything the pipeline runs per frame: [hazard stack, coco]."""
    out = []
    hz = build_hazard_detector(spec)
    if hz is not None:
        out.append(hz)
    if spec.coco_model is not None:
        if spec.coco_model.exists():
            coco = OnnxDetector(spec.coco_model, labels_for(spec.coco_model, spec.coco_labels), "coco",
                                spec.conf, spec.iou, threads=spec.threads)
            if spec.zoom_crop and spec.zoom_coco:
                coco = ZoomDetector(coco, spec.zoom_crop, spec.iou)
            out.append(coco)
        else:
            print(f"[warn] coco model not found at {spec.coco_model}; running without it", file=sys.stderr)
    return out


DEFAULT_VEHICLE_MODEL = ROOT / "models" / "vehicle_cls.onnx"
DEFAULT_VEHICLE_MODEL2 = ROOT / "models" / "vehicle_cls2.onnx"     # averaged with the first when present
DEFAULT_WEATHER_MODEL = ROOT / "models" / "weather_cls.onnx"
DEFAULT_SURFACE_MODEL = ROOT / "models" / "surface_cls.onnx"
SCENE_FEATURES = ("traffic", "crash", "lanes", "signals", "weather")


def add_scene_args(ap) -> None:
    """Flags for the second stage (vehicle classifier), the weather classifier and the scene rules."""
    ap.add_argument("--vehicle-model", help=f"police / ambulance / fire classifier for COCO vehicles; a comma list "
                                            f"averages several (default {DEFAULT_VEHICLE_MODEL.relative_to(ROOT)}, "
                                            f"plus {DEFAULT_VEHICLE_MODEL2.relative_to(ROOT)} if present)")
    ap.add_argument("--no-vehicle", action="store_true", help="no vehicle classifier: no police, ambulance, fire")
    ap.add_argument("--vehicle-crops", type=int, default=4, help="largest vehicles classified per frame")
    ap.add_argument("--weather-model", help=f"whole-frame weather classifier "
                                            f"(default {DEFAULT_WEATHER_MODEL.relative_to(ROOT)} if present)")
    ap.add_argument("--surface-model", help=f"road-surface classifier (default "
                                            f"{DEFAULT_SURFACE_MODEL.relative_to(ROOT)} if present)")
    ap.add_argument("--no-weather", action="store_true", help="no weather and no road-surface classifier")
    ap.add_argument("--no-scene", action="store_true", help="no scene rules (traffic, crash, lanes, signals, weather)")
    ap.add_argument("--scene-features", default=",".join(SCENE_FEATURES),
                    help="comma list of scene rules to run: " + ",".join(SCENE_FEATURES))
    ap.add_argument("--police-other-side", action="store_true",
                    help="report a police car far on the opposite side as Police > Other side")
    ap.add_argument("--drive-side", default="right", choices=["right", "left"])
    ap.add_argument("--camera-geometry", metavar="HFOV,HEIGHT,HORIZON",
                    help="camera geometry for the blocked-lane rule: horizontal field of view in degrees, mount "
                         "height in metres, horizon row as a fraction of the frame (default 70,1.3,0.5)")


def _optional_model(explicit: str | None, default: Path, what: str):
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"{what} not found: {p} (see models/README.md)")
        return p
    if default.exists():
        return default
    print(f"[warn] {what} not found at {default}; reports that need it are off", file=sys.stderr)
    return None


def build_second_stage(args):
    """(vehicle classifier, weather classifier, scene analyzer) from the add_scene_args flags; any may be None."""
    from .classifiers import RoadSurface, SceneClassifier, SceneClassifiers, VehicleClassifier
    from .scene import SceneAnalyzer, SceneConfig

    vehicle = weather = scene = None
    geo = {}
    if getattr(args, "camera_geometry", None):
        hfov, height, horizon = (float(v) for v in args.camera_geometry.split(","))
        geo = {"hfov_deg": hfov, "cam_height_m": height, "horizon_y": horizon}
    if not getattr(args, "no_vehicle", False) and not getattr(args, "no_coco", False):
        explicit = getattr(args, "vehicle_model", None)
        if explicit and "," in explicit:
            paths = [_optional_model(m.strip(), DEFAULT_VEHICLE_MODEL, "vehicle classifier") for m in explicit.split(",")]
        else:
            p = _optional_model(explicit, DEFAULT_VEHICLE_MODEL, "vehicle classifier")
            paths = [p] if p is not None else []
            if p is not None and not explicit and DEFAULT_VEHICLE_MODEL2.exists():
                paths.append(DEFAULT_VEHICLE_MODEL2)
        if paths:
            vehicle = VehicleClassifier(paths, max_crops=getattr(args, "vehicle_crops", 4))
    if not getattr(args, "no_weather", False):
        p = _optional_model(getattr(args, "weather_model", None), DEFAULT_WEATHER_MODEL, "weather classifier")
        q = _optional_model(getattr(args, "surface_model", None), DEFAULT_SURFACE_MODEL, "road-surface classifier")
        surface = None
        if q is not None:
            surface = RoadSurface(q)
            if "damp" in surface.clf.labels:       # a RoadSaW / RoadSC model: it wants the bird's-eye patch
                cfg = SceneConfig(**geo)
                surface.topview = (cfg.hfov_deg, cfg.cam_height_m, cfg.horizon_y)
        parts = [SceneClassifier(p) if p is not None else None, surface]
        if any(parts):
            weather = SceneClassifiers(*parts)
    if not getattr(args, "no_scene", False):
        feats = tuple(f.strip() for f in str(getattr(args, "scene_features", "")).split(",") if f.strip())
        unknown = [f for f in feats if f not in SCENE_FEATURES]
        if unknown:
            raise SystemExit(f"unknown --scene-features {unknown}; choose from {SCENE_FEATURES}")
        scene = SceneAnalyzer(SceneConfig(enabled=feats, drive_side=getattr(args, "drive_side", "right"),
                                          police_other_side=bool(getattr(args, "police_other_side", False)), **geo))
    return vehicle, weather, scene
