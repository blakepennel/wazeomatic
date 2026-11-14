"""Image classifiers on top of the detectors: what kind of vehicle a box is, and what the weather is.

Both are Ultralytics ``-cls`` models exported to ONNX by ``tools/train_classifier.py``: input ``[N, 3, S, S]``
RGB in [0, 1], output ``[N, C]`` class probabilities (softmax inside the graph). The class names are in the
``.labels.txt`` next to the model, one per line, in output order.

* ``VehicleClassifier`` takes the car / truck / bus boxes of the stock detector, crops the largest few, pads
  each crop to a square and asks the model whether it is a civilian vehicle, a police car, an ambulance or a
  fire engine. A police car comes back as a ``police_vehicle`` detection with the box of the vehicle, so the
  report manager treats it like any other hazard class (3 hits in 10 frames, label floor, cooldown).
* ``SceneClassifier`` classifies the whole frame (clear, rain, snow, fog) a few times a second.

The Kotlin twins are ``VehicleClassifier.kt`` and ``SceneClassifier.kt``; keep the crop rule in sync.
"""
from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from .schema import Detection

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None


def read_labels(model_path: str | Path) -> list[str]:
    p = Path(model_path)
    lab = p.with_suffix(".labels.txt")
    if not lab.exists():
        raise SystemExit(f"labels file missing next to the classifier: {lab}")
    return [ln.strip() for ln in lab.read_text(encoding="utf-8").splitlines() if ln.strip()]


def square_crop(img: np.ndarray, bbox: tuple[float, float, float, float], pad: float = 0.12,
                size: int = 128, fill: int = 114) -> np.ndarray:
    """Crop a normalised xywh box grown by ``pad`` on each side, letterbox it into ``size`` x ``size``.

    The same rule builds the training crops (tools/build_vehicle_crops.py), so the model sees at run time
    exactly what it was trained on."""
    h, w = img.shape[:2]
    cx, cy, bw, bh = bbox
    bw, bh = bw * (1 + 2 * pad), bh * (1 + 2 * pad)
    # clamp the corner inside the frame: a box past the right / bottom edge still gives a 1 px crop
    x1, y1 = int(min(w - 1, max(0, (cx - bw / 2) * w))), int(min(h - 1, max(0, (cy - bh / 2) * h)))
    x2, y2 = int(min(w, (cx + bw / 2) * w)), int(min(h, (cy + bh / 2) * h))
    crop = img[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)]
    ch, cw = crop.shape[:2]
    s = size / max(ch, cw)
    nw, nh = max(1, int(round(cw * s))), max(1, int(round(ch * s)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    out = np.full((size, size, 3), fill, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    out[py:py + nh, px:px + nw] = resized
    return out


def trained_size(meta: dict) -> Optional[int]:
    """Square input size from Ultralytics export metadata ("imgsz": "[160, 160]" or "160"), else None."""
    nums = re.findall(r"\d+", str(meta.get("imgsz", "")))
    return int(nums[0]) if nums and int(nums[0]) > 0 else None


class OnnxClassifier:
    def __init__(self, model_path: str | Path, threads: int = 2, providers: Optional[Sequence[str]] = None,
                 size: Optional[int] = None):
        if ort is None:
            raise SystemExit("onnxruntime is not installed")
        self.path = Path(model_path)
        if not self.path.exists():
            raise SystemExit(f"classifier not found: {self.path} (see models/README.md)")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(self.path), so, providers=list(providers or ["CPUExecutionProvider"]))
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # a fixed input wins; a dynamic one (train_classifier exports dynamic=True) runs at the size the model
        # was trained at, which Ultralytics writes into the metadata as "imgsz": "[160, 160]"
        if isinstance(inp.shape[2], int):
            self.size = int(inp.shape[2])
        else:
            self.size = int(size or trained_size(self.session.get_modelmeta().custom_metadata_map) or 128)
        self.dynamic_batch = not isinstance(inp.shape[0], int)
        self.labels = read_labels(self.path)
        self.last_ms = 0.0

    def predict(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """[N, C] probabilities for N square BGR crops already at ``self.size``."""
        if not crops_bgr:
            return np.zeros((0, len(self.labels)), np.float32)
        t0 = time.perf_counter()
        x = np.stack([c[:, :, ::-1] for c in crops_bgr]).astype(np.float32) / 255.0
        x = np.ascontiguousarray(x.transpose(0, 3, 1, 2))
        if self.dynamic_batch:
            out = self.session.run(None, {self.input_name: x})[0]
        else:
            out = np.concatenate([self.session.run(None, {self.input_name: x[i:i + 1]})[0] for i in range(len(x))])
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        out = np.asarray(out, np.float32)
        if out.min() < 0 or abs(float(out[0].sum()) - 1.0) > 0.05:      # logits: softmax here
            e = np.exp(out - out.max(axis=1, keepdims=True))
            out = e / e.sum(axis=1, keepdims=True)
        return out


VEHICLE_LABELS = ("car", "truck", "bus")
# classifier class -> detection label handed to the report manager / scene rules
VEHICLE_CLASS_TO_LABEL = {"police": "police_vehicle", "ambulance": "ambulance", "fire": "fire_truck"}


class VehicleClassifier:
    """Second stage for the stock detector's vehicles. ``max_crops`` keeps the cost bounded: the largest boxes
    are the closest vehicles and the only ones whose livery and light bar are resolvable."""

    name = "vehicle"

    def __init__(self, model_path: str | Path | Sequence[str | Path], max_crops: int = 4, min_box_px: int = 28,
                 min_vehicle_conf: float = 0.35, threads: int = 2, pad: float = 0.12):
        # several models (same classes) are averaged: two models trained on different hard negatives make
        # different mistakes on look-alikes, so the mean is far more often below the police floor on them
        paths = [model_path] if isinstance(model_path, (str, Path)) else list(model_path)
        self.clfs = [OnnxClassifier(m, threads=threads) for m in paths]
        self.clf = self.clfs[0]
        for c in self.clfs[1:]:
            if sorted(c.labels) != sorted(self.clf.labels):
                raise SystemExit(f"vehicle classifiers disagree on classes: {c.path} {c.labels} vs {self.clf.labels}")
        self.max_crops = max_crops
        self.min_box_px = min_box_px
        self.min_vehicle_conf = min_vehicle_conf
        self.pad = pad
        self.last_ms = 0.0
        self.last_probs: list[tuple[Detection, dict[str, float]]] = []

    def candidates(self, image: np.ndarray, dets: Sequence[Detection]) -> list[Detection]:
        h, w = image.shape[:2]
        c = [d for d in dets if d.label in VEHICLE_LABELS and d.confidence >= self.min_vehicle_conf
             and min(d.bbox[2] * w, d.bbox[3] * h) >= self.min_box_px]
        return sorted(c, key=lambda d: -(d.bbox[2] * d.bbox[3]))[: self.max_crops]

    def __call__(self, image: np.ndarray, dets: Sequence[Detection]) -> list[Detection]:
        cand = self.candidates(image, dets)
        self.last_probs = []
        if not cand:
            self.last_ms = 0.0
            return []
        sums = [dict.fromkeys(self.clf.labels, 0.0) for _ in cand]
        self.last_ms = 0.0
        crops: dict[int, list[np.ndarray]] = {}
        for clf in self.clfs:
            if clf.size not in crops:
                crops[clf.size] = [square_crop(image, d.bbox, self.pad, clf.size) for d in cand]
            for acc, p in zip(sums, clf.predict(crops[clf.size])):
                for lab, v in zip(clf.labels, p):
                    acc[lab] += float(v)
            self.last_ms += clf.last_ms
        out: list[Detection] = []
        for d, acc in zip(cand, sums):
            pd = {lab: v / len(self.clfs) for lab, v in acc.items()}
            self.last_probs.append((d, pd))
            for cls, label in VEHICLE_CLASS_TO_LABEL.items():
                if cls in pd and pd[cls] >= 0.05:
                    out.append(Detection(label=label, confidence=pd[cls], bbox=d.bbox, model="vehicle"))
        return out


class SceneClassifier:
    """Whole-frame weather classifier, run every ``every_s`` seconds (the weather does not change per frame)."""

    name = "scene"

    def __init__(self, model_path: str | Path, every_s: float = 0.5, threads: int = 1):
        self.clf = OnnxClassifier(model_path, threads=threads)
        self.every_s = every_s
        self._last_t = -1e9
        self.last: dict[str, float] = {}
        self.last_ms = 0.0

    def __call__(self, image: np.ndarray, t: float) -> Optional[dict[str, float]]:
        if t - self._last_t < self.every_s:
            self.last_ms = 0.0
            return None
        self._last_t = t
        s = self.clf.size
        img = cv2.resize(image, (s, s), interpolation=cv2.INTER_AREA)   # whole frame, squashed, as in training
        p = self.clf.predict([img])[0]
        self.last_ms = self.clf.last_ms
        self.last = {lab: float(v) for lab, v in zip(self.clf.labels, p)}
        return self.last


# The patch of road the surface classifier looks at: centre of the lane, just above the hood zone.
ROAD_PATCH = (0.40, 0.60, 0.60, 0.86)     # x1, y1, x2, y2 as fractions of the frame


# boxes that hide the road: a top-view patch they overlap is not classified
OCCLUDERS = ("car", "truck", "bus", "motorcycle", "bicycle", "person")


def topview_quad(w: int, h: int, hfov_deg: float, height_m: float, horizon_y: float, dist_m: float,
                 size_m: float) -> np.ndarray:
    """Image corners (far left, far right, near right, near left; pixels) of a ``size_m`` square of flat road
    centred ``dist_m`` ahead of the camera: the same pinhole model as the blocked-lane rule (scene.py)."""
    fx = 0.5 / math.tan(math.radians(hfov_deg) / 2)
    fy = fx * w / h
    near, far, half = dist_m - size_m / 2, dist_m + size_m / 2, size_m / 2

    def px(x: float, z: float) -> list[float]:
        return [(0.5 + fx * x / z) * w, (horizon_y + fy * height_m / z) * h]
    return np.float32([px(-half, far), px(half, far), px(half, near), px(-half, near)])


class RoadSurface:
    """Road-surface classifier on the road ahead. Class names come back prefixed ``surface_`` so they cannot
    collide with the weather classes.

    Two kinds of model. A **top-view** model (``topview=(hfov, height, horizon)``; trained on RoadSaW / RoadSC,
    classes dry, damp, wet, snow, slush) sees the road the way those datasets were made: a 3.6 m square 15 m
    ahead, warped to a bird's-eye patch with the flat-road camera model. A frame where a vehicle or a person
    covers that square is skipped. Otherwise (RSCD-style models) the classifier gets a fixed patch of the frame
    just above the hood zone, which is moved up when the learnt hood zone is taller than the default."""

    name = "surface"

    def __init__(self, model_path: str | Path, every_s: float = 0.5, threads: int = 1,
                 patch: tuple[float, float, float, float] = ROAD_PATCH,
                 topview: Optional[tuple[float, float, float]] = None, dist_m: float = 15.0, size_m: float = 3.6,
                 max_occlusion: float = 0.10):
        self.clf = OnnxClassifier(model_path, threads=threads)
        self.every_s = every_s
        self.patch = patch
        self.topview = topview
        self.dist_m, self.size_m, self.max_occlusion = dist_m, size_m, max_occlusion
        self.hood_frac = 0.0
        self._last_t = -1e9
        self.last: dict[str, float] = {}
        self.last_ms = 0.0
        self.skipped = 0

    def occluded(self, quad: np.ndarray, w: int, h: int, dets: Sequence[Detection]) -> bool:
        x1, y1 = quad[:, 0].min() / w, quad[:, 1].min() / h
        x2, y2 = quad[:, 0].max() / w, quad[:, 1].max() / h
        area = max((x2 - x1) * (y2 - y1), 1e-9)
        for d in dets:
            if d.label not in OCCLUDERS:
                continue
            cx, cy, bw, bh = d.bbox
            ix = max(0.0, min(x2, cx + bw / 2) - max(x1, cx - bw / 2))
            iy = max(0.0, min(y2, cy + bh / 2) - max(y1, cy - bh / 2))
            if ix * iy / area > self.max_occlusion:
                return True
        return False

    def region(self, h: int, w: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.patch
        bottom = min(y2, 1.0 - max(self.hood_frac, 0.10) - 0.02)
        top = max(0.45, bottom - (y2 - y1))
        return int(x1 * w), int(top * h), int(x2 * w), int(bottom * h)

    def __call__(self, image: np.ndarray, t: float, dets: Sequence[Detection] = ()) -> Optional[dict[str, float]]:
        if t - self._last_t < self.every_s:
            self.last_ms = 0.0
            return None
        h, w = image.shape[:2]
        s = self.clf.size
        if self.topview is not None:
            quad = topview_quad(w, h, *self.topview, self.dist_m, self.size_m)
            if quad[:, 1].max() > (1.0 - max(self.hood_frac, 0.10)) * h or self.occluded(quad, w, h, dets):
                self.last_ms = 0.0
                self.skipped += 1
                return None                   # try again on the next frame
            m = cv2.getPerspectiveTransform(quad, np.float32([[0, 0], [s, 0], [s, s], [0, s]]))
            img = cv2.warpPerspective(image, m, (s, s), flags=cv2.INTER_LINEAR)
        else:
            x1, y1, x2, y2 = self.region(h, w)
            img = cv2.resize(image[y1:y2, x1:x2], (s, s), interpolation=cv2.INTER_AREA)
        self._last_t = t
        p = self.clf.predict([img])[0]
        self.last_ms = self.clf.last_ms
        self.last = {f"surface_{lab}": float(v) for lab, v in zip(self.clf.labels, p)}
        return self.last


class SceneClassifiers:
    """Weather and road surface together, as one ``(image, t) -> probs`` callable for the pipeline."""

    def __init__(self, *parts):
        self.parts = [p for p in parts if p is not None]
        self.last_ms = 0.0

    @property
    def clf(self):                      # for log lines that print the first model
        return self.parts[0].clf

    def set_hood_frac(self, frac: float) -> None:
        for p in self.parts:
            if hasattr(p, "hood_frac"):
                p.hood_frac = frac

    def __call__(self, image: np.ndarray, t: float, dets: Sequence[Detection] = ()) -> Optional[dict[str, float]]:
        merged: dict[str, float] = {}
        ms = 0.0
        for p in self.parts:
            r = p(image, t, dets) if isinstance(p, RoadSurface) else p(image, t)
            ms += p.last_ms
            if r:
                merged.update(r)
        self.last_ms = ms
        return merged or None
