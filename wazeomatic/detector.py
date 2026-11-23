"""ONNX Runtime wrapper for Ultralytics-style YOLO detection models.

Works with any model exported by ``tools/export_onnx.py`` (output shape
``[1, 4 + num_classes, num_anchors]``, boxes in letterboxed pixel xywh). The same
pre/post-processing is mirrored in Kotlin in the Android app, so keep them in sync.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .schema import Detection

try:  # onnxruntime is optional at import time so tests can run without a model
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None


@dataclass
class Letterbox:
    scale: float
    pad_x: float
    pad_y: float
    in_w: int
    in_h: int
    src_w: int
    src_h: int


def rect_shape(h: int, w: int, size: int, stride: int = 32) -> tuple[int, int]:
    """Smallest (height, width) input, in multiples of ``stride``, that holds an ``h`` x ``w`` image scaled so its
    long side is ``size``: (384, 640) for a 16:9 frame at 640, (320, 640) for the 640 x 288 zoom crop. The image is
    seen at the same scale as in the square ``size`` x ``size`` letterbox; only the grey padding is left out."""
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return -(-nh // stride) * stride, -(-nw // stride) * stride


def letterbox(img: np.ndarray, size: int, fill: int = 114,
              shape: tuple[int, int] | None = None) -> tuple[np.ndarray, Letterbox]:
    """Resize keeping aspect ratio and pad to ``size`` x ``size``, or to ``shape`` = (height, width) when given
    (BGR uint8 in/out)."""
    h, w = img.shape[:2]
    oh, ow = shape or (size, size)
    scale = min(oh / h, ow / w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    import cv2

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((oh, ow, 3), fill, dtype=np.uint8)
    px, py = (ow - nw) // 2, (oh - nh) // 2
    out[py : py + nh, px : px + nw] = resized
    return out, Letterbox(scale, px, py, ow, oh, w, h)


def to_input_tensor(img_bgr: np.ndarray) -> np.ndarray:
    """HWC BGR uint8 -> NCHW RGB float32 in [0, 1]."""
    x = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    """Plain greedy NMS. boxes: [N,4] xyxy, scores: [N]. Returns kept indices."""
    if len(boxes_xyxy) == 0:
        return []
    x1, y1, x2, y2 = boxes_xyxy.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thres]
    return keep


# Per pass, best first (the phone app's YoloDecoder has the same caps). A normal frame has far fewer: on 1,600
# frames of road, work-zone and city clips the most of one class was 22 delineators, the most in one pass 38
# boxes. They bound a frame gone wrong (a bad model or a burst of clutter), whose hundreds of boxes would
# otherwise stall NMS and everything after the detectors.
MAX_CANDIDATES = 1000   # boxes over the threshold that go into NMS
MAX_DET = 40            # boxes out of one pass
MAX_PER_CLASS = 12      # of one class in one pass: a 13th cone adds nothing a report needs


def cap_detections(dets: Sequence[Detection], max_det: int = MAX_DET, max_per_class: int = MAX_PER_CLASS) -> list[Detection]:
    """The best ``max_per_class`` of each label and ``max_det`` in all (0 = no cap), best first."""
    out: list[Detection] = []
    seen: dict[str, int] = {}
    for d in sorted(dets, key=lambda d: -d.confidence):
        if max_det and len(out) >= max_det:
            break
        n = seen.get(d.label, 0)
        if max_per_class and n >= max_per_class:
            continue
        seen[d.label] = n + 1
        out.append(d)
    return out


def decode_yolo_output(
    raw: np.ndarray,
    lb: Letterbox,
    labels: Sequence[str],
    conf_thres: float,
    iou_thres: float,
    model_name: str = "hazard",
    class_agnostic: bool = False,
    max_det: int = MAX_DET,
    max_per_class: int = MAX_PER_CLASS,
) -> list[Detection]:
    """Turn a ``[1, 4+nc, N]`` (or ``[1, N, 4+nc]``) tensor into Detections.

    Returned bboxes are normalized center-xywh in the *source* image frame, best first, capped by
    ``max_det`` / ``max_per_class`` (0 = no cap).
    """
    out = raw[0]
    if out.shape[0] == 4 + len(labels):  # channel-first export [4+nc, N] -> [N, 4+nc]
        out = out.T  # -> [N, 4+nc]
    boxes = out[:, :4]
    cls_scores = out[:, 4:]
    cls_id = cls_scores.argmax(1)
    conf = cls_scores[np.arange(len(cls_id)), cls_id]
    m = conf >= conf_thres
    boxes, cls_id, conf = boxes[m], cls_id[m], conf[m]
    if len(boxes) == 0:
        return []
    if len(conf) > MAX_CANDIDATES:
        top = np.argsort(-conf, kind="stable")[:MAX_CANDIDATES]
        boxes, cls_id, conf = boxes[top], cls_id[top], conf[top]
    # xywh (letterboxed px) -> xyxy
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    # per-class NMS via coordinate offset trick
    offs = np.zeros_like(cls_id, dtype=np.float32) if class_agnostic else cls_id.astype(np.float32) * 4096.0
    keep = nms(xyxy + offs[:, None], conf, iou_thres)
    dets: list[Detection] = []
    for i in keep:
        bx1 = (xyxy[i, 0] - lb.pad_x) / lb.scale
        by1 = (xyxy[i, 1] - lb.pad_y) / lb.scale
        bx2 = (xyxy[i, 2] - lb.pad_x) / lb.scale
        by2 = (xyxy[i, 3] - lb.pad_y) / lb.scale
        bx1, bx2 = np.clip([bx1, bx2], 0, lb.src_w)
        by1, by2 = np.clip([by1, by2], 0, lb.src_h)
        w, h = bx2 - bx1, by2 - by1
        if w <= 1 or h <= 1:
            continue
        dets.append(
            Detection(
                label=labels[int(cls_id[i])],
                confidence=float(conf[i]),
                bbox=(
                    float((bx1 + w / 2) / lb.src_w),
                    float((by1 + h / 2) / lb.src_h),
                    float(w / lb.src_w),
                    float(h / lb.src_h),
                ),
                model=model_name,
            )
        )
    return cap_detections(dets, max_det, max_per_class) if (max_det or max_per_class) else dets


def _meta_size(session) -> int | None:
    """Training size from Ultralytics export metadata ("imgsz": "[640, 640]"), for an export with open height/width."""
    import re as _re
    nums = _re.findall(r"\d+", str(session.get_modelmeta().custom_metadata_map.get("imgsz", "")))
    return int(nums[0]) if nums else None


class OnnxDetector:
    """Loads a YOLO ONNX file and runs it on BGR frames."""

    def __init__(
        self,
        model_path: str | Path,
        labels: Sequence[str],
        name: str = "hazard",
        conf_thres: float = 0.35,
        iou_thres: float = 0.5,
        providers: Sequence[str] | None = None,
        threads: int | None = None,
        rect: bool = False,
        max_det: int = MAX_DET,
        max_per_class: int = MAX_PER_CLASS,
    ):
        if ort is None:
            raise RuntimeError("onnxruntime is not installed")
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=so, providers=list(providers or ["CPUExecutionProvider"])
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        self.size = int(shape[2]) if isinstance(shape[2], int) else _meta_size(self.session) or 640
        # rect: an export with open height/width (export_onnx.py --dynamic) runs each image at its own aspect,
        # as the phone app does; a static export always gets the square
        self.rect = bool(rect) and not isinstance(shape[2], int) and not isinstance(shape[3], int)
        self.labels = list(labels)
        self.name = name
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.max_per_class = max_per_class
        self.last_ms = 0.0
        nc = self.session.get_outputs()[0].shape[1]
        if isinstance(nc, int) and nc - 4 != len(self.labels):
            raise ValueError(f"{model_path}: model has {nc - 4} classes but {len(self.labels)} labels given")

    def __call__(self, frame_bgr: np.ndarray) -> list[Detection]:
        shape = rect_shape(*frame_bgr.shape[:2], self.size) if self.rect else None
        padded, lb = letterbox(frame_bgr, self.size, shape=shape)
        x = to_input_tensor(padded)
        t0 = time.perf_counter()
        raw = self.session.run(None, {self.input_name: x})[0]
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return decode_yolo_output(raw, lb, self.labels, self.conf_thres, self.iou_thres, self.name,
                                  max_det=self.max_det, max_per_class=self.max_per_class)


class ZoomDetector:
    """Two-pass wrapper for small, distant hazards: runs the detector on the whole frame and a
    second time on a zoomed crop of the road ahead, then merges both with per-class NMS.

    With a 1280x720 frame and a 640 px model the full pass sees the scene at half resolution;
    the default crop (middle 50 % of the width, 35-75 % of the height) is 640x288 px, so the
    second pass sees that region at full resolution: a 20 px cone becomes 40 px to the network.
    Costs one extra inference per frame. The crop stays above the hood zone on purpose.
    ``every`` = n runs the crop pass on every n-th frame only (the phone's zoom_every); the frames
    between get the whole-frame pass alone.
    """

    def __init__(self, base: "OnnxDetector", crop: tuple[float, float, float, float] = (0.25, 0.35, 0.75, 0.75),
                 iou_thres: float = 0.5, every: int = 1):
        self.base = base
        self.every = max(1, int(every))
        self._calls = 0
        self.crop = crop
        self.iou_thres = iou_thres
        self.labels = base.labels
        self.name = base.name
        self.size = base.size
        self.conf_thres = base.conf_thres
        self.last_ms = 0.0

    def __call__(self, frame_bgr: np.ndarray) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        full = self.base(frame_bgr)
        ms = self.base.last_ms
        self._calls += 1
        if (self._calls - 1) % self.every:
            self.last_ms = ms
            return full
        cx1, cy1, cx2, cy2 = self.crop
        x1, y1 = int(round(cx1 * w)), int(round(cy1 * h))
        x2, y2 = int(round(cx2 * w)), int(round(cy2 * h))
        if x2 - x1 < 32 or y2 - y1 < 32:
            self.last_ms = ms
            return full
        sub = frame_bgr[y1:y2, x1:x2]
        zoomed = self.base(sub)
        ms += self.base.last_ms
        self.last_ms = ms
        sw, sh = (x2 - x1) / w, (y2 - y1) / h
        mapped = [
            Detection(
                label=d.label,
                confidence=d.confidence,
                bbox=(cx1 + d.bbox[0] * sw, cy1 + d.bbox[1] * sh, d.bbox[2] * sw, d.bbox[3] * sh),
                model=d.model,
            )
            for d in zoomed
        ]
        merged = merge_detections(full + mapped, self.iou_thres)
        return cap_detections(merged, getattr(self.base, "max_det", MAX_DET),
                              getattr(self.base, "max_per_class", MAX_PER_CLASS))


def merge_detections(dets: list[Detection], iou_thres: float = 0.5) -> list[Detection]:
    """Per-class NMS over detections from several passes (normalized xywh boxes)."""
    if len(dets) <= 1:
        return list(dets)
    labels = sorted({d.label for d in dets})
    xyxy = np.array([[d.bbox[0] - d.bbox[2] / 2, d.bbox[1] - d.bbox[3] / 2,
                      d.bbox[0] + d.bbox[2] / 2, d.bbox[1] + d.bbox[3] / 2] for d in dets], dtype=np.float32)
    conf = np.array([d.confidence for d in dets], dtype=np.float32)
    offs = np.array([labels.index(d.label) * 4.0 for d in dets], dtype=np.float32)
    keep = nms(xyxy + offs[:, None], conf, iou_thres)
    return [dets[i] for i in sorted(keep, key=lambda i: -conf[i])]


class EnsembleDetector:
    """Runs several detectors on the same frame (in parallel threads; ONNX Runtime releases the
    GIL) and fuses their boxes, by default with Weighted Boxes Fusion (see ``fusion.py``).

    Typical members: the 640 px hazard model plus the 416 px one, or a model plus its
    ``ZoomDetector`` wrapper. Members may have different label sets; fusion is per label, so a
    label only one member knows simply gets that member's votes against the full weight sum.
    ``class_conf`` applies a per-label confidence floor after fusion (``{"pothole": 0.5}``).
    """

    def __init__(self, members: Sequence, weights: Sequence[float] | None = None, fusion: str = "wbf",
                 iou_thres: float = 0.55, conf_type: str = "avg", parallel: bool = True,
                 class_conf: dict[str, float] | None = None, name: str = "ensemble"):
        if not members:
            raise ValueError("EnsembleDetector needs at least one member")
        self.members = list(members)
        self.weights = list(weights) if weights is not None else [1.0] * len(self.members)
        if len(self.weights) != len(self.members):
            raise ValueError("one weight per member expected")
        if fusion not in ("wbf", "nms"):
            raise ValueError("fusion must be 'wbf' or 'nms'")
        self.fusion = fusion
        self.iou_thres = iou_thres
        self.conf_type = conf_type
        self.parallel = parallel and len(self.members) > 1
        self.class_conf = dict(class_conf or {})
        self.name = name
        seen: list[str] = []
        for m in self.members:
            for lb in m.labels:
                if lb not in seen:
                    seen.append(lb)
        self.labels = seen
        self.size = max(m.size for m in self.members)
        self.conf_thres = min(m.conf_thres for m in self.members)
        self.last_ms = 0.0          # summed member inference time (compute cost)
        self.last_wall_ms = 0.0     # wall time of the parallel run (latency)
        self._pool = None
        if self.parallel:
            from concurrent.futures import ThreadPoolExecutor

            self._pool = ThreadPoolExecutor(max_workers=len(self.members), thread_name_prefix="ensemble")

    def __call__(self, frame_bgr: np.ndarray) -> list[Detection]:
        from .fusion import apply_class_thresholds, fuse_nms, fuse_wbf

        t0 = time.perf_counter()
        if self._pool is not None:
            results = list(self._pool.map(lambda m: m(frame_bgr), self.members))
        else:
            results = [m(frame_bgr) for m in self.members]
        self.last_ms = float(sum(m.last_ms for m in self.members))
        self.last_wall_ms = (time.perf_counter() - t0) * 1000.0
        if self.fusion == "nms":
            fused = fuse_nms(results, self.iou_thres)
        else:
            fused = fuse_wbf(results, self.weights, self.iou_thres, conf_type=self.conf_type, model_name=self.name)
        return apply_class_thresholds(fused, self.class_conf)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None


class ClassFilter:
    """Wraps any detector and applies per-label confidence floors to its output."""

    def __init__(self, base, class_conf: dict[str, float], default: float = 0.0):
        self.base = base
        self.class_conf = dict(class_conf)
        self.default = default
        self.labels = base.labels
        self.name = base.name
        self.size = base.size
        self.conf_thres = base.conf_thres

    @property
    def last_ms(self) -> float:
        return self.base.last_ms

    def __call__(self, frame_bgr: np.ndarray) -> list[Detection]:
        from .fusion import apply_class_thresholds

        return apply_class_thresholds(self.base(frame_bgr), self.class_conf, self.default)
