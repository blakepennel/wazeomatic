"""Glue: frames -> detectors -> ReportManager -> sinks, with optional display/recording."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

from .detector import OnnxDetector
from .report_manager import ReportManager
from .schema import Detection, HazardEvent
from .sources import Frame

COLORS = {
    "pothole": (0, 0, 255),
    "cone": (0, 140, 255),
    "construction_sign": (0, 200, 255),
    "delineator": (0, 255, 255),
    "car": (200, 200, 200),
    "truck": (200, 200, 200),
    "bus": (200, 200, 200),
    "person": (255, 0, 255),
}


def draw(frame: np.ndarray, dets: Iterable[Detection], events: Iterable[HazardEvent], info: str = "") -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    for d in dets:
        cx, cy, bw, bh = d.bbox
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        col = COLORS.get(d.label, (0, 255, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(out, f"{d.label} {d.confidence:.2f}", (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    y = 22
    if info:
        cv2.putText(out, info, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 24
    for ev in events:
        cv2.putText(out, f"REPORT {ev.hazard_type} ({ev.confidence:.2f})", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y += 26
    return out


@dataclass
class PipelineStats:
    frames: int = 0
    detections: int = 0
    events: int = 0
    infer_ms: list = field(default_factory=list)
    started: float = field(default_factory=time.time)

    def summary(self) -> dict:
        ms = sorted(self.infer_ms)

        def pct(p: float) -> float:
            return ms[min(len(ms) - 1, int(p * len(ms)))] if ms else 0.0

        wall = time.time() - self.started
        return {
            "frames": self.frames,
            "detections": self.detections,
            "events": self.events,
            "infer_ms_mean": round(sum(ms) / len(ms), 2) if ms else 0.0,
            "infer_ms_p50": round(pct(0.5), 2),
            "infer_ms_p90": round(pct(0.9), 2),
            "wall_s": round(wall, 1),
            "pipeline_fps": round(self.frames / wall, 2) if wall else 0.0,
        }


class Pipeline:
    def __init__(self, detectors: Sequence[OnnxDetector], manager: ReportManager, sinks: Sequence,
                 show: bool = False, record_path: Optional[str | Path] = None, record_fps: float = 10.0,
                 on_frame=None):
        self.detectors = list(detectors)
        self.manager = manager
        self.sinks = list(sinks)
        self.show = show
        self.record_path = Path(record_path) if record_path else None
        self.record_fps = record_fps
        self._writer = None
        self.on_frame = on_frame
        self.stats = PipelineStats()
        self.recent_events: list[tuple[float, HazardEvent]] = []

    def process(self, frame: Frame) -> tuple[list[Detection], list[HazardEvent]]:
        dets: list[Detection] = []
        ms = 0.0
        for det in self.detectors:
            dets.extend(det(frame.image))
            ms += det.last_ms
        events = self.manager.update(dets, frame_index=frame.index)
        for ev in events:
            for s in self.sinks:
                s.send(ev)
        self.stats.frames += 1
        self.stats.detections += len(dets)
        self.stats.events += len(events)
        self.stats.infer_ms.append(ms)
        now = time.time()
        self.recent_events += [(now, e) for e in events]
        self.recent_events = [(t, e) for t, e in self.recent_events if now - t < 3.0]
        if self.show or self.record_path or self.on_frame:
            info = f"frame {frame.index}  t={frame.ts:6.1f}s  infer {ms:5.1f} ms  dets {len(dets)}"
            vis = draw(frame.image, dets, [e for _, e in self.recent_events], info)
            if self.record_path:
                if self._writer is None:
                    self.record_path.parent.mkdir(parents=True, exist_ok=True)
                    h, w = vis.shape[:2]
                    self._writer = cv2.VideoWriter(str(self.record_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                                   self.record_fps, (w, h))
                self._writer.write(vis)
            if self.show:
                cv2.imshow("wazeomatic", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt
            if self.on_frame:
                self.on_frame(vis, dets, events)
        return dets, events

    def run(self, source: Iterable[Frame]) -> PipelineStats:
        try:
            for frame in source:
                self.process(frame)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()
        return self.stats

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        for s in self.sinks:
            s.close()
        if self.show:
            cv2.destroyAllWindows()
