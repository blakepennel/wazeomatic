"""Turns noisy per-frame detections into rare, trustworthy HazardEvents.

Three layers, in order:

1. **Persistence** - a hazard class must be seen in at least ``min_hits`` of the last
   ``window`` inference frames (per class). Single-frame flickers never report.
2. **Cooldown / de-duplication** - after a hazard type is emitted it is muted for
   ``cooldown_s`` seconds. This mirrors the macro's own one-report-per-minute guard and
   stops the same pothole being reported ten times while we drive over it.
3. **Heuristics** - things no single-frame detector can label, currently the
   stopped-vehicle-on-shoulder check (experimental, off by default).

Everything here is pure Python so it can be unit tested without a model, and the
Kotlin ``ReportManager`` in the Android app implements the same rules.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from .schema import DEFAULT_CLASS_TO_HAZARD, Detection, HazardEvent


@dataclass
class ReportConfig:
    window: int = 10                 # inference frames considered
    min_hits: int = 3                # hits required inside the window
    min_confidence: float = 0.45     # per-detection confidence to count as a hit
    cooldown_s: float = 60.0         # per hazard type
    global_cooldown_s: float = 5.0   # between any two events
    class_to_hazard: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CLASS_TO_HAZARD))
    # stopped-vehicle heuristic
    stopped_vehicle_enabled: bool = False
    shoulder_x_min: float = 0.62     # right-hand shoulder region of interest (normalized)
    shoulder_y_min: float = 0.40
    vehicle_labels: tuple[str, ...] = ("car", "truck", "bus")
    vehicle_min_area: float = 0.02   # bbox area fraction - close enough to matter
    vehicle_min_growth: float = 1.35 # area must grow by this factor across the window


@dataclass
class _Track:
    hits: deque = field(default_factory=deque)        # (frame_idx, confidence, bbox)
    last_emitted_ts: float = -1e9


class ReportManager:
    def __init__(self, config: ReportConfig | None = None, source: str = "sim",
                 clock: Callable[[], float] | None = None):
        import time

        self.cfg = config or ReportConfig()
        self.source = source
        self.clock = clock or time.time
        self._tracks: dict[str, _Track] = defaultdict(_Track)
        self._last_global_ts = -1e9
        self._frame = 0

    # ------------------------------------------------------------------ public
    def update(self, detections: Iterable[Detection], frame_index: Optional[int] = None,
               lat: float | None = None, lon: float | None = None) -> list[HazardEvent]:
        """Feed one inference frame; returns zero or more events to report."""
        self._frame = self._frame + 1 if frame_index is None else frame_index
        now = self.clock()
        best: dict[str, Detection] = {}
        for d in detections:
            hazard = self._hazard_for(d)
            if hazard is None or d.confidence < self.cfg.min_confidence:
                continue
            if hazard not in best or d.confidence > best[hazard].confidence:
                best[hazard] = d
        # record hits (one per hazard per frame) and trim the window
        for hazard, d in best.items():
            self._tracks[hazard].hits.append((self._frame, d.confidence, d.bbox, d.label))
        for hazard, tr in self._tracks.items():
            while tr.hits and tr.hits[0][0] <= self._frame - self.cfg.window:
                tr.hits.popleft()

        events: list[HazardEvent] = []
        for hazard, tr in self._tracks.items():
            if len(tr.hits) < self.cfg.min_hits:
                continue
            if now - tr.last_emitted_ts < self.cfg.cooldown_s:
                continue
            if now - self._last_global_ts < self.cfg.global_cooldown_s:
                continue
            if hazard == "HAZARD_SHOULDER_CAR_STOPPED" and not self._stopped_vehicle_confirmed(tr):
                continue
            _, conf, bbox, label = max(tr.hits, key=lambda h: h[1])
            ev = HazardEvent(hazard_type=hazard, confidence=conf, source=self.source, ts=now,
                             frame_index=self._frame, bbox=bbox, label=label, lat=lat, lon=lon)
            events.append(ev)
            tr.last_emitted_ts = now
            self._last_global_ts = now
            tr.hits.clear()
        return events

    # ----------------------------------------------------------------- helpers
    def _hazard_for(self, d: Detection) -> Optional[str]:
        if d.label in self.cfg.class_to_hazard:
            return self.cfg.class_to_hazard[d.label]
        if self.cfg.stopped_vehicle_enabled and d.label in self.cfg.vehicle_labels and self._in_shoulder(d):
            return "HAZARD_SHOULDER_CAR_STOPPED"
        return None

    def _in_shoulder(self, d: Detection) -> bool:
        cx, cy, w, h = d.bbox
        return cx >= self.cfg.shoulder_x_min and cy >= self.cfg.shoulder_y_min and w * h >= self.cfg.vehicle_min_area

    def _stopped_vehicle_confirmed(self, tr: _Track) -> bool:
        """A parked car on the shoulder grows in the frame as we approach it and stays
        on the right edge; a car in the next lane moving with us does neither."""
        if len(tr.hits) < 2:
            return False
        first, last = tr.hits[0][2], tr.hits[-1][2]
        a0, a1 = first[2] * first[3], last[2] * last[3]
        return a1 >= a0 * self.cfg.vehicle_min_growth and last[0] >= first[0] - 0.02
