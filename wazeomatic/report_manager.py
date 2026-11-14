"""Turns noisy per-frame detections into rare, trustworthy HazardEvents.

Three layers, in order:

1. **Persistence** - a hazard class must be seen in at least ``min_hits`` of the last
   ``window`` inference frames (per class). Single-frame flickers never report.
2. **Cooldown / de-duplication** - after a hazard type is emitted it is muted for
   ``cooldown_s`` seconds. This mirrors the macro's own one-report-per-minute guard and
   stops the same pothole being reported ten times while we drive over it.
3. **Heuristics** - a hood exclusion zone (boxes touching the bottom edge of the frame are
   the car's own bonnet reflection, not road), and the stopped-vehicle-on-shoulder check
   (experimental, off by default).
4. **Spatial consistency** (``track_iou`` > 0) - the hits that make up a report must come from
   boxes that overlap frame to frame, i.e. from one object the car is approaching, not from a
   flicker here and a flicker there. Per-label confidence floors (``min_confidence_by_label``)
   let a noisy class such as cone demand more than a clean one such as construction sign.

Everything here is pure Python so it can be unit tested without a model, and the
Kotlin ``ReportManager`` in the Android app implements the same rules.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from .schema import DEFAULT_CLASS_TO_HAZARD, Detection, HazardEvent


DEFAULT_LABEL_FLOORS: dict[str, float] = {"pothole": 0.45, "police_vehicle": 0.85}
# per-type overrides of ReportConfig.min_hits. Police: two classifier hits above the 0.85 floor in ten frames;
# with the classifiers at their trained 160 px input that found 7 of the 9 police cars on the held-out
# Argoverse clips with no false report on the other 154, where three hits found 5 (docs/waze-coverage.md)
DEFAULT_MIN_HITS_BY_HAZARD: dict[str, int] = {"POLICE": 2}


def _scene_cooldowns() -> dict[str, float]:
    from .scene import SCENE_COOLDOWNS   # scene imports cv2; keep it out of the module import path

    return SCENE_COOLDOWNS


def default_label_floor_spec() -> str:
    """The shipped per-label floors as a CLI string (``"pothole=0.45"``): the default value of
    ``--report-class-conf`` so the command-line tools run the same operating point as the phone."""
    return ",".join(f"{k}={v:g}" for k, v in DEFAULT_LABEL_FLOORS.items())


@dataclass
class ReportConfig:
    window: int = 10                 # inference frames considered
    min_hits: int = 3                # hits required inside the window
    min_hits_by_hazard: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_MIN_HITS_BY_HAZARD))
    min_confidence: float = 0.40     # per-detection confidence to count as a hit
    # per-label floors on top of min_confidence (the higher of the two applies). Potholes need
    # more: the v3 model is clean on 12 minutes of hazard-free footage at 0.45 (the older 640c
    # needed 0.55; docs/model-research.md)
    min_confidence_by_label: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LABEL_FLOORS))
    track_iou: float = 0.0           # > 0: hits must chain through boxes with at least this IoU
                                     # (frame to frame, boxes grown by track_grow), or whose centres
                                     # lie within track_grow box sizes of each other
    track_grow: float = 1.5          # box growth for the IoU test / centre-distance tolerance
    hood_frac: float = 0.10          # ignore boxes whose bottom edge is in the lowest 10 % of the frame
                                     # (the car's own hood / dashboard reflection looks like a pothole)
    cooldown_s: float = 60.0         # per hazard type
    # longer per-type cooldowns for conditions that last minutes (scene.SCENE_COOLDOWNS): one report per
    # traffic jam or fog bank, not one a minute for as long as it lasts
    cooldown_by_hazard: dict[str, float] = field(default_factory=lambda: dict(_scene_cooldowns()))
    global_cooldown_s: float = 60.0  # between any two events. The macro files at most one report a
                                     # minute in total (docs/pipeline-to-macro.md), so anything sent
                                     # sooner would be dropped there unseen; holding it back here
                                     # keeps the loss visible (ReportManager.held_back)
    class_to_hazard: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CLASS_TO_HAZARD))
    # stopped-vehicle heuristic
    stopped_vehicle_enabled: bool = False
    shoulder_x_min: float = 0.62     # right-hand shoulder region of interest (normalized)
    shoulder_y_min: float = 0.40
    vehicle_labels: tuple[str, ...] = ("car", "truck", "bus")
    # red-and-white chevrons on the back of a works van look like a barricade; a box of these labels that lies
    # almost wholly inside a vehicle box is the vehicle's livery, not a barrier on the road
    livery_labels: tuple[str, ...] = ("barricade",)
    livery_inside: float = 0.8
    # a dog or cat beside a person standing on the same ground is a pet being walked on the pavement, not an
    # animal loose on the road (the one Animal report on 38 minutes of held-out Argoverse city clips)
    pet_labels: tuple[str, ...] = ("dog", "cat")
    pet_reach: float = 0.75          # horizontal gap to the person's box, in person heights
    vehicle_min_area: float = 0.02   # bbox area fraction - close enough to matter
    vehicle_min_growth: float = 1.35 # area must grow by this factor across the window


@dataclass
class _Track:
    hits: deque = field(default_factory=deque)        # (frame_idx, confidence, bbox, label)
    last_emitted_ts: float = -1e9
    tracklets: list = field(default_factory=list)     # spatial mode: [_Tracklet]


@dataclass
class _Tracklet:
    bbox: tuple
    hits: deque = field(default_factory=deque)        # (frame_idx, confidence, bbox, label)


def _grown_iou(a: tuple, b: tuple, grow: float) -> float:
    ax1, ay1 = a[0] - a[2] * grow / 2, a[1] - a[3] * grow / 2
    ax2, ay2 = a[0] + a[2] * grow / 2, a[1] + a[3] * grow / 2
    bx1, by1 = b[0] - b[2] * grow / 2, b[1] - b[3] * grow / 2
    bx2, by2 = b[0] + b[2] * grow / 2, b[1] + b[3] * grow / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _associated(a: tuple, b: tuple, iou_thr: float, grow: float) -> bool:
    """Same object across two frames: grown boxes overlap enough, or the centres are within
    ``grow`` box sizes of each other (an object close to the car moves a lot per frame)."""
    if _grown_iou(a, b, grow) >= iou_thr:
        return True
    dist = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
    return dist <= grow * max(a[2], a[3], b[2], b[3])


def _inside(a, b) -> float:
    """Fraction of box ``a`` (normalised xywh) that lies inside box ``b``."""
    ix = max(0.0, min(a[0] + a[2] / 2, b[0] + b[2] / 2) - max(a[0] - a[2] / 2, b[0] - b[2] / 2))
    iy = max(0.0, min(a[1] + a[3] / 2, b[1] + b[3] / 2) - max(a[1] - a[3] / 2, b[1] - b[3] / 2))
    return ix * iy / max(a[2] * a[3], 1e-9)


def _beside(pet, person, reach: float) -> bool:
    """``pet`` (normalised xywh) stands next to ``person``: a horizontal gap of at most ``reach`` person heights,
    and the bottoms of the two boxes (where they touch the ground) within half a person height."""
    gap = max(0.0, abs(pet[0] - person[0]) - (pet[2] + person[2]) / 2)
    ground = abs((pet[1] + pet[3] / 2) - (person[1] + person[3] / 2))
    return gap <= reach * person[3] and ground <= 0.5 * person[3]


class ReportManager:
    def __init__(self, config: ReportConfig | None = None, source: str = "sim",
                 clock: Callable[[], float] | None = None, waze=None):
        import time

        self.cfg = config or ReportConfig()
        # read-only view of the Waze screen (waze_reader.WazeAlerts): a hazard Waze already shows nearby is
        # a confirmation, not a new report
        self.waze = waze
        self.already_reported = 0
        self.confirmations: list[dict] = []
        self.source = source
        self.clock = clock or time.time
        self._tracks: dict[str, _Track] = defaultdict(_Track)
        self._last_global_ts = -1e9
        self._frame = 0
        self._hood_frac_dynamic = 0.0
        # hazards that qualified while another type's report held the global cooldown, counted once
        # per hazard type per cooldown window: reports the driver would have expected and did not get
        self.held_back = 0
        self._held: set[str] = set()

    # ------------------------------------------------------------------ public
    def update(self, detections: Iterable[Detection], frame_index: Optional[int] = None,
               lat: float | None = None, lon: float | None = None) -> list[HazardEvent]:
        """Feed one inference frame; returns zero or more events to report."""
        self._frame = self._frame + 1 if frame_index is None else frame_index
        now = self.clock()
        best: dict[str, Detection] = {}
        accepted: dict[str, list[Detection]] = defaultdict(list)
        detections = list(detections)
        vehicles = [d.bbox for d in detections if d.label in self.cfg.vehicle_labels]
        people = [d.bbox for d in detections if d.label == "person"]
        for d in detections:
            hazard = self._hazard_for(d)
            if hazard is None or d.confidence < self._min_conf(d.label):
                continue
            if d.model != "scene" and self._in_hood_zone(d):
                continue
            if d.label in self.cfg.livery_labels and any(_inside(d.bbox, v) >= self.cfg.livery_inside
                                                         for v in vehicles):
                continue
            if d.label in self.cfg.pet_labels and any(_beside(d.bbox, p, self.cfg.pet_reach) for p in people):
                continue
            accepted[hazard].append(d)
            if hazard not in best or d.confidence > best[hazard].confidence:
                best[hazard] = d
        # record hits (one per hazard per frame) and trim the window
        for hazard, d in best.items():
            self._tracks[hazard].hits.append((self._frame, d.confidence, d.bbox, d.label))
        for hazard, tr in self._tracks.items():
            while tr.hits and tr.hits[0][0] <= self._frame - self.cfg.window:
                tr.hits.popleft()
        if self.cfg.track_iou > 0:
            self._update_tracklets(accepted)

        events: list[HazardEvent] = []
        for hazard, tr in self._tracks.items():
            hits = self._qualifying_hits(tr, self.cfg.min_hits_by_hazard.get(hazard, self.cfg.min_hits))
            if hits is None:
                continue
            if now - tr.last_emitted_ts < self.cfg.cooldown_by_hazard.get(hazard, self.cfg.cooldown_s):
                continue
            if hazard == "HAZARD_SHOULDER_CAR_STOPPED" and not self._stopped_vehicle_confirmed(tr):
                continue
            cover = self.waze.covering(hazard) if self.waze is not None else None
            if cover is not None:
                # Waze already shows it: count the agreement, mute the type for a cooldown, send nothing.
                # The global cooldown is left alone because no report goes out.
                _, conf, _, label = max(hits, key=lambda h: h[1])
                self.already_reported += 1
                self.confirmations.append({"hazard_type": hazard, "confidence": round(conf, 3), "label": label,
                                           "ts": now, "frame_index": self._frame, "waze": cover.to_dict()})
                tr.last_emitted_ts = now
                tr.hits.clear()
                tr.tracklets.clear()
                continue
            if now - self._last_global_ts < self.cfg.global_cooldown_s:
                if hazard not in self._held:
                    self._held.add(hazard)
                    self.held_back += 1
                continue
            _, conf, bbox, label = max(hits, key=lambda h: h[1])
            ev = HazardEvent(hazard_type=hazard, confidence=conf, source=self.source, ts=now,
                             frame_index=self._frame, bbox=bbox, label=label, lat=lat, lon=lon)
            events.append(ev)
            tr.last_emitted_ts = now
            self._last_global_ts = now
            self._held.clear()
            tr.hits.clear()
            tr.tracklets.clear()
        return events

    # ------------------------------------------------------------ spatial tracking
    def _min_conf(self, label: str) -> float:
        """A label floor only ever raises the bar: the higher of min_confidence and the label value."""
        return max(self.cfg.min_confidence, self.cfg.min_confidence_by_label.get(label, 0.0))

    def _update_tracklets(self, accepted: dict[str, list[Detection]]) -> None:
        """Greedy IoU association of this frame's accepted boxes to per-hazard tracklets."""
        for hazard, dets in accepted.items():
            tr = self._tracks[hazard]
            used: set[int] = set()
            for d in sorted(dets, key=lambda x: -x.confidence):
                best_i, best_iou = -1, -1.0
                for i, tl in enumerate(tr.tracklets):
                    if i in used or not _associated(tl.bbox, d.bbox, self.cfg.track_iou, self.cfg.track_grow):
                        continue
                    iou = _grown_iou(tl.bbox, d.bbox, self.cfg.track_grow)
                    if iou > best_iou:
                        best_i, best_iou = i, iou
                hit = (self._frame, d.confidence, d.bbox, d.label)
                if best_i < 0:
                    tr.tracklets.append(_Tracklet(bbox=d.bbox, hits=deque([hit])))
                    used.add(len(tr.tracklets) - 1)
                else:
                    tl = tr.tracklets[best_i]
                    tl.bbox = d.bbox
                    tl.hits.append(hit)
                    used.add(best_i)
        # trim windows and drop stale tracklets (for every hazard, not only those seen now)
        for tr in self._tracks.values():
            for tl in tr.tracklets:
                while tl.hits and tl.hits[0][0] <= self._frame - self.cfg.window:
                    tl.hits.popleft()
            tr.tracklets = [tl for tl in tr.tracklets if tl.hits]

    def _qualifying_hits(self, tr: _Track, need: int):
        """The hits that justify a report now (at least ``need`` in the window), or None."""
        if self.cfg.track_iou <= 0:
            return tr.hits if len(tr.hits) >= need else None
        ready = [tl for tl in tr.tracklets if len(tl.hits) >= need]
        if not ready:
            return None
        return max(ready, key=lambda tl: max(h[1] for h in tl.hits)).hits

    # ----------------------------------------------------------------- helpers
    def _hazard_for(self, d: Detection) -> Optional[str]:
        if d.label in self.cfg.class_to_hazard:
            return self.cfg.class_to_hazard[d.label]
        if self.cfg.stopped_vehicle_enabled and d.label in self.cfg.vehicle_labels and self._in_shoulder(d):
            return "HAZARD_SHOULDER_CAR_STOPPED"
        return None

    @property
    def hood_frac(self) -> float:
        """Active hood zone: the configured minimum, or a larger value learnt from the video
        (see ``hood.HoodEstimator``; the pipeline calls ``set_hood_frac``)."""
        return max(self.cfg.hood_frac, self._hood_frac_dynamic)

    def set_hood_frac(self, frac: float) -> None:
        self._hood_frac_dynamic = float(frac)

    def _in_hood_zone(self, d: Detection) -> bool:
        cy, h = d.bbox[1], d.bbox[3]
        hood = self.hood_frac
        if hood <= 0:
            return False
        if d.model == "vehicle":
            # a police car or ambulance close by reaches the bottom of the frame; only a vehicle box that lies
            # entirely in the hood zone is the car itself
            return (cy - h / 2) > (1.0 - hood)
        return (cy + h / 2) > (1.0 - hood)

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
