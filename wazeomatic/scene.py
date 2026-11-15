"""Scene-level reports: conditions that are not one box in one frame.

The detectors see objects; several Waze reports are about a situation instead. ``SceneAnalyzer`` turns the
per-frame detections, a cheap ego-motion estimate and the weather classifier into ``scene_*`` detections,
one per frame while a condition holds, which the report manager then treats like any other class (with longer
cooldowns, ``SCENE_COOLDOWNS``):

=========================  ===============================================================================
label                      condition
=========================  ===============================================================================
scene_traffic_standstill   not moving for ``standstill_s`` with a vehicle close ahead and no red light seen
scene_traffic_heavy        over ``heavy_window_s``: mostly slow or stopped, repeatedly, among many vehicles,
                           and no red light seen
scene_crash                an ambulance or fire engine with lights, or two kinds of responder with lights,
                           for ``crash_hold_s``: that we approach, or that stand still while we stand
scene_lane_left/center/    at least ``lane_min_objects`` cones, drums or barricades on the road 3-40 m ahead,
right                      two of them inside our lane (flat-road camera model), on one side for 1.5 s
scene_dark_signal          two or more signal heads close ahead, none lit, for ``dark_signal_s``; a single lit
                           head anywhere in view vetoes it (a signal seen from behind is dark too)
scene_fog, scene_slippery  the weather classifier's smoothed probability above a threshold for a hold time
...
=========================  ===============================================================================

Everything is driven by the frame timestamp, so a replayed clip and a live camera behave the same. The Kotlin
twin is ``SceneAnalyzer.kt``.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

from .schema import Detection

CONSTRUCTION_LABELS = ("cone", "barricade", "delineator", "construction_sign", "barrel")
VEHICLE_LABELS = ("car", "truck", "bus")
EMERGENCY_LABELS = ("police_vehicle", "ambulance", "fire_truck")

# per-hazard cooldowns for situations that last minutes: one report per jam, per fog bank, per crash
SCENE_COOLDOWNS: dict[str, float] = {
    "TRAFFIC_HEAVY": 900.0, "TRAFFIC_STANDSTILL": 900.0,
    "WEATHER_FOG": 1800.0, "WEATHER_FLOOD": 900.0, "WEATHER_SLIPPERY": 1800.0, "WEATHER_ICE": 1800.0,
    "WEATHER_UNPLOWED": 1800.0, "CRASH": 300.0, "CRASH_OTHER_SIDE": 300.0,
    "BLOCKED_LANE_LEFT": 300.0, "BLOCKED_LANE_CENTER": 300.0, "BLOCKED_LANE_RIGHT": 300.0,
    "HAZARD_ON_ROAD_TRAFFIC_LIGHT": 600.0,
}

# classifier class -> (scene label, probability threshold, hold seconds). Weather classes are bare (fog, rain,
# snow), road-surface classes carry a ``surface_`` prefix (classifiers.RoadSurface). A key "a&b|c" is the
# smaller of a and (b + c): rain falling AND the road wet or slushy. A term whose classes no loaded model has
# is left out, so without the surface model the surface rules never fire and Slippery road rests on rain alone.
DEFAULT_WEATHER_RULES: dict[str, tuple[str, float, float]] = {
    "fog": ("scene_fog", 0.70, 20.0),
    "surface_water": ("scene_flood", 0.85, 8.0),
    "surface_ice": ("scene_ice", 0.80, 10.0),
    "surface_snow": ("scene_unplowed", 0.75, 15.0),
    "rain&surface_wet|surface_water|surface_slush": ("scene_slippery", 0.65, 60.0),
}


def _whole(label: str, conf: float, bbox=(0.5, 0.5, 1.0, 1.0)) -> Detection:
    return Detection(label=label, confidence=float(min(1.0, max(0.0, conf))), bbox=tuple(bbox), model="scene")


def _standing(boxes: Sequence[tuple], max_shift: float) -> bool:
    """Whether a run of boxes (oldest first) stays put: the centre of the last third is within ``max_shift``
    box widths of the first third's and the size changed by less than 40 %."""
    if len(boxes) < 3:
        return False
    third = max(1, len(boxes) // 3)
    first, last = boxes[:third], boxes[-third:]
    width = float(np.mean([b[2] for b in boxes]))
    shift = abs(float(np.mean([b[0] for b in last])) - float(np.mean([b[0] for b in first])))
    a0 = float(np.mean([b[2] * b[3] for b in first]))
    a1 = float(np.mean([b[2] * b[3] for b in last]))
    return width > 0 and shift <= max_shift * width and 0.7 <= a1 / max(a0, 1e-9) <= 1.4


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


# ----------------------------------------------------------------------------------------------- colours
def lamp_fractions(img: np.ndarray, bbox, top_frac: float = 1.0) -> dict[str, float]:
    """Share of the box (its upper ``top_frac``) that is a bright, saturated red / amber / green / blue light."""
    h, w = img.shape[:2]
    cx, cy, bw, bh = bbox
    x1, x2 = int(max(0, (cx - bw / 2) * w)), int(min(w, (cx + bw / 2) * w))
    y1 = int(max(0, (cy - bh / 2) * h))
    y2 = int(min(h, y1 + bh * top_frac * h))
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return {"red": 0.0, "amber": 0.0, "green": 0.0, "blue": 0.0, "lit": 0.0}
    if roi.shape[0] * roi.shape[1] > 96 * 96:
        s = 96.0 / max(roi.shape[:2])
        roi = cv2.resize(roi, (max(1, int(roi.shape[1] * s)), max(1, int(roi.shape[0] * s))),
                         interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[..., 0].astype(np.int16), hsv[..., 1], hsv[..., 2]
    bright = (vv >= 200) & (ss >= 120)
    n = float(hh.size)
    red = bright & ((hh <= 8) | (hh >= 170))
    amber = bright & (hh > 8) & (hh <= 30)
    green = bright & (hh >= 45) & (hh <= 95)
    blue = bright & (hh >= 100) & (hh <= 135)
    lit = (vv >= 235)
    return {"red": red.sum() / n, "amber": amber.sum() / n, "green": green.sum() / n,
            "blue": blue.sum() / n, "lit": lit.sum() / n}


@dataclass
class _LightTrack:
    bbox: tuple
    samples: deque = field(default_factory=lambda: deque(maxlen=40))   # (t, red, blue)
    last_t: float = 0.0


def _alternations(xs: Sequence[float], min_amp: float) -> int:
    """How many times a series crosses its midpoint by at least ``min_amp`` each way (on/off cycles)."""
    if len(xs) < 4:
        return 0
    lo, hi = min(xs), max(xs)
    if hi - lo < min_amp:
        return 0
    mid = (hi + lo) / 2
    band = (hi - lo) * 0.25
    state, flips = None, 0
    for x in xs:
        s = True if x > mid + band else False if x < mid - band else state
        if state is not None and s is not None and s != state:
            flips += 1
        state = s if s is not None else state
    return flips


class EmergencyLights:
    """Red / blue flashing on a vehicle. Brake lights are red and steady, indicators and hazard flashers amber;
    a light bar alternates red and blue (or flashes one of them) several times a second. A box qualifies
    when, within ``window_s``, blue or red flips on and off at least ``min_flips`` times."""

    def __init__(self, window_s: float = 2.0, min_flips: int = 3, min_amp: float = 0.004, min_box_px: int = 24):
        self.window_s = window_s
        self.min_flips = min_flips
        self.min_amp = min_amp
        self.min_box_px = min_box_px
        self._tracks: list[_LightTrack] = []

    def update(self, t: float, img: np.ndarray, dets: Sequence[Detection]) -> list[Detection]:
        h, w = img.shape[:2]
        boxes = [d for d in dets if d.label in VEHICLE_LABELS + EMERGENCY_LABELS
                 and min(d.bbox[2] * w, d.bbox[3] * h) >= self.min_box_px]
        out: list[Detection] = []
        used: set[int] = set()
        for d in sorted(boxes, key=lambda x: -(x.bbox[2] * x.bbox[3])):
            best, best_iou = -1, 0.3
            for i, tr in enumerate(self._tracks):
                if i in used:
                    continue
                v = _iou(tr.bbox, d.bbox)
                if v > best_iou:
                    best, best_iou = i, v
            if best < 0:
                self._tracks.append(_LightTrack(bbox=d.bbox))
                best = len(self._tracks) - 1
            used.add(best)
            tr = self._tracks[best]
            tr.bbox, tr.last_t = d.bbox, t
            f = lamp_fractions(img, d.bbox, top_frac=0.6)
            tr.samples.append((t, f["red"], f["blue"]))
            recent = [s for s in tr.samples if t - s[0] <= self.window_s]
            reds, blues = [s[1] for s in recent], [s[2] for s in recent]
            flips = max(_alternations(blues, self.min_amp), _alternations(reds, self.min_amp * 1.5))
            both = max(blues, default=0) > self.min_amp and max(reds, default=0) > self.min_amp
            if flips >= self.min_flips or (both and flips >= self.min_flips - 1):
                conf = min(1.0, 0.5 + 0.1 * flips + (0.15 if both else 0.0))
                out.append(Detection(label="emergency_lights", confidence=conf, bbox=d.bbox, model="lights"))
        self._tracks = [tr for tr in self._tracks if t - tr.last_t <= self.window_s]
        return out


# ------------------------------------------------------------------------------------------- ego motion
class EgoMotion:
    """Is the car moving? From how much the image's horizontal gradients change between frames.

    The frame is box-averaged to 80 x 45 grey, the lowest 15 % (hood) and every vehicle / person box are masked,
    and the score is the median absolute change of the central-difference gradient, scaled to a 10 fps frame
    gap. A parked car's view barely changes (95 % of stopped frames score under 0.45-0.85 on two real clips,
    slow driving 0.65 and up); driving moves every edge (2 and up). Gradients rather than raw brightness keep
    exposure drift out of it. Pure array arithmetic, so the phone (``SceneAnalyzer.kt``) computes the same number
    when the grey conversion matches: the Kotlin uses OpenCV 5's BGR2GRAY formula (OpenCV 4 differs by one
    level on some colours, which moves the score by far less than the thresholds). A GPS speed, when there is
    one, replaces it.
    """

    W, H = 80, 45

    def __init__(self, stationary_thr: float = 0.6, slow_thr: float = 2.5,
                 stationary_mps: float = 0.8, slow_mps: float = 6.0, hood_frac: float = 0.15):
        self.stationary_thr = stationary_thr
        self.slow_thr = slow_thr
        self.stationary_mps = stationary_mps
        self.slow_mps = slow_mps
        self.hood_frac = hood_frac
        self._prev: Optional[np.ndarray] = None
        self._prev_t = 0.0
        self.flow = 0.0
        self.gps_mps: Optional[float] = None
        self._gps_t = -1e9

    def set_gps_speed(self, mps: Optional[float], t: float) -> None:
        self.gps_mps, self._gps_t = mps, t

    @classmethod
    def grey_small(cls, img: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        return cv2.resize(g.astype(np.float32), (cls.W, cls.H), interpolation=cv2.INTER_AREA)

    def update(self, t: float, img: np.ndarray, dets: Sequence[Detection]) -> float:
        g = self.grey_small(img)
        gx = g[:, 2:] - g[:, :-2]
        if self._prev is not None and t > self._prev_t:
            mask = np.ones(gx.shape, bool)
            mask[int(self.H * (1 - self.hood_frac)):, :] = False
            for d in dets:
                if d.label in VEHICLE_LABELS + EMERGENCY_LABELS + ("person",):
                    cx, cy, bw, bh = d.bbox
                    y1, y2 = int(max(0, (cy - bh / 2) * self.H)), int(min(self.H, np.ceil((cy + bh / 2) * self.H)))
                    x1, x2 = int(max(0, (cx - bw / 2) * self.W) - 1), int(min(self.W - 2, np.ceil((cx + bw / 2) * self.W)))
                    mask[y1:y2, max(0, x1):x2] = False
            vals = np.abs(gx - self._prev)[mask]
            if vals.size >= 200:
                score = float(np.median(vals)) * min(3.0, 0.1 / (t - self._prev_t))
                self.flow = 0.6 * self.flow + 0.4 * score
        self._prev, self._prev_t = gx, t
        return self.flow

    def state(self, t: float) -> str:
        """stationary | slow | moving"""
        if self.gps_mps is not None and t - self._gps_t < 5.0:
            v = self.gps_mps
            return "stationary" if v < self.stationary_mps else "slow" if v < self.slow_mps else "moving"
        f = self.flow
        return "stationary" if f < self.stationary_thr else "slow" if f < self.slow_thr else "moving"


# ---------------------------------------------------------------------------------------------- analyzer
@dataclass
class SceneConfig:
    enabled: tuple[str, ...] = ("traffic", "crash", "lanes", "signals", "weather")
    drive_side: str = "right"             # right-hand traffic (US); "left" mirrors the side rules
    # traffic
    standstill_s: float = 75.0            # longer than a long red light
    standstill_stationary_frac: float = 0.9   # share of standstill_s read as stationary
    standstill_moving_max: float = 0.01       # share read as moving
    heavy_window_s: float = 150.0
    heavy_slow_frac: float = 0.5          # share of the window read slow or stopped (free city driving reads
                                          # slow 70 % of the time, so this alone says little)
    heavy_stationary_frac: float = 0.3    # share read stationary: what separates a jam from free driving
    heavy_min_stops: int = 2              # stop-and-go: separate stops inside the window
    heavy_min_vehicles: float = 3.0       # mean vehicles in view
    ahead_min_area: float = 0.015         # a vehicle "close ahead"
    red_veto_s: float = 20.0              # a red signal seen this recently explains a stop
    # crash
    crash_hold_s: float = 3.0
    crash_window_s: float = 10.0
    crash_max_shift: float = 0.5       # standing still: box centre moves less than this many box widths
    other_side_x: float = 0.22            # emergency scene entirely left of this (right-hand traffic)
    police_other_side: bool = False       # relabel far-side police as Police > Other side (opt in)
    # lanes
    lane_min_objects: int = 3
    lane_window_s: float = 3.0
    # Flat-road pinhole model of the camera. Where a box meets the road (its bottom edge, row gy) the point lies
    # d = f_y * cam_height / (gy - horizon_y) metres ahead and (cx - 0.5) * cam_height * aspect / (gy - horizon_y)
    # metres to the side (aspect = width / height, square pixels). The lateral distance needs no field of view,
    # only the mount height and the horizon row; the distance needs the horizontal field of view too.
    cam_height_m: float = 1.3             # phone on the windscreen of a car
    horizon_y: float = 0.50               # row of the horizon with the phone mounted level
    hfov_deg: float = 70.0                # Pixel main camera in 16:9 video
    lane_width_m: float = 3.6
    lane_max_dist_m: float = 40.0         # further away a cone is a few pixels and its row unreliable
    lane_min_dist_m: float = 3.0
    # offsets u are in half lane widths from our centre line: |u| <= 1 is our lane, 1 to 3 the next
    lane_max_offset: float = 2.0          # count objects in our lane and in the near half of the next one
    lane_reach: float = 0.8               # at least two objects well inside our lane (tuned on Argoverse 2 3D labels)
    lane_side_u: float = 0.6              # median offset beyond this: left / right; inside: center
    lane_hold_s: float = 1.5              # the same side for this long
    # dark signals
    dark_signal_s: float = 4.0
    signal_min_px: int = 18
    # weather
    weather_rules: dict = field(default_factory=lambda: dict(DEFAULT_WEATHER_RULES))
    weather_ema: float = 0.15


class SceneAnalyzer:
    def __init__(self, cfg: SceneConfig | None = None):
        self.cfg = cfg or SceneConfig()
        self.ego = EgoMotion()
        self.lights = EmergencyLights()
        self._hist: deque = deque()            # (t, ego state, n vehicles, vehicle close ahead)
        self._last_red = -1e9
        self._emergency: deque = deque()       # (t, label, bbox, lights)
        self._construction: deque = deque()    # (t, u, ground y, label)
        self._lane_side: Optional[str] = None
        self._lane_since = 0.0
        self._dark_since: Optional[float] = None
        self._weather: dict[str, float] = {}
        self._weather_since: dict[str, float] = {}
        self.state: dict[str, object] = {}

    # --------------------------------------------------------------------------- one frame
    def update(self, t: float, img: np.ndarray, dets: Sequence[Detection],
               weather: Optional[dict[str, float]] = None) -> list[Detection]:
        """Returns extra detections: emergency_lights boxes and scene_* labels."""
        cfg = self.cfg
        out: list[Detection] = []
        h, w = img.shape[:2]
        self.ego.update(t, img, dets)
        ego = self.ego.state(t)
        lights = self.lights.update(t, img, dets)
        out += lights
        vehicles = [d for d in dets if d.label in VEHICLE_LABELS and d.confidence >= 0.35]
        ahead = any(abs(d.bbox[0] - 0.5) < 0.18 and d.bbox[2] * d.bbox[3] >= cfg.ahead_min_area for d in vehicles)
        signals = [d for d in dets if d.label == "traffic light" and d.confidence >= 0.35
                   and d.bbox[3] * h >= cfg.signal_min_px]
        sig_colours = [lamp_fractions(img, d.bbox) for d in signals]
        if any(c["red"] > 0.02 or c["amber"] > 0.03 for c in sig_colours):
            self._last_red = t
        self._hist.append((t, ego, len(vehicles), ahead))
        while self._hist and t - self._hist[0][0] > max(cfg.heavy_window_s, cfg.standstill_s) + 5:
            self._hist.popleft()
        self.state = {"ego": ego, "flow": round(self.ego.flow, 4), "vehicles": len(vehicles)}

        if "traffic" in cfg.enabled:
            out += self._traffic(t)
        if "crash" in cfg.enabled:
            out += self._crash(t, dets, lights, ego)
        if "lanes" in cfg.enabled:
            out += self._lanes(t, dets, w / h)
        if "signals" in cfg.enabled:
            out += self._signals(t, signals, sig_colours)
        if "weather" in cfg.enabled and weather is not None:
            self._update_weather(t, weather)
        if "weather" in cfg.enabled:
            out += self._weather_labels(t)
        if cfg.police_other_side:
            out += self._police_side(dets)
        return out

    # --------------------------------------------------------------------------- traffic
    def _traffic(self, t: float) -> list[Detection]:
        cfg = self.cfg
        if t - self._last_red < cfg.red_veto_s:
            return []
        out = []
        # The camera-only ego estimate tells stopped from moving well but not slow from moving (its score stops
        # rising above about 3 m/s at 5 fps; tools/eval_ego_argoverse.py), so both rules lean on stationary time.
        # standstill: nearly every sample of the last standstill_s is stationary (a stop reads stationary on
        # 91-97 % of frames, so "every one" almost never held), hardly any moving, most with a vehicle ahead
        span = [s for s in self._hist if t - s[0] <= cfg.standstill_s]
        if span and t - span[0][0] >= cfg.standstill_s * 0.95:
            n = len(span)
            if (sum(s[1] == "stationary" for s in span) >= cfg.standstill_stationary_frac * n
                    and sum(s[1] == "moving" for s in span) <= cfg.standstill_moving_max * n
                    and sum(s[3] for s in span) >= 0.7 * n):
                out.append(_whole("scene_traffic_standstill", 0.9))
                return out
        win = [s for s in self._hist if t - s[0] <= cfg.heavy_window_s]
        if win and t - win[0][0] >= cfg.heavy_window_s * 0.95:
            slow = sum(s[1] != "moving" for s in win) / len(win)
            stationary = sum(s[1] == "stationary" for s in win) / len(win)
            stops, prev = 0, None
            for s in win:
                if s[1] == "stationary" and prev not in (None, "stationary"):
                    stops += 1
                prev = s[1]
            mean_veh = sum(s[2] for s in win) / len(win)
            if (slow >= cfg.heavy_slow_frac and stationary >= cfg.heavy_stationary_frac
                    and stops >= cfg.heavy_min_stops and mean_veh >= cfg.heavy_min_vehicles):
                out.append(_whole("scene_traffic_heavy", min(1.0, 0.5 + 0.5 * slow)))
        return out

    # --------------------------------------------------------------------------- crash
    def _crash(self, t: float, dets: Sequence[Detection], lights: Sequence[Detection], ego: str) -> list[Detection]:
        cfg = self.cfg
        # the vehicle classifier scores every class on a box; a box is one responder, of its best class
        best: dict[tuple, Detection] = {}
        for d in dets:
            key = tuple(d.bbox)
            if d.label in EMERGENCY_LABELS and (key not in best or d.confidence > best[key].confidence):
                best[key] = d
        for d in best.values():
            if d.confidence >= 0.5:
                # lights count only on a responder: a work truck's beacon or a flickering tail light is not one
                lit = any(_iou(d.bbox, L.bbox) > 0.5 for L in lights)
                self._emergency.append((t, d.label, d.bbox, lit))
        while self._emergency and t - self._emergency[0][0] > cfg.crash_window_s:
            self._emergency.popleft()
        # responders still in view: the window keeps them for 10 s, the scene ends when they leave the frame
        if not self._emergency or t - self._emergency[-1][0] > 1.0:
            return []
        span = self._emergency[-1][0] - self._emergency[0][0]
        if span < cfg.crash_hold_s:
            return []
        labels = [e[1] for e in self._emergency]
        heavy = sum(1 for lab in labels if lab in ("ambulance", "fire_truck"))
        lit = sum(1 for e in self._emergency if e[3])
        # a kind counts when seen at least twice: one truck the classifier calls police once and fire engine
        # once is not two responders
        kinds = {lab for lab in set(labels) if labels.count(lab) >= 2}
        # responders on scene, not responders on their way. While we drive, we approach them: their boxes
        # grow (an ambulance overtaking us shrinks). While we stand, they must stand too: an ambulance that
        # drives past a car waiting at a light crosses the frame and is not a crash.
        areas = [e[2][2] * e[2][3] for e in self._emergency]
        third = max(1, len(areas) // 3)
        approaching = float(np.mean(areas[-third:])) >= 1.3 * float(np.mean(areas[:third]))
        standing = _standing([e[2] for e in self._emergency], cfg.crash_max_shift)
        on_scene = (approaching and ego != "stationary") or (standing and ego != "moving")
        # an ambulance or fire engine with lights, or two kinds of responder; police alone is a traffic stop
        responders = (heavy >= 3 and lit >= 2) or (len(kinds) >= 2 and lit >= 2)
        if not (responders and on_scene):
            return []
        xs = [e[2][0] + (e[2][2] / 2 if cfg.drive_side == "right" else -e[2][2] / 2) for e in self._emergency]
        far = all(x < cfg.other_side_x for x in xs) if cfg.drive_side == "right" else \
            all(x > 1 - cfg.other_side_x for x in xs)
        conf = min(1.0, 0.55 + 0.05 * lit + 0.05 * heavy)
        return [_whole("scene_crash_other_side" if far else "scene_crash", conf)]

    def _police_side(self, dets: Sequence[Detection]) -> list[Detection]:
        cfg, out = self.cfg, []
        for d in dets:
            if d.label != "police_vehicle":
                continue
            right_edge = d.bbox[0] + d.bbox[2] / 2
            left_edge = d.bbox[0] - d.bbox[2] / 2
            far = right_edge < cfg.other_side_x if cfg.drive_side == "right" else left_edge > 1 - cfg.other_side_x
            if far:
                out.append(Detection("scene_police_other_side", d.confidence, d.bbox, model="scene"))
        return out

    # --------------------------------------------------------------------------- lanes
    def ground_point(self, bbox, aspect: float) -> Optional[tuple[float, float]]:
        """(distance ahead, lateral offset to the right) in metres of the point where a box meets the road, from
        the flat-road camera model; None above the horizon."""
        cfg = self.cfg
        dv = bbox[1] + bbox[3] / 2 - cfg.horizon_y
        if dv <= 1e-3:
            return None
        fy = 0.5 / math.tan(math.radians(cfg.hfov_deg) / 2) * aspect
        return fy * cfg.cam_height_m / dv, (bbox[0] - 0.5) * cfg.cam_height_m * aspect / dv

    def lane_offset(self, bbox, aspect: float = 16 / 9) -> Optional[float]:
        """Lateral offset in half lane widths (negative: left), or None when out of the distance range."""
        cfg = self.cfg
        g = self.ground_point(bbox, aspect)
        if g is None or not cfg.lane_min_dist_m <= g[0] <= cfg.lane_max_dist_m:
            return None
        return g[1] / (cfg.lane_width_m / 2)

    def _lanes(self, t: float, dets: Sequence[Detection], aspect: float) -> list[Detection]:
        cfg = self.cfg
        for d in dets:
            if d.label in CONSTRUCTION_LABELS and d.label != "construction_sign" and d.confidence >= 0.45:
                u = self.lane_offset(d.bbox, aspect)
                if u is not None and abs(u) <= cfg.lane_max_offset:
                    self._construction.append((t, u, d.bbox[1] + d.bbox[3] / 2, d.label))
        while self._construction and t - self._construction[0][0] > cfg.lane_window_s:
            self._construction.popleft()
        now = [c for c in self._construction if t - c[0] <= 0.5]
        us = sorted(c[1] for c in now)
        side = None
        if len(us) >= cfg.lane_min_objects and sum(1 for u in us if abs(u) <= cfg.lane_reach) >= 2:
            mid = us[len(us) // 2]
            side = "left" if mid < -cfg.lane_side_u else "right" if mid > cfg.lane_side_u else "center"
        if side != self._lane_side:
            self._lane_side, self._lane_since = side, t
        if side is None or t - self._lane_since < cfg.lane_hold_s:
            return []
        conf = min(1.0, 0.5 + 0.08 * len(now))
        return [_whole(f"scene_lane_{side}", conf)]

    # --------------------------------------------------------------------------- dark signals
    def _signals(self, t: float, signals: Sequence[Detection], colours: Sequence[dict]) -> list[Detection]:
        cfg = self.cfg
        lit_any = any(c["red"] > 0.01 or c["amber"] > 0.01 or c["green"] > 0.01 or c["lit"] > 0.03 for c in colours)
        close = [s for s in signals if s.bbox[3] >= 0.04]
        if len(close) >= 2 and not lit_any:
            if self._dark_since is None:
                self._dark_since = t
            if t - self._dark_since >= cfg.dark_signal_s:
                return [_whole("scene_dark_signal", 0.6)]
        else:
            self._dark_since = None
        return []

    # --------------------------------------------------------------------------- weather
    def _update_weather(self, t: float, probs: dict[str, float]) -> None:
        a = self.cfg.weather_ema
        for k, v in probs.items():
            self._weather[k] = (1 - a) * self._weather.get(k, v) + a * v
        for cls, (_label, thr, _hold) in self.cfg.weather_rules.items():
            if self._rule_value(cls) >= thr:
                self._weather_since.setdefault(cls, t)
            else:
                self._weather_since.pop(cls, None)

    def _rule_value(self, key: str) -> float:
        # a term none of whose classes any loaded model has is left out: without the road-surface model,
        # Slippery road rests on rain alone. A rule with no term left never fires.
        terms = [sum(self._weather.get(k, 0.0) for k in part.split("|")) for part in key.split("&")
                 if any(k in self._weather for k in part.split("|"))]
        return min(terms) if terms else 0.0

    def _weather_labels(self, t: float) -> list[Detection]:
        out = []
        for cls, (label, thr, hold) in self.cfg.weather_rules.items():
            since = self._weather_since.get(cls)
            if since is not None and t - since >= hold:
                out.append(_whole(label, self._rule_value(cls)))
        return out

    @property
    def weather(self) -> dict[str, float]:
        return dict(self._weather)
