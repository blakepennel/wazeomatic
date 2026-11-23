"""Is a police car standing still? Waze's Police report is for a police car that stands (a speed trap, a traffic
stop), not one driving past, so the classifier's police boxes go through this check before they count.

``PoliceMotion`` follows each police box from frame to frame and asks whether its image motion is what a
stationary object at that place would do while we drive:

* **We are moving.** Sparse corners on the road (Shi-Tomasi, pyramidal Lucas-Kanade with a forward-backward
  check) give the expansion of the static world row by row: two road points on the same row spread apart at a
  rate that grows linearly with the row (flat road), fitted as ``rate = A * y + B``. A spread rate needs no
  horizon and no rotation, which is why the road is measured with pairs; a fit whose zero row ``-B / A`` is
  far from the horizon is not trusted (it happens at walking pace, in turns and with few road corners). A static point at the bottom of the
  car's box should flow away from the focus of expansion at ``(p - foe) * (A * y_bottom + B) * dt``. Whatever
  road flow that model leaves over, its median, is the camera's rotation (yaw and pitch shift the whole image),
  and is taken off the car's flow too, so no gyro is needed. The residual is
  ``|car flow - predicted static flow| / (|predicted| + 2 px)``; its median over the track is compared with a
  threshold that is stricter on the oncoming side, where slow oncoming cars look most like parked ones.
  Flow alone is blind along our own direction of travel (a car ahead driving at our pace sits near the focus
  of expansion, where a static object barely moves either), so the car's own points must also spread apart at
  least 0.3 times as fast as the road on the row it stands on (``min_spread_ratio``; a static car scores
  about 1.5, a car driving ahead at our pace about 0).
* **We stand.** A car that stands too keeps its box: centre shift under ``still_shift`` box widths and size
  change under ``still_growth`` over ``still_span_s``; most such checks failing means it drives.

Each police track is *pending* until it has ``min_measures`` measurements in one of the two regimes, then
*stopped* or *moving*. Police boxes of pending and moving tracks are relabelled (``police_vehicle_pending``,
``police_vehicle_moving``), which no report maps, so only a car read as standing reaches the Police report.

Measured on 163 held-out Argoverse 2 clips against the 3D tracks (``tools/label_vehicle_motion_argoverse.py``,
``tools/eval_police_motion.py``); the Kotlin twin is ``PoliceMotion.kt``.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from .schema import Detection

POLICE = "police_vehicle"
PENDING = "police_vehicle_pending"
MOVING = "police_vehicle_moving"
OTHER_SIDE = "scene_police_other_side"    # SceneAnalyzer's far-side police: held back the same way
MASK_LABELS = ("car", "truck", "bus", "person", "police_vehicle", "ambulance", "fire_truck", "bicycle",
               "motorcycle")


@dataclass
class MotionConfig:
    width: int = 640                   # frames are measured at this width
    horizon_y: float = 0.50            # row of the horizon (SceneConfig.horizon_y)
    drive_side: str = "right"
    track_min_conf: float = 0.3        # police score that starts or extends a track
    link_iou: float = 0.3              # against the box predicted from the car's own flow
    track_gap_s: float = 1.5
    min_measures: int = 3              # measurements before a track is decided
    window: int = 12                   # newest measurements per regime that count
    # moving regime
    road_corners: int = 250
    car_corners: int = 40
    min_road_pairs: int = 20
    horizon_tol: float = 0.12          # a flat road's spread rate is zero at the horizon: a fit that puts
                                       # that row further off is not the road (low speed, a wall, a turn);
                                       # 0 turns the check off
    min_car_points: int = 8
    road_lk: tuple = (21, 4)           # LK window (px) and pyramid levels for road corners
    car_lk: tuple = (15, 3)            # smaller for the car: a 21 px window on a small box mostly sees the road
    min_expansion: float = 0.02        # 1/s at the car's bottom row; below it we hardly move, nothing to compare
    oncoming_x: float = 0.4            # box centre left of this (right-hand traffic): the oncoming side
    resid_oncoming: float = 0.3
    resid_other: float = 0.6
    min_spread_ratio: float = 0.3      # car spread rate / static rate at its row: a car we hardly gain on
    # standing regime
    still_span_s: float = 1.0
    still_shift: float = 0.1           # box widths
    still_growth: float = 0.1          # |log area ratio| / 2, i.e. relative width change


LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)


def track_points(a: np.ndarray, b: np.ndarray, pts: np.ndarray, lk: tuple = (21, 4),
                 fb_px: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Pyramidal LK (window, levels) from grey frame a to b with a forward-backward check; returns matched (N, 2)
    point arrays."""
    LK = dict(winSize=(lk[0], lk[0]), maxLevel=lk[1], criteria=LK_CRITERIA)
    if pts is None or len(pts) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    p0 = pts.reshape(-1, 1, 2).astype(np.float32)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None, **LK)
    p0r, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, p1, None, **LK)
    ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(p0r - p0, axis=2)[:, 0] < fb_px)
    return p0[ok][:, 0], p1[ok][:, 0]


def road_pair_rate(a: np.ndarray, b: np.ndarray, h: int, dt: float, min_pairs: int = 20) -> Optional[tuple[float, float]]:
    """Fit the horizontal log-spread rate of road point pairs on nearly the same row as ``A * y + B`` (y as a
    fraction of the frame height, rate per second), robustly: three least-squares passes, each keeping the
    60 % of pairs closest to the previous fit."""
    if len(a) < 2:
        return None
    ya = a[:, 1] / h
    i, j = np.triu_indices(len(a), 1)
    sel = (np.abs(ya[i] - ya[j]) < 0.015) & (np.abs(a[i, 0] - a[j, 0]) > 15)
    if int(sel.sum()) < min_pairs:
        return None
    i, j = i[sel], j[sel]
    da, db = np.abs(a[i, 0] - a[j, 0]), np.abs(b[i, 0] - b[j, 0])
    ok = db > 1e-3
    if int(ok.sum()) < min_pairs:
        return None
    rate = np.log(db[ok] / da[ok]) / dt
    y = ((ya[i] + ya[j]) / 2)[ok]
    X = np.stack([y, np.ones(len(y))], 1)
    sol = np.linalg.lstsq(X, rate, rcond=None)[0]
    for _ in range(2):
        res = np.abs(X @ sol - rate)
        keep = res <= np.percentile(res, 60)
        sol = np.linalg.lstsq(X[keep], rate[keep], rcond=None)[0]
    return float(sol[0]), float(sol[1])


class _Track:
    def __init__(self, t: float, bbox):
        self.boxes: deque = deque(maxlen=64)       # (t, bbox, ego stationary)
        self.boxes.append((t, tuple(bbox), False))
        self.resid: deque = deque()                # (residual, centre x, spread ratio)
        self.still: deque = deque()                # True when the box stood over still_span_s
        self.state = "pending"

    @property
    def last_t(self) -> float:
        return self.boxes[-1][0]

    @property
    def last_box(self):
        return self.boxes[-1][1]


class StandingTracker:
    """Follows a chosen set of vehicle boxes from frame to frame and reads each as pending, stopped or moving
    (the method in the module docstring). ``PoliceMotion`` and ``shoulder.ShoulderCars`` use it."""

    def __init__(self, cfg: MotionConfig | None = None):
        self.cfg = cfg or MotionConfig()
        self._prev: Optional[np.ndarray] = None
        self._prev_t = 0.0
        self._tracks: list[_Track] = []
        self.last_road: Optional[tuple[float, float, float, float]] = None   # A, B, shift x, shift y
        self.last_tracks: list[_Track] = []

    def states(self, t: float, img: np.ndarray, dets: Sequence[Detection], targets: Sequence[Detection],
               ego_state: str, hood_frac: float = 0.0, force_road: bool = False) -> list[_Track]:
        """One frame: the track of each of ``targets`` (boxes out of ``dets``), with its state. Call it on every
        frame, with no targets when there are none, so the frame-to-frame flow has the previous frame.
        ``force_road`` fits the road (``last_road``) even when no target needs it."""
        cfg = self.cfg
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        h = int(round(g.shape[0] * cfg.width / g.shape[1]))
        g = cv2.resize(g, (cfg.width, h), interpolation=cv2.INTER_AREA)
        prev, prev_t = self._prev, self._prev_t
        self._prev, self._prev_t = g, t
        self._tracks = [tr for tr in self._tracks if t - tr.last_t <= cfg.track_gap_s]

        police = list(targets)
        stationary = ego_state == "stationary"
        h, w = g.shape
        dt = t - prev_t
        road = None
        if force_road and prev is not None and dt > 0 and not stationary:
            road = self._road(prev, g, dets, dt, hood_frac) or False
        if not police:
            self.last_tracks = []
            return []
        # where each track seen in the last frame should be now: its own corners followed with LK (a near car
        # moves a box width between frames at 5 fps, so the last box alone links poorly)
        follow, pred = {}, {}
        for tr in self._tracks:
            box = tr.last_box
            if prev is not None and not stationary and dt > 0 and abs(tr.last_t - prev_t) < 1e-6:
                f = self._follow(prev, g, box)
                if f is not None:
                    follow[id(tr)] = f
                    pa, pb = f
                    d = np.median(pb - pa, 0)
                    sc = float(np.exp(_spread(pa, pb) or 0.0))
                    box = (box[0] + d[0] / w, box[1] + d[1] / h, box[2] * sc, box[3] * sc)
            pred[id(tr)] = box
        cand = sorted(((_iou(pred[id(tr)], d.bbox), k, n) for k, d in enumerate(police)
                       for n, tr in enumerate(self._tracks) if tr.last_t < t), reverse=True)
        assigned: dict[int, _Track] = {}
        taken: set[int] = set()
        for ov, k, n in cand:
            if ov < cfg.link_iou:
                break
            if k in assigned or n in taken:
                continue
            assigned[k] = self._tracks[n]
            taken.add(n)
        for k, d in enumerate(police):
            tr = assigned.get(k)
            if tr is None:
                assigned[k] = _Track(t, d.bbox)
                self._tracks.append(assigned[k])
                continue
            prev_box = tr.last_box
            tr.boxes.append((t, tuple(d.bbox), stationary))
            if stationary:
                self._still_check(tr, t)
            elif id(tr) in follow:
                if road is None:
                    road = self._road(prev, g, dets, dt, hood_frac) or False
                if road:
                    m = self._measure(*follow[id(tr)], prev_box, road, dt, h, w)
                    if m is not None:
                        tr.resid.append((m[0], prev_box[0], m[1]))
                        while len(tr.resid) > cfg.window:
                            tr.resid.popleft()
            self._decide(tr)

        self.last_tracks = [assigned[k] for k in range(len(police))]     # for evaluation tools
        return self.last_tracks


    # ------------------------------------------------------------------------------------------- regimes
    def _still_check(self, tr: _Track, t: float) -> None:
        cfg = self.cfg
        old = [b for b in tr.boxes if t - b[0] >= cfg.still_span_s * 0.8 and b[2]]
        if not old:
            return
        t0, b0, _ = old[-1]
        b1 = tr.boxes[-1][1]
        shift = abs(b1[0] - b0[0]) / max(1e-6, (b0[2] + b1[2]) / 2)
        growth = abs(np.log(max(1e-9, b1[2] * b1[3]) / max(1e-9, b0[2] * b0[3]))) / 2
        tr.still.append(shift < cfg.still_shift and growth < cfg.still_growth)
        while len(tr.still) > cfg.window:
            tr.still.popleft()

    def _decide(self, tr: _Track) -> None:
        cfg = self.cfg
        n_m, n_s = len(tr.resid), len(tr.still)
        if max(n_m, n_s) < cfg.min_measures:
            return
        if n_m >= n_s:
            r = float(np.median([x[0] for x in tr.resid]))
            x = float(np.median([x[1] for x in tr.resid]))
            oncoming = x < cfg.oncoming_x if cfg.drive_side == "right" else x > 1 - cfg.oncoming_x
            spread = float(np.median([x[2] for x in tr.resid]))
            moving = r > (cfg.resid_oncoming if oncoming else cfg.resid_other) or spread < cfg.min_spread_ratio
            tr.state = "moving" if moving else "stopped"
        else:
            tr.state = "stopped" if sum(tr.still) * 2 >= n_s else "moving"

    # ------------------------------------------------------------------------------------------- flow
    def _road(self, a: np.ndarray, b: np.ndarray, dets: Sequence[Detection], dt: float,
              hood_frac: float) -> Optional[tuple[float, float, float, float]]:
        """Road expansion fit and camera rotation shift (px) between grey frames a and b."""
        cfg = self.cfg
        h, w = a.shape
        mask = np.zeros_like(a)
        y1, y2 = int((cfg.horizon_y + 0.06) * h), int((0.97 - hood_frac) * h)
        if y2 - y1 < 10:
            return None
        mask[y1:y2, int(0.1 * w):int(0.9 * w)] = 255
        for d in dets:
            if d.label in MASK_LABELS:
                cx, cy, bw, bh = d.bbox
                mask[max(0, int((cy - bh / 2) * h)):int((cy + bh / 2) * h) + 1,
                     max(0, int((cx - bw / 2) * w)):int((cx + bw / 2) * w) + 1] = 0
        pts = cv2.goodFeaturesToTrack(a, cfg.road_corners, 0.005, 6, mask=mask)
        if pts is None:
            return None
        pa, pb = track_points(a, b, pts, cfg.road_lk)
        if len(pa) < 15:
            return None
        fit = road_pair_rate(pa, pb, h, dt, cfg.min_road_pairs)
        if fit is None:
            return None
        A, B = fit
        if cfg.horizon_tol > 0 and (not A > 0 or abs(-B / A - cfg.horizon_y) > cfg.horizon_tol):
            return None
        e = A * pa[:, 1] / h + B
        pred = (pa - np.array([w / 2, cfg.horizon_y * h])) * e[:, None] * dt
        sx, sy = np.median(pb - pa - pred, 0)
        self.last_road = (A, B, float(sx), float(sy))
        return self.last_road

    def _follow(self, a: np.ndarray, b: np.ndarray, box) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """The car's corners (inner 80 % of its box in frame a) followed into frame b."""
        cfg = self.cfg
        h, w = a.shape
        cx, cy, bw, bh = box
        if bw * w < 24 or bh * h < 16:
            return None
        m = np.zeros_like(a)
        m[max(0, int((cy - .4 * bh) * h)):int((cy + .4 * bh) * h), max(0, int((cx - .4 * bw) * w)):int((cx + .4 * bw) * w)] = 255
        pts = cv2.goodFeaturesToTrack(a, cfg.car_corners, 0.01, 3, mask=m)
        if pts is None or len(pts) < cfg.min_car_points:
            return None
        pa, pb = track_points(a, b, pts, cfg.car_lk)
        return (pa, pb) if len(pa) >= cfg.min_car_points else None

    def _measure(self, pa: np.ndarray, pb: np.ndarray, box, road, dt: float, h: int, w: int
                 ) -> Optional[tuple[float, float]]:
        """(flow residual, spread ratio) of the car whose corners moved from pa to pb, against a static object
        standing where the box meets the road."""
        cfg = self.cfg
        cx, cy, bw, bh = box
        A, B, sx, sy = road
        e = A * (cy + bh / 2) + B
        if not e > cfg.min_expansion:
            return None
        ctr = np.median(pa, 0)
        pred = (ctr - np.array([w / 2, cfg.horizon_y * h])) * e * dt
        act = np.median(pb - pa, 0) - np.array([sx, sy])
        resid = float(np.linalg.norm(act - pred) / (np.linalg.norm(pred) + 2.0))
        spread = _spread(pa, pb)
        if spread is None:
            return None
        return resid, spread / dt / e


class PoliceMotion(StandingTracker):
    """Relabels the police boxes of cars not read as standing (``police_vehicle_pending`` / ``_moving``)."""

    def update(self, t: float, img: np.ndarray, dets: Sequence[Detection], ego_state: str,
               hood_frac: float = 0.0) -> list[Detection]:
        """Returns ``dets`` with the police boxes of pending and moving tracks relabelled."""
        police = [d for d in dets if d.label == POLICE and d.confidence >= self.cfg.track_min_conf]
        tracks = self.states(t, img, dets, police, ego_state, hood_frac)
        assigned = dict(enumerate(tracks))
        out = []
        index = {id(d): k for k, d in enumerate(police)}
        held = {tuple(d.bbox): assigned[k].state for k, d in enumerate(police) if assigned[k].state != "stopped"}
        for d in dets:
            k = index.get(id(d))
            state = assigned[k].state if k is not None else held.get(tuple(d.bbox)) if d.label == OTHER_SIDE else None
            if state is None or state == "stopped":
                out.append(d)
            else:
                tag = "_moving" if state == "moving" else "_pending"
                out.append(Detection(label=d.label + tag, confidence=d.confidence, bbox=d.bbox, model=d.model))
        return out


def _spread(pa: np.ndarray, pb: np.ndarray, min_dx: float = 8.0, min_pairs: int = 10) -> Optional[float]:
    """Median log change of the horizontal distance between point pairs: how much a set of points grew."""
    i, j = np.triu_indices(len(pa), 1)
    da, db = np.abs(pa[i, 0] - pa[j, 0]), np.abs(pb[i, 0] - pb[j, 0])
    sel = (da > min_dx) & (db > 1e-3)
    if int(sel.sum()) < min_pairs:
        return None
    return float(np.median(np.log(db[sel] / da[sel])))


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0
