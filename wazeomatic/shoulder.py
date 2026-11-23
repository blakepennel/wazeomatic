"""Hazard > Car on shoulder: a vehicle standing beside the road on our side, while we drive on a fast road.

In town every kerb is lined with parked cars that look exactly like a car on a shoulder, so the rule only runs
on a fast road. ``ShoulderCars`` holds three conditions:

* **Fast road.** Our speed stays at or above ``highway_mps`` (50 km/h) on at least ``highway_frac`` of the
  samples of the last ``highway_hold_s``. The GPS speed is used when there is a fresh one (``use_gps_speed``).
  Without it the camera measures the speed itself: the road-pair spread fit of ``motion.py`` gives
  ``rate = A * y + B`` per row, and on a flat road ``A`` times the focal length (in frame heights) times the
  mount height is our speed. On Argoverse 2 that reads 0.87 of the true speed at the median, so its threshold is
  ``camera_highway_mps``; Argoverse has almost no driving above 50 km/h, so the camera speed is unmeasured there.
* **Beside the road, our side.** The point where the box meets the road (flat-road camera model of
  ``SceneConfig``) lies ``min_dist_m`` to ``max_dist_m`` ahead and ``min_offset_m`` to ``max_offset_m`` to the
  drive side (right in right-hand traffic) of our camera.
* **Standing.** ``motion.StandingTracker`` reads the vehicle as standing still (optical flow against the road).

Each frame in which a vehicle meets all three gives a ``scene_car_on_shoulder`` box; the report manager's hit
rule and cooldown do the rest. The Kotlin twin is ``ShoulderCars.kt``.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .motion import MotionConfig, StandingTracker
from .schema import Detection

LABEL = "scene_car_on_shoulder"
VEHICLES = ("car", "truck", "bus")


@dataclass
class ShoulderConfig:
    require_highway: bool = True
    highway_mps: float = 13.9          # 50 km/h
    highway_hold_s: float = 15.0
    highway_frac: float = 0.8
    camera_speed: bool = True          # without GPS, the camera's own speed estimate
    camera_highway_mps: float = 12.0   # it reads about 0.87 of the true speed
    camera_speed_every_s: float = 0.5
    min_conf: float = 0.35
    min_box_px: int = 28
    min_dist_m: float = 8.0
    max_dist_m: float = 60.0
    min_offset_m: float = 2.5          # beyond our own lane (half a lane, 1.8 m, plus a margin)
    max_offset_m: float = 8.0


class ShoulderCars:
    def __init__(self, cfg: ShoulderConfig | None = None, horizon_y: float = 0.5, hfov_deg: float = 70.0,
                 cam_height_m: float = 1.3, drive_side: str = "right"):
        self.cfg = cfg or ShoulderConfig()
        self.horizon_y, self.hfov_deg, self.cam_height_m = horizon_y, hfov_deg, cam_height_m
        self.drive_side = drive_side
        self.tracker = StandingTracker(MotionConfig(horizon_y=horizon_y, drive_side=drive_side))
        self._speeds: deque = deque()      # (t, m/s, source)
        self._last_cam_t = -1e9
        self.highway = False
        self.speed: Optional[float] = None   # latest sample (m/s), for the status line
        self.last_candidates: list[Detection] = []

    # ---------------------------------------------------------------------------------- geometry
    def _fy(self, aspect: float) -> float:
        """Focal length in frame heights."""
        return 0.5 / math.tan(math.radians(self.hfov_deg) / 2) * aspect

    def ground_point(self, bbox, aspect: float) -> Optional[tuple[float, float]]:
        """(metres ahead, metres to the right) where the box meets a flat road; None at or above the horizon."""
        dv = bbox[1] + bbox[3] / 2 - self.horizon_y
        if dv <= 1e-3:
            return None
        return self._fy(aspect) * self.cam_height_m / dv, (bbox[0] - 0.5) * self.cam_height_m * aspect / dv

    def is_candidate(self, d: Detection, w: int, h: int) -> bool:
        cfg = self.cfg
        if d.label not in VEHICLES or d.confidence < cfg.min_conf or min(d.bbox[2] * w, d.bbox[3] * h) < cfg.min_box_px:
            return False
        g = self.ground_point(d.bbox, w / h)
        if g is None:
            return False
        side = g[1] if self.drive_side == "right" else -g[1]
        return cfg.min_dist_m <= g[0] <= cfg.max_dist_m and cfg.min_offset_m <= side <= cfg.max_offset_m

    # ---------------------------------------------------------------------------------- speed
    def camera_speed(self, aspect: float) -> Optional[float]:
        """Our speed from the tracker's last road fit (m/s), or None when there was no trustworthy fit."""
        road = self.tracker.last_road
        if road is None:
            return None
        return float(road[0]) * self._fy(aspect) * self.cam_height_m

    def _update_speed(self, t: float, gps_mps: Optional[float], gps_fresh: bool) -> None:
        cfg = self.cfg
        if gps_fresh and gps_mps is not None:
            self._speeds.append((t, float(gps_mps), "gps"))
            self.speed = float(gps_mps)
        while self._speeds and t - self._speeds[0][0] > cfg.highway_hold_s:
            self._speeds.popleft()
        if not self._speeds or t - self._speeds[0][0] < cfg.highway_hold_s * 0.6:
            self.highway = False
            return
        gps = [s for s in self._speeds if s[2] == "gps"]
        use, thr = (gps, cfg.highway_mps) if gps else (list(self._speeds), cfg.camera_highway_mps)
        self.highway = sum(s[1] >= thr for s in use) >= cfg.highway_frac * len(use)

    # ---------------------------------------------------------------------------------- frame
    def update(self, t: float, img: np.ndarray, dets: Sequence[Detection], ego_state: str,
               gps_mps: Optional[float] = None, gps_fresh: bool = False, hood_frac: float = 0.0) -> list[Detection]:
        cfg = self.cfg
        h, w = img.shape[:2]
        cand = [d for d in dets if self.is_candidate(d, w, h)]
        self.last_candidates = cand
        # the road fit (and so the camera speed) is made only when there is a target to measure; without GPS
        # ask for one every camera_speed_every_s by measuring the nearest vehicle, or the road alone
        need_speed = cfg.camera_speed and not gps_fresh and t - self._last_cam_t >= cfg.camera_speed_every_s
        before = self.tracker.last_road
        # off the fast road only the speed is measured: tracking every parked car in town would cost a car LK
        # pass a frame for nothing (a track needs three measurements, under a second once the gate opens)
        idle = cfg.require_highway and not self.highway
        targets = [] if idle else cand
        tracks = self.tracker.states(t, img, dets, targets, ego_state, hood_frac, force_road=need_speed)
        if need_speed and self.tracker.last_road is not None and self.tracker.last_road is not before:
            self._last_cam_t = t
            v = self.camera_speed(w / h)
            if v is not None:
                self._speeds.append((t, v, "camera"))
                self.speed = v
        self._update_speed(t, gps_mps, gps_fresh)
        if idle or (cfg.require_highway and not self.highway):
            return []
        return [Detection(LABEL, d.confidence, d.bbox, model="scene") for d, tr in zip(targets, tracks)
                if tr.state == "stopped"]
