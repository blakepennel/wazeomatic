"""Frame sources: video files, webcams, and a camera-style degradation filter.

The simulator feeds recorded dashcam video through the exact code path the phone
uses. ``CameraSim`` makes clean footage look like a phone camera on a dash mount:
lower resolution, mild blur, JPEG blocking and exposure drift.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


@dataclass
class Frame:
    image: np.ndarray      # BGR uint8
    index: int             # index in the source (decoded frame number)
    ts: float              # seconds since source start


@dataclass
class CameraSim:
    """Cheap approximation of a phone camera at 720p on a windshield mount."""

    width: int = 1280
    height: int = 720
    blur_sigma: float = 0.8
    jpeg_quality: int = 70
    exposure_jitter: float = 0.12     # +-fraction of brightness drift
    noise_sigma: float = 2.0
    seed: Optional[int] = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._gain = 1.0

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if img.shape[1] != self.width or img.shape[0] != self.height:
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.blur_sigma > 0:
            img = cv2.GaussianBlur(img, (0, 0), self.blur_sigma)
        # slow random-walk exposure drift, like auto-exposure hunting
        self._gain += self._rng.uniform(-0.02, 0.02)
        lo, hi = 1 - self.exposure_jitter, 1 + self.exposure_jitter
        self._gain = min(max(self._gain, lo), hi)
        out = img.astype(np.float32) * self._gain
        if self.noise_sigma > 0:
            out += np.random.default_rng(self._rng.randrange(1 << 30)).normal(0, self.noise_sigma, out.shape)
        img = np.clip(out, 0, 255).astype(np.uint8)
        if self.jpeg_quality < 100:
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if ok:
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img


class VideoSource:
    """Iterate frames of a video file at a fixed step (every ``step``-th frame)."""

    def __init__(self, path: str | Path, step: int = 1, loop: bool = False,
                 start_s: float = 0.0, max_frames: Optional[int] = None,
                 sim: Optional[CameraSim] = None):
        self.path = str(path)
        self.step = max(1, step)
        self.loop = loop
        self.start_s = start_s
        self.max_frames = max_frames
        self.sim = sim
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {self.path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def effective_fps(self) -> float:
        return self.fps / self.step

    def __iter__(self) -> Iterator[Frame]:
        emitted = 0
        while True:
            if self.start_s:
                self.cap.set(cv2.CAP_PROP_POS_MSEC, self.start_s * 1000.0)
            idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
            while True:
                ok = self.cap.grab()
                if not ok:
                    break
                if idx % self.step == 0:
                    ok, img = self.cap.retrieve()
                    if not ok:
                        break
                    if self.sim is not None:
                        img = self.sim(img)
                    yield Frame(img, idx, idx / self.fps)
                    emitted += 1
                    if self.max_frames and emitted >= self.max_frames:
                        return
                idx += 1
            if not self.loop:
                return
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self) -> None:
        self.cap.release()


class CameraSource:
    """Webcam / capture device by index (desktop or Raspberry Pi USB camera)."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, step: int = 1,
                 sim: Optional[CameraSim] = None, max_frames: Optional[int] = None):
        self.cap = cv2.VideoCapture(index)
        self.max_frames = max_frames
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {index}")
        self.step = max(1, step)
        self.sim = sim
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def effective_fps(self) -> float:
        return self.fps / self.step

    def __iter__(self) -> Iterator[Frame]:
        import time

        t0 = time.time()
        idx = 0
        emitted = 0
        while True:
            ok, img = self.cap.read()
            if not ok:
                return
            if idx % self.step == 0:
                if self.sim is not None:
                    img = self.sim(img)
                yield Frame(img, idx, time.time() - t0)
                emitted += 1
                if self.max_frames and emitted >= self.max_frames:
                    return
            idx += 1

    def release(self) -> None:
        self.cap.release()
