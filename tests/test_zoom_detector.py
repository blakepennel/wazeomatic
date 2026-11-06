import numpy as np

from wazeomatic.detector import ZoomDetector, merge_detections
from wazeomatic.schema import Detection


class FakeBase:
    """Returns one fixed box per call; records the frame sizes it was called with."""

    labels = ["cone"]
    name = "fake"
    size = 640
    conf_thres = 0.3

    def __init__(self, boxes_per_call):
        self.boxes_per_call = list(boxes_per_call)
        self.calls = []
        self.last_ms = 1.0

    def __call__(self, frame):
        self.calls.append(frame.shape[:2])
        boxes = self.boxes_per_call.pop(0)
        return [Detection("cone", c, b, "fake") for c, b in boxes]


def test_zoom_detector_maps_crop_boxes_back_to_frame():
    # full pass: nothing; crop pass: a box in the middle of the crop
    base = FakeBase([[], [(0.9, (0.5, 0.5, 0.2, 0.5))]])
    zd = ZoomDetector(base, crop=(0.25, 0.35, 0.75, 0.75))
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    dets = zd(frame)
    assert base.calls == [(720, 1280), (288, 640)]
    assert len(dets) == 1
    cx, cy, w, h = dets[0].bbox
    assert abs(cx - 0.5) < 1e-6 and abs(cy - 0.55) < 1e-6
    assert abs(w - 0.1) < 1e-6 and abs(h - 0.2) < 1e-6
    assert zd.last_ms == 2.0


def test_zoom_detector_merges_duplicates_keeping_higher_confidence():
    # the same cone seen by both passes: full pass at 0.4, crop pass at 0.8
    base = FakeBase([[(0.4, (0.5, 0.55, 0.1, 0.2))], [(0.8, (0.5, 0.5, 0.2, 0.5))]])
    zd = ZoomDetector(base)
    dets = zd(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert len(dets) == 1
    assert dets[0].confidence == 0.8


def test_merge_detections_is_per_class():
    a = Detection("cone", 0.9, (0.5, 0.5, 0.1, 0.1))
    b = Detection("pothole", 0.8, (0.5, 0.5, 0.1, 0.1))
    assert len(merge_detections([a, b])) == 2
