"""The second-stage classifiers without a model: crop rule, candidate choice, label mapping, the road patch."""
import numpy as np

from wazeomatic.classifiers import RoadSurface, SceneClassifier, SceneClassifiers, VehicleClassifier, square_crop
from wazeomatic.schema import Detection


class FakeClf:
    """Stands in for OnnxClassifier: returns fixed probabilities and records the crops it was given."""

    def __init__(self, labels, probs, size=64):
        self.labels, self.probs, self.size = list(labels), np.asarray(probs, np.float32), size
        self.seen = []
        self.last_ms = 1.0

    def predict(self, crops):
        self.seen.append(crops)
        return np.repeat(self.probs[None], len(crops), axis=0)


def vehicle_clf(probs, **kw):
    vc = VehicleClassifier.__new__(VehicleClassifier)
    vc.clf = FakeClf(["civilian", "police", "ambulance", "fire"], probs)
    vc.clfs = [vc.clf] + kw.get("more", [])
    vc.max_crops, vc.min_box_px, vc.min_vehicle_conf, vc.pad = kw.get("max_crops", 4), 28, 0.35, 0.12
    vc.last_ms, vc.last_probs = 0.0, []
    return vc


def test_square_crop_letterboxes_wide_box():
    img = np.zeros((100, 200, 3), np.uint8)
    img[40:60, 50:150] = 255                              # a wide white bar
    out = square_crop(img, (0.5, 0.5, 0.5, 0.2), pad=0.0, size=64)
    assert out.shape == (64, 64, 3)
    assert (out[0, 32] == 114).all() and (out[-1, 32] == 114).all()   # grey bars above and below
    assert out[32, 32].min() == 255                        # the bar fills the middle


def test_square_crop_clamps_at_frame_edge():
    img = np.full((100, 100, 3), 50, np.uint8)
    out = square_crop(img, (0.02, 0.98, 0.2, 0.2), size=32)
    assert out.shape == (32, 32, 3)
    out = square_crop(img, (1.3, 1.3, 0.1, 0.1), size=32)   # entirely outside: no division by zero
    assert out.shape == (32, 32, 3)


def test_vehicle_candidates_largest_vehicles_only():
    vc = vehicle_clf([0.1, 0.8, 0.05, 0.05], max_crops=2)
    img = np.zeros((480, 640, 3), np.uint8)
    dets = [
        Detection("car", 0.9, (0.5, 0.5, 0.10, 0.10)),
        Detection("car", 0.9, (0.3, 0.5, 0.30, 0.30)),
        Detection("truck", 0.9, (0.7, 0.5, 0.20, 0.20)),
        Detection("car", 0.2, (0.5, 0.5, 0.40, 0.40)),      # too unsure to crop
        Detection("car", 0.9, (0.5, 0.5, 0.02, 0.02)),      # under min_box_px
        Detection("person", 0.9, (0.5, 0.5, 0.5, 0.5)),
    ]
    c = vc.candidates(img, dets)
    assert [d.bbox[2] for d in c] == [0.30, 0.20]


def test_vehicle_emits_police_with_vehicle_box():
    vc = vehicle_clf([0.1, 0.8, 0.06, 0.04])
    img = np.zeros((480, 640, 3), np.uint8)
    box = (0.4, 0.6, 0.2, 0.2)
    out = vc(img, [Detection("car", 0.9, box)])
    labels = {d.label: d for d in out}
    assert labels["police_vehicle"].confidence == np.float32(0.8)
    assert labels["police_vehicle"].bbox == box and labels["police_vehicle"].model == "vehicle"
    assert "ambulance" in labels and "fire_truck" not in labels     # 0.04 is under the 0.05 report floor
    assert vc.clf.seen[0][0].shape == (64, 64, 3)
    assert vc.last_probs[0][1]["civilian"] == np.float32(0.1)


def test_vehicle_averages_several_models():
    # a second model with its classes in another order and a larger input: averaged by class name
    other = FakeClf(["police", "civilian", "fire", "ambulance"], [0.4, 0.5, 0.1, 0.0], size=96)
    vc = vehicle_clf([0.1, 0.8, 0.06, 0.04], more=[other])
    out = {d.label: d.confidence for d in vc(np.zeros((480, 640, 3), np.uint8), [Detection("car", 0.9, (0.4, 0.6, 0.2, 0.2))])}
    assert abs(out["police_vehicle"] - 0.6) < 1e-6         # (0.8 + 0.4) / 2: one sure model alone is not enough
    assert abs(out["fire_truck"] - 0.07) < 1e-6
    assert "ambulance" not in out                            # (0.06 + 0) / 2 = 0.03, under the 0.05 floor
    assert other.seen[0][0].shape == (96, 96, 3)             # each model gets crops at its own size
    assert vc.last_ms == 2.0


def test_vehicle_no_candidates_no_model_call():
    vc = vehicle_clf([1, 0, 0, 0])
    assert vc(np.zeros((100, 100, 3), np.uint8), [Detection("person", 0.9, (0.5, 0.5, 0.3, 0.3))]) == []
    assert vc.clf.seen == []


def scene_part(cls, labels, probs):
    p = cls.__new__(cls)
    p.clf = FakeClf(labels, probs)
    p.every_s, p._last_t, p.last, p.last_ms = 0.5, -1e9, {}, 0.0
    if cls is RoadSurface:
        p.patch, p.hood_frac, p.topview, p.skipped = (0.40, 0.60, 0.60, 0.86), 0.0, None, 0
        p.dist_m, p.size_m, p.max_occlusion = 15.0, 3.6, 0.10
    return p


def test_scene_classifier_rate_limited():
    sc = scene_part(SceneClassifier, ["clear", "fog"], [0.3, 0.7])
    img = np.zeros((90, 160, 3), np.uint8)
    assert sc(img, 0.0) == {"clear": np.float32(0.3), "fog": np.float32(0.7)}
    assert sc(img, 0.2) is None
    assert sc(img, 0.6) is not None


def test_road_patch_moves_above_tall_hood():
    rs = scene_part(RoadSurface, ["dry", "wet"], [0.9, 0.1])
    x1, y1, x2, y2 = rs.region(1000, 1000)
    assert (x1, x2) == (400, 600) and y2 == 860
    rs.hood_frac = 0.30
    _, y1b, _, y2b = rs.region(1000, 1000)
    assert abs(y2b - 680) <= 1 and abs(y1b - 450) <= 1                    # bottom above the hood, top never above 0.45


def test_scene_classifiers_merge_and_prefix():
    w = scene_part(SceneClassifier, ["clear", "rain"], [0.2, 0.8])
    s = scene_part(RoadSurface, ["dry", "wet"], [0.4, 0.6])
    both = SceneClassifiers(w, None, s)
    both.set_hood_frac(0.2)
    assert s.hood_frac == 0.2
    r = both(np.zeros((90, 160, 3), np.uint8), 0.0)
    assert set(r) == {"clear", "rain", "surface_dry", "surface_wet"}
    assert both.last_ms == 2.0


def test_topview_quad_matches_the_camera_model():
    from wazeomatic.classifiers import topview_quad
    q = topview_quad(1280, 720, 70.0, 1.3, 0.5, 15.0, 3.6)
    # far edge above the near edge, symmetric about the centre, narrower far away
    assert q[0, 1] < q[3, 1] and abs((q[0, 0] + q[1, 0]) / 2 - 640) < 1e-3
    assert (q[1, 0] - q[0, 0]) < (q[2, 0] - q[3, 0])
    # the near edge sits where the blocked-lane rule puts a point 13.2 m ahead
    from wazeomatic.scene import SceneAnalyzer, SceneConfig
    sa = SceneAnalyzer(SceneConfig(hfov_deg=70.0, cam_height_m=1.3, horizon_y=0.5))
    d, _ = sa.ground_point((0.5, q[3, 1] / 720, 0.0, 0.0), 1280 / 720)
    assert abs(d - 13.2) < 0.05


def test_topview_skips_a_covered_road():
    s = scene_part(RoadSurface, ["dry", "wet"], [0.4, 0.6])
    s.topview = (70.0, 1.3, 0.5)
    img = np.zeros((720, 1280, 3), np.uint8)
    car = Detection("car", 0.9, (0.5, 0.6, 0.3, 0.2))
    assert s(img, 0.0, [car]) is None and s.skipped == 1
    assert s(img, 0.1, []) == {"surface_dry": np.float32(0.4), "surface_wet": np.float32(0.6)}
