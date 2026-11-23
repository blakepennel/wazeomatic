import numpy as np

from test_motion import H, W, box_after, render, step, texture
from wazeomatic.motion import MotionConfig, StandingTracker
from wazeomatic.report_manager import ReportConfig, ReportManager
from wazeomatic.scene import SceneAnalyzer, SceneConfig
from wazeomatic.schema import Detection
from wazeomatic.shoulder import LABEL, ShoulderCars, ShoulderConfig

ASPECT = W / H


def car(box, conf=0.8, label="car"):
    return Detection(label, conf, box, model="coco")


def test_ground_point_and_candidates():
    sc = ShoulderCars(horizon_y=0.5, hfov_deg=70.0, cam_height_m=1.3)
    # 20 m ahead: the box bottom is fy * 1.3 / 20 below the horizon
    fy = 0.5 / np.tan(np.radians(35)) * ASPECT
    dv = fy * 1.3 / 20
    def at(lateral_m, w=0.08, h=0.08):
        cx = 0.5 + lateral_m * dv / (1.3 * ASPECT)
        return (cx, 0.5 + dv - h / 2, w, h)
    g = sc.ground_point(at(4.0), ASPECT)
    assert abs(g[0] - 20) < 1e-6 and abs(g[1] - 4.0) < 1e-6
    assert sc.is_candidate(car(at(4.0)), W, H)              # beside the road on the right
    assert not sc.is_candidate(car(at(0.0)), W, H)          # ahead in our lane
    assert not sc.is_candidate(car(at(-4.0)), W, H)         # the other side
    assert not sc.is_candidate(car(at(12.0)), W, H)         # too far out: a car park
    assert not sc.is_candidate(car(at(4.0), conf=0.2), W, H)
    assert not sc.is_candidate(car(at(4.0, w=0.02, h=0.02)), W, H)     # a few pixels
    assert not sc.is_candidate(car(at(4.0), label="person"), W, H)
    left = ShoulderCars(horizon_y=0.5, hfov_deg=70.0, cam_height_m=1.3, drive_side="left")
    assert left.is_candidate(car(at(-4.0)), W, H) and not left.is_candidate(car(at(4.0)), W, H)


def test_camera_speed_from_the_road_fit():
    sc = ShoulderCars(horizon_y=0.5, hfov_deg=70.0, cam_height_m=1.3)
    assert sc.camera_speed(ASPECT) is None
    fy = 0.5 / np.tan(np.radians(35)) * ASPECT
    sc.tracker.last_road = np.array([20.0 / (fy * 1.3), -10.0, 0.0, 0.0])
    assert abs(sc.camera_speed(ASPECT) - 20.0) < 1e-9


def _gate(speeds, dt=0.5, **cfg):
    sc = ShoulderCars(ShoulderConfig(**cfg))
    img = np.zeros((H, W, 3), np.uint8)
    for k, v in enumerate(speeds):
        sc.update(k * dt, img, [], "moving", gps_mps=v, gps_fresh=v is not None)
    return sc


def test_fast_road_needs_sustained_gps_speed():
    assert _gate([20.0] * 32).highway                          # 16 s at 72 km/h
    assert not _gate([20.0] * 10).highway                      # 5 s is too short to call it a highway
    assert not _gate([10.0] * 32).highway                      # 36 km/h: town
    assert not _gate([20.0] * 20 + [8.0] * 12).highway         # slowing down for a junction: 6 s of 30 s under
    assert _gate([20.0] * 28 + [12.0] * 4).highway             # one slow patch is fine
    # without GPS and without the camera speed there is nothing to open the condition
    assert not _gate([None] * 40, camera_speed=False).highway


def _drive(sc: ShoulderCars, car_static: bool, frames: int = 8, gps=None):
    """A textured world we drive into with a car beside the road on the right (part of the world, standing) or
    keeping its place in the image (driving beside us)."""
    world = texture(1)
    patch = texture(2, 60, 40)
    box0 = (0.69, 0.58, 60 / W, 40 / H)      # 14 m ahead, 3.7 m to the right
    x0, y0 = int(box0[0] * W - 30), int(box0[1] * H - 20)
    if car_static:
        world[y0:y0 + 40, x0:x0 + 60] = patch
    m, outs = np.eye(3), []
    for k in range(frames):
        if k:
            m = step(m, 1.03)
        img = render(world, m)
        if car_static:
            box = box_after(box0, m)
        else:
            img = img.copy()
            img[y0:y0 + 40, x0:x0 + 60] = patch
            box = box0
        outs.append(sc.update(0.2 * k, img, [car(box)], "moving", gps_mps=gps, gps_fresh=gps is not None))
    return outs


def _shoulder(**cfg):
    # the synthetic world zooms uniformly, a wall rather than a road, so the tracker's horizon check is off
    sc = ShoulderCars(ShoulderConfig(**cfg), horizon_y=0.5, hfov_deg=70.0, cam_height_m=1.5)
    sc.tracker = StandingTracker(MotionConfig(horizon_tol=0.0))
    return sc


def test_standing_car_beside_the_road_is_marked():
    outs = _drive(_shoulder(require_highway=False), car_static=True)
    assert not outs[0] and not outs[1]                        # pending until three measurements
    assert all(len(o) == 1 and o[0].label == LABEL for o in outs[4:])


def test_car_driving_beside_us_is_not_marked():
    outs = _drive(_shoulder(require_highway=False), car_static=False)
    assert not any(outs)


def test_nothing_in_town():
    # the same standing car, at 30 km/h on GPS: every kerb in town looks like this
    outs = _drive(_shoulder(), car_static=True, gps=8.0)
    assert not any(outs)


def test_scene_label_reports_car_on_shoulder():
    rm = ReportManager(ReportConfig(window=10, min_hits=3, global_cooldown_s=0, track_iou=0.0))
    box = (0.8, 0.66, 0.09, 0.11)
    events = []
    for _ in range(3):
        events += rm.update([Detection(LABEL, 0.8, box, model="scene")])
    assert [e.hazard_type for e in events] == ["HAZARD_SHOULDER_CAR_STOPPED"]
    # a plain car box is never a hazard on its own
    rm2 = ReportManager(ReportConfig(window=10, min_hits=1, global_cooldown_s=0))
    for _ in range(5):
        assert rm2.update([car(box, 0.9)]) == []


def test_scene_analyzer_runs_it_only_when_asked():
    img = np.zeros((H, W, 3), np.uint8)
    off = SceneAnalyzer(SceneConfig())
    assert off.shoulder is None and off.shoulder_cars(0.0, img, []) == []
    on = SceneAnalyzer(SceneConfig(shoulder_cars=True))
    for k in range(32):
        on.ego.set_gps_speed(20.0, k * 0.5)
        on.update(k * 0.5, img, [])
        on.shoulder_cars(k * 0.5, img, [])
    assert on.state["fast_road"] is True and on.state["speed"] == 20.0


def test_other_side_from_the_ground_point():
    sc = SceneAnalyzer(SceneConfig(horizon_y=0.5, hfov_deg=70.0, cam_height_m=1.3))
    fy = 0.5 / np.tan(np.radians(35)) * ASPECT
    dv = fy * 1.3 / 25
    def at(lateral_m):
        return (0.5 + lateral_m * dv / (1.3 * ASPECT), 0.5 + dv - 0.04, 0.06, 0.08)
    assert sc.other_side(at(-5.0), ASPECT)                  # oncoming lane, 25 m ahead
    assert not sc.other_side(at(-1.0), ASPECT)              # ahead in our lane
    assert not sc.other_side(at(4.0), ASPECT)               # our side
    assert not sc.other_side((0.1, 0.3, 0.05, 0.05), ASPECT)  # above the horizon: no ground point
    # left-hand traffic: mirrored
    lht = SceneAnalyzer(SceneConfig(horizon_y=0.5, hfov_deg=70.0, cam_height_m=1.3, drive_side="left"))
    assert lht.other_side(at(5.0), ASPECT) and not lht.other_side(at(-5.0), ASPECT)
