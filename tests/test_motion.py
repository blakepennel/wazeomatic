import cv2
import numpy as np

from wazeomatic.motion import MOVING, PENDING, POLICE, MotionConfig, PoliceMotion, road_pair_rate
from wazeomatic.schema import Detection

W, H = 640, 360
FOE = np.array([W / 2, H / 2])          # horizon_y 0.5


def texture(seed: int, w: int = W, h: int = H) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 255, (h // 4, w // 4)).astype(np.uint8)
    return cv2.GaussianBlur(cv2.resize(t, (w, h), interpolation=cv2.INTER_CUBIC), (3, 3), 0)


def step(m: np.ndarray, s: float, shift=(0.0, 0.0)) -> np.ndarray:
    """World -> next frame: the last frame scaled about the focus of expansion (driving) and shifted (turning)."""
    z = np.array([[s, 0, (1 - s) * FOE[0] + shift[0]], [0, s, (1 - s) * FOE[1] + shift[1]], [0, 0, 1]])
    return z @ m


def render(world: np.ndarray, m: np.ndarray) -> np.ndarray:
    return cv2.warpAffine(world, m[:2].astype(np.float32), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def box_after(box, m: np.ndarray):
    cx, cy, bw, bh = box
    x, y, _ = m @ [cx * W, cy * H, 1.0]
    return (x / W, y / H, bw * m[0, 0], bh * m[1, 1])


def drive(car_static: bool, frames: int = 6, s: float = 1.03, turn=(0.0, 0.0), ego="moving", cfg=None):
    """A textured world we drive into; a police car at the right of the road that is part of the world (static)
    or keeps its place and size in the image (driving ahead at our speed)."""
    world = texture(1)
    car = texture(2, 60, 40)
    box0 = (0.75, 0.68, 60 / W, 40 / H)
    x0, y0 = int(box0[0] * W - 30), int(box0[1] * H - 20)
    if car_static:
        world[y0:y0 + 40, x0:x0 + 60] = car
    # the synthetic world zooms uniformly, a wall rather than a road: its spread rate has no zero row, so the
    # horizon check is off here (test_uniform_zoom_is_not_a_road checks it)
    pm, states = PoliceMotion(cfg or MotionConfig(horizon_tol=0.0)), []
    m = np.eye(3)
    for k in range(frames):
        if k:
            m = step(m, s, turn)
        img = render(world, m)
        if car_static:
            box = box_after(box0, m)
        else:
            img = img.copy()
            img[y0:y0 + 40, x0:x0 + 60] = car
            box = box0
        out = pm.update(0.2 * k, img, [Detection(POLICE, 0.9, box, model="vehicle")], ego)
        states.append(out[0].label)
    return states


def test_road_pair_rate_recovers_uniform_expansion():
    rng = np.random.default_rng(0)
    a = np.c_[rng.uniform(50, 590, 300), rng.uniform(200, 340, 300)]
    b = (a - FOE) * 1.05 + FOE + [3.0, -1.0]          # expansion plus a rotation shift
    A, B = road_pair_rate(a, b, H, 0.2)
    assert abs(A) < 1e-6 and abs(B - np.log(1.05) / 0.2) < 1e-6


def test_uniform_zoom_is_not_a_road():
    assert drive(True, cfg=MotionConfig())[-1] == PENDING


def test_pending_until_enough_measurements():
    states = drive(True, frames=3)
    assert states == [PENDING] * 3


def test_static_car_passes_while_we_drive():
    assert drive(True)[-1] == POLICE


def test_car_keeping_pace_is_moving():
    assert drive(False)[-1] == MOVING


def test_rotation_is_taken_off():
    # turning shifts the whole image by 6 px a frame; the static car must still pass
    assert drive(True, turn=(6.0, 0.0))[-1] == POLICE
    assert drive(False, turn=(6.0, 0.0))[-1] == MOVING


def test_while_we_stand_a_still_box_passes_and_a_crossing_one_does_not():
    img = texture(3)
    for dx, want in ((0.0, POLICE), (0.03, MOVING)):
        pm = PoliceMotion()
        for k in range(12):
            box = (0.3 + dx * k, 0.6, 0.15, 0.12)
            out = pm.update(0.2 * k, img, [Detection(POLICE, 0.9, box, model="vehicle")], "stationary")
        assert out[0].label == want


def test_other_labels_pass_untouched():
    pm = PoliceMotion()
    dets = [Detection("car", 0.8, (0.5, 0.6, 0.1, 0.1)), Detection(POLICE, 0.2, (0.2, 0.6, 0.1, 0.1))]
    out = pm.update(0.0, texture(4), dets, "moving")
    assert [d.label for d in out] == ["car", POLICE]      # below track_min_conf: left alone (under the floor)
