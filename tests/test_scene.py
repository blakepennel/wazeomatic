import numpy as np

from wazeomatic.scene import EgoMotion, EmergencyLights, SceneAnalyzer, SceneConfig, _alternations, lamp_fractions
from wazeomatic.schema import DEFAULT_CLASS_TO_HAZARD, HAZARD_TYPES, Detection

CAR = (0.5, 0.5, 0.3, 0.3)


def frame_with_bar(colour_bgr=None, h=360, w=640):
    img = np.full((h, w, 3), 90, np.uint8)
    if colour_bgr is not None:      # a light bar across the top of the car box
        y1, x1 = int((0.5 - 0.15) * h), int((0.5 - 0.15) * w)
        img[y1:y1 + 12, x1:x1 + int(0.3 * w)] = colour_bgr
    return img


def test_every_scene_label_maps_to_a_hazard():
    for label, key in DEFAULT_CLASS_TO_HAZARD.items():
        assert key in HAZARD_TYPES, label


def test_lamp_fractions_colours():
    f = lamp_fractions(frame_with_bar((255, 40, 0)), CAR, top_frac=0.6)
    assert f["blue"] > 0.02 and f["red"] == 0
    f = lamp_fractions(frame_with_bar((0, 0, 255)), CAR, top_frac=0.6)
    assert f["red"] > 0.02 and f["blue"] == 0


def test_alternations():
    assert _alternations([0, 1, 0, 1, 0, 1], 0.5) == 5
    assert _alternations([1, 1, 1, 1, 1], 0.1) == 0
    assert _alternations([0, 0, 1, 1, 1, 1], 0.5) == 1     # brake lights coming on once


def test_light_bar_flashing_is_detected_and_brake_lights_are_not():
    car = [Detection("car", 0.9, CAR)]
    lights = EmergencyLights()
    hits = []
    for i in range(20):        # red / blue alternating at 10 fps
        img = frame_with_bar((255, 40, 0) if i % 2 else (0, 0, 255))
        hits += lights.update(i * 0.1, img, car)
    assert hits and hits[-1].label == "emergency_lights"
    steady = EmergencyLights()
    hits = []
    for i in range(20):
        hits += steady.update(i * 0.1, frame_with_bar((0, 0, 255)), car)
    assert hits == []


def texture(shift, h=180, w=320):
    rng = np.random.default_rng(0)
    base = (rng.random((h, w * 2)) * 255).astype(np.uint8)
    base = np.repeat(np.repeat(base[::4, ::4], 4, 0), 4, 1)[:h, : w * 2]
    g = base[:, shift:shift + w]
    return np.dstack([g, g, g])


def test_ego_motion_stationary_vs_moving():
    still, moving = EgoMotion(), EgoMotion()
    for i in range(10):
        still.update(i * 0.1, texture(0), [])
        moving.update(i * 0.1, texture(i * 6), [])
    assert still.state(1.0) == "stationary"
    assert moving.state(1.0) != "stationary"
    gps = EgoMotion()
    gps.set_gps_speed(20.0, 0.0)
    assert gps.state(1.0) == "moving"


def run_scene(cfg, frames, dets_fn, weather_fn=None, dt=0.5, image=None):
    sa = SceneAnalyzer(cfg)
    labels = []
    img = texture(0) if image is None else image
    for i in range(frames):
        t = i * dt
        out = sa.update(t, img, dets_fn(t), weather_fn(t) if weather_fn else None)
        labels.append({d.label for d in out})
    return labels


def queue_ahead(t):
    return [Detection("car", 0.9, (0.5, 0.55, 0.25, 0.2)), Detection("car", 0.8, (0.2, 0.55, 0.1, 0.1)),
            Detection("car", 0.8, (0.8, 0.55, 0.1, 0.1))]


def run_speeds(speed_fn, seconds, dt=0.5):
    """Traffic rules on a scripted speed (the GPS hook sets the ego state); queue_ahead in every frame."""
    sa = SceneAnalyzer(SceneConfig(enabled=("traffic",)))
    labels = []
    for i in range(int(seconds / dt)):
        t = i * dt
        sa.ego.set_gps_speed(speed_fn(i, t), t)
        labels.append({d.label for d in sa.update(t, texture(0), queue_ahead(t))})
    return labels


def test_standstill_survives_a_few_slow_readings():
    # a real stop reads stationary on 91-97 % of frames; one frame in twelve read slow must not hide it
    labels = run_speeds(lambda i, t: 2.0 if i % 12 == 0 else 0.0, 85)
    assert "scene_traffic_standstill" in labels[-1]
    creeping = run_speeds(lambda i, t: 2.0 if i % 4 == 0 else 0.0, 85)      # a quarter of the time rolling
    assert not any("scene_traffic_standstill" in s for s in creeping)


def test_free_driving_that_reads_slow_is_not_heavy_traffic():
    # what the camera-only estimate makes of ordinary city driving: mostly "slow", the odd stationary flicker
    pattern = ["slow"] * 14 + ["moving"] * 5 + ["stationary"]
    speed = {"slow": 3.0, "moving": 10.0, "stationary": 0.0}
    labels = run_speeds(lambda i, t: speed[pattern[i % len(pattern)]], 170)
    assert not any("scene_traffic_heavy" in s for s in labels)


def test_stop_and_go_is_heavy_traffic():
    labels = run_speeds(lambda i, t: 0.0 if t % 40 < 20 else 3.0, 170)        # 20 s stopped, 20 s crawling
    assert "scene_traffic_heavy" in labels[-1]
    assert not any("scene_traffic_standstill" in s for s in labels)


def test_standstill_after_long_stop():
    labels = run_scene(SceneConfig(enabled=("traffic",)), 170, queue_ahead)
    assert "scene_traffic_standstill" not in labels[100]          # 50 s: could be a red light
    assert "scene_traffic_standstill" in labels[-1]               # 85 s stopped behind a car


def test_red_light_vetoes_standstill():
    def with_red(t):
        img_dets = queue_ahead(t)
        return img_dets + [Detection("traffic light", 0.9, (0.5, 0.2, 0.04, 0.14))]
    sa = SceneAnalyzer(SceneConfig(enabled=("traffic",)))
    img = texture(0).copy()
    h, w = img.shape[:2]
    img[int(0.14 * h):int(0.18 * h), int(0.485 * w):int(0.515 * w)] = (0, 0, 255)    # lit red lamp
    last = set()
    for i in range(170):
        last = {d.label for d in sa.update(i * 0.5, img, with_red(i * 0.5))}
    assert "scene_traffic_standstill" not in last


def test_dark_signals_need_two_heads_and_no_lit_one():
    two_dark = lambda t: [Detection("traffic light", 0.9, (0.4, 0.2, 0.04, 0.14)),
                          Detection("traffic light", 0.9, (0.6, 0.2, 0.04, 0.14))]
    grey = np.full((180, 320, 3), 60, np.uint8)            # unlit housings
    labels = run_scene(SceneConfig(enabled=("signals",)), 12, two_dark, image=grey)
    assert "scene_dark_signal" in labels[-1] and "scene_dark_signal" not in labels[3]
    one_dark = lambda t: two_dark(t)[:1]
    assert all("scene_dark_signal" not in s
               for s in run_scene(SceneConfig(enabled=("signals",)), 12, one_dark, image=grey))
    lit = grey.copy()
    lit[int(0.14 * 180):int(0.18 * 180), int(0.585 * 320):int(0.615 * 320)] = (0, 200, 0)   # one green lamp
    assert all("scene_dark_signal" not in s
               for s in run_scene(SceneConfig(enabled=("signals",)), 12, two_dark, image=lit))


def test_blocked_lane_side():
    def cones(xs):
        return lambda t: [Detection("cone", 0.8, (x, 0.7, 0.03, 0.06)) for x in xs]
    right = run_scene(SceneConfig(enabled=("lanes",)), 4, cones([0.55, 0.62, 0.7, 0.78]))
    assert "scene_lane_right" in right[-1]
    left = run_scene(SceneConfig(enabled=("lanes",)), 4, cones([0.22, 0.3, 0.38, 0.46]))
    assert "scene_lane_left" in left[-1]
    shoulder = run_scene(SceneConfig(enabled=("lanes",)), 4, cones([0.85, 0.9, 0.95]))
    assert all(not any(lab.startswith("scene_lane") for lab in s) for s in shoulder)


def test_blocked_lane_ignores_distant_curbside_cones():
    # drums along a parking lane 50 m ahead sit near the vanishing point: x looks central, but on the ground
    # they are three lane widths to the right
    far = lambda t: [Detection("cone", 0.8, (x, 0.52, 0.01, 0.03)) for x in (0.58, 0.6, 0.62, 0.64)]
    labels = run_scene(SceneConfig(enabled=("lanes",)), 8, far)
    assert all(not any(lab.startswith("scene_lane") for lab in s) for s in labels)


def test_blocked_lane_needs_one_side_held():
    # the cone line swings from one side to the other every second (turning through a junction)
    flip = lambda t: [Detection("cone", 0.8, (x if int(t) % 2 else 1 - x, 0.7, 0.03, 0.06))
                      for x in (0.55, 0.62, 0.7)]
    labels = run_scene(SceneConfig(enabled=("lanes",)), 24, flip, dt=0.25)
    assert all(not any(lab.startswith("scene_lane") for lab in s) for s in labels)
    one = lambda t: [Detection("cone", 0.8, (x, 0.7, 0.03, 0.06)) for x in (0.55, 0.62, 0.7)]
    labels = run_scene(SceneConfig(enabled=("lanes",)), 4, one)
    assert not any(lab.startswith("scene_lane") for lab in labels[2])     # 1.0 s: not yet
    assert "scene_lane_right" in labels[3]                                # 1.5 s held


def test_weather_needs_hold_time():
    fog = lambda t: {"clear": 0.05, "fog": 0.95}
    labels = run_scene(SceneConfig(enabled=("weather",)), 60, lambda t: [], fog)
    assert "scene_fog" not in labels[20] and "scene_fog" in labels[-1]
    brief = lambda t: {"fog": 0.95 if t < 5 else 0.0, "clear": 0.05 if t < 5 else 1.0}
    assert all("scene_fog" not in s for s in run_scene(SceneConfig(enabled=("weather",)), 60, lambda t: [], brief))


def test_crash_needs_responders_on_scene():
    def responders(t):
        grow = 1 + t * 0.2
        return [Detection("ambulance", 0.9, (0.6, 0.5, 0.1 * grow, 0.1 * grow), model="vehicle"),
                Detection("police_vehicle", 0.9, (0.7, 0.5, 0.08 * grow, 0.08 * grow), model="vehicle")]
    sa = SceneAnalyzer(SceneConfig(enabled=("crash",)))
    sa.lights.update = lambda t, img, dets: [Detection("emergency_lights", 0.8, d.bbox, model="lights")
                                             for d in dets if d.label != "car"]
    out = set()
    for i in range(12):
        out = {d.label for d in sa.update(i * 0.5, texture(i * 6), responders(i * 0.5))}
    assert "scene_crash" in out
    # police alone at the roadside is a traffic stop: Police, not Crash
    sb = SceneAnalyzer(SceneConfig(enabled=("crash",)))
    sb.lights.update = sa.lights.update
    out = set()
    for i in range(12):
        out = {d.label for d in sb.update(i * 0.5, texture(i * 6), responders(i * 0.5)[1:])}
    assert "scene_crash" not in out


def test_crash_not_for_responders_passing_a_waiting_car():
    lit = lambda t, img, dets: [Detection("emergency_lights", 0.8, d.bbox, model="lights") for d in dets]

    def run(boxes):
        sa = SceneAnalyzer(SceneConfig(enabled=("crash",)))
        sa.lights.update = lit
        fired = False
        for i in range(16):
            t = i * 0.5
            fired |= any(d.label == "scene_crash" for d in sa.update(t, texture(0), boxes(t)))
        return fired
    # we wait at a light; an ambulance and a police car drive across the frame
    passing = lambda t: [Detection("ambulance", 0.9, (0.1 + 0.1 * t, 0.5, 0.12, 0.12), model="vehicle"),
                         Detection("police_vehicle", 0.9, (0.05 + 0.1 * t, 0.5, 0.1, 0.1), model="vehicle")]
    assert not run(passing)
    # the same two parked at the scene ahead of us
    parked = lambda t: [Detection("ambulance", 0.9, (0.6 + 0.002 * np.sin(t), 0.5, 0.12, 0.12), model="vehicle"),
                        Detection("police_vehicle", 0.9, (0.75, 0.5, 0.1, 0.1), model="vehicle")]
    assert run(parked)


def test_crash_needs_lights_on_the_responders_and_ends_with_them():
    amb = Detection("ambulance", 0.9, (0.6, 0.5, 0.12, 0.12), model="vehicle")
    pol = Detection("police_vehicle", 0.9, (0.75, 0.5, 0.1, 0.1), model="vehicle")
    car = Detection("car", 0.9, (0.3, 0.5, 0.1, 0.1), model="coco")

    def run(lit_label, present=lambda t: True):
        sa = SceneAnalyzer(SceneConfig(enabled=("crash",)))
        sa.lights.update = lambda t, img, dets: [Detection("emergency_lights", 0.8, d.bbox, model="lights")
                                                 for d in dets if d.label == lit_label]
        fired = []
        for i in range(24):
            t = i * 0.5
            ds = [amb, pol, car] if present(t) else [car]
            fired.append(any(d.label == "scene_crash" for d in sa.update(t, texture(0), ds)))
        return fired
    # parked responders, but the flashing light is on a work truck next to them
    assert not any(run("car"))
    # one truck that the classifier calls police in one frame and fire engine in the next is one responder
    sb = SceneAnalyzer(SceneConfig(enabled=("crash",)))
    sb.lights.update = lambda t, img, dets: [Detection("emergency_lights", 0.8, d.bbox, model="lights")
                                             for d in dets]
    box = (0.5, 0.5, 0.2, 0.2)
    for i in range(16):
        guess = [Detection("police_vehicle", 0.55 if i % 2 else 0.4, box, model="vehicle"),
                 Detection("fire_truck", 0.4 if i % 2 else 0.55, box, model="vehicle")]
        sb.update(i * 0.5, texture(0), guess)
    stamps = [e[0] for e in sb._emergency]
    assert len(stamps) == len(set(stamps))       # one entry per frame for the one box, of its best class
    # lights on the responders: a crash, which stops once they are out of view
    fired = run("ambulance", present=lambda t: t < 7.0)
    assert any(fired[:14]) and not any(fired[16:])


def test_slippery_needs_rain_and_wet_road():
    both = lambda t: {"rain": 0.9, "clear": 0.1, "surface_wet": 0.7, "surface_dry": 0.3}
    rain_only = lambda t: {"rain": 0.9, "clear": 0.1, "surface_dry": 0.95, "surface_wet": 0.05}
    wet_only = lambda t: {"clear": 0.95, "rain": 0.05, "surface_wet": 0.9}
    cfg = SceneConfig(enabled=("weather",))
    assert "scene_slippery" in run_scene(cfg, 140, lambda t: [], both)[-1]
    for fn in (rain_only, wet_only):
        assert all("scene_slippery" not in s for s in run_scene(cfg, 140, lambda t: [], fn))
    # no road-surface model loaded: rain alone, held, is Slippery road; the surface-only rules stay silent
    no_surface = lambda t: {"rain": 0.9, "clear": 0.1}
    labels = run_scene(cfg, 140, lambda t: [], no_surface)
    assert "scene_slippery" in labels[-1] and all("scene_flood" not in s for s in labels)
    assert all("scene_slippery" not in s for s in run_scene(cfg, 140, lambda t: [], lambda t: {"clear": 0.95, "rain": 0.05}))


def test_ground_point_inverts_the_camera_model():
    import math
    sa = SceneAnalyzer(SceneConfig(cam_height_m=1.3, horizon_y=0.5, hfov_deg=70.0))
    aspect = 16 / 9
    fy = 0.5 / math.tan(math.radians(35.0)) * aspect
    dist, lat = 10.0, 1.8                                  # 10 m ahead, half a lane to the right
    gy = 0.5 + fy * 1.3 / dist
    cx = 0.5 + lat * (gy - 0.5) / (1.3 * aspect)
    d, y = sa.ground_point((cx, gy - 0.02, 0.02, 0.04), aspect)
    assert abs(d - dist) < 1e-6 and abs(y - lat) < 1e-6
    assert abs(sa.lane_offset((cx, gy - 0.02, 0.02, 0.04), aspect) - 1.0) < 1e-6
    assert sa.lane_offset((0.5, 0.49, 0.02, 0.02), aspect) is None          # above the horizon
