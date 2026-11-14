from wazeomatic.report_manager import DEFAULT_LABEL_FLOORS, ReportConfig, ReportManager
from wazeomatic.schema import Detection


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


def det(label="pothole", conf=0.9, bbox=(0.5, 0.6, 0.2, 0.1)):
    return Detection(label=label, confidence=conf, bbox=bbox)


def make(**kw):
    clock = FakeClock()
    cfg = ReportConfig(**kw)
    return ReportManager(cfg, source="sim", clock=clock), clock


def test_single_frame_does_not_report():
    rm, _ = make(window=10, min_hits=3)
    assert rm.update([det()]) == []
    assert rm.update([]) == []


def test_persistence_then_report_once():
    rm, _ = make(window=10, min_hits=3)
    assert rm.update([det(conf=0.6)]) == []     # above the 0.45 pothole floor, one hit
    assert rm.update([det(conf=0.7)]) == []
    events = rm.update([det(conf=0.9)])
    assert len(events) == 1
    ev = events[0]
    assert ev.hazard_type == "HAZARD_ON_ROAD_POTHOLE"
    assert ev.confidence == 0.9           # best hit in the window
    assert ev.waze.waze_category == "Hazard"
    # continued detections inside the cooldown stay silent
    for _ in range(20):
        assert rm.update([det()]) == []


def test_low_confidence_hits_do_not_count():
    rm, _ = make(window=10, min_hits=3, min_confidence=0.6)
    for _ in range(10):
        assert rm.update([det(conf=0.55)]) == []


def test_window_expires_old_hits():
    rm, _ = make(window=5, min_hits=3)
    rm.update([det()])
    rm.update([det()])
    for _ in range(5):
        rm.update([])
    assert rm.update([det()]) == []  # old two hits fell out of the window


def test_cooldown_per_type_and_global():
    rm, clock = make(window=10, min_hits=1, cooldown_s=60, global_cooldown_s=5)
    assert len(rm.update([det("pothole")])) == 1
    # different type blocked by the global cooldown
    assert rm.update([det("cone")]) == []
    clock.advance(6)
    assert len(rm.update([det("cone")])) == 1
    clock.advance(6)
    assert rm.update([det("pothole")]) == []   # still in per-type cooldown
    clock.advance(60)
    assert len(rm.update([det("pothole")])) == 1


def test_default_global_cooldown_matches_the_macro_and_counts_what_it_holds_back():
    from wazeomatic.report_manager import ReportConfig
    assert ReportConfig().global_cooldown_s == 60.0     # the macro: one report a minute in total
    rm, clock = make(window=10, min_hits=1)
    assert len(rm.update([det("pothole")])) == 1
    for _ in range(5):                                  # a cone qualifies on five frames in the window
        assert rm.update([det("cone")]) == []
    assert rm.held_back == 1                            # counted once, not once per frame
    clock.advance(61)
    assert len(rm.update([det("cone")])) == 1
    assert rm.held_back == 1


def test_types_take_turns_for_the_global_cooldown():
    # a work zone that qualifies every minute must not hold back a pothole (or a police car) for good
    rm, clock = make(window=10, min_hits=1)
    assert [e.hazard_type for e in rm.update([det("cone")])] == ["HAZARD_ON_ROAD_CONSTRUCTION"]
    for _ in range(3):
        assert rm.update([det("cone"), det("pothole")]) == []
    clock.advance(61)
    assert [e.hazard_type for e in rm.update([det("cone"), det("pothole")])] == ["HAZARD_ON_ROAD_POTHOLE"]
    clock.advance(61)
    assert [e.hazard_type for e in rm.update([det("cone"), det("pothole")])] == ["HAZARD_ON_ROAD_CONSTRUCTION"]


def test_unmapped_labels_ignored():
    rm, _ = make(window=10, min_hits=1)
    assert rm.update([det("toothbrush"), det("car")]) == []


def test_stopped_vehicle_heuristic():
    rm, clock = make(window=10, min_hits=3, global_cooldown_s=0, stopped_vehicle_enabled=True)
    # car in the right shoulder region growing as we approach it
    assert rm.update([det("car", 0.9, (0.8, 0.6, 0.15, 0.15))]) == []
    assert rm.update([det("car", 0.9, (0.82, 0.62, 0.18, 0.18))]) == []
    events = rm.update([det("car", 0.9, (0.85, 0.65, 0.22, 0.22))])
    assert [e.hazard_type for e in events] == ["HAZARD_SHOULDER_CAR_STOPPED"]
    # a car ahead in our lane (centre of frame) is never a shoulder hazard
    rm2, _ = make(window=10, min_hits=3, global_cooldown_s=0, stopped_vehicle_enabled=True)
    for _ in range(5):
        assert rm2.update([det("car", 0.9, (0.5, 0.6, 0.2, 0.2))]) == []


def test_event_json_and_intent_extras():
    rm, _ = make(window=10, min_hits=1)
    ev = rm.update([det("cone", 0.77)])[0]
    d = ev.to_dict()
    assert d["hazard_type"] == "HAZARD_ON_ROAD_CONSTRUCTION"
    assert d["waze_subcategory"] == "Construction"
    assert d["schema"] == 1
    extras = ev.intent_extras()
    assert extras["hazard_type"] == "HAZARD_ON_ROAD_CONSTRUCTION"
    assert extras["confidence"] == "0.770"
    assert all(isinstance(v, str) for v in extras.values())


def test_hood_zone_ignored():
    rm, _ = make(window=10, min_hits=1)
    # bottom edge at 0.93 -> inside the lowest 10 % of the frame: the bonnet reflection
    assert rm.update([det("pothole", 0.9, (0.49, 0.88, 0.15, 0.10))]) == []
    # same box a little higher is a real road detection
    assert len(rm.update([det("pothole", 0.9, (0.49, 0.80, 0.15, 0.10))])) == 1
    rm2, _ = make(window=10, min_hits=1, hood_frac=0.0)
    assert len(rm2.update([det("pothole", 0.9, (0.49, 0.88, 0.15, 0.10))])) == 1


def test_hood_zone_keeps_close_vehicles():
    rm, _ = make(window=10, min_hits=1, min_hits_by_hazard={})
    # a police car beside us reaches the bottom edge of the frame: still a police car
    close = Detection("police_vehicle", 0.95, (0.7, 0.78, 0.4, 0.43), model="vehicle")
    assert len(rm.update([close])) == 1
    # a vehicle box wholly inside the hood zone is our own bonnet
    rm2, _ = make(window=10, min_hits=1)
    assert rm2.update([Detection("police_vehicle", 0.95, (0.5, 0.96, 0.4, 0.05), model="vehicle")]) == []


def test_barricade_on_a_van_is_livery():
    rm, _ = make(window=10, min_hits=1)
    van = Detection("truck", 0.6, (0.26, 0.52, 0.19, 0.26), model="coco")
    chevrons = Detection("barricade", 0.6, (0.215, 0.53, 0.10, 0.17), model="hazard")
    assert rm.update([van, chevrons]) == []
    # the same barricade on its own, or only partly behind a car, is on the road
    assert len(rm.update([chevrons])) == 1
    rm2, _ = make(window=10, min_hits=1)
    car = Detection("car", 0.6, (0.40, 0.53, 0.19, 0.20), model="coco")
    assert len(rm2.update([car, chevrons])) == 1


def test_default_pothole_floor_is_045():
    from wazeomatic.report_manager import ReportConfig, ReportManager
    from wazeomatic.schema import Detection

    assert DEFAULT_LABEL_FLOORS == {"pothole": 0.45, "police_vehicle": 0.85, "worker": 0.6, "debris": 0.5}
    cfg = ReportConfig(global_cooldown_s=0)
    assert cfg.min_confidence_by_label == {"pothole": 0.45, "police_vehicle": 0.85, "worker": 0.6, "debris": 0.5}
    rm = ReportManager(cfg, clock=lambda: 1000.0)
    weak = Detection("pothole", 0.42, (0.5, 0.5, 0.1, 0.1))
    for i in range(3):
        assert rm.update([weak], frame_index=i) == []
    strong = Detection("pothole", 0.46, (0.5, 0.5, 0.1, 0.1))
    events = []
    for i in range(3, 6):
        events += rm.update([strong], frame_index=i)
    assert len(events) == 1
    # the default dict is copied per config, not shared
    cfg.min_confidence_by_label["cone"] = 0.9
    assert "cone" not in ReportConfig().min_confidence_by_label


def test_dog_on_a_walk_is_not_an_animal_on_the_road():
    # Argoverse 3f9796e9: a man walking a dog on the pavement, person and dog at the same ground line
    person = Detection("person", 0.9, (0.88, 0.645, 0.08, 0.25))
    dog = Detection("dog", 0.75, (0.976, 0.733, 0.048, 0.07))
    rm, _ = make(window=10)
    assert all(rm.update([dog, person]) == [] for _ in range(5))
    # the same dog alone, or a person far off on another ground line, is reported
    rm, _ = make(window=10)
    assert [e.hazard_type for _ in range(3) for e in rm.update([dog])] == ["HAZARD_SHOULDER_ANIMALS"]
    far = Detection("person", 0.9, (0.3, 0.45, 0.03, 0.09))
    rm, _ = make(window=10)
    assert [e.hazard_type for _ in range(3) for e in rm.update([dog, far])] == ["HAZARD_SHOULDER_ANIMALS"]


def test_debris_must_lie_on_the_road():
    # Argoverse 2583a8ee: a vent on a building front; 5bc5e7b0: the trouser legs of a man crossing
    vent = Detection("debris", 0.8, (0.93, 0.21, 0.1, 0.09))
    legs = Detection("debris", 0.8, (0.48, 0.82, 0.05, 0.05))
    walker = Detection("person", 0.9, (0.48, 0.75, 0.06, 0.2))
    rm, _ = make(window=10)
    assert all(rm.update([vent, legs, walker]) == [] for _ in range(5))
    # the same box on the road with nobody standing in it is reported
    rm, _ = make(window=10)
    assert [e.hazard_type for _ in range(3) for e in rm.update([legs])] == ["HAZARD_ON_ROAD_OBJECT"]


def test_police_needs_two_hits_by_default():
    rm, _ = make(window=10)
    cop = Detection("police_vehicle", 0.9, (0.5, 0.5, 0.2, 0.2), model="vehicle")
    cone = Detection("cone", 0.9, (0.5, 0.6, 0.05, 0.05))
    assert rm.update([cop, cone]) == []
    out = rm.update([cop, cone])
    assert [e.hazard_type for e in out] == ["POLICE"]          # the cone still needs its third hit
