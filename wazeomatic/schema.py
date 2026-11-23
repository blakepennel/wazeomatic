"""Hazard taxonomy shared by the desktop pipeline, the Android app and the macro.

The ``hazard_type`` strings are the contract with the MacroDroid macro: the macro
switches on this value to decide which Waze menu path to tap. Keep them stable.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

SCHEMA_VERSION = 1

# Broadcast action the Android app sends and MacroDroid listens for.
INTENT_ACTION = "com.wazeomatic.HAZARD_DETECTED"


@dataclass(frozen=True)
class HazardType:
    """One reportable thing, mapped onto Waze's report menu."""

    key: str              # value of the ``hazard_type`` variable the macro reads
    waze_category: str    # top-level Waze report tile
    waze_subcategory: str # second-level choice inside that tile ("" if none)
    description: str


# Waze report menu (Android, 2024 redesign; see docs/pipeline-to-macro.md). Under "Hazard"
# Waze offers a flat list of subtypes: Construction, Car on shoulder, Broken traffic light,
# Pothole, Object. Animals/roadkill subtypes were removed, so animals report as a plain
# Hazard with no subtype. Blocked lane takes Left / Center / Right lane
# (support.google.com/waze/answer/13811250); Traffic, Police, Crash and Bad weather sub-tiles
# follow the 2024 redesign coverage (docs/waze-coverage.md). Keys are kept stable because the
# macro switches on them; new keys only ever get added.
HAZARD_TYPES: dict[str, HazardType] = {
    h.key: h
    for h in [
        HazardType("HAZARD_ON_ROAD_POTHOLE", "Hazard", "Pothole", "Pothole / road damage"),
        HazardType("HAZARD_ON_ROAD_CONSTRUCTION", "Hazard", "Construction", "Cones, work-zone signs, delineators"),
        HazardType("HAZARD_ON_ROAD_OBJECT", "Hazard", "Object", "Debris or object on the road"),
        HazardType("HAZARD_ON_ROAD_TRAFFIC_LIGHT", "Hazard", "Broken traffic light",
                   "Signal head in view with no lamp lit for several seconds"),
        HazardType("HAZARD_SHOULDER_CAR_STOPPED", "Hazard", "Car on shoulder", "Vehicle stopped on the shoulder"),
        HazardType("HAZARD_SHOULDER_ANIMALS", "Hazard", "", "Animal on or near the road (generic Hazard)"),
        HazardType("POLICE", "Police", "", "Police vehicle visible"),
        HazardType("POLICE_OTHER_SIDE", "Police", "Other side", "Police vehicle on the opposite carriageway"),
        HazardType("CRASH", "Crash", "", "Crash / collision: emergency vehicles at stopped traffic"),
        HazardType("CRASH_OTHER_SIDE", "Crash", "Other side", "Crash scene on the opposite carriageway"),
        HazardType("TRAFFIC_HEAVY", "Traffic", "Heavy", "Slow, dense traffic for a sustained stretch"),
        HazardType("TRAFFIC_STANDSTILL", "Traffic", "Standstill", "Traffic not moving, not at a red light"),
        HazardType("WEATHER_FOG", "Bad weather", "Fog", "Fog limiting visibility"),
        HazardType("WEATHER_FLOOD", "Bad weather", "Flood", "Standing water across the road"),
        HazardType("WEATHER_SLIPPERY", "Bad weather", "Slippery road", "Wet road in heavy rain"),
        HazardType("WEATHER_ICE", "Bad weather", "Icy road", "Ice on the road surface"),
        HazardType("WEATHER_UNPLOWED", "Bad weather", "Unplowed road", "Snow covering the lanes"),
        HazardType("BLOCKED_LANE_LEFT", "Blocked lane", "Left lane", "Left lane blocked by a work zone or vehicle"),
        HazardType("BLOCKED_LANE_CENTER", "Blocked lane", "Center lane", "Center lane blocked"),
        HazardType("BLOCKED_LANE_RIGHT", "Blocked lane", "Right lane", "Right lane blocked"),
    ]
}

# Waze report types that are not camera observations and are deliberately not automated:
# Map issue, Closure (needs a duration and reason the camera cannot know), Gas prices, Place,
# Roadside help, Map chat. docs/waze-coverage.md explains each.
NOT_AUTOMATED = ("Map issue", "Closure", "Gas prices", "Place", "Roadside help", "Map chat")

# Model class name -> hazard key. Model classes come from configs/classes.yaml.
DEFAULT_CLASS_TO_HAZARD: dict[str, str] = {
    # custom hazard model
    "pothole": "HAZARD_ON_ROAD_POTHOLE",
    "cone": "HAZARD_ON_ROAD_CONSTRUCTION",
    "construction_sign": "HAZARD_ON_ROAD_CONSTRUCTION",
    "delineator": "HAZARD_ON_ROAD_CONSTRUCTION",
    "barricade": "HAZARD_ON_ROAD_CONSTRUCTION",
    "worker": "HAZARD_ON_ROAD_CONSTRUCTION",
    "police_vehicle": "POLICE",            # the vehicle classifier's label for a police car
    "debris": "HAZARD_ON_ROAD_OBJECT",
    "deer": "HAZARD_SHOULDER_ANIMALS",
    # COCO classes used from the stock detector
    "dog": "HAZARD_SHOULDER_ANIMALS",
    "horse": "HAZARD_SHOULDER_ANIMALS",
    "sheep": "HAZARD_SHOULDER_ANIMALS",
    "cow": "HAZARD_SHOULDER_ANIMALS",
    "bear": "HAZARD_SHOULDER_ANIMALS",
    # COCO has no deer: on Open Images deer the stock model says sheep, cow, horse and, a sixth of the time,
    # giraffe (docs/waze-coverage.md). On a North American road those are deer, elk or moose.
    "giraffe": "HAZARD_SHOULDER_ANIMALS",
    "zebra": "HAZARD_SHOULDER_ANIMALS",
    "elephant": "HAZARD_SHOULDER_ANIMALS",
    # scene-level labels (wazeomatic/scene.py): one per frame while a condition holds
    "scene_traffic_heavy": "TRAFFIC_HEAVY",
    "scene_traffic_standstill": "TRAFFIC_STANDSTILL",
    "scene_fog": "WEATHER_FOG",
    "scene_flood": "WEATHER_FLOOD",
    "scene_slippery": "WEATHER_SLIPPERY",
    "scene_ice": "WEATHER_ICE",
    "scene_unplowed": "WEATHER_UNPLOWED",
    "scene_crash": "CRASH",
    "scene_crash_other_side": "CRASH_OTHER_SIDE",
    "scene_police_other_side": "POLICE_OTHER_SIDE",
    "scene_lane_left": "BLOCKED_LANE_LEFT",
    "scene_lane_center": "BLOCKED_LANE_CENTER",
    "scene_lane_right": "BLOCKED_LANE_RIGHT",
    "scene_dark_signal": "HAZARD_ON_ROAD_TRAFFIC_LIGHT",
    "scene_car_on_shoulder": "HAZARD_SHOULDER_CAR_STOPPED",
    # "car"/"truck"/"bus" are NOT mapped directly: a vehicle becomes Car on shoulder only through
    # shoulder.ShoulderCars (scene_car_on_shoulder: standing, beside the road, on a fast road).
}


@dataclass
class Detection:
    """A single box from one inference pass. bbox is normalized xywh (center)."""

    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    model: str = "hazard"


@dataclass
class HazardEvent:
    """What the pipeline emits once a detection is trusted enough to report."""

    hazard_type: str
    confidence: float
    source: str = "sim"            # "phone" | "pi" | "sim"
    ts: float = field(default_factory=time.time)
    frame_index: Optional[int] = None
    bbox: Optional[tuple[float, float, float, float]] = None
    label: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    schema: int = SCHEMA_VERSION

    @property
    def waze(self) -> HazardType:
        return HAZARD_TYPES[self.hazard_type]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["waze_category"] = self.waze.waze_category
        d["waze_subcategory"] = self.waze.waze_subcategory
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    def intent_extras(self) -> dict[str, str]:
        """String extras for the Android broadcast (MacroDroid reads strings)."""
        return {
            "hazard_type": self.hazard_type,
            "confidence": f"{self.confidence:.3f}",
            "ts": f"{self.ts:.3f}",
            "source": self.source,
            "waze_category": self.waze.waze_category,
            "waze_subcategory": self.waze.waze_subcategory,
        }


def load_class_to_hazard(classes_yaml=None, warn=None) -> dict[str, str]:
    """DEFAULT_CLASS_TO_HAZARD plus the ``hazard_map`` of configs/classes.yaml.

    A hazard key that is not in HAZARD_TYPES is dropped with a warning, the same rule as the phone
    app (Settings.kt): a typo in the YAML must not surface as a KeyError at the first event,
    hours into a run.
    """
    import sys
    from pathlib import Path

    m = dict(DEFAULT_CLASS_TO_HAZARD)
    path = Path(classes_yaml) if classes_yaml else None
    if path is not None and path.exists():
        import yaml

        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for label, key in (cfg.get("hazard_map") or {}).items():
            if key in HAZARD_TYPES:
                m[str(label)] = key
            else:
                msg = f"[warn] {path.name}: unknown hazard key {key!r} for class {label!r}; mapping ignored"
                (warn or (lambda t: print(t, file=sys.stderr)))(msg)
                m.pop(str(label), None)
    return m
