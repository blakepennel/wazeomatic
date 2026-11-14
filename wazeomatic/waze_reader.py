"""Read-only view of what the Waze app is showing: alerts ahead, "Still there?" prompts, the report sheet.

The reader never taps, types or scrolls. It looks at the text the app puts on screen:

* on a desktop with a phone or emulator on adb, ``AdbWazeReader`` polls ``uiautomator dump`` (the same
  accessibility tree a screen reader sees) while Waze is in the foreground;
* on the phone, ``WazeWatcherService.kt`` is an AccessibilityService with no gesture or action rights that
  feeds the same parser (``WazeText.kt`` is the Kotlin twin of ``group_cards`` / ``parse_cards``).

What it is used for:

1. **No duplicates.** Before an event goes to the macro, ``WazeAlerts.covering`` says whether Waze already
   shows that report nearby. If it does, the event is not sent; it is counted as a confirmation (our camera
   agrees with what other drivers reported) instead of piling a second report on the same spot.
2. **What is ahead.** ``WazeAlerts.ahead`` is the list of reports Waze announced recently, nearest first, for
   the status line and the logs.

The strings below are the ones Waze shows in English (alert cards, the "Still there?" prompt, the list of
reports on the route). They have been checked against the mock app and against the few real-app phrases
quoted in press reports ("Police reported ahead", "Car on shoulder reported ahead", "Are they still there?" with
"Not there"; tests/test_waze_reader.py), not against a live session of the real app: its wording changes
between versions and languages, so every pattern is data, and an unknown text is kept as an alert of unknown
type rather than dropped (``hazard_type`` None) so a log shows what to add.
"""
from __future__ import annotations

import re
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from .schema import HAZARD_TYPES

WAZE_PACKAGES = ("com.waze", "com.wazeomatic.mockwaze")

# (pattern, hazard key). First match wins, so specific phrases come before the bare category words.
# Patterns are matched case-insensitively against one on-screen text at a time.
ALERT_PATTERNS: list[tuple[str, str]] = [
    (r"police.*other side|other side.*police", "POLICE_OTHER_SIDE"),
    (r"\bpolice\b|speed trap|\bcops?\b", "POLICE"),
    (r"(crash|accident).*other side|other side.*(crash|accident)", "CRASH_OTHER_SIDE"),
    (r"pile-?up|\bcrash\b|\baccident\b|collision", "CRASH"),
    (r"standstill|stopped traffic|traffic (is )?stopped", "TRAFFIC_STANDSTILL"),
    (r"heavy traffic|traffic jam|\bjam\b|slow traffic", "TRAFFIC_HEAVY"),
    (r"pothole", "HAZARD_ON_ROAD_POTHOLE"),
    (r"construction|road ?work|work zone", "HAZARD_ON_ROAD_CONSTRUCTION"),
    (r"(car|vehicle) (stopped )?on (the )?shoulder|stopped (car|vehicle)|car on shoulder", "HAZARD_SHOULDER_CAR_STOPPED"),
    (r"(broken|faulty|dark) traffic light|traffic light (out|not working|broken)", "HAZARD_ON_ROAD_TRAFFIC_LIGHT"),
    (r"object on (the )?road|\bdebris\b|\bobject\b|lost load", "HAZARD_ON_ROAD_OBJECT"),
    (r"\banimals?\b|\bdeer\b|roadkill", "HAZARD_SHOULDER_ANIMALS"),
    (r"\bfog\b|foggy", "WEATHER_FOG"),
    (r"\bflood", "WEATHER_FLOOD"),
    (r"\bic(e|y)\b", "WEATHER_ICE"),
    (r"unplowed|snow on (the )?road|\bsnow\b", "WEATHER_UNPLOWED"),
    (r"slippery", "WEATHER_SLIPPERY"),
    (r"left lane (closed|blocked)|blocked.*left lane", "BLOCKED_LANE_LEFT"),
    (r"right lane (closed|blocked)|blocked.*right lane", "BLOCKED_LANE_RIGHT"),
    (r"(center|centre|middle) lane (closed|blocked)|blocked.*(center|centre|middle) lane", "BLOCKED_LANE_CENTER"),
]
_COMPILED = [(re.compile(p, re.I), k) for p, k in ALERT_PATTERNS]

# Texts that mark the thing on screen as a report somebody else filed (an alert), as opposed to a tile of our
# own report sheet. A bare "Police" on the report sheet is a button, not an alert.
ALERT_CONTEXT = re.compile(r"reported|ahead|still there|\bago\b|\bthanks\b|not there|on your route|"
                           r"\b\d+(\.\d+)?\s?(mi|ft|km|m|yd)\b", re.I)
STILL_THERE = re.compile(r"still there\??|is it still there|not there", re.I)
# The report sheet's own titles. Anchored on the category names: the round report button on the map is
# described as "Report" or "Report button" and must not count as an open sheet.
REPORT_SHEET = re.compile(r"^what do you see\??$|^report (a |an )?(traffic|police|crash|hazard|bad weather|"
                          r"blocked lane|map issue|closure|gas prices)$", re.I)

_DIST = re.compile(r"(\d+(?:[.,]\d+)?)\s?(mi|miles?|ft|feet|yd|km|m|meters?|metres?)\b", re.I)
_TO_M = {"mi": 1609.344, "mile": 1609.344, "miles": 1609.344, "ft": 0.3048, "feet": 0.3048, "yd": 0.9144,
         "km": 1000.0, "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0}


def parse_distance(text: str) -> Optional[float]:
    """'0.3 mi' -> 482.8, '500 ft' -> 152.4, '1,2 km' -> 1200. None if the text carries no distance."""
    m = _DIST.search(text)
    if not m:
        return None
    return float(m.group(1).replace(",", ".")) * _TO_M[m.group(2).lower()]


def classify_text(text: str) -> Optional[str]:
    for rx, key in _COMPILED:
        if rx.search(text):
            return key
    return None


@dataclass
class WazeAlert:
    hazard_type: Optional[str]      # schema key, None when the wording is not one we know
    text: str                       # the texts of the card, joined
    distance_m: Optional[float]
    kind: str = "alert"             # alert | still_there
    ts: float = field(default_factory=time.time)

    @property
    def category(self) -> str:
        return HAZARD_TYPES[self.hazard_type].waze_category if self.hazard_type else ""

    def to_dict(self) -> dict:
        return {"hazard_type": self.hazard_type, "category": self.category, "text": self.text,
                "distance_m": None if self.distance_m is None else round(self.distance_m, 1),
                "kind": self.kind, "ts": round(self.ts, 3)}


@dataclass
class WazeScreen:
    """One look at the screen."""

    package: str = ""
    alerts: list[WazeAlert] = field(default_factory=list)
    report_sheet_open: bool = False
    texts: list[str] = field(default_factory=list)


def group_cards(nodes: list[tuple[str, tuple[int, int, int, int]]], gap_px: int = 40,
                max_height_frac: float = 0.33) -> list[list[str]]:
    """Group on-screen texts into cards by vertical proximity: an alert card is a title, a distance and a
    'reported ... ago' line stacked within a few dozen pixels. ``nodes`` is (text, (x1, y1, x2, y2))."""
    cards: list[list[str]] = []
    last_bottom = None
    # a node as tall as a good part of the screen (the map, a full-screen container with a description) is
    # not a line of a card; letting it in would stretch one card over every text on screen
    screen_h = max((b[3] for _, b in nodes), default=0)
    nodes = [n for n in nodes if screen_h <= 0 or (n[1][3] - n[1][1]) <= max_height_frac * screen_h]
    for text, (_, y1, _, y2) in sorted(nodes, key=lambda n: (n[1][1], n[1][0])):
        if last_bottom is None or y1 - last_bottom > gap_px:
            cards.append([])
        cards[-1].append(text)
        last_bottom = y2 if last_bottom is None else max(last_bottom, y2)
    return cards


def parse_cards(cards: Iterable[list[str]], now: Optional[float] = None) -> list[WazeAlert]:
    """Turn groups of texts into alerts. A group is an alert when it names a report type and carries alert
    context (a distance, 'reported', 'ahead', 'Still there?'), so tiles of the report sheet are not alerts."""
    now = time.time() if now is None else now
    out: list[WazeAlert] = []
    for card in cards:
        joined = " · ".join(t.strip() for t in card if t and t.strip())
        if not joined or not ALERT_CONTEXT.search(joined):
            continue
        key = None
        for t in card:                       # the title line names the type; try it before the joined text
            key = classify_text(t)
            if key:
                break
        key = key or classify_text(joined)
        kind = "still_there" if STILL_THERE.search(joined) else "alert"
        if key is None and kind == "alert" and not re.search(r"reported", joined, re.I):
            continue                         # a distance alone (the speedometer, a turn instruction)
        out.append(WazeAlert(hazard_type=key, text=joined, distance_m=parse_distance(joined), kind=kind, ts=now))
    return out


_BOUNDS = re.compile(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]")


def parse_uiautomator_xml(xml: str, now: Optional[float] = None) -> WazeScreen:
    """Parse ``uiautomator dump`` output. Only nodes of a Waze package count."""
    start = xml.find("<?xml")
    if start < 0:
        start = xml.find("<hierarchy")
    end = xml.rfind("</hierarchy>")
    if start < 0 or end < 0:
        return WazeScreen()
    root = ET.fromstring(xml[start:end + len("</hierarchy>")])
    nodes, package = [], ""
    for n in root.iter("node"):
        pkg = n.get("package", "")
        if pkg not in WAZE_PACKAGES:
            continue
        package = package or pkg
        text = (n.get("text") or "").strip() or (n.get("content-desc") or "").strip()
        m = _BOUNDS.match(n.get("bounds", ""))
        if text and m:
            nodes.append((text, tuple(int(v) for v in m.groups())))
    texts = [t for t, _ in nodes]
    sheet = any(REPORT_SHEET.search(t) for t in texts)
    # while our own report sheet is open its tiles are not alerts; nothing else is read from that screen
    alerts = [] if sheet else parse_cards(group_cards(nodes), now)
    return WazeScreen(package=package, alerts=alerts, report_sheet_open=sheet, texts=texts)


def _same_report(alert_key: str, event_key: str) -> bool:
    """Does an alert of ``alert_key`` already stand for an event of ``event_key``?

    Same key, always. Traffic and Bad weather are one condition per stretch of road whatever the sub-tile, so
    any alert of the category covers. A Police alert covers police on either side (Waze shows one icon).
    """
    if alert_key == event_key:
        return True
    a, e = HAZARD_TYPES[alert_key], HAZARD_TYPES[event_key]
    if a.waze_category != e.waze_category:
        return False
    return a.waze_category in ("Traffic", "Bad weather", "Police", "Crash")


class WazeAlerts:
    """What Waze has shown recently, with expiry. Thread safe; the reader thread writes, the pipeline reads."""

    def __init__(self, ttl_s: float = 120.0, max_distance_m: float = 1200.0,
                 clock: Callable[[], float] = time.time):
        self.ttl_s = ttl_s
        self.max_distance_m = max_distance_m
        self.clock = clock
        self._alerts: list[WazeAlert] = []
        self._lock = threading.Lock()
        self.report_sheet_open = False
        self.last_seen = 0.0             # last time a Waze screen was read at all

    def update(self, screen: WazeScreen) -> list[WazeAlert]:
        """Merge one screen read. Returns the alerts that are new (not seen within the TTL)."""
        now = self.clock()
        new: list[WazeAlert] = []
        with self._lock:
            self.last_seen = now if screen.package else self.last_seen
            self.report_sheet_open = screen.report_sheet_open
            self._expire(now)
            for a in screen.alerts:
                a.ts = now
                same = [b for b in self._alerts if b.hazard_type == a.hazard_type and b.text == a.text]
                if same:
                    same[0].ts, same[0].distance_m = now, a.distance_m
                else:
                    self._alerts.append(a)
                    new.append(a)
        return new

    def _expire(self, now: float) -> None:
        self._alerts = [a for a in self._alerts if now - a.ts <= self.ttl_s]

    def covering(self, hazard_type: str) -> Optional[WazeAlert]:
        """The alert that already reports ``hazard_type`` nearby, if any."""
        now = self.clock()
        with self._lock:
            self._expire(now)
            for a in sorted(self._alerts, key=lambda x: -x.ts):
                if a.hazard_type is None or not _same_report(a.hazard_type, hazard_type):
                    continue
                if a.distance_m is not None and a.distance_m > self.max_distance_m:
                    continue
                return a
        return None

    def ahead(self) -> list[WazeAlert]:
        now = self.clock()
        with self._lock:
            self._expire(now)
            return sorted(self._alerts, key=lambda a: (a.distance_m is None, a.distance_m or 0.0))

    def summary(self, limit: int = 3) -> str:
        items = self.ahead()[:limit]
        if not items:
            return ""
        def one(a: WazeAlert) -> str:
            name = HAZARD_TYPES[a.hazard_type].waze_category if a.hazard_type else "?"
            sub = HAZARD_TYPES[a.hazard_type].waze_subcategory if a.hazard_type else ""
            d = "" if a.distance_m is None else f" {a.distance_m:,.0f} m"
            return f"{name}{(' ' + sub) if sub else ''}{d}"
        return "Waze ahead: " + ", ".join(one(a) for a in items)


class AdbWazeReader:
    """Polls the screen of an adb device while Waze (or the mock) is in front. Read only: ``uiautomator dump``
    and ``dumpsys window`` observe, nothing is injected. Runs in a daemon thread."""

    def __init__(self, alerts: WazeAlerts, serial: Optional[str] = None, adb: str = "adb",
                 interval_s: float = 2.0, on_new: Optional[Callable[[WazeAlert], None]] = None):
        self.alerts = alerts
        self.adb = [adb] + (["-s", serial] if serial else [])
        self.interval_s = interval_s
        self.on_new = on_new
        self.reads = 0
        self.errors = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self, *args: str, timeout: float = 10.0) -> str:
        return subprocess.run(self.adb + list(args), capture_output=True, text=True, timeout=timeout,
                              encoding="utf-8", errors="replace").stdout

    def foreground_package(self) -> str:
        out = self._run("shell", "dumpsys", "window", "displays")
        m = re.search(r"mCurrentFocus=Window\{[^ ]+ [^ ]+ ([\w.]+)/", out) or \
            re.search(r"mFocusedApp=.*? ([\w.]+)/", out)
        return m.group(1) if m else ""

    def read_once(self) -> Optional[WazeScreen]:
        if self.foreground_package() not in WAZE_PACKAGES:
            return None
        xml = self._run("exec-out", "uiautomator", "dump", "/dev/tty", timeout=20.0)
        screen = parse_uiautomator_xml(xml)
        self.reads += 1
        for a in self.alerts.update(screen):
            if self.on_new:
                self.on_new(a)
        return screen

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.read_once()
            except Exception:  # noqa: BLE001 - a missed read is not fatal; the counter shows it
                self.errors += 1
            self._stop.wait(self.interval_s)

    def start(self) -> "AdbWazeReader":
        self._thread = threading.Thread(target=self._loop, name="waze-reader", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
