"""End-to-end test on the pixel9pro_test emulator.

Stage A (mock Waze only): start mock Waze with auto macro on, inject HAZARD_DETECTED
broadcasts with adb, pull reports.jsonl / macro_events.jsonl and check
  - a manual rehearsal through `adb shell input tap` (like MacroDroid coordinate taps) works,
  - a real event is reported exactly once with the right category/subcategory,
  - a second event inside 60 s is suppressed,
  - dry_run=true reaches the confirm sheet, cancels and is logged with dry_run=true,
  - the same for hazard types whose menu path has no subcategory tile.

Stage B (needs the detector APK): push a dashcam clip, run the detector in video mode, read
its WazeomaticEvent logcat lines and check mock Waze received them.

Stage C (Waze reader; C1 needs only mock Waze, the rest the debug detector APK): mock Waze shows a
Police alert card; the desktop reader (wazeomatic.waze_reader over `uiautomator dump`) and the
detector's read-only accessibility service must both read it; a police_vehicle detection injected into
the detector while the card shows must become a CONFIRMED line and send nothing, and once the card is
gone the same injection must reach mock Waze as a dry-run event. The accessibility settings are put
back at the end.

Usage (from the repo root):
  .venv\\Scripts\\python.exe tools\\emulator\\run_e2e.py              # Stage A + B + C
  .venv\\Scripts\\python.exe tools\\emulator\\run_e2e.py --stage a
  .venv\\Scripts\\python.exe tools\\emulator\\run_e2e.py --stage b --video-seconds 90
  .venv\\Scripts\\python.exe tools\\emulator\\run_e2e.py --stage c

Every check of the requested stages shows up in the summary. A check that could not run (no
detector APK, no detector events, an earlier step failed) is a SKIP row, and a run with a SKIP
is INCOMPLETE, not PASS. Exit code: 0 = PASS, 1 = FAIL, 2 = INCOMPLETE (0 with --allow-skip).

adb is taken from --adb, else ANDROID_HOME / ANDROID_SDK_ROOT (platform-tools), else PATH, else
sdk.dir in android/*/local.properties.

Everything (logcat, screenshots, pulled jsonl, summary) goes to runs/emulator/<timestamp>/.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ADB = "adb"  # replaced by resolve_adb() in main()
VIDEO_DIR = REPO / "assets" / "videos"
DEFAULT_VIDEOS = ("dashcam_urban_1080p.mp4", "dashcam_urban_1080p.ogv")
MOCK_PKG = "com.wazeomatic.mockwaze"
DET_PKG = "com.wazeomatic.detector"
MOCK_APK = REPO / "android" / "mockwaze" / "app" / "build" / "outputs" / "apk" / "debug" / "app-debug.apk"
DET_APK_GLOB = str(REPO / "android" / "detector" / "app" / "build" / "outputs" / "apk" / "debug" / "*.apk")
REMOTE_DIR = f"/sdcard/Android/data/{MOCK_PKG}/files"
HAZARD_ACTION = "com.wazeomatic.HAZARD_DETECTED"

# Expected menu paths, must match wazeomatic/schema.py (checked at start-up if importable).
EXPECTED = {
    "HAZARD_ON_ROAD_POTHOLE": ("Hazard", "Pothole"),
    "HAZARD_ON_ROAD_CONSTRUCTION": ("Hazard", "Construction"),
    "HAZARD_ON_ROAD_OBJECT": ("Hazard", "Object"),
    "HAZARD_ON_ROAD_TRAFFIC_LIGHT": ("Hazard", "Broken traffic light"),
    "HAZARD_SHOULDER_CAR_STOPPED": ("Hazard", "Car on shoulder"),
    "HAZARD_SHOULDER_ANIMALS": ("Hazard", ""),
    "POLICE": ("Police", ""),
    "POLICE_OTHER_SIDE": ("Police", "Other side"),
    "CRASH": ("Crash", ""),
    "CRASH_OTHER_SIDE": ("Crash", "Other side"),
    "TRAFFIC_HEAVY": ("Traffic", "Heavy"),
    "TRAFFIC_STANDSTILL": ("Traffic", "Standstill"),
    "WEATHER_FOG": ("Bad weather", "Fog"),
    "WEATHER_FLOOD": ("Bad weather", "Flood"),
    "WEATHER_SLIPPERY": ("Bad weather", "Slippery road"),
    "WEATHER_ICE": ("Bad weather", "Icy road"),
    "WEATHER_UNPLOWED": ("Bad weather", "Unplowed road"),
    "BLOCKED_LANE_LEFT": ("Blocked lane", "Left lane"),
    "BLOCKED_LANE_CENTER": ("Blocked lane", "Center lane"),
    "BLOCKED_LANE_RIGHT": ("Blocked lane", "Right lane"),
}

# Stage C: the detector's read-only Waze reader (accessibility service) and the debug injection hook.
WAZE_SERVICE = f"{DET_PKG}/{DET_PKG}.WazeWatcherService"
SHOW_ALERT_ACTION = f"{MOCK_PKG}.SHOW_ALERT"
CLEAR_ALERTS_ACTION = f"{MOCK_PKG}.CLEAR_ALERTS"
INJECT_ACTION = f"{DET_PKG}.INJECT_DETECTION"
WAZE_TAG = "WazeomaticWaze"
# 0.3 mi as the alert card says it; the readers must turn it into about 483 m
C_DISTANCE_TEXT = "0.3 mi"
C_DISTANCE_M = 0.3 * 1609.344
# a video file that does not exist: the detector loads its models and builds the pipeline, but no frame
# arrives, so the only detections are the injected ones
C_NO_CLIP = "e2e_no_clip.mp4"

# Labels of the shipped six-class hazard model. A detector event with one of these labels came
# from the hazard model itself, not from a COCO class remapped with the class_map extra.
HAZARD_MODEL_LABELS = {"pothole", "cone", "construction_sign", "delineator", "barricade", "worker"}

# Macro outcomes that end a run of the auto macro.
MACRO_END = {"sent", "dry_run_cancelled", "failed", "obstructed"}

# Every check of a stage, in run order. All of them appear in the summary of a run that asked
# for the stage: PASS/FAIL when they ran, SKIP when they could not.
CHECKS_A = {
    "A0": "A0 auto macro service running",
    "A1": "A1 manual tap rehearsal logged as dry_run=true",
    "A1_police": "A1 manual Police send logged as a touch report",
    "A2_outcome": "A2 macro finished with outcome=sent",
    "A2_once": "A2 Hazard > Pothole reported exactly once",
    "A3_suppressed": "A3 second event inside 60 s suppressed",
    "A3_once": "A3 still exactly one real report",
    "A4_outcome": "A4 dry run finished with outcome=dry_run_cancelled",
    "A4_logged": "A4 dry run logged dry_run=true, action=cancel, Hazard > Construction",
    "A4_no_send": "A4 no real send from the dry run",
    "A4_police": "A4 POLICE dry run (no subcategory step) cancelled as Police, empty subcategory",
    "A4_animals": "A4 HAZARD_SHOULDER_ANIMALS dry run (no subcategory step) cancelled as Hazard, empty subcategory",
    "A5": "A5 unknown hazard_type ignored",
    "A6": "A6 MockWazeReport lines in logcat",
}
CHECKS_B = {
    "B0": "B0 detector installed",
    "B1": "B1 video pushed",
    "B2": "B2 detector emitted WazeomaticEvent",
    "B3": "B3 mock Waze received the detector broadcasts",
    "B4": "B4 sent or dry-run reports match the hazard_type mapping",
    "B5": "B5 every report traces back to a detector event",
    "B6": "B6 one-per-minute limit held",
}
CHECKS_C = {
    "C0": "C0 mock Waze showed the POLICE alert card (alerts.jsonl)",
    "C1": "C1 desktop reader (uiautomator dump) sees one POLICE alert at about 483 m",
    "C2_enabled": "C2 Waze reader accessibility service connected",
    "C2": "C2 phone reader logged WAZE_ALERT for POLICE",
    "C3_confirmed": "C3 injected police_vehicle logged as CONFIRMED against the Waze alert",
    "C3_no_event": "C3 no POLICE event reached mock Waze while Waze showed the alert",
    "C4": "C4 control: with the alert gone the POLICE event reached mock Waze (dry run)",
    "C5": "C5 accessibility settings restored",
}

# Timings used for the run (ms). macro_wait_ms must be longer than sheet_ms/page_ms.
TIMING = {
    "sheet_ms": 300,
    "page_ms": 200,
    "toast_ms": 1500,
    # Waze sends by itself after a few seconds (mock default 5000). The manual rehearsal in
    # Stage A is slowed down by uiautomator dumps, so give it more room here.
    "auto_send_ms": 10000,
    "macro_start_delay_ms": 500,
    "macro_wait_ms": 1000,
    "dry_run_hold_ms": 1500,
    "rate_limit_ms": 60000,
}


# ---------------------------------------------------------------- pure helpers (unit tested)
def b7_name(hazard_type: str) -> str:
    return f"B7 detector reported {hazard_type} from the hazard model"


def planned_checks(stage: str, expect_hazards: list[str]) -> list[str]:
    """Names of all checks a run of `stage` has to produce. A stage that was not asked for
    contributes nothing (it is not a skip)."""
    plan: list[str] = []
    if stage in ("a", "all"):
        plan += CHECKS_A.values()
    if stage in ("b", "all"):
        plan += CHECKS_B.values()
        plan += [b7_name(t) for t in expect_hazards]
    if stage in ("c", "all"):
        plan += CHECKS_C.values()
    return plan


def missing_rows(results: list[tuple[str, str, str]], checks: dict[str, str], why: str) -> list[tuple[str, str, str]]:
    """SKIP rows for the checks of one stage that have no row yet, so a stage that stops early names its
    own reason instead of the run-wide not_run_reason."""
    have = {r[1] for r in results}
    return [("SKIP", name, why) for name in checks.values() if name not in have]


def fill_skips(results: list[tuple[str, str, str]], plan: list[str], why: str) -> list[tuple[str, str, str]]:
    """Rows in plan order with a SKIP row for every planned check that produced none; rows that
    are not in the plan (a crash, a failed install) go last."""
    by_name = {name: (status, name, detail) for status, name, detail in results}
    rows = [by_name.get(name, ("SKIP", name, why or "not run")) for name in plan]
    return rows + [r for r in results if r[1] not in plan]


def verdict(results: list[tuple[str, str, str]]) -> str:
    """PASS only when nothing failed and nothing was skipped."""
    statuses = {r[0] for r in results}
    if "FAIL" in statuses:
        return "FAIL"
    if "SKIP" in statuses or not results:
        return "INCOMPLETE"
    return "PASS"


def exit_code(v: str, allow_skip: bool = False) -> int:
    if v == "PASS":
        return 0
    if v == "INCOMPLETE":
        return 0 if allow_skip else 2
    return 1


def parse_focus(dumpsys: str) -> str:
    """Title of the focused window from `dumpsys window` (mCurrentFocus), '' when there is none.
    Example lines:
      mCurrentFocus=Window{1f2e3d u0 com.wazeomatic.mockwaze/com.wazeomatic.mockwaze.MainActivity}
      mCurrentFocus=Window{9ab7c1 u0 Application Not Responding: com.android.systemui}
      mCurrentFocus=null
    """
    for m in re.finditer(r"mCurrentFocus=Window\{\S+\s+\S+\s+([^}\r\n]*)\}", dumpsys):
        if m.group(1).strip():
            return m.group(1).strip()
    return ""


def focus_package(title: str) -> str:
    """Package that owns a focused activity window ('pkg/activity'), '' for anything else."""
    m = re.match(r"([\w.]+)/", title)
    return m.group(1) if m else ""


def anr_package(title: str) -> str:
    m = re.search(r"Not Responding: ([\w.]+)", title)
    return m.group(1) if m else ""


def parse_event_lines(lines: list[str]) -> list[dict]:
    """WazeomaticEvent logcat lines -> the JSON objects they carry."""
    events = []
    for line in lines:
        i = line.find("{")
        if i < 0:
            continue
        try:
            obj = json.loads(line[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


def check_expected_hazard(hazard_type: str, events: list[dict], macro: list[dict]) -> tuple[bool, str]:
    """B7: the detector reported `hazard_type` from a hazard-model class (not a remapped COCO
    class) and mock Waze took one of those events through to the end of the macro."""
    of_type = [e for e in events if e.get("hazard_type") == hazard_type]
    real = [e for e in of_type if e.get("label") in HAZARD_MODEL_LABELS]
    labels = sorted({str(e.get("label")) for e in of_type})
    got = [m for m in macro if m.get("hazard_type") == hazard_type]
    outcomes = sorted({str(m.get("outcome")) for m in got})
    done = [m for m in got if m.get("outcome") in ("sent", "dry_run_cancelled")]

    def from_real_event(m: dict) -> bool:
        # the broadcast carries the event ts with three decimals; the mock logs it as event_ts
        try:
            ts = float(m.get("event_ts"))
            return any(abs(ts - float(e.get("ts"))) < 0.002 for e in real)
        except (TypeError, ValueError):
            return False

    traced = [m for m in done if from_real_event(m)]
    detail = (f"{len(of_type)} detector event(s) of this type, labels={labels}, {len(real)} from the hazard "
              f"model; mock outcomes={outcomes}, {len(traced)} completed macro run(s) for a hazard-model event")
    return bool(real) and bool(traced), detail


def import_waze_reader():
    """wazeomatic.waze_reader from the repo (the desktop reader Stage C checks against)."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from wazeomatic import waze_reader  # type: ignore
    return waze_reader


def accessibility_list_with(existing: str, component: str) -> str:
    """Value for `settings put secure enabled_accessibility_services`: what was enabled before (other
    services stay on) plus `component`. `settings get` prints 'null' when the key is unset."""
    items = [s for s in (existing or "").strip().split(":") if s and s != "null"]
    short = component.replace("/" + component.split("/")[0] + ".", "/.")
    if component not in items and short not in items:
        items.append(component)
    return ":".join(items)


def accessibility_restore_commands(services: str, enabled: str) -> list[str]:
    """Shell commands that put both accessibility keys back to what `settings get` printed before."""
    cmds = []
    for key, value in (("enabled_accessibility_services", services), ("accessibility_enabled", enabled)):
        value = (value or "").strip()
        if not value or value == "null":
            cmds.append(f"settings delete secure {key}")
        else:
            cmds.append(f"settings put secure {key} '{value}'")
    return cmds


def parse_tagged_json(lines: list[str], marker: str) -> list[dict]:
    """JSON objects of logcat lines that carry `marker` (e.g. 'WAZE_ALERT', 'CONFIRMED')."""
    return parse_event_lines([l for l in lines if marker in l])


def check_desktop_alerts(alerts: list[dict], hazard_type: str = "POLICE", distance_m: float = C_DISTANCE_M,
                         tol_m: float = 10.0) -> tuple[bool, str]:
    """C1: the screen holds exactly one alert, of `hazard_type`, about `distance_m` away.
    `alerts` are WazeAlert.to_dict() rows."""
    ok = (len(alerts) == 1 and alerts[0].get("hazard_type") == hazard_type
          and alerts[0].get("distance_m") is not None and abs(float(alerts[0]["distance_m"]) - distance_m) <= tol_m)
    return ok, f"{len(alerts)} alert(s): {alerts}"


def check_waze_alert_lines(lines: list[str], hazard_type: str = "POLICE") -> tuple[bool, str]:
    """C2: the detector's reader logged a WAZE_ALERT of `hazard_type` about the distance of the card."""
    rows = parse_tagged_json(lines, "WAZE_ALERT")
    hits = [r for r in rows if r.get("hazard_type") == hazard_type and r.get("distance_m") is not None
            and abs(float(r["distance_m"]) - C_DISTANCE_M) <= 10.0]
    return bool(hits), f"{len(rows)} WAZE_ALERT line(s): {rows[:3]}"


def check_confirmed_lines(lines: list[str], hazard_type: str = "POLICE") -> tuple[bool, str]:
    """C3: a CONFIRMED line of `hazard_type` whose Waze alert is of the same type."""
    rows = parse_tagged_json(lines, "CONFIRMED")
    hits = [r for r in rows if r.get("hazard_type") == hazard_type
            and (r.get("waze") or {}).get("hazard_type") == hazard_type]
    return bool(hits), f"{len(rows)} CONFIRMED line(s): {rows[:3]}"


def check_nothing_sent(event_lines: list[str], macro: list[dict], hazard_type: str, after_ts: float) -> tuple[bool, str]:
    """C3: no detector event and no mock Waze macro row of `hazard_type` since `after_ts`."""
    events = [e for e in parse_event_lines(event_lines) if e.get("hazard_type") == hazard_type]
    rows = [m for m in macro if m.get("hazard_type") == hazard_type and float(m.get("ts", 0)) >= after_ts - 1]
    return not events and not rows, f"{len(events)} detector event(s), {len(rows)} macro row(s): {rows}"


def check_control_event(macro: list[dict], hazard_type: str, after_ts: float) -> tuple[bool, str]:
    """C4: mock Waze got the event and the macro ran it as a dry run to the end."""
    rows = [m for m in macro if m.get("hazard_type") == hazard_type and float(m.get("ts", 0)) >= after_ts - 1]
    done = [m for m in rows if m.get("outcome") == "dry_run_cancelled" and m.get("dry_run") is True]
    return bool(done), f"{len(rows)} macro row(s): {[(m.get('outcome'), m.get('dry_run')) for m in rows]}"


def _sdk_dirs_from_local_properties() -> list[str]:
    dirs = []
    for prop in sorted((REPO / "android").glob("*/local.properties")):
        try:
            text = prop.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        m = re.search(r"^sdk\.dir\s*=\s*(.+)$", text, re.M)
        if m:
            # undo the java properties escaping of ':' and '\'
            dirs.append(re.sub(r"\\(.)", r"\1", m.group(1).strip()))
    return dirs


def resolve_adb(explicit: str | None = None) -> str:
    """--adb, else ANDROID_HOME / ANDROID_SDK_ROOT, else PATH, else sdk.dir of local.properties."""
    exe = "adb.exe" if os.name == "nt" else "adb"
    if explicit:
        found = explicit if Path(explicit).is_file() else shutil.which(explicit)
        if not found:
            raise SystemExit(f"adb not found: --adb {explicit} is neither a file nor a command on PATH")
        return str(found)
    tried = []
    for var in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        root = os.environ.get(var)
        if root:
            cand = Path(root) / "platform-tools" / exe
            if cand.is_file():
                return str(cand)
            tried.append(f"{var}={root} (no platform-tools/{exe})")
        else:
            tried.append(f"{var} not set")
    on_path = shutil.which("adb")
    if on_path:
        return on_path
    tried.append("adb not on PATH")
    for root in _sdk_dirs_from_local_properties():
        cand = Path(root) / "platform-tools" / exe
        if cand.is_file():
            return str(cand)
        tried.append(f"sdk.dir={root} (no platform-tools/{exe})")
    raise SystemExit("adb not found. Set ANDROID_HOME (or ANDROID_SDK_ROOT) to the Android SDK, put adb on "
                     "PATH, or pass --adb <path to adb>. Tried: " + "; ".join(tried))


def resolve_video(explicit: str | None = None) -> str:
    """--video, else the test clip that tools/fetch_test_videos.py puts in assets/videos/."""
    if explicit:
        path = Path(explicit)
        if not path.is_file() and not path.is_absolute() and (REPO / path).is_file():
            path = REPO / path
        if not path.is_file():
            raise SystemExit(f"video not found: {explicit}")
        return str(path.resolve())
    for name in DEFAULT_VIDEOS:
        if (VIDEO_DIR / name).is_file():
            return str(VIDEO_DIR / name)
    raise SystemExit(f"no test clip in {VIDEO_DIR} (looked for {', '.join(DEFAULT_VIDEOS)}). Run "
                     "tools/fetch_test_videos.py to download the test clips, or pass --video <file>. "
                     "Stage A alone (--stage a) needs no clip.")


def script_env() -> dict:
    """Environment for the .ps1 helpers, so they use the SDK the resolved adb belongs to."""
    env = dict(os.environ)
    tools = Path(ADB).parent
    if tools.name == "platform-tools" and not (env.get("ANDROID_HOME") or env.get("ANDROID_SDK_ROOT")):
        env["ANDROID_HOME"] = str(tools.parent)
    return env


class Runner:
    def __init__(self, serial: str, out: Path):
        self.serial = serial
        self.detector_extras = ""
        self.expect_hazards: list[str] = []
        self.plan: list[str] = []   # names of all checks this run has to produce
        self.not_run_reason = ""    # why the remaining planned checks did not run
        self.last_tap_error = ""
        self.out = out
        self.results: list[tuple[str, str, str]] = []  # (status, name, detail)
        self.logcat_proc: subprocess.Popen | None = None
        self.log_lines: list[str] = []
        self.detector_installed = False
        self.c_why = ""                          # why the rest of Stage C did not run
        self.c_orig: tuple[str | None, str | None] = (None, None)  # accessibility settings before Stage C

    # ------------------------------------------------------------ adb helpers
    def adb(self, *args: str, timeout: float = 60, check: bool = False) -> str:
        cmd = [ADB, "-s", self.serial, *args]
        p = subprocess.run(cmd, capture_output=True, timeout=timeout)
        out = p.stdout.decode("utf-8", "replace") + p.stderr.decode("utf-8", "replace")
        if check and p.returncode != 0:
            raise RuntimeError(f"adb {' '.join(args)} failed: {out.strip()}")
        return out

    def sh(self, cmd: str, timeout: float = 60) -> str:
        return self.adb("shell", cmd, timeout=timeout)

    def log(self, msg: str) -> None:
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        self.log_lines.append(line)

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append(("PASS" if ok else "FAIL", name, detail))
        self.log(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
        return ok

    def skip(self, name: str, why: str) -> None:
        self.results.append(("SKIP", name, why))
        self.log(f"SKIP  {name}  ({why})")

    def screenshot(self, name: str) -> Path:
        remote = f"/sdcard/e2e_{name}.png"
        self.sh(f"screencap -p {remote}")
        local = self.out / f"{name}.png"
        self.adb("pull", remote, str(local))
        self.sh(f"rm {remote}")
        return local

    def start_logcat(self) -> None:
        self.adb("logcat", "-c")
        f = open(self.out / "logcat.txt", "wb")
        self.logcat_proc = subprocess.Popen([ADB, "-s", self.serial, "logcat", "-v", "time"],
                                            stdout=f, stderr=subprocess.STDOUT)

    def stop_logcat(self) -> None:
        if self.logcat_proc:
            self.logcat_proc.terminate()
            try:
                self.logcat_proc.wait(5)
            except subprocess.TimeoutExpired:
                self.logcat_proc.kill()

    def logcat_text(self) -> str:
        try:
            return (self.out / "logcat.txt").read_bytes().decode("utf-8", "replace")
        except FileNotFoundError:
            return ""

    def pull_jsonl(self, name: str) -> list[dict]:
        text = self.sh(f"cat {REMOTE_DIR}/{name} 2>/dev/null")
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return rows

    def save_jsonl(self, suffix: str = "") -> None:
        for name in ("reports.jsonl", "macro_events.jsonl"):
            rows = self.pull_jsonl(name)
            (self.out / name.replace(".jsonl", f"{suffix}.jsonl")).write_text(
                "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def broadcast(self, hazard_type: str, confidence: str = "0.87", dry_run: bool = False,
                  source: str = "e2e") -> None:
        ts = f"{time.time():.3f}"
        cat, sub = EXPECTED.get(hazard_type, ("", ""))
        cmd = (f"am broadcast -a {HAZARD_ACTION} --es hazard_type {hazard_type} "
               f"--es confidence {confidence} --es ts {ts} --es source {source} "
               f"--es dry_run {'true' if dry_run else 'false'}")
        if cat:
            cmd += f" --es waze_category {cat}"
        if sub:
            cmd += f" --es waze_subcategory '{sub}'"
        out = self.sh(cmd)
        self.log(f"broadcast {hazard_type} dry_run={dry_run}: {out.strip().splitlines()[-1] if out.strip() else ''}")

    def wait_macro_outcome(self, hazard_type: str, outcomes: set[str], after_ts: float,
                           timeout: float = 25) -> dict | None:
        end = time.time() + timeout
        while time.time() < end:
            for row in self.pull_jsonl("macro_events.jsonl"):
                if (row.get("hazard_type") == hazard_type and row.get("outcome") in outcomes
                        and float(row.get("ts", 0)) >= after_ts - 2):
                    return row
            time.sleep(0.5)
        return None

    def macro_detail(self, row: dict | None) -> str:
        """Check detail for a macro outcome row; spells out an obstructed run."""
        if row and row.get("outcome") == "obstructed":
            return (f"macro obstructed: mock Waze was not the focused window ({row.get('detail', '')}); "
                    f"focused now: '{self.focused_window() or 'none'}'; row={row}")
        return str(row)

    def device_time(self) -> float:
        out = self.sh("date +%s").strip().splitlines()
        try:
            return float(out[-1])
        except (ValueError, IndexError):
            return time.time()

    # ------------------------------------------------------------ UI helpers (uiautomator)
    def ui_nodes(self) -> list[dict]:
        self.sh("uiautomator dump /sdcard/e2e_ui.xml", timeout=30)
        xml = self.sh("cat /sdcard/e2e_ui.xml")
        start = xml.find("<?xml")
        if start < 0:
            return []
        try:
            root = ET.fromstring(xml[start:].strip())
        except ET.ParseError:
            return []
        nodes = []
        for n in root.iter("node"):
            m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", n.get("bounds", ""))
            if not m:
                continue
            x1, y1, x2, y2 = map(int, m.groups())
            nodes.append({"text": n.get("text", ""), "desc": n.get("content-desc", ""),
                          "pkg": n.get("package", ""), "rid": n.get("resource-id", ""),
                          "clickable": n.get("clickable") == "true",
                          "cx": (x1 + x2) // 2, "cy": (y1 + y2) // 2, "bounds": (x1, y1, x2, y2)})
        return nodes

    def focused_window(self) -> str:
        return parse_focus(self.sh("dumpsys window | grep -E 'mCurrentFocus'"))

    def clear_obstruction(self, allowed: tuple[str, ...] = (MOCK_PKG,)) -> str:
        """Look at the focused window. Returns '' when it belongs to one of the `allowed` apps,
        otherwise a description of what is in the way.

        A host overload (training, a second emulator) leaves "<app> isn't responding" dialogs on
        top of everything; they swallow every tap. For a foreign app the offending process is
        killed so the dialog goes away (the launcher restarts by itself). The mock and the
        detector are never killed: their dialog is answered with its Wait button."""
        title = self.focused_window()
        if not title:
            return "no focused window (mCurrentFocus=null)"
        if focus_package(title) in allowed:
            return ""
        pkg = anr_package(title)
        if not pkg:
            return f"window '{title}'"
        if pkg in (MOCK_PKG, DET_PKG):
            wait = [n for n in self.ui_nodes() if n["rid"].endswith(":id/aerr_wait") or n["text"] == "Wait"]
            if wait:
                self.sh(f"input tap {wait[0]['cx']} {wait[0]['cy']}")
                self.log(f"not-responding dialog of {pkg}: pressed Wait")
                time.sleep(1)
                return f"window '{title}' (pressed Wait)"
            self.log(f"not-responding dialog of {pkg}: no Wait button found, cannot dismiss it")
            return f"window '{title}' (no Wait button found, not dismissed)"
        self.log(f"dismissing not-responding dialog of {pkg}")
        self.sh(f"am force-stop {pkg}")
        time.sleep(2)
        return f"window '{title}' (force-stopped {pkg})"

    def tap_label(self, label: str, retries: int = 6) -> tuple[int, int] | None:
        """Find a view by text or content description and tap its centre with `input tap`.
        On failure last_tap_error says which window was in the way."""
        blocker = ""
        for _ in range(retries):
            # a foreign window (ANR dialog, permission prompt, ...) hides the mock from the dump
            blocker = self.clear_obstruction()
            nodes = [n for n in self.ui_nodes() if n["pkg"] == MOCK_PKG and label in (n["text"], n["desc"])]
            nodes.sort(key=lambda n: not n["clickable"])
            if nodes:
                n = nodes[0]
                self.sh(f"input tap {n['cx']} {n['cy']}")
                self.log(f"tap '{label}' at ({n['cx']},{n['cy']})")
                return n["cx"], n["cy"]
            time.sleep(0.7)
        blocker = self.clear_obstruction() or blocker
        if blocker:
            self.last_tap_error = f"could not tap '{label}': obstructed by {blocker}"
        else:
            self.last_tap_error = f"could not find '{label}' on screen (mock Waze had window focus)"
        self.log(self.last_tap_error)
        return None

    # ------------------------------------------------------------ setup
    def ensure_installed(self, install: bool) -> bool:
        have = MOCK_PKG in self.sh(f"pm path {MOCK_PKG}")
        if install or not have:
            if not MOCK_APK.exists():
                self.log(f"mock Waze APK missing: {MOCK_APK} (build it with gradlew.bat assembleDebug)")
                return False
            out = self.adb("install", "-r", "-g", str(MOCK_APK), timeout=180)
            self.log(f"install mock Waze: {out.strip().splitlines()[-1] if out.strip() else '?'}")
        return MOCK_PKG in self.sh(f"pm path {MOCK_PKG}")

    def launch_mock(self, **extra) -> None:
        cfg = dict(TIMING)
        cfg.update(extra)
        parts = [f"am start -W -n {MOCK_PKG}/.MainActivity"]
        for k, v in cfg.items():
            if isinstance(v, bool):
                parts.append(f"--ez {k} {'true' if v else 'false'}")
            else:
                parts.append(f"--ei {k} {int(v)}")
        out = self.sh(" ".join(parts), timeout=60)
        self.log("launch mock Waze: " + " | ".join(l.strip() for l in out.splitlines() if "Status" in l or "Error" in l))

    def service_running(self) -> bool:
        return "AutoMacroService" in self.sh(f"dumpsys activity services {MOCK_PKG}")

    def install_detector(self) -> tuple[bool, str]:
        """Installs the newest built detector APK once per run. (ok, detail); detail starts with
        'detector APK not built' when there is neither an APK nor an installed detector."""
        if self.detector_installed:
            return True, "installed earlier in this run"
        apks = sorted(glob.glob(DET_APK_GLOB), key=os.path.getmtime)
        installed = DET_PKG in self.sh(f"pm path {DET_PKG}")
        if not apks and not installed:
            return False, "detector APK not built (android/detector/app/build/outputs/apk/debug/*.apk)"
        if not apks:
            self.detector_installed = True
            return True, "already on the device, no APK built to reinstall"
        out = self.adb("install", "-r", "-g", apks[-1], timeout=300)
        self.detector_installed = "Success" in out
        return self.detector_installed, out.strip().splitlines()[-1] if out.strip() else ""

    # ------------------------------------------------------------ stage A
    def stage_a(self) -> None:
        self.log("=== Stage A: mock Waze auto macro ===")
        self.sh(f"am force-stop {MOCK_PKG}")
        # a detector left running by an earlier run would keep broadcasting its own events
        # into this stage and the mock would report "skipped_busy" for ours
        self.sh(f"am force-stop {DET_PKG}")
        self.sh("input keyevent 82")  # make sure the lock screen is gone
        self.sh("svc power stayon true")
        self.clear_obstruction()  # the launcher is in front here; only a dialog gets handled
        self.launch_mock(auto_macro=True, require_foreground=True, bring_to_front=False,
                         clear_reports=True, reset_rate_limit=True)
        time.sleep(2.5)
        self.check(CHECKS_A["A0"], self.service_running())
        self.screenshot("A0_map")

        # --- A1: manual rehearsal with input taps, the way the MacroDroid macro taps ------
        self.log("A1: manual rehearsal with adb input tap (Report button > Hazard > Pothole > Cancel)")
        t0 = self.device_time()
        self.last_tap_error = ""
        ok = self.tap_label("Report button") is not None
        time.sleep(0.8)
        self.screenshot("A1_report_grid")
        ok = ok and self.tap_label("Hazard") is not None
        time.sleep(0.8)
        self.screenshot("A1_hazard_sheet")
        ok = ok and self.tap_label("Pothole") is not None
        time.sleep(0.6)
        self.screenshot("A1_pothole_selected")
        ok = ok and self.tap_label("Cancel") is not None
        time.sleep(1.0)
        rows = [r for r in self.pull_jsonl("reports.jsonl") if r.get("via") == "touch" and r.get("ts", 0) >= t0 - 2]
        good = [r for r in rows if r.get("category") == "Hazard" and r.get("subcategory") == "Pothole"
                and r.get("dry_run") is True and r.get("action") == "cancel"]
        self.check(CHECKS_A["A1"], ok and len(good) == 1,
                   (self.last_tap_error + "; " if not ok else "") + f"{len(rows)} touch row(s): {rows}")

        # --- A1: a one-step category and the Thanks! toast via a manual Police send --------
        self.log("A1: manual Police send with adb input tap (Report button > Police > Report)")
        t1 = self.device_time()
        self.last_tap_error = ""
        ok = self.tap_label("Report button") is not None
        time.sleep(0.8)
        ok = ok and self.tap_label("Police") is not None
        time.sleep(0.8)
        self.screenshot("A1_police_confirm")
        ok = ok and self.tap_label("Report") is not None
        time.sleep(0.4)
        self.screenshot("A1_thanks_toast")
        time.sleep(1.5)
        rows = [r for r in self.pull_jsonl("reports.jsonl") if r.get("via") == "touch" and r.get("ts", 0) >= t1 - 2]
        good = [r for r in rows if r.get("category") == "Police" and r.get("action") == "send"
                and r.get("dry_run") is False]
        self.check(CHECKS_A["A1_police"], ok and len(good) == 1,
                   (self.last_tap_error + "; " if not ok else "") + f"{len(rows)} touch row(s): {rows}")

        # --- A2: real event via broadcast -> exactly one report ---------------------------
        self.log("A2: HAZARD_ON_ROAD_POTHOLE dry_run=false")
        t_a2 = self.device_time()
        self.broadcast("HAZARD_ON_ROAD_POTHOLE", "0.87", dry_run=False)
        # timeline: start delay 0.5 s, then a tap every 1.0 s -> confirm sheet at ~2.5-3.4 s
        time.sleep(3.0)
        self.screenshot("A2_macro_confirm_sheet")
        row = self.wait_macro_outcome("HAZARD_ON_ROAD_POTHOLE", MACRO_END, t_a2)
        self.check(CHECKS_A["A2_outcome"], bool(row and row["outcome"] == "sent"), self.macro_detail(row))
        time.sleep(0.5)
        self.screenshot("A2_after_send")
        sends = [r for r in self.pull_jsonl("reports.jsonl") if r.get("via") == "auto_macro" and r.get("action") == "send"]
        pothole = [r for r in sends if r.get("category") == "Hazard" and r.get("subcategory") == "Pothole"
                   and r.get("dry_run") is False and r.get("hazard_type") == "HAZARD_ON_ROAD_POTHOLE"]
        self.check(CHECKS_A["A2_once"], len(pothole) == 1 and len(sends) == 1,
                   f"{len(sends)} auto send(s): {sends}")

        # --- A3: second event inside 60 s is suppressed ------------------------------------
        self.log("A3: CRASH dry_run=false inside the 60 s window")
        t_a3 = self.device_time()
        self.broadcast("CRASH", "0.91", dry_run=False)
        row = self.wait_macro_outcome("CRASH", {"suppressed_rate_limit", "accepted", "sent", "failed"}, t_a3, 10)
        self.check(CHECKS_A["A3_suppressed"], bool(row and row["outcome"] == "suppressed_rate_limit"),
                   str(row))
        time.sleep(4.5)  # long enough that a non-suppressed macro would have sent
        sends = [r for r in self.pull_jsonl("reports.jsonl") if r.get("via") == "auto_macro" and r.get("action") == "send"]
        self.check(CHECKS_A["A3_once"], len(sends) == 1, f"{len(sends)} send(s)")

        # --- A4: dry run -> confirm sheet, wait, cancel, logged dry_run=true ----------------
        self.log("A4: HAZARD_ON_ROAD_CONSTRUCTION dry_run=true (inside the window, dry runs are not limited)")
        t_a4 = self.device_time()
        self.broadcast("HAZARD_ON_ROAD_CONSTRUCTION", "0.66", dry_run=True)
        time.sleep(3.6)  # confirm sheet is held for dry_run_hold_ms before Cancel
        self.screenshot("A4_dry_run_confirm_sheet")
        row = self.wait_macro_outcome("HAZARD_ON_ROAD_CONSTRUCTION", MACRO_END | {"suppressed_rate_limit"}, t_a4)
        self.check(CHECKS_A["A4_outcome"],
                   bool(row and row["outcome"] == "dry_run_cancelled"), self.macro_detail(row))
        rows = [r for r in self.pull_jsonl("reports.jsonl") if r.get("via") == "auto_macro"
                and r.get("hazard_type") == "HAZARD_ON_ROAD_CONSTRUCTION"]
        self.check(CHECKS_A["A4_logged"],
                   len(rows) == 1 and rows[0].get("dry_run") is True and rows[0].get("action") == "cancel"
                   and rows[0].get("subcategory") == "Construction", str(rows))
        sends = [r for r in self.pull_jsonl("reports.jsonl") if r.get("action") == "send" and r.get("via") == "auto_macro"]
        self.check(CHECKS_A["A4_no_send"], len(sends) == 1, f"{len(sends)} auto send(s)")

        # --- A4 again for the types whose menu path has no subcategory tile ----------------
        # (the macro then goes report button > category > Cancel, without a "sub:" step)
        self.dry_run_without_sub("A4_police", "POLICE")
        self.dry_run_without_sub("A4_animals", "HAZARD_SHOULDER_ANIMALS")

        # --- A5: unknown hazard type is ignored ------------------------------------------
        t_a5 = self.device_time()
        self.broadcast("NOT_A_REAL_HAZARD", "0.5", dry_run=False)
        row = self.wait_macro_outcome("NOT_A_REAL_HAZARD", {"unknown_hazard_type"}, t_a5, 8)
        self.check(CHECKS_A["A5"], row is not None, str(row))

        # --- logcat ------------------------------------------------------------------------
        time.sleep(1)
        text = self.logcat_text()
        n_report = len(re.findall(r"MockWazeReport", text))
        self.check(CHECKS_A["A6"], n_report >= 3, f"{n_report} line(s)")
        self.save_jsonl("_stage_a")

    def dry_run_without_sub(self, key: str, hazard_type: str) -> None:
        category, sub = EXPECTED[hazard_type]
        assert sub == "", hazard_type
        self.log(f"{key}: {hazard_type} dry_run=true ({category}, no subcategory)")
        time.sleep(1.0)  # let the sheet of the previous macro close
        t = self.device_time()
        self.broadcast(hazard_type, "0.70", dry_run=True)
        time.sleep(2.4)  # confirm sheet is up from ~1.5 s until Cancel at ~4 s
        self.screenshot(f"{key}_dry_run_confirm_sheet")
        row = self.wait_macro_outcome(hazard_type, MACRO_END | {"suppressed_rate_limit", "skipped_busy",
                                                                "skipped_not_foreground"}, t)
        reports = self.pull_jsonl("reports.jsonl")
        rows = [r for r in reports if r.get("via") == "auto_macro" and r.get("hazard_type") == hazard_type]
        sends = [r for r in reports if r.get("action") == "send" and r.get("via") == "auto_macro"]
        ok = (bool(row) and row.get("outcome") == "dry_run_cancelled"
              and row.get("category") == category and row.get("subcategory") == ""
              and len(rows) == 1 and rows[0].get("action") == "cancel" and rows[0].get("dry_run") is True
              and rows[0].get("category") == category and rows[0].get("subcategory") == ""
              and len(sends) == 1)
        self.check(CHECKS_A[key], ok,
                   f"{self.macro_detail(row)}; report row(s): {rows}; {len(sends)} auto send(s)")

    # ------------------------------------------------------------ stage B
    def stage_b(self, video: str, video_seconds: int, wait_s: int) -> None:
        self.log("=== Stage B: detector video mode -> mock Waze ===")
        blocker = self.clear_obstruction(allowed=(MOCK_PKG, DET_PKG))
        if "Not Responding" in blocker:
            self.log(f"Stage B started behind {blocker}")
        ok, detail = self.install_detector()
        if detail.startswith("detector APK not built"):
            self.not_run_reason = detail
            self.log(f"Stage B cannot run: {self.not_run_reason}")
            return
        if not self.check(CHECKS_B["B0"], ok, detail):
            self.not_run_reason = "detector install failed (B0)"
            return
        ps = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(HERE / "push_video.ps1"),
              "-InputFile", video, "-Port", self.serial.split("-")[-1]]
        if video_seconds:
            ps += ["-Seconds", str(video_seconds)]
        p = subprocess.run(ps, capture_output=True, text=True, timeout=900, env=script_env())
        (self.out / "push_video.log").write_text(p.stdout + p.stderr, encoding="utf-8")
        if not self.check(CHECKS_B["B1"], p.returncode == 0, (p.stdout + p.stderr).strip().splitlines()[-1:][0]
                          if (p.stdout + p.stderr).strip() else ""):
            self.not_run_reason = "video not pushed (B1)"
            return

        # Mock Waze must take events while the detector is on screen; bring_to_front needs the
        # "display over other apps" app-op to start an activity from the background.
        self.sh(f"appops set {MOCK_PKG} SYSTEM_ALERT_WINDOW allow")
        self.sh(f"am force-stop {MOCK_PKG}")
        self.launch_mock(auto_macro=True, require_foreground=True, bring_to_front=True,
                         clear_reports=True, reset_rate_limit=True)
        time.sleep(2)
        mark = self.logcat_text().count("\n")
        self.sh(f"am force-stop {DET_PKG}")
        out = self.sh(f"am start -n {DET_PKG}/.MainActivity --es source video --es video test.mp4 "
                      f"--ez reset_settings true {self.detector_extras}".strip())
        self.log("start detector: " + out.strip().replace("\n", " | "))
        self.screenshot("B_detector_start")
        # On a phone Waze is in front and the detector works from its foreground service; reproduce that
        time.sleep(5)
        self.launch_mock(auto_macro=True, require_foreground=True, bring_to_front=True)
        time.sleep(1)  # window focus follows the launch a moment later
        blocker = self.clear_obstruction()
        if blocker:
            self.log(f"mock Waze is not in front after its launch: obstructed by {blocker}")
        self.screenshot("B_mock_front")
        self.log(f"waiting {wait_s} s for the detector to run through the clip...")
        t_end = time.time() + wait_s
        shot = 0
        while time.time() < t_end:
            time.sleep(min(30, max(1, t_end - time.time())))
            shot += 1
            self.screenshot(f"B_progress_{shot:02d}")
        lines = self.logcat_text().splitlines()[mark:]
        ev_lines = [l for l in lines if "WazeomaticEvent" in l]
        (self.out / "wazeomatic_events.txt").write_text("\n".join(ev_lines), encoding="utf-8")
        det_types = set()
        for l in ev_lines:
            det_types.update(re.findall(r"\b(" + "|".join(sorted(EXPECTED, key=len, reverse=True)) + r")\b", l))
        macro = self.pull_jsonl("macro_events.jsonl")
        reports = self.pull_jsonl("reports.jsonl")
        self.save_jsonl("_stage_b")
        received = [m for m in macro if m.get("hazard_type") in EXPECTED]
        self.log(f"detector event lines: {len(ev_lines)} types={sorted(det_types)}; mock received {len(received)}; "
                 f"reports {len(reports)}")
        if not ev_lines:
            self.skip(CHECKS_B["B2"], "no WazeomaticEvent lines (nothing detected or tag differs)")
            self.not_run_reason = "the detector logged no events (B2), nothing to compare"
            return
        self.check(CHECKS_B["B2"], True, f"{len(ev_lines)} line(s)")
        obstructed = [m for m in received if m.get("outcome") == "obstructed"]
        detail = f"{len(received)} received, outcomes={sorted({str(m.get('outcome')) for m in received})}"
        if obstructed:
            detail += (f"; {len(obstructed)} macro run(s) obstructed: mock Waze was not the focused window "
                       f"({obstructed[0].get('detail', '')})")
        self.check(CHECKS_B["B3"], len(received) > 0 and not obstructed, detail)
        # the app ships with dry_run on, so a fresh install files dry runs: the macro walks the same menu path
        # and cancels on the confirm sheet, which is as good a check of the mapping as a real send
        sends = [r for r in reports if r.get("via") == "auto_macro" and
                 (r.get("action") == "send" or (r.get("action") == "cancel" and r.get("dry_run") is True))]
        mismatched = [r for r in sends if (r.get("category"), r.get("subcategory")) !=
                      EXPECTED.get(r.get("hazard_type", ""), ("?", "?"))]
        n_dry = sum(1 for r in sends if r.get("dry_run") is True)
        self.check(CHECKS_B["B4"], len(sends) > 0 and not mismatched,
                   f"{len(sends)} report(s) ({n_dry} dry run), {len(mismatched)} mismatched")
        unknown = [r for r in sends if r.get("hazard_type") not in det_types]
        self.check(CHECKS_B["B5"], not unknown, f"{len(unknown)} untraced")
        limit_ok = len(sends) <= int(wait_s / 60) + 1
        self.check(CHECKS_B["B6"], limit_ok, f"{len(sends)} sends in {wait_s} s")
        events = parse_event_lines(ev_lines)
        for hazard_type in self.expect_hazards:
            ok, detail = check_expected_hazard(hazard_type, events, macro)
            self.check(b7_name(hazard_type), ok, detail)

    # ------------------------------------------------------------ stage C
    def log_lines_since(self, mark: int) -> list[str]:
        return self.logcat_text().splitlines()[mark:]

    def wait_log(self, mark: int, predicate, timeout: float) -> tuple[bool, str]:
        """Polls logcat.txt from line `mark` until predicate(lines) -> (ok, detail) holds."""
        end = time.time() + timeout
        ok, detail = predicate(self.log_lines_since(mark))
        while not ok and time.time() < end:
            time.sleep(0.5)
            ok, detail = predicate(self.log_lines_since(mark))
        return ok, detail

    def show_alert(self, hazard_type: str = "POLICE", distance: str = C_DISTANCE_TEXT, minutes_ago: int = 2,
                   duration_ms: int = 180000) -> None:
        out = self.sh(f"am broadcast -a {SHOW_ALERT_ACTION} -p {MOCK_PKG} --es alert_type {hazard_type} "
                      f"--es distance '{distance}' --ei minutes_ago {minutes_ago} --ez still_there false "
                      f"--ei duration_ms {duration_ms}")
        self.log(f"show alert {hazard_type} {distance}: {out.strip().splitlines()[-1] if out.strip() else ''}")

    def clear_alerts(self) -> None:
        self.sh(f"am broadcast -a {CLEAR_ALERTS_ACTION} -p {MOCK_PKG}")

    def inject(self, label: str, confidence: float, count: int, reset: bool = False, clear_waze: bool = False) -> None:
        cmd = (f"am broadcast -a {INJECT_ACTION} -p {DET_PKG} --es label {label} --es confidence {confidence} "
               f"--ei count {count}")
        if reset:
            cmd += " --ez reset true"
        if clear_waze:
            cmd += " --ez clear_waze true"
        out = self.sh(cmd)
        self.log(f"inject {count} x {label}@{confidence} reset={reset} clear_waze={clear_waze}: "
                 f"{out.strip().splitlines()[-1] if out.strip() else ''}")

    def stage_c(self) -> None:
        """Waze reader: the alert card of mock Waze read by the desktop reader (uiautomator) and by the
        detector's accessibility service, an injected police detection turned into a confirmation while
        the alert shows, and sent once it is gone."""
        self.log("=== Stage C: Waze reader ===")
        try:
            reader = import_waze_reader()
        except Exception as e:  # noqa: BLE001
            reader = None
            self.log(f"could not import wazeomatic.waze_reader: {e!r}")
        self.c_orig = (None, None)
        try:
            self._stage_c(reader)
        finally:
            # whatever happened, the emulator's accessibility settings go back to what they were
            orig_services, orig_enabled = self.c_orig
            self.clear_alerts()
            if orig_services is not None:
                for cmd in accessibility_restore_commands(orig_services, orig_enabled):
                    self.sh(cmd)
                now_services = self.sh("settings get secure enabled_accessibility_services").strip()
                now_enabled = self.sh("settings get secure accessibility_enabled").strip()
                self.check(CHECKS_C["C5"], now_services == orig_services.strip() and now_enabled == orig_enabled.strip(),
                           f"enabled_accessibility_services={now_services!r} (was {orig_services.strip()!r}), "
                           f"accessibility_enabled={now_enabled!r} (was {orig_enabled.strip()!r})")
            self.sh(f"am force-stop {DET_PKG}")
            self.results += missing_rows(self.results, CHECKS_C, self.c_why or "an earlier Stage C step failed")
            self.save_jsonl("_stage_c")
            (self.out / "alerts_stage_c.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in self.pull_jsonl("alerts.jsonl")), encoding="utf-8")

    def _stage_c(self, reader) -> None:
        self.sh(f"am force-stop {MOCK_PKG}")
        self.sh(f"am force-stop {DET_PKG}")
        self.sh("input keyevent 82")
        self.sh("svc power stayon true")
        ok, detail = self.install_detector()
        have_detector = ok
        if not ok:
            self.log(f"Stage C without the detector: {detail}")
            self.c_why = detail
        else:
            # pipeline with models but no frames: the only detections are the injected ones
            out = self.sh(f"am start -n {DET_PKG}/.MainActivity --es source video --es video {C_NO_CLIP} "
                          f"--ez reset_settings true --ez dry_run true --ez waze_reader true --ez tcp false")
            self.log("start detector: " + out.strip().replace("\n", " | "))
            time.sleep(6)  # model loading
        self.launch_mock(auto_macro=True, require_foreground=True, bring_to_front=False,
                         clear_reports=True, reset_rate_limit=True)
        time.sleep(2.5)
        self.clear_obstruction()

        # --- C0 / C1: the card and the desktop reader -----------------------------------------
        t0 = self.device_time()
        self.show_alert()
        time.sleep(1.5)
        self.screenshot("C0_alert_card")
        shown = [r for r in self.pull_jsonl("alerts.jsonl")
                 if r.get("action") == "shown" and r.get("alert_type") == "POLICE" and float(r.get("ts", 0)) >= t0 - 2]
        if not self.check(CHECKS_C["C0"], len(shown) >= 1, str(shown)):
            self.c_why = "mock Waze did not show the alert card (C0)"
            return
        if reader is None:
            self.skip(CHECKS_C["C1"], "wazeomatic.waze_reader could not be imported")
        else:
            xml = self.adb("exec-out", "uiautomator", "dump", "/dev/tty", timeout=40)
            (self.out / "C1_uiautomator.xml").write_text(xml, encoding="utf-8")
            screen = reader.parse_uiautomator_xml(xml)
            ok, detail = check_desktop_alerts([a.to_dict() for a in screen.alerts])
            self.check(CHECKS_C["C1"], ok, f"package={screen.package!r}; {detail}")
        if not have_detector:
            return

        # --- C2: switch the accessibility service on and let it read the card ---------------------
        orig_services = self.sh("settings get secure enabled_accessibility_services").strip()
        orig_enabled = self.sh("settings get secure accessibility_enabled").strip()
        self.c_orig = (orig_services, orig_enabled)
        mark = self.logcat_text().count("\n")
        value = accessibility_list_with(orig_services, WAZE_SERVICE)
        self.sh(f"settings put secure enabled_accessibility_services '{value}'")
        self.sh("settings put secure accessibility_enabled 1")
        self.log(f"accessibility services: {orig_services!r} -> {value!r}")
        ok, detail = self.wait_log(mark, lambda ls: (any(WAZE_TAG in l and "Waze reader connected" in l for l in ls), ""), 15)
        if not ok:
            dump = self.sh("dumpsys accessibility")
            ok = "WazeWatcherService" in dump
            detail = "no 'Waze reader connected' line; dumpsys accessibility " + ("lists" if ok else "does not list") + " it"
        if not self.check(CHECKS_C["C2_enabled"], ok, detail):
            self.c_why = "the Waze reader service did not connect (C2)"
            return
        # the service only reads on a window change: show the card again now that it is listening
        self.clear_alerts()
        time.sleep(1.0)
        self.show_alert()
        ok, detail = self.wait_log(mark, check_waze_alert_lines, 15)
        self.screenshot("C2_alert_read")
        if not self.check(CHECKS_C["C2"], ok, detail):
            self.c_why = "the phone reader did not see the alert (C2)"
            return

        # --- C3: a police car while Waze shows Police nearby -> confirmation, nothing sent ---------
        t3 = self.device_time()
        mark3 = self.logcat_text().count("\n")
        self.inject("police_vehicle", 0.9, 5)
        ok, detail = self.wait_log(mark3, check_confirmed_lines, 10)
        self.check(CHECKS_C["C3_confirmed"], ok, detail)
        time.sleep(4.0)  # a macro started by a stray event would have logged by now
        ev_lines = [l for l in self.log_lines_since(mark3) if "WazeomaticEvent" in l]
        ok, detail = check_nothing_sent(ev_lines, self.pull_jsonl("macro_events.jsonl"), "POLICE", t3)
        self.check(CHECKS_C["C3_no_event"], ok, detail)

        # --- C4: control. Alert gone, fresh ReportManager (as after a service restart) -> sent ----
        self.clear_alerts()
        time.sleep(1.5)
        t4 = self.device_time()
        self.inject("police_vehicle", 0.9, 5, reset=True, clear_waze=True)
        row = self.wait_macro_outcome("POLICE", MACRO_END | {"suppressed_rate_limit", "skipped_busy",
                                                             "skipped_not_foreground"}, t4, 25)
        self.screenshot("C4_control")
        ok, detail = check_control_event(self.pull_jsonl("macro_events.jsonl"), "POLICE", t4)
        self.check(CHECKS_C["C4"], ok, f"{detail}; last outcome row: {self.macro_detail(row)}")

    # ------------------------------------------------------------ summary
    def summary(self, allow_skip: bool = False) -> int:
        recorded = {r[1] for r in self.results}
        self.results = fill_skips(self.results, self.plan, self.not_run_reason)
        for status, name, detail in self.results:
            if status == "SKIP" and name not in recorded:
                self.log(f"SKIP  {name}  ({detail})")
        v = verdict(self.results)
        count = {st: sum(1 for r in self.results if r[0] == st) for st in ("PASS", "FAIL", "SKIP")}
        lines = ["", "=" * 64, "E2E SUMMARY  " + v, "=" * 64]
        for status, name, detail in self.results:
            lines.append(f"{status:5} {name}" + (f"  ({detail})" if status == "SKIP" and detail else ""))
        lines.append(f"{len(self.results)} check(s): {count['PASS']} PASS, {count['FAIL']} FAIL, {count['SKIP']} SKIP")
        if v == "INCOMPLETE":
            lines.append("INCOMPLETE: the skipped checks produced no evidence, so this run is not a pass"
                         + (" (exit code 0 because of --allow-skip)" if allow_skip else ""))
        lines.append(f"artifacts: {self.out}")
        for l in lines:
            print(l)
        (self.out / "summary.txt").write_text("\n".join(self.log_lines + lines) + "\n", encoding="utf-8")
        return exit_code(v, allow_skip)


def check_schema() -> None:
    """Fail fast if schema.py changed and EXPECTED here is stale."""
    sys.path.insert(0, str(REPO))
    try:
        from wazeomatic.schema import HAZARD_TYPES, INTENT_ACTION  # type: ignore
    except Exception as e:  # pragma: no cover - schema optional for the runner
        print(f"(could not import wazeomatic.schema: {e}; using built-in table)")
        return
    assert INTENT_ACTION == HAZARD_ACTION, INTENT_ACTION
    for k, h in HAZARD_TYPES.items():
        if EXPECTED.get(k) != (h.waze_category, h.waze_subcategory):
            raise SystemExit(f"run_e2e.py EXPECTED is out of date for {k}: schema says "
                             f"{(h.waze_category, h.waze_subcategory)}")


def ensure_device(serial: str, start: bool) -> bool:
    out = subprocess.run([ADB, "devices"], capture_output=True, text=True).stdout
    if re.search(rf"^{re.escape(serial)}\s+device$", out, re.M):
        return True
    if not start:
        return False
    port = serial.split("-")[-1]
    print(f"{serial} not running, starting it...")
    p = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                        str(HERE / "start_emulator.ps1"), "-Port", port, "-Wait"], text=True, env=script_env())
    return p.returncode == 0


def main() -> int:
    global ADB
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serial", default="emulator-5554")
    ap.add_argument("--stage", choices=["a", "b", "c", "all"], default="all",
                    help="a = mock Waze, b = detector video mode, c = Waze reader, all = a + b + c (default)")
    ap.add_argument("--install", action="store_true", help="reinstall mock Waze even if present")
    ap.add_argument("--no-start", action="store_true", help="do not start the emulator if it is not running")
    ap.add_argument("--adb", default=None,
                    help="adb executable (default: ANDROID_HOME / ANDROID_SDK_ROOT platform-tools, then PATH, "
                         "then sdk.dir of android/*/local.properties)")
    ap.add_argument("--video", default=None,
                    help="clip for Stage B (default: assets/videos/dashcam_urban_1080p.mp4, which "
                         "tools/fetch_test_videos.py downloads)")
    ap.add_argument("--video-seconds", type=int, default=0, help="trim the clip (0 = whole 120 s)")
    ap.add_argument("--stage-b-wait", type=int, default=150, help="seconds to let the detector run")
    ap.add_argument("--detector-extras", default="",
                    help="extra `am start` arguments for the detector in Stage B, e.g. "
                         "\"--es class_map car=HAZARD_ON_ROAD_OBJECT --es conf 0.4\" to force events on a hazard-free clip")
    ap.add_argument("--expect-hazard", action="append", default=[], metavar="HAZARD_TYPE",
                    help="Stage B: add check B7 that the detector reported this hazard_type from a hazard-model "
                         "class (not a remapped COCO class) and mock Waze took it. Repeatable.")
    ap.add_argument("--allow-skip", action="store_true",
                    help="exit 0 for an INCOMPLETE run (nothing failed, but checks were skipped)")
    args = ap.parse_args()

    for t in args.expect_hazard:
        if t not in EXPECTED:
            ap.error(f"--expect-hazard {t}: unknown hazard_type (known: {', '.join(EXPECTED)})")
    if args.expect_hazard and args.stage not in ("b", "all"):
        ap.error(f"--expect-hazard is a Stage B check, it cannot be used with --stage {args.stage}")
    check_schema()
    ADB = resolve_adb(args.adb)
    video = resolve_video(args.video) if args.stage in ("b", "all") else ""
    out = REPO / "runs" / "emulator" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    if not ensure_device(args.serial, not args.no_start):
        print(f"FAIL: {args.serial} is not online")
        return 1
    r = Runner(args.serial, out)
    r.detector_extras = args.detector_extras
    r.expect_hazards = list(dict.fromkeys(args.expect_hazard))
    r.plan = planned_checks(args.stage, r.expect_hazards)
    r.log(f"adb: {ADB}")
    r.start_logcat()
    try:
        if not r.ensure_installed(args.install):
            r.check("mock Waze installed", False)
            r.not_run_reason = "mock Waze is not installed"
            return r.summary(args.allow_skip)
        if args.stage in ("a", "all"):
            r.stage_a()
        if args.stage in ("b", "all"):
            r.stage_b(video, args.video_seconds, args.stage_b_wait)
        if args.stage in ("c", "all"):
            r.stage_c()
    except Exception as e:  # keep the summary even if something blows up
        r.check("runner crashed", False, repr(e))
        r.not_run_reason = r.not_run_reason or "the runner crashed before this check"
    finally:
        r.stop_logcat()
    return r.summary(args.allow_skip)


if __name__ == "__main__":
    sys.exit(main())
