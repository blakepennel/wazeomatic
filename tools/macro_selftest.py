"""Fire every hazard type at a connected phone or emulator as a DRY RUN, one at a time, so the
macro's branches can be checked by eye against the trigger contract (docs/pipeline-to-macro.md).

    python tools/macro_selftest.py                       # the device adb sees, every hazard type
    python tools/macro_selftest.py --serial emulator-5554 --only HAZARD_ON_ROAD_POTHOLE POLICE
    python tools/macro_selftest.py --print               # just print the adb commands

Every broadcast carries dry_run=true and there is no option to change that: with it the macro must
walk to the confirm sheet and cancel whatever its own DRY_RUN variable says. Open Waze (or the mock)
first; the macro only acts when it is in the foreground. For each type the script prints what the
macro should do, sends the broadcast, and waits so the result can be watched before the next one.

On the emulator with the mock Waze installed, --check-mock reads the mock's own macro log and
verifies each outcome instead of relying on the eye.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wazeomatic.schema import HAZARD_TYPES, INTENT_ACTION, HazardEvent  # noqa: E402
from wazeomatic.sinks import AdbBroadcastSink  # noqa: E402

MOCK_PKG = "com.wazeomatic.mockwaze"


def find_adb(explicit: str | None) -> str:
    if explicit:
        return explicit
    for var in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        root = os.environ.get(var)
        if root:
            for name in ("adb.exe", "adb"):
                cand = Path(root) / "platform-tools" / name
                if cand.exists():
                    return str(cand)
    found = shutil.which("adb")
    if not found:
        raise SystemExit("adb not found: pass --adb, set ANDROID_HOME, or put platform-tools on PATH")
    return found


def expectation(key: str) -> str:
    h = HAZARD_TYPES[key]
    path = h.waze_category + (f" > {h.waze_subcategory}" if h.waze_subcategory else " (no subtype to pick)")
    return f"open the report menu, tap {path}, stop at the confirm sheet, tap Cancel"


def mock_macro_events(adb: str, serial: str | None) -> list[dict]:
    cmd = [adb] + (["-s", serial] if serial else []) + [
        "shell", f"cat /sdcard/Android/data/{MOCK_PKG}/files/macro_events.jsonl 2>/dev/null"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    out = []
    for line in r.stdout.splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adb")
    ap.add_argument("--serial", help="adb serial when more than one device is connected")
    ap.add_argument("--package", help="deliver only to this package, e.g. com.arlosoft.macrodroid")
    ap.add_argument("--only", nargs="+", choices=list(HAZARD_TYPES), help="hazard types to send (default: all)")
    ap.add_argument("--pause", type=float, default=12.0, help="seconds to watch each run before the next")
    ap.add_argument("--print", action="store_true", dest="print_only", help="print the commands, send nothing")
    ap.add_argument("--check-mock", action="store_true",
                    help="emulator with mock Waze: verify each outcome from the mock's macro log")
    args = ap.parse_args()

    adb = "adb" if args.print_only and not args.adb else find_adb(args.adb)
    sink = AdbBroadcastSink(adb=adb, serial=args.serial, package=args.package, dry_run=True)
    keys = args.only or list(HAZARD_TYPES)
    print(f"action {INTENT_ACTION}, {len(keys)} hazard type(s), every broadcast dry_run=true\n")
    failures = 0
    for i, key in enumerate(keys, 1):
        ev = HazardEvent(hazard_type=key, confidence=0.9, source="sim")
        cmd = sink.command(ev)
        assert cmd[cmd.index("dry_run") + 1] == "true"
        print(f"[{i}/{len(keys)}] {key}\n    expect: {expectation(key)}")
        if args.print_only:
            print("    " + " ".join(cmd))
            continue
        before = len(mock_macro_events(adb, args.serial)) if args.check_mock else 0
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0 or "Broadcast completed" not in r.stdout:
            print(f"    !! adb: {(r.stderr or r.stdout).strip()}")
            failures += 1
            continue
        time.sleep(args.pause)
        if args.check_mock:
            new = [e for e in mock_macro_events(adb, args.serial)[before:] if e.get("hazard_type") == key]
            outcomes = [e.get("outcome") for e in new]
            ok = "dry_run_cancelled" in outcomes and "sent" not in outcomes
            print(f"    mock outcomes: {outcomes} -> {'OK' if ok else 'FAIL'}")
            failures += 0 if ok else 1
    if args.check_mock and not args.print_only:
        print(f"\n{len(keys) - failures}/{len(keys)} hazard types walked to the confirm sheet and cancelled")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
