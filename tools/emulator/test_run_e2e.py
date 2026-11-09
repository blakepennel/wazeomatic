"""Unit tests for the parts of run_e2e.py that need no emulator.

  .venv\\Scripts\\python.exe -m unittest discover -s tools\\emulator -p "test_*.py"
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_e2e as e2e  # noqa: E402


def rows(plan, **status_by_name):
    return [(status_by_name.get(n, "PASS"), n, "") for n in plan]


class PlanAndVerdict(unittest.TestCase):
    def test_default_full_run_has_21_checks(self):
        self.assertEqual(len(e2e.CHECKS_A), 14)
        self.assertEqual(len(e2e.CHECKS_B), 7)
        plan = e2e.planned_checks("all", [])
        self.assertEqual(len(plan), 21)
        self.assertEqual(len(set(plan)), 21)

    def test_stage_not_requested_is_not_planned(self):
        self.assertEqual(e2e.planned_checks("a", []), list(e2e.CHECKS_A.values()))
        self.assertEqual(e2e.planned_checks("b", []), list(e2e.CHECKS_B.values()))

    def test_expect_hazard_adds_b7(self):
        plan = e2e.planned_checks("b", ["HAZARD_ON_ROAD_CONSTRUCTION"])
        self.assertEqual(plan[-1], "B7 detector reported HAZARD_ON_ROAD_CONSTRUCTION from the hazard model")
        self.assertEqual(len(plan), 8)

    def test_all_pass(self):
        plan = e2e.planned_checks("all", [])
        res = e2e.fill_skips(rows(plan), plan, "")
        self.assertEqual(e2e.verdict(res), "PASS")
        self.assertEqual(e2e.exit_code("PASS"), 0)

    def test_stage_b_without_evidence_is_incomplete(self):
        plan = e2e.planned_checks("all", [])
        done = rows(list(e2e.CHECKS_A.values()))
        res = e2e.fill_skips(done, plan, "detector APK not built")
        self.assertEqual(len(res), 21)
        self.assertEqual([r[0] for r in res].count("SKIP"), 7)
        self.assertTrue(all(r[2] == "detector APK not built" for r in res if r[0] == "SKIP"))
        self.assertEqual(e2e.verdict(res), "INCOMPLETE")
        self.assertEqual(e2e.exit_code("INCOMPLETE"), 2)
        self.assertEqual(e2e.exit_code("INCOMPLETE", allow_skip=True), 0)

    def test_stage_a_only_is_a_pass(self):
        plan = e2e.planned_checks("a", [])
        self.assertEqual(e2e.verdict(e2e.fill_skips(rows(plan), plan, "")), "PASS")

    def test_fail_wins_over_skip_and_allow_skip(self):
        plan = e2e.planned_checks("all", [])
        done = rows(list(e2e.CHECKS_A.values()), **{e2e.CHECKS_A["A1"]: "FAIL"})
        res = e2e.fill_skips(done, plan, "x")
        self.assertEqual(e2e.verdict(res), "FAIL")
        self.assertEqual(e2e.exit_code("FAIL", allow_skip=True), 1)

    def test_unplanned_rows_are_kept(self):
        plan = e2e.planned_checks("a", [])
        res = e2e.fill_skips([("FAIL", "runner crashed", "boom")], plan, "the runner crashed")
        self.assertEqual(len(res), 15)
        self.assertEqual(res[-1][1], "runner crashed")
        self.assertEqual(e2e.verdict(res), "FAIL")

    def test_no_rows_is_not_a_pass(self):
        self.assertEqual(e2e.verdict([]), "INCOMPLETE")

    def test_every_check_call_uses_a_planned_name(self):
        src = Path(e2e.__file__).read_text(encoding="utf-8")
        keys_a = set(re.findall(r'CHECKS_A\["(\w+)"\]', src))
        keys_b = set(re.findall(r'CHECKS_B\["(\w+)"\]', src))
        # A4_police / A4_animals go through dry_run_without_sub(key, ...)
        keys_a |= set(re.findall(r'dry_run_without_sub\("(\w+)"', src))
        self.assertEqual(keys_a, set(e2e.CHECKS_A))
        self.assertEqual(keys_b, set(e2e.CHECKS_B))

    def test_summary_of_a_run_without_stage_b_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = e2e.Runner("emulator-0", Path(tmp))
            r.plan = e2e.planned_checks("all", [])
            for name in e2e.CHECKS_A.values():
                r.results.append(("PASS", name, ""))
            r.not_run_reason = "detector APK not built"
            with mock.patch("builtins.print"):
                self.assertEqual(r.summary(), 2)
            text = (Path(tmp) / "summary.txt").read_text(encoding="utf-8")
            self.assertIn("E2E SUMMARY  INCOMPLETE", text)
            self.assertNotIn("E2E SUMMARY  PASS", text)
            self.assertIn("21 check(s): 14 PASS, 0 FAIL, 7 SKIP", text)
            r2 = e2e.Runner("emulator-0", Path(tmp))
            r2.plan = list(r.plan)
            r2.results = [("PASS", n, "") for n in e2e.CHECKS_A.values()]
            with mock.patch("builtins.print"):
                self.assertEqual(r2.summary(allow_skip=True), 0)


class Focus(unittest.TestCase):
    MOCK = "  mCurrentFocus=Window{1f2e3d u0 com.wazeomatic.mockwaze/com.wazeomatic.mockwaze.MainActivity}\n"
    ANR = "  mCurrentFocus=Window{9ab7c1 u0 Application Not Responding: com.android.systemui}\n"

    def test_app_window(self):
        title = e2e.parse_focus(self.MOCK)
        self.assertEqual(e2e.focus_package(title), e2e.MOCK_PKG)
        self.assertEqual(e2e.anr_package(title), "")

    def test_anr_dialog(self):
        title = e2e.parse_focus(self.ANR)
        self.assertEqual(title, "Application Not Responding: com.android.systemui")
        self.assertEqual(e2e.focus_package(title), "")
        self.assertEqual(e2e.anr_package(title), "com.android.systemui")

    def test_null_focus(self):
        self.assertEqual(e2e.parse_focus("  mCurrentFocus=null\n"), "")
        self.assertEqual(e2e.parse_focus(""), "")

    def test_first_real_window_of_several_displays(self):
        self.assertEqual(e2e.focus_package(e2e.parse_focus("mCurrentFocus=null\r\n" + self.MOCK)), e2e.MOCK_PKG)


class FakeDevice(e2e.Runner):
    """Runner with a scripted focus and UI dump instead of adb."""

    def __init__(self, focus, nodes=()):
        super().__init__("emulator-0", Path("."))
        self.focus = focus
        self.nodes = list(nodes)
        self.commands = []

    def sh(self, cmd, timeout=60):
        self.commands.append(cmd)
        return self.focus if "mCurrentFocus" in cmd else ""

    def ui_nodes(self):
        return self.nodes

    def log(self, msg):
        self.log_lines.append(msg)


@mock.patch("time.sleep", lambda s: None)
class Obstruction(unittest.TestCase):
    def test_mock_in_front(self):
        self.assertEqual(FakeDevice(Focus.MOCK).clear_obstruction(), "")

    def test_foreign_anr_is_force_stopped(self):
        d = FakeDevice(Focus.ANR)
        what = d.clear_obstruction()
        self.assertIn("am force-stop com.android.systemui", d.commands)
        self.assertIn("Not Responding: com.android.systemui", what)

    def test_own_anr_is_never_force_stopped(self):
        for pkg in (e2e.MOCK_PKG, e2e.DET_PKG):
            anr = f"mCurrentFocus=Window{{9ab7c1 u0 Application Not Responding: {pkg}}}"
            wait = {"text": "Wait", "desc": "", "pkg": "android", "rid": "android:id/aerr_wait",
                    "clickable": True, "cx": 640, "cy": 1500, "bounds": (0, 0, 0, 0)}
            d = FakeDevice(anr, [wait])
            what = d.clear_obstruction()
            self.assertFalse([c for c in d.commands if "force-stop" in c])
            self.assertIn("input tap 640 1500", d.commands)
            self.assertIn("pressed Wait", what)
            d = FakeDevice(anr, [])
            self.assertIn("not dismissed", d.clear_obstruction())
            self.assertFalse([c for c in d.commands if "force-stop" in c])

    def test_other_window_is_described(self):
        d = FakeDevice("mCurrentFocus=Window{1 u0 com.android.launcher3/com.android.launcher3.Launcher}")
        self.assertEqual(d.clear_obstruction(), "window 'com.android.launcher3/com.android.launcher3.Launcher'")
        self.assertEqual(d.clear_obstruction(allowed=("com.android.launcher3",)), "")

    def test_tap_failure_names_the_window(self):
        d = FakeDevice(Focus.ANR)
        self.assertIsNone(d.tap_label("Hazard", retries=2))
        self.assertIn("could not tap 'Hazard': obstructed by window 'Application Not Responding", d.last_tap_error)

    def test_tap_failure_with_mock_in_front(self):
        d = FakeDevice(Focus.MOCK)
        self.assertIsNone(d.tap_label("Hazard", retries=2))
        self.assertIn("could not find 'Hazard' on screen", d.last_tap_error)
        self.assertNotIn("obstructed", d.last_tap_error)

    def test_tap_checks_focus_before_every_attempt(self):
        d = FakeDevice(Focus.MOCK)
        d.tap_label("Hazard", retries=3)
        self.assertGreaterEqual(len([c for c in d.commands if "mCurrentFocus" in c]), 3)

    def test_tap_found(self):
        node = {"text": "Hazard", "desc": "", "pkg": e2e.MOCK_PKG, "rid": "", "clickable": True,
                "cx": 10, "cy": 20, "bounds": (0, 0, 20, 40)}
        d = FakeDevice(Focus.MOCK, [node])
        self.assertEqual(d.tap_label("Hazard"), (10, 20))
        self.assertIn("input tap 10 20", d.commands)

    def test_obstructed_macro_row_is_spelled_out(self):
        d = FakeDevice(Focus.ANR)
        text = d.macro_detail({"outcome": "obstructed", "detail": "step 'tile:Hazard': window has no focus"})
        self.assertIn("macro obstructed", text)
        self.assertIn("tile:Hazard", text)
        self.assertIn("Not Responding", text)


class ExpectedHazard(unittest.TestCase):
    T = "HAZARD_ON_ROAD_CONSTRUCTION"

    def ev(self, label, ts=100.0, hazard_type=None):
        return {"hazard_type": hazard_type or self.T, "label": label, "ts": ts}

    def macro(self, outcome, event_ts="100.000", hazard_type=None):
        return {"hazard_type": hazard_type or self.T, "outcome": outcome, "event_ts": event_ts}

    def test_parse_logcat_lines(self):
        line = ('11-06 17:47:10.680 I/WazeomaticEvent(13552): {"hazard_type":"HAZARD_ON_ROAD_OBJECT",'
                '"confidence":0.52,"ts":1794008830.676,"label":"car"}')
        self.assertEqual(e2e.parse_event_lines([line, "junk", "x {not json"])[0]["label"], "car")

    def test_hazard_model_event_sent(self):
        ok, _ = e2e.check_expected_hazard(self.T, [self.ev("cone")], [self.macro("accepted"), self.macro("sent")])
        self.assertTrue(ok)

    def test_dry_run_counts(self):
        ok, _ = e2e.check_expected_hazard(self.T, [self.ev("barricade")], [self.macro("dry_run_cancelled")])
        self.assertTrue(ok)

    def test_coco_remap_does_not_count(self):
        ok, detail = e2e.check_expected_hazard(self.T, [self.ev("car")], [self.macro("sent")])
        self.assertFalse(ok)
        self.assertIn("car", detail)

    def test_sent_event_must_be_the_hazard_model_one(self):
        events = [self.ev("car", ts=100.0), self.ev("cone", ts=200.0)]
        ok, _ = e2e.check_expected_hazard(self.T, events, [self.macro("sent", "100.000"),
                                                           self.macro("suppressed_rate_limit", "200.000")])
        self.assertFalse(ok)

    def test_not_received(self):
        self.assertFalse(e2e.check_expected_hazard(self.T, [self.ev("cone")], [])[0])

    def test_other_type_does_not_count(self):
        events = [self.ev("pothole", hazard_type="HAZARD_ON_ROAD_POTHOLE")]
        macro = [self.macro("sent", hazard_type="HAZARD_ON_ROAD_POTHOLE")]
        self.assertFalse(e2e.check_expected_hazard(self.T, events, macro)[0])


class Resolve(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.exe = "adb.exe" if os.name == "nt" else "adb"
        self.no_sdk_env = {k: v for k, v in os.environ.items() if k not in ("ANDROID_HOME", "ANDROID_SDK_ROOT")}

    def sdk(self, name):
        tools = Path(self.tmp.name) / name / "platform-tools"
        tools.mkdir(parents=True)
        (tools / self.exe).write_text("")
        return tools.parent

    def test_explicit_adb(self):
        adb = self.sdk("a") / "platform-tools" / self.exe
        self.assertEqual(e2e.resolve_adb(str(adb)), str(adb))
        with self.assertRaises(SystemExit):
            e2e.resolve_adb(str(Path(self.tmp.name) / "nope" / self.exe))

    def test_android_home_then_sdk_root_then_path(self):
        home, root = self.sdk("home"), self.sdk("root")
        with mock.patch.dict(os.environ, {"ANDROID_HOME": str(home), "ANDROID_SDK_ROOT": str(root)}):
            self.assertEqual(Path(e2e.resolve_adb()), home / "platform-tools" / self.exe)
        with mock.patch.dict(os.environ, dict(self.no_sdk_env, ANDROID_SDK_ROOT=str(root)), clear=True):
            self.assertEqual(Path(e2e.resolve_adb()), root / "platform-tools" / self.exe)
        with mock.patch.dict(os.environ, self.no_sdk_env, clear=True), \
                mock.patch("shutil.which", lambda n: "/somewhere/adb"):
            self.assertEqual(e2e.resolve_adb(), "/somewhere/adb")

    def test_nothing_works(self):
        with mock.patch.dict(os.environ, self.no_sdk_env, clear=True), mock.patch("shutil.which", lambda n: None), \
                mock.patch.object(e2e, "_sdk_dirs_from_local_properties", lambda: []):
            with self.assertRaises(SystemExit) as cm:
                e2e.resolve_adb()
        self.assertIn("ANDROID_HOME", str(cm.exception))
        self.assertIn("--adb", str(cm.exception))

    def test_local_properties_is_the_last_resort(self):
        sdk = self.sdk("props")
        with mock.patch.dict(os.environ, self.no_sdk_env, clear=True), mock.patch("shutil.which", lambda n: None), \
                mock.patch.object(e2e, "_sdk_dirs_from_local_properties", lambda: [str(sdk)]):
            self.assertEqual(Path(e2e.resolve_adb()), sdk / "platform-tools" / self.exe)

    def test_default_video_missing_points_at_the_fetch_script(self):
        with mock.patch.object(e2e, "VIDEO_DIR", Path(self.tmp.name)):
            with self.assertRaises(SystemExit) as cm:
                e2e.resolve_video(None)
            self.assertIn("tools/fetch_test_videos.py", str(cm.exception))
            (Path(self.tmp.name) / "dashcam_urban_1080p.ogv").write_text("")
            self.assertTrue(e2e.resolve_video(None).endswith("dashcam_urban_1080p.ogv"))
            (Path(self.tmp.name) / "dashcam_urban_1080p.mp4").write_text("")
            self.assertTrue(e2e.resolve_video(None).endswith("dashcam_urban_1080p.mp4"))

    def test_explicit_video(self):
        with self.assertRaises(SystemExit):
            e2e.resolve_video(str(Path(self.tmp.name) / "missing.mp4"))

    def test_no_machine_paths_in_the_runner(self):
        src = Path(e2e.__file__).read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"\b[A-Za-z]:\\\\?[A-Za-z]", src))


if __name__ == "__main__":
    unittest.main()
