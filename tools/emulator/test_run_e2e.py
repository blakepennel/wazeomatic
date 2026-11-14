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
    def test_default_full_run_has_29_checks(self):
        self.assertEqual(len(e2e.CHECKS_A), 14)
        self.assertEqual(len(e2e.CHECKS_B), 7)
        self.assertEqual(len(e2e.CHECKS_C), 8)
        plan = e2e.planned_checks("all", [])
        self.assertEqual(len(plan), 29)
        self.assertEqual(len(set(plan)), 29)

    def test_stage_not_requested_is_not_planned(self):
        self.assertEqual(e2e.planned_checks("a", []), list(e2e.CHECKS_A.values()))
        self.assertEqual(e2e.planned_checks("b", []), list(e2e.CHECKS_B.values()))
        self.assertEqual(e2e.planned_checks("c", []), list(e2e.CHECKS_C.values()))

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
        done = rows(list(e2e.CHECKS_A.values()) + list(e2e.CHECKS_C.values()))
        res = e2e.fill_skips(done, plan, "detector APK not built")
        self.assertEqual(len(res), 29)
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
        keys_c = set(re.findall(r'CHECKS_C\["(\w+)"\]', src))
        # A4_police / A4_animals go through dry_run_without_sub(key, ...)
        keys_a |= set(re.findall(r'dry_run_without_sub\("(\w+)"', src))
        self.assertEqual(keys_a, set(e2e.CHECKS_A))
        self.assertEqual(keys_b, set(e2e.CHECKS_B))
        self.assertEqual(keys_c, set(e2e.CHECKS_C))

    def test_summary_of_a_run_without_stage_b_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = e2e.Runner("emulator-0", Path(tmp))
            r.plan = e2e.planned_checks("all", [])
            for name in list(e2e.CHECKS_A.values()) + list(e2e.CHECKS_C.values()):
                r.results.append(("PASS", name, ""))
            r.not_run_reason = "detector APK not built"
            with mock.patch("builtins.print"):
                self.assertEqual(r.summary(), 2)
            text = (Path(tmp) / "summary.txt").read_text(encoding="utf-8")
            self.assertIn("E2E SUMMARY  INCOMPLETE", text)
            self.assertNotIn("E2E SUMMARY  PASS", text)
            self.assertIn("29 check(s): 22 PASS, 0 FAIL, 7 SKIP", text)
            r2 = e2e.Runner("emulator-0", Path(tmp))
            r2.plan = list(r.plan)
            r2.results = [("PASS", n, "") for n in list(e2e.CHECKS_A.values()) + list(e2e.CHECKS_C.values())]
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


class StageC(unittest.TestCase):
    """Parsing and verdict code of the Waze reader stage."""

    SVC = e2e.WAZE_SERVICE

    def test_accessibility_list_keeps_what_was_on(self):
        self.assertEqual(e2e.accessibility_list_with("null", self.SVC), self.SVC)
        self.assertEqual(e2e.accessibility_list_with("", self.SVC), self.SVC)
        other = "com.example.tts/.Reader"
        self.assertEqual(e2e.accessibility_list_with(other, self.SVC), f"{other}:{self.SVC}")
        # already on, in either the long or the short form: not added twice
        self.assertEqual(e2e.accessibility_list_with(f"{other}:{self.SVC}", self.SVC), f"{other}:{self.SVC}")
        short = "com.wazeomatic.detector/.WazeWatcherService"
        self.assertEqual(e2e.accessibility_list_with(short, self.SVC), short)

    def test_restore_commands(self):
        self.assertEqual(e2e.accessibility_restore_commands("null", "null"),
                         ["settings delete secure enabled_accessibility_services",
                          "settings delete secure accessibility_enabled"])
        self.assertEqual(e2e.accessibility_restore_commands("a/.B\n", "0"),
                         ["settings put secure enabled_accessibility_services 'a/.B'",
                          "settings put secure accessibility_enabled '0'"])

    def test_desktop_alerts_verdict(self):
        good = {"hazard_type": "POLICE", "distance_m": 482.8, "text": "Police · 0.3 mi · Reported 2 min ago"}
        self.assertTrue(e2e.check_desktop_alerts([good])[0])
        self.assertFalse(e2e.check_desktop_alerts([])[0])
        self.assertFalse(e2e.check_desktop_alerts([good, good])[0])
        self.assertFalse(e2e.check_desktop_alerts([dict(good, hazard_type="CRASH")])[0])
        self.assertFalse(e2e.check_desktop_alerts([dict(good, distance_m=3218.7)])[0])
        self.assertFalse(e2e.check_desktop_alerts([dict(good, distance_m=None)])[0])

    def test_desktop_reader_on_the_mock_card(self):
        """The card layout of AlertCard.kt as uiautomator dumps it, through the real desktop parser."""
        reader = e2e.import_waze_reader()
        xml = ("UI hierchary dumped to: /dev/tty<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>"
               "<hierarchy rotation=\"0\"><node text=\"\" package=\"com.wazeomatic.mockwaze\" bounds=\"[0,0][1280,2856]\">"
               "<node text=\"\" content-desc=\"Menu\" package=\"com.wazeomatic.mockwaze\" bounds=\"[46,193][203,350]\"/>"
               "<node text=\"MOCK WAZE · nothing is sent\" package=\"com.wazeomatic.mockwaze\" bounds=\"[400,210][880,260]\"/>"
               "<node text=\"Police\" package=\"com.wazeomatic.mockwaze\" bounds=\"[230,450][420,520]\"/>"
               "<node text=\"0.3 mi\" package=\"com.wazeomatic.mockwaze\" bounds=\"[230,520][360,575]\"/>"
               "<node text=\"Reported 2 min ago\" package=\"com.wazeomatic.mockwaze\" bounds=\"[230,575][620,625]\"/>"
               "<node text=\"Where to?\" package=\"com.wazeomatic.mockwaze\" bounds=\"[150,2600][900,2700]\"/>"
               "</node></hierarchy>")
        screen = reader.parse_uiautomator_xml(xml)
        ok, detail = e2e.check_desktop_alerts([a.to_dict() for a in screen.alerts])
        self.assertTrue(ok, detail)

    def test_mock_alert_titles_classify_back_to_their_key(self):
        reader = e2e.import_waze_reader()
        src = (e2e.REPO / "android" / "mockwaze" / "app" / "src" / "main" / "java" / "com" / "wazeomatic"
               / "mockwaze" / "WazeMenu.kt").read_text(encoding="utf-8")
        block = src[src.index("val alertTitles"):]
        block = block[:block.index(")\n")]
        titles = dict(re.findall(r'"(\w+)" to "([^"]+)"', block))
        self.assertEqual(set(titles), set(e2e.EXPECTED))
        for key, title in titles.items():
            self.assertEqual(reader.classify_text(title), key, title)

    def test_mock_menu_covers_every_hazard_key(self):
        src = (e2e.REPO / "android" / "mockwaze" / "app" / "src" / "main" / "java" / "com" / "wazeomatic"
               / "mockwaze" / "WazeMenu.kt").read_text(encoding="utf-8")
        block = src[src.index("val hazardTypes"):]
        block = block[:block.index(")\n\n")]
        paths = {k: (c, s) for k, c, s in re.findall(r'"(\w+)" to MenuPath\("([^"]+)"(?:, "([^"]*)")?\)', block)}
        self.assertEqual(paths, e2e.EXPECTED)

    WAZE = ('11-13 10:00:01.000 I/WazeomaticWaze( 4242): WAZE_ALERT {"hazard_type":"POLICE","category":"Police",'
            '"text":"Police · 0.3 mi · Reported 2 min ago","distance_m":482.8,"kind":"alert","ts":1794500000.000}')
    CONF = ('11-13 10:00:05.000 I/WazeomaticWaze( 4242): CONFIRMED {"hazard_type":"POLICE","confidence":0.900,'
            '"label":"police_vehicle","ts":1794500004.000,"frame_index":3,"waze":{"hazard_type":"POLICE",'
            '"category":"Police","text":"Police · 0.3 mi","distance_m":482.8,"kind":"alert","ts":1794500003.500}}')

    def test_waze_alert_lines(self):
        self.assertTrue(e2e.check_waze_alert_lines(["junk", self.WAZE])[0])
        self.assertFalse(e2e.check_waze_alert_lines([self.WAZE.replace("POLICE", "CRASH")])[0])
        self.assertFalse(e2e.check_waze_alert_lines([self.WAZE.replace("482.8", "3218.7")])[0])
        self.assertFalse(e2e.check_waze_alert_lines([self.CONF])[0])

    def test_confirmed_lines(self):
        self.assertTrue(e2e.check_confirmed_lines([self.WAZE, self.CONF])[0])
        self.assertFalse(e2e.check_confirmed_lines([self.WAZE])[0])
        other = self.CONF.replace('"waze":{"hazard_type":"POLICE"', '"waze":{"hazard_type":"CRASH"')
        self.assertFalse(e2e.check_confirmed_lines([other])[0])

    def test_nothing_sent(self):
        ev = '11-13 10:00:06.000 I/WazeomaticEvent( 4242): {"hazard_type":"POLICE","confidence":0.9,"ts":105.0}'
        self.assertTrue(e2e.check_nothing_sent([], [], "POLICE", 100.0)[0])
        self.assertFalse(e2e.check_nothing_sent([ev], [], "POLICE", 100.0)[0])
        self.assertFalse(e2e.check_nothing_sent([], [{"hazard_type": "POLICE", "outcome": "accepted", "ts": 101.0}],
                                                "POLICE", 100.0)[0])
        # rows from before the injection and other types do not count
        self.assertTrue(e2e.check_nothing_sent([], [{"hazard_type": "POLICE", "outcome": "sent", "ts": 50.0},
                                                    {"hazard_type": "CRASH", "outcome": "sent", "ts": 101.0}],
                                               "POLICE", 100.0)[0])

    def test_control_event(self):
        done = {"hazard_type": "POLICE", "outcome": "dry_run_cancelled", "dry_run": True, "ts": 110.0}
        self.assertTrue(e2e.check_control_event([done], "POLICE", 100.0)[0])
        self.assertFalse(e2e.check_control_event([dict(done, ts=50.0)], "POLICE", 100.0)[0])
        self.assertFalse(e2e.check_control_event([dict(done, outcome="skipped_not_foreground")], "POLICE", 100.0)[0])
        self.assertFalse(e2e.check_control_event([dict(done, dry_run=False, outcome="sent")], "POLICE", 100.0)[0])

    def test_missing_rows_name_the_stage_reason(self):
        done = [("PASS", e2e.CHECKS_C["C0"], ""), ("FAIL", e2e.CHECKS_C["C1"], "x")]
        extra = e2e.missing_rows(done, e2e.CHECKS_C, "detector APK not built")
        self.assertEqual(len(extra), 6)
        self.assertTrue(all(r[0] == "SKIP" and r[2] == "detector APK not built" for r in extra))
        self.assertEqual(e2e.verdict(done + extra), "FAIL")
        self.assertEqual(e2e.verdict([("PASS", n, "") for n in e2e.CHECKS_C.values()]), "PASS")


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
