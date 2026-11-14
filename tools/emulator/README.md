# Emulator test suite

Scripts to run the Wazeomatic Android pieces on an emulator set up like the Pixel 9 Pro
(1280x2856, 495 dpi) against **mock Waze** (`android/mockwaze`) instead of real Waze.
Outputs go to `runs\emulator\`; the SDK and AVD locations come from the environment (below).

Our emulator is AVD `pixel9pro_test` on port **5554** (`emulator-5554`). Every script only uses
that serial, so the detector team's `wazeomatic_dev` on 5560 is never touched.

## Quick start (PowerShell, repo root)

```powershell
$ps = 'powershell -NoProfile -ExecutionPolicy Bypass -File'
iex "$ps tools\emulator\create_avd.ps1"            # once; -Webcam for back cam = webcam0, -Force to recreate
iex "$ps tools\emulator\start_emulator.ps1 -Wait"  # -Headless (no window), -Gpu auto|host|swiftshader_indirect, -ColdBoot
iex "$ps tools\emulator\install_apps.ps1 -Build"   # mock Waze + detector APK if built
.venv\Scripts\python.exe tools\fetch_test_videos.py # once; test clips for Stage B into assets\videos\
.venv\Scripts\python.exe tools\emulator\run_e2e.py  # Stage A + B + C (B and most of C need the detector APK)
```

## Environment

Nothing in `run_e2e.py` is tied to one machine. The `.ps1` scripts dot-source `_env.ps1`, which
uses these variables when they are set and otherwise falls back to the original dev PC layout:

| variable | used for | fallback in `_env.ps1` |
|---|---|---|
| `ANDROID_HOME` (or `ANDROID_SDK_ROOT`) | SDK root: `platform-tools\adb.exe`, `emulator\emulator.exe`, `cmdline-tools` | `F:\dev\android-sdk` |
| `ANDROID_USER_HOME` | where AVDs live (`<it>\avd`) | `<SDK>\.android` |
| `JAVA_HOME` | JDK for `avdmanager` and gradle | `C:\Program Files\OpenLogic\jdk-21.0.7.6-hotspot` |
| `GRADLE_USER_HOME` | gradle cache for `install_apps.ps1 -Build` | `F:\dev\gradle\home` |
| `FFMPEG` | ffmpeg executable for `push_video.ps1` | `ffmpeg` on `PATH`, then the chocolatey install |

`run_e2e.py` finds adb in this order: `--adb <path>`, `%ANDROID_HOME%\platform-tools`,
`%ANDROID_SDK_ROOT%\platform-tools`, `adb` on `PATH`, `sdk.dir` from `android\*\local.properties`
(the file Android Studio writes). If none has adb it stops with a message that lists what it
tried. When it starts a `.ps1` helper it passes the SDK of that adb on as `ANDROID_HOME` unless
one is set. The default Stage B clip is `assets\videos\dashcam_urban_1080p.mp4`, which
`tools\fetch_test_videos.py` downloads; without it (and without `--video`) a run that includes
Stage B stops before it starts and says so. `--stage a` needs no clip.

| script | what it does |
|---|---|
| `create_avd.ps1` | creates `pixel9pro_test` from the `pixel_9_pro` profile (falls back to the closest Pixel) and forces 1280x2856 @ 495 dpi, 4 GB emulator RAM, 6 cores, back camera `virtualscene` (or `webcam0` with `-Webcam`) in `config.ini` |
| `start_emulator.ps1` | starts it in the background on 5554 (`-gpu auto`), log in `runs\emulator\emulator_5554.log` |
| `wait_for_boot.ps1` | polls `sys.boot_completed`, then unlocks and keeps the screen on |
| `install_apps.ps1` | installs mock Waze and `android/detector/app/build/outputs/apk/debug/*.apk` if present |
| `push_video.ps1` | ffmpeg: `.ogv` -> H.264 720p 30 fps mp4 in `runs\emulator\`, pushed to `/sdcard/Android/data/com.wazeomatic.detector/files/test.mp4` and `/sdcard/Download/` (`-Seconds 60` to trim) |
| `screenshot.ps1` / `screenrecord.ps1` | PNG / mp4 into `runs\emulator\` |
| `run_e2e.py` | the end-to-end test (below) |

## run_e2e.py

```powershell
.venv\Scripts\python.exe tools\emulator\run_e2e.py --stage a        # mock Waze only
.venv\Scripts\python.exe tools\emulator\run_e2e.py --stage c        # Waze reader (debug detector APK)
.venv\Scripts\python.exe tools\emulator\run_e2e.py --stage b --stage-b-wait 150
.venv\Scripts\python.exe tools\emulator\run_e2e.py --stage b --video-seconds 60 --stage-b-wait 100 --detector-extras "--es class_map car=HAZARD_ON_ROAD_OBJECT --es conf 0.4 --ez dry_run false"
# (the CC0 clips contain no real hazard; the class_map extra makes the detector treat COCO cars as a hazard so the whole chain is exercised)
.venv\Scripts\python.exe tools\emulator\run_e2e.py --stage b --video assets/videos/labelled/argo_2ff4f798_workzone.mp4 --expect-hazard HAZARD_ON_ROAD_CONSTRUCTION --detector-extras "--ez dry_run false"
# (a real work-zone clip through the shipped hazard model, no class_map; the clip is a local file, it is not in the repo)
.venv\Scripts\python.exe tools\emulator\run_e2e.py --stage b --video assets/videos/labelled/argo_2ff4f798_workzone.mp4 --stage-b-wait 90 --detector-extras "--es class_conf cone=1.01,construction_sign=1.01,delineator=1.01,barricade=1.01,worker=1.01,pothole=1.01,debris=1.01"
# (the same clip with every hazard-model class muted: the parked police car goes through the vehicle classifier to a
#  Police dry run. Unmuted, the work zone holds each minute's report and the car, in view two seconds a loop, is held back)
```

If `adb install` of the detector fails with `INSTALL_FAILED_UPDATE_INCOMPATIBLE`, the APK already on the emulator was signed with a different debug keystore (another PC or user). Run `adb -s emulator-5554 uninstall com.wazeomatic.detector` once and install again.

Starts the emulator if it is not running (unless `--no-start`), installs mock Waze if missing
(`--install` forces it), and writes `logcat.txt`, screenshots, the pulled `reports.jsonl` /
`macro_events.jsonl` and `summary.txt` to `runs\emulator\<timestamp>\`.

**Result and exit code.** A default full run has **29 checks** (14 in Stage A, 7 in Stage B, 8 in
Stage C, plus one B7 per `--expect-hazard`), and every check of the requested stages is listed in the summary.
A check that could not run (no detector APK, the detector logged no events, an earlier step of
the stage failed, the runner crashed) is listed as `SKIP` with the reason. A stage that was not
requested (`--stage a` leaves out Stages B and C) is not listed and is not a skip.

| `--stage` | stages | checks |
|---|---|---|
| `a` | A (mock Waze only) | 14 |
| `b` | B (detector video mode) | 7 + one B7 per `--expect-hazard` |
| `c` | C (Waze reader) | 8 |
| `all` (default) | A, B, C | 29 + B7 rows |

| summary line | meaning | exit code |
|---|---|---|
| `E2E SUMMARY  PASS` | every listed check passed | 0 |
| `E2E SUMMARY  FAIL` | at least one `FAIL` (or the emulator is not online) | 1 |
| `E2E SUMMARY  INCOMPLETE` | no `FAIL`, but at least one `SKIP`: part of the run produced no evidence | 2 (0 with `--allow-skip`) |

**Blocking windows.** Before every tap attempt the runner reads `mCurrentFocus` from
`dumpsys window`. An "<app> isn't responding" dialog of a foreign app is dismissed by
force-stopping that app; for mock Waze or the detector the dialog's Wait button is pressed
instead. If a label still cannot be tapped, the failure says which window was in the way
(`could not tap 'Hazard': obstructed by window '...'`). The mock's own macro refuses to tap
while the app is not resumed and focused and logs outcome `obstructed`; the runner reports that
as a failure of the check.

**Stage A** (mock Waze only):
* A1 taps Report > Hazard > Pothole > Cancel with `adb shell input tap` at the coordinates
  from `uiautomator dump` (same kind of taps the MacroDroid macro does) and expects a
  `dry_run=true` row. A manual Police send (Report > Police > Report) grabs the "Thanks!" toast
  screenshot and must leave a `touch` row with category Police and `action=send`.
* A2 broadcasts `HAZARD_ON_ROAD_POTHOLE` (`dry_run=false`) and expects exactly one
  `Hazard / Pothole` report from the auto macro.
* A3 broadcasts `CRASH` straight after: must be `suppressed_rate_limit` and nothing sent.
* A4 broadcasts `HAZARD_ON_ROAD_CONSTRUCTION` with `dry_run=true`: must reach the confirm
  sheet, cancel, and log `dry_run=true` with no extra send. Then the same dry run for `POLICE`
  and `HAZARD_SHOULDER_ANIMALS`, whose menu path has no subcategory tile (the macro goes
  Report > category > Cancel): outcome `dry_run_cancelled`, category Police / Hazard, empty
  subcategory, still no extra send.
* A5 unknown hazard_type is ignored; A6 `MockWazeReport` lines show up in logcat.

The broadcasts look exactly like what the detector sends:

```
adb -s emulator-5554 shell am broadcast -a com.wazeomatic.HAZARD_DETECTED --es hazard_type HAZARD_ON_ROAD_POTHOLE --es confidence 0.87 --es dry_run false --es source e2e --es waze_category Hazard --es waze_subcategory Pothole
```

**Stage B** (all of B0-B6 are `SKIP` when no detector APK is built or installed): pushes the clip, starts mock
Waze with `bring_to_front=true` (+ `appops set ... SYSTEM_ALERT_WINDOW allow` so it may come to
the front from the background), launches
`am start -n com.wazeomatic.detector/.MainActivity --es source video --es video test.mp4`,
waits, reads logcat tag `WazeomaticEvent` and checks mock Waze received those events, the
reports match the schema mapping and the one-per-minute limit held. The app ships with `dry_run`
on, so without `--detector-extras "--ez dry_run false"` the macro walks the same menu path and
cancels on the confirm sheet; B4-B6 count those dry runs like sends. If the detector logs no
events at all, B2-B6 are reported as SKIP (nothing to compare) and the run is INCOMPLETE. The
default clip is a drive without hazards, so that is its expected result; for events use the
work-zone clip (`--video assets/videos/labelled/argo_2ff4f798_workzone.mp4`), which gives a
Hazard > Construction report a minute on the emulator.

`--expect-hazard <HAZARD_TYPE>` (repeatable) adds check "B7 detector reported <type> from the
hazard model": at least one detector event of that type whose `label` is a class of the shipped
hazard model (pothole, cone, construction_sign, delineator, barricade, worker), not a COCO class
remapped with `class_map`, and a mock Waze macro run for one of those events that finished
(`sent`, or `dry_run_cancelled` when the detector runs with dry_run on). An event that mock Waze
only suppressed (rate limit, busy) does not count. Without the option there is no B7.

**Stage C** (Waze reader; C0/C1 need only mock Waze, C2-C5 a **debug** detector APK because the
injection hook only exists in debug builds):
* C0 `am broadcast -a com.wazeomatic.mockwaze.SHOW_ALERT -p com.wazeomatic.mockwaze --es alert_type POLICE --es distance '0.3 mi' --ei minutes_ago 2`
  must leave a `shown` row in `alerts.jsonl` (next to `reports.jsonl`).
* C1 `adb exec-out uiautomator dump /dev/tty`, parsed by `wazeomatic.waze_reader.parse_uiautomator_xml`,
  must hold exactly one alert: POLICE at about 483 m.
* C2 adds `com.wazeomatic.detector/com.wazeomatic.detector.WazeWatcherService` to
  `enabled_accessibility_services` (keeping whatever was already in the list) and sets
  `accessibility_enabled 1`; the service must connect and log `WAZE_ALERT {...}` with POLICE (logcat
  tag `WazeomaticWaze`).
* C3 injects 5 frames of `police_vehicle` @ 0.9 with
  `am broadcast -a com.wazeomatic.detector.INJECT_DETECTION -p com.wazeomatic.detector --es label police_vehicle --es confidence 0.9 --ei count 5`:
  the detector must log `CONFIRMED {...}` and send nothing (no `WazeomaticEvent` POLICE line, no
  POLICE row in `macro_events.jsonl`).
* C4 clears the card (`CLEAR_ALERTS`) and injects again with `--ez reset true --ez clear_waze true`
  (a fresh ReportManager, as after a service restart, and an empty Waze reader): the event must reach
  mock Waze and finish as `dry_run_cancelled`.
* C5 the two accessibility settings are back to their values from before the stage (also after a
  failure). The detector is started in video mode on a clip that does not exist, so the pipeline (and the
  injection hook) runs with no real frames.

Note: once mock Waze jumps to the front, the detector goes to the background. If the detector
pauses video playback when it is not visible, run it as a foreground service (its owner's call).

## Camera path instead of video mode (webcam0 + OBS)

The detector's in-app video mode is the easy way. To test the real camera code path instead:

1. Install OBS Studio, add a *Media Source* with `assets\videos\dashcam_urban_1080p.mp4`
   (loop on), and click **Start Virtual Camera**.
2. Recreate or start the AVD with the webcam as back camera:
   `create_avd.ps1 -Webcam -Force` (permanent) or `start_emulator.ps1 -Webcam` (this run only;
   passes `-camera-back webcam0`). Check the name with
   `%ANDROID_HOME%\emulator\emulator.exe -webcam-list`; if OBS is not `webcam0`, use the
   name it shows (e.g. `webcam1`) in `config.ini` (`hw.camera.back`).
3. Start the detector in camera mode; the emulator back camera now shows the OBS clip.

The emulator only enumerates webcams at start-up, so start the OBS virtual camera first.
The default `virtualscene` camera is a 3D room, useful only to check the camera opens.

## Troubleshooting

* `adb devices` shows `emulator-5554 offline`: wait, or `adb -s emulator-5554 emu kill` and start again with `-ColdBoot`.
* Stuck at boot with `-gpu auto`: try `-Gpu swiftshader_indirect` (software, slower).
* `uiautomator dump` fails while an animation runs: A1 retries a few times.
* `obstructed by window '...'` in a failed check: something else had focus (a dialog, the
  launcher, another app). Look at the screenshot of that step in the run folder.
* Reports file: `adb -s emulator-5554 shell cat /sdcard/Android/data/com.wazeomatic.mockwaze/files/reports.jsonl`.

## Real Waze on the emulator (pixel9pro_play)

`pixel9pro_test` uses the plain Google APIs image, which has no Play Store, so the real Waze
app cannot be installed on it. A second profile, `pixel9pro_play`, uses the Google Play image:

```powershell
powershell -ExecutionPolicy Bypass -File tools\emulator\create_avd.ps1 -Name pixel9pro_play -Image "system-images;android-35;google_apis_playstore;x86_64"
powershell -ExecutionPolicy Bypass -File tools\emulator\start_emulator.ps1 -Avd pixel9pro_play -Port 5556 -Wait
```

Then, in the emulator window, open the Play Store, sign in with a Google account and install
Waze. Nothing in this repo automates that step on purpose (it is a personal account).

**Known limit (checked 2026-11-05, Waze 5.26.5, API 35 Play image):** Waze installs and starts,
but both "Get started" and "Log in" request a Play Integrity token first and the emulator is not
a certified device, so the verdict has no device integrity and Waze refuses to continue
("Something went wrong. Try again later." on the guest path, a network error on the login path).
The emulator's network itself is fine (ping, DNS and the Play Store all work), so there is nothing
to fix on that side, and we do not try to work around the integrity check. Use the Play emulator
for look-and-feel comparisons of the installed app's resources only; real-app dry runs and
reference screenshots of the report sheets have to come from a real phone:

```powershell
adb -s <phone serial> exec-out screencap -p > runs\waze_ref\hazard_sheet.png
```

Once Waze runs somewhere (real phone):

* The real app is the reference for the mock: compare screenshots of the report sheets and
  adjust `WazeMenu.kt` / `Ui.kt` if a label, tile order or position differs.
* Macro rehearsals against the real app stay in dry-run: tap through to the confirm sheet and
  Cancel. Never let a test tap Report or wait out the auto-send countdown in real Waze, as that
  publishes a report to other drivers. Automated full sends stay on mock Waze (`pixel9pro_test`).
* The two emulators can run side by side (serials `emulator-5554` and `emulator-5556`); pass
  `--serial emulator-5556` to `run_e2e.py --stage b` to replay a clip into the detector app on
  the Play device while Waze is in front.
