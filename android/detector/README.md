# Wazeomatic detector (Android)

Kotlin app (package `com.wazeomatic.detector`, minSdk 29, target 35). Runs the YOLO ONNX
models with ONNX Runtime (XNNPACK EP, CPU fallback) inside a foreground service, applies the
same reporting rules as `wazeomatic/report_manager.py` and broadcasts
`com.wazeomatic.HAZARD_DETECTED` for the MacroDroid macro.

## Build

Needs JDK 17+ and the Android SDK (platform 35, build-tools 35.0.0).

```powershell
$env:JAVA_HOME = 'C:\Program Files\OpenLogic\jdk-21.0.7.6-hotspot'
$env:ANDROID_HOME = 'F:\dev\android-sdk'
$env:GRADLE_USER_HOME = 'F:\dev\gradle\home'
cd android\detector
.\gradlew.bat assembleDebug          # -> app\build\outputs\apk\debug\app-debug.apk
.\gradlew.bat testDebugUnitTest      # JVM tests: decoder/NMS port + ReportManager
```

## Models

`preBuild` runs the `syncModels` task, which copies from `<repo>/models` into
`app/src/main/assets` (not committed):

| repo file | asset |
|-----------|-------|
| `hazard_yolo11n.onnx` + `hazard_yolo11n.labels.txt` | `hazard.onnx` + `hazard_labels.txt` |
| `coco_yolo11n.onnx` + `coco_yolo11n.labels.txt` | `coco.onnx` + `coco_labels.txt` |

Other files: `.\gradlew.bat assembleDebug -PcocoModel=coco_yolo11n.int8.onnx -PhazardModel=...`
(the labels file is looked up as `<name>.labels.txt`). Labels are one per line; if the file is
missing the class names are read from the Ultralytics `names` metadata in the model. The input
size comes from the model's input shape. Both models run when present and their detections are
merged. With no model the app still runs and shows "model missing".

## Install and launch

```powershell
adb install -r -g app\build\outputs\apk\debug\app-debug.apk   # -g grants camera/notification permissions

# camera (default)
adb shell am start -n com.wazeomatic.detector/.MainActivity --es source camera

# video: push an h264 mp4 into the app's external files dir first
adb push clip.mp4 /sdcard/Android/data/com.wazeomatic.detector/files/test.mp4
adb shell am start -n com.wazeomatic.detector/.MainActivity --es source video --es video test.mp4 --ez dry_run true

# tcp only (events from the Pi / desktop simulator)
adb shell am start -n com.wazeomatic.detector/.MainActivity --es source tcp
adb forward tcp:8765 tcp:8765    # emulator / USB: then send JSON lines to localhost:8765

# stop
adb shell am start -n com.wazeomatic.detector/.MainActivity --ez stop true
```

Passing `source` (or `--ez autostart true`) starts the detector without taps. Extras are saved
into the settings; `--ez reset_settings true` restores defaults first.

| extra | default | meaning |
|-------|---------|---------|
| `source` | camera | `camera`, `video` or `tcp` |
| `video` | test.mp4 | file in the app external files dir (or absolute path) |
| `frame_step` / `loop` | 5 / true | video: infer every Nth frame, restart at end |
| `max_fps` | 6 | camera: inference rate cap (extra frames are dropped) |
| `conf` | 0.45 | report min confidence (detector box threshold is min(0.30, conf)) |
| `cooldown` / `global_cooldown` | 60 / 5 | seconds |
| `min_hits` / `window` | 3 / 10 | persistence rule |
| `stopped_vehicle` | false | experimental parked-car heuristic |
| `class_map` | "" | extra mappings for testing, e.g. `car=HAZARD_ON_ROAD_OBJECT` |
| `zoom_crop` | "" | two-pass zoom: `--ez zoom_crop true` = default crop `0.25,0.35,0.75,0.75`, `--es zoom_crop x1,y1,x2,y2` = that region (frame fractions); empty or malformed = off |
| `dry_run` | true | passed to the macro |
| `target_package` | "" | empty = implicit broadcast, e.g. `com.arlosoft.macrodroid` |
| `tcp` / `tcp_port` | true / 8765 | JSON-lines listener, also runs next to camera/video |

## Outputs

Each event is
- broadcast as `com.wazeomatic.HAZARD_DETECTED` with `FLAG_INCLUDE_STOPPED_PACKAGES`,
- logged as one JSON line under logcat tag `WazeomaticEvent` (same shape as `HazardEvent.to_json()`),
- appended to `/sdcard/Android/data/com.wazeomatic.detector/files/events.jsonl`.

Per-frame latency goes to logcat tag `WazeomaticPerf` and is shown on screen.

### Two-pass zoom

`zoom_crop` runs every model a second time on a crop of the road ahead (same as `--zoom-crop`
in `wazeomatic/run_video.py`): with a 1280x720 frame and a 640 px model the default crop is
640x288 px, so a distant 20 px cone is 40 px to the network. Crop detections are mapped back
to frame coordinates and merged with the full-frame ones by per-class NMS (IoU 0.5, the higher
confidence wins). The crop is skipped when it would be under 32 px in either dimension.
Inference cost roughly doubles; the second pass shows up as `hazard_zoom` / `coco_zoom` in the
`WazeomaticPerf` line and on screen.

```powershell
adb shell am start -n com.wazeomatic.detector/.MainActivity --es source camera --ez zoom_crop true
adb shell am start -n com.wazeomatic.detector/.MainActivity --es source video --es zoom_crop 0.2,0.3,0.8,0.7
adb shell am start -n com.wazeomatic.detector/.MainActivity --es zoom_crop ""    # off again
```

```powershell
adb logcat -s WazeomaticEvent
adb shell cat /sdcard/Android/data/com.wazeomatic.detector/files/events.jsonl
```

## MacroDroid pairing

Trigger: *Intent Received*, action `com.wazeomatic.HAZARD_DETECTED`. All extras are strings:
`hazard_type`, `confidence`, `ts`, `source`, `waze_category`, `waze_subcategory`, `dry_run`
("true"/"false"). The macro switches on `hazard_type` (keys from `wazeomatic/schema.py`). If
implicit broadcasts don't reach MacroDroid on a phone, set `target_package` to
`com.arlosoft.macrodroid`.
