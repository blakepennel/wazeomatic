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
| `vehicle_cls.onnx` + `vehicle_cls.labels.txt` (optional) | `vehicle_cls.onnx` + `vehicle_cls_labels.txt` |
| `vehicle_cls2.onnx` + `vehicle_cls2.labels.txt` (optional, averaged with the first) | `vehicle_cls2.onnx` + `vehicle_cls2_labels.txt` |
| `weather_cls.onnx` + `weather_cls.labels.txt` (optional) | `weather_cls.onnx` + `weather_cls_labels.txt` |
| `surface_cls.onnx` + `surface_cls.labels.txt` (optional) | `surface_cls.onnx` + `surface_cls_labels.txt` |

Other files: `.\gradlew.bat assembleDebug -PcocoModel=coco_yolo11n.int8.onnx -PhazardModel=...`
(the labels file is looked up as `<name>.labels.txt`). Labels are one per line; if the file is
missing the class names are read from the Ultralytics `names` metadata in the model. The input
size comes from the model's input shape. Both models run when present and their detections are
merged. With no model the app still runs and shows "model missing".

The three classifiers (`tools/train_classifier.py`) are copied only as a model + labels pair; a
missing one is left out of the build with a note in the Gradle log (and a stale asset from an earlier
build is deleted), and the app switches that part off and says so on the status line
(`vehicle: missing`, `weather: missing`, `surface: missing`). Other file names:
`-PvehicleModel=...`, `-PweatherModel=...`, `-PsurfaceModel=...` (empty = leave it out).

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

### Using the app

The main screen (dark Material 3, laid out for portrait and for a landscape car mount) shows the analysed
frames with their boxes, a state pill (stopped / starting / running / stalled / problem), the frame rate,
the last report, the model and health line from the service (warnings in amber; tap it to expand) and the
reports of this session. The big button starts and stops the detector. The amber **Dry run** button in the
top bar switches dry run on in one tap; switching it off (real Waze reports) asks first and turns the
button red (**LIVE**). Changing the source, the clip or dry run while the detector runs restarts it.

The **Video** source shows the selected clip. **Choose** lists the clips in the app's files folder
(thumbnail, size, length; a clip can be deleted there) and **Add a clip from the phone** opens the system
file picker. The picked clip is copied into the folder, because the video source reads plain files, and
then selected. Sharing a video to Wazeomatic, or opening one with it, does the same.

**Settings** (the gear) groups every launch extra under Source, Reporting, Detection, Scene and Advanced,
with switches, sliders and dropdowns. Each change is written through the same `Settings.withExtras` path
that `am start --es` uses. A value set over adb that is outside a slider's range, or not in a dropdown's
list, is shown as it is and kept unless you change it.

### Mounting the phone

Mount the phone in landscape, like a dashcam: the models are trained on landscape frames and the
zoom crop assumes a wide picture. With `orientation` on `auto` the frames are turned upright from
the accelerometer, whichever way the phone is held and whether or not auto-rotate is on (a flat
phone keeps the last orientation; before the first reading landscape is assumed); a portrait
mount still runs but the status line warns about portrait frames. Set the app's battery usage to
Unrestricted (the app shows a hint with a button to the system screen while it is not): the
service holds a wake lock while it runs, the status line and the notification show
`battery: restricted` / `battery: unrestricted`, and `stalled Ns` appears when no frame has been
processed for more than 5 s.

With the `video` or `tcp` source the service runs as a `dataSync` foreground service, which
Android 15 limits to about 6 hours a day; when the limit is reached the detector stops and the
status line says so (press Start to run again). The camera source has no such limit.

Passing `source` (or `--ez autostart true`) starts the detector without taps. Extras are saved
into the settings; `--ez reset_settings true` restores defaults first.

| extra | default | meaning |
|-------|---------|---------|
| `source` | camera | `camera`, `video` or `tcp` |
| `video` | test.mp4 | file in the app external files dir (or absolute path) |
| `frame_step` / `loop` | 5 / true | video: infer every Nth frame, restart at end |
| `max_fps` | 6 | camera: inference rate cap (extra frames are dropped) |
| `conf` | 0.40 | report min confidence (detector box threshold is min(0.35, conf), the desktop default) |
| `cooldown` / `global_cooldown` | 60 / 60 | seconds per hazard type / between any two events; the second matches the macro, which files at most one report a minute, so a held-back hazard is counted here (`heldBack`) instead of vanishing there |
| `min_hits` / `window` | 3 / 10 | persistence rule; Police needs 2 hits whatever `min_hits` says (`DEFAULT_MIN_HITS_BY_HAZARD` in `ReportManager.kt`) |
| `shoulder_cars` | false | Hazard > Car on shoulder: a car, truck or bus standing beside the road on our side while we drive above 50 km/h (`ShoulderCars.kt`) |
| `shoulder_camera_speed` | true | without a GPS speed, the camera's own speed estimate may open the 50 km/h condition (it reads about 0.87 of the true speed) |
| `class_map` | "" | extra mappings for testing, e.g. `car=HAZARD_ON_ROAD_OBJECT` |
| `zoom_crop` | true | two-pass zoom (on by default since the 2026-11-06 evaluation: +6 points of event recall, no false events on the clips with the adaptive hood zone): `--ez zoom_crop true` = default crop `0.25,0.35,0.75,0.75`, `--es zoom_crop x1,y1,x2,y2` = that region (frame fractions); empty or malformed = off |
| `rect_input` | true | a model exported with an open input shape (`export_onnx.py --dynamic`, bundled by default when present) runs at the frame's own aspect: 640x384 for a 16:9 frame and 640x288 for the zoom crop instead of 640x640, the same pixels without the grey bands (`--rect` on the desktop). No effect on a square export |
| `ort_threads` | 0 | XNNPACK threads per model session; 0 = the cores divided by the passes that run at once, at most 4 (two on an 8-core phone with the zoom pass on) |
| `ort_backend` | xnnpack | `xnnpack`, `cpu`, or `nnapi` with `_fp16` / `_cpu` / `_nchw` added; per model as `hazard=nnapi,coco=xnnpack`. NNAPI is experimental: on the OnePlus N10 the Qualcomm drivers reject both models unless NNAPI may use its own CPU device, and then it is slower (docs/model-research.md) |
| `max_det` | 40 | boxes per detector pass, best first (0 = no cap), as `--max-det` on the desktop |
| `max_per_class` | 12 | boxes of one class per detector pass, best first (0 = no cap), as `--max-per-class` |
| `zoom_every` | 0 | zoom pass on every n-th frame; 0 = auto: every frame while frames with it fit the `max_fps` interval, every other frame while they don't |
| `zoom_coco` | false | also run the zoom pass on the COCO model (`--zoom-coco` on the desktop); off by default, the second pass is for the hazard model(s) |
| `orientation` | auto | camera: how the phone is mounted. `auto` follows the accelerometer (independent of the system auto-rotate setting), `landscape` (top edge to the left), `landscape_reverse` (top edge to the right) and `portrait` pin it |
| `ensemble` | true | also run `hazard2.onnx` when the build bundled one (`syncModels` copies `-PhazardModel2`; the default is none since the v3 model, `-PhazardModel2=hazard_yolo11s_v3.onnx` adds the v3 small) and fuse it with `hazard.onnx`; no second asset = single model |
| `fusion` | wbf | `wbf` = weighted boxes fusion (boxes averaged, confidence = weighted mean over all members, so a box one member missed is scored down), `nms` = keep the best box |
| `ensemble_weights` | "" | per-member weights, e.g. `3,1` (default equal) |
| `class_conf` | "" | per-label confidence floors after fusion, e.g. `cone=0.5,delineator=0.5` |
| `report_class_conf` | pothole=0.55,police_vehicle=0.85,worker=0.6,debris=0.5 | per-label hit floors inside the Report Manager; the higher of the floor and `conf` applies, so a floor can raise the bar for a label but never lower it (same as `_min_conf` in `wazeomatic/report_manager.py`); both floors are the shipped default on the phone and the desktop pipeline. At the police floor the vehicle classifier calls two dashcam civilian vehicles in 5,784 police in a single frame (see `docs/waze-coverage.md`) |
| `hood_frac` | 0.10 | minimum hood / dashboard zone at the bottom of the frame (fraction of the height); boxes whose bottom edge is inside it are ignored, except police / ambulance / fire boxes, which are ignored only when they lie wholly inside it (a police car close beside us reaches the bottom of the frame) |
| `auto_hood` | true | grow the hood zone from the picture itself (`HoodEstimator.kt`): the dashboard keeps sharp edges in a running average of the frames while the road blurs away; on the test clips this finds a 22 % dashboard that the fixed 10 % missed (an air vent was reported as a pothole) |
| `track_iou` | 0 | > 0 turns on spatial tracking: the hits behind a report must come from boxes that overlap frame to frame (grown by `track_grow`, default 1.5) or sit within 1.5 box sizes; three unrelated flickers no longer add up to a report |
| `dry_run` | true | passed to the macro |
| `target_package` | "" | empty = implicit broadcast, e.g. `com.arlosoft.macrodroid` |
| `tcp` / `tcp_port` | true / 8765 | JSON-lines listener, also runs next to camera/video |
| `waze_reader` | true | before an event is sent, ask the Waze reader whether Waze already shows that report nearby; if so it is counted as a confirmation and not sent (same rule as `waze=` in `wazeomatic/report_manager.py`). Needs the accessibility service below |
| `vehicle_classifier` / `vehicle_crops` | true / 4 | second stage on the COCO car / truck / bus boxes (needs `vehicle_cls.onnx` and the COCO model): police, ambulance, fire engine (`--no-vehicle` / `--vehicle-crops` on the desktop) |
| `weather` | true | weather (whole frame) and road-surface (patch of road ahead) classifiers, twice a second (`--no-weather`) |
| `scene_features` | traffic,crash,lanes,signals,weather | scene rules to run, a comma list like `--scene-features`; `none` switches the scene layer off (`--no-scene`), an empty list keeps only ego motion and the emergency-light check |
| `police_other_side` | false | a police car far on the opposite side becomes Police > Other side (`--police-other-side`) |
| `police_stopped_only` | true | Police only for a police car read as standing still (`PoliceMotion`, optical flow); false reports moving ones too (`--police-any-motion`) |
| `drive_side` | right | `left` mirrors the side rules for left-hand traffic (`--drive-side`) |
| `camera_geometry` | 70,1.3,0.5 | horizontal field of view (degrees), mount height (metres) and horizon row (fraction of the height) for the flat-road camera model of the blocked-lane rule and a top-view road-surface model (`--camera-geometry`); a malformed value falls back to the default |
| `use_gps_speed` | false | camera source: feed the GPS speed into the stopped / moving estimate. Only with this on does the app ask for location; the speed is used on the phone and no position is stored or sent |

## Second stage and scene layer

Per frame, in the order of `Pipeline.process` in `wazeomatic/pipeline.py`: detectors, then the vehicle
classifier (`VehicleClassifier.kt`: the largest 4 COCO vehicles with confidence >= 0.35 and both sides
>= 28 px, cropped with `square_crop`'s rule, `police_vehicle` / `ambulance` / `fire_truck` detections for
probabilities >= 0.05), the scene classifiers (`SceneClassifier.kt`, `RoadSurface.kt`; every 0.5 s of
frame time, classes of the surface model prefixed `surface_`), then `SceneAnalyzer.kt` (ego motion,
flashing emergency lights, standstill / heavy traffic, crash, blocked lane, dark signals, weather), then
the hood zone and the Report Manager. The scene rules run on the frame time: the clip position for the
`video` source, `elapsedRealtime` for the camera. The status line shows the ego state (`ego stationary`,
`ego slow`, `ego moving`) and any scene label active on the last frame; per-frame timings go to the
`WazeomaticPerf` log as `vehicle=`, `weather=` and `scene=`. The image maths (`ImageOps.kt`) follows
OpenCV's (grey, HSV, INTER_AREA), checked against cv2 numbers in `ClassifierTest`.

Scene conditions (traffic, weather, crash, blocked lane, dark signal) use the longer per-type
cooldowns of `SCENE_COOLDOWNS` in `wazeomatic/scene.py` (e.g. 900 s for heavy traffic) instead of
`cooldown`, and their whole-frame `scene` detections skip the hood zone.

## Waze reader (read only)

`WazeWatcherService` is an accessibility service that only reads the screen of `com.waze` and
`com.wazeomatic.mockwaze` (window content, no gestures, no key events, at most two reads a second)
and never taps or changes anything. It parses the alert cards with `WazeText.kt` (the Kotlin twin of
`wazeomatic/waze_reader.py`) into a process-wide `WazeAlerts` (TTL 120 s, 1200 m). Switch it on under
Settings > Accessibility > "Wazeomatic Waze reader"; the app shows a hint while it is off. The status
line shows what Waze has ahead and how many events were confirmed instead of sent. Logcat tag
`WazeomaticWaze`: `WAZE_ALERT {json}` per new alert, `CONFIRMED {json}` per suppressed event.

Debug builds also accept a test broadcast that feeds synthetic frames into the running pipeline
(release builds do not register it):

```powershell
adb shell am broadcast -a com.wazeomatic.detector.INJECT_DETECTION -p com.wazeomatic.detector --es label police_vehicle --es confidence 0.9 --ei count 5
# --ez reset true: fresh ReportManager (cooldowns gone); --ez clear_waze true: forget the Waze alerts seen so far
```

## Outputs

Each event is
- broadcast as `com.wazeomatic.HAZARD_DETECTED` with `FLAG_INCLUDE_STOPPED_PACKAGES`,
- logged as one JSON line under logcat tag `WazeomaticEvent` (same shape as `HazardEvent.to_json()`),
- appended to `/sdcard/Android/data/com.wazeomatic.detector/files/events.jsonl`.

Per-frame latency goes to logcat tag `WazeomaticPerf` and is shown on screen. Each line has the model passes
(`hazard`, `hazard_zoom`, `coco`), `wall` (all passes, run in parallel), `infer` (models and fusion), the second
stage (`vehicle`, `weather`, `scene_rules`, `police`, `scene`), `post` (the whole second stage, hood and Report
Manager), `wait` (the frame source blocked on the previous frame's second stage) and `src` (time between frames
spent decoding and converting the next one; near zero once the video source decodes ahead).

A frame goes through two stages on two threads: the caller (camera analyzer or video thread) runs the models,
then hands the frame to a second-stage thread and goes on to the next frame. At most one frame waits, so the
order is kept. The video source also decodes one frame ahead. With the int8 hazard model this took a 16:9 clip on
the OnePlus N10 from 3.1 to 6.3 frames per second ([model-research.md](../../docs/model-research.md#phone-latency-2026-11-23)); a
release build and a second round (camera throttle, police gate, zoom pass when time is short) took it to 7.1, and the
live camera holds its 6 fps cap ([second round](../../docs/model-research.md#phone-speed-second-round-2026-11-23)).

Time the app with a release build: a debug build runs its Kotlin code much slower on the phone.
`./gradlew assembleRelease -PdebugSignedRelease` signs the release build with the local debug key, so it installs over
the debug build (`adb install -r app/build/outputs/apk/release/app-release.apk`). The debug-only parts (the
`INJECT_DETECTION` receiver, models from `files/models/`) are off in it.

The build bundles `models/hazard_yolo11n_dyn.qdq.onnx` (int8, from `tools/quantize_onnx.py --static`, see
[models/README.md](../../models/README.md)) when it exists, else the float `hazard_yolo11n_dyn.onnx`. The COCO
model stays float.

Debug builds load a model from `<external files>/models/<asset name>` instead of the bundled one when that file
exists, so a model can be tried without a rebuild:

```powershell
adb shell mkdir -p /sdcard/Android/data/com.wazeomatic.detector/files/models
adb push models\hazard_yolo11n_dyn.qdq.onnx /sdcard/Android/data/com.wazeomatic.detector/files/models/hazard.onnx
adb shell rm -r /sdcard/Android/data/com.wazeomatic.detector/files/models    # back to the bundled models
```

### Two-pass zoom

All models, and each zoom pass, run on their own threads with their own ONNX Runtime session; the `WazeomaticPerf` log line reports each pass's time plus `wall`, the latency of the whole frame (measured on a phone in [model-research.md](../../docs/model-research.md#phone-latency-2026-11-23)). The hazard members are fused (`BoxFusion.kt`, a port of `wazeomatic/fusion.py`) before the Report Manager sees them.

`zoom_crop` runs the hazard model(s) a second time on a crop of the road ahead (same as
`--zoom-crop` in `wazeomatic/run_video.py`); the COCO model is only zoomed with `zoom_coco`
(`--zoom-coco`), off by default. With a 1280x720 frame and a 640 px model the default crop is
640x288 px, so a distant 20 px cone is 40 px to the network. Crop detections are mapped back
to frame coordinates and merged with the full-frame ones by per-class NMS (IoU 0.5, the higher
confidence wins). The crop is skipped when it would be under 32 px in either dimension.
The second pass runs at the same time as the first and shows up as `hazard_zoom` (and
`coco_zoom` with `zoom_coco`) in the `WazeomaticPerf` line and on screen.

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
