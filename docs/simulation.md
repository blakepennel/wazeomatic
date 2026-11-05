# Simulation and testing

We test without driving and without sending real Waze reports. There are three levels:

| Level | What runs | Waze | Good for |
|-------|-----------|------|----------|
| Desktop simulator | `wazeomatic` on a PC with a dashcam clip | none | Model and Report Manager tuning, fast iteration |
| Emulator suite | Detector app + MacroDroid + Mock Waze on an emulator | Mock Waze | End-to-end automation, repeatable |
| Real phone, dry-run | Detector app + MacroDroid + real Waze | real, never sent | Checking real menu labels and tap positions |

## Why real Waze cannot be used for automated tests

Waze has no sandbox, test server or developer mode for reports. A report sent from the app is live and shown to other drivers, and repeated false reports can get an account rate-limited. So automation stops one tap before send on real Waze (dry-run), and anything that must actually "send" goes to the mock app.

## Desktop simulator

```powershell
python tools/fetch_test_videos.py                      # downloads the two clips and converts them to 720p MP4 in assets/videos/
python -m wazeomatic.run_video assets/videos/dashcam_urban_1080p.mp4 --sink stdout
```

| Option | Meaning |
|--------|---------|
| `--sink stdout` | Print each event as JSON (default); `--sink jsonl:<path>` appends to a file |
| `--sink tcp:<host>:8765` | Send NDJSON to the phone app's TCP server, same path as the Pi (`--event-source pi`) |
| `--sink adb[:<serial>]` | Fire `am broadcast` on the connected device or emulator for each event; `--adb-dry-run` sets `dry_run=true`, `--adb-package` targets one app |
| `--step 3` | Infer on every 3rd frame (30 fps clip -> 10 inferences/s, the phone's budget) |
| `--conf`, `--report-conf`, `--min-hits`, `--window`, `--cooldown`, `--global-cooldown` | Detector and Report Manager thresholds (defaults match the phone app) |
| `--stopped-vehicle` | Enable the experimental car-on-shoulder heuristic |
| `--camera-sim [--sim-res 1280x720]` | Camera-style degradation, see below |
| `--show`, `--record <mp4>`, `--stats-json <file>` | Live window, annotated video, latency/FPS summary |
| `--no-coco`, `--no-hazard`, `--threads 4` | Pick models; ORT thread count (4 approximates a phone's big cores) |

`run_video.py` runs the hazard and COCO models on each sampled frame, feeds the Report Manager, and prints a `[stats]` line with mean/p50/p90 inference latency and pipeline FPS so results can go into [model-research.md](model-research.md#measured-results).

### Camera-style degradation

The test clips come from real dashcams, but a phone behind a windscreen produces worse frames. The simulator can degrade each frame before inference with `--camera-sim` (`CameraSim` in `wazeomatic/sources.py`):

| Effect | Purpose |
|--------|---------|
| Resize to 1280x720 | Phone analysis resolution |
| Gaussian blur (sigma 0.8) and sensor noise | Vibration, focus, low light |
| JPEG re-encode at quality 70 | Compression artefacts |
| Slow random-walk exposure gain (+-12 %) | Auto-exposure hunting, tunnels, glare |

Run each clip clean and degraded. If event counts change a lot, the thresholds in `ReportConfig` are too tight.

## Emulator suite

Details and scripts: [tools/emulator/README.md](../tools/emulator/README.md).

- AVD: Pixel 9 Pro device profile, API 35 `google_apis` x86_64 system image, hardware acceleration through WHPX on Windows.
- Installed: detector app, Mock Waze, MacroDroid (with the macro imported, `DRY_RUN` may be false here because Waze is not installed).
- Video into the emulator, two options:
  1. **Video-source mode (deterministic):** `adb push` an MP4 to the emulator and select it in the detector app. Same frames every run; use this for automated assertions.
  2. **Emulator webcam:** play the clip in OBS and start OBS Virtual Camera, then set the AVD back camera to that webcam. Exercises the real camera path, but timing varies.
- Assertion: Mock Waze logs every report it receives (type, time). The script compares that log against the expected hazards for the clip.

Emulator latency is not representative of the phone; only use it for function.

## Mock Waze app

`android/mockwaze` copies the parts of Waze the macro touches: a map screen with a Report button, the category grid, the hazard sub-menu and the send button, using the same labels and approximate positions. It records each completed report and shows it on screen and in logcat. Because it is our app, the macro can run with `DRY_RUN` off against it and tests can check that the right item was chosen.

When Waze changes its menu, update the mock app first, then the macro, then check on a real phone in dry-run.

## Dry-run on a real phone

1. Set the MacroDroid global variable `DRY_RUN` to `true` and keep the detector app's dry-run setting on.
2. Open Waze (parked, or as a passenger).
3. Send a test event: `adb shell am broadcast -a com.wazeomatic.HAZARD_DETECTED --es hazard_type HAZARD_ON_ROAD_CONSTRUCTION --es dry_run true`.
4. Watch the macro open the report menu, select the item, take a screenshot and back out. Check the screenshot in the gallery shows the right item selected.
5. Repeat for each sprint 1 `hazard_type`. Record the Waze version tested.

Never turn `DRY_RUN` off while testing. See [pipeline-to-macro.md](pipeline-to-macro.md#dry_run-semantics).

## Test videos

| Clip | Resolution | Length | Licence |
|------|------------|--------|---------|
| [Dashcam Recording (urban).ogv](https://commons.wikimedia.org/wiki/File:Dashcam_Recording_(urban).ogv) | 1920x1080, 30 fps | 120 s | CC0 1.0 |
| [Dashcam recording example - Panorama II BlackBox Dashcam daylight.mp4.ogv](https://commons.wikimedia.org/wiki/File:Dashcam_recording_example_-_Panorama_II_BlackBox_Dashcam_daylight.mp4.ogv) | 1280x720, 30 fps | 120 s | CC0 1.0 |

Both are public domain, so they can be used, cut and shared freely. They are not committed; `tools/fetch_test_videos.py` downloads them. Neither clip was filmed for this project, so they contain few potholes or cones; hazard-heavy footage recorded by the team should be added to `assets/videos/` (also not committed).
