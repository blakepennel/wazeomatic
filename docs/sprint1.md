# Sprint 1

## Goals

The team set four goals for sprint 1:

1. Document the work done so far.
2. Write a setup guide for the MacroDroid macro.
3. Describe how the CV pipeline hands detections to the macro.
4. Research models that fit the phone (and later the Pi), including memory budgets.

## Delivered

| Goal | Area | Where | Status |
|------|------|-------|--------|
| 1 | Project overview, repo layout, quick start | [README.md](../README.md) | done |
| 1 | Architecture and diagrams (component, sequence, class) | [architecture.md](architecture.md), [diagrams/](diagrams/) | done |
| 1 | User stories with estimates | [user-stories.md](user-stories.md) | done |
| 2 | Macro setup guide | [android_setup/macro_setup_guide.md](../android_setup/macro_setup_guide.md) | done (teammate) |
| 2 | Intent Received trigger, `DRY_RUN`, permissions | [pipeline-to-macro.md](pipeline-to-macro.md#macrodroid-setup) | documented; needs to be applied to the macro |
| 3 | Event JSON, Intent contract, webhook and TCP fallbacks, cooldowns | [pipeline-to-macro.md](pipeline-to-macro.md) | done |
| 3 | Shared hazard schema | `wazeomatic/schema.py` | done |
| 3 | Report Manager (persistence, cooldowns, stopped-vehicle heuristic) | `wazeomatic/report_manager.py` | done |
| 4 | Model, runtime and dataset comparison; memory budgets | [model-research.md](model-research.md) | done |
| 4 | ONNX detector (desktop) | `wazeomatic/detector.py` | done |
| 4 | Dataset merge, training, export, quantisation, benchmarks | `tools/` | done; merged dataset `wazeomatic_hazards_v3` = 45,855 train / 14,597 val files, 6 classes (the first build had 5,322 / 2,259 images and 4 classes) |
| 4 | Measured latency and accuracy | [model-research.md](model-research.md#measured-results) | desktop latency and validation mAP recorded; Pixel 9 Pro numbers pending a phone session |
| extra | Desktop simulator with camera degradation | `wazeomatic/run_video.py`, [simulation.md](simulation.md) | done; 62 unit tests in `tests/` |
| extra | Detector app (camera / video / TCP sources, ONNX Runtime XNNPACK, foreground service, broadcast) | `android/detector/` | builds; 54 JVM unit tests in `app/src/test`; video mode verified on the emulator |
| extra | Mock Waze app and emulator suite (Pixel 9 Pro AVD, API 35) | `android/mockwaze/`, [tools/emulator/](../tools/emulator/README.md) | done; the end-to-end suite ([tools/emulator/README.md](../tools/emulator/README.md)) covers Stage A (mock macro: send once, 60 s rate limit, dry-run cancel, unknown type ignored) and Stage B (detector app in video mode -> broadcast -> mock Waze report with matching hazard type) |

| extra | Model fusion (weighted boxes fusion across hazard models, parallel members), spatial tracking and per-label report floors, prefetching source | `wazeomatic/fusion.py`, `build.py`, `report_manager.py`; `BoxFusion.kt`, `ReportManager.kt` in the app; [model-research.md](model-research.md#ensembling-zoom-and-tracking-pipeline-pass-2026-11-06) | done; scored with `tools/eval_events.py`, same flags as `run_video.py`; fusion and tracking are available but off by default; shipped operating point: one hazard model (v3 nano), zoom pass on, report floor 0.40 with a pothole floor of 0.45 |
| extra | Mock Waze matched to the 2024 Waze report flow (tile size, spacing, colours, auto-send countdown measured from reference screenshots) | `android/mockwaze/` | done; emulator suite re-run against it |

## Sprint 2

The sprint 2 plan and backlog are in [sprint2.md](sprint2.md).

## History

- Done 2026-11-09: audit pass. Desktop/phone parity fixes (pothole floor wired through both pipelines, a label floor only raises the bar, zoom pass only on the hazard model, global cooldown aligned with the macro's one report per minute) and safe defaults (the adb sink sends dry run unless `--adb-live`). Failure visibility (frame-source errors and a missing hazard model stop the run, stalled and battery state shown on the phone) and reproducibility (dataset fetcher, configurable dataset path, export naming).
- Done 2026-11-06 (afternoon): data expansion. ROADWork (CMU), Argoverse 2 front-camera logs with projected cuboids and a Roboflow pothole set added (`tools/fetch_argoverse_hazards.py`, `tools/argoverse_to_yolo.py`, `tools/prepare_datasets.py`); six-class schema (barricade, worker added); v3 nano and small models trained on 45.9k images; v3 nano shipped on both pipelines with the report floor at 0.40 and the pothole floor at 0.45; UK dashcam and Argoverse clips added to the real-footage check (zero false reports in 12 minutes, one correct Construction event on a real lane closure). Numbers in [model-research.md](model-research.md#data-expansion-and-the-v3-models-2026-11-06-afternoon). Still open: Pixel 9 Pro latency of the small model.
- Done 2026-11-06: cone/delineator recall. StarLine tiled at native resolution, 640 px training, full RDD2022 potholes added, two-pass zoom inference in Python and the Android app, event-level evaluation (`tools/eval_events.py`); numbers in [model-research.md](model-research.md#small-object-pass-tiled-data-640-px-two-pass-zoom). Shipped model: run `hazard640c` with a pothole-only report floor of 0.55 (zero false events on the hazard-free clips). Still open: Pixel 9 Pro latency of 640 + zoom; real-footage check of the pothole floor.
- Done 2026-11-06: live camera path verified on the desktop (OBS virtual camera) and on the emulator (CameraX, 4.4 FPS with both models and the zoom pass on a loaded host). Still open: the same on the phone with real footage.
