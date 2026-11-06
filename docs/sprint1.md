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
| 4 | Dataset merge, training, export, quantisation, benchmarks | `tools/` | done; merged dataset = 5,322 train / 2,259 val images, 4 classes |
| 4 | Measured latency and accuracy | [model-research.md](model-research.md#measured-results) | desktop latency and validation mAP recorded; Pixel 9 Pro numbers pending a phone session |
| extra | Desktop simulator with camera degradation | `wazeomatic/run_video.py`, [simulation.md](simulation.md) | done; 18 unit tests in `tests/` |
| extra | Detector app (camera / video / TCP sources, ONNX Runtime XNNPACK, foreground service, broadcast) | `android/detector/` | builds; 19 JVM tests pass; video mode verified on the emulator |
| extra | Mock Waze app and emulator suite (Pixel 9 Pro AVD, API 35) | `android/mockwaze/`, [tools/emulator/](../tools/emulator/README.md) | done; `run_e2e.py` passes all 18 checks: Stage A (mock macro: send once, 60 s rate limit, dry-run cancel, unknown type ignored) and Stage B (detector app in video mode -> broadcast -> mock Waze report with matching hazard type) |

Update the status column at the sprint review.

## Open items for sprint 2

- Fill in the measured results table (desktop, emulator, Pixel 9 Pro) and decide 320 vs 416 and fp32 vs int8 for the phone.
- Confirm the Roadwork_Cones dataset licence with its authors before any public release of the weights.
- Apply the Intent Received trigger and `DRY_RUN` to the macro; re-check tap targets against the current Waze menu labels (Construction, Car on shoulder, Pothole, Object directly under Hazard).
- Run the full emulator suite with Mock Waze on both test clips and record false/missed events.
- Real-phone dry-run for every sprint 1 hazard type; note the Waze version.
- Done 2026-11-06: cone/delineator recall. StarLine tiled at native resolution, 640 px training, full RDD2022 potholes added, two-pass zoom inference in Python and the Android app, event-level evaluation (`tools/eval_events.py`); numbers in [model-research.md](model-research.md#small-object-pass-tiled-data-640-px-two-pass-zoom). Shipped model: run `hazard640c` with a pothole-only report floor of 0.55 (zero false events on the hazard-free clips). Still open: Pixel 9 Pro latency of 640 + zoom; real-footage check of the pothole floor.
- Measure latency on the Pixel 9 Pro and decide fp32 vs int8 (int8 is slower than fp32 on the desktop CPU; the phone's ARM cores may differ).
- Live camera path verified on the desktop (OBS virtual camera) and on the emulator (CameraX, 4.4 FPS with both models and the zoom pass on a loaded host); still to do on the phone with real footage.
- Tune the stopped-vehicle heuristic on footage with parked cars before enabling it.
- Record team dashcam footage with real potholes and work zones.
- Pi variant: run the Python pipeline on a Pi 4/5 and connect it to the phone over TCP 8765.
- Police and crash classes stay deferred until a usable dataset is found.
