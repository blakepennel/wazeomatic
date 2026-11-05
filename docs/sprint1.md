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
| 4 | Dataset merge, training, export, quantisation, benchmarks | `tools/` | export done; others in progress |
| 4 | Measured latency and accuracy | [model-research.md](model-research.md#measured-results) | TBD |
| extra | Desktop simulator with camera degradation | `wazeomatic/run_video.py`, [simulation.md](simulation.md) | in progress |
| extra | Detector app | `android/detector/` | in progress |
| extra | Mock Waze app and emulator suite | `android/mockwaze/`, [tools/emulator/](../tools/emulator/README.md) | in progress |

Update the status column at the sprint review.

## Open items for sprint 2

- Fill in the measured results table (desktop, emulator, Pixel 9 Pro) and decide 320 vs 416 and fp32 vs int8 for the phone.
- Merge the Roadwork_Cones dataset (after confirming its licence) to improve `barrel` and `construction_sign`.
- Apply the Intent Received trigger and `DRY_RUN` to the macro; re-check tap targets against the current Waze menu labels (Construction, Car on shoulder, Pothole, Object directly under Hazard).
- Run the full emulator suite with Mock Waze on both test clips and record false/missed events.
- Real-phone dry-run for every sprint 1 hazard type; note the Waze version.
- Tune the stopped-vehicle heuristic on footage with parked cars before enabling it.
- Record team dashcam footage with real potholes and work zones.
- Pi variant: run the Python pipeline on a Pi 4/5 and connect it to the phone over TCP 8765.
- Police and crash classes stay deferred until a usable dataset is found.
- Regenerate the PNG/SVG diagram renders from the `.mmd` files.
