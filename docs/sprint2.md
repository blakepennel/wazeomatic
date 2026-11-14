# Sprint 2 plan

Sprint 1 delivered the detection side and a tested loop against a mock Waze
([sprint1.md](sprint1.md)). Nothing has yet run on the real phone against the real macro. Sprint 2
is about closing that gap and measuring what a real drive does.

## Goal

A dry-run drive: the Pixel 9 Pro in a mount, the detector and the MacroDroid macro running next to
real Waze, the macro stopping at the confirm sheet for hazards that are really there, and numbers
for latency, heat and where the report would have landed.

## Where the sprint 1 acceptance line stands

[model-research.md](model-research.md) set three criteria for sprint 2 before any of this was
built. Measured against the shipped v3 nano model:

| Criterion | State |
|---|---|
| Pothole and cone mAP50 >= 0.5 | Cone 0.51: met. Pothole 0.39: **not met** (box AP50 on the 1,201-image held-out subset, zoom pass on). Pothole event recall is 0.59 at 0.98 precision. |
| No more than one false event per test clip | Met: zero events on five hazard-free clips, twelve minutes. Daylight only. |
| Phone latency under 100 ms per frame | **Unmeasured.** No number exists for the Pixel 9 Pro. |

## Backlog

Owners are proposals to be confirmed at sprint planning. "Needs" lists what blocks the item.
Points use the scale of [user-stories.md](user-stories.md).

### Must have

| # | Item | Done when | Needs | Owner | Pts |
|---|---|---|---|---|---|
| 1 | Macro matches the trigger contract | The macro has the Intent Received trigger, `dry_run_extra`, `DRY_RUN` and the foreground and rate constraints; `tools/macro_selftest.py` shows every branch stopping at the confirm sheet on real Waze; the `.macro` export is committed in `android_setup/` | Phone | Drew | 5 |
| 2 | Detector on the Pixel 9 Pro | App installed with the v3 model; orientation correct in the mount (`orientation` setting); status line shows frames, no "stalled", battery unrestricted; a 30-minute run next to Waze without the service dying | Phone, mount | Blake | 5 |
| 3 | Phone latency and heat | Per-frame latency (p50, p90) for the nano with and without the zoom pass and for the small model, plus battery drop and temperature over 30 minutes, recorded in model-research.md; decision on zoom and frame rate for the phone | Item 2 | Blake | 3 |
| 4 | Dry-run drive | One drive of 20 minutes or more past at least one real work zone, with detector events, macro screenshots and a screen recording kept; every event traced to a macro run or to a counted reason it was not (`heldBack`, not in foreground, rate limit) | Items 1 and 2, a car, a passenger | Both | 8 |
| 5 | Report offset | For each event of item 4: time from first detection to the confirm sheet, vehicle speed, and the distance travelled in between; a stated budget and, if it is exceeded, a decision (shorter window, earlier report, or accept) | Item 4 | Blake | 3 |

### Should have

| # | Item | Done when | Needs | Owner | Pts |
|---|---|---|---|---|---|
| 6 | Own footage in the false-report check | At least 30 minutes of the team's own footage from the mount, including dusk, in `assets/videos/`; the clip check rerun on it; operating point confirmed or changed | Phone, mount | Both | 5 |
| 7 | Night and rain envelope | A measured false-report rate on night and wet-road footage, or a written decision that the app is daylight only with a guard that says so in the status line | Footage | Blake | 5 |
| 8 | Pothole accuracy | Pothole box AP50 >= 0.5 on the held-out subset, or a documented reason it cannot be reached with public data; candidates: own labelled frames from item 6, the small model if item 3 allows | Items 3 and 6 | Blake | 8 |
| 9 | Loss accounting on the phone | The app shows events emitted, held back and broadcast; after a drive the three numbers explain every report and every non-report | Item 2 | Blake | 3 |
| 14 | Every report type on real Waze | The macro taps all 20 `hazard_type` keys of [pipeline-to-macro.md](pipeline-to-macro.md) to the confirm sheet on the installed Waze (`tools/macro_selftest.py`), and the sub-tile labels in [waze-coverage.md](waze-coverage.md) are checked against it | Item 1 | Drew | 5 |
| 15 | Waze reader on real Waze | `WazeWatcherService` reads the alert cards of the installed Waze during item 4; every alert logged as "unknown" is turned into a pattern; at least one hazard counted as confirmed instead of reported | Item 4 | Blake | 3 |
| 16 | Camera geometry in the mount | Mount height, field of view and horizon row of the Pixel in the car measured and set (`--camera-geometry` / settings), so the blocked-lane rule places cones in the right lane | Item 2 | Blake | 2 |
| 17 | Police and crash on own footage | The vehicle classifier and the crash rule run over the drive of item 4 and item 6; every POLICE and CRASH event checked by eye; the police floor confirmed or raised | Items 4 and 6 | Both | 3 |

### Could have

| # | Item | Done when | Needs | Owner | Pts |
|---|---|---|---|---|---|
| 10 | Raspberry Pi camera path | A Pi running the Python pipeline sends events over TCP to the phone app, which re-broadcasts them; the macro is unchanged | Pi, camera | Blake | 8 |
| 11 | Stopped-vehicle heuristic | Evaluated on footage with parked cars; on by default or removed | Footage | Blake | 5 |
| 12 | Continuous integration green | The workflow in `.github/workflows/ci.yml` passes on main and the README badge points at it | - | Blake | 1 |
| 13 | Weights release | A licence answer from the StarLine dataset authors; then either a release of the weights under terms the datasets allow, or a model retrained without the sources that forbid it ([NOTICE.md](../NOTICE.md)) | Reply from the authors | Blake | 3 |

### Not in this sprint

Live reports of any kind (see [responsible-use.md](responsible-use.md)), an iOS version. Police, crash,
traffic, weather and blocked-lane reports were added after this plan was first written; what fires each one
and how well it is measured is in [waze-coverage.md](waze-coverage.md).

## What the report offset is made of

Nobody has measured it; these are the parts, from the code, so that item 5 has something to check
against. The camera is sampled at up to 6 frames a second on the phone, so three hits inside a
ten-frame window take between about 0.5 s and 1.7 s. The macro then taps the report button, the
category and the subtype with waits of about a second each. Waze pins the report at the position of
the phone when the sheet sends, which in a live run is after its own countdown of a few seconds.
At 100 km/h every second is 28 m, so the pin is plausibly 100 m or more past where the hazard was
first seen, plus however far ahead it was. The event JSON has `lat` and `lon` fields that are not
filled in; filling them at first detection is the obvious first step if the offset matters.

## Risks

| Risk | Effect | Mitigation |
|---|---|---|
| The macro's coordinate taps do not match the installed Waze version | Item 1 slips, and item 4 with it | Self-test on the first day; record the Waze version in the macro guide |
| The phone throttles with Waze and two models running | Latency over budget | Item 3 early; frame rate, the zoom pass and the COCO model are all switchable from settings |
| One person holds the phone, the macro and the car | Items 1, 2 and 4 serialise | Agree the drive date at planning; everything not marked "Phone" can proceed in parallel |
| Only daylight data | A demo at dusk misfires | Item 7, and the demo is scheduled in daylight |

## Definition of done

For every item: the result is written down in the doc the table names, with the numbers; tests pass
on both pipelines; nothing was filed to Waze.
