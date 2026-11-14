# Waze report coverage

Which Waze reports Wazeomatic can file, what it looks at for each one, and how well that is measured. The
Waze menu is the Android one after the 2024 redesign: the top-level tiles and the Hazard subtypes are from
[Waze Help](https://support.google.com/waze/answer/13739290), the Blocked lane options from
[Report a blocked lane](https://support.google.com/waze/answer/13811250); the Traffic, Police, Crash and Bad
weather sub-tiles follow press coverage of the redesign and must be checked on the real phone (the macro taps
by label). The `hazard_type` keys are the contract with the macro (`wazeomatic/schema.py`, `Schema.kt`).

## How each report is detected

| Waze report | `hazard_type` | What fires it | Default |
|---|---|---|---|
| Hazard > Pothole | `HAZARD_ON_ROAD_POTHOLE` | hazard model, class pothole, floor 0.45 | on |
| Hazard > Construction | `HAZARD_ON_ROAD_CONSTRUCTION` | hazard model: cone, sign, delineator, barricade, worker | on |
| Hazard > Object | `HAZARD_ON_ROAD_OBJECT` | hazard model class debris, when the model has it | on if the model has the class |
| Hazard > Car on shoulder | `HAZARD_SHOULDER_CAR_STOPPED` | COCO vehicle on the right shoulder whose box grows as we approach | off (`--stopped-vehicle`) |
| Hazard > Broken traffic light | `HAZARD_ON_ROAD_TRAFFIC_LIGHT` | two or more signal heads close ahead, none lit, for 4 s; any lit head vetoes | on |
| Hazard (animal) | `HAZARD_SHOULDER_ANIMALS` | COCO animal classes; deer show up as sheep, cow, horse or giraffe | on |
| Police | `POLICE` | vehicle classifier says police on a COCO car / truck / bus, 3 hits in 10 frames | on |
| Police > Other side | `POLICE_OTHER_SIDE` | police box entirely in the far-left 22 % of the frame (right-hand traffic) | off (`--police-other-side`) |
| Crash | `CRASH` | ambulance / fire engine with flashing lights, or two kinds of responder with lights, that we approach or that stand by stopped traffic | on |
| Crash > Other side | `CRASH_OTHER_SIDE` | the same, entirely on the far left | on |
| Traffic > Standstill | `TRAFFIC_STANDSTILL` | not moving for 75 s behind a vehicle, no red or amber signal seen in the last 20 s | on |
| Traffic > Heavy | `TRAFFIC_HEAVY` | 150 s mostly slow or stopped, at least two separate stops, three or more vehicles in view, no red signal | on |
| Bad weather > Fog | `WEATHER_FOG` | weather classifier fog, smoothed, above 0.70 for 20 s | on |
| Bad weather > Flood | `WEATHER_FLOOD` | road-surface classifier water above 0.85 for 8 s | on |
| Bad weather > Icy road | `WEATHER_ICE` | road-surface classifier ice above 0.80 for 10 s | on |
| Bad weather > Unplowed road | `WEATHER_UNPLOWED` | road-surface classifier fresh snow above 0.75 for 15 s | on |
| Bad weather > Slippery road | `WEATHER_SLIPPERY` | rain (weather) and a wet, flooded or slushy road (surface) both, for 60 s | on |
| Blocked lane > Left / Center / Right | `BLOCKED_LANE_*` | three or more cones, barricades or delineators on the road surface reaching into our lane; the side is where their median lies | on |

Not automated, on purpose: Map issue, Closure (Waze asks for a reason and a duration the camera cannot know),
Gas prices, Place, Roadside help, Map chat.

Situations that last minutes have longer per-type cooldowns (`SCENE_COOLDOWNS` in `wazeomatic/scene.py`):
15 minutes for a traffic jam, 30 for fog, 5 for a crash or a blocked lane. A jam is reported once, not once a
minute while it lasts.

## Reading Waze before reporting

`wazeomatic/waze_reader.py` (desktop, over adb) and `WazeWatcherService.kt` (phone, an accessibility service
with no rights to tap) read the alert cards Waze shows: "Police · 0.3 mi · Reported 2 min ago",
"Still there?". When a hazard qualifies and Waze already shows that report within 1.2 km in the last two
minutes, nothing is sent: it is counted as a confirmation (`already_reported`, `CONFIRMED` in the log) and the
one-report-a-minute budget stays free for something Waze does not know yet. Traffic, Bad weather, Police and
Crash alerts cover any sub-tile of their category; Hazard subtypes only cover themselves. The status line shows
what Waze has announced ahead.

The alert wording is from the mock app. Real Waze words alerts differently between versions and languages, so
an alert whose text matches no pattern is kept as "unknown" in the log rather than dropped; that log is how the
patterns get extended after a real drive.

## Measured

Filled in by the evaluation runs below; every number names the data it comes from.
