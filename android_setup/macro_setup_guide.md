# Setup guide for the Android Macro on Waze
***
## Installation:
***
The program used to write the macros is [MacroDroid](https://macrodroid.com/) downloaded from the Google Playstore.

The macro has 3 distinct parts
* The logic that checks which report to make
* The actions that interact with the UI elements
* The variable that needs updated with the report type

To setup the variable, a local variable named ```hazard_type``` is declared at the bottom of the app inside the macro.

For actions on the device, they are classified as device interactions that click at the location of the report button, after checking its location.
Waits are also inserted to allow for dynamic element movements.

The conditional logic checks which UI element to press based on the ```hazard_type``` variable. It also constrains the user to only
 1 report per minute to prevent triggering Waze timeouts. It also checks that the foreground focus is on the Waze app before making
any clicks.
## Usage:
***
The Macro is performed automatically when it is called from the CV pipeline.

## Trigger contract and self-test
***
The macro is started by a broadcast from the detector. The contract is fixed and specified in
[docs/pipeline-to-macro.md](../docs/pipeline-to-macro.md); the parts the macro has to implement are:

| Item | Value |
|---|---|
| Trigger | *Intent Received*, action `com.wazeomatic.HAZARD_DETECTED` |
| Extras to save | `hazard_type` into the local variable ```hazard_type```, `dry_run` into ```dry_run_extra``` |
| Global variable | boolean ```DRY_RUN```, default `true` |
| Dry-run rule | if ```DRY_RUN``` is true **or** ```dry_run_extra``` is `true`: stop at the confirm sheet and tap Cancel, never Report |
| Constraints | Waze in the foreground; at most one real report per minute |

`hazard_type` values and the tiles they map to:

| `hazard_type` | Taps after the report button |
|---|---|
| `HAZARD_ON_ROAD_POTHOLE` | Hazard, Pothole |
| `HAZARD_ON_ROAD_CONSTRUCTION` | Hazard, Construction |
| `HAZARD_ON_ROAD_OBJECT` | Hazard, Object |
| `HAZARD_SHOULDER_CAR_STOPPED` | Hazard, Car on shoulder |
| `HAZARD_SHOULDER_ANIMALS` | Hazard (no subtype) |
| `HAZARD_ON_ROAD_TRAFFIC_LIGHT` | Hazard, Broken traffic light |
| `POLICE` | Police |
| `POLICE_OTHER_SIDE` | Police, Other side |
| `CRASH` | Crash |
| `CRASH_OTHER_SIDE` | Crash, Other side |
| `TRAFFIC_HEAVY` | Traffic, Heavy |
| `TRAFFIC_STANDSTILL` | Traffic, Standstill |
| `WEATHER_FOG` | Bad weather, Fog |
| `WEATHER_FLOOD` | Bad weather, Flood |
| `WEATHER_SLIPPERY` | Bad weather, Slippery road |
| `WEATHER_ICE` | Bad weather, Icy road |
| `WEATHER_UNPLOWED` | Bad weather, Unplowed road |
| `BLOCKED_LANE_LEFT` | Blocked lane, Left lane |
| `BLOCKED_LANE_CENTER` | Blocked lane, Center lane |
| `BLOCKED_LANE_RIGHT` | Blocked lane, Right lane |

Construction and pothole are the most frequent, then police, so those branches matter first. The
tiles are tapped by their label; check each label against the installed Waze before enabling a branch
(the sub-tiles of Traffic, Police, Crash and Bad weather are not confirmed on the real phone yet).

To check the branches, open Waze with the phone connected over adb and run

```
python tools/macro_selftest.py
```

It sends every `hazard_type` in turn with `dry_run=true`, prints what the macro should do for each,
and waits so the taps can be watched. It cannot send a live broadcast.

Export the macro from MacroDroid (macro menu, *Share macro*, as a `.macro` file) and commit it here
as `android_setup/wazeomatic.macro` whenever it changes, so that it can be imported, reviewed and
compared against the contract. It is not in the repository yet.
