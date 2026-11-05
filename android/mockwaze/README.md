# Mock Waze (`com.wazeomatic.mockwaze`)

A look-alike of the Android Waze report flow so the MacroDroid macro can be rehearsed without
sending real reports (real Waze publishes a report the moment you tap Send). Shows up in the
launcher as **Waze (Mock)**. Kotlin, plain framework views (no AndroidX), minSdk 29, target 35.

## Build

```powershell
cd android\mockwaze
.\gradlew.bat assembleDebug
# -> app\build\outputs\apk\debug\app-debug.apk
```

Needs `JAVA_HOME` (JDK 17+) and the SDK at `F:\dev\android-sdk` (`local.properties`, not committed;
create it with `sdk.dir=F\:/dev/android-sdk` or set `ANDROID_HOME`).

## What it looks like

* Main screen: fake map, "Where to?" bottom bar, orange round **Report** button bottom-right.
* Report -> bottom sheet slides up (`sheet_ms`, default 300 ms) with 12 tiles: Traffic, Police,
  Crash, Hazard, Closure, Blocked lane, Bad weather, Gas prices, Map chat, Map issue, Place,
  Roadside help. The order of the last row may differ from the real app (see `WazeMenu.kt`).
* Hazard -> confirm sheet with Hazard preselected and a flat list of subtypes: Construction,
  Car on shoulder, Broken traffic light, Pothole, Object (tap one to select it, tap again to go
  back to plain Hazard). Every other tile -> one-step confirm sheet.
* Confirm sheet: **Send** and **Cancel**. Send shows a "Thanks!" toast (`toast_ms`) and closes.
* Every tap target has its label as text and content description, so MacroDroid "UI interaction"
  by text and plain coordinate taps both work.

## Logs

Each Send or Cancel on the confirm sheet appends one JSON line to
`/sdcard/Android/data/com.wazeomatic.mockwaze/files/reports.jsonl` and logs it to logcat tag
`MockWazeReport`:

```json
{"category":"Hazard","subcategory":"Pothole","ts":1762360000.12,"dry_run":false,"via":"auto_macro","action":"send","hazard_type":"HAZARD_ON_ROAD_POTHOLE","confidence":"0.87","source":"e2e","event_ts":"1762359999.8"}
```

`action=cancel` rows are rehearsals and always have `dry_run=true`. `via` is `touch` for real
taps (you or MacroDroid) and `auto_macro` for the built-in macro. What the auto macro did with
every broadcast (accepted / sent / suppressed_rate_limit / skipped_not_foreground / ...) goes to
`macro_events.jsonl` and logcat tag `MockWazeMacro`.

```powershell
adb -s emulator-5554 shell cat /sdcard/Android/data/com.wazeomatic.mockwaze/files/reports.jsonl
adb -s emulator-5554 logcat -s MockWazeReport MockWazeMacro
```

## Auto macro mode

MacroDroid is a Play Store app and the `google_apis` emulator image has no Play Store, so the
mock can replay the macro itself. When `auto_macro` is on, a foreground service registers a
receiver for `com.wazeomatic.HAZARD_DETECTED` **at runtime**. (A receiver declared in the
manifest would never fire: since Android 8 implicit broadcasts are not delivered to manifest
receivers. MacroDroid works around this the same way.) For each event it:

1. maps `hazard_type` to the menu path (same table as `wazeomatic/schema.py`; falls back to
   the `waze_category`/`waze_subcategory` extras),
2. drops it if a real report was made less than `rate_limit_ms` (60 s) ago,
3. drops it if mock Waze is not in the foreground (like the macro's foreground check),
4. taps Report, waits `macro_wait_ms`, taps the category, waits, taps the subtype, waits,
   taps Send. Taps are real touch events at the view centre, so tapping while the sheet is
   still animating misses, exactly like a too-short wait in MacroDroid would,
5. with `dry_run=true` it stops on the confirm sheet for `dry_run_hold_ms`, taps Cancel and
   logs `dry_run=true`. Dry runs do not use up the one-per-minute slot.

Test helper: `bring_to_front=true` pulls mock Waze in front when an event arrives while another
app (e.g. the detector) is on screen. Android 10+ blocks background activity starts unless the
app may draw over other apps: `adb shell appops set com.wazeomatic.mockwaze SYSTEM_ALERT_WINDOW allow`.

## Settings / timing

Settings button (top-left) or launch extras or a broadcast, all with the same keys:

| key | default | |
|---|---|---|
| `sheet_ms` | 300 | sheet slide up/down |
| `page_ms` | 200 | fade between sheet pages |
| `toast_ms` | 1500 | "Thanks!" toast |
| `macro_start_delay_ms` | 500 | event -> first tap |
| `macro_wait_ms` | 1000 | wait after each tap |
| `dry_run_hold_ms` | 1500 | time on the confirm sheet before Cancel |
| `rate_limit_ms` | 60000 | one real report per window |
| `auto_macro` | false | listen for HAZARD_DETECTED |
| `require_foreground` | true | only tap when on screen |
| `bring_to_front` | false | see above |

One-shot extras: `clear_reports`, `reset_rate_limit`.

```powershell
adb shell am start -n com.wazeomatic.mockwaze/.MainActivity --ez auto_macro true --ei sheet_ms 300 --ez clear_reports true
adb shell am broadcast -a com.wazeomatic.mockwaze.SET_TIMING -p com.wazeomatic.mockwaze --ei sheet_ms 500 --ei macro_wait_ms 1200
adb shell am broadcast -a com.wazeomatic.HAZARD_DETECTED --es hazard_type HAZARD_ON_ROAD_POTHOLE --es confidence 0.87 --es dry_run true
```

Use `-p com.wazeomatic.mockwaze` on SET_TIMING when the app is not open (then only the manifest
receiver can get it, and only for broadcasts aimed at our package).

## Rehearsing the real MacroDroid macro

On a real phone with MacroDroid, install this APK next to Waze, point the macro's
"application in foreground" check at `com.wazeomatic.mockwaze` instead of `com.waze`, leave
`auto_macro` off, and the macro's taps land on the mock. Coordinates were laid out for a
1280x2856 screen (Pixel 9 Pro); recheck them against the real Waze before going live.
