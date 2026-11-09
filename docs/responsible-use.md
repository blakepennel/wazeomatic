# Responsible use

Wazeomatic taps through another company's app on a public road, and a report it files is shown to
other drivers. This page says what the project does and does not do about that, and what is still
unknown. It is not legal advice.

## What the project is allowed to do, by its own rules

1. **Automation never files a report.** Every sender defaults to `dry_run=true` (the phone app, the
   desktop adb sink, the emulator suite) and the macro's own `DRY_RUN` variable defaults to on. In
   a dry run the macro walks to the confirm sheet and cancels. A live report needs both switches
   turned off by hand ([pipeline-to-macro.md](pipeline-to-macro.md)).
2. **Tests run against the mock.** The whole loop is tested on an emulator against
   `android/mockwaze`, which sends nothing anywhere.
3. **Testing against the real app is dry run only.** That is a project rule, not a setting someone
   may flip for a demo.
4. **A live report is a driver's decision, on a real drive, about a hazard that is really there.**
   Nothing in this repository makes that decision.

## Waze's terms

The Waze terms of use (read on 2026-11-09, last modified 2026-07-08,
<https://support.google.com/waze/answer/12373727>) matter in three places:

- *Rules of Conduct* and *Use Restrictions* prohibit intentionally submitting false information or
  reports, and fraudulent or deceptive use including fake reports. A false positive from a detector
  is not intentional, but a system that files reports nobody looked at is close enough to that line
  that the project keeps a person in the loop: see the rule above and the open item below.
- *Use Restrictions* prohibits using automated means to access content from the service. The macro
  does not read or collect anything from Waze; it presses the same buttons a user would. Whether
  scripted input counts is for Waze to say, not for this project, and nobody has asked them.
- Waze may remove content, stop an account from posting, or end access at its discretion. Anyone
  who turns dry run off does so with their own account.

The terms also put responsibility for attentive driving on the driver. The point of the project is
to remove the reason to touch the phone, not to add a screen to watch.

## What a wrong report costs

Other drivers slow down, change lane or lose trust in the pin. The pipeline is built to make wrong
reports rare (three detections in ten frames, per-class confidence floors, a learnt dashboard zone,
one report a minute at most), and the measured false-report rate is zero events in twelve minutes
of real footage ([model-research.md](model-research.md)). That measurement has limits that a user
must know:

| Limit | State |
|---|---|
| Light and weather | All measured footage is daylight or overcast daylight. Night, rain, wet roads, glare and wiper smear are **unmeasured**. Do not run live outside daylight. |
| Where the pin lands | Waze pins a report at the phone's position when the sheet sends, not where the hazard was first seen. With the persistence window, the macro's taps and Waze's own countdown that is several seconds of travel. The offset is **unmeasured**; see the sprint 2 plan. |
| Place | The pipeline has no notion of place. A cone in a car park is still a cone. |
| Retraction | There is no way to withdraw a report from the pipeline. Waze's own countdown on the confirm sheet is the last chance to cancel. |
| Footage volume | Twelve minutes from three cars is a small sample for a claim about false reports. |

## Before anyone turns dry run off

These are conditions, not suggestions. They are tracked in [sprint2.md](sprint2.md).

- The macro has been watched stopping at the confirm sheet on the real app, for every hazard type
  it has a branch for.
- The report offset has been measured on a real drive and is acceptable, or compensated.
- The run is in daylight, on a road and in conditions like the ones measured.
- A passenger, not the driver, watches the first live runs and can cancel on the countdown.

## The mock app

`android/mockwaze` reproduces the geometry and wording of the report flow because coordinate taps
can only be tested against something laid out the same way. It uses its own drawn icons, is
labelled "MOCK WAZE · nothing is sent" on screen, and is for test devices only. It should not be
published to a store or presented as Waze.

## Privacy

The detector processes camera frames on the device and stores no video. Events (hazard type,
confidence, time, box) are broadcast locally to the macro. Nothing is uploaded by this project.
Clips a developer records for evaluation contain other people's vehicles and faces and should not
be committed or shared; `assets/videos/` is git-ignored for that reason.
