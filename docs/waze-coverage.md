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
| Hazard > Pothole | `HAZARD_ON_ROAD_POTHOLE` | hazard model, class pothole, floor 0.55 | on |
| Hazard > Construction | `HAZARD_ON_ROAD_CONSTRUCTION` | hazard model: cone, sign, delineator, barricade, worker (a worker needs 0.6) | on |
| Hazard > Object | `HAZARD_ON_ROAD_OBJECT` | hazard model class debris (0.5), lying on the road: the box reaches below the horizon less 0.15 and is not mostly inside a person or vehicle box | on |
| Hazard > Car on shoulder | `HAZARD_SHOULDER_CAR_STOPPED` | COCO vehicle on the right shoulder whose box grows as we approach | off (`--stopped-vehicle`) |
| Hazard > Broken traffic light | `HAZARD_ON_ROAD_TRAFFIC_LIGHT` | two or more signal heads close ahead, none lit, for 4 s; any lit head vetoes | on |
| Hazard (animal) | `HAZARD_SHOULDER_ANIMALS` | COCO animal classes; deer show up as sheep, cow, horse or giraffe. A dog or cat beside a person on the same ground line is a pet on a walk and does not count | on |
| Police | `POLICE` | vehicle classifier says police (0.85 or more) on one of the four largest COCO car / truck / bus boxes, 2 hits in 10 frames, and the car is read as standing still (Waze's Police is for a police car that stands; see Measured, `--police-any-motion` turns the check off) | on |
| Police > Other side | `POLICE_OTHER_SIDE` | police box entirely in the far-left 22 % of the frame (right-hand traffic) | off (`--police-other-side`) |
| Crash | `CRASH` | an ambulance or fire engine, or two kinds of responder, with red / blue flashing lights on them, for 3 s: boxes that grow while we drive towards them, or that stand still while we stand (a responder driving past a waiting car is not a crash); ends as soon as they leave the frame | on |
| Crash > Other side | `CRASH_OTHER_SIDE` | the same, entirely on the far left | on |
| Traffic > Standstill | `TRAFFIC_STANDSTILL` | 75 s with at least 90 % of frames read as stopped and at most 1 % as moving, behind a vehicle, no red or amber signal seen in the last 20 s | on |
| Traffic > Heavy | `TRAFFIC_HEAVY` | 150 s with at least 30 % of frames read as stopped and half read slow or stopped, at least two separate stops, three or more vehicles in view, no red signal | on |
| Bad weather > Fog | `WEATHER_FOG` | weather classifier fog, smoothed, above 0.70 for 20 s | on |
| Bad weather > Flood | `WEATHER_FLOOD` | road-surface classifier water above 0.85 for 8 s | off: no road-surface model ships (see Measured) |
| Bad weather > Icy road | `WEATHER_ICE` | road-surface classifier ice above 0.80 for 10 s | off: no road-surface model ships |
| Bad weather > Unplowed road | `WEATHER_UNPLOWED` | road-surface classifier fresh snow above 0.75 for 15 s | off: no road-surface model ships |
| Bad weather > Slippery road | `WEATHER_SLIPPERY` | rain (weather classifier, smoothed, above 0.65) for 60 s; with a road-surface model loaded, a wet, flooded or slushy road as well | on |
| Blocked lane > Left / Center / Right | `BLOCKED_LANE_*` | three or more cones, drums or barricades 3-40 m ahead, placed on the road with a flat-road camera model (mount height, horizon, field of view: `--camera-geometry`), two of them well inside our lane, the same side for 1.5 s; the side is where their median lies | on |

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

Every number names the data it comes from and the command that produced it. None of this data was used to
pick the thresholds unless the row says so.

### Police and crash

**Police on dashcam footage.** Argoverse 2 front-camera logs (16 s each, 5 fps; six US cities) split in two
by a hash of the clip name: one half was mined for the hard negatives of v5, v6 and v8, the other half was
never trained on and is only scored (v8 also learnt from 300 further logs that are not in either half). The
police cars in that half were found by scoring every 5th frame with a classifier at a low threshold and
checking every candidate by eye (`configs/police_truth_argoverse.txt`); the full pipeline then runs over every
frame (`tools/eval_vehicle.py clips --clip-half 1 --step 1 --police-truth ...`). The first rows were scored
against the list as it was then (9 police clips); v8 turned up four more that no earlier model had scored 0.2,
all checked by eye, and the last two rows use the list of 13:

| 163 held-out clips, 37.9 minutes | crop size | hits in 10 frames | police clips found | clips without police that got a Police report |
|---|---|---|---|---|
| v4 | 128 px | 3 | 5 of 9 | 16 of 154 |
| v5 | 128 px | 3 | 7 of 9 | 12 of 154 |
| v6 | 128 px | 3 | 8 of 9 | 4 of 154 |
| v5 and v6 averaged | 128 px | 3 | 8 of 9 | 2 of 154 |
| v5 and v6 averaged | 160 px | 3 | 5 of 9 | 0 of 154 |
| v5 and v6 averaged (shipped until v8) | 160 px | 2 | 7 of 9 | 0 of 154 |
| v5 and v6 averaged, list of 13 | 160 px | 2 | 7 of 13 | 0 of 150 |
| v8 (ships), list of 13 | 160 px | 2 | 12 of 13 | 0 of 150 |

The classifiers are trained on 160 px crops. Their ONNX export has a dynamic input size, and until this
measurement both the desktop and the phone read that as "no size" and fed them 128 px crops; they now take the
size from the export's metadata (`imgsz`). At 160 px the models are less sure of the police cars on these
clips, so three hits above 0.85 found only five of them; two hits find seven with no false report. The v5, the
averaged and the 160 px rows are the pipeline itself; v4 and v6 come from a cache of the classifier's
per-frame scores on the same frames run through the same rule, which gave the same counts as the pipeline
where both were run. The averaging, the 160 px input and the two-hit rule were chosen on these clips, so the
last rows are in-sample for those choices (not for the models, which never saw these clips). At 160 px no
floor between 0.6 and 0.9 with two to four hits finds more than seven without a false report; at 128 px with
three hits the same clips give eight with two false reports (an old white Chevrolet Impala and a silver Dodge
seen from behind). The two police cars the pair missed are a DC Metropolitan Police SUV seen mostly from behind
(0.72 at most at 160 px) and the car in clip 931b76ee, which the 160 px models score no higher than 0.46.

v8 finds both, and the four the list lacked: two parked sedans with their light bars on, one from behind
(`assets/videos/labelled/argo_2ff4f798_workzone.mp4` is one of them), a DC SUV and a Miami SUV half out of
the frame, the one it misses. It came from mining whole tracks instead of single crops ([models/README.md](../models/README.md#building-the-models));
averaging it with v5 or v6 gives back most of the gain (7 or 8 of 13), so it ships alone. The v8 row is the
pipeline as shipped (`runs/eval/argo_h1_vehicle_v8.json`); the pair's row on the list of 13 is from the cache
of per-frame scores. The rule (0.85, two hits) was kept from the pair, not retuned for v8.

**Police: standing or driving.** Waze's Police report is for a police car that stands (a speed trap, a traffic
stop), so a police box only counts once `wazeomatic/motion.py` (`PoliceMotion.kt` on the phone) has read the
car as standing still. While we drive, it follows corners on the road and on the car with pyramidal
Lucas-Kanade: road point pairs on one row give how fast the static world spreads apart at each row, the road
flow left over is the camera's turn (no gyro), and the car must move the way a static object standing on its
row would (flow residual at most 0.3 on the oncoming side, 0.6 elsewhere) and grow at least 0.3 times as fast
(a car driving ahead at our pace barely moves in the image, like a static one near the focus of expansion, but
does not grow). While we stand, its box must stay put. A car stays *pending*, and is not reported, until three
measurements agree. Truth comes from the Argoverse 3D tracks: each detector box of the 163 held-out clips is
matched to a projected cuboid and takes its ground speed (`tools/label_vehicle_motion_argoverse.py`, 41,996
boxes). Police cars are too few to score the check on, so every vehicle stands in for one
(`tools/eval_police_motion.py`); a track is standing under 0.5 m/s, driving over 2 m/s:

| vehicles within 30 m, verdict on the track's last frame | standing (752 tracks) | driving (314 tracks) |
|---|---|---|
| read standing (would be reported) | 46 % | 4 % |
| still pending (too few measurements: short tracks, small boxes) | 49 % | 56 % |
| of those decided while we drive: read standing | 95 % (of 343) | 10 % (of 121) |
| of those decided while we stand: read standing | 54 % (of 37) | 6 % (of 17) |

From 30 to 60 m, 37 % of standing and 10 % of driving tracks are read standing. The road-based turn estimate
is within 1.1 px a frame of the true rotation (median; from the logs' poses) and separates as well as the true
rotation does. The thresholds were chosen on these clips, so the table is in-sample for them. Through the whole
pipeline (`runs/eval/police_stopped_h1_b.json`, `--camera-geometry 47.2,1.44,0.53` for this camera), Police is
reported on all 11 clips with a standing police car that v8 finds (`ad870270` stays the one v8 misses), not on
`eb222d5d`, whose DC police SUV drives past, and on none of the 150 clips without police. The check is weakest
while we stand (the box test reads only about half the parked cars as standing) and on slow oncoming traffic.
It costs nothing until a police box is in view; then about 40 ms a frame on a desktop JVM.

**Vehicle classifier, crop level.** Held-out crops of `<datasets>/vehicle_crops/val` (split by source file;
`tools/eval_vehicle.py crops --data ...`), scored at the shipped police floor of 0.85, at the 160 px input the
models were trained at:

| | v3 | v4 | v5 | v6 | v5 + v6 (ships) | v5 + v6 at 128 px |
|---|---|---|---|---|---|---|
| police crops called police (889) | 74.8 % | 74.1 % | 76.4 % | 75.0 % | 75.6 % | 68.1 % |
| dashcam civilian vehicles called police (BDD100K, 5,784) | 0.02 % | 0.03 % | 0.02 % | 0.02 % | 0.02 % | 0.02 % |
| civilian photos called police (Open Images cars, taxis, vans, 1,542) | 1.4 % | 1.2 % | 1.2 % | 1.2 % | 1.0 % | 1.6 % |
| civilian look-alikes called police (Commons: Tahoes, Explorers, taxis, tow trucks, 193) | 7.3 % | 6.2 % | 5.7 % | 7.8 % | 6.2 % | 7.3 % |
| ambulances called police (554) | 1.3 % | 0.7 % | 0.9 % | 0.9 % | 0.9 % | 2.2 % |
| fire engines called police (416) | 1.0 % | 1.0 % | 1.0 % | 1.0 % | 1.0 % | 1.0 % |

The last column is what ran before the input size was fixed. The averaged pair calls one dashcam vehicle in
5,784 police in a single frame. A report needs two such frames out of ten, so isolated misses do not reach
Waze; what does is a look-alike that stays in view, which is why the floor is 0.85 and not the 0.5 of the
argmax. v4 differs from v3 by 3,000 more ambulance photos from
six countries; v3 called Edinburgh, Toronto and Japanese ambulances police at 0.9 and above.

The val folder has grown since (303 taxi photos, 91 Argoverse look-alikes, 55 frames of police tracks and 28
of look-alike tracks), so v8 is compared with the pair on the current one (`runs/eval/vehicle_v8_crops_E.json`,
`vehicle_v5v6_crops_E.json`), at 0.85:

| | v5 + v6 | v8 (ships) |
|---|---|---|
| police crops called police (944) | 73.4 % | 71.4 % |
| dashcam civilian vehicles called police (BDD100K, 5,784) | 0.02 % (1) | 0.03 % (2) |
| civilian photos called police (Open Images, 1,542) | 0.97 % | 0.71 % |
| civilian look-alikes called police (Commons, 193) | 6.2 % | 3.6 % |
| taxis called police (Commons, 303) | 8.9 % | 1.7 % |
| Argoverse look-alikes called police (91 crops, 28 track frames) | 1.1 %, 0 % | 0 %, 0 % |
| ambulances called police (554) | 0.90 % | 0.54 % |
| fire engines called police (416) | 0.96 % | 0.48 % |

Most police crops are hand-held Commons photos; on those v8 is a little behind the pair, as on the Commons
clips below. Its gain is on dashcam views, which this folder has few of.

**Clips.** The full pipeline (`tools/eval_vehicle.py clips`, report manager as shipped, no global cooldown),
v4 at 128 px / v5 + v6 at 128 px / v5 + v6 at 160 px with two hits / v8 (ships):

| Footage | Minutes | POLICE | CRASH |
|---|---|---|---|
| dashcam drives in `assets/videos`: two Commons drives at two encodings, two UK drives, two Argoverse logs | 14.6 | 0 / 0 / 0 / 1 | 0 / 0 / 0 / 0 |
| Commons videos of police vehicles, mostly hand-held (19 clips) | | 9 / 8 / 10 / 8 clips | 2 / 0 / 1 / 2 clips |
| Commons videos of ambulances (13 clips) | | 0 / 1 / 2 / 2 clips | 6 / 7 / 7 / 6 clips |
| Commons videos of other responders and emergency lighting (11 clips) | | 3 / 3 / 1 / 3 clips | 7 / 6 / 5 / 3 clips |

The v8 Police report on the drives is right: the Argoverse work-zone log
(`labelled/argo_2ff4f798_workzone.mp4`) has a police sedan with its light bar on parked by the cones, which no
earlier model reported. The rows were once described as drives without police; that clip never was one.

The Police report among the other responders is "Toronto Police responding", a real police car. With the pair
the two ambulance clips that get one are false: the box is a taxi in both, a dark Edinburgh private-hire car
with a door badge (0.89) and a white Japanese taxi in Sendai (0.86). v8 drops the Edinburgh car and keeps
Sendai, and adds three false ones: a French ambulance clip (the box is road surface in a blurred pan), dark
cars under blue light at a Berlin fire scene at night, and a Toronto box ambulance, white with blue graphics,
parked at a scene (0.85 to 0.91 for several seconds). The Commons police clips v8 misses and the pair found are
a Poissy municipal police car and the unmarked New York cars.

Taxis also lead the civilian photos called police at the crop level: the pair calls 8.9 % of 303 held-out
Commons taxi photos (Japan, UK, Canada, Germany, France, Australia, New York) police at 0.85. A v7 trained with
1,894 of those photos as civilian, and with softened copies of half the crops (`--res-jitter 0.5`), brings that
to 1.3 % and, averaged with v5 and v6, takes the Commons ambulance and lighting clips to no Police report; but
on the held-out Argoverse clips it finds 6 of 9 police cars, alone or averaged, against 7 of 9 for the pair,
and the Commons police clips from 10 to 8. The pair stayed then; v8 replaced it (see above). The cached scores behind these counts are per-frame classifier scores on the pipeline's own vehicle
boxes, run through the two-hit rule.

The police clips that stay silent show unmarked cars (a diplomatic convoy, black saloons), a car far away, or
only a light bar at night. The CRASH events on the Commons clips are responders driving towards a camera that
pans with them: to the rule that looks like approaching a scene. A dashcam meets that only with an oncoming
responder, which leaves the frame within a second or two; the rule now also requires the responders to stand
still while we stand and ends the scene as soon as they leave the frame, which took the Commons count from 20
clips to 13 with v3, and the dashcam clips have none. There is no public dashcam footage of real crash scenes
to measure recall on; that is item 17 of the sprint 2 plan.

### Blocked lane

Scored against the 3D cuboids of Argoverse 2 (100 logs selected for road works, 26.7 minutes, front camera at
5 fps; `tools/eval_lanes_argoverse.py`). The truth is a moment with at least three cones or drums 3-40 m ahead,
two of them inside our lane, held without a break; the side is where their median stands.

| Rule | Truth held 1.5 s: precision / recall | Truth held 1.0 s: precision / recall | Side right on hits (1.5 s / 1.0 s) |
|---|---|---|---|
| before: cones in fixed image strips | 0.16 / 1.00 | 0.27 / 1.00 | 0.75 / 0.69 |
| now: flat-road camera model | 0.46 / 0.75 | 0.77 / 0.77 | 0.83 / 0.80 |

The old rule fired in 49 of the 100 clips, the new one in 13. `lane_reach` 0.8 was chosen from ten settings on
these same clips (`--cache ... --sweep`), so the second row is tuned in-sample; the camera model itself and the
other parameters were set before. Most remaining false alarms are cones on a curve, where a flat straight road
puts them in the wrong lane. A Waze user who sees a closed lane holds the 1.0 s definition; the 1.5 s one is
there to show how the numbers move.

The second Argoverse fetch added logs that were never used for tuning. On the held-out half of all logs
(`--clip-half 1`, 163 clips, same geometry), the 109 clips outside the original 100 have no blocked lane by
the label definition and get no Blocked lane report; the 54 inside it give 4 of 6 found with no false report
(1.0 s) and 3 of 5 with one false report (1.5 s) (`runs/eval/lanes_h1*.json`).

### Construction and the other hazard rules

Every rule at once (hazard model, COCO, vehicle classifiers, scene rules; `tools/eval_vehicle.py clips --hazard
--clip-half 1 --step 1`) on the 163 held-out Argoverse clips, 37.9 minutes of city driving in six US cities.
Construction is scored against the 3D labels with `tools/eval_construction_argoverse.py`: a clip is a work
zone when a cone, drum, message-board trailer or flagger with lidar points stands 3-40 m ahead in view and at
most 10 m to the side for a second, and has none when no such object is within 80 m; the 60 clips in between
are left out.

| Report | Result |
|---|---|
| Construction | 85 of 86 work-zone clips reported (recall 0.99); 3 of the 17 clips without a work zone get one (precision 0.97). The v3 model: 81 of 86, the same 3 false |
| Animal | none (a dog on a walk beside its owner was reported before the pet rule) |
| Object | none (two false ones before the debris road rules: a vent on a building front and the legs of a man crossing) |
| Pothole, Broken traffic light, Crash, Traffic | none (Argoverse has no pothole labels, so this only says there was no false one; a real dashcam pothole is in the next section) |
| Police | 12 of 13, none false (vehicle v8; the table above) |

Construction reports without a labelled work zone, checked by eye: striped delineator posts on the pavement
and a row of posts with fencing at a work site (both labelled as bollards in Argoverse, so real work zones the
labels miss), and an "END ROAD WORK" sign (a construction sign, but it marks where the work zone ends: telling
the two apart needs the text). A fourth, a pedestrian in an orange T-shirt taken for a worker at 0.61, is why a
worker box now needs 0.6 to count: replayed on cached detections (`tools/cache_detections.py --clip-half 1`)
that drops it and no work-zone clip, and on the val clips of `tools/eval_events.py` (1,201 clips, `--zoom-crop
--limit 120 --frames 8 --seed 1`) it keeps all 310 Construction events and drops one false one.

These rows are the shipped hazard model v4 (`runs/eval/argo_h1_hazard_v4b.json`). Object on the road is also
measured on Lost and Found, frames of small obstacles on German roads (boxes, tyres, crates, bins) from a
camera like a dashcam: with `--data <datasets>/laf_yolo` on 120 frames with an obstacle and 120 without (8
jittered frames each), 66 of 120 obstacles are reported and 8 of the 120 frames without one get a report
(recall 0.55, precision 0.89). The debris floor of 0.5 is chosen on these frames (0.4: 0.59 / 0.84; 0.6: 0.50 /
0.92), so they are in-sample for it; how v4 compares with v3 is in [model-research.md](model-research.md#hazard-v4-debris-2026-11-14).

### Traffic: what the camera can tell about our own speed

Both Traffic reports rest on the camera-only speed estimate (`EgoMotion`; GPS speed replaces it only when
`use_gps_speed` is on, and it is off by default). Argoverse 2 logs carry the car's true pose, so every frame of
the 163 held-out clips has a true speed (`tools/eval_ego_argoverse.py --cache runs/cache/argo_h1.jsonl`, 5 fps
like the phone, first second of each clip and speeds within 0.3 m/s of a cut-off not scored):

| true speed | frames | read stopped | read slow | read moving |
|---|---|---|---|---|
| stopped (< 0.8 m/s) | 1,047 | 91.4 % | 8.3 % | 0.3 % |
| slow (0.8-6 m/s) | 3,482 | 11.2 % | 59.1 % | 29.6 % |
| moving (> 6 m/s) | 5,216 | 5.5 % | 69.9 % | 24.6 % |

The estimate tells stopped from moving, but not slow from moving: its score stops rising above about 3 m/s
(median 1.6-1.8 at every speed from 3 to 15 m/s). The rules as first written leaned on exactly that. Replayed on
150 s windows chained from held-out clips (no red-light veto, so a worst case), the old Heavy rule fired on
293 of 462 windows of free-flowing traffic (mean true speed above 6 m/s), and the old Standstill rule, which
wanted every frame of 75 s read as stopped, fired on 1 of 500 75 s windows chained from real stops. Both rules now count stopped time
instead:

| rule | fires on slow windows (mean < 4 m/s) | on free-flowing windows (> 6 m/s) |
|---|---|---|
| Heavy, old (slow share 0.75, two stops) | 241 of 459 | 293 of 462 |
| Heavy, shipped (also 30 % stopped, slow share 0.5) | 215 of 459 | 1 of 462 |
| Heavy, old rule on the true speed (what GPS would give) | 430 of 459 | 0 of 462 |

| rule | fires on 75 s of real stops | on 75 s windows of driving |
|---|---|---|
| Standstill, old (every frame stopped) | 1 of 500 | 0 of 1,000 |
| Standstill, shipped (90 % stopped, at most 1 % moving) | 499 of 500 | 0 of 1,000 |

The thresholds were chosen on these windows. The other Argoverse cache (`runs/cache/argoverse.jsonl`, the first 100
logs, 54 of them also in the held-out half) gives the same picture: old Heavy 311 of 471 free-flowing windows,
shipped 4 of 471 with 406 of 463 slow ones; shipped Standstill 208 of 500 stop windows (old 68), none on
driving. The stopped stretches are short (about 3 minutes in all, chained), the windows are made of 15 s clips joined end
to end, and none of this is a real jam: it measures the speed estimate under the rules, not a Traffic report
on a real jam, which still needs footage of one. The eight dashcam drives of `assets/videos` (14.6 min)
give no Traffic report before or after the change. Turning on `use_gps_speed` is the bigger gain where the phone has a
fix.

### Pothole on a real dashcam

The pothole numbers in [model-research.md](model-research.md#training-and-evaluation-procedure) come from
labelled still frames. The one public dashcam video with potholes found so far, `Dashcam-Front.mp4` of the
Mendeley pothole set (tp95cdvgm8 v1, CC BY 4.0; 34 s, 1080p, a UK residential street, not redistributed here),
gives **no report** with every rule on (`tools/eval_vehicle.py clips --hazard --step 1`): the pothole is
missed, and nothing false is reported either. The potholes in it are shallow and small (frames 840-880, in the
bottom fifth of the picture); the model gives them under 0.3 in the full frame and in the zoom crop. Moving the
zoom crop down to rows 35-85 % raises the best pothole score in that stretch to 0.65, but every box above the
0.45 floor is on the painted give-way triangle beside the potholes, not on a pothole, so the crop stays where it
is. Small UK-style potholes seen from a car are a gap in the training data; footage from the team's own mount
(sprint 2 item 6) is the planned fix.

The other way round, hazard v4 makes one false Pothole report on real footage: 9 s into
`assets/videos/uk_dashcam_006_1080p.mp4`, on wet asphalt with tyre tracks (scores 0.47-0.63 over about a
second; v3 reports nothing there). The rest of the 14.6 minutes of dashcam drives give no false Pothole. What the floor
buys (`tools/eval_events.py --zoom-crop --limit 120 --frames 8 --seed 1`, `runs/eval/pothole_floor_*`):

| pothole floor | pothole event recall (synthetic clips) | precision | false Pothole, 14.6 min of real drives |
|---|---|---|---|
| 0.45 (shipped until 2026-11-22) | 0.56 | 0.99 | 1 |
| 0.50 | 0.51 | 0.99 | 1 |
| 0.55 (ships) | 0.43 | 0.99 | 0 |

The floor went up to 0.55 on 2026-11-22: a false report on real footage counts for more than 13 points of
recall on synthetic clips, and the one real dashcam pothole is missed at either floor. `report_class_conf`
(phone) and `--report-class-conf pothole=0.45` (desktop) set it back.

### Animal on a real dashcam

`Metskitsed autoteel.webm` from Wikimedia Commons (CC0; 8 s, 1080p, a dashcam on a snowy forest road in
Estonia) has a roe deer crossing the road about 20 m ahead for about a second (frames 80-102). With every rule
on it gives **no report**. COCO has no deer class, and the deer in its winter coat against snow is found in
1 of 17 frames by the shipped nano (dog 0.70), 2 by yolo11s and 3 by yolo11l (dog 0.54-0.66), short of the 3
hits in 10 frames a report needs. A bigger COCO model does not close this, so the fix is an animal class trained on
dashcam frames of deer, elk and moose; until then Animal is reported only for animals the COCO model knows
when they stay in view, and a deer crossing quickly is likely to be missed.

### Weather

The weather classifier on 2,425 BDD100K frames held out by file hash (`tools/eval_scene.py`), one frame at a
time, before the smoothing and hold times of the rules:

| BDD label | frames | classified as that label |
|---|---|---|
| clear (clear, partly cloudy, overcast) | 800 | 96 % |
| rain | 800 | 75 % |
| snow | 800 | 82 % |
| fog | 25 | 12 % (3 frames) |

On clear frames no single frame clears the Fog threshold (fog above 0.70), and 1.5 % clear the rain threshold
of Slippery road (rain above 0.65); on rain frames the rain threshold holds in 70 %. The reports need the
smoothed probability above the threshold for 20 s (fog) and 60 s (rain), so isolated frames do not reach
Waze. These numbers are at the 224 px input the model was trained at (`runs/eval/scene_topview_224w_160s.json`);
fed 128 px frames, as the pipeline did before the input size was read from the model, clear frames met the
fog threshold in 1.0 % and the rain threshold in 3.3 %, and rain frames 62 %. Real fog is rare in BDD100K
(25 val frames): at 224 px a fog threshold as low as 0.4 still catches only 3 of them. A second check
uses DAWN (real road pictures in adverse weather, gathered from the web; evaluation only, see NOTICE.md):
a single picture clears the fog threshold in 46 % of its 300 fog pictures (52 % at 128 px) and the rain
threshold in 60 % of its 200 rain pictures (62 %), and 17 rain pictures come out as fog (31 at 128 px). A
fog threshold of 0.6 would add two points of fog recall, so it stays at 0.70.

### Road surface

A road-surface classifier trained on RSCD (close-up photos of the road surface, 92 % top-1 on RSCD's own
validation split) does not carry over to a dashcam. On the same BDD100K frames, fed the patch of road just
above the hood, it calls 73 % of clear-weather patches "water" and gives the same class mix whatever the
weather label (`runs/eval/scene_v1.json`): the Flood rule's single-frame condition would hold on 49 % of clear
frames. The patch from a dashcam holds the road far ahead in perspective, cars, lane paint and at night mostly
darkness; the RSCD pictures are textures seen from close above.

The second attempt matched the training pictures to the camera instead. RoadSaW and RoadSC (VISCODA, CC BY-NC-SA
4.0) are road patches seen from a car and warped to a bird's-eye view; a YOLO11n-cls model on them (dry, damp,
wet, snow, slush; `tools/build_scene_sets.py topview`) reaches 87 % top-1 on their validation split, and the
pipeline produces the same kind of patch with the blocked-lane camera model, skipping frames where a vehicle or
a person covers it (`RoadSurface` with `topview`, on both the desktop and the phone). On the BDD100K frames it
still does not separate the weather (`runs/eval/scene_topview_d75.json`, square 7.5 m ahead):

| BDD label | wet or slush (Slippery's surface half) | snow (Unplowed) |
|---|---|---|
| clear | 49 % | 1.6 % |
| rain | 59 % | 0.3 % |
| snow | 47 % | 2.2 % |

At 15 m the square is about 20 pixel rows of a 720p frame and often lands on the car ahead or past the
horizon, since BDD100K mixes many mounts; at 7.5 m it is about 80 rows and still mostly wrong. Most clear
frames it calls wet are at night, which RoadSaW (daylight) never shows. Those runs fed the model 128 px
patches; at the 160 px it was trained at, the 15 m square is still called wet or slush on 61 % of clear frames
and 60 % of rain frames (`runs/eval/scene_topview_224w_160s.json`). So no surface model ships, Flood, Icy
road and Unplowed road stay off, and Slippery road rests on rain alone. The code path stays: a top-view model
dropped in as `surface_cls` is picked up from its class names and warped with the mount's `--camera-geometry`
(`camera_geometry` on the phone). Training it on patches from our own camera, day and night, is the next step.
