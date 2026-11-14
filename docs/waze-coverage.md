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
| Police | `POLICE` | vehicle classifier (two models averaged) says police (0.85 or more) on one of the four largest COCO car / truck / bus boxes, 2 hits in 10 frames | on |
| Police > Other side | `POLICE_OTHER_SIDE` | police box entirely in the far-left 22 % of the frame (right-hand traffic) | off (`--police-other-side`) |
| Crash | `CRASH` | an ambulance or fire engine, or two kinds of responder, with red / blue flashing lights on them, for 3 s: boxes that grow while we drive towards them, or that stand still while we stand (a responder driving past a waiting car is not a crash); ends as soon as they leave the frame | on |
| Crash > Other side | `CRASH_OTHER_SIDE` | the same, entirely on the far left | on |
| Traffic > Standstill | `TRAFFIC_STANDSTILL` | not moving for 75 s behind a vehicle, no red or amber signal seen in the last 20 s | on |
| Traffic > Heavy | `TRAFFIC_HEAVY` | 150 s mostly slow or stopped, at least two separate stops, three or more vehicles in view, no red signal | on |
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
by a hash of the clip name: one half was mined for the hard negatives of v5 and v6, the other half was never
trained on and is only scored. The police cars in that half were found by scoring every 5th frame with a
classifier at a low threshold and checking every candidate by eye (`configs/police_truth_argoverse.txt`); the
full pipeline then runs over every frame (`tools/eval_vehicle.py clips --clip-half 1 --step 1
--police-truth ...`):

| 163 held-out clips, 37.9 minutes | crop size | hits in 10 frames | police clips found (of 9) | clips without police that got a Police report (of 154) |
|---|---|---|---|---|
| v4 | 128 px | 3 | 5 | 16 |
| v5 | 128 px | 3 | 7 | 12 |
| v6 | 128 px | 3 | 8 | 4 |
| v5 and v6 averaged | 128 px | 3 | 8 | 2 |
| v5 and v6 averaged | 160 px | 3 | 5 | 0 |
| v5 and v6 averaged (ships) | 160 px | 2 | 7 | 0 |

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
seen from behind). The two police cars missed are a DC Metropolitan Police SUV seen mostly from behind (0.72
at most at 160 px) and the car in clip 931b76ee, which the 160 px models score no higher than 0.46.

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

**Clips.** The full pipeline (`tools/eval_vehicle.py clips`, report manager as shipped, no global cooldown),
v4 at 128 px / v5 + v6 at 128 px / v5 + v6 at 160 px with two hits (ships):

| Footage | Minutes | POLICE | CRASH |
|---|---|---|---|
| dashcam clips without police or responders (`assets/videos`: two Commons drives at two encodings, two UK drives, two Argoverse logs) | 14.6 | 0 / 0 / 0 | 0 / 0 / 0 |
| Commons videos of police vehicles, mostly hand-held (19 clips) | | 9 / 8 / 10 clips | 2 / 0 / 1 clips |
| Commons videos of ambulances (13 clips) | | 0 / 1 / 2 clips | 6 / 7 / 7 clips |
| Commons videos of other responders and emergency lighting (11 clips) | | 3 / 3 / 1 clips | 7 / 6 / 5 clips |

The Police report among the other responders is "Toronto Police responding", a real police car. The two
ambulance clips that get one (Edinburgh and Sendai) peak at 0.89 and 0.86, just over the floor.

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
(25 val frames): at 224 px a fog threshold as low as 0.4 still catches only 3 of them, so fog recall is a
rough figure and the threshold stays at 0.70.

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
