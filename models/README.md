# Model files

Exported models live here but are not committed (see `.gitignore`); only this file and the
`*.labels.txt` files are. The weights are not published either, see [Why the weights are not
published](#why-the-weights-are-not-published). Build them locally with the steps below.

| File | What |
|------|------|
| `hazard_yolo11n.onnx` | The shipped hazard detector: YOLO11n, 640 px input, fp32, seven classes (pothole, cone, construction_sign, delineator, barricade, worker, debris). A copy of `hazard_yolo11n_v4.onnx`; the desktop pipeline and the Android build load this name. |
| `hazard_yolo11n_v4.onnx` | The same model under its versioned name: the v3 set plus Lost and Found road obstacles as `debris`. |
| `hazard_yolo11n_v3.onnx` | The previous shipped model, six classes (no debris). Kept for comparison. |
| `hazard_yolo11s_v3.onnx` | YOLO11s trained on the same v3 set, 640 px, six classes. Not the default: about 3x the compute of the nano. Use it with `--hazard-model` on the desktop or `-PhazardModel` / `-PhazardModel2` in the Android build. |
| `coco_yolo11n.onnx` | Stock COCO YOLO11n, 416 px, used for vehicles, people and animals. |
| `hazard_yolo11n_dyn.onnx`, `coco_yolo11n_dyn.onnx` | The same two detectors exported with an open input height and width (`export_onnx.py --dynamic`). The Android build bundles these when they exist: the app fixes the shape per session to the frame's aspect (640x384 for 16:9), which needs 40 % less work than the square for the same pixels. The desktop loads the square files unless given these with `--rect`. Identical output to the square export at 640x640. |
| `hazard_yolo11n_dyn.qdq.onnx` | int8 copy of `hazard_yolo11n_dyn.onnx` (`tools/quantize_onnx.py --static`), the hazard model the Android build bundles when it exists: about 1.5x the frames a second on a mid-range phone, with the same Construction and Police scores on the held-out Argoverse clips and no new false pothole (`docs/model-research.md`). |
| `vehicle_cls.onnx` | Second stage on COCO car / truck / bus boxes: YOLO11n-cls, 160 px (the pipeline reads the size from the export's `imgsz` metadata, since the input is dynamic), four classes (ambulance, civilian, fire, police). A copy of `vehicle_v8.onnx`. An optional `vehicle_cls2.onnx` is averaged with it per class (desktop and app); none ships since v8. Turns a COCO vehicle into `police_vehicle`, `ambulance` or `fire_truck`. |
| `vehicle_yolo11n.onnx`, `vehicle_v2.onnx` ... `vehicle_v8.onnx`, `vehicle_s192.onnx` | The rounds of the vehicle classifier (see below); only v8 is loaded. |
| `weather_cls.onnx` | Whole-frame weather: YOLO11n-cls, 224 px, four classes (clear, fog, rain, snow). A copy of `weather_yolo11n.onnx`. Feeds the Fog and Slippery road rules. |
| `surface_cls.onnx` (optional) | Road-surface patch classifier (dry, wet, water, slush, snow, ice) for Flood, Icy road, Unplowed road and Slippery road. Without it those rules stay silent. |

Each model has a `<name>.labels.txt` next to it (one class per line, in output order).

## Building the models

Setup: `pip install -r requirements.txt -r requirements-train.txt -c constraints.txt`. Training
needs a CUDA build of torch (see the comment in `requirements-train.txt`); the export alone runs
on the CPU build. The datasets folder is `WAZEOMATIC_DATASETS`, else `datasets:` in
`configs/local.yaml`, else `<repo>/datasets` (`wazeomatic/paths.py`).

Shipped hazard model:

```powershell
python tools/fetch_datasets.py                 # six archive-based sources, about 23 GB; --list shows sizes and licences
python tools/fetch_rdd2022_potholes.py         # pothole images of the complete RDD2022 release
python tools/fetch_argoverse_hazards.py --stage annotations
python tools/fetch_argoverse_hazards.py --stage images --logs 100 --every 4
python tools/argoverse_to_yolo.py
python tools/prepare_datasets.py --tile 1024 --tile-sources starline roadwork argoverse --tile-size roadwork=1080 --tile-size argoverse=1280
python tools/train.py --name hazard_v3n        # yolo11n.pt, 640 px, 40 epochs, batch 32
python tools/export_onnx.py --weights runs/train/hazard_v3n/weights/best.pt --name hazard_yolo11n_v3 --ship
```

`prepare_datasets.py` writes `<datasets>/wazeomatic_hazards_v3` and stops if a source folder is
missing (`--allow-missing` builds a smaller, different set). `export_onnx.py` writes
`models/<name>.onnx`; `--ship` also copies it to `models/hazard_yolo11n.onnx` and refuses a
checkpoint that is not a YOLO11n unless `--force` is given. Without `--name` the file is called
`hazard_<arch>_<imgsz>.onnx`, for example `hazard_yolo11n_640.onnx`.

The v3 nano was continued from an earlier checkpoint rather than trained from the COCO weights
(`docs/model-research.md`, data expansion section), so a run with the defaults above gives a model
of the same kind, not the identical weights.

The shipped v4 adds a seventh class, `debris`, from Lost and Found (`tools/laf_to_yolo.py --class-id
6`), continued from the v3 weights at a low learning rate. Its data file lists the v3 train images
once and the Lost and Found train tiles twice (to lift their share), and the two val subsets; the
class names are the six above plus `debris`:

```yaml
train: [<datasets>/wazeomatic_hazards_v3/images/train, <datasets>/laf_yolo/images/train, <datasets>/laf_yolo/images/train]
val: [<datasets>/wazeomatic_hazards_v3/val_subset.txt, <datasets>/laf_yolo/val_subset.txt]
names: {0: pothole, 1: cone, 2: construction_sign, 3: delineator, 4: barricade, 5: worker, 6: debris}
```

```powershell
python tools/train.py --data <datasets>/hazard_v4.yaml --model runs/train/hazard_v3n/weights/best.pt --epochs 25 --lr0 0.002 --warmup 1 --name hazard_v4n
python tools/export_onnx.py --weights runs/train/hazard_v4n/weights/best.pt --name hazard_yolo11n_v4 --ship
python tools/export_onnx.py --weights runs/train/hazard_v4n/weights/best.pt --name hazard_yolo11n_v4_dyn --dynamic --ship
```

The second line writes the open-shape copy (`hazard_yolo11n_dyn.onnx`). The phone build prefers an int8 copy of
it, made from 300 calibration images (100 from each folder; about 10 minutes):

```powershell
python tools/quantize_onnx.py models/hazard_yolo11n_dyn.onnx --static --calib <datasets>/roadwork/images --calib <datasets>/rdd2022_us/United_States/train/images --calib <datasets>/bdd100k_weather/data/images/train
```

That writes `hazard_yolo11n_dyn.qdq.onnx` (3.3 MB) and its labels. The head's box decoding and class branch stay
float; why, and the accuracy checks behind it, are in `docs/model-research.md` (int8 hazard model). Re-check
those before shipping a new int8 file.

How v4 compares with v3 is in `docs/model-research.md` (hazard v4).

Small model:

```powershell
python tools/train.py --model yolo11s.pt --batch 24 --name hazard_v3s
python tools/export_onnx.py --weights runs/train/hazard_v3s/weights/best.pt --name hazard_yolo11s_v3
```

COCO model (downloads the stock `yolo11n.pt` through ultralytics, so it needs
`requirements-train.txt`; no dataset or training):

```powershell
python tools/export_onnx.py --coco             # models/coco_yolo11n_416.onnx, copied to models/coco_yolo11n.onnx
python tools/export_onnx.py --coco --dynamic   # models/coco_yolo11n_416_dyn.onnx, copied to models/coco_yolo11n_dyn.onnx
```

Second-stage classifiers (image-folder sets, then `tools/train_classifier.py`):

```powershell
python tools/fetch_commons_vehicles.py         # police / ambulance / fire photos from Wikimedia Commons
python tools/fetch_openimages.py               # Open Images Ambulance, Car, Taxi, Van boxes
python tools/build_vehicle_crops.py            # <datasets>/vehicle_crops
python tools/train_classifier.py vehicle --ship
# optional second model, averaged with the first: copy it to models/vehicle_cls2.onnx
#   (+ vehicle_cls2.labels.txt); the app bundles it when present (v5 + v6 shipped that way before v8)
# whole-track mining on clips no evaluation scores (see v8 below):
python tools/mine_vehicle_tracks.py scan --clips <datasets>/argoverse2/clips_mine --work <datasets>/mining_tracks/new
python tools/mine_vehicle_tracks.py sheets --work <datasets>/mining_tracks/new
python tools/mine_vehicle_tracks.py label --work <datasets>/mining_tracks/new --yes 0,3 --skip 5
python tools/build_vehicle_crops.py --sources tracks --track-repeat 3
python tools/build_scene_sets.py weather       # BDD100K frames by weather label, plus synthetic fog in train
python tools/train_classifier.py weather --ship
python tools/fetch_rscd.py; python tools/build_scene_sets.py surface
python tools/train_classifier.py surface --ship
```

The vehicle classifier took eight rounds. v1 learnt photo style instead of livery: every police picture was a
hand-held photo and every civilian car a dashcam crop, so it called 22 % of hand-held civilian look-alikes
police. v2 added Open Images photos of ordinary cars, taxis and vans as civilian. v3 added crops mined from
40,000 BDD100K dashcam frames: `tools/mine_vehicle_crops.py scan` scores every vehicle with the current model,
`sheets` lays the high scorers out as contact sheets, and the ones checked by hand go to
`<datasets>/mined_vehicles/<class>` (51 police, 23 ambulance, 1 fire engine, 279 civilian look-alikes) with an
exclude list that takes the real police cars out of the civilian negatives. v4 added ambulances from the UK,
Japan, Germany, France, Australia and the Netherlands (Commons): v3 called Battenberg and box-bodied
ambulances police, because police crops outnumbered ambulance crops four to one.

v4 still called police on a parked grey sedan, an Infiniti behind a UPS truck or a white SUV in a quarter of
the Argoverse 2 clips. v5 and v6 are hard-negative rounds on that footage: the clips are split in two by a
hash of their name (`tools/mine_vehicle_crops.py --clip-half`), one half is mined and checked by eye (v5: 100
road-work logs, 121 civilian look-alikes and 16 police crops; v6: 200 more logs fetched with
`tools/fetch_argoverse_hazards.py --skip 100`, 178 look-alikes, 2 police, 1 ambulance), and the other half is
only ever scored (`configs/police_truth_argoverse.txt`). Each round fixed the look-alikes it was shown and
found new ones, but not the same ones, so the two models are averaged: on the 163 held-out clips that cut the
clips with a false Police report from 12 (v5) and 4 (v6) to 2, with 8 of the 9 police clips found, all at 128 px
crops (before the pipeline read the trained input size from the export). At the trained 160 px, with Police on
two hits in ten frames, the pair finds 7 of 9 with no false report. A YOLO11s-cls at 192 px (`vehicle_s192`)
was not better than either. Measured numbers are in
[docs/waze-coverage.md](../docs/waze-coverage.md#measured).

v7 (not loaded) tried two fixes at once: 1,894 taxi and private-hire photos from Japan, the UK, Canada,
Germany, France, Australia and New York as civilian (`fetch_commons_vehicles.py --only civilian_taxi`, crops
named `tx_*`), because taxis were the false Police reports on the Commons clips, and `--res-jitter 0.5`,
because the two police cars missed on Argoverse score lower on larger, sharper crops. Taxis called police
dropped from 8.9 % (v5 + v6) to 1.3 %, but on the held-out Argoverse clips v7 found 6 of 9 police cars, alone
or averaged with v5 and v6, against 7 of 9 for the pair, and still missed the same two. The taxi crops stay in
the set for the next round.

v8 (loaded) fixed the misses by changing how crops are mined. Both crop miners so far ranked single crops by the
classifier's own score, so a police car it scored 0.00 from behind never reached a contact sheet, and the two
missed cars were exactly that: slicktop SUVs and sedans seen from the rear. `tools/mine_vehicle_tracks.py`
follows each COCO vehicle from frame to frame (box overlap, split where the colour histogram or the box size
jumps, so an ID swap does not mix two cars), shows a track when any one of its frames scores 0.3, and a single
yes or no labels every frame of it, rear and far views included (`<datasets>/mined_tracks/<class>/<clip>_t<k>/`,
crops `mt_*`, three degraded copies of each train frame; for v8 the split went by track, it now goes by clip). On the mined
Argoverse half and on 300 new logs that no evaluation uses (`fetch_argoverse_hazards.py --clips-out
clips_mine`), 24 tracks were police (406 frames: Miami, Pittsburgh, DC and Detroit cars, many that v5 and v6
scored below 0.2) and 98 look-alike tracks gave 209 civilian frames. v8 is trained like v5 to v7 on the whole
set with these added. It found three police clips in the held-out half that the truth list did not have,
because the list was built from crops some earlier model had scored 0.2: a sedan with its light bar on at a
work zone, another from behind with a bike rack and a DC SUV. A fourth (a Miami SUV half out of the frame)
turned up while checking v8's near misses. With those added (13 police clips), v8 alone finds 12 with no false
clip; v5 + v6 find 7 and every average with v5 or v6 loses most of the gain. On held-out crops v8 keeps 70 %
of police at 0.3 % false civilians against 60 % for the pair, and calls 12 of 554 ambulances police against 17.
Its new weak spot is a Toronto box ambulance, white with blue graphics, that it scores 0.85 to 0.91 police
([docs/waze-coverage.md](../docs/waze-coverage.md#measured)).

## Retired files

These are earlier four-class models (pothole, cone, construction_sign, delineator) from before the
v3 set. They are kept for comparison only and nothing loads them by default. Their label files are
still in this folder:

- `hazard_yolo11n_320.labels.txt`, `hazard_yolo11n_320.int8.labels.txt`
- `hazard_yolo11n_416.labels.txt`, `hazard_yolo11n_416.int8.labels.txt`
- `hazard_yolo11n_640.labels.txt`
- `hazard_yolo11n_640b.labels.txt`, `hazard_yolo11n_640b.int8.labels.txt`
- `hazard_yolo11n_640c.labels.txt`, `hazard_yolo11n_640c.int8.labels.txt`

The `coco_yolo11n_{320,416,640}` files and their `.int8` copies are the stock COCO model exported at
other sizes for the latency tables in `docs/model-research.md` (`tools/quantize_onnx.py` makes the
int8 copies). No int8 copy and no 320 or 416 px export of the six-class model exists.

## Why the weights are not published

The trained weights are neither committed nor released. Two of the training sources do not allow
it as things stand: the StarLine Roadwork_Cones dataset card states no licence, and Argoverse 2 is
CC BY-NC-SA 4.0 (non-commercial, share-alike). Until that is settled the models are built locally
from the public sources with the pipeline above. The code's licence is in [LICENSE](../LICENSE);
third-party datasets, models and their terms are listed in [NOTICE.md](../NOTICE.md) and in the
Datasets table of `docs/model-research.md`.

See `docs/model-research.md` for the reasoning behind the model choice and the measured sizes,
latencies and accuracy.
