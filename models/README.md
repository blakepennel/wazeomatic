# Model files

Exported models live here but are not committed (see `.gitignore`); only this file and the
`*.labels.txt` files are. The weights are not published either, see [Why the weights are not
published](#why-the-weights-are-not-published). Build them locally with the steps below.

| File | What |
|------|------|
| `hazard_yolo11n.onnx` | The shipped hazard detector: YOLO11n, 640 px input, fp32, six classes (pothole, cone, construction_sign, delineator, barricade, worker). A copy of `hazard_yolo11n_v3.onnx`; the desktop pipeline and the Android build load this name. |
| `hazard_yolo11n_v3.onnx` | The same model under its versioned name. |
| `hazard_yolo11s_v3.onnx` | YOLO11s trained on the same v3 set, 640 px, six classes. Not the default: about 3x the compute of the nano. Use it with `--hazard-model` on the desktop or `-PhazardModel` / `-PhazardModel2` in the Android build. |
| `coco_yolo11n.onnx` | Stock COCO YOLO11n, 416 px, used for vehicles, people and animals. |
| `vehicle_cls.onnx` | Second stage on COCO car / truck / bus boxes: YOLO11n-cls, 160 px, four classes (ambulance, civilian, fire, police). A copy of `vehicle_v4.onnx`. Turns a COCO vehicle into `police_vehicle`, `ambulance` or `fire_truck`. |
| `vehicle_yolo11n.onnx`, `vehicle_v2.onnx`, `vehicle_v3.onnx`, `vehicle_v4.onnx` | The four rounds of the vehicle classifier (see below); only v4 is loaded. |
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

The nano that ships was continued from an earlier checkpoint rather than trained from the COCO
weights (`docs/model-research.md`, data expansion section), so a run with the defaults above gives
a model of the same kind, not the identical weights.

Small model:

```powershell
python tools/train.py --model yolo11s.pt --batch 24 --name hazard_v3s
python tools/export_onnx.py --weights runs/train/hazard_v3s/weights/best.pt --name hazard_yolo11s_v3
```

COCO model (downloads the stock `yolo11n.pt` through ultralytics, so it needs
`requirements-train.txt`; no dataset or training):

```powershell
python tools/export_onnx.py --coco             # models/coco_yolo11n_416.onnx, copied to models/coco_yolo11n.onnx
```

Second-stage classifiers (image-folder sets, then `tools/train_classifier.py`):

```powershell
python tools/fetch_commons_vehicles.py         # police / ambulance / fire photos from Wikimedia Commons
python tools/fetch_openimages.py               # Open Images Ambulance, Car, Taxi, Van boxes
python tools/build_vehicle_crops.py            # <datasets>/vehicle_crops
python tools/train_classifier.py vehicle --ship
python tools/build_scene_sets.py weather       # BDD100K frames by weather label, plus synthetic fog in train
python tools/train_classifier.py weather --ship
python tools/fetch_rscd.py; python tools/build_scene_sets.py surface
python tools/train_classifier.py surface --ship
```

The vehicle classifier took four rounds. v1 learnt photo style instead of livery: every police picture was a
hand-held photo and every civilian car a dashcam crop, so it called 22 % of hand-held civilian look-alikes
police. v2 added Open Images photos of ordinary cars, taxis and vans as civilian. v3 added crops mined from
40,000 BDD100K dashcam frames: `tools/mine_vehicle_crops.py scan` scores every vehicle with the current model,
`sheets` lays the high scorers out as contact sheets, and the ones checked by hand go to
`<datasets>/mined_vehicles/<class>` (51 police, 23 ambulance, 1 fire engine, 279 civilian look-alikes) with an
exclude list that takes the real police cars out of the civilian negatives. v4 added ambulances from the UK,
Japan, Germany, France, Australia and the Netherlands (Commons): v3 called Battenberg and box-bodied
ambulances police, because police crops outnumbered ambulance crops four to one. Measured numbers are in
[docs/waze-coverage.md](../docs/waze-coverage.md#measured).

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
