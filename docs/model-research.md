# Model research

Goal: a detector that runs on a Pixel 9 Pro alongside Waze, uses well under 1 GB of memory, and reaches at least a few inferences per second on road hazards. The same model should also run on a desktop for simulation and on a Raspberry Pi later.

## Decision

- **Model:** Ultralytics YOLO11n, fine-tuned on seven hazard classes (pothole, cone, construction_sign, delineator, barricade, worker, debris), plus a stock COCO YOLO11n for vehicles and animals.
- **Runtime:** ONNX Runtime everywhere. On Android, the XNNPACK execution provider on the CPU.
- **Input size:** 640 px for the hazard model, 416 px for the COCO model. The hazard model also runs on a zoom crop of the road ahead (two-pass zoom). Static shape export, opset 17, no NMS in the graph (NMS is done in our code).
- **Precision:** fp32. The seven-class model has no int8 copy and no 320 or 416 px export; the int8 and smaller files measured below belong to the earlier four-class models and the stock COCO model.

The sections under "Measured results" are a dated record of how this was reached, starting from a 416 px four-class model. The current defaults are summarised at the [end of this page](#current-defaults-2026-11-09).

## Candidate models

Latency class is relative, for CPU inference at 320–640 px. Ultralytics reports 56 ms for YOLO11n at 640 px on CPU with ONNX ([YOLO11 docs](https://docs.ultralytics.com/models/yolo11/)). Sizes are approximate fp32 ONNX files.

| Model | Params | ONNX size (fp32) | Licence | Latency class | Notes |
|-------|--------|------------------|---------|---------------|-------|
| **YOLO11n** (chosen) | 2.6 M | ~10 MB | AGPL-3.0 | fast | 39.5 mAP50-95 on COCO at 640; mature training and export tooling |
| YOLO11s | 9.4 M | ~36 MB | AGPL-3.0 | medium | 47.0 mAP50-95; fallback if nano accuracy is too low |
| YOLOX-nano / tiny | 0.9 M / 5.1 M | ~4 / ~20 MB | Apache-2.0 | fast | Licence-friendly alternative, older training code |
| RT-DETR (R18 / v2-S) | ~20 M | ~80 MB | Apache-2.0 | slow on mobile CPU | No NMS needed; too heavy for the phone at our frame rate |
| EfficientDet-Lite0 | 3.2 M | ~13 MB (TFLite) | Apache-2.0 | medium | Native to LiteRT; would need a TFLite path instead of ONNX |
| MobileNet-SSD v2 | ~4–6 M | ~17 MB | Apache-2.0 | fast | Weak on small objects such as distant potholes |

**Licence:** YOLO11 is published under AGPL-3.0 or an enterprise licence ([Ultralytics](https://docs.ultralytics.com/models/yolo11/)). For a course project with public source this is acceptable. If the project is ever distributed under other terms, switch to one of the Apache-2.0 models above. The pre/post-processing in `wazeomatic/detector.py` is model-agnostic enough that YOLOX needs only a different decoder.

## Runtime options per device

| Device | Runtime | Accelerator | Notes |
|--------|---------|-------------|-------|
| Desktop (Windows) | onnxruntime (Python) | CPU EP; CUDA EP optional | Reference implementation and benchmarks |
| Pixel 9 Pro | onnxruntime-android (Maven) | **XNNPACK EP** on CPU | XNNPACK ships in the Android package and must be registered on the session ([ORT XNNPACK](https://onnxruntime.ai/docs/execution-providers/Xnnpack-ExecutionProvider.html)). Recommended: XNNPACK threads = physical cores, ORT intra-op threads = 1, spinning off. |
| Pixel 9 Pro (not used) | onnxruntime-android | NNAPI EP | NNAPI is deprecated since Android 15 ([Android NNAPI migration guide](https://developer.android.com/ndk/guides/neuralnetworks/migration-guide)), so we do not rely on it. |
| Pixel 9 Pro (fallback) | LiteRT (TFLite) | GPU delegate | Google's recommended path after NNAPI. Only if XNNPACK latency is not enough; needs a TFLite export. |
| Raspberry Pi 4/5 | onnxruntime (Python, aarch64) | CPU | Same Python pipeline as the desktop |
| Emulator (x86_64) | onnxruntime-android | CPU | Functional testing only; timings are not representative |

## Memory and compute budget

| Device | CPU | RAM | Notes |
|--------|-----|-----|-------|
| Pixel 9 Pro | Tensor G4, 8 cores | 16 GB ([Tom's Guide](https://www.tomsguide.com/phones/google-pixel-phones/google-pixel-9-pro)) | Teammate measured about 10.5 GB available |
| Raspberry Pi 5 | 4x Cortex-A76 @ 2.4 GHz | 1, 2, 4, 8 or 16 GB ([raspberrypi.com](https://www.raspberrypi.com/products/raspberry-pi-5/)) | |
| Raspberry Pi 4 B | 4x Cortex-A72 @ 1.8 GHz | 1, 2, 3, 4 or 8 GB ([raspberrypi.com](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/)) | |
| Raspberry Pi Zero 2 W | 4x Cortex-A53 | 512 MB | Too slow; listed for completeness |

A YOLO11n ONNX file is about 10 MB (about 3 MB int8) and the session needs well under 200 MB at runtime, even with two models loaded. **RAM is not the constraint on any of these devices; CPU latency is.** Expectations to be confirmed by measurement:

- Pi 4 CPU, nano model at 320–416 px: a few FPS.
- Tensor G4 CPU with XNNPACK: roughly 10–30 inferences/s (TBD).

Since the Report Manager needs 3 hits in 10 frames, even 3–5 inferences/s confirms a hazard within 1–3 s.

## Datasets

The datasets are not committed. Every tool resolves the datasets folder the same way (`wazeomatic/paths.py`): the `WAZEOMATIC_DATASETS` environment variable, else `datasets:` in `configs/local.yaml` (git-ignored), else `<repo>/datasets`. `python tools/fetch_datasets.py` downloads the archive-based sources (1 to 5 and 7 below) into the layout `tools/prepare_datasets.py` reads; `--list` shows each source's size, licence and whether it is already on disk. The complete RDD2022 pothole images and Argoverse 2 (6) are filtered while downloading and keep their own scripts (`tools/fetch_rdd2022_potholes.py`, `tools/fetch_argoverse_hazards.py`).

| # | Dataset | Source | Licence | Size | Used for |
|---|---------|--------|---------|------|----------|
| 1 | Pothole Dataset IVCNZ | [jaygala24/pothole-detection](https://github.com/jaygala24/pothole-detection) release v1.0.0 | MIT | 1,243 images, YOLO labels | pothole |
| 2 | RDD2022, United States subset (Arya, Maeda et al.) | HF mirror [cvtechniques/Road_Damage_Detection_USA](https://huggingface.co/datasets/cvtechniques/Road_Damage_Detection_USA); original S3 links are dead, [figshare](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547) full set is 13 GB | CC BY 4.0 (to confirm on figshare; the HF mirror card states no licence) | 4,805 images, Pascal VOC; D00/D10/D20 cracks, D40 pothole (only 86 boxes) | pothole; crack-only images as hard negatives (cracks are not reportable in Waze) |
| 3 | Traffic_Cone | [Junhan0518/Traffic_Cone](https://huggingface.co/datasets/Junhan0518/Traffic_Cone) | Apache-2.0 | only `cone_dataset.tar.gz` used: 263 Pascal VOC images, 48 annotated with `trafficcone` (the repo's `archive.zip` turned out to be a forklift/person dataset and was discarded) | cone |
| 4 | Roadwork_Cones_Dataset | [StarLineResearch/Roadwork_Cones_Dataset](https://huggingface.co/datasets/StarLineResearch/Roadwork_Cones_Dataset) | not stated on the card; research/course use only until confirmed | 5.7 GB, 4,674 images from four vehicle-mounted cameras (2880x1860), 31,402 boxes: cone 11,520, roadworks 2,813, vertical_pannel 17,069; session-based train/test split kept as our train/val | cone, construction_sign, delineator (best domain match: real road scenes) |
| 5 | ROADWork (CMU) | HF mirror [anuragxel/roadwork-dataset](https://huggingface.co/datasets/anuragxel/roadwork-dataset), project page cs.cmu.edu/~roadwork | ODC-By 1.0 (images from external sources keep their own terms) | `images.zip` 10.6 GB, 8,550 images; we use the GPS-based split with sign subtypes: 5,318 train / 2,098 val dashcam frames (1920x1080) from 18 US cities, 82k boxes (Cone 20k, Vertical Panel 10.7k, Tubular Marker 8.8k, Drum 6.6k, TTC signs 7k, Barricade 5.2k, Barrier 4.1k, Worker 4k, Arrow Board 0.8k ...) | cone, delineator (drum, tubular marker, vertical panel), construction_sign (all TTC sign subtypes, arrow and message boards), barricade (barricade, barrier), worker. Fence, work vehicles, police and equipment are left unlabelled (hard negatives) |
| 6 | Argoverse 2 sensor dataset | public S3 bucket `s3://argoverse/datasets/av2/sensor` (anonymous HTTPS listing works; `tools/fetch_argoverse_hazards.py`) | CC BY-NC-SA 4.0 | 1,000 logs with 3D cuboids; 630 contain CONSTRUCTION_CONE / CONSTRUCTION_BARREL. We fetched the 100 richest logs, every 4th front-camera frame (1550x2048 portrait, cropped to rows 256-1536), projected the cuboids with the av2 API (`tools/argoverse_to_yolo.py`, motion compensated, cuboids without lidar returns skipped): 5,501 frames, 20k cone and 25.8k barrel boxes, 957 hazard-free frames kept as negatives; log-level train/val split kept | cone, delineator (barrels) at every distance, six US cities, the only source with the full approach sequence of each object |
| 7 | Potholes Detection (Roboflow export) | HF mirror [Ryukijano/Pothole-detection-Yolov8](https://huggingface.co/datasets/Ryukijano/Pothole-detection-Yolov8) | CC BY 4.0 | 300 images, 225 train boxes | pothole (small; variety) |
| - | UK Road DashCam | [aap9002/UK-Road-DashCam](https://huggingface.co/datasets/aap9002/UK-Road-DashCam) | MIT | 4K Nextbase front-camera clips, 3 min each; two converted to 1080p in `assets/videos/uk_dashcam_002_1080p.mp4` and `..._006_...` | hazard-free real footage for the false-report rate (overcast, suburban and rural, dashboard in the bottom 15 %) |
| - | Argoverse 2 clips | as 6 | CC BY-NC-SA 4.0 | two full-rate (20 Hz) val logs encoded as video: `assets/videos/argo_02678d04_clean.mp4` (no hazards labelled) and `assets/videos/labelled/argo_2ff4f798_workzone.mp4` (120 cone/barrel tracks) | real-sequence false-report rate and a real work-zone event check |

Gaps at sprint 1: **police** had no adequate open dataset (the only Hugging Face set found had 8 images) and **crash** neither. Both were added later without a detector class: a classifier on the COCO vehicle boxes (Wikimedia Commons photos, Open Images and hand-checked dashcam crops, `models/README.md`) and scene rules on top of it ([waze-coverage.md](waze-coverage.md)). Pothole positives come from dataset 1 (close-up, varied cameras), from the pothole images of the complete RDD2022 release (7,440 images, 10,702 pothole boxes, added in the small-object pass below) and from dataset 7; dataset 2 adds US street-level context but only 86 pothole boxes, so its value is mainly as negatives. Barrels (drums) are labelled as `delineator` since the v3 set (ROADWork drums, Argoverse barrels).

## Class schema and Waze mapping

Defined in `wazeomatic/schema.py`; model classes in `configs/classes.yaml`.

| Model class | Model | hazard_type | Waze path (schema.py) | Sprint 1 |
|-------------|-------|-------------|------------------------|----------|
| pothole | hazard | `HAZARD_ON_ROAD_POTHOLE` | Hazard > Pothole | yes |
| cone, construction_sign, delineator, barricade, worker | hazard | `HAZARD_ON_ROAD_CONSTRUCTION` | Hazard > Construction | yes (barricade and worker since the v3 model) |
| dog, horse, sheep, cow, bear | COCO | `HAZARD_SHOULDER_ANIMALS` | Hazard (no subtype; Waze removed Animals) | yes |
| car, truck, bus, standing beside the road above 50 km/h | COCO | `HAZARD_SHOULDER_CAR_STOPPED` | Hazard > Car on shoulder | off (`--shoulder-cars`) |
| debris (v4 model) | hazard | `HAZARD_ON_ROAD_OBJECT` | Hazard > Object | added later |
| police_vehicle | vehicle classifier on COCO boxes | `POLICE` | Police | added later |
| ambulance, fire_truck + flashing lights | vehicle classifier + scene rule | `CRASH` | Crash | added later |

The table shows the sprint 1 view; the complete list of the 20 report types, including the scene rules for
traffic, weather, blocked lanes and dark signals, is [waze-coverage.md](waze-coverage.md). The shipped hazard
model (v4) has seven classes, in the order pothole, cone, construction_sign, delineator, barricade, worker,
debris (`configs/classes.yaml`, `models/hazard_yolo11n.labels.txt`).

**Menu names:** since the 2024 redesign Waze lists the hazard subtypes Construction, Car on shoulder, Broken traffic light, Pothole and Object directly under Report > Hazard, with Hazard preselected ([Waze Help](https://support.google.com/waze/answer/13739290?hl=en)); Animals/Roadkill were removed, so animals go out as a plain Hazard. The `waze_subcategory` strings in `schema.py` follow this flat layout. The `hazard_type` keys are the contract and never change; the macro and the mock app tap by on-screen label.

## Training and evaluation procedure

This is the procedure as the tools work now. Install the training dependencies first: `pip install -r requirements.txt -r requirements-train.txt -c constraints.txt` (`constraints.txt` holds the tested versions; training needs a CUDA build of torch, see the comment in `requirements-train.txt`).

1. **Fetch.** `python tools/fetch_datasets.py` downloads what is missing of the six archive-based sources (`pothole_ivcnz`, `rdd2022_us`, `cone_voc`, `starline`, `roadwork`, `pothole_ryukijano`; about 23 GB), `--only <source ...>` picks some, `--list` shows sizes, licences and what is on disk. Two sources have their own scripts: `python tools/fetch_rdd2022_potholes.py`, and for Argoverse 2 `python tools/fetch_argoverse_hazards.py --stage annotations`, `python tools/fetch_argoverse_hazards.py --stage images --logs 100 --every 4`, then `python tools/argoverse_to_yolo.py`.
2. **Prepare.** `python tools/prepare_datasets.py` converts every source to YOLO txt using the class order in `configs/classes.yaml`, keeps up to 1,500 crack-only RDD images with empty labels as hard negatives, and makes a random 85/15 train/val split for sources without their own (StarLine keeps its session-based split, ROADWork its GPS-based split, Argoverse its log-level split). Output goes to `<datasets>/wazeomatic_hazards_v3` (`--out` and `--datasets` override). It stops when a source folder is missing; `--allow-missing` builds a smaller set from what is there. It writes `data.yaml`, plus `data_train.yaml` and `val_subset.txt`: the same set with a fixed 3,000-image subset of the val split for the check after every epoch. Without `--tile` large images are downscaled to 1280 px; the v3 set described below was tiled: `--tile 1024 --tile-sources starline roadwork argoverse --tile-size roadwork=1080 --tile-size argoverse=1280` (ROADWork and Argoverse also keep the downscaled full frame by default). `--check <source>` renders labelled samples to `runs/dataset_check/` to verify box conventions.
3. **Train.** `python tools/train.py --name hazard_v3n` fine-tunes `yolo11n.pt` on `<datasets>/wazeomatic_hazards_v3/data_train.yaml` at 640 px for 40 epochs with batch 32 (no vertical flips, mild rotation/shear, HSV jitter for exposure, mosaic off for the last 10 epochs, cosine LR). The small model: `python tools/train.py --model yolo11s.pt --batch 24 --name hazard_v3s`. Keep the dataset on an SSD. The shipped nano was not trained from the COCO weights but continued from the earlier 640c checkpoint (`--model <640c best.pt> --lr0 0.005 --close-mosaic 8`, see below), so a run with the defaults will not reproduce its numbers exactly.
4. **Export.** `python tools/export_onnx.py --weights runs/train/hazard_v3n/weights/best.pt --name hazard_yolo11n_v3 --ship` writes `models/hazard_yolo11n_v3.onnx` (640 px, opset 17, static shape, no NMS) and its labels file, and copies both to the stable name `models/hazard_yolo11n.onnx` that the desktop pipeline and the Android build load. Without `--ship` only `models/<name>.onnx` is written; without `--name` the name is `hazard_<arch>_<imgsz>` with the architecture read from the checkpoint (for example `hazard_yolo11s_640`). `--ship` refuses a checkpoint that is not a YOLO11n unless `--force` is given. The small model: `--weights runs/train/hazard_v3s/weights/best.pt --name hazard_yolo11s_v3`. The stock model: `python tools/export_onnx.py --coco` (416 px, refreshes `models/coco_yolo11n.onnx`).
5. **Evaluate.** `python tools/eval_events.py` scores a model on the v3 val split by default (box AP by object size, report events on synthetic clips, false reports on the hazard-free clips in `assets/videos/`) with the shipped report floors; it takes the same detector flags as `run_video.py`. The v3 tables below used `--zoom-crop --limit 120 --frames 8 --seed 1`. Then replay the test videos with `python -m wazeomatic.run_video`.

`tools/quantize_onnx.py` (ONNX Runtime dynamic int8) and `tools/bench_onnx.py` (mean/p50/p90 ms per frame, `--compare <clip>` for fp32 vs int8 agreement) produced the size and latency tables below. They are not part of the shipped path: no int8 copy of the seven-class model is exported.

The acceptance line written in sprint 1 for sprint 2 was: pothole and cone mAP50 >= 0.5, no more than one false event per test clip, phone latency under 100 ms per frame. The sprint-1 figure (0.56 for pothole) was Ultralytics `val`, which ranks every box down to confidence 0.001. Measured the same way on the whole v3 validation split (14,597 images, 4,368 pothole and 21,089 cone boxes), the shipped v4 nano has pothole mAP50 0.676 and cone 0.579: met (construction_sign 0.58, delineator 0.53, barricade 0.43, worker 0.37; `YOLO.val`, 640 px). The tables below use `tools/eval_events.py`, which keeps only boxes at the detector's 0.25 cut-off and so reports a lower AP for the same model (v3 nano on the 1,201-image subset: cone 0.513, pothole 0.394; v4: 0.505 and 0.375). That is the operating point the phone runs at, and it is the number to compare between models; it is not the metric the target was written in. False events 0 across 5 clips and 12 minutes, met; phone latency not measured yet.

## Measured results

The subsections below are in the order the work happened and are kept as a dated record. "Shipped", "default" and the thresholds named in each one are those of that date; where they differ from today, the [current defaults](#current-defaults-2026-11-09) at the end apply.

Desktop numbers: AMD Ryzen 7 9800X3D, onnxruntime CPU provider, 4 intra-op threads (phone-like). Emulator and Pixel numbers come from the detector app's latency readout.

### Latency (ms per frame, inference only)

Desktop: stock COCO YOLO11n exported at each size, onnxruntime 1.23 CPU provider, 4 intra-op threads, 60 runs after warm-up (`tools/bench_onnx.py`). The fine-tuned hazard model has the same backbone and a 4-class head, so its cost is the same within noise.

| Setup | 320 fp32 | 320 int8 | 416 fp32 | 416 int8 | 640 fp32 | 640 int8 |
|-------|----------|----------|----------|----------|----------|----------|
| Desktop, ORT CPU (Ryzen 7 9800X3D, 4 threads), mean | 6.1 | 14.7 | 8.4 | 16.4 | 16.3 | 33.2 |
| Desktop, p90 | 8.3 | 16.5 | 13.4 | 18.7 | 20.3 | 35.7 |
| Desktop, hazard model (4 classes), mean | 3.3 | 9.3 | 5.3 | 12.8 | 11.3 | |
| Desktop, both models in `run_video` on a 720p clip, mean | | | 13.1 (p90 16.2) | | | |
| Emulator (x86_64, ORT XNNPACK), detector app readout | | | 12–40 (COCO model, 9–11 FPS end to end) | | | |
| Pixel 9 Pro (ORT XNNPACK) | TBD | TBD | TBD | TBD | TBD | TBD |
| Raspberry Pi (if available) | TBD | TBD | TBD | TBD | TBD | TBD |

Take-aways so far:

- At 416 px the whole two-model pipeline (hazard + COCO) costs about 17 ms per frame on the desktop, far below the 100 ms budget at 10 inferences/s.
- **Dynamic int8 is slower than fp32 on this x86 CPU** (the quantised convolutions fall back to slower kernels), even though the file shrinks from 10.6 MB to 2.9 MB. Int8 is therefore a size optimisation on desktop; whether it helps on the phone's ARM cores must be measured there before shipping the int8 file.
- 640 px fp32 is still only 16 ms on the desktop; the phone decides whether 416 or 640 is affordable.

### Phone latency (2026-11-23)

The first real-phone numbers: a OnePlus Nord N10 5G (Snapdragon 690: 2 x Cortex-A77 at 2.0 GHz, 6 x A55; 6 GB; Android 11), the slowest phone we have. Detector app with default settings on `urban_720p.mp4` (every 5th frame), ONNX Runtime 1.20 with XNNPACK, from the `WazeomaticPerf` log, median / p90 in ms after warm-up. The live camera (1280x960) gives the same times.

| Build | Hazard 640 | Zoom pass | COCO 416 | Whole frame | Frames/s |
|---|---|---|---|---|---|
| Before: square inputs, zoom pass after the full pass, 4 threads per model | 207 / 217 | 175 / 180 | 126 / 146 | **446 / 456** | 1.9 |
| Before, zoom pass off | 206 / 217 | | 124 / 144 | 240 / 251 | 3.1 |
| Zoom pass on its own thread, square inputs, 2 threads | 280 / 296 | 285 / 312 | 289 / 298 | 320 / 338 | 2.4 |
| Zoom pass on its own thread, rect inputs, 4 threads | 239 / 261 | 219 / 251 | 186 / 223 | 250 / 272 | 3.0 |
| ... 1 thread | 178 / 180 | 135 / 138 | 171 / 176 | 196 / 201 | 3.5 |
| **... 2 threads (shipped default)** | 158 / 165 | 129 / 176 | 154 / 159 | **178 / 186** | 3.9 |
| ... 3 threads | 185 / 201 | 159 / 197 | 150 / 170 | 198 / 212 | 3.6 |
| Shipped default, zoom pass off (4 threads) | 135 / 145 | | 94 / 120 | 142 / 154 | 4.4 |

What changed (`OnnxDetector.kt`, `Pipeline.kt`):

- **Each pass has its own session**, so the zoom pass runs alongside the full frame and COCO instead of after the full frame. Alone this took the frame from 446 to 320 ms.
- **Rect inputs.** The detectors are exported with an open input height and width (`tools/export_onnx.py --dynamic`; identical output to the square export at 640x640) and the app fixes the shape per session to what it is fed: 640x384 for a 16:9 frame, 640x288 for the zoom crop, 416x256 for COCO, instead of 640x640 and 416x416. A 16:9 frame in a 640 square is 44 % grey padding, the zoom crop 55 %. The image is at the same scale either way. This is the shape Ultralytics `val` uses too (rectangular batches).
- **Two XNNPACK threads per session** when three passes run at once (`ort_threads` = 0: the cores divided by the passes, at most 4). With four each, twelve threads fight over the two fast cores.
- Preprocessing writes the input through a float array in one copy instead of three `FloatBuffer.put` calls per pixel.

Accuracy of the rect inputs, desktop, every 3rd frame of the six real clips with the zoom pass (`--rect` against the same open-shape model at the square; box matched at IoU 0.5, same label):

| | Square boxes | Rect boxes | Matched |
|---|---|---|---|
| Hazard boxes at 0.45 | 2,336 | 2,252 | 2,164 (93 % of square, 96 % of rect) |
| COCO car / truck / bus / person at 0.45 | 4,280 | 4,079 | 3,787 (88 % / 93 %) |

- On the labelled work-zone clip the two agree within 1 % (cones 617 against 625, delineators 1,343 against 1,346).
- Most of the hazard difference is on the hazard-free UK 002 clip: 229 pothole boxes at 0.45 with the square, 132 with rect. Those are all false, so rect is the cleaner of the two there.
- Rect finds about 5 % fewer vehicle and person boxes at 0.45. The lost ones are mostly near the floor.
- Desktop time for hazard + zoom + COCO run one after the other: 48 ms square, 27 ms rect.

- Reports on the hazard-free clips (`tools/eval_events.py --rect`, 14.6 min, with and without camera simulation): 6 with the square input as before, 5 with rect. All of them are Construction.

**Busier clips and the second stage.** `urban_720p.mp4` is mostly a parking garage, so the next round used two held-out clips with more going on: UK 006 (1080p, 16:9, hazard-free) and an Argoverse work-zone clip (1548x1278, about 25 boxes a frame). Frames a second are counted from the log timestamps over 60 s. The `fps` field in the log is a moving average and reads high.

| Build | UK 006 | Work zone |
|---|---|---|
| Rect inputs, 2 threads (the table above) | 3.05 | 2.17 |
| + second stage on its own thread | 3.76 | 2.57 |
| + video decoded one frame ahead (all float) | 4.22 | 2.84 |
| **+ int8 hazard model (shipped default)** | **6.25** | **4.03** |
| + int8 COCO model too (not shipped, see below) | 7.40 | 4.40 |

In the shipped build the detectors take 148 / 205 ms a frame (UK / work zone, median) and the second stage 102 / 166 ms alongside them, so a frame's report is out about 250 / 370 ms after it arrives. On ordinary roads the N10 now keeps up with the 6 fps camera cap (`max_fps`).

- **Second stage on its own thread** (`Pipeline.kt`). Everything after the detectors runs on one "post" thread while the next frame's detectors run: the vehicle and weather classifiers, the scene rules, the police gate, the hood mask, the report manager and the UI. On the work zone that stage is about 160 ms a frame (vehicle 35, weather 31, scene rules 20, police 34). At most one frame waits for it, so reports still come out in frame order and nothing queues up. The log line has `post=` (that stage), `wait=` (time the detectors spent waiting for it) and `src=` (time waiting for the next frame).
- **Video decoded one frame ahead** (`VideoSource.kt`): the decoder fills a one-frame queue on its own thread. Test runs only. The live camera already delivers frames on its own thread.
- **Live camera at 16:9** (`CameraSource.kt`). CameraX gave 1280x960 (4:3) by default, which makes 640x480 inputs. It now asks for 16:9 first (1280x720, 640x384 inputs), the same shape as a dashcam clip.

**Police gate bug (fixed).** Since the stopped-police check (2026-11-22) the phone dropped almost every detection on any frame without a police car in view. `PoliceMotion.gate` returned the list it was given, and `Pipeline` cleared that list before adding the result back. The gate and `SceneAnalyzer` now always return a new list, and `PoliceMotionTest.gateKeepsEveryBoxWhenNoPoliceIsInView` covers it. The desktop twin was not affected. Phone numbers taken before the fix (the table above) are only valid for timing.

**int8 hazard model.** `tools/quantize_onnx.py --static` writes a QDQ model (3.3 MB against 10.8 MB): per-channel int8 weights, int8 activations, percentile calibration on 300 images, which XNNPACK runs with the A77's int8 dot-product instructions. The hazard detector alone goes from 197 to 115 ms on UK 006 on the phone. Two things had to stay in float or the reports changed:

| int8 variant | Pothole boxes >= 0.55, UK 002 / UK 006 (all false) | Work-zone boxes matched to float at 0.45 | UK 006 on the phone |
|---|---|---|---|
| Float (reference) | 37 / 39 | | no pothole report |
| Calibrated on ROADWork only, class branch int8 | 82 / 41 | 96 % | false pothole report every run |
| Work zones + road damage + ordinary driving, class branch int8 | 62 / 27 | 94 % | false pothole report |
| **Same calibration, class branch (cv3) in float (shipped)** | **33 / 31** | 94 % (cones at 0.55: 89 against 92) | no pothole report |

- Calibrating on work zones alone left the pothole branch wrongly scaled: twice the false pothole boxes on the hazard-free clips. The calibration set is now 100 ROADWork, 100 RDD2022 (US road damage) and 100 BDD100K images (the command is in `models/README.md`). Repeat runs of the same build on the phone vary by about 0.1 frames a second.
- The class scores are where int8 rounding hurt. With the head's class branch in float the pothole boxes drop below float's count, and the int8 build stops reporting the worn lane line on UK 006 (the borderline spot from 2026-11-22, which float itself scores up to 0.63). Box decoding and the attention block were already float.
- Construction on the 163 held-out Argoverse clips (half 1, `eval_construction_argoverse.py`, int8 hazard with float COCO): precision 0.966, recall 0.977 (84 of 86 clips), the same as float, with the same three false clips. int8 misses one clip float finds and finds one float misses; two blocked-lane reports move. Police reports: 10, as float. (An earlier int8 build from a hand-picked calibration set of the same mix scored 83 of 86.)
- COCO stays float. An int8 COCO model adds 17 % more frames a second, but its slightly different car box on the work-zone clip scored a parked white Ford Explorer (no light bar, no markings) at 1.0 police, where float gives 0.26: one false Police report in 38 minutes. False police reports are the expensive kind, so it is not worth it.
- The app bundles `hazard_yolo11n_dyn.qdq.onnx` when it exists (`build.gradle.kts`), else the float model. A debug build also loads `<external files>/models/hazard.onnx` or `coco.onnx` in place of the bundled asset, for trying a model without a rebuild.

**Pixel 9 Pro estimate (not measured).** The Pixel 9 Pro's Tensor G4 has 1 x Cortex-X4 at 3.1 GHz, 3 x A720 and 4 x A520, against the N10's 2 x A77. In Geekbench 6 it scores about 2.5 times the N10: about 1,950 / 4,550 single / multi-core against about 785 / 1,870 ([Pixel 9 Pro baseline](https://browser.geekbench.com/v6/cpu/baseline/18451925), [N10 5G](https://browser.geekbench.com/v6/cpu/10428574)). The X4 and A720 also have more vector units per core than the A77, which favours convolution kernels. The X4, A720 and A520 all have the int8 dot-product (and matrix-multiply) instructions, so the int8 model should gain at least as much there. So expect 2 to 3 times faster than the shipped N10 numbers:

- detectors about **50 to 75 ms a frame** on an ordinary road (70 to 100 ms in a busy work zone), second stage 35 to 50 ms alongside;
- so **13 to 20 frames a second** possible (10 to 14 in the work zone), well over the 6 fps camera cap. At the cap the Pixel is busy about a third to half of the time;
- a frame's report out **85 to 125 ms** after it arrives (125 to 185 ms in the work zone), against 250 / 370 ms on the N10.

Tensor G4 is known to throttle early under sustained load, and Waze will be running too, so the upper end is the safer planning number. The 100 ms detector target is met with room to spare, but it has to be measured on the Pixel.

### Phone speed, second round (2026-11-23)

Same phone, clips and counting as above, plus the live camera (1280x720, dry run, a desk scene).

| Build | UK 006 | Work zone | Live camera |
|---|---|---|---|
| Shipped before this round (debug build) | 6.25 | 4.03 | 5.01 |
| + camera throttle, police gate, grey image, box caps (debug) | 6.48 | 4.43 | 5.99 (the cap) |
| + zoom pass on every other frame while frames run long (debug) | 6.74 | 4.59 | |
| **Release build, with the pixel copy (shipped)** | **7.06** | **5.06** | **6.01 (the cap)** |
| Release build, zoom pass forced on every frame | | 4.76 | |

On the live camera the detectors take 123 ms and the second stage 15 ms, so a report is out about 140 ms after its frame (250 ms before), and the phone waits about 40 ms a frame for the camera.

- **Time a release build.** Debuggable apps are not compiled ahead of time, and their Kotlin loops run much slower: on UK 006 the release build takes the scene rules from 17 to 8 ms, the police gate from 20 to 8 ms and the whole second stage from 85 to 64 ms. Model times do not change (native code). `./gradlew assembleRelease -PdebugSignedRelease` signs the release build with the local debug key so it installs over the debug one. Compiling it fully ahead of time (`adb shell cmd package compile -m speed -f com.wazeomatic.detector`) changed nothing more.
- **Camera throttle** (`FrameThrottle` in `CameraSource.kt`). The `max_fps` check took a frame only once 1/6 s had passed since the last one. On a 30 fps camera the frame that lands a hair early was dropped and the next one came 200 ms later: 5 frames a second. It now keeps a schedule and takes a frame up to 20 ms before its slot.
- **Police gate** (`Flow.lk`). With a police car in view the road fit tracks 250 corners with a 21 px Lucas-Kanade window over 5 levels, forward and back. Every window sample went through the general bilinear sample (clamp, floor and weights each time), about 14 million a frame. A window wholly inside the image now works out its fraction once, with the same arithmetic (within 1e-6 of the old path, `FlowTest`). Work zone, mean per frame: 80 to 56 ms (debug), 43 ms in release.
- **Grey image.** 1920 to 640 and 1280 to 640 are whole-number ratios, where INTER_AREA is the plain block mean, now done in one pass (police gate on UK 006: 26 to 20 ms, debug). The police gate and the shoulder-car check share one grey pyramid a frame.
- **Box caps** (`--max-det` / `--max-per-class` on the desktop, `max_det` / `max_per_class` in the app). Per detector pass at most 12 boxes of a class and 40 in all, best first, and at most 1000 candidates into NMS. Normal frames stay below them: on 1,616 frames of road, work-zone and city clips the most of one class was 22 delineators and the most in one pass 38 boxes (99th percentile 11). The 163 Argoverse clips give the same reports with the caps, clip for clip. The caps are there for a frame gone wrong: the broken fp16 COCO build below gave 763 boxes a frame and took the second stage to 275 ms. The lane rule needs 3 objects and its confidence tops out at 7, so a cap of 12 cones costs it nothing. Tighter caps (10 a pass, 4 a class) do not make the phone faster (work zone 5.01 against 5.06 frames a second; the boxes a frame halve, 18 to 9, but the model runs are the cost) and lose a blocked-lane report on one Argoverse clip, so the defaults stay.
- **Zoom pass when time is short** (`zoom_every`: 1 = every frame, n = every n-th, 0 = auto, the app default; `--zoom-every` on the desktop). Auto runs the zoom pass on every frame while the frames that run it stay inside the `max_fps` budget (167 ms), and on every other frame while they don't. On the N10 that means every frame on a 16:9 road and every other frame in the busy work zone. On the held-out Argoverse half (Construction, `eval_construction_argoverse.py`):

  | Zoom pass | Precision | Recall | False clips | Police reports |
  |---|---|---|---|---|
  | Off | 0.966 | 0.977 (84 of 86) | 3 | 10 |
  | Every frame | 0.944 | 0.988 (85 of 86) | 5 | 10 |
  | Every other frame | 0.966 | 0.988 (85 of 86) | 3 | 10 |

  Half rate keeps the zoom pass's extra find without its two extra false clips, so falling back to it costs nothing here. This set has no potholes or debris, the small things the zoom pass was added for, so the default stays every frame wherever the phone has the time.
- **Pixel copy.** The input bitmap is read with `copyPixelsToBuffer` (a plain memory copy) instead of `getPixels`, which converts every pixel: hazard preprocessing 27 to 24 ms. The first frame of each input shape checks the copy against `getPixels` and falls back to it if they differ (they match exactly on the N10).

Tried and not kept:

- **NNAPI** (`ort_backend`, e.g. `nnapi`, `nnapi_fp16_cpu`). The N10 offers its Hexagon DSP, Adreno GPU and HTA through NNAPI (`qti-dsp`, `qti-gpu`, `qti-hta`). With NNAPI's own CPU device off, the drivers reject both models, the int8 hazard and the float COCO, also with fp16 and NCHW. With it allowed, the hazard model compiles but runs at 174 ms against 117 ms on XNNPACK. The setting stays for trying other phones.
- **fp16 COCO on XNNPACK** (`onnxruntime.transformers.float16`, float input and output kept). On the desktop its class scores match float to 1e-4. On the phone it gave hundreds of boxes a frame and a false Animals report, and it was slower anyway (134 against 118 ms).
- **Threads.** Two XNNPACK threads per pass is still best in release: 6.98 frames a second, against 6.73 with three and 6.66 with one.
- **int8 COCO** stays out. Its problem was the vehicle classifier flipping to police on a slightly different box, not the COCO boxes themselves. A classifier that holds steady under small box changes would let it in (17 % more frames a second).

The three model runs (about 105 to 150 ms each, side by side) are now nearly all of a frame. What would cut them further is a smaller model: quantization-aware training, so the class branch and COCO can go int8 without the problems above, or a lighter backbone (fewer channels, more depthwise convolutions). Both need a training run and a new accuracy check. YOLO11n's head already uses depthwise convolutions, and on int8 phone kernels they gain less than their FLOP count suggests.

**Falling behind** (`realtime`, video only; off by default). The camera never builds a queue: CameraX keeps only the newest frame, and the app takes a frame only when the pipeline is free and the `max_fps` slot has come, so a slow phone sees fewer frames but always recent ones. The video source used to hand over every `frame_step`-th frame in order, so a phone slower than the clip fell further behind it. With `realtime` on, it plays the clip at its own speed and drops frames the same way: it converts a frame only when the pipeline is free, or about to be (it learns how long a frame takes and starts converting just before then), and only when its `max_fps` slot has come. Test runs keep it off, so every run sees the same frames.

How far behind the road a report is, from the time a frame was captured (camera) or due (video) to its reports, the `age` field in the perf log, release build:

| Source | Frames a second | Frame age, median (90th percentile) |
|---|---|---|
| Live camera, 1280x720 | 6.00 (the cap) | 247 ms (324) |
| UK 006, real time | 6.01 (the cap) | 225 ms |
| Work zone, real time | 4.99 | 332 ms (416) |

Age is more than the detector and second-stage time (about 140 ms on the camera) because a frame waits for the one before it to finish, and on the camera also waits in the camera pipeline. Simpler versions were worse: converting only once the pipeline was idle left it waiting 80 ms a frame (4.36 frames a second), and always replacing the waiting frame with a newer one kept the frame rate (5.25) but put the age at 450 ms.

With these numbers the Pixel 9 Pro estimate becomes detectors about 45 to 70 ms on the live camera, and a report about 50 to 75 ms after its frame (still to be measured).

### Model size

| File | Size |
|------|------|
| coco_yolo11n_{320,416,640}.onnx (fp32) | 10.6 / 10.6 / 10.7 MB |
| coco_yolo11n_{320,416,640}.int8.onnx | 2.9 / 2.9 / 3.0 MB |
| hazard_yolo11n_{320,416,640}.onnx (fp32) | 10.5 / 10.6 / 10.7 MB |
| hazard_yolo11n_{320,416,640}.int8.onnx | 2.9 / 2.9 / 3.0 MB |

### Accuracy (validation split, 416 px)

YOLO11n fine-tuned at 416 px, 43 epochs (early stop, best epoch kept), batch 32, RTX 5070 Ti, 18 min. Validation split = 2,259 images (StarLine session-based test split plus a random 15 % of the other sources).

| Class | Val images | Boxes | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|-------|-----------|--------|-------|----------|
| pothole | 226 | 665 | 0.60 | 0.52 | **0.56** | 0.27 |
| cone | 314 | 2,713 | 0.40 | 0.17 | 0.16 | 0.08 |
| construction_sign | 1,193 | 1,252 | 0.81 | 0.66 | **0.69** | 0.34 |
| delineator | 302 | 2,127 | 0.52 | 0.19 | 0.21 | 0.09 |
| all | 2,259 | 6,757 | 0.58 | 0.38 | 0.40 | 0.19 |

Int8 copy: on 60 frames of the urban clip the int8 model produced the identical label set to fp32 on every frame (`bench_onnx.py --compare`), so quantisation does not change behaviour at our thresholds.

Reading the numbers: pothole already meets the sprint 2 target (mAP50 >= 0.5). Cone and delineator look poor because the StarLine validation images are 2880 px wide and most cones are distant, a few pixels tall after resizing to 416; the model finds the near ones (which are the ones that matter for a report) and misses the far ones. Training at 640 px or cropping the StarLine images into tiles is the obvious next step. Plots: [training curves](images/hazard_training_results.png), [confusion matrix](images/hazard_confusion_matrix.png), [validation predictions](images/hazard_val_predictions.jpg).

### Video replay

Both models, every 3rd frame (10 inferences/s), default Report Manager thresholds. Neither clip contains a visible pothole or work zone, so the correct number of events is zero.

| Clip | Events (no hood mask) | Events (hood mask, default) | Correct | False | Missed |
|------|----------------|----------------|---------|-------|--------|
| Dashcam Recording (urban), 120 s | 1 (pothole, conf 0.50) | 0 | 0 | 0 | 0 |
| Panorama II BlackBox daylight, 120 s | 1 (pothole, conf 0.49) | 0 | 0 | 0 | 0 |

The single event in each clip was the same false positive: the reflection on the car's own bonnet at the very bottom of the frame, persisting for 3 frames just above the 0.45 threshold. The Report Manager now ignores boxes whose bottom edge lies in the lowest 10 % of the frame (`hood_frac`), which removes it without touching anything on the road. With `--camera-sim` degradation the urban clip also produced no events.

### Small-object pass (tiled data, 640 px, two-pass zoom)

Why the first model missed cones: in the StarLine source the cones are a median 43 px wide at 2880 px (delineators 61 px). The first dataset build downscaled every frame to 1280 px and training ran at 416 px, so a median cone was about 6 px to the network. Measured with `tools/eval_events.py`, the 416 model had AP50 0.08 on boxes under 24 px against 0.70 on boxes over 64 px, and 7,220 of the 9,248 validation boxes were that small. The merged set also had only 145 near-range cone boxes and 106 RDD potholes.

What changed:

1. `prepare_datasets.py --tile 1024` cuts the StarLine frames into 1024 px tiles at native resolution (stride 896, boxes kept when 60 % visible, 8 % of empty tiles kept as negatives): cone boxes 8k -> 22.6k, delineator 13k -> 24k. Dataset `<datasets>/wazeomatic_hazards_640`, validation = StarLine test-session tiles plus the other sources' validation images (5,167 images).
2. Training at 640 px (`train.py --imgsz 640 --batch 16`; about 1.2 min per epoch on the RTX 5070 Ti once the disk cache is warm). Run `hazard640` early-stopped at epoch 27; run `hazard640b` fine-tuned it for 25 epochs at lr0 0.002 with mosaic off for the last 8. `hazard640b` was the model shipped as `models/hazard_yolo11n.onnx` at that point (kept as `models/hazard_yolo11n_640b.onnx`).
3. The pothole images of the complete RDD2022 release (all six countries, 7,440 images, 10,702 pothole boxes, CC BY-SA 4.0, mirror dronefreak/RDD2022, `tools/fetch_rdd2022_potholes.py`) were added for a third run (`hazard640c`); see the note at the end of this section.
4. Two-pass zoom inference (`ZoomDetector`, `run_video.py --zoom-crop`, `zoom_crop` launch extra of the Android app): each model runs on the full frame and again on the middle 50 % x 35-75 % of the frame (the road ahead, above the hood zone); boxes are mapped back and merged with per-class NMS. On a 1280x720 frame with a 640 px model the crop pass sees the road at full resolution, so a 20 px cone becomes 40 px to the network, for one extra inference per model and frame.

Box metrics on the tiled validation split (IoU 0.5, detector conf 0.25; `runs/eval/baseline416_tiledval`, `stage2_640_tiledval`):

| Class | AP50, 416 model | AP50, 640b model |
|-------|-----------------|------------------|
| pothole | 0.44 | 0.49 |
| cone | 0.18 | 0.31 |
| construction_sign | 0.50 | 0.47 |
| delineator | 0.09 | 0.21 |
| all, boxes < 24 px | 0.10 | 0.13 (recall 0.30 vs 0.17) |

Report events on synthetic 12-frame clips from the same images (Report Manager defaults at the time: 3 hits in 10 frames, min confidence 0.45):

| Hazard type | 416 model P / R / F1 | 640b model P / R / F1 |
|-------------|----------------------|------------------------|
| HAZARD_ON_ROAD_CONSTRUCTION (cone, sign, delineator) | 0.99 / 0.75 / 0.85 | 0.99 / 0.83 / 0.90 |
| HAZARD_ON_ROAD_POTHOLE | 0.91 / 0.54 / 0.68 | 1.00 / 0.62 / 0.77 |

Hazard-free clips (both models + COCO, every 3rd frame, clean and camera-degraded, 4 x 120 s): zero events for the 416 model and zero for the 640b model. The threshold sweep says min confidence 0.35 buys another 4 to 6 points of event recall for one extra false event per 2,000 clips; the clip false-positive rate at 0.35 has not been measured yet, so 0.45 stays the default.

Cost: hazard model alone on the desktop 5.4 ms at 416 vs 11.3 ms at 640 (fp32, 4 threads). On the Pixel 9 Pro emulator in live camera mode (x86 host under load) the full stack was hazard 23 ms + zoom 24 ms + COCO 18 ms + zoom 13 ms at 4.4 FPS; the real phone number is still pending.

Run `hazard640c` (fine-tune of 640b on the set with the full RDD2022 potholes, 30 epochs, lr0 0.005, mosaic off for the last 8) was shipped as `models/hazard_yolo11n.onnx` on the morning of 2026-11-06, until the v3 model replaced it that afternoon. Scored on the enlarged validation split (7,360 images, now including 2,193 RDD2022 pothole images; `runs/eval/stage2_640b_val2` vs `stage3_640c_val2`, same split for both):

| | 640b | 640c |
|---|---|---|
| AP50 pothole / cone / construction_sign / delineator | 0.10 / 0.31 / 0.47 / 0.21 | 0.58 / 0.31 / 0.46 / 0.28 |
| AP50 all classes (boxes < 24 px) | 0.24 (0.13) | 0.35 (0.15) |
| Event P / R / F1, HAZARD_ON_ROAD_POTHOLE | 1.00 / 0.10 / 0.19 | 1.00 / 0.46 / 0.63 |
| Event P / R / F1, HAZARD_ON_ROAD_CONSTRUCTION | 0.98 / 0.83 / 0.90 | 0.99 / 0.80 / 0.88 |
| False events on the 4 hazard-free clip runs, min conf 0.45 | 0 | 2 (one pothole each in two runs) |
| Same with a pothole-only floor of 0.55 (`--report-class-conf pothole=0.55`) | 0 | 0 |

The 640b pothole number on this split is low because 2,193 of its 2,418 pothole images are the new RDD2022 ones, which 640b never saw; on the IVCNZ potholes 640c reaches event recall 0.69. The two extra false events are what the RDD training buys at 0.45: the shipped operating point is therefore min confidence 0.45 for every class and 0.55 for potholes (`min_confidence_by_label` in `ReportConfig`, `report_class_conf` in the app). With that floor the synthetic-clip pothole event recall is 0.37 (precision 1.00, F1 0.54; `runs/eval/stage3_640c_val2_p55`), still 3.5x the 640b model, and the clip false-event count is zero. Desktop cost is unchanged at about 11 ms per 640 px inference (4 threads).

### Ensembling, zoom and tracking (pipeline pass, 2026-11-06)

The pipeline can now combine several detectors and reason about where a hazard sits across frames (`wazeomatic/fusion.py`, `EnsembleDetector` and `ClassFilter` in `detector.py`, spatial tracking in `report_manager.py`, `Prefetch` in `sources.py`; the Android app has the same in `BoxFusion.kt`, `ReportManager.kt` and `Pipeline.kt`):

- **Weighted Boxes Fusion** across hazard models. Boxes of the same class that overlap (IoU >= 0.55) are averaged, weighted by confidence and by a per-model weight. With `conf_type=avg` the fused confidence is the weighted mean over *all* members, so a box only one of two models saw drops to half its confidence; `max` keeps the best member's confidence; `box_avg` averages only the boxes present. `fusion=nms` keeps the best box unchanged.
- **Parallel members.** Members of an ensemble, and the hazard stack and the COCO model, run on their own threads (ONNX Runtime releases the GIL); the stats line now reports both the summed model time (compute) and the wall-clock latency of a frame. Frames are decoded and degraded on a background thread (`--prefetch`).
- **Spatial tracking** (`--track-iou 0.3`): the hits that justify a report must come from boxes that overlap frame to frame (boxes grown by 1.5x, or centres within 1.5 box sizes), i.e. from one object the car approaches. Three flickers in three corners of the frame no longer add up to a report.
- **Per-label floors** on the detector (`--class-conf`) and in the Report Manager (`--report-class-conf`), so a noisy class can demand more confidence than a clean one.

All variants are scored with `tools/eval_events.py` (same flags as `run_video.py`, built by `wazeomatic/build.py`) on the same subset of the tiled 640 px validation split: 633 images (120 per source plus all 6 cone_voc images, seed 1, 8 jittered frames per image), Report Manager defaults unless stated. The subset includes the RDD2022 potholes added on 2026-11-06, which is why the absolute pothole numbers are lower than in the section above. `runs/eval/ens_*`; table by `tools/summarize_evals.py`.

| Stack | AP50 pothole / cone / sign / delineator | Event P / R / F1 | Construction R (FP) | Pothole R (FP) | Model ms per frame |
|---|---|---|---|---|---|
| 640b alone (reference) | 0.33 / 0.30 / 0.50 / 0.15 | 0.943 / 0.508 / 0.660 | 0.82 (12) | 0.36 (0) | 15 |
| 640b + zoom pass | 0.29 / 0.30 / 0.48 / 0.12 | 0.944 / **0.564** / **0.706** | 0.86 (13) | 0.42 (0) | 30 |
| 640b + 416, WBF avg | **0.36** / **0.31** / **0.73** / 0.12 | 0.952 / 0.408 / 0.571 | 0.64 (8) | 0.30 (0) | 22 |
| 640b + 416, WBF max | 0.35 / 0.27 / 0.72 / 0.15 | 0.933 / 0.536 / 0.681 | 0.86 (15) | 0.38 (0) | 37 |
| 640b + 416, WBF avg + zoom | 0.32 / 0.24 / 0.69 / 0.11 | 0.948 / 0.464 / 0.623 | 0.73 (10) | 0.34 (0) | 55 |
| 640b + tracking 0.3 | as 640b | 0.943 / 0.508 / 0.660 | 0.82 (12) | 0.36 (0) | 15 |

Reading the numbers:

- The zoom pass is the cheapest real gain: +5.6 points of event recall for one extra inference, no extra false events.
- Fusing the 416 model is a calibration question, not a detection one. At box level the fused scores are far better ranked (construction-sign AP50 0.50 -> 0.73) because two models agreeing is strong evidence, but `avg` halves every box the weaker 416 model misses, and at the 0.45 report floor that costs 10 points of recall. `max` keeps the recall and gains 2 points of F1 over the single model. Per the sweep on the single model, the floor itself is worth revisiting: 0.35 / 2 hits gives F1 0.697 on synthetic clips, at the price of 2 more false events per 633 clips; whether that holds on real footage is what the tracking runs below measure.
- Tracking changes nothing on the synthetic clips, by construction: every jittered frame shows the same object in nearly the same place, so all hits are already consistent. Its job is the real-clip false-positive rate, where the hits behind a false report come from different places in the frame.

#### Second and third batch: the 640c model, floors, zoom plus tracking, and the real clips

Same subset and protocol as the table above (`runs/eval/ens_f*`, `ens_g*`); the pothole floor of 0.55 that shipped with 640c is applied in every `g` row. 640c is the model promoted to `models/hazard_yolo11n.onnx` on the morning of 2026-11-06 (RDD2022 potholes added to training).

| Stack | Report floor | Event P / R / F1 | Construction R (FP) | Pothole R (FP) | Model ms |
|---|---|---|---|---|---|
| 640b, pothole floor 0.55 | 0.45 | 0.937 / 0.454 / 0.612 | 0.82 (12) | 0.28 (0) | 13 |
| 640b + zoom + tracking 0.3 | 0.35 | 0.949 / 0.620 / 0.750 | 0.87 (13) | 0.50 (0) | 54 |
| 640b + 416 (box_avg) + zoom + tracking | 0.35 | 0.906 / 0.691 / 0.784 | 0.90 (22) | 0.59 (6) | 110 |
| **640c** | 0.45 | 0.957 / 0.513 / 0.668 | 0.79 (9) | 0.38 (0) | 15 |
| **640c + zoom** | 0.45 | 0.959 / 0.592 / 0.732 | 0.85 (10) | 0.47 (0) | 29 |
| 640c + 640b (WBF max) | 0.45 | 0.949 / 0.566 / 0.709 | 0.86 (12) | 0.43 (0) | 37 |
| **640c + 640b (WBF max) + zoom** | 0.45 | 0.950 / 0.635 / 0.761 | 0.90 (13) | 0.51 (0) | 60 |
| 640c + zoom + tracking 0.3 | 0.35 | 0.956 / 0.602 / 0.739 | 0.88 (11) | 0.47 (0) | 27 |

Real footage (`assets/videos`, every 3rd frame, clean and camera-degraded, 640c unless stated, pothole floor 0.55; `runs/eval/vid_*`):

| Stack | Report floor | Hood zone | False events in 4 x 120 s |
|---|---|---|---|
| 640c | 0.45 | fixed 10 % | 0 |
| 640c + zoom | 0.45 | fixed 10 % | 2 (pothole) |
| 640c + 640b + zoom | 0.45 | fixed 10 % | 4 (2 pothole, 2 construction) |
| 640c + zoom, any stack | 0.35 | fixed 10 % | 2 to 5 |
| 640c | 0.45 | learnt (22 %) | 0 |
| **640c + zoom** | 0.45 | learnt (22 %) | **0** |
| 640c + 640b + zoom | 0.45 | learnt (22 %) | 0 false; 2 real cones (see below) |
| 640c + zoom | 0.35 | learnt (22 %) | 1 pothole + the cones |

What the clips taught us:

- The "pothole" that the zoom pass kept producing was the car's own dashboard air vent at 82 % of the frame height, just above the fixed 10 % hood zone (`runs/demo/zoom_fp_event.jpg`). The zoom pass did not detect it; it added enough extra hits on the road for the vent's full-frame hits to reach three in ten frames. The dashboard of that car reaches 22 % up the frame. The hood zone is therefore now learnt from the picture (`wazeomatic/hood.py`, `HoodEstimator.kt`, on by default): a running average of the frames keeps the dashboard's edges sharp while the road blurs away; the ratio of edge energy in the average to the instantaneous edge energy is 0.87 / 0.92 / 0.73 / 0.56 for the bottom four 5.6 % bands and 0.31 above them, so the zone settles at 22 % within about 80 s of driving and never drops below the configured minimum. With it, every stack is clean at the 0.45 floor.
- The one event the fused stack still emits is a real traffic cone standing in the parking garage where the recording begins (`runs/demo/fusion_fp_urban_5s_crop.jpg`, an 11 x 29 px box the single model misses); the clip is not entirely hazard-free. Reporting it as "Construction" would be wrong in context, which is a reminder that the report manager has no notion of place yet.
- Report floor 0.35 is not safe on real footage even with the learnt hood zone (one false pothole at 99 s plus more cone hits); the floor stays at 0.45.
- The two Commons clips are the same recording at two resolutions and encodings (identical dashboard, identical cone at 5.5 s), so the "4 x 120 s" above is really one two-minute drive, clean and degraded, twice. More and different footage is the most valuable thing the team can add before the real-phone drive.

**Shipped operating point (2026-11-06, morning; superseded below):** 640c (`models/hazard_yolo11n.onnx`), zoom pass on, learnt hood zone on, detector threshold 0.35, report floor 0.45 with pothole 0.55, 3 hits in 10 frames, 60 s per-type and 5 s global cooldown, tracking off. On the desktop that is 29 ms of model time per frame. The phone app additionally fuses 640b (`hazard2.onnx`, WBF max) because the two run in parallel threads and the fused stack has the higher recall at the same false-event count; disable it with the `ensemble` setting on a phone where latency matters more than the extra recall. Spatial tracking stays available (`--track-iou`) but is off: it changed nothing on these clips because every false event came from one persistent object, which is exactly what tracking cannot reject.

### Data expansion and the v3 models (2026-11-06, afternoon)

The pipeline pass left two problems that were data problems: potholes hallucinated on real roads, and the whole
"construction" family trained on one source (StarLine, one country, four cameras). The afternoon went into
finding more real road footage with labels and retraining. Datasets 5 to 7 above (ROADWork, Argoverse 2, a
small Roboflow pothole set) and the UK dashcam and Argoverse clips are the result; the merged set
`wazeomatic_hazards_v3` has 45,855 training and 14,597 validation files (StarLine tiles, ROADWork and
Argoverse as both 1024/1080/1280 px tiles and a downscaled full frame, RDD2022 and the pothole sets as
before) and two new classes, **barricade** (ROADWork Barricade + Barrier) and **worker**, both mapped to
Hazard > Construction.

Training: YOLO11n from the 640c weights (40 epochs, lr0 0.005, batch 32, cosine, mosaic off for the last 8)
and YOLO11s from COCO weights (same schedule, batch 24), 640 px, on the NVMe copy of the set (from the 14 TB
HDD the dataloader starved the GPU to 2 % and an epoch took 45 minutes; from the SSD 2.5 minutes). Per-epoch
validation uses a fixed 3,000-image subset of the val split (`val_subset.txt`); the tables below use the
event harness on a 1,201-image seed-1 subset of the full val split (`--limit 120 --frames 8 --seed 1`,
zoom pass on), so the shipped model and the new ones are measured on identical images.

Box AP50 on the v3 val subset (`runs/eval/v3val_*`):

| class | 640c (shipped this morning) | v3 nano | v3 small (YOLO11s) |
|---|---|---|---|
| pothole | 0.323 | 0.394 | 0.431 |
| cone | 0.279 | 0.513 | 0.523 |
| construction_sign | 0.250 | 0.577 | 0.579 |
| delineator | 0.039 | 0.334 | 0.358 |
| barricade | - | 0.296 | 0.368 |
| worker | - | 0.360 | 0.520 |
| all, small (< 24 px on the 640 input) | 0.035 | 0.246 | 0.267 |
| all, medium | 0.194 | 0.497 | 0.520 |
| all, large | 0.374 | 0.542 | 0.569 |
| **all** | **0.159** | **0.392** | **0.415** |
| model time per frame, desktop CPU, zoom on | 29 ms | 25 ms | 86 ms |

Event-level (synthetic clips, 3 hits in 10, zoom on; precision / recall / F1):

| Stack | Report floor | Overall | Construction | Pothole |
|---|---|---|---|---|
| 640c | 0.45, pothole 0.55 | 0.960 / 0.508 / 0.664 | 0.952 / 0.546 / 0.694 | 0.968 / 0.471 / 0.634 |
| v3 nano | 0.45, pothole 0.55 | 0.950 / 0.633 / 0.760 | 0.935 / 0.820 / 0.873 | 0.978 / 0.456 / 0.622 |
| **v3 nano (shipped floors)** | 0.40, pothole 0.45 | **0.947 / 0.717 / 0.816** | 0.925 / 0.847 / 0.884 | 0.979 / 0.593 / 0.739 |
| v3 nano + 640c (WBF max) | 0.40, pothole 0.45 | 0.922 / 0.742 / 0.822 | 0.915 / 0.858 / 0.886 | 0.931 / 0.632 / 0.753 |
| v3 small | 0.40, pothole 0.45 | 0.952 / 0.769 / 0.851 | 0.926 / 0.893 / 0.910 | 0.988 / 0.650 / 0.784 |
| v3 nano, threshold sweep | 0.45 / 3 | 0.953 / 0.704 / 0.809 | | |
| v3 nano, threshold sweep | 0.35 / 3 | 0.933 / 0.757 / 0.836 | | |
| v3 nano, threshold sweep | 0.35 / 2 | 0.924 / 0.779 / 0.846 | | |

(The sweep rows apply one floor to every label, which is why their overall numbers differ from the first
v3 row.) Fusing the old 640c into the v3 nano buys 0.6 F1 overall and costs 2.7x the model time (68 ms
against 25 ms per frame on the desktop CPU) and 13 extra false pothole events, so the old models are no
longer ensemble partners; the phone build fuses the v3 nano with the v3 small instead (below) or runs the
nano alone. Construction recall went from 0.55 to 0.82 at the same precision; the delineator class, which
640c had effectively never seen in ROADWork's drum / tubular-marker / vertical-panel form, is now found a
third of the time even when small. Potholes did not move: the new pothole data is small and the pothole
subset of the val split (IVCNZ close-ups and RDD street views) is the same as before.

Real footage, v3 nano, learnt hood zone, every 6th frame, clean and camera-degraded (`runs/eval/clips_*`):

| Report floor | 5 clips, 12 min, 10 runs | Raw pothole detections above 0.25 on the two UK clips |
|---|---|---|
| 0.45, pothole 0.55 (640c, for reference) | 0 false events | 497 / 565 |
| 0.45, pothole 0.55 | 0 | 275 / 185 |
| 0.45, pothole 0.45 | 0 | |
| 0.35, pothole 0.45 | 0 | |
| 0.40, pothole 0.45, v3 small | 0 | |

The UK clips (`uk_dashcam_002/006`, overcast suburban and rural England, dashboard in the bottom 15 % of the
frame, wiper in view) are the first footage from a different car and country; the learnt hood zone settles
at about 25 % on them and nothing is reported. On the Argoverse work-zone clip
(`assets/videos/labelled/argo_2ff4f798_workzone.mp4`, a Miami lane closure with a "Right lane closed ahead"
sign, a barrel line and cones) the live pipeline emits exactly one Construction event and draws the sign, the
barrels and the distant markers (`runs/clips/argo_workzone_v3n.mp4`); the 60 s per-type cooldown absorbs the
rest of the zone.

The small model is better everywhere (overall event F1 0.851 against 0.816, pothole recall 0.65 against
0.59 with fewer false events) at 3.4x the model time. It is exported as `models/hazard_yolo11s_v3.onnx`;
the desktop pipeline takes it with `--hazard-model models/hazard_yolo11s_v3.onnx` and the phone build with
`-PhazardModel2=hazard_yolo11s_v3.onnx` (fused with the nano) or `-PhazardModel=hazard_yolo11s_v3.onnx`
(alone). It is not the default because its Pixel 9 Pro latency is unmeasured; the nano with the zoom pass
already costs two inferences per frame there.

**Operating point after this pass (2026-11-06):** v3 nano (`models/hazard_yolo11n.onnx` = `hazard_yolo11n_v3.onnx`,
six classes), zoom on, learnt hood zone, detector threshold 0.35, report floor **0.40** with pothole
**0.45**, 3 hits in 10 frames, cooldowns 60 / 5 s. 0.35 was clean on the clips too, but twelve minutes of
footage is a thin basis for the lowest setting; 0.40 keeps a margin and still lifts synthetic F1 above the
0.45 point. The pothole floor comes down from 0.55 because the v3 model emits half as many weak pothole
boxes on real roads and none of them survive three hits at 0.45.

### Hazard v4: debris (2026-11-14)

The v4 nano is the v3 nano trained on for 25 more epochs (learning rate 0.002) on the v3 set plus Lost and
Found road obstacles as a seventh class, `debris` (models/README.md has the data file). It is the first model
with the class behind Hazard > Object. Measured against v3 with the shipped report settings:

| | v3 | v4 |
|---|---|---|
| boxes on the v3 val subset (1,201 images): AP50, all six shared classes | 0.392 | 0.381 |
| events on the same images as 8-frame clips (`tools/eval_events.py --zoom-crop --limit 120 --frames 8 --seed 1`): F1 | 0.817 | 0.802 |
| Construction events on those clips: found / false | 310 / 24 | 312 / 31 |
| Pothole events on those clips: found / false | 229 / 5 | 215 / 3 |
| reports on the hazard-free clips (14.6 min, each clip with and without the camera simulation) | 4 | 6 |
| Construction on the 163 held-out Argoverse clips (3D labels): recall / precision | 0.94 / 0.96 | 0.99 / 0.97 |
| Object on Lost and Found val (120 frames with an obstacle, 120 without; debris floor 0.5): recall / precision | no class | 0.55 / 0.89 |

On the synthetic clips v4 is a little worse than v3 on the classes they share; on real city driving it finds
more work zones with no more false reports, and it adds Object. It ships. On Lost and Found the debris floor
trades precision for recall: 0.4 gives 0.59 / 0.84, 0.5 gives 0.55 / 0.89, 0.6 gives 0.50 / 0.92. Before the
road rules the held-out Argoverse clips got two Object reports, both false: a vent on a building front (0.80)
and the trouser legs of a man crossing (0.59). A debris box now has to reach below `road_top` (the horizon of
the camera geometry less 0.15; 0.35 by default) and must not lie mostly inside a person or vehicle box;
replayed, that cost 2 of 73 Object events on Lost and Found.

### Current defaults (2026-11-14)

The same on the desktop (`ReportConfig`, `run_video.py`) and in the phone app (`Settings.kt`, `ReportManager.kt`) unless stated:

| Setting | Value |
|---|---|
| Hazard model | v4 nano, `models/hazard_yolo11n.onnx` (a copy of `hazard_yolo11n_v4.onnx`), seven classes, 640 px, fp32 |
| COCO model | stock YOLO11n, `models/coco_yolo11n.onnx`, 416 px |
| Zoom pass | on, crop `0.25,0.35,0.75,0.75`, hazard model only (`--zoom-coco` / `zoom_coco` adds the COCO model) |
| Fusion | off: one hazard model ships. A second one (`--ensemble-model`, `-PhazardModel2=hazard_yolo11s_v3.onnx`) is fused with Weighted Boxes Fusion |
| Hood zone | learnt from the picture, minimum 10 % of the frame height |
| Detector box threshold | 0.35 on the desktop and in the app |
| Report floor | 0.40; pothole 0.55 (0.45 until 2026-11-22), police_vehicle 0.85, worker 0.6, debris 0.5 (the higher of the general floor and a label floor applies) |
| Persistence | 3 hits in 10 inference frames (Police 2) |
| Cooldowns | 60 s per hazard type and 60 s between any two events |
| Spatial tracking | off (`track_iou` 0) |

Two things changed after the 2026-11-06 pass. The global cooldown went from 5 s to 60 s so that it matches the macro's one report per minute; a hazard that qualifies during another report's cooldown is counted as held back instead of being sent and dropped by the macro ([pipeline-to-macro.md](pipeline-to-macro.md#cooldown-responsibilities)). And the pothole floor is now the default of the command-line tools as well as of the app (`--report-class-conf` defaulted to `pothole=0.45`; 0.55 since 2026-11-22), so `run_video.py` and `tools/eval_events.py` run the shipped operating point without extra flags. The tables above were measured before these changes; the event tables use single-hazard synthetic clips and name their floors explicitly.
