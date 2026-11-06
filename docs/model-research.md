# Model research

Goal: a detector that runs on a Pixel 9 Pro alongside Waze, uses well under 1 GB of memory, and reaches at least a few inferences per second on road hazards. The same model should also run on a desktop for simulation and on a Raspberry Pi later.

## Decision

- **Model:** Ultralytics YOLO11n, fine-tuned on hazard classes, plus a stock COCO YOLO11n for vehicles and animals.
- **Runtime:** ONNX Runtime everywhere. On Android, the XNNPACK execution provider on the CPU.
- **Input size:** 416x416 on the phone, 320x320 as the fast option. Static shape export, opset 17, no NMS in the graph (NMS is done in our code).
- **Precision:** fp32 plus a dynamically quantised int8 copy of each model.

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

All datasets are stored on the lead's machine under `F:\dev\datasets` and are not committed.

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

Gaps: **police** has no adequate open dataset (the only Hugging Face set found had 8 images), so it is deferred. **Crash** is deferred for the same reason. Pothole positives are mostly from dataset 1 (close-up, varied cameras); dataset 2 adds US street-level context but only 86 pothole boxes, so its value is mainly as negatives. Barrels (drums) have no labelled source yet; they will usually be picked up as `cone` or `delineator`, which map to the same Waze report.

## Class schema and Waze mapping

Defined in `wazeomatic/schema.py`; model classes in `configs/classes.yaml`.

| Model class | Model | hazard_type | Waze path (schema.py) | Sprint 1 |
|-------------|-------|-------------|------------------------|----------|
| pothole | hazard | `HAZARD_ON_ROAD_POTHOLE` | Hazard > Pothole | yes |
| cone, construction_sign, delineator | hazard | `HAZARD_ON_ROAD_CONSTRUCTION` | Hazard > Construction | yes |
| dog, horse, sheep, cow, bear | COCO | `HAZARD_SHOULDER_ANIMALS` | Hazard (no subtype; Waze removed Animals) | yes |
| car, truck, bus + heuristic | COCO | `HAZARD_SHOULDER_CAR_STOPPED` | Hazard > Car on shoulder | experimental, off |
| (none yet) | | `HAZARD_ON_ROAD_OBJECT` | Hazard > Object | no |
| police_vehicle | hazard | `POLICE` | Police | deferred |
| (none) | | `CRASH` | Crash | deferred |

**Menu names:** since the 2024 redesign Waze lists the hazard subtypes Construction, Car on shoulder, Broken traffic light, Pothole and Object directly under Report > Hazard, with Hazard preselected ([Waze Help](https://support.google.com/waze/answer/13739290?hl=en)); Animals/Roadkill were removed, so animals go out as a plain Hazard. The `waze_subcategory` strings in `schema.py` follow this flat layout. The `hazard_type` keys are the contract and never change; the macro and the mock app tap by on-screen label.

## Training and evaluation plan

1. `python tools/prepare_datasets.py`: convert VOC/parquet to YOLO txt using the class order in `configs/classes.yaml`, keep up to 1,500 crack-only RDD images with empty labels as hard negatives, downscale images to 1280 px, random 85/15 train/val split (StarLine keeps its own session-based split), write `F:\dev\datasets\wazeomatic_hazards\data.yaml`. `--check <source>` renders labelled samples to `runs/dataset_check/` to verify box conventions.
2. `python tools/train.py`: fine-tune `yolo11n.pt` at 416 px (no vertical flips, mild rotation/shear, HSV jitter for exposure, mosaic off for the last 10 epochs, cosine LR).
3. `python tools/export_onnx.py --weights runs/train/hazard/weights/best.pt --imgsz 416` (and `--imgsz 320`); `--coco` for the stock model.
4. `python tools/quantize_onnx.py <model>`: ONNX Runtime dynamic int8 quantisation, written as `*.int8.onnx`.
5. `python tools/bench_onnx.py <models...> --threads 4`: mean/p50/p90 ms per frame as a markdown table; `--compare <clip>` measures fp32 vs int8 agreement on real frames.
6. Evaluate mAP50 per class on the held-out split, then run both test videos through `wazeomatic.run_video` and count events against a hand-labelled list of hazards in each clip.

Acceptance for sprint 2: pothole and cone mAP50 >= 0.5, no more than one false event per test clip, phone latency under 100 ms per frame.

## Measured results

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

1. `prepare_datasets.py --tile 1024` cuts the StarLine frames into 1024 px tiles at native resolution (stride 896, boxes kept when 60 % visible, 8 % of empty tiles kept as negatives): cone boxes 8k -> 22.6k, delineator 13k -> 24k. Dataset `F:/dev/datasets/wazeomatic_hazards_640`, validation = StarLine test-session tiles plus the other sources' validation images (5,167 images).
2. Training at 640 px (`train.py --imgsz 640 --batch 16`; about 1.2 min per epoch on the RTX 5070 Ti once the disk cache is warm). Run `hazard640` early-stopped at epoch 27; run `hazard640b` fine-tuned it for 25 epochs at lr0 0.002 with mosaic off for the last 8. `hazard640b` is the model shipped as `models/hazard_yolo11n.onnx`.
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

Report events on synthetic 12-frame clips from the same images (Report Manager defaults: 3 hits in 10 frames, min confidence 0.45):

| Hazard type | 416 model P / R / F1 | 640b model P / R / F1 |
|-------------|----------------------|------------------------|
| HAZARD_ON_ROAD_CONSTRUCTION (cone, sign, delineator) | 0.99 / 0.75 / 0.85 | 0.99 / 0.83 / 0.90 |
| HAZARD_ON_ROAD_POTHOLE | 0.91 / 0.54 / 0.68 | 1.00 / 0.62 / 0.77 |

Hazard-free clips (both models + COCO, every 3rd frame, clean and camera-degraded, 4 x 120 s): zero events for the 416 model and zero for the 640b model. The threshold sweep says min confidence 0.35 buys another 4 to 6 points of event recall for one extra false event per 2,000 clips; the clip false-positive rate at 0.35 has not been measured yet, so 0.45 stays the default.

Cost: hazard model alone on the desktop 5.4 ms at 416 vs 11.3 ms at 640 (fp32, 4 threads). On the Pixel 9 Pro emulator in live camera mode (x86 host under load) the full stack was hazard 23 ms + zoom 24 ms + COCO 18 ms + zoom 13 ms at 4.4 FPS; the real phone number is still pending.

Run `hazard640c` (fine-tune of 640b on the set with the full RDD2022 potholes, 30 epochs, lr0 0.005, mosaic off for the last 8) is the model shipped as `models/hazard_yolo11n.onnx` since 2026-11-06. Scored on the enlarged validation split (7,360 images, now including 2,193 RDD2022 pothole images; `runs/eval/stage2_640b_val2` vs `stage3_640c_val2`, same split for both):

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

Same subset and protocol as the table above (`runs/eval/ens_f*`, `ens_g*`); the pothole floor of 0.55 that now ships is applied in every `g` row. 640c is the model promoted to `models/hazard_yolo11n.onnx` on 2026-11-06 (RDD2022 potholes added to training).

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

**Shipped operating point (2026-11-06):** 640c (`models/hazard_yolo11n.onnx`), zoom pass on, learnt hood zone on, detector threshold 0.35, report floor 0.45 with pothole 0.55, 3 hits in 10 frames, 60 s per-type and 5 s global cooldown, tracking off. On the desktop that is 29 ms of model time per frame. The phone app additionally fuses 640b (`hazard2.onnx`, WBF max) because the two run in parallel threads and the fused stack has the higher recall at the same false-event count; disable it with the `ensemble` setting on a phone where latency matters more than the extra recall. Spatial tracking stays available (`--track-iou`) but is off: it changed nothing on these clips because every false event came from one persistent object, which is exactly what tracking cannot reject.

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

**Operating point after this pass:** v3 nano (`models/hazard_yolo11n.onnx` = `hazard_yolo11n_v3.onnx`,
six classes), zoom on, learnt hood zone, detector threshold 0.35, report floor **0.40** with pothole
**0.45**, 3 hits in 10 frames, cooldowns 60 / 5 s. 0.35 was clean on the clips too, but twelve minutes of
footage is a thin basis for the lowest setting; 0.40 keeps a margin and still lifts synthetic F1 above the
0.45 point. The pothole floor comes down from 0.55 because the v3 model emits half as many weak pothole
boxes on real roads and none of them survive three hits at 0.45.
