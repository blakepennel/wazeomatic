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
| Desktop, hazard model (4 classes), mean | 3.3 | 9.3 | 5.3 | 12.8 | | |
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
| hazard_yolo11n_{320,416}.onnx (fp32) | 10.5 / 10.6 MB |
| hazard_yolo11n_{320,416}.int8.onnx | 2.9 / 2.9 MB |

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
