# Third-party terms

The [MIT licence](LICENSE) covers the source code written for this project. It does not cover the
things the project is built on or trained with. Those keep their own terms, listed here.

## Nothing trained is distributed

No model weights and no dataset are in this repository or in any release of it. `models/*.onnx` is
git-ignored; the models are built locally with the documented pipeline
([models/README.md](models/README.md)). That is deliberate: several of the training sources below
(StarLine, Argoverse 2, BDD100K, Lost and Found) do not allow a free redistribution of what is derived from
them.

## Software

| Component | Used for | Licence |
|---|---|---|
| [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) | Training and ONNX export (`tools/train.py`, `tools/export_onnx.py`); the architecture of the exported models | AGPL-3.0, or an Ultralytics enterprise licence. Weights trained with it are a derivative under the same terms. Anyone who distributes a trained model, or an app that bundles one, has to comply with AGPL-3.0 (or hold the enterprise licence). The inference code in `wazeomatic/` and `android/detector/` does not import Ultralytics; it runs any ONNX file with the same output layout. |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | Inference on the desktop and in the Android app | MIT |
| [OpenCV](https://opencv.org) | Video decoding, drawing | Apache-2.0 |
| [AndroidX CameraX](https://developer.android.com/jetpack/androidx/releases/camera) and the Android Gradle toolchain | The Android apps | Apache-2.0 |
| [Argoverse 2 API (`av2`)](https://github.com/argoverse/av2-api) | Projecting 3D labels into the camera (`tools/argoverse_to_yolo.py`) | MIT |

## Datasets used to train the hazard models

Sizes, download locations and how each is used are in [docs/model-research.md](docs/model-research.md).

| Dataset | Licence | What that means here |
|---|---|---|
| Pothole Dataset IVCNZ (jaygala24/pothole-detection) | MIT | Attribution |
| RDD2022, Road Damage Dataset (Arya, Maeda et al.) | CC BY 4.0 (the United States mirror card states none; the pothole mirror is CC BY-SA 4.0) | Attribution; share-alike for the mirror |
| Traffic_Cone (Junhan0518) | Apache-2.0 | Attribution |
| Roadwork_Cones_Dataset (StarLineResearch) | **None stated on the dataset card** | Treated as research and course use only. Weights trained on it are not released until its authors confirm a licence. |
| ROADWork (Ghosh et al., Carnegie Mellon University) | Open Data Commons Attribution 1.0 | Attribution; images it draws from other sources keep their own terms |
| Argoverse 2 Sensor Dataset (Argo AI) | CC BY-NC-SA 4.0 | **Non-commercial** and share-alike. A model trained on it must not be used commercially. |
| Potholes Detection (Roboflow Universe export) | CC BY 4.0 | Attribution |
| Lost and Found (Pinggera et al., IROS 2016; mirror kumuji/lost_and_found) | Research use per the dataset page; the mirror states none | The `debris` class. Treated as research and course use only. |

## Datasets used to train the classifiers

The vehicle classifier (police, ambulance, fire engine), the weather classifier and the road-surface
classifier ([docs/waze-coverage.md](docs/waze-coverage.md)).

| Dataset | Licence | What that means here |
|---|---|---|
| Wikimedia Commons photos of police cars, ambulances, fire engines and look-alike vehicles (`tools/fetch_commons_vehicles.py`) | Per file: CC0, public domain, CC BY or CC BY-SA only; other files are skipped | Attribution per file; author and licence of each photo are written to `meta.jsonl` next to it. Share-alike files make a redistributed dataset CC BY-SA. |
| Open Images V7 boxes, Ambulance and Deer (`tools/fetch_openimages.py`) | Annotations CC BY 4.0; images CC BY 2.0 as listed by Open Images | Attribution |
| BDD100K (Yu et al., UC Berkeley), weather attribute via the dronefreak mirrors | BDD100K data licence: educational, research and not-for-profit use; commercial use only for BDD and BAIR Commons members | **Non-commercial.** Used for the weather classes and as civilian-vehicle negatives. |
| RSCD, Road Surface Condition Dataset (Zhao et al., Tsinghua University), mirror rezzzq/RSCD-1million | MIT on the mirror card | Attribution; cite the original paper |

## Test footage

Not committed either. `tools/fetch_test_videos.py` downloads two CC0 dashcam clips from Wikimedia
Commons. The UK Road DashCam clips (aap9002, MIT) and the Argoverse 2 sequences (CC BY-NC-SA 4.0)
used for the real-footage checks are fetched by hand as described in
[assets/videos/README.md](assets/videos/README.md). The emergency-vehicle clips used to test police and
crash reporting come from Wikimedia Commons under CC BY / CC BY-SA (`tools/fetch_commons_videos.py`
records author and licence per clip).

## Waze

Waze is a trademark of Google LLC. This project is not affiliated with, endorsed by or sponsored
by Waze or Google. The mock app in `android/mockwaze` is a test double that reproduces the layout
of the report flow so that automated taps can be tested without filing reports; it ships none of
Waze's assets and is installed only on test devices. See
[docs/responsible-use.md](docs/responsible-use.md).
