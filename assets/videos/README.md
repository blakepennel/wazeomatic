# Test videos

Not committed; anything in this folder except this file is ignored by git. Fetch the two
public-domain dashcam clips used by the simulator with:

    python tools/fetch_test_videos.py

The script needs ffmpeg on the PATH (or the `FFMPEG` environment variable) to convert the
downloads to MP4. Without it the `.ogv` files are kept: the desktop pipeline can read them, the
Android app cannot.

Both clips are CC0 from Wikimedia Commons (see the script for URLs). Put your own
dashcam recordings here too.

Note (2026-11-06): the two clips are the same two-minute recording published twice on Commons (720p and 1080p encodes); treat replays on both as one drive, clean and camera-degraded. The recording starts in a parking garage with a real traffic cone at about 5.5 s, so it is not entirely hazard-free.

## Other clips

There is no fetch script for these. Sources and licences are in the Datasets
table of `docs/model-research.md`.

| File | Source | Licence | Content |
|------|--------|---------|---------|
| `uk_dashcam_002_1080p.mp4`, `uk_dashcam_006_1080p.mp4` | [aap9002/UK-Road-DashCam](https://huggingface.co/datasets/aap9002/UK-Road-DashCam), 4K clips converted to 1080p | MIT | Hazard-free real footage, overcast suburban and rural England |
| `argo_02678d04_clean.mp4` | Argoverse 2 sensor dataset, one validation log encoded as video | CC BY-NC-SA 4.0 | No hazards labelled |
| `labelled/argo_2ff4f798_workzone.mp4` | Argoverse 2 sensor dataset, one validation log encoded as video | CC BY-NC-SA 4.0 | A real work zone (cones and barrels) |

## The labelled folder

`assets/videos/labelled/` is for clips that contain hazards. `tools/eval_events.py` counts every
report on the `.mp4` files directly in `assets/videos/` as a false report and does not look into
subfolders, so a clip with real hazards must go into `labelled/`, not here.
