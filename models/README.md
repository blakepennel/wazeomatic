# Model files

Exported models live here but are not committed (see `.gitignore`).

| File | What | How to produce |
|------|------|----------------|
| `hazard_yolo11n.onnx` | Fine-tuned road hazard detector (pothole, cone, construction sign, delineator) | `python tools/train.py` then `python tools/export_onnx.py` |
| `coco_yolo11n.onnx` | Stock COCO detector used for vehicles, people, animals | `python tools/export_onnx.py --coco` |
| `*.int8.onnx` | Quantized copies for the phone | `python tools/quantize_onnx.py <model>` |

See `docs/model-research.md` for the reasoning behind the model choice and the
measured sizes and latencies.
