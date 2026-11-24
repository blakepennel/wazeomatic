"""Make an int8 copy of an ONNX model for the phone/Pi.

    python tools/quantize_onnx.py models/hazard_yolo11n.onnx
    -> models/hazard_yolo11n.int8.onnx (+ .labels.txt copied)

    python tools/quantize_onnx.py models/hazard_yolo11n_dyn.onnx --static         --calib <datasets>/roadwork/images --calib <datasets>/rdd2022_us/United_States/train/images         --calib <datasets>/bdd100k_weather/data/images/train
    -> models/hazard_yolo11n_dyn.qdq.onnx

Dynamic quantization (the default) needs no calibration data and roughly halves the file, but its
ConvInteger nodes have no XNNPACK kernel, so on the phone they run on the slow default CPU kernels.

--static writes a QDQ model (int8 activations, per-channel int8 weights) from calibration images. ONNX Runtime
fuses each DequantizeLinear -> op -> QuantizeLinear group into the XNNPACK int8 kernels, which use the dot-product
instructions of ARMv8.2 cores (Cortex-A77 and later). The detection head's box decoding (DFL, anchors, the
concatenated output), its class branch (cv3) and the attention block's MatMul / Softmax stay in float: box
coordinates, class scores and attention are where int8 rounding costs the most accuracy. With the class branch
in int8 the pothole scores came out higher on road texture, nearly twice the false pothole boxes on hazard-free
clips; --int8-classes quantizes it anyway.

Calibrate on a mix of scenes: --n images are split evenly over the --calib folders. Work zones alone (ROADWork)
left the pothole branch badly calibrated; work zones + road damage (RDD2022) + ordinary driving (BDD100K) is
what the shipped model uses. Calibration images are letterboxed like the detector does (open-shape
models at the 16:9 shape the phone feeds them). Always re-check accuracy before shipping the file (docs/model-research.md).

--calib-cache keeps the calibrated ranges in a JSON file and reuses them on later runs without calibrating;
tools/qat_onnx.py tunes a model against those fixed ranges (the int8 COCO model is made that way). --stride
calibrates that many images at a time: ONNX Runtime's histogram calibrators keep every intermediate tensor of the
images they hold, about 150 MB an image for the nano at 640 x 384.
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def calibration_reader(model_path: Path, images: list[Path], size: int, open_shape: bool):
    import cv2
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader

    from wazeomatic.detector import letterbox, rect_shape, to_input_tensor

    name = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"]).get_inputs()[0].name

    class Reader(CalibrationDataReader):
        # set_range lets quantize_static calibrate in strides (CalibStridedMinMax): it keeps every intermediate
        # tensor of the images it has read, about 150 MB an image for a YOLO11n at 640 x 384
        def __init__(self):
            self.it = iter(images)

        def __len__(self):
            return len(images)

        def set_range(self, start_index: int, end_index: int):
            self.it = iter(images[start_index:end_index])

        def get_next(self):
            for p in self.it:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                # one shape for every image (the calibrator stacks the outputs): a 16:9 frame as the phone feeds it
                shape = rect_shape(1080, 1920, size) if open_shape else None
                padded, _ = letterbox(img, size, shape=shape)
                return {name: to_input_tensor(padded)}
            return None

    return Reader()


def calibration_files(folders: list[str], n: int, seed: int) -> list[Path]:
    """``n`` images drawn at random, split evenly over ``folders`` (the same draw for the same arguments)."""
    rng, files = random.Random(seed), []
    for i, d in enumerate(folders):
        found = sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not found:
            raise ValueError(f"no images under {d}")
        rng.shuffle(found)
        share = n // len(folders) + (1 if i < n % len(folders) else 0)
        files += found[:share]
    return files


def float_nodes(graph, int8_classes: bool = False) -> list[str]:
    """Nodes left in float: the Detect head's box decoding (not its cv2 box branch), its cv3 class branch unless
    ``int8_classes``, and the attention block's MatMul / Softmax."""
    return [n.name for n in graph.node
            if ("/model.23/" in n.name and "/cv2." not in n.name and ("/cv3." not in n.name or not int8_classes))
            or n.op_type in ("MatMul", "Softmax")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--out")
    ap.add_argument("--per-channel", action="store_true", help="dynamic mode: per-channel weights")
    ap.add_argument("--static", action="store_true", help="QDQ int8 from calibration images (for XNNPACK)")
    ap.add_argument("--calib", action="append", default=[], help="folder of calibration images (repeatable; --n is split evenly over them)")
    ap.add_argument("--n", type=int, default=300, help="calibration images in all, drawn at random")
    ap.add_argument("--size", type=int, help="input size (default: the model's, or its imgsz metadata)")
    ap.add_argument("--method", default="percentile", choices=["minmax", "entropy", "percentile"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--int8-classes", action="store_true",
                    help="static: quantize the Detect head's class branch (cv3) too (more false pothole boxes)")
    ap.add_argument("--stride", type=int, default=10, help="static: images held in memory at once while calibrating")
    ap.add_argument("--calib-cache", help="static: JSON of calibrated tensor ranges; written if missing, else reused "
                                          "without calibrating (tools/qat_onnx.py trains against these ranges)")
    args = ap.parse_args()

    from onnxruntime.quantization.preprocess import quant_pre_process

    src = Path(args.model)
    suffix = ".qdq.onnx" if args.static else ".int8.onnx"
    out = Path(args.out) if args.out else src.with_name(src.stem + suffix)
    pre = src.with_suffix(".preprocessed.onnx")
    # symbolic shape inference cannot follow the open height / width of a --dynamic export; ONNX inference is enough
    quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
    if not args.static:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(str(pre), str(out), weight_type=QuantType.QUInt8, per_channel=args.per_channel)
    else:
        import onnx
        import onnxruntime as ort
        from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

        cache = Path(args.calib_cache) if args.calib_cache else None
        if not args.calib and not (cache and cache.is_file()):
            ap.error("--static needs --calib <folder of images> (or an existing --calib-cache)")
        try:
            files = calibration_files(args.calib, args.n, args.seed)
        except ValueError as e:
            ap.error(str(e))
        sess = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
        shape = sess.get_inputs()[0].shape
        open_shape = not isinstance(shape[2], int)
        meta = sess.get_modelmeta().custom_metadata_map.get("imgsz", "")
        nums = re.findall(r"\d+", meta)
        size = args.size or (int(shape[2]) if not open_shape else int(nums[0]) if nums else 640)
        keep_float = float_nodes(onnx.load(str(pre)).graph, args.int8_classes)
        # calibrate --stride images at a time (the stride must divide the image count)
        stride = next((d for d in range(min(args.stride, len(files)), 0, -1) if len(files) % d == 0), 0) if files else 0
        method = {"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy,
                  "percentile": CalibrationMethod.Percentile}[args.method]
        quantize_static(
            str(pre), str(out), calibration_reader(src, files, size, open_shape),
            quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
            per_channel=True, calibrate_method=method, nodes_to_exclude=keep_float,
            extra_options={"ActivationSymmetric": False, "WeightSymmetric": True,
                           **({"CalibStridedMinMax": stride} if stride else {})},
            **({"calibration_cache_path": str(cache)} if cache else {}),
        )
        print(f"calibrated on {len(files)} images at {size} px ({'own aspect' if open_shape else 'square'}), "
              f"{len(keep_float)} nodes kept in float, method {args.method}")
    pre.unlink(missing_ok=True)
    labels = src.with_suffix(".labels.txt")
    if labels.exists():
        shutil.copyfile(labels, out.with_name(out.name.removesuffix(".onnx") + ".labels.txt"))
    print(f"{src.name}: {src.stat().st_size / 1e6:.1f} MB -> {out.name}: {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
