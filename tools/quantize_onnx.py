"""Make an int8 (dynamic quantization) copy of an ONNX model for the phone/Pi.

    python tools/quantize_onnx.py models/hazard_yolo11n.onnx
    -> models/hazard_yolo11n.int8.onnx (+ .labels.txt copied)

Dynamic quantization needs no calibration data and roughly halves the file; on CPU
execution providers it usually speeds up the conv-heavy backbone. Always re-check
accuracy with tools/bench_onnx.py --compare before shipping the int8 file.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out")
    ap.add_argument("--per-channel", action="store_true")
    args = ap.parse_args()

    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.preprocess import quant_pre_process

    src = Path(args.model)
    out = Path(args.out) if args.out else src.with_suffix(".int8.onnx")
    pre = src.with_suffix(".preprocessed.onnx")
    quant_pre_process(str(src), str(pre))
    quantize_dynamic(str(pre), str(out), weight_type=QuantType.QUInt8, per_channel=args.per_channel)
    pre.unlink(missing_ok=True)
    labels = src.with_suffix(".labels.txt")
    if labels.exists():
        shutil.copyfile(labels, out.with_suffix(".labels.txt"))
    print(f"{src.name}: {src.stat().st_size / 1e6:.1f} MB -> {out.name}: {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
