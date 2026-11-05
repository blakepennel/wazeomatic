"""Benchmark ONNX models with ONNX Runtime on this machine (CPU, phone-like thread count).

    python tools/bench_onnx.py models/*.onnx --threads 4 --runs 50
    python tools/bench_onnx.py models/hazard_yolo11n.onnx models/hazard_yolo11n.int8.onnx --compare clip.mp4

Prints a markdown table that can be pasted into docs/model-research.md.
"""
from __future__ import annotations

import argparse
import glob
import statistics
import time
from pathlib import Path

import numpy as np


def bench(path: Path, threads: int, runs: int, warmup: int = 10) -> dict:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    x = np.random.rand(*shape).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {inp.name: x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {inp.name: x})
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "model": path.name,
        "size_mb": round(path.stat().st_size / 1e6, 1),
        "input": "x".join(str(s) for s in shape[2:]),
        "classes": sess.get_outputs()[0].shape[1] - 4 if isinstance(sess.get_outputs()[0].shape[1], int) else "?",
        "mean_ms": round(statistics.mean(times), 2),
        "p50_ms": round(times[len(times) // 2], 2),
        "p90_ms": round(times[int(len(times) * 0.9)], 2),
        "fps": round(1000 / statistics.mean(times), 1),
    }


def compare_on_video(a: Path, b: Path, video: str, frames: int, threads: int) -> dict:
    """Agreement between two models (e.g. fp32 vs int8): how often top detections match."""
    from wazeomatic.detector import OnnxDetector
    from wazeomatic.sources import VideoSource

    labels = [ln.strip() for ln in a.with_suffix(".labels.txt").read_text().splitlines() if ln.strip()]
    da = OnnxDetector(a, labels, "a", 0.3, 0.5, threads=threads)
    db = OnnxDetector(b, labels, "b", 0.3, 0.5, threads=threads)
    same = total = 0
    for fr in VideoSource(video, step=10, max_frames=frames):
        ra = sorted((d.label for d in da(fr.image)))
        rb = sorted((d.label for d in db(fr.image)))
        total += 1
        same += ra == rb
    return {"frames": total, "identical_label_sets": same, "agreement": round(same / max(total, 1), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--compare", help="video path: compare first two models' detections")
    ap.add_argument("--frames", type=int, default=100)
    args = ap.parse_args()

    paths = [Path(p) for pat in args.models for p in (glob.glob(pat) or [pat])]
    paths = [p for p in paths if p.suffix == ".onnx"]
    rows = [bench(p, args.threads, args.runs) for p in paths]
    cols = ["model", "size_mb", "input", "classes", "mean_ms", "p50_ms", "p90_ms", "fps"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")
    print(f"\n(onnxruntime CPU, {args.threads} threads, {args.runs} runs after warm-up)")
    if args.compare and len(paths) >= 2:
        print("\ncompare:", compare_on_video(paths[0], paths[1], args.compare, args.frames, args.threads))


if __name__ == "__main__":
    main()
