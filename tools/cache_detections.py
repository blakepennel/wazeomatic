"""Run the detectors over a folder of clips once and store every frame's boxes, for replaying rules quickly.

    python tools/cache_detections.py --clips <datasets>/argoverse2/clips --out runs/cache/argoverse.jsonl

One JSON line per frame: clip name, frame time, frame width and height, and the detections (label, confidence,
normalised cx, cy, w, h) of the hazard model and the COCO model with the shipped settings. Scene rules that need
only boxes (blocked lanes) can then be replayed with other parameters in seconds
(``tools/eval_lanes_argoverse.py --cache``). Extra detector flags go after ``--``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wazeomatic.build import DetectorSpec, add_detector_args, build_detectors  # noqa: E402
from wazeomatic.sources import SourceError, VideoSource  # noqa: E402

VIDEO_EXT = {".mp4", ".webm", ".ogv", ".ogg", ".mkv", ".mov", ".avi"}


def main() -> None:
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=1)
    args = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    dap = argparse.ArgumentParser()
    add_detector_args(dap)
    det = build_detectors(DetectorSpec.from_args(dap.parse_args(extra)))
    clips = sorted(p for p in Path(args.clips).rglob("*") if p.suffix.lower() in VIDEO_EXT)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for clip in clips:
            src = VideoSource(str(clip), step=args.step)
            n = 0
            try:
                for fr in src:
                    ds = [d for dd in det for d in dd(fr.image)]
                    f.write(json.dumps({"clip": clip.name, "t": round(fr.ts, 3), "w": fr.image.shape[1],
                                        "h": fr.image.shape[0],
                                        "dets": [[d.label, round(float(d.confidence), 3)] +
                                                 [round(float(v), 4) for v in d.bbox] for d in ds]}) + "\n")
                    n += 1
            except SourceError as e:
                print(f"[warn] {e}", file=sys.stderr)
            src.release()
            print(f"{clip.name}: {n} frames", flush=True)


if __name__ == "__main__":
    main()
