"""Convert Lost and Found (road obstacles) into YOLO labels for a ``debris`` class.

    python tools/laf_to_yolo.py --out C:/dev/datasets/laf_yolo --class-id 6

Lost and Found (Pinggera et al., IROS 2016; Hugging Face mirror kumuji/lost_and_found) has 2,100 frames of
small objects lying on the road - boxes, tyres, bins, crates, styrofoam, balls - with polygon labels in the
Cityscapes layout. Every object of an obstacle category (category 2 in its ``labels.py``) becomes a box of
class ``--class-id``. The 2048 x 1024 frames are cut into three overlapping 1024 x 1024 tiles so a distant
obstacle keeps its pixels at 640 px training size; a box is kept in a tile when at least 60 % of it is inside.
The dataset's own train / test split becomes our train / val. Licence: research use (see the dataset page);
it is listed in NOTICE.md with the other non-commercial sources.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wazeomatic.paths import datasets_root  # noqa: E402


def obstacle_labels(labels_py: Path) -> set[str]:
    """Names of the labels whose category is 2 (obstacle) in the dataset's labels.py."""
    out = set()
    for m in re.finditer(r"Label\(\s*'([^']+)'\s*,\s*(\d+)\s*,\s*(\d+)", labels_py.read_text(encoding="utf-8")):
        if int(m.group(3)) == 2:
            out.add(m.group(1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="default <datasets>/lost_and_found")
    ap.add_argument("--out", help="default <datasets>/laf_yolo")
    ap.add_argument("--class-id", type=int, default=6)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--min-inside", type=float, default=0.6)
    ap.add_argument("--min-px", type=int, default=6, help="drop boxes smaller than this on either side")
    args = ap.parse_args()
    src = Path(args.src) if args.src else datasets_root() / "lost_and_found"
    out = Path(args.out) if args.out else datasets_root() / "laf_yolo"
    keep = obstacle_labels(src / "labels.py")
    stats = {"frames": 0, "tiles": 0, "boxes": 0, "empty_tiles": 0}
    for split, ours in (("train", "train"), ("test", "val")):
        for pj in sorted((src / "gtCoarse" / split).rglob("*_gtCoarse_polygons.json")):
            d = json.loads(pj.read_text(encoding="utf-8"))
            stem = pj.name.replace("_gtCoarse_polygons.json", "")
            img_path = src / "leftImg8bit" / split / pj.parent.name / f"{stem}_leftImg8bit.png"
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            boxes = []
            for o in d.get("objects", []):
                if o.get("label") not in keep or not o.get("polygon"):
                    continue
                xs = [p[0] for p in o["polygon"]]
                ys = [p[1] for p in o["polygon"]]
                x1, y1, x2, y2 = max(0, min(xs)), max(0, min(ys)), min(w, max(xs)), min(h, max(ys))
                if x2 - x1 >= args.min_px and y2 - y1 >= args.min_px:
                    boxes.append((x1, y1, x2, y2))
            stats["frames"] += 1
            t = min(args.tile, h, w)
            for x0 in sorted({0, (w - t) // 2, w - t}):
                y0 = max(0, h - t)
                lines = []
                for x1, y1, x2, y2 in boxes:
                    ix1, iy1, ix2, iy2 = max(x1, x0), max(y1, y0), min(x2, x0 + t), min(y2, y0 + t)
                    if ix2 <= ix1 or iy2 <= iy1:
                        continue
                    if (ix2 - ix1) * (iy2 - iy1) < args.min_inside * (x2 - x1) * (y2 - y1):
                        continue
                    cx, cy = ((ix1 + ix2) / 2 - x0) / t, ((iy1 + iy2) / 2 - y0) / t
                    lines.append(f"{args.class_id} {cx:.6f} {cy:.6f} {(ix2 - ix1) / t:.6f} {(iy2 - iy1) / t:.6f}")
                name = f"laf_{stem}_x{x0}"
                (out / "images" / ours).mkdir(parents=True, exist_ok=True)
                (out / "labels" / ours).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out / "images" / ours / f"{name}.jpg"), img[y0:y0 + t, x0:x0 + t],
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                (out / "labels" / ours / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                                                 encoding="utf-8")
                stats["tiles"] += 1
                stats["boxes"] += len(lines)
                stats["empty_tiles"] += not lines
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
