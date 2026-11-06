"""Turn the Argoverse 2 logs fetched by fetch_argoverse_hazards.py into 2D boxes.

Every cuboid labelled CONSTRUCTION_CONE or CONSTRUCTION_BARREL is projected into the front camera
with the av2 API (motion compensated between the lidar timestamp of the label and the camera
timestamp). The 2D box is the extent of the eight projected corners, clipped to the image. Cuboids
without lidar returns inside them (``num_interior_pts == 0``) are skipped: they are occluded or too
far to see, and a box there would teach the model to hallucinate.

The front camera is portrait (1550 x 2048) with the hood in the bottom quarter and sky in the top;
the frame is cropped to rows [--crop-top, --crop-bottom) so it looks like a dashcam picture.

Output: ``<root>/yolo/images/{train,val}/<log8>_<timestamp>.jpg`` and matching ``labels`` (YOLO format,
class ids from configs/classes.yaml), read by ``tools/prepare_datasets.py`` (source ``argoverse``).

    python tools/argoverse_to_yolo.py            # write labels
    python tools/argoverse_to_yolo.py --check 8  # also render 8 labelled frames to runs/dataset_check/argoverse
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
AV2 = Path("F:/dev/datasets/argoverse2")
CAMERA = "ring_front_center"
CLASS_MAP = {"CONSTRUCTION_CONE": "cone", "CONSTRUCTION_BARREL": "delineator"}


def load_classes() -> list[str]:
    names = yaml.safe_load((ROOT / "configs" / "classes.yaml").read_text(encoding="utf-8"))["names"]
    return [names[i] for i in sorted(names)]


def nearest(sorted_ts: list[int], t: int) -> int:
    import bisect
    i = bisect.bisect_left(sorted_ts, t)
    cands = [sorted_ts[j] for j in (i - 1, i) if 0 <= j < len(sorted_ts)]
    return min(cands, key=lambda c: abs(c - t))


def convert_log(log_dir: Path, split: str, classes: list[str], out_lbl: Path, max_dt_ns: int,
                min_px: float, min_visible: float, keep_empty: float, rng: random.Random,
                crop: tuple[int, int] = (256, 1536)):
    from av2.geometry.camera.pinhole_camera import PinholeCamera
    from av2.structures.cuboid import CuboidList
    from av2.utils.io import read_city_SE3_ego

    cam = PinholeCamera.from_feather(log_dir, CAMERA)
    import pandas as pd
    cuboids = CuboidList.from_feather(log_dir / "annotations.feather").cuboids
    table = pd.read_feather(log_dir / "annotations.feather")
    assert len(table) == len(cuboids), "cuboid list and table out of step"
    interior = table["num_interior_pts"].to_numpy()
    poses = read_city_SE3_ego(log_dir)
    pose_ts = sorted(poses)
    by_ts: dict[int, list] = {}
    for c, n_pts in zip(cuboids, interior):
        if c.category in CLASS_MAP:
            by_ts.setdefault(int(c.timestamp_ns), []).append((c, int(n_pts)))
    ann_ts = sorted({int(c.timestamp_ns) for c in cuboids})
    cam_dir = log_dir / "sensors" / "cameras" / CAMERA
    counts = Counter()
    rows = []
    w = cam.width_px
    top, bottom = crop
    h = bottom - top
    for img in sorted(cam_dir.glob("*.jpg")):
        cam_t = int(img.stem)
        lidar_t = nearest(ann_ts, cam_t)
        if abs(lidar_t - cam_t) > max_dt_ns:
            continue
        pose_cam = poses.get(cam_t) or poses[nearest(pose_ts, cam_t)]
        pose_lidar = poses.get(lidar_t) or poses[nearest(pose_ts, lidar_t)]
        lines = []
        for c, n_pts in by_ts.get(lidar_t, []):
            if n_pts == 0:
                counts["(skipped: no lidar points)"] += 1
                continue
            uv, pts_cam, valid = cam.project_ego_to_img_motion_compensated(c.vertices_m, pose_cam, pose_lidar)
            if (pts_cam[:, 2] <= 0.5).any():
                continue
            x1, y1 = uv[:, 0].min(), uv[:, 1].min() - top
            x2, y2 = uv[:, 0].max(), uv[:, 1].max() - top
            full = max(1e-6, (x2 - x1) * (y2 - y1))
            cx1, cy1, cx2, cy2 = max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)
            if cx2 - cx1 < min_px or cy2 - cy1 < min_px:
                continue
            if (cx2 - cx1) * (cy2 - cy1) / full < min_visible:
                continue
            cls = CLASS_MAP[c.category]
            cid = classes.index(cls)
            bw, bh = cx2 - cx1, cy2 - cy1
            lines.append(f"{cid} {(cx1 + bw / 2) / w:.6f} {(cy1 + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
            counts[cls] += 1
        if not lines:
            if rng.random() >= keep_empty:
                continue
            counts["(negative)"] += 1
        stem = f"{log_dir.name[:8]}_{cam_t}"
        lbl = out_lbl / split / f"{stem}.txt"
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        out_img = out_lbl.parent / "images" / split / f"{stem}.jpg"
        if not out_img.exists():
            out_img.parent.mkdir(parents=True, exist_ok=True)
            im = cv2.imread(str(img))
            cv2.imwrite(str(out_img), im[top:bottom], [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append({"image": str(out_img).replace("\\", "/"), "label": str(lbl).replace("\\", "/"), "split": split})
        counts["(frames)"] += 1
    return rows, counts


def render_check(rows: list[dict], classes: list[str], n: int) -> None:
    out = ROOT / "runs" / "dataset_check" / "argoverse"
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    positives = [r for r in rows if Path(r["label"]).read_text().strip()]
    for r in rng.sample(positives, min(n, len(positives))):
        im = cv2.imread(r["image"])
        h, w = im.shape[:2]
        for line in Path(r["label"]).read_text().splitlines():
            cid, cx, cy, bw, bh = line.split()
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            p1 = (int(cx - bw / 2), int(cy - bh / 2))
            p2 = (int(cx + bw / 2), int(cy + bh / 2))
            cv2.rectangle(im, p1, p2, (0, 255, 0), 3)
            cv2.putText(im, classes[int(cid)], (p1[0], max(0, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imwrite(str(out / (Path(r["image"]).stem + ".jpg")), cv2.resize(im, (w // 2, h // 2)))
    print(f"check images in {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(AV2))
    ap.add_argument("--max-dt-ms", type=float, default=60.0, help="max lidar/camera timestamp gap")
    ap.add_argument("--min-px", type=float, default=8.0, help="drop boxes smaller than this (either side)")
    ap.add_argument("--min-visible", type=float, default=0.5, help="drop boxes mostly outside the image")
    ap.add_argument("--keep-empty", type=float, default=0.35, help="share of hazard-free frames kept as negatives")
    ap.add_argument("--check", type=int, default=0, help="render N labelled frames for a visual check")
    args = ap.parse_args()
    root = Path(args.root)
    classes = load_classes()
    out_lbl = root / "yolo" / "labels"
    rng = random.Random(0)
    all_rows, total = [], Counter()
    for split in ("train", "val"):
        for log_dir in sorted((root / "logs" / split).glob("*")):
            if not (log_dir / "annotations.feather").exists():
                continue
            rows, counts = convert_log(log_dir, split, classes, out_lbl, int(args.max_dt_ms * 1e6), args.min_px,
                                       args.min_visible, args.keep_empty, rng)
            all_rows += rows
            total.update(counts)
            print(f"{split} {log_dir.name[:8]}: {dict(counts)}", flush=True)
    (root / "yolo" / "index.json").write_text(json.dumps(all_rows, indent=0))
    print("total:", dict(total))
    if args.check:
        render_check(all_rows, classes, args.check)


if __name__ == "__main__":
    main()
