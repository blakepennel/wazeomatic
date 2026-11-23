"""Label every detected vehicle in the Argoverse 2 clips as moving or stopped, from the 3D tracks.

    python tools/cache_detections.py --clips <datasets>/argoverse2/clips --clip-half 1 --out runs/cache/argo_h1.jsonl
    python tools/label_vehicle_motion_argoverse.py --cache runs/cache/argo_h1.jsonl --out runs/cache/argo_h1_motion.jsonl

Waze wants police reported where they stand, not while they drive, so the Police report needs to know whether
the police car is moving in the world. The rule sees only what the phone sees: the detector's box in each frame.
This tool gives those boxes the truth. Each annotated vehicle cuboid (``annotations.feather``) is projected into
the front camera (motion compensated, with the av2 API, as ``tools/argoverse_to_yolo.py`` does) and into the
clip's crop (rows 15-80 % of the portrait image, ``tools/fetch_argoverse_hazards.py --stage clips``); a cached
detector box (car, truck, bus, conf >= 0.35) that overlaps a projected cuboid with IoU >= 0.4 takes its track
and the track's speed over the ground: the city-frame position of the cuboid centre, differenced over +-0.5 s.
Our own speed comes from the ego pose the same way.

One JSON line per matched box: clip, frame time, detector box, label, confidence, track id, the track's ground
speed (m/s), our speed (m/s) and the cuboid's distance ahead of the camera (m). Boxes without a matching cuboid
(parked cars beyond the labelled range, false boxes) are left out.

Where the log's vector map is on disk (``tools/fetch_argoverse_hazards.py --stage maps``) each line also says
where the vehicle is on the road: ``in_lane`` (its centre lies in a vehicle, bus or bike lane segment; a car
parked at the kerb or on a shoulder lies beside them) and ``side``, the direction of the nearest lane against
ours: ``ours`` (within 45 degrees), ``other`` (the opposite carriageway), ``cross`` (a side street) or null (no
lane within 8 m). ``lat_m`` is the cuboid centre's offset to our left (m, ego frame) and ``yaw`` its heading
against ours (degrees).
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wazeomatic.paths import datasets_root  # noqa: E402

CAMERA = "ring_front_center"
VEHICLE_CATEGORIES = {"REGULAR_VEHICLE", "LARGE_VEHICLE", "BUS", "BOX_TRUCK", "TRUCK", "VEHICULAR_TRAILER",
                      "TRUCK_CAB", "SCHOOL_BUS", "ARTICULATED_BUS", "MESSAGE_BOARD_TRAILER"}
DET_LABELS = ("car", "truck", "bus")
CROP = (0.15, 0.80)              # rows of the portrait image kept in the clips


def nearest(sorted_ts: list[int], t: int) -> int:
    i = bisect.bisect_left(sorted_ts, t)
    cands = [sorted_ts[j] for j in (i - 1, i) if 0 <= j < len(sorted_ts)]
    return min(cands, key=lambda s: abs(s - t))


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    return inter / max(1e-9, a[2] * a[3] + b[2] * b[3] - inter)


def speed_at(ts: np.ndarray, xy: np.ndarray, t: float, half: float = 0.5) -> float:
    """Ground speed at t (seconds) from positions xy at times ts, over [t - half, t + half] clipped to the track."""
    a, b = max(ts[0], t - half), min(ts[-1], t + half)
    if b - a < 0.3:
        return float("nan")
    pa = np.array([np.interp(a, ts, xy[:, 0]), np.interp(a, ts, xy[:, 1])])
    pb = np.array([np.interp(b, ts, xy[:, 0]), np.interp(b, ts, xy[:, 1])])
    return float(np.linalg.norm(pb - pa) / (b - a))


def _inside(poly: np.ndarray, x: float, y: float) -> bool:
    """Point in polygon (ray casting), poly an (N, 2+) array of vertices."""
    px, py = poly[:, 0], poly[:, 1]
    qx, qy = np.roll(px, -1), np.roll(py, -1)
    cross = (py > y) != (qy > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xs = px + (y - py) * (qx - px) / (qy - py)
    return bool(np.count_nonzero(cross & (x < xs)) % 2)


def road_context(avm, xy: np.ndarray, heading: np.ndarray) -> tuple[bool, str | None]:
    """(in a lane, side of the road) of a city-frame point, against our heading (unit vector)."""
    lanes = avm.get_nearby_lane_segments(xy, 8.0)
    if not lanes:
        return False, None
    in_lane = any(_inside(ls.polygon_boundary, xy[0], xy[1]) for ls in lanes)
    best, tangent = 1e9, None
    for ls in lanes:
        c = avm.get_lane_segment_centerline(ls.id)[:, :2]
        if len(c) < 2:
            continue
        seg = c[1:] - c[:-1]
        t = np.clip(((xy - c[:-1]) * seg).sum(1) / np.maximum((seg * seg).sum(1), 1e-9), 0, 1)
        d = np.linalg.norm(c[:-1] + seg * t[:, None] - xy, axis=1)
        k = int(np.argmin(d))
        if d[k] < best:
            best, tangent = d[k], seg[k] / max(np.linalg.norm(seg[k]), 1e-9)
    if tangent is None:
        return in_lane, None
    cos = float(tangent @ heading)
    return in_lane, "ours" if cos > 0.707 else "other" if cos < -0.707 else "cross"


def label_clip(log: Path, rows: list[dict], max_dt_ns: int = 60_000_000) -> list[dict]:
    import pandas as pd
    from av2.geometry.camera.pinhole_camera import PinholeCamera
    from av2.structures.cuboid import CuboidList
    from av2.utils.io import read_city_SE3_ego

    cam = PinholeCamera.from_feather(log, CAMERA)
    table = pd.read_feather(log / "annotations.feather")
    cuboids = CuboidList.from_feather(log / "annotations.feather").cuboids
    poses = read_city_SE3_ego(log)
    pose_ts = sorted(poses)
    avm = None
    maps = list((log / "map").glob("log_map_archive_*.json"))
    if maps:
        from av2.map.map_api import ArgoverseStaticMap
        avm = ArgoverseStaticMap.from_json(maps[0])
    W, H = cam.width_px, cam.height_px
    top, bot = int(H * CROP[0]), int(H * CROP[1])
    ch = bot - top

    # tracks in the city frame
    tracks: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    by_ts: dict[int, list] = defaultdict(list)
    for c, (_, r) in zip(cuboids, table.iterrows()):
        if r.category not in VEHICLE_CATEGORIES:
            continue
        ts = int(r.timestamp_ns)
        pose = poses.get(ts) or poses[nearest(pose_ts, ts)]
        p = pose.transform_point_cloud(np.array([[r.tx_m, r.ty_m, r.tz_m]]))[0]
        tracks[r.track_uuid].append((ts / 1e9, p[0], p[1]))
        if r.num_interior_pts > 0:
            yaw = np.degrees(np.arctan2(2 * (r.qw * r.qz + r.qx * r.qy), 1 - 2 * (r.qy ** 2 + r.qz ** 2)))
            by_ts[ts].append((c, r.track_uuid, (float(r.ty_m), float(yaw), p[:2].copy(), pose)))
    track_arr = {}
    for k, v in tracks.items():
        v.sort()
        a = np.array(v)
        track_arr[k] = (a[:, 0], a[:, 1:3])
    ego_ts = np.array([t / 1e9 for t in pose_ts])
    ego_xy = np.array([poses[t].translation[:2] for t in pose_ts])
    ann_ts = sorted(by_ts)

    jpgs = sorted((log / "sensors" / "cameras" / CAMERA).glob("*.jpg"))
    out = []
    for k, r in enumerate(sorted(rows, key=lambda r: r["t"])):
        if k >= len(jpgs) or not ann_ts:
            break
        cam_t = int(jpgs[k].stem)
        lidar_t = nearest(ann_ts, cam_t)
        if abs(lidar_t - cam_t) > max_dt_ns:
            continue
        pose_cam = poses.get(cam_t) or poses[nearest(pose_ts, cam_t)]
        pose_lidar = poses.get(lidar_t) or poses[nearest(pose_ts, lidar_t)]
        proj = []
        for c, tid, ctx in by_ts[lidar_t]:
            uv, pts_cam, _ = cam.project_ego_to_img_motion_compensated(c.vertices_m, pose_cam, pose_lidar)
            if (pts_cam[:, 2] <= 0.5).any():
                continue
            x1, x2 = max(0.0, uv[:, 0].min()), min(float(W), uv[:, 0].max())
            y1, y2 = max(0.0, uv[:, 1].min() - top), min(float(ch), uv[:, 1].max() - top)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            box = ((x1 + x2) / 2 / W, (y1 + y2) / 2 / ch, (x2 - x1) / W, (y2 - y1) / ch)
            proj.append((box, tid, float(pts_cam[:, 2].mean()), ctx))
        dets = [d for d in r["dets"] if d[0] in DET_LABELS and d[1] >= 0.35]
        pairs = sorted(((iou(tuple(d[2:6]), p[0]), i, j) for i, d in enumerate(dets) for j, p in enumerate(proj)),
                       reverse=True)
        used_d, used_p = set(), set()
        t_s = cam_t / 1e9
        for ov, i, j in pairs:
            if ov < 0.4:
                break
            if i in used_d or j in used_p:
                continue
            used_d.add(i)
            used_p.add(j)
            d, (box, tid, depth, (lat, yaw, cxy, pose)) = dets[i], proj[j]
            ts_arr, xy = track_arr[tid]
            in_lane, side = (None, None)
            if avm is not None:
                in_lane, side = road_context(avm, cxy, pose.rotation[:2, 0] / np.linalg.norm(pose.rotation[:2, 0]))
            out.append({"clip": r["clip"], "t": r["t"], "bbox": d[2:6], "label": d[0], "conf": d[1],
                        "track": tid[:8], "speed": round(speed_at(ts_arr, xy, t_s), 3),
                        "ego_speed": round(speed_at(ego_ts, ego_xy, t_s), 3), "depth_m": round(depth, 2),
                        "iou": round(ov, 3), "lat_m": round(lat, 2), "yaw": round(yaw, 1),
                        "in_lane": in_lane, "side": side})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--root", default=None, help="argoverse2 folder (default <datasets>/argoverse2)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.root) if args.root else datasets_root() / "argoverse2"
    logs = {p.name[:8]: p for split in ("train", "val") for p in (root / "logs" / split).glob("*") if p.is_dir()}
    frames = defaultdict(list)
    with open(args.cache, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            frames[r["clip"]].append(r)
    n = 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i, (clip, rows) in enumerate(sorted(frames.items())):
            log = logs.get(clip[:8])
            if log is None or not (log / "annotations.feather").exists():
                continue
            for rec in label_clip(log, rows):
                f.write(json.dumps(rec) + "\n")
                n += 1
            print(f"[{i + 1}/{len(frames)}] {clip}: {n} labelled boxes so far", flush=True)
    print(f"{n} boxes -> {args.out}")


if __name__ == "__main__":
    main()
