"""Mine whole vehicle tracks from dashcam clips: one look by eye labels every frame of a vehicle.

    python tools/mine_vehicle_tracks.py scan --clips <datasets>/argoverse2/clips --clip-half 0 \\
        --models models/vehicle_v6.onnx models/vehicle_v5.onnx --work <datasets>/mining_tracks/h0
    python tools/mine_vehicle_tracks.py sheets --work <datasets>/mining_tracks/h0 --min-score 0.3
    python tools/mine_vehicle_tracks.py label --work <datasets>/mining_tracks/h0 --yes 0,3,7-9 --skip 4

``tools/mine_vehicle_crops.py`` ranks single crops by the classifier's own score, so it can only find what the
classifier already half believes: a police car it scores 0.00 from behind never reaches a contact sheet. Here
the unit is a track (the same COCO vehicle box followed frame to frame). A track is shown when any of its frames
scores at least ``--min-score`` with any model; a person checks the strip of its frames, and ``label`` files
every frame of a confirmed track, including the rear and far views the classifier got wrong, under
``<datasets>/mined_tracks/<class>/<clip>_t<k>/``. Tracks shown and not confirmed go to ``civilian``, but only
their frames that scored at least ``--neg-score``: those are the look-alikes worth learning from.
``tools/build_vehicle_crops.py`` (source ``tracks``) keeps each track on one side of the train / val split.

``scan`` uses the pipeline's own candidate rule (the four largest COCO car / truck / bus boxes,
``VehicleClassifier.candidates``), so the crops are the ones the phone classifies. Use it only on clips that no
evaluation scores: the Argoverse half that ``--clip-half 1`` evaluations leave out, or new logs fetched with
``tools/fetch_argoverse_hazards.py --stage clips --clips-out clips_mine``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wazeomatic.classifiers import OnnxClassifier, VehicleClassifier, square_crop  # noqa: E402
from wazeomatic.paths import datasets_root  # noqa: E402

VIDEO_EXT = {".mp4", ".webm", ".mov", ".mkv"}


def parse_ranges(spec: str | None) -> set[int]:
    out: set[int] = set()
    for part in filter(None, (p.strip() for p in (spec or "").split(","))):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    return inter / max(1e-9, a[2] * a[3] + b[2] * b[3] - inter)


def link(frames: list[list[dict]], min_iou: float = 0.3, max_gap: int = 2) -> list[list[tuple[int, dict]]]:
    """Greedy frame-to-frame association: a box joins the open track whose last box it overlaps most (IoU at
    least ``min_iou``), a track survives ``max_gap`` frames without a box. Returns tracks as (frame, box)."""
    tracks: list[list[tuple[int, dict]]] = []
    open_: list[int] = []
    for t, boxes in enumerate(frames):
        open_ = [k for k in open_ if t - tracks[k][-1][0] <= max_gap + 1]
        free = list(range(len(boxes)))
        pairs = sorted(((iou(tracks[k][-1][1]["bbox"], boxes[j]["bbox"]), k, j) for k in open_ for j in free),
                       reverse=True)
        used_k, used_j = set(), set()
        for s, k, j in pairs:
            if s < min_iou or k in used_k or j in used_j:
                continue
            tracks[k].append((t, boxes[j]))
            used_k.add(k)
            used_j.add(j)
        for j in free:
            if j not in used_j:
                tracks.append([(t, boxes[j])])
                open_.append(len(tracks) - 1)
    return tracks


def _hist(im: np.ndarray, bbox) -> np.ndarray:
    hsv = cv2.cvtColor(square_crop(im, tuple(bbox), 0.0, 64), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [12, 6, 6], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(h, h).flatten()


def split_by_look(track: list, ims: list[np.ndarray], max_dist: float) -> list[list]:
    """Cut a track where two consecutive crops stop looking alike (Bhattacharyya distance of their HSV
    histograms above ``max_dist``): box linking alone hands a track to the next car when a vehicle leaves the
    frame next to another one."""
    parts, prev = [[track[0]]], _hist(ims[track[0][0]], track[0][1]["bbox"])
    for t, b in track[1:]:
        h = _hist(ims[t], b["bbox"])
        last = parts[-1][-1][1]["bbox"]
        area = (b["bbox"][2] * b["bbox"][3]) / max(1e-9, last[2] * last[3])
        if cv2.compareHist(prev, h, cv2.HISTCMP_BHATTACHARYYA) > max_dist or not 0.6 <= area <= 1.67:
            parts.append([])
        parts[-1].append((t, b))
        prev = h
    return parts


def inside_frame(bbox, margin: float = 0.01) -> bool:
    """The box does not touch the left or right edge: a vehicle cut off by the frame is half a vehicle, and
    where tracks swap to a neighbour."""
    return bbox[0] - bbox[2] / 2 > margin and bbox[0] + bbox[2] / 2 < 1 - margin


def clip_list(folder: Path, half: int | None) -> list[Path]:
    clips = sorted(p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXT)
    if half is not None:
        from mine_vehicle_crops import clip_half
        clips = [p for p in clips if clip_half(p.stem) == half]
    return clips


def scan(args) -> None:
    from wazeomatic.build import DetectorSpec, add_detector_args, build_detectors
    from wazeomatic.sources import VideoSource

    dp = argparse.ArgumentParser()
    add_detector_args(dp)
    dets = build_detectors(DetectorSpec.from_args(dp.parse_args(["--no-hazard"])))
    vc = VehicleClassifier(args.models[0])
    clfs = [OnnxClassifier(m, threads=4) for m in args.models]
    out = {"models": [Path(m).stem for m in args.models], "clips": {}}
    clips = clip_list(Path(args.clips), args.clip_half)
    for n, c in enumerate(clips):
        rows = []
        src = VideoSource(str(c), step=1)
        for fr in src:
            cand = vc.candidates(fr.image, [d for dm in dets for d in dm(fr.image)])
            per = []
            for clf in clfs:
                crops = [square_crop(fr.image, d.bbox, vc.pad, clf.size) for d in cand]
                pi = clf.labels.index(args.cls)
                per.append([float(p[pi]) for p in clf.predict(crops)] if crops else [])
            rows.append([{"bbox": [round(v, 4) for v in d.bbox], "score": [round(pm[k], 4) for pm in per]}
                         for k, d in enumerate(cand)])
        src.release()
        out["clips"][str(c)] = rows
        print(f"[{n + 1}/{len(clips)}] {c.stem}: {len(rows)} frames", flush=True)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    (work / "scan.json").write_text(json.dumps(out), encoding="utf-8")


def read_frames(path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        out.append(im)
    cap.release()
    return out


def sheets(args) -> None:
    work = Path(args.work)
    data = json.loads((work / "scan.json").read_text(encoding="utf-8"))
    shown = []
    for clip, rows in data["clips"].items():
        tracks = link(rows)
        if any(max(max(b["score"]) for _, b in tr) >= args.min_score for tr in tracks):
            ims = read_frames(clip)
            tracks = [p for tr in tracks for p in split_by_look(tr, ims, args.max_look)]
        # chosen on every frame (a police car often scores best as it passes at the edge), filed without the
        # frames cut off by the edge
        tracks = [(max(max(b["score"]) for _, b in tr), [(t, b) for t, b in tr if inside_frame(b["bbox"])])
                  for tr in tracks]
        for k, (best, tr) in enumerate((s, tr) for s, tr in tracks if tr):
            if best >= args.min_score and len(tr) >= args.min_len:
                shown.append({"clip": clip, "track": k, "best": round(best, 3),
                              "frames": [[t, b["bbox"], b["score"]] for t, b in tr]})
    shown.sort(key=lambda s: -s["best"])
    (work / "tracks.json").write_text(json.dumps(shown), encoding="utf-8")
    # one row per track: up to 8 frames spread over it, each crop 120 px, best score on the first
    cache: dict[str, list[np.ndarray]] = {}
    rows_img = []
    for i, s in enumerate(shown):
        if s["clip"] not in cache:
            cache = {s["clip"]: read_frames(s["clip"])}
        ims = cache[s["clip"]]
        pick = np.linspace(0, len(s["frames"]) - 1, min(8, len(s["frames"]))).round().astype(int)
        tiles = []
        for j in pick:
            t, bb, sc = s["frames"][j]
            tile = cv2.resize(square_crop(ims[t], tuple(bb), 0.12, 120), (120, 120))
            cv2.putText(tile, f"{max(sc):.2f}", (3, 115), 0, 0.4, (0, 255, 255), 1)
            tiles.append(tile)
        tiles += [np.zeros((120, 120, 3), np.uint8)] * (8 - len(tiles))
        label = np.zeros((120, 70, 3), np.uint8)
        cv2.putText(label, str(i), (5, 50), 0, 0.9, (255, 255, 255), 2)
        cv2.putText(label, Path(s["clip"]).stem[:8], (2, 80), 0, 0.35, (200, 200, 200), 1)
        rows_img.append(np.hstack([label] + tiles))
    for p in range(0, len(rows_img), args.per_sheet):
        cv2.imwrite(str(work / f"sheet_{p:04d}.jpg"), np.vstack(rows_img[p:p + args.per_sheet]))
    print(f"{len(shown)} tracks with a frame at or above {args.min_score}; sheets in {work}")


def label(args) -> None:
    work = Path(args.work)
    shown = json.loads((work / "tracks.json").read_text(encoding="utf-8"))
    yes, skip = parse_ranges(args.yes), parse_ranges(args.skip)
    reviewed = parse_ranges(args.reviewed) if args.reviewed else set(range(len(shown)))
    root = (Path(args.datasets) if args.datasets else datasets_root()) / "mined_tracks"
    counts = {args.cls: 0, "civilian": 0}
    cache: dict[str, list[np.ndarray]] = {}
    for i, s in enumerate(shown):
        if i not in reviewed or i in skip:
            continue
        cls = args.cls if i in yes else "civilian"
        frames = s["frames"] if cls == args.cls else [f for f in s["frames"] if max(f[2]) >= args.neg_score]
        if not frames:
            continue
        if s["clip"] not in cache:
            cache = {s["clip"]: read_frames(s["clip"])}
        dest = root / cls / f"{Path(s['clip']).stem}_t{s['track']}"
        dest.mkdir(parents=True, exist_ok=True)
        for t, bb, _ in frames:
            if min(bb[2] * cache[s["clip"]][t].shape[1], bb[3] * cache[s["clip"]][t].shape[0]) < args.min_px:
                continue
            cv2.imwrite(str(dest / f"f{t:03d}.jpg"), square_crop(cache[s["clip"]][t], tuple(bb), 0.12, 192))
            counts[cls] += 1
    print(json.dumps(counts))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["scan", "sheets", "label"])
    ap.add_argument("--work", required=True, help="folder for scan.json, tracks.json and the sheets")
    ap.add_argument("--clips", help="scan: folder of clips (searched recursively)")
    ap.add_argument("--clip-half", type=int, choices=[0, 1], help="scan: only this half (as mine_vehicle_crops)")
    ap.add_argument("--models", nargs="+", default=["models/vehicle_cls.onnx", "models/vehicle_cls2.onnx"])
    ap.add_argument("--cls", default="police")
    ap.add_argument("--min-score", type=float, default=0.3, help="sheets: show tracks with a frame this high")
    ap.add_argument("--min-len", type=int, default=3, help="sheets: frames a track needs")
    ap.add_argument("--per-sheet", type=int, default=10)
    ap.add_argument("--max-look", type=float, default=0.5, help="sheets: split a track at this histogram distance")
    ap.add_argument("--yes", help="label: track indices that ARE the class, e.g. 0,3,7-9")
    ap.add_argument("--skip", help="label: indices too unclear to file either way")
    ap.add_argument("--reviewed", help="label: indices looked at (default all); the rest are left alone")
    ap.add_argument("--neg-score", type=float, default=0.5, help="label: frames of a rejected track kept")
    ap.add_argument("--min-px", type=int, default=28)
    ap.add_argument("--datasets")
    args = ap.parse_args()
    {"scan": scan, "sheets": sheets, "label": label}[args.cmd](args)


if __name__ == "__main__":
    main()
