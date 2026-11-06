"""Pull road-work frames out of the Argoverse 2 sensor dataset (anonymous S3, no login).

Argoverse 2 labels every log with 3D cuboids, including CONSTRUCTION_CONE, CONSTRUCTION_BARREL,
BOLLARD and SIGN, from a real car driving six US cities. Projecting those cuboids into the front
camera gives 2D boxes on genuine dashcam-style footage at every distance, which is exactly what the
StarLine set lacks (one country, four cameras) and what we cannot label by hand.

Stage 1 (--stage annotations): list the train/val logs and fetch every ``annotations.feather``
(about 300 KB each) so we can rank logs by how many cones and barrels they contain.
Stage 2 (--stage images): for the top ``--logs`` logs download the calibration, ego poses and every
``--every``-th front-camera frame, then write YOLO labels by projecting the cuboids.

    python tools/fetch_argoverse_hazards.py --stage annotations
    python tools/fetch_argoverse_hazards.py --stage images --logs 60 --every 5

Licence: Argoverse 2 is CC BY-NC-SA 4.0 (research use).
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BUCKET = "https://s3.amazonaws.com/argoverse"
PREFIX = "datasets/av2/sensor"
OUT = Path("F:/dev/datasets/argoverse2")
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
CAMERA = "ring_front_center"

# Argoverse category -> our class (configs/classes.yaml order)
CLASS_MAP = {
    "CONSTRUCTION_CONE": "cone",
    "CONSTRUCTION_BARREL": "delineator",   # drums map to the same Waze report as delineators
}
NEGATIVE_CATS = {"BOLLARD", "SIGN"}  # present in the scene but not something we report; kept as context


def list_prefixes(prefix: str) -> list[str]:
    out, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix, "delimiter": "/", "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = requests.get(BUCKET, params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        out += [p.find(NS + "Prefix").text for p in root.findall(NS + "CommonPrefixes")]
        if root.find(NS + "IsTruncated").text != "true":
            return out
        token = root.find(NS + "NextContinuationToken").text


def list_keys(prefix: str) -> list[tuple[str, int]]:
    out, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = requests.get(BUCKET, params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        out += [(c.find(NS + "Key").text, int(c.find(NS + "Size").text)) for c in root.findall(NS + "Contents")]
        if root.find(NS + "IsTruncated").text != "true":
            return out
        token = root.find(NS + "NextContinuationToken").text


def fetch(key: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(f"{BUCKET}/{key}", timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def stage_annotations(workers: int) -> None:
    logs = []
    for split in ("train", "val"):
        logs += [(split, p.rstrip("/").split("/")[-1]) for p in list_prefixes(f"{PREFIX}/{split}/")]
    print(f"{len(logs)} logs")
    jobs = {}
    with ThreadPoolExecutor(workers) as ex:
        for split, log in logs:
            key = f"{PREFIX}/{split}/{log}/annotations.feather"
            jobs[ex.submit(fetch, key, OUT / "annotations" / split / f"{log}.feather")] = (split, log)
        done = 0
        for f in as_completed(jobs):
            f.result()
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(jobs)}")
    rank_logs()


def rank_logs() -> list[dict]:
    import pandas as pd
    rows = []
    for split in ("train", "val"):
        for f in sorted((OUT / "annotations" / split).glob("*.feather")):
            df = pd.read_feather(f)
            counts = df["category"].value_counts()
            cones = int(counts.get("CONSTRUCTION_CONE", 0))
            barrels = int(counts.get("CONSTRUCTION_BARREL", 0))
            # distinct tracks matter more than cuboid rows (one object = many timestamps)
            tracks = df[df["category"].isin(CLASS_MAP)]["track_uuid"].nunique()
            rows.append({"split": split, "log": f.stem, "cone_rows": cones, "barrel_rows": barrels,
                         "hazard_tracks": int(tracks), "frames": int(df["timestamp_ns"].nunique())})
    rows.sort(key=lambda r: (r["hazard_tracks"], r["cone_rows"] + r["barrel_rows"]), reverse=True)
    (OUT / "log_ranking.json").write_text(json.dumps(rows, indent=1))
    with_hazards = [r for r in rows if r["hazard_tracks"] > 0]
    print(f"{len(with_hazards)}/{len(rows)} logs contain cones or barrels; top 10:")
    for r in rows[:10]:
        print("  ", r)
    return rows


def stage_images(n_logs: int, every: int, workers: int, min_tracks: int) -> None:
    rows = json.loads((OUT / "log_ranking.json").read_text())
    chosen = [r for r in rows if r["hazard_tracks"] >= min_tracks][:n_logs]
    print(f"downloading {len(chosen)} logs, every {every}th front frame")
    for i, r in enumerate(chosen):
        split, log = r["split"], r["log"]
        base = f"{PREFIX}/{split}/{log}"
        dest = OUT / "logs" / split / log
        keys = [k for k, _ in list_keys(f"{base}/calibration/")] + [f"{base}/city_SE3_egovehicle.feather"]
        frames = sorted(k for k, _ in list_keys(f"{base}/sensors/cameras/{CAMERA}/"))
        keys += frames[::every]
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(lambda k: fetch(k, dest / k[len(base) + 1:]), keys))
        # the annotations fetched in stage 1
        ann = OUT / "annotations" / split / f"{log}.feather"
        (dest / "annotations.feather").write_bytes(ann.read_bytes())
        print(f"  [{i + 1}/{len(chosen)}] {log}: {len(frames[::every])} frames, {r['hazard_tracks']} tracks")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["annotations", "rank", "images"], required=True)
    ap.add_argument("--logs", type=int, default=60)
    ap.add_argument("--every", type=int, default=5, help="keep every Nth front frame (20 Hz source)")
    ap.add_argument("--min-tracks", type=int, default=3)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "annotations":
        stage_annotations(args.workers)
    elif args.stage == "rank":
        rank_logs()
    else:
        stage_images(args.logs, args.every, args.workers, args.min_tracks)


if __name__ == "__main__":
    main()
