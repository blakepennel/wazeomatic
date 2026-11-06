"""Fetch the pothole images of RDD2022 (all six countries) from the Hugging Face mirror.

The mirror dronefreak/RDD2022 holds the 38,385 publicly labelled RDD2022 images as YOLO
files (classes 0 longitudinal_crack, 1 transverse_crack, 2 alligator_crack, 3 pothole;
CC BY-SA 4.0, cite arXiv:2209.08538). We only need the images that contain a pothole, about
a fifth of the set, so this script downloads the label files first, picks the images with a
class-3 box and fetches just those.

    python tools/fetch_rdd2022_potholes.py                 # -> F:/dev/datasets/rdd2022_full
    python tools/fetch_rdd2022_potholes.py --max-images 3000 --workers 8

Afterwards tools/prepare_datasets.py --sources ... rdd2022_full merges them (class 3 -> pothole,
cracks dropped from the boxes).
"""
from __future__ import annotations

import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = "dronefreak/RDD2022"
DEFAULT_OUT = Path("F:/dev/datasets/rdd2022_full")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--max-images", type=int, default=0, help="cap on pothole images (0 = all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download, snapshot_download

    out = Path(args.out)
    snapshot_download(REPO, repo_type="dataset", local_dir=str(out),
                      allow_patterns=["data/labels/**", "data/data.yaml", "README.md"], max_workers=args.workers)

    wanted: list[Path] = []
    total = 0
    for lbl in sorted((out / "data" / "labels").rglob("*.txt")):
        total += 1
        for line in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.split()[:1] == ["3"]:
                wanted.append(lbl)
                break
    print(f"{total} label files, {len(wanted)} with a pothole")
    if args.max_images and len(wanted) > args.max_images:
        random.Random(args.seed).shuffle(wanted)
        wanted = wanted[: args.max_images]

    def fetch(lbl: Path) -> str | None:
        rel = lbl.relative_to(out).with_suffix(".jpg")
        rel = Path(str(rel).replace("labels", "images", 1))
        target = out / rel
        if target.exists():
            return None
        try:
            hf_hub_download(REPO, repo_type="dataset", filename=rel.as_posix(), local_dir=str(out))
            return None
        except Exception as e:  # keep going, report at the end
            return f"{rel}: {e}"

    errors = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(fetch, l) for l in wanted]):
            done += 1
            err = fut.result()
            if err:
                errors.append(err)
            if done % 500 == 0:
                print(f"  {done}/{len(wanted)} images...", flush=True)
    (out / "pothole_images.txt").write_text(
        "\n".join(str(l.relative_to(out).with_suffix(".jpg")).replace("labels", "images", 1) for l in wanted) + "\n",
        encoding="utf-8")
    print(f"done: {len(wanted) - len(errors)} images in {out / 'data' / 'images'}, {len(errors)} errors")
    for e in errors[:10]:
        print("  ", e)


if __name__ == "__main__":
    main()
