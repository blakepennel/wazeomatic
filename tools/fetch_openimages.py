"""Download the images of a few Open Images V7 classes with their boxes (CC BY 4.0 annotations).

    python tools/fetch_openimages.py                                  # Ambulance and Deer, all splits
    python tools/fetch_openimages.py --classes Ambulance --max 400

Needs the box CSVs in ``<datasets>/openimages`` (``oidv6-train-annotations-bbox.csv``,
``validation-annotations-bbox.csv``, ``test-annotations-bbox.csv`` from storage.googleapis.com/openimages);
``--fetch-csv`` downloads them (2.3 GB). Images come from the public ``open-images-dataset`` S3 bucket. The
output is ``<datasets>/openimages/<Class>/<split>/<id>.jpg`` plus ``boxes.jsonl`` (normalised x1, y1, x2, y2,
and the group-of / truncated / occluded flags). Ambulances feed the vehicle classifier, deer the animal check.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wazeomatic.paths import datasets_root  # noqa: E402

CLASS_IDS = {"Ambulance": "/m/012n7d", "Deer": "/m/09kx5", "Traffic light": "/m/015qff",
             "Fire hydrant": "/m/01pns0", "Car": "/m/0k4j", "Taxi": "/m/0pg52", "Van": "/m/0h2r6"}
CSVS = {"train": ("oidv6-train-annotations-bbox.csv", "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"),
        "validation": ("validation-annotations-bbox.csv", "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"),
        "test": ("test-annotations-bbox.csv", "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv")}
S3 = "https://open-images-dataset.s3.amazonaws.com/{split}/{id}.jpg"


def scan(csv_path: Path, wanted: dict[str, str]) -> dict[str, dict[str, list]]:
    """class name -> image id -> [box dicts]"""
    by_mid = {mid: name for name, mid in wanted.items()}
    out: dict[str, dict[str, list]] = {n: defaultdict(list) for n in wanted}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = by_mid.get(row["LabelName"])
            if name is None:
                continue
            out[name][row["ImageID"]].append({
                "box": [float(row["XMin"]), float(row["YMin"]), float(row["XMax"]), float(row["YMax"])],
                "group": row.get("IsGroupOf") == "1", "truncated": row.get("IsTruncated") == "1",
                "occluded": row.get("IsOccluded") == "1", "depiction": row.get("IsDepiction") == "1"})
    return out


def fetch(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "wazeomatic-fetch"}),
                                    timeout=60) as r:
            dest.write_bytes(r.read())
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  skip {url}: {e}", file=sys.stderr)
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes", nargs="+", default=["Ambulance", "Deer"], choices=list(CLASS_IDS))
    ap.add_argument("--splits", nargs="+", default=list(CSVS), choices=list(CSVS))
    ap.add_argument("--max", type=int, default=5000, help="images per class and split at most")
    ap.add_argument("--root", help="default <datasets>/openimages")
    ap.add_argument("--fetch-csv", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    root = Path(args.root) if args.root else datasets_root() / "openimages"
    root.mkdir(parents=True, exist_ok=True)
    wanted = {c: CLASS_IDS[c] for c in args.classes}
    for split in args.splits:
        name, url = CSVS[split]
        path = root / name
        if not path.exists():
            if not args.fetch_csv:
                raise SystemExit(f"{path} missing; pass --fetch-csv or download {url}")
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, path)
        found = scan(path, wanted)
        for cls, imgs in found.items():
            folder = root / cls / split
            folder.mkdir(parents=True, exist_ok=True)
            ids = sorted(imgs)[: args.max]
            print(f"[{cls}/{split}] {len(ids)} images, {sum(len(imgs[i]) for i in ids)} boxes")
            s3split = "train" if split == "train" else split
            with ThreadPoolExecutor(args.workers) as ex:
                ok = list(ex.map(lambda i: fetch(S3.format(split=s3split, id=i), folder / f"{i}.jpg"), ids))
            with open(folder / "boxes.jsonl", "w", encoding="utf-8") as f:
                for i, good in zip(ids, ok):
                    if good:
                        f.write(json.dumps({"image": f"{i}.jpg", "boxes": imgs[i]}) + "\n")
            print(f"[{cls}/{split}] {sum(ok)} downloaded to {folder}")


if __name__ == "__main__":
    main()
