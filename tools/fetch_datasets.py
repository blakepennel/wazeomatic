"""Download the public source datasets into the layout tools/prepare_datasets.py reads.

    python tools/fetch_datasets.py --list                  # what there is, sizes, licences, what is on disk
    python tools/fetch_datasets.py                         # everything that is missing (about 17 GB)
    python tools/fetch_datasets.py --only cone_voc pothole_ivcnz

Two sources have their own scripts because they are filtered while downloading:

    python tools/fetch_rdd2022_potholes.py                 # pothole images of RDD2022, all countries
    python tools/fetch_argoverse_hazards.py --stage annotations
    python tools/fetch_argoverse_hazards.py --stage images --logs 100 --every 4
    python tools/argoverse_to_yolo.py

The target folder is the datasets root of wazeomatic/paths.py (WAZEOMATIC_DATASETS, configs/local.yaml
or <repo>/datasets). Archives are kept next to what they unpack to; a source whose marker path exists
is skipped. Read the licence column before using a dataset for anything but course work: StarLine
states no licence and Argoverse 2 is non-commercial (docs/model-research.md, Datasets).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wazeomatic.paths import datasets_root  # noqa: E402

HF = "https://huggingface.co/datasets/{repo}/resolve/main/{file}"


@dataclass
class Source:
    key: str                 # the source name prepare_datasets.py uses
    folder: str              # folder under the datasets root
    marker: str              # path (under folder) that exists once the source is unpacked
    files: list[tuple[str, str]]   # (url, extract_to relative to folder; "" = folder itself)
    size: str
    licence: str
    note: str = ""


SOURCES = [
    Source("pothole_ivcnz", "pothole_ivcnz", "Pothole Dataset",
           [("https://github.com/jaygala24/pothole-detection/releases/download/v1.0.0/Pothole.Dataset.IVCNZ.zip", "")],
           "174 MB", "MIT"),
    Source("rdd2022_us", "rdd2022_us", "United_States/train",
           [(HF.format(repo="cvtechniques/Road_Damage_Detection_USA", file="United_States.zip"), "")],
           "0.5 GB", "CC BY 4.0 upstream (mirror card states none)"),
    Source("cone_voc", "cones", "cone_dataset/cone_dataset",
           [(HF.format(repo="Junhan0518/Traffic_Cone", file="cone_dataset.tar.gz"), "cone_dataset")],
           "49 MB", "Apache-2.0"),
    Source("starline", "starline_roadwork", "roadwork_cones_dataset/annotations.parquet",
           [(HF.format(repo="StarLineResearch/Roadwork_Cones_Dataset", file="roadwork_cones_dataset.zip"), "")],
           "5.7 GB", "NOT STATED on the dataset card",
           "research / course use only until the authors confirm a licence"),
    Source("roadwork", "roadwork", "images",
           [(HF.format(repo="anuragxel/roadwork-dataset", file="annotations.zip"), ""),
            (HF.format(repo="anuragxel/roadwork-dataset", file="images.zip"), "")],
           "10.7 GB", "ODC-By 1.0", "images drawn from external sources keep their own terms"),
    Source("pothole_ryukijano", "pothole_ryukijano", "train/images", [], "30 MB", "CC BY 4.0",
           "small Roboflow export, fetched file by file with huggingface_hub"),
]
BY_KEY = {s.key: s for s in SOURCES}


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "wazeomatic-fetch"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, last = 0, -1
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            pct = int(100 * done / total) if total else -1
            if pct != last and (pct % 5 == 0 or not total):
                print(f"  {dest.name}: {done / 1e6:,.0f} MB" + (f" ({pct} %)" if total else ""), flush=True)
                last = pct
    tmp.replace(dest)


def unpack(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  unpacking {archive.name} -> {dest}")
    if archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive) as t:
            try:
                t.extractall(dest, filter="data")     # refuse paths that leave the folder (Python 3.12+)
            except TypeError:
                t.extractall(dest)
    else:
        with zipfile.ZipFile(archive) as z:
            root = dest.resolve()
            for name in z.namelist():
                if not (dest / name).resolve().is_relative_to(root):
                    raise SystemExit(f"{archive.name}: entry {name!r} would unpack outside {dest}")
            z.extractall(dest)


def fetch(src: Source, base: Path, keep_archives: bool) -> None:
    folder = base / src.folder
    if (folder / src.marker).exists():
        print(f"[have] {src.key}: {folder / src.marker}")
        return
    print(f"[fetch] {src.key} ({src.size}, licence: {src.licence})")
    if src.key == "pothole_ryukijano":
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise SystemExit("pip install huggingface_hub (it is in requirements-train.txt)") from None
        snapshot_download("Ryukijano/Pothole-detection-Yolov8", repo_type="dataset", local_dir=str(folder))
    for url, sub in src.files:
        archive = folder / url.rsplit("/", 1)[-1]
        download(url, archive)
        unpack(archive, folder / sub if sub else folder)
        if not keep_archives:
            archive.unlink()
    if not (folder / src.marker).exists():
        raise SystemExit(f"{src.key}: expected {folder / src.marker} after unpacking; the upstream layout changed")
    print(f"[done] {src.key}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", help="target folder (default: see wazeomatic/paths.py)")
    ap.add_argument("--only", nargs="+", choices=list(BY_KEY), help="fetch only these sources")
    ap.add_argument("--list", action="store_true", help="show the sources and what is already on disk")
    ap.add_argument("--discard-archives", action="store_true", help="delete each archive after unpacking")
    args = ap.parse_args()
    base = Path(args.datasets) if args.datasets else datasets_root()
    if args.list:
        print(f"datasets root: {base}")
        for s in SOURCES:
            state = "on disk" if (base / s.folder / s.marker).exists() else "missing"
            print(f"  {s.key:18} {s.size:>8}  {state:8} {s.licence}" + (f"  ({s.note})" if s.note else ""))
        for key, folder, marker, how in (("rdd2022_full", "rdd2022_full", "data", "tools/fetch_rdd2022_potholes.py"),
                                         ("argoverse", "argoverse2", "yolo", "tools/fetch_argoverse_hazards.py + argoverse_to_yolo.py")):
            state = "on disk" if (base / folder / marker).exists() else "missing"
            print(f"  {key:18} {'':>8}  {state:8} {how}")
        return
    free = shutil.disk_usage(base if base.exists() else base.parent).free / 1e9
    print(f"datasets root: {base} ({free:,.0f} GB free)")
    for key in (args.only or list(BY_KEY)):
        fetch(BY_KEY[key], base, keep_archives=not args.discard_archives)


if __name__ == "__main__":
    main()
