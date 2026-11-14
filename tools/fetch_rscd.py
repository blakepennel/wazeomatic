"""Download a balanced sample of RSCD (Road Surface Condition Dataset) from its Hugging Face mirror.

    python tools/fetch_rscd.py                       # up to 3000 images per folder into <datasets>/rscd/train/<folder>
    python tools/fetch_rscd.py --per-folder 1000

RSCD images are close views of the road surface ahead of a car, labelled by moisture and material
(dry / wet / water x asphalt / concrete / gravel / mud) plus fresh snow, melted snow and ice. The road-surface
classifier (tools/build_scene_sets.py) groups them into dry, wet, water, slush, snow and ice and runs on the
patch of road just above the hood. Mirror: rezzzq/RSCD-1million (MIT on the card; the original is from
Tsinghua University, cite Zhao et al. 2023 when you publish anything trained on it).
"""
from __future__ import annotations

import argparse
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wazeomatic.paths import datasets_root  # noqa: E402

REPO = "rezzzq/RSCD-1million"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-folder", type=int, default=3000)
    ap.add_argument("--out", help="default <datasets>/rscd")
    ap.add_argument("--workers", type=int, default=4, help="parallel downloads (the mirror rate limits anonymous clients)")
    ap.add_argument("--skip-damage", action="store_true", default=True,
                    help="only the smooth/slight asphalt and concrete folders (severity is not what we classify)")
    args = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download

    out = Path(args.out) if args.out else datasets_root() / "rscd"
    api = HfApi()
    folders = [e.path for e in api.list_repo_tree(REPO, repo_type="dataset", path_in_repo="train")]
    if args.skip_damage:
        folders = [f for f in folders if not f.endswith("_severe")]
    # the rare conditions first: if the mirror starts rate limiting, dry asphalt is what goes missing
    order = ("fresh_snow", "ice", "melted_snow", "water", "wet", "dry")
    folders.sort(key=lambda f: next((i for i, k in enumerate(order) if Path(f).name.startswith(k)), len(order)))
    rng = random.Random(0)
    for folder in folders:
        dest = out / folder
        have = len(list(dest.glob("*.jpg"))) if dest.exists() else 0
        if have >= args.per_folder:
            print(f"[have] {folder}: {have}")
            continue
        files = [e.path for e in api.list_repo_tree(REPO, repo_type="dataset", path_in_repo=folder)
                 if e.path.endswith(".jpg")]
        rng.shuffle(files)
        files = files[: args.per_folder]
        print(f"[fetch] {folder}: {len(files)} of the folder", flush=True)
        with ThreadPoolExecutor(args.workers) as ex:
            list(ex.map(lambda f: hf_hub_download(REPO, f, repo_type="dataset", local_dir=str(out)), files))
    print("RSCD sample done")


if __name__ == "__main__":
    main()
