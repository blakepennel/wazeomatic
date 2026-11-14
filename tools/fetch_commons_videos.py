"""Download short, openly licensed videos of emergency vehicles from Wikimedia Commons as test clips.

    python tools/fetch_commons_videos.py                 # into <datasets>/commons_videos/<category>/
    python tools/fetch_commons_videos.py --max-seconds 60 --list

They are the event-level test for the police classifier and the flashing-light detector
(tools/eval_police.py): every clip shows police, an ambulance or a fire engine, most with lights on. Many are
European, so the livery differs from the US cars the classifier was trained on; the light detector should not
care. Licence and author of each file go to ``meta.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wazeomatic.paths import datasets_root  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_commons_vehicles import FREE, api, fetch_one  # noqa: E402

CATEGORIES = ["Videos of emergency vehicle lighting", "Videos of police vehicles", "Videos of ambulances",
              "Videos of emergency service vehicles", "Videos of rescue vehicles"]


def list_videos(cat: str) -> list[dict]:
    out, cont = [], {}
    while True:
        d = api({"action": "query", "generator": "categorymembers", "gcmtitle": f"Category:{cat}", "gcmtype": "file",
                 "gcmlimit": "200", "prop": "imageinfo", "iiprop": "url|size|mediatype|extmetadata",
                 "iiextmetadatafilter": "LicenseShortName|Artist|LicenseUrl", **cont})
        for p in (d.get("query") or {}).get("pages", []):
            ii = (p.get("imageinfo") or [{}])[0]
            if ii.get("mediatype") != "VIDEO":
                continue
            em = ii.get("extmetadata") or {}
            out.append({"title": p["title"], "thumb": ii.get("url", ""), "page": ii.get("descriptionurl", ""),
                        "duration": float(ii.get("duration") or 0), "width": ii.get("width", 0),
                        "height": ii.get("height", 0), "size": ii.get("size", 0),
                        "licence": (em.get("LicenseShortName") or {}).get("value", ""),
                        "author": re.sub(r"<[^>]+>", "", (em.get("Artist") or {}).get("value", "")).strip()[:200]})
        if "continue" not in d:
            return out
        cont = {"gcmcontinue": d["continue"]["gcmcontinue"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="default <datasets>/commons_videos")
    ap.add_argument("--max-seconds", type=float, default=90.0)
    ap.add_argument("--min-height", type=int, default=480)
    ap.add_argument("--max-mb", type=float, default=150.0)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    out = Path(args.out) if args.out else datasets_root() / "commons_videos"
    seen = set()
    for cat in CATEGORIES:
        vids = [v for v in list_videos(cat) if v["title"] not in seen and 0 < v["duration"] <= args.max_seconds
                and v["height"] >= args.min_height and v["size"] <= args.max_mb * 1e6 and FREE.search(v["licence"])]
        seen.update(v["title"] for v in vids)
        print(f"[{cat}] {len(vids)} clips, {sum(v['duration'] for v in vids):.0f} s")
        if args.list:
            continue
        folder = out / re.sub(r"\W+", "_", cat.lower()).strip("_")
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "meta.jsonl", "w", encoding="utf-8") as f:
            for v in vids:
                name = fetch_one(v, folder)
                # fetch_one names files .jpg; keep the real container extension
                if name:
                    src = folder / name
                    ext = Path(v["title"]).suffix.lower() or ".webm"   # the URL carries a query string
                    dst = src.with_suffix(ext)
                    if src.exists():
                        src.replace(dst)
                    f.write(json.dumps({"file": dst.name, **{k: v[k] for k in ("title", "page", "licence", "author",
                                                                                "duration", "width", "height")}}) + "\n")
        print(f"[{cat}] saved to {folder}")


if __name__ == "__main__":
    main()
