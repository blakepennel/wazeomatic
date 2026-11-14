"""Download openly licensed photos of emergency vehicles from Wikimedia Commons.

    python tools/fetch_commons_vehicles.py --list            # categories and how many files each class would get
    python tools/fetch_commons_vehicles.py                   # police, ambulance, fire into <datasets>/commons_vehicles
    python tools/fetch_commons_vehicles.py --only police --max 6000

These are the positives for the vehicle classifier (tools/build_vehicle_crops.py): the stock detector finds
cars, trucks and buses, and the classifier decides whether one is a police car, an ambulance or a fire engine.
Only files under CC0, public domain, CC BY or CC BY-SA are kept; author and licence of each one are written to
``<class>/meta.jsonl`` so the attribution travels with the data. Thumbnails are fetched at 640 px wide, which is
larger than any crop the classifier sees.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wazeomatic.paths import datasets_root  # noqa: E402

API = "https://commons.wikimedia.org/w/api.php"
UA = "wazeomatic-dataset-builder/0.2 (university course project; https://github.com/blakepennel/wazeomatic)"

# class -> (root categories, depth). Depth is how far into subcategories the walk goes.
CLASSES: dict[str, list[tuple[str, int]]] = {
    "police": [
        ("Police automobiles in the United States", 4),
        ("Police trucks of the United States", 3),
        ("Police vans of the United States", 2),
        ("Police vehicles of Boston", 2),
        ("Florida Highway Patrol vehicles", 2),
        ("Vehicles of the Indianapolis Metropolitan Police Department", 2),
        ("Vehicles of the Las Vegas Metropolitan Police Department", 2),
        ("Vehicles of the Los Angeles Police Department", 2),
        ("Vehicles of the Chicago Police Department", 2),
        ("Vehicles of the New York City Police Department", 2),
        ("San Francisco Police Department vehicles", 2),
        ("Vehicles of the Seattle Police Department", 2),
        ("Vehicles of the Memphis Police Department", 2),
        ("University and college police vehicles of the United States", 2),
        ("Police vehicles of the United States", 1),
        ("Police vehicles of Canada", 3),
    ],
    "ambulance": [
        ("Ambulances in the United States", 4),
        ("Ambulances in Canada", 3),
        # other countries' liveries: Battenberg and box-bodied ambulances were called police until these joined
        ("Ambulances in the United Kingdom", 3),
        ("Ambulances in Japan", 2),
        ("Ambulances in Germany", 2),
        ("Ambulances in France", 2),
        ("Ambulances in Australia", 2),
        ("Ambulances in the Netherlands", 2),
    ],
    "fire": [
        ("Fire engines of the United States", 4),
        ("Fire engines in Canada", 2),
    ],
    # look-alikes that are not police: the civilian versions of the usual patrol cars, taxis, tow trucks,
    # school buses. They join the civilian class as hard negatives.
    "civilian_hard": [
        ("Ford Explorer (sixth generation)", 1),
        ("Ford Explorer (fifth generation)", 1),
        ("Dodge Charger (LD)", 1),
        ("Chevrolet Tahoe (fifth generation)", 1),
        ("Chevrolet Tahoe (fourth generation)", 1),
        ("Ford Taurus (sixth generation)", 1),
        ("Chevrolet Impala (ninth generation)", 1),
        ("Ford F-150 (fourteenth generation)", 1),
        ("Taxis in the United States", 2),
        ("Tow trucks in the United States", 2),
        ("School buses in the United States", 1),
        ("Security vehicles", 1),
    ],
    # taxis and private-hire cars of other countries: a roof sign reads like a light bar, and they were the
    # two false Police reports on the Commons clips (docs/waze-coverage.md)
    "civilian_taxi": [
        ("Taxis in Japan", 2),
        ("Taxis in the United Kingdom", 2),
        ("Taxis in Canada", 2),
        ("Taxis in Germany", 1),
        ("Taxis in France", 1),
        ("Taxis in Australia", 1),
        ("Taxis in New York City", 1),
    ],
}
# extra subcategory filter per class: the look-alike models have police subcategories of their own
_NOT_CIVILIAN = re.compile(r"police|sheriff|patrol|trooper|law enforcement|interceptor|emergency|ambulance|fire",
                           re.I)
# a taxi walk also reaches water taxis, rickshaws, protests, ranks and views out of a taxi (which may show a
# police car ahead): none of them is a taxi on the road
_NOT_A_TAXI = re.compile(_NOT_CIVILIAN.pattern + r"|water|ferry|rickshaw|cycle|protest|views from|drivers|memorial|"
                         r"road signs|offices|ranks|stands|race|zusatzzeichen|electrobat", re.I)
CLASS_SKIP = {"civilian_hard": _NOT_CIVILIAN, "civilian_taxi": _NOT_A_TAXI}

# Subcategories whose name matches this are not road vehicles a dashcam meets, or not real ones.
SKIP_CAT = re.compile(
    r"aircraft|helicopter|airplane|\bboats?\b|\bships?\b|bicycle|\bbikes?\b|horse|mounted|armou?red|\btanks?\b|"
    r"museum|\btoys?\b|\bmodels?\b|scale model|lego|interior|drawing|\bart\b|painting|logo|insignia|badge|"
    r"uniform|military|\barmy\b|\bnavy\b|air force|historic|antique|vintage|19[0-6]\d|parade|segway|golf|"
    r"snow ?mobile|\batvs?\b|\btrains?\b|\brail|motorcycle|scooter",
    re.I,
)
FREE = re.compile(r"^(cc0|public domain|pd\b|pd-|cc[ -]by(-sa)?[ -]?\d|cc by(-sa)? \d|cc-by(-sa)?-\d)", re.I)


_last_call = [0.0]
API_INTERVAL_S = 1.0          # one API request a second; Commons answers 429 to faster anonymous clients


def _polite_wait(err: Exception, k: int) -> float:
    retry_after = getattr(err, "headers", None) and err.headers.get("Retry-After")
    if retry_after and str(retry_after).isdigit():
        return float(retry_after) + 1
    return (30.0 if "429" in str(err) else 2.0) * (2 ** k)


def api(params: dict, tries: int = 7) -> dict:
    params = {**params, "format": "json", "formatversion": "2", "maxlag": "5"}
    url = API + "?" + urllib.parse.urlencode(params)
    for k in range(tries):
        time.sleep(max(0.0, _last_call[0] + API_INTERVAL_S - time.time()))
        _last_call[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read().decode("utf-8"))
            if "error" in d and d["error"].get("code") == "maxlag":
                raise RuntimeError("maxlag")
            return d
        except Exception as e:  # noqa: BLE001 - network, JSON or lag: back off and retry
            wait = min(_polite_wait(e, k), 900.0)
            print(f"  api retry in {wait:.0f}s: {e}", file=sys.stderr)
            time.sleep(wait)
    raise SystemExit(f"Commons API kept failing for {params}")


def members(cat: str, kind: str) -> list[str]:
    out, cont = [], {}
    while True:
        d = api({"action": "query", "list": "categorymembers", "cmtitle": f"Category:{cat}",
                 "cmtype": kind, "cmlimit": "500", **cont})
        out += [m["title"] for m in d["query"]["categorymembers"]]
        if "continue" not in d:
            return out
        cont = {"cmcontinue": d["continue"]["cmcontinue"]}


def walk(roots: list[tuple[str, int]], extra_skip=None) -> tuple[list[str], list[str]]:
    seen_cats, files = set(), []
    queue = [(c, d) for c, d in roots]
    while queue:
        cat, depth = queue.pop(0)
        if cat in seen_cats:
            continue
        seen_cats.add(cat)
        files += [f for f in members(cat, "file") if re.search(r"\.(jpe?g|png)$", f, re.I)]
        if depth > 0:
            for sub in members(cat, "subcat"):
                name = sub.split(":", 1)[1]
                if not SKIP_CAT.search(name) and name not in seen_cats and \
                        not (extra_skip is not None and extra_skip.search(name)):
                    queue.append((name, depth - 1))
    return sorted(set(files)), sorted(seen_cats)


def image_info(titles: list[str], width: int) -> list[dict]:
    out = []
    for i in range(0, len(titles), 50):
        d = api({"action": "query", "titles": "|".join(titles[i:i + 50]), "prop": "imageinfo",
                 "iiprop": "url|extmetadata|size", "iiurlwidth": str(width),
                 "iiextmetadatafilter": "LicenseShortName|Artist|LicenseUrl"})
        for p in d["query"]["pages"]:
            ii = (p.get("imageinfo") or [{}])[0]
            em = ii.get("extmetadata") or {}
            lic = (em.get("LicenseShortName") or {}).get("value", "")
            artist = re.sub(r"<[^>]+>", "", (em.get("Artist") or {}).get("value", "")).strip()
            if ii.get("thumburl"):
                out.append({"title": p["title"], "thumb": ii["thumburl"], "page": ii.get("descriptionurl", ""),
                            "licence": lic, "licence_url": (em.get("LicenseUrl") or {}).get("value", ""),
                            "author": artist[:200], "free": bool(FREE.search(lic))})
    return out


def fetch_one(item: dict, folder: Path) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", item["title"].split(":", 1)[1])[:150]
    name = Path(name).stem + ".jpg"
    dest = folder / name
    if dest.exists() and dest.stat().st_size > 0:
        return name
    for k in range(6):
        try:
            req = urllib.request.Request(item["thumb"], headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            dest.write_bytes(data)
            time.sleep(0.25)      # a few files a second per worker
            return name
        except Exception as e:  # noqa: BLE001
            if k < 5:
                time.sleep(min(_polite_wait(e, k), 300.0))
            else:
                print(f"  skip {item['title']}: {e}", file=sys.stderr)
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="default <datasets>/commons_vehicles")
    ap.add_argument("--only", nargs="+", choices=list(CLASSES))
    ap.add_argument("--max", type=int, default=8000, help="files per class at most")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--list", action="store_true", help="walk the categories and report counts only")
    args = ap.parse_args()
    out = Path(args.out) if args.out else datasets_root() / "commons_vehicles"
    for cls in args.only or list(CLASSES):
        listing = out / cls / "listing.json"       # the walk costs hundreds of requests; keep it
        if listing.exists() and not args.list:
            d = json.loads(listing.read_text(encoding="utf-8"))
            files, cats = d["files"], d["categories"]
        else:
            files, cats = walk(CLASSES[cls], CLASS_SKIP.get(cls))
            listing.parent.mkdir(parents=True, exist_ok=True)
            listing.write_text(json.dumps({"files": files, "categories": cats}), encoding="utf-8")
        print(f"[{cls}] {len(files)} files in {len(cats)} categories")
        if args.list:
            continue
        files = files[: args.max]
        folder = out / cls
        folder.mkdir(parents=True, exist_ok=True)
        infos = image_info(files, args.width)
        free = [i for i in infos if i["free"]]
        print(f"[{cls}] {len(free)} of {len(infos)} under a free licence; downloading")
        with ThreadPoolExecutor(args.workers) as ex:
            names = list(ex.map(lambda it: fetch_one(it, folder), free))
        with open(folder / "meta.jsonl", "w", encoding="utf-8") as f:
            for it, name in zip(free, names):
                if name:
                    f.write(json.dumps({"file": name, **{k: it[k] for k in ("title", "page", "licence",
                                                                             "licence_url", "author")}}) + "\n")
        (folder / "categories.txt").write_text("\n".join(cats) + "\n", encoding="utf-8")
        print(f"[{cls}] {sum(1 for n in names if n)} files in {folder}")


if __name__ == "__main__":
    main()
