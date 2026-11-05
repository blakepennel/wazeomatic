"""Merge the public source datasets into one YOLO-format hazard dataset.

Sources (download locations are the ones used on the team machine; override with flags):

  pothole_ivcnz   github.com/jaygala24/pothole-detection release (MIT)      YOLO txt, class 0 = pothole
  rdd2022_us      RDD2022 United States subset (CC BY 4.0)                 Pascal VOC; D40 = pothole,
                  D00/D10/D20 = cracks -> not reportable, image kept as a hard negative
  cone_voc        Junhan0518/Traffic_Cone cone_dataset.tar.gz (Apache-2.0) Pascal VOC, "trafficcone"
  starline        StarLineResearch/Roadwork_Cones_Dataset                  parquet; cone / roadworks /
                  vertical_pannel from vehicle-mounted cameras (2880x1860)

Output: <out>/images/{train,val}, <out>/labels/{train,val}, <out>/data.yaml using the class
order from configs/classes.yaml. Large images are downscaled to --max-side to speed training.

    python tools/prepare_datasets.py                 # build everything
    python tools/prepare_datasets.py --check starline  # render a few labelled samples to runs/dataset_check
"""
from __future__ import annotations

import argparse
import random
from concurrent.futures import ThreadPoolExecutor
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATASETS = Path("F:/dev/datasets")

SRC = {
    "pothole_ivcnz": DATASETS / "pothole_ivcnz" / "Pothole Dataset",
    "rdd2022_us": DATASETS / "rdd2022_us" / "United_States" / "train",
    "cone_voc": DATASETS / "cones" / "cone_dataset" / "cone_dataset",
    "starline": DATASETS / "starline_roadwork" / "roadwork_cones_dataset",
}

# source label -> our class name (None = keep image, drop the box: hard negative)
LABEL_MAP = {
    "pothole_ivcnz": {"0": "pothole"},
    "rdd2022_us": {"D40": "pothole", "D00": None, "D10": None, "D20": None, "D43": None, "D44": None, "D50": None},
    "cone_voc": {"trafficcone": "cone", "person": None},
    "starline": {"cone": "cone", "roadworks": "construction_sign", "vertical_pannel": "delineator"},
}

IMG_EXT = {".jpg", ".jpeg", ".png", ".JPG", ".PNG"}


def load_classes() -> list[str]:
    cfg = yaml.safe_load((ROOT / "configs" / "classes.yaml").read_text(encoding="utf-8"))
    names = cfg["names"]
    return [names[i] for i in sorted(names)]


# ----------------------------------------------------------------------------- readers
# Each reader yields (image_path, [(class_name_or_None, x1, y1, x2, y2 in pixels)], split_hint)

def read_pothole_ivcnz(root: Path):
    for img in sorted(p for p in root.iterdir() if p.suffix in IMG_EXT):
        txt = img.with_suffix(".txt")
        boxes = []
        if txt.exists():
            h, w = cv2.imread(str(img)).shape[:2]
            for line in txt.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                c, cx, cy, bw, bh = parts
                cx, cy, bw, bh = (float(v) for v in (cx, cy, bw, bh))
                boxes.append((LABEL_MAP["pothole_ivcnz"].get(c), (cx - bw / 2) * w, (cy - bh / 2) * h,
                              (cx + bw / 2) * w, (cy + bh / 2) * h))
        yield img, boxes, None


def _read_voc(img_dir: Path, xml_dir: Path, label_map: dict):
    for xml in sorted(xml_dir.glob("*.xml")):
        tree = ET.parse(xml)
        fname = tree.findtext("filename") or xml.stem
        img = img_dir / fname
        if not img.exists():
            cands = [p for p in img_dir.glob(xml.stem + ".*") if p.suffix in IMG_EXT]
            if not cands:
                continue
            img = cands[0]
        boxes = []
        for obj in tree.findall("object"):
            name = obj.findtext("name")
            bb = obj.find("bndbox")
            if bb is None:
                continue
            x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            boxes.append((label_map.get(name), x1, y1, x2, y2))
        yield img, boxes, None


def read_rdd2022_us(root: Path):
    yield from _read_voc(root / "images", root / "annotations" / "xmls", LABEL_MAP["rdd2022_us"])


def read_cone_voc(root: Path):
    yield from _read_voc(root, root / "annotations", LABEL_MAP["cone_voc"])


def read_starline(root: Path):
    import pyarrow.parquet as pq

    table = pq.read_table(root / "annotations.parquet", columns=["label", "bbox_coords", "image_path"])
    by_img: dict[str, list] = defaultdict(list)
    for label, bbox, path in zip(table["label"].to_pylist(), table["bbox_coords"].to_pylist(),
                                 table["image_path"].to_pylist()):
        x, y, w, h = bbox  # top-left x, y, width, height in pixels (verified with --check)
        by_img[path].append((LABEL_MAP["starline"].get(label), x, y, x + w, y + h))
    for rel, boxes in sorted(by_img.items()):
        img = root / rel
        if img.exists():
            yield img, boxes, ("val" if rel.startswith("test/") else "train")


READERS = {
    "pothole_ivcnz": read_pothole_ivcnz,
    "rdd2022_us": read_rdd2022_us,
    "cone_voc": read_cone_voc,
    "starline": read_starline,
}


# ----------------------------------------------------------------------------- writer

def write_sample(img_path: Path, boxes, out_img: Path, out_lbl: Path, classes: list[str], max_side: int) -> Counter:
    im = cv2.imread(str(img_path))
    if im is None:
        return Counter()
    h, w = im.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        im = cv2.resize(im, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        h, w = im.shape[:2]
    lines, counts = [], Counter()
    for cls, x1, y1, x2, y2 in boxes:
        if cls is None:
            continue
        x1, y1, x2, y2 = (max(0.0, min(v * scale, lim)) for v, lim in ((x1, w), (y1, h), (x2, w), (y2, h)))
        bw, bh = x2 - x1, y2 - y1
        if bw < 2 or bh < 2:
            continue
        cid = classes.index(cls)
        lines.append(f"{cid} {(x1 + bw / 2) / w:.6f} {(y1 + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
        counts[cls] += 1
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_lbl.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
    out_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    if not lines:
        counts["(negative)"] += 1
    return counts


def check(source: str, n: int = 6) -> None:
    out = ROOT / "runs" / "dataset_check"
    out.mkdir(parents=True, exist_ok=True)
    k = 0
    for img, boxes, _ in READERS[source](SRC[source]):
        if not any(b[0] for b in boxes):
            continue
        im = cv2.imread(str(img))
        for cls, x1, y1, x2, y2 in boxes:
            if cls is None:
                continue
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
            cv2.putText(im, cls, (int(x1), max(20, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        h, w = im.shape[:2]
        if w > 1280:
            im = cv2.resize(im, (1280, int(h * 1280 / w)))
        cv2.imwrite(str(out / f"{source}_{k}.jpg"), im)
        k += 1
        if k >= n:
            break
    print(f"wrote {k} samples to {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATASETS / "wazeomatic_hazards"))
    ap.add_argument("--sources", nargs="+", default=list(READERS), choices=list(READERS))
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--max-negatives", type=int, default=1500, help="cap on crack-only RDD negatives")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", choices=list(READERS), help="render labelled samples and exit")
    ap.add_argument("--clean", action="store_true", help="delete the output folder first")
    ap.add_argument("--workers", type=int, default=8, help="parallel image conversions")
    args = ap.parse_args()

    if args.check:
        check(args.check)
        return

    classes = load_classes()
    out = Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    rng = random.Random(args.seed)
    totals: dict[str, Counter] = defaultdict(Counter)
    per_split = Counter()
    for source in args.sources:
        root = SRC[source]
        if not root.exists():
            print(f"[skip] {source}: {root} not found")
            continue
        negatives = 0
        jobs = []
        for img, boxes, split_hint in READERS[source](root):
            positive = any(b[0] for b in boxes)
            if not positive:
                if negatives >= args.max_negatives:
                    continue
                negatives += 1
            split = split_hint or ("val" if rng.random() < args.val_frac else "train")
            stem = f"{source}_{img.stem}"
            jobs.append((img, boxes, out / "images" / split / f"{stem}.jpg",
                         out / "labels" / split / f"{stem}.txt", split))
        for split in ("train", "val"):
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        n = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(write_sample, img, boxes, oi, ol, classes, args.max_side) for img, boxes, oi, ol, _ in jobs]
            for (_, _, _, _, split), fut in zip(jobs, futures):
                totals[source].update(fut.result())
                per_split[split] += 1
                n += 1
                if n % 500 == 0:
                    print(f"  {source}: {n}/{len(jobs)} images...", flush=True)
        print(f"[done] {source}: {n} images, boxes {dict(totals[source])}", flush=True)

    data = {
        "path": str(out).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {i: c for i, c in enumerate(classes)},
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    grand = Counter()
    for c in totals.values():
        grand.update(c)
    print("images per split:", dict(per_split))
    print("boxes per class:", dict(grand))
    print("wrote", out / "data.yaml")


if __name__ == "__main__":
    main()
