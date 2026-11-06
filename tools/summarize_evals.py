"""Collect several ``tools/eval_events.py`` runs into one comparison table.

    python tools/summarize_evals.py runs/eval/ens_*/ --out runs/eval/ens_summary.md

Each row: detector stack, box AP50 per class, event precision/recall/F1 overall and per hazard
type, false events on the hazard-free clips (if that section ran) and the mean inference time.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def load(d: Path) -> dict | None:
    f = d / "results.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text(encoding="utf-8"))
    r["_name"] = d.name
    return r


def row(r: dict, classes: list[str], types: list[str]) -> list:
    boxes = r.get("boxes", {})
    per_class = boxes.get("per_class", boxes)  # tolerate both layouts
    out = [r["_name"], r.get("stack", Path(r["model"]).name)]
    for c in classes:
        ap = None
        entry = per_class.get(c) if isinstance(per_class, dict) else None
        if isinstance(entry, dict):
            ap = entry.get("all", entry).get("ap50") if isinstance(entry.get("all", entry), dict) else None
        out.append("-" if ap is None else f"{ap:.3f}")
    ev = r.get("events", {}).get("overall", {})
    ov = ev.get("overall", {})
    out += [f"{ov.get('precision', 0):.3f}", f"{ov.get('recall', 0):.3f}", f"{ov.get('f1', 0):.3f}"]
    for t in types:
        pt = ev.get("per_type", {}).get(t, {})
        out.append(f"{pt.get('recall', 0):.2f}/{pt.get('fp', 0)}")
    vids = r.get("videos")
    out.append("-" if not vids else str(sum(v.get("events", 0) for v in vids)))
    out.append(str(r.get("infer_ms_mean", "-")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out")
    args = ap.parse_args()
    dirs = [Path(p) for pat in args.dirs for p in sorted(glob.glob(pat))]
    runs = [r for r in (load(d) for d in dirs if d.is_dir()) if r]
    if not runs:
        raise SystemExit("no results.json found")
    classes = runs[0].get("labels", [])
    types = runs[0].get("hazard_types", [])
    cols = ["run", "stack"] + [f"AP50 {c}" for c in classes] + ["P", "R", "F1"] + \
           [f"{t.replace('HAZARD_ON_ROAD_', '').lower()} R/FP" for t in types] + ["clip events", "ms"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in runs:
        lines.append("| " + " | ".join(str(v) for v in row(r, classes, types)) + " |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
