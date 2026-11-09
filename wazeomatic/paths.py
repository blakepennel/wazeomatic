"""Where the large, uncommitted things live.

Datasets are several tens of gigabytes and are never committed. Every tool resolves the datasets
folder the same way, first match wins:

1. the ``WAZEOMATIC_DATASETS`` environment variable,
2. ``datasets:`` in ``configs/local.yaml`` (git-ignored, one line, for a machine-specific drive),
3. ``<repo>/datasets`` (git-ignored).

The merged training set built by ``tools/prepare_datasets.py`` is ``<datasets>/wazeomatic_hazards_v3``.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
MERGED_DATASET = "wazeomatic_hazards_v3"


def datasets_root() -> Path:
    env = os.environ.get("WAZEOMATIC_DATASETS")
    if env:
        return Path(env)
    local = ROOT / "configs" / "local.yaml"
    if local.exists():
        import yaml

        cfg = yaml.safe_load(local.read_text(encoding="utf-8")) or {}
        if cfg.get("datasets"):
            return Path(str(cfg["datasets"]))
    return ROOT / "datasets"


def merged_dataset() -> Path:
    """The merged YOLO-format training set (``data.yaml``, ``images/``, ``labels/``)."""
    return datasets_root() / MERGED_DATASET
