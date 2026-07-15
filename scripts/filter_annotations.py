#!/usr/bin/env python3
"""filter index.jsonl down to the zarr_paths configured in configs/train.yaml.

writes index_filtered.jsonl at the repo root and prints the resulting behavior
class and per-dataset counts, so the probe scripts only see annotations for
datasets the current config actually trains on.

run from the repo root:
    python scripts/filter_annotations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# allow running as `python scripts/filter_annotations.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from configs.config import Config

# resolve paths relative to the repo root, not the caller's cwd.
repo_root = Path(__file__).resolve().parent.parent
index_path = repo_root / "index.jsonl"
output_path = repo_root / "index_filtered.jsonl"

# datasets the current config trains on; only these annotations are kept.
cfg = Config.from_yaml(str(repo_root / "configs" / "train.yaml"))
train_paths = set(cfg.zarr_paths)

print(f"train.yaml has {len(train_paths)} datasets:")
for path in sorted(train_paths):
    print(f"  - {Path(path).name}")

df = pd.read_json(index_path, lines=True)
print(f"\nannotation file has {len(df)} total entries")

df_filtered = df[df["zarr_path"].isin(train_paths)]
print(f"after filtering: {len(df_filtered)} entries")

print("\nbehavior classes in filtered annotations:")
for label, count in df_filtered["label"].value_counts().items():
    print(f"  {label}: {count}")

print("\ndatasets with annotations:")
for path in sorted(df_filtered["zarr_path"].unique()):
    count = len(df_filtered[df_filtered["zarr_path"] == path])
    print(f"  {Path(path).name}: {count}")

df_filtered.to_json(output_path, orient="records", lines=True)
print(f"\nsaved filtered annotations to {output_path}")
