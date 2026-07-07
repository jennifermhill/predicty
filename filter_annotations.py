#!/usr/bin/env python3
"""Filter annotation file to only include zarr_paths from train.yaml"""

import pandas as pd
from pathlib import Path
from configs.config import Config

# Load config to get actual zarr_paths
cfg = Config.from_yaml('configs/train.yaml')
train_paths = set(cfg.zarr_paths)

print(f"Train.yaml has {len(train_paths)} datasets:")
for p in sorted(train_paths):
    print(f"  - {Path(p).name}")

# Load annotations
df = pd.read_json('index.jsonl', lines=True)
print(f"\nAnnotation file has {len(df)} total entries")

# Filter to matching paths
df_filtered = df[df['zarr_path'].isin(train_paths)]
print(f"After filtering: {len(df_filtered)} entries")

# Show class distribution
print(f"\nBehavior classes in filtered annotations:")
for label, count in df_filtered['label'].value_counts().items():
    print(f"  {label}: {count}")

print(f"\nDatasets with annotations:")
for path in sorted(df_filtered['zarr_path'].unique()):
    count = len(df_filtered[df_filtered['zarr_path'] == path])
    print(f"  {Path(path).name}: {count}")

# Save filtered version
output_path = 'index_filtered.jsonl'
df_filtered.to_json(output_path, orient='records', lines=True)
print(f"\nSaved filtered annotations to {output_path}")
