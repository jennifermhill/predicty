"""
Short script aiming to simplify multi-model evaluation processes.

For a given set of models, visualizes latent_space and runs linear probes
checking dataset and behavior classification capabilities. The dataset probe
should fail (model has learned dataset-invariant characteristics) and the 
behavior probe should succeed (model embeddings translate well to downstream
classification tasks).

Can check if evaluation scripts have previously been run for individual models 
to prevent redundant evaluation.

Usage:
    python eval/evaluation_pipeline.py --config checkpoints/<run_name>/config.yaml --run_names 0626-NoClassBalancing 06261422

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import subprocess

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to the training yaml config")
    p.add_argument("--check_evaluated", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run_names", required=True, nargs="+")

    return p.parse_args()

def main():
    args = parse_args()
    runs = args.run_names

    for run in runs:
        matches = [p for p in Path("latent_space").glob(f"{run}*") if p.is_dir()]
        if not matches or not args.check_evaluated:
            result = subprocess.run(["python", "eval/latent_space.py", 
                "--config", args.config, 
                "--checkpoint", str(Path("checkpoints") / run / "latest.pt")], 
                check=True)
            
            print(f"[pipeline] Latent space processed for run {run}")

        else:
            print(f"[pipeline] Skipping latent space processing for run {run}")

        run_dirs = [p for p in Path("latent_space").glob(f"{run}*") if p.is_dir()]
        run_dir = max(run_dirs, key=lambda p: p.name) if run_dirs else None

        matches = list(run_dir.glob("dataset_probe*")) if run_dir else []
        if not matches or not args.check_evaluated:
            result = subprocess.run(["python", "eval/probe_dataset.py", 
                "--config", args.config, 
                "--checkpoint", str(Path("checkpoints") / run / "latest.pt")], 
                check=True)
            
            print(f"[pipeline] Dataset probe completed for run {run}")
        else:
            print(f"[pipeline] Skipped dataset probe for run {run}")

        matches = list(run_dir.glob("behavior_probe*")) if run_dir else []
        if not matches or not args.check_evaluated:
            result = subprocess.run(["python", "eval/probe_behavior.py", 
                "--config", args.config, 
                "--checkpoint", str(Path("checkpoints") / run / "latest.pt")], 
                check=True)
            
            print(f"[pipeline] Behavior probe completed for run {run}")
        else:
            print(f"[pipeline] Skipped behavior probe for run {run}")



if __name__ == "__main__":
    main()
