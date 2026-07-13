"""
short script aiming to simplify multi-model evaluation processes.

for a given set of models, visualizes latent_space and runs linear probes
checking dataset and behavior classification capabilities. The dataset probe
should fail (model has learned dataset-invariant characteristics) and the
behavior probe should succeed (model embeddings translate well to downstream
classification tasks). Also runs Grad-CAM on both probes to visualize where
the encoder is looking.

can check if evaluation scripts have previously been run for individual models
to prevent redundant evaluation.

each run's config is read from its frozen snapshot at
checkpoints/<run_name>/config.yaml (the one launch.sh wrote at training time),
so this script only needs run names.

Usage:
    python eval/evaluation_pipeline.py --run_names 0626-NoClassBalancing 06261422

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import subprocess

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--check_evaluated", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run_names", required=True, nargs="+")

    return p.parse_args()

def main():
    args = parse_args()
    runs = args.run_names

    for run in runs:
        config_path = Path("checkpoints") / run / "config.yaml"
        if not config_path.exists():
            print(f"[pipeline] ERROR: {config_path} not found, skipping run {run}")
            continue
        checkpoint_path = Path("checkpoints") / run / "latest.pt"

        matches = [p for p in Path("latent_space").glob(f"{run}*") if p.is_dir()]
        if not matches or not args.check_evaluated:
            result = subprocess.run(["python", "eval/latent_space.py",
                "--config", str(config_path),
                "--checkpoint", str(checkpoint_path)],
                check=True)

            print(f"[pipeline] Latent space processed for run {run}")

        else:
            print(f"[pipeline] Skipping latent space processing for run {run}")

        run_dirs = [p for p in Path("latent_space").glob(f"{run}*") if p.is_dir()]
        run_dir = max(run_dirs, key=lambda p: p.name) if run_dirs else None

        matches = list(run_dir.glob("dataset_probe*")) if run_dir else []
        if not matches or not args.check_evaluated:
            result = subprocess.run(["python", "eval/probe_dataset.py",
                "--config", str(config_path),
                "--checkpoint", str(checkpoint_path)],
                check=True)

            print(f"[pipeline] Dataset probe completed for run {run}")
        else:
            print(f"[pipeline] Skipped dataset probe for run {run}")

        matches = list(run_dir.glob("behavior_probe*")) if run_dir else []
        if not matches or not args.check_evaluated:
            result = subprocess.run(["python", "eval/probe_behavior.py",
                "--config", str(config_path),
                "--checkpoint", str(checkpoint_path)],
                check=True)

            print(f"[pipeline] Behavior probe completed for run {run}")
        else:
            print(f"[pipeline] Skipped behavior probe for run {run}")

        # re-glob run_dir in case the probe steps above just created it
        # (e.g. a run whose latent_space dir didn't exist until now).
        run_dirs = [p for p in Path("latent_space").glob(f"{run}*") if p.is_dir()]
        run_dir = max(run_dirs, key=lambda p: p.name) if run_dirs else None

        gradcam_dir = (run_dir / "gradcam") if run_dir else None
        matches = list(gradcam_dir.glob("*")) if gradcam_dir and gradcam_dir.exists() else []
        if not matches or not args.check_evaluated:
            if run_dir is None:
                print(f"[pipeline] Skipping Grad-CAM for run {run}: no latent_space dir found")
                continue

            behavior_probe = run_dir / "behavior_probe.pt"
            dataset_probe = run_dir / "dataset_probe.pt"
            gradcam_cmd = ["python", "eval/run_gradcam.py",
                "--config", str(config_path),
                "--checkpoint", str(checkpoint_path),
                "--output-dir", str(gradcam_dir)]
            if behavior_probe.exists():
                gradcam_cmd += ["--behavior-probe", str(behavior_probe)]
            if dataset_probe.exists():
                gradcam_cmd += ["--dataset-probe", str(dataset_probe)]

            result = subprocess.run(gradcam_cmd, check=True)

            print(f"[pipeline] Grad-CAM completed for run {run}")
        else:
            print(f"[pipeline] Skipped Grad-CAM for run {run}")



if __name__ == "__main__":
    main()
