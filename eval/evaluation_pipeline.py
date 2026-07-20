"""run the full eval suite (latent space + dataset/behavior probes) over runs.

for each run this visualizes the latent space and runs the two linear probes.
the dataset probe should fail (the encoder learned dataset-invariant features)
and the behavior probe should succeed (embeddings transfer to the downstream
task). already-evaluated runs are skipped unless --no-check-evaluated is passed,
so re-running the pipeline does not redo finished work.

usage:
    python eval/evaluation_pipeline.py --config checkpoints/<run_name>/config.yaml --run-names 0626-NoClassBalancing 06261422
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to the training yaml config")
    p.add_argument(
        "--check-evaluated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip runs whose latent_space / probe outputs already exist",
    )
    p.add_argument("--run-names", required=True, nargs="+", help="run names to evaluate")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    for run in args.run_names:
        checkpoint = str(Path("checkpoints") / run / "latest.pt")

        # latent-space visualization; also writes the embeddings cache the probes reuse.
        matches = [p for p in Path("latent_space").glob(f"{run}*") if p.is_dir()]
        if not matches or not args.check_evaluated:
            subprocess.run(
                ["python", "eval/latent_space.py", "--config", args.config, "--checkpoint", checkpoint],
                check=True,
            )
            print(f"[pipeline] latent space processed for run {run}")
        else:
            print(f"[pipeline] skipping latent space processing for run {run}")

        # newest latent_space folder for this run; probes look for their outputs inside it.
        run_dirs = [p for p in Path("latent_space").glob(f"{run}*") if p.is_dir()]
        run_dir = max(run_dirs, key=lambda p: p.name) if run_dirs else None

        dataset_done = list(run_dir.glob("dataset_probe*")) if run_dir else []
        if not dataset_done or not args.check_evaluated:
            subprocess.run(
                ["python", "eval/probe_dataset.py", "--config", args.config, "--checkpoint", checkpoint],
                check=True,
            )
            print(f"[pipeline] dataset probe completed for run {run}")
        else:
            print(f"[pipeline] skipped dataset probe for run {run}")

        behavior_done = list(run_dir.glob("behavior_probe*")) if run_dir else []
        if not behavior_done or not args.check_evaluated:
            subprocess.run(
                ["python", "eval/probe_behavior.py", "--config", args.config, "--checkpoint", checkpoint],
                check=True,
            )
            print(f"[pipeline] behavior probe completed for run {run}")
        else:
            print(f"[pipeline] skipped behavior probe for run {run}")


if __name__ == "__main__":
    main()
