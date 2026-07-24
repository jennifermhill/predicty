"""run the full eval suite over runs: latent space, dataset + behavior linear
probes, non-parametric k-NN behavior eval, and grad-cam + expected-gradients
attribution.

for each run this visualizes the latent space and runs the two linear probes.
the dataset probe should fail (the encoder learned dataset-invariant features)
and the behavior probe should succeed (embeddings transfer to the downstream
task). already-evaluated runs are skipped unless --no-check-evaluated is passed,
so re-running the pipeline does not redo finished work.

usage:
    python eval/evaluation_pipeline.py --run-names 0626-NoClassBalancing 06261422

each run uses checkpoints/<run_name>/config.yaml by default; pass --config to
override the config for every listed run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default=None,
        help="optional path to a training yaml config; by default each run uses "
        "checkpoints/<run_name>/config.yaml",
    )
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
        # each run uses its own frozen config unless a global --config override
        # is passed. restores the per-run behavior described in submit_eval.sh.
        config = args.config or str(Path("checkpoints") / run / "config.yaml")

        # latent-space visualization; also writes the embeddings cache the probes reuse.
        matches = [p for p in Path("latent_space").glob(f"{run}*") if p.is_dir()]
        if not matches or not args.check_evaluated:
            subprocess.run(
                ["python", "eval/latent_space.py", "--config", config, "--checkpoint", checkpoint],
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
                ["python", "eval/probe_dataset.py", "--config", config, "--checkpoint", checkpoint],
                check=True,
            )
            print(f"[pipeline] dataset probe completed for run {run}")
        else:
            print(f"[pipeline] skipped dataset probe for run {run}")

        behavior_done = list(run_dir.glob("behavior_probe*")) if run_dir else []
        if not behavior_done or not args.check_evaluated:
            subprocess.run(
                ["python", "eval/probe_behavior.py", "--config", config, "--checkpoint", checkpoint],
                check=True,
            )
            print(f"[pipeline] behavior probe completed for run {run}")
        else:
            print(f"[pipeline] skipped behavior probe for run {run}")

        # non-parametric k-NN behavior eval (no probe .pt needed; it re-extracts embeddings).
        knn_done = list(run_dir.glob("knn_eval*")) if run_dir else []
        if run_dir is not None and (not knn_done or not args.check_evaluated):
            subprocess.run(
                ["python", "eval/knn_behavior.py", "--config", config, "--checkpoint", checkpoint, "--weighted", "--class-balanced"],
                check=True,
            )
            print(f"[pipeline] knn probe completed for run {run}")
        else:
            print(f"[pipeline] skipped knn probe for run {run}")

        # grad-cam attribution maps. pass the probe paths from THIS run's dir
        # explicitly so it never auto-detects a stale probe of the wrong latent_dim.
        gradcam_done = list(run_dir.glob("gradcam")) if run_dir else []
        if run_dir is not None and (not gradcam_done or not args.check_evaluated):
            subprocess.run(
                [
                    "python", "eval/run_gradcam.py",
                    "--config", config, "--checkpoint", checkpoint,
                    "--output-dir", str(run_dir / "gradcam"),
                    "--behavior-probe", str(run_dir / "behavior_probe.pt"),
                    "--dataset-probe", str(run_dir / "dataset_probe.pt"),
                ],
                check=True,
            )
            print(f"[pipeline] grad-cam completed for run {run}")
        else:
            print(f"[pipeline] skipped grad-cam for run {run}")

        # expected-gradients attribution. pass this run's probe paths explicitly
        # (same reason as grad-cam); run_eg auto-places its output in run_dir/eg.
        eg_done = list(run_dir.glob("eg")) if run_dir else []
        if run_dir is not None and (not eg_done or not args.check_evaluated):
            subprocess.run(
                [
                    "python", "eval/run_eg.py",
                    "--config", config, "--checkpoint", checkpoint,
                    "--behavior-probe", str(run_dir / "behavior_probe.pt"),
                    "--dataset-probe", str(run_dir / "dataset_probe.pt"),
                ],
                check=True,
            )
            print(f"[pipeline] expected-gradients completed for run {run}")
        else:
            print(f"[pipeline] skipped expected-gradients for run {run}")


if __name__ == "__main__":
    main()
