"""run the full eval suite over runs: latent space, the dataset and behavior
linear probes, the non-parametric k-NN behavior eval, and the grad-cam and
expected-gradients attribution.

the dataset probe must fail and the behavior probe must succeed. the pipeline
skips a run that it evaluated already, unless you pass --no-check-evaluated.

by default the probes run with settings that control leakage. the dataset probe
runs at --max-contamination 0.0. the behavior probe runs once on all annotations
and once on --restrict unseen. compare runs on the *unseen* macro_F1 and on kNN,
and not on the full-set number. see README "How to read the metrics" sections 1
and 8.

usage:
    python eval/evaluation_pipeline.py --run-names 0626-NoClassBalancing 06261422

each run uses checkpoints/<run_name>/config.yaml by default. pass --config to
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
        # each run uses its own frozen config, unless --config overrides the
        # config. the frozen snapshot is the truth about what a run used.
        config = args.config or str(Path("checkpoints") / run / "config.yaml")

        # the latent-space visualization step also writes the embeddings cache
        # that the probes reuse.
        matches = [p for p in Path("latent_space").glob(f"{run}*") if p.is_dir()]
        if not matches or not args.check_evaluated:
            subprocess.run(
                ["python", "eval/latent_space.py", "--config", config, "--checkpoint", checkpoint],
                check=True,
            )
            print(f"[pipeline] latent space processed for run {run}")
        else:
            print(f"[pipeline] skipping latent space processing for run {run}")

        # the newest latent_space folder for this run. the probes look for their
        # outputs inside that folder.
        run_dirs = [p for p in Path("latent_space").glob(f"{run}*") if p.is_dir()]
        run_dir = max(run_dirs, key=lambda p: p.name) if run_dirs else None

        # warning: a 3x3-era checkpoint errors out here. --max-contamination 0.0
        # keeps only the probe crops with zero training-pixel overlap, which is
        # possible on a 2x2 grid only. see README "How to read the metrics"
        # section 2. probe a 3x3-era checkpoint manually with a looser cap.
        dataset_done = list(run_dir.glob("dataset_probe*")) if run_dir else []
        if not dataset_done or not args.check_evaluated:
            subprocess.run(
                ["python", "eval/probe_dataset.py", "--config", config,
                 "--checkpoint", checkpoint, "--max-contamination", "0.0"],
                check=True,
            )
            print(f"[pipeline] dataset probe completed for run {run}")
        else:
            print(f"[pipeline] skipped dataset probe for run {run}")

        # run the behavior probe twice: first on the full annotation set, then
        # on the unseen experiments only. the two runs measure different things,
        # and one run cannot replace the other. see README "How to read the
        # metrics" sections 1 and 8. the order does not matter, because the
        # restrict run writes behavior_probe_unseen.pt and cannot overwrite the
        # head that grad-cam loads.
        behavior_done = list(run_dir.glob("behavior_probe*")) if run_dir else []
        if not behavior_done or not args.check_evaluated:
            for restrict in ("all", "unseen"):
                subprocess.run(
                    ["python", "eval/probe_behavior.py", "--config", config,
                     "--checkpoint", checkpoint, "--restrict", restrict],
                    check=True,
                )
                print(f"[pipeline] behavior probe ({restrict}) completed for run {run}")
        else:
            print(f"[pipeline] skipped behavior probe for run {run}")

        # the non-parametric k-NN behavior eval needs no probe .pt, because the
        # eval extracts the embeddings again.
        knn_done = list(run_dir.glob("knn_eval*")) if run_dir else []
        if run_dir is not None and (not knn_done or not args.check_evaluated):
            subprocess.run(
                ["python", "eval/knn_behavior.py", "--config", config, "--checkpoint", checkpoint, "--weighted", "--class-balanced"],
                check=True,
            )
            print(f"[pipeline] knn probe completed for run {run}")
        else:
            print(f"[pipeline] skipped knn probe for run {run}")

        # pass the probe paths explicitly from the dir of THIS run. then grad-cam
        # never auto-detects a stale probe with the wrong latent_dim.
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

        # pass explicit probe paths for the same reason as grad-cam. run_eg puts
        # its own output in run_dir/eg.
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
