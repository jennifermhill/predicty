"""sliced-Wasserstein batch-effect probe over cached embeddings.

this probe reuses the `embeddings.npz` that `latent_space.py` writes. therefore
you can get the batch-effect report, or run it again, and not embed the crops
through the encoder a second time. the probe computes the pairwise
sliced-Wasserstein distance between the per-dataset embedding clouds in the full
latent space. the probe then writes a report next to the embeddings.

a lower distance means that the clouds overlap, that is, the encoder scrubbed
dataset identity and the mixing is good. a higher distance means that the
datasets stay separable, that is, a residual batch effect remains.

usage:
    # point at an existing latent_space run folder (the folder that contains the embeddings.npz)
    python eval/probe_wasserstein.py --config configs/train.yaml --run-dir latent_space/final_20260711_161004

    # or point at the npz directly. the probe writes the report next to the npz
    python eval/probe_wasserstein.py --config configs/train.yaml --embeddings latent_space/final_20260711_161004/embeddings.npz

    # or let the probe auto-find the newest cached run for a checkpoint
    python eval/probe_wasserstein.py --config configs/train.yaml --checkpoint checkpoints/z_filtering/latest.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# the path insert lets you run `python eval/probe_wasserstein.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from configs.config import Config
from eval.wasserstein import print_batch_effect_wasserstein


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="path to the training yaml config")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", default=None, help="latent_space run folder containing embeddings.npz")
    src.add_argument("--embeddings", default=None, help="path to an embeddings.npz directly")
    src.add_argument(
        "--checkpoint",
        default=None,
        help="auto-find the newest cached latent_space run for this checkpoint",
    )
    p.add_argument(
        "--embedding-preprocess",
        choices=("raw", "l2", "standardize"),
        default="l2",
        help="must match how the run was projected; l2 is the latent_space default",
    )
    p.add_argument("--seed", type=int, default=0, help="random-projection seed")
    p.add_argument("--n-projections", type=int, default=128, help="number of random 1d slices")
    p.add_argument(
        "--min-per-dataset",
        type=int,
        default=10,
        help="drop datasets with fewer points than this from the comparison",
    )
    return p.parse_args()


def find_cached_run(checkpoint: str) -> Path | None:
    """the newest latent_space/<run>_* folder for this checkpoint that has embeddings."""
    base = Path("latent_space")
    if not base.is_dir():
        return None
    prefix = f"{Path(checkpoint).parent.name}_"
    candidates = [
        d for d in base.iterdir()
        if d.is_dir() and d.name.startswith(prefix) and (d / "embeddings.npz").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: (d / "embeddings.npz").stat().st_mtime)


def resolve_npz(args: argparse.Namespace) -> Path:
    if args.embeddings is not None:
        return Path(args.embeddings)
    if args.run_dir is not None:
        return Path(args.run_dir) / "embeddings.npz"
    cached = find_cached_run(args.checkpoint)
    if cached is None:
        raise FileNotFoundError(
            f"no cached latent_space run with embeddings.npz found for {args.checkpoint}"
        )
    print(f"[wasserstein] auto-selected cached run {cached}")
    return cached / "embeddings.npz"


def preprocess_embeddings(X: np.ndarray, mode: str) -> np.ndarray:
    # warning: this function is a duplicate of
    # eval/pca_utils.preprocess_embeddings, which latent_space.py uses. keep the
    # two functions the same, or the distances here no longer match a full run.
    # note also that this copy works in float32 and that the pca_utils copy
    # works in float64.
    X = X.astype(np.float32, copy=False)
    if mode == "raw":
        return X
    if mode == "l2":
        denom = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(denom, 1e-12)
    if mode == "standardize":
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        return (X - mean) / np.maximum(std, 1e-6)
    raise ValueError(f"unknown embedding preprocess mode: {mode}")


def load_dataset_rows(npz_path: Path, cfg: Config) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """load raw embeddings and the minimal per-point rows that the metric needs.

    print_batch_effect_wasserstein reads only dataset_idx and dataset_name.
    therefore this function does not rebuild the full metadata that latent_space
    carries.
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"embeddings not found at {npz_path}")
    data = np.load(npz_path, allow_pickle=False)
    X = data["embeddings"]
    dataset_idx = data["dataset_idx"]

    zarr_paths = list(cfg.zarr_paths) if cfg.zarr_paths else [cfg.zarr_path or ""]
    if not zarr_paths or all(p == "" for p in zarr_paths):
        raise ValueError("no zarr paths configured; needed to name datasets")

    rows = [
        {
            "dataset_idx": int(d),
            "dataset_name": Path(zarr_paths[int(d)]).name if int(d) < len(zarr_paths) else str(int(d)),
        }
        for d in dataset_idx
    ]
    return X, rows


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)

    npz_path = resolve_npz(args)
    out_dir = npz_path.parent
    print(f"[wasserstein] embeddings : {npz_path}")
    print(f"[wasserstein] out_dir    : {out_dir}")
    print(f"[wasserstein] preprocess : {args.embedding_preprocess}")

    X, rows = load_dataset_rows(npz_path, cfg)
    print(f"[wasserstein] loaded {X.shape[0]} points, dim={X.shape[1]}")

    X_projected = preprocess_embeddings(X, args.embedding_preprocess)
    print_batch_effect_wasserstein(
        X_projected,
        rows,
        out_dir=out_dir,
        n_projections=args.n_projections,
        seed=args.seed,
        min_per_dataset=args.min_per_dataset,
    )


if __name__ == "__main__":
    main()
