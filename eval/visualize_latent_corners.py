"""Reload and visualize representative crops near three corners of a 2D latent map.

The input is ``metadata.csv`` written by ``eval/latent_space.py``. Sparse
projection outliers are removed before a maximum-area triangle is fit to the
dense manifold. Samples near each vertex are chosen dataset-first so a corner
contact sheet does not accidentally contain only one experiment.

Example:
    python eval/visualize_latent_corners.py \
        --config configs/train.yaml \
        --metadata latent_space/latest/metadata.csv \
        --samples-per-corner 4
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dask.array as da
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from configs.config import Config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize exact crops sampled near three latent-space corners."
    )
    p.add_argument("--config", required=True, help="training YAML used for latent extraction")
    p.add_argument("--metadata", required=True, help="metadata.csv from eval/latent_space.py")
    p.add_argument(
        "--out-dir",
        default=None,
        help="output directory; defaults to <metadata-dir>/corner_crops",
    )
    p.add_argument("--samples-per-corner", type=int, default=4)
    p.add_argument(
        "--candidate-pool",
        type=int,
        default=200,
        help="nearest points considered at each corner before dataset-diverse selection",
    )
    p.add_argument("--density-neighbors", type=int, default=15)
    p.add_argument(
        "--outlier-mad",
        type=float,
        default=6.0,
        help="exclude points whose local kNN radius exceeds median + this many robust SDs",
    )
    p.add_argument("--plot-size", type=float, default=9.0)
    return p.parse_args()


def read_metadata(path: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 3:
        raise ValueError(f"need at least 3 metadata rows; got {len(rows)}")

    required = {
        "coord_x",
        "coord_y",
        "dataset_idx",
        "dataset_name",
        "view",
        "t",
        "z",
        "z_end",
        "global_y",
        "global_x",
        "frame_gap",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"metadata is missing required columns: {sorted(missing)}")

    coords = np.asarray(
        [[float(row["coord_x"]), float(row["coord_y"])] for row in rows],
        dtype=np.float64,
    )
    if not np.isfinite(coords).all():
        raise ValueError("metadata contains non-finite latent coordinates")
    return rows, coords


def dense_manifold_mask(
    coords: np.ndarray, density_neighbors: int, outlier_mad: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Keep points with a typical local kNN radius.

    UMAP occasionally places a few samples far from the main manifold. A
    maximum-area triangle would otherwise lock onto those sparse points.
    """
    k = min(max(2, density_neighbors), len(coords) - 1)
    distances, _ = cKDTree(coords).query(coords, k=k + 1)
    local_radius = np.asarray(distances[:, -1])
    median = float(np.median(local_radius))
    mad = float(np.median(np.abs(local_radius - median)))
    robust_sd = max(1.4826 * mad, np.finfo(np.float64).eps)
    threshold = median + outlier_mad * robust_sd
    mask = local_radius <= threshold
    if int(mask.sum()) < 3:
        raise ValueError("density filtering retained fewer than three points")
    return mask, local_radius, threshold


def maximum_area_triangle(coords: np.ndarray, max_hull_points: int = 240) -> np.ndarray:
    """Return indices of an approximate maximum-area triangle in ``coords``."""
    hull = ConvexHull(coords)
    candidates = np.asarray(hull.vertices, dtype=np.int64)
    if len(candidates) > max_hull_points:
        # Preserve coverage around the perimeter without cubic work on a very
        # detailed hull. ConvexHull vertices are returned in boundary order.
        positions = np.linspace(0, len(candidates), max_hull_points, endpoint=False)
        candidates = candidates[np.floor(positions).astype(int)]

    points = coords[candidates]
    best_area2 = -1.0
    best = (0, 1, 2)
    for i in range(len(points) - 2):
        for j in range(i + 1, len(points) - 1):
            edge = points[j] - points[i]
            rel = points[j + 1 :] - points[i]
            area2 = np.abs(edge[0] * rel[:, 1] - edge[1] * rel[:, 0])
            k_rel = int(np.argmax(area2))
            if float(area2[k_rel]) > best_area2:
                best_area2 = float(area2[k_rel])
                best = (i, j, j + 1 + k_rel)
    return candidates[np.asarray(best)]


def name_corners(vertices: np.ndarray) -> dict[str, np.ndarray]:
    """Assign stable visual names to three distinct triangle vertices."""
    top_idx = int(np.argmax(vertices[:, 1]))
    bottom_idx = int(np.argmin(vertices[:, 1]))
    if top_idx == bottom_idx:
        raise ValueError("could not distinguish top and bottom corners")
    remaining = ({0, 1, 2} - {top_idx, bottom_idx}).pop()
    return {
        "top": vertices[top_idx],
        "left": vertices[remaining],
        "bottom": vertices[bottom_idx],
    }


def select_diverse_samples(
    coords: np.ndarray,
    rows: list[dict[str, Any]],
    corner_targets: dict[str, np.ndarray],
    samples_per_corner: int,
    candidate_pool: int,
    eligible_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    if samples_per_corner < 1:
        raise ValueError("--samples-per-corner must be at least 1")
    if eligible_mask is None:
        eligible_indices = np.arange(len(coords))
    else:
        eligible_indices = np.flatnonzero(eligible_mask)
    pool_size = min(len(eligible_indices), max(samples_per_corner, candidate_pool))
    selected: list[dict[str, Any]] = []

    for corner, target in corner_targets.items():
        distances = np.linalg.norm(coords - target[None, :], axis=1)
        candidates = eligible_indices[np.argsort(distances[eligible_indices])[:pool_size]]
        chosen: list[int] = []
        seen_datasets: set[str] = set()
        seen_anchors: set[tuple[str, ...]] = set()

        # First pass: nearest point from each distinct dataset.
        for idx in candidates:
            row = rows[int(idx)]
            dataset = row["dataset_idx"]
            anchor = (
                dataset,
                row["t"],
                row["z"],
                row["global_y"],
                row["global_x"],
            )
            if dataset in seen_datasets or anchor in seen_anchors:
                continue
            chosen.append(int(idx))
            seen_datasets.add(dataset)
            seen_anchors.add(anchor)
            if len(chosen) == samples_per_corner:
                break

        # Second pass: fill from nearest unused anchors if the pool contains
        # fewer datasets than requested samples.
        if len(chosen) < samples_per_corner:
            for idx in candidates:
                row = rows[int(idx)]
                anchor = (
                    row["dataset_idx"],
                    row["t"],
                    row["z"],
                    row["global_y"],
                    row["global_x"],
                )
                if int(idx) in chosen or anchor in seen_anchors:
                    continue
                chosen.append(int(idx))
                seen_anchors.add(anchor)
                if len(chosen) == samples_per_corner:
                    break

        for rank, idx in enumerate(chosen, start=1):
            selected.append(
                {
                    "corner": corner,
                    "sample_id": f"{corner[0].upper()}{rank}",
                    "corner_rank": rank,
                    "corner_distance": float(distances[idx]),
                    "row_index": idx,
                    **rows[idx],
                }
            )
    return selected


def zarr_paths_from_config(cfg: Config) -> list[str]:
    if cfg.zarr_paths:
        return list(cfg.zarr_paths)
    if cfg.zarr_path:
        return [cfg.zarr_path]
    raise ValueError("config must define zarr_path or zarr_paths")


def temporal_pair(row: dict[str, Any], cfg: Config) -> tuple[int, int]:
    t = int(row["t"])
    gap = int(row["frame_gap"])
    sign = 1 if cfg.temporal_direction == "forward" else -1
    if row["view"] == "t1":
        return t, t + sign * gap
    if row["view"] == "t2":
        return t - sign * gap, t
    raise ValueError(f"unknown view {row['view']!r}")


def load_raw_crop(
    arrays: dict[int, da.Array],
    zarr_paths: list[str],
    row: dict[str, Any],
    cfg: Config,
    t: int,
) -> np.ndarray:
    dataset_idx = int(row["dataset_idx"])
    if not 0 <= dataset_idx < len(zarr_paths):
        raise IndexError(f"dataset_idx={dataset_idx} is outside configured zarr_paths")
    if dataset_idx not in arrays:
        arrays[dataset_idx] = da.from_zarr(f"{zarr_paths[dataset_idx]}/0/0")
    arr = arrays[dataset_idx]
    z0, z1 = int(row["z"]), int(row["z_end"])
    y0, x0 = int(row["global_y"]), int(row["global_x"])
    size = cfg.global_crop_size
    crop = arr[t, cfg.channel, z0:z1, y0 : y0 + size, x0 : x0 + size]
    return np.asarray(crop.compute(), dtype=np.float32)


def robust_limits(*images: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([image.reshape(-1) for image in images])
    lo, hi = np.percentile(values, [1.0, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = float(values.min()), float(values.max())
    if lo >= hi:
        hi = lo + 1.0
    return float(lo), float(hi)


def save_corner_sheet(
    out_path: Path,
    corner: str,
    samples: list[dict[str, Any]],
    arrays: dict[int, da.Array],
    zarr_paths: list[str],
    cfg: Config,
) -> None:
    n = len(samples)
    fig, axes = plt.subplots(4, n, figsize=(3.2 * n, 11.0), dpi=170, squeeze=False)
    row_titles = ["t1 mean-Z", "t2 mean-Z", "t1 max-Z", "t2 max-Z"]

    for col, sample in enumerate(samples):
        t1, t2 = temporal_pair(sample, cfg)
        crop_t1 = load_raw_crop(arrays, zarr_paths, sample, cfg, t1)
        crop_t2 = load_raw_crop(arrays, zarr_paths, sample, cfg, t2)
        mean_t1, mean_t2 = crop_t1.mean(axis=0), crop_t2.mean(axis=0)
        max_t1, max_t2 = crop_t1.max(axis=0), crop_t2.max(axis=0)
        mean_limits = robust_limits(mean_t1, mean_t2)
        max_limits = robust_limits(max_t1, max_t2)
        images = [mean_t1, mean_t2, max_t1, max_t2]
        for row_idx, image in enumerate(images):
            ax = axes[row_idx, col]
            limits = mean_limits if row_idx < 2 else max_limits
            ax.imshow(image, cmap="gray", vmin=limits[0], vmax=limits[1])
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(row_titles[row_idx], fontsize=9)
        title = (
            f"{sample['sample_id']} | ds={sample['dataset_idx']}\n"
            f"{sample['dataset_name'][:24]}\n"
            f"t=({t1},{t2}) z=[{sample['z']},{sample['z_end']})"
        )
        axes[0, col].set_title(title, fontsize=8)

    fig.suptitle(f"Latent {corner} corner: exact temporal crop pairs", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_annotated_umap(
    out_path: Path,
    coords: np.ndarray,
    dense_mask: np.ndarray,
    corner_targets: dict[str, np.ndarray],
    selected: list[dict[str, Any]],
    plot_size: float,
) -> None:
    colors = {"top": "#d62728", "left": "#2ca02c", "bottom": "#1f77b4"}
    fig, ax = plt.subplots(figsize=(plot_size, plot_size), dpi=180)
    ax.scatter(coords[~dense_mask, 0], coords[~dense_mask, 1], s=7, c="#dddddd", alpha=0.45)
    ax.scatter(coords[dense_mask, 0], coords[dense_mask, 1], s=5, c="#999999", alpha=0.35)
    for corner, target in corner_targets.items():
        color = colors[corner]
        ax.scatter(*target, marker="X", s=180, c=color, edgecolor="black", linewidth=0.8)
        corner_rows = [row for row in selected if row["corner"] == corner]
        for row in corner_rows:
            idx = int(row["row_index"])
            x, y = coords[idx]
            ax.scatter(x, y, s=75, facecolor=color, edgecolor="black", linewidth=0.7)
            ax.annotate(
                row["sample_id"],
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                weight="bold",
            )
    ax.set_title("Latent-space corner samples")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_selection_csv(path: Path, selected: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in selected:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata)
    out_dir = Path(args.out_dir) if args.out_dir else metadata_path.parent / "corner_crops"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.from_yaml(args.config)
    rows, coords = read_metadata(metadata_path)
    dense_mask, local_radius, threshold = dense_manifold_mask(
        coords, args.density_neighbors, args.outlier_mad
    )
    dense_indices = np.flatnonzero(dense_mask)
    vertex_local_indices = maximum_area_triangle(coords[dense_mask])
    vertices = coords[dense_indices[vertex_local_indices]]
    corner_targets = name_corners(vertices)
    selected = select_diverse_samples(
        coords,
        rows,
        corner_targets,
        args.samples_per_corner,
        args.candidate_pool,
        dense_mask,
    )

    print(
        f"[corners] dense points: {dense_mask.sum()}/{len(coords)} "
        f"(kNN-radius threshold={threshold:.4g}, max={local_radius.max():.4g})"
    )
    for corner, target in corner_targets.items():
        names = [row["sample_id"] + ":ds" + row["dataset_idx"] for row in selected if row["corner"] == corner]
        print(f"[corners] {corner:6s} target=({target[0]:.3f}, {target[1]:.3f}) samples={names}")

    save_annotated_umap(
        out_dir / "latent_corners_annotated.png",
        coords,
        dense_mask,
        corner_targets,
        selected,
        args.plot_size,
    )
    write_selection_csv(out_dir / "corner_selections.csv", selected)

    zarr_paths = zarr_paths_from_config(cfg)
    arrays: dict[int, da.Array] = {}
    for corner in corner_targets:
        corner_samples = [row for row in selected if row["corner"] == corner]
        save_corner_sheet(
            out_dir / f"corner_{corner}_crops.png",
            corner,
            corner_samples,
            arrays,
            zarr_paths,
            cfg,
        )

    print(f"[corners] wrote {out_dir}")
    for path in sorted(out_dir.iterdir()):
        print(f"  - {path}")


if __name__ == "__main__":
    main()
