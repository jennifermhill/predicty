"""sliced-Wasserstein batch-effect metric for embedding clouds.

standalone, numpy-only module so it can be reused without pulling in the heavy
latent-space visualization stack (torch, umap, matplotlib). both
`latent_space.py` (as part of a full run) and `probe_wasserstein.py` (as a
standalone probe over cached embeddings) import from here.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np


def sliced_wasserstein_distance(
    x_a: np.ndarray,
    x_b: np.ndarray,
    n_projections: int = 128,
    n_quantiles: int = 100,
    seed: int = 0,
) -> float:
    """sliced Wasserstein-1 distance between two point clouds.

    projects both clouds onto many random 1d directions, computes the exact 1d
    wasserstein-1 distance on each slice by matching quantiles, and averages. the
    1d case reduces to sorting, so this approximates the full high-dim optimal
    transport distance without a solver or extra dependency. quantile matching
    also handles clouds with different sample counts.
    """
    dim = x_a.shape[1]
    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((dim, n_projections)).astype(np.float64)
    directions /= np.linalg.norm(directions, axis=0, keepdims=True) + 1e-12

    proj_a = x_a.astype(np.float64) @ directions  # (n_a, n_projections)
    proj_b = x_b.astype(np.float64) @ directions  # (n_b, n_projections)

    q = np.linspace(0.0, 1.0, n_quantiles)
    qa = np.quantile(proj_a, q, axis=0)  # (n_quantiles, n_projections)
    qb = np.quantile(proj_b, q, axis=0)
    # mean |Δquantile| per slice IS the 1d wasserstein-1 distance; average slices.
    return float(np.abs(qa - qb).mean())


def _format_wasserstein_report(
    mat: np.ndarray,
    label_list: list[str],
    counts: list[int],
    excluded: list[tuple[str, int]],
    n_projections: int,
    seed: int,
    min_per_dataset: int,
    n_top_pairs: int = 8,
) -> str:
    """render the pairwise distance matrix as a human-readable text report.

    datasets are shown as short [i] tags in the matrix with a legend mapping each
    tag to its full name, so long zarr filenames don't blow out the column width.
    """
    n = len(label_list)
    lines: list[str] = []
    rule = "=" * 78
    lines.append(rule)
    lines.append(" sliced-Wasserstein batch-effect report")
    lines.append(rule)
    lines.append("")
    lines.append(" what this measures:")
    lines.append("   pairwise distance between each dataset's embedding cloud in the full")
    lines.append("   latent space. lower = clouds overlap = dataset identity scrubbed (good")
    lines.append("   mixing); higher = datasets stay separable = residual batch effect.")
    lines.append("")
    lines.append(
        f" config: n_projections={n_projections}  seed={seed}  "
        f"min_per_dataset={min_per_dataset}"
    )
    lines.append(f" datasets included: {n}   excluded (too few points): {len(excluded)}")
    for name, cnt in excluded:
        lines.append(f"   - {name}  (n={cnt})")
    lines.append("")

    # legend: short tag -> full dataset name + point count.
    lines.append(" legend:")
    for i, (name, cnt) in enumerate(zip(label_list, counts)):
        lines.append(f"   [{i}]  {name}  (n={cnt})")
    lines.append("")

    # matrix with compact [i] headers; diagonal shown as a dot.
    tags = [f"[{i}]" for i in range(n)]
    col_w = max(8, max(len(t) for t in tags) + 2)
    row_label_w = max(len(t) for t in tags) + 2
    lines.append(" pairwise distance matrix:")
    header = " " * row_label_w + "".join(f"{t:>{col_w}}" for t in tags)
    lines.append(header)
    for i in range(n):
        cells = "".join(
            f"{'.':>{col_w}}" if i == j else f"{mat[i, j]:>{col_w}.4f}"
            for j in range(n)
        )
        lines.append(f"{tags[i]:>{row_label_w}}{cells}")
    lines.append("")

    iu = np.triu_indices(n, k=1)
    vals = mat[iu]
    lines.append(f" summary over {vals.size} dataset pairs:")
    lines.append(f"   mean = {vals.mean():.4f}")
    lines.append(f"   min  = {vals.min():.4f}")
    lines.append(f"   max  = {vals.max():.4f}")
    lines.append("")

    # ranked pairs make the matrix actionable: which datasets are worst / best mixed.
    pairs = sorted(
        ((mat[i, j], i, j) for i, j in zip(*iu)), key=lambda p: p[0], reverse=True
    )
    k = min(n_top_pairs, len(pairs))
    lines.append(f" most separated pairs (largest batch effect, top {k}):")
    for dist, i, j in pairs[:k]:
        lines.append(f"   {dist:.4f}   [{i}] vs [{j}]")
    lines.append("")
    lines.append(f" best mixed pairs (smallest distance, top {k}):")
    for dist, i, j in pairs[-k:][::-1]:
        lines.append(f"   {dist:.4f}   [{i}] vs [{j}]")
    lines.append("")
    lines.append(rule)
    return "\n".join(lines) + "\n"


def print_batch_effect_wasserstein(
    X: np.ndarray,
    rows: list[dict[str, Any]],
    out_dir: Path | None = None,
    n_projections: int = 128,
    seed: int = 0,
    min_per_dataset: int = 10,
) -> None:
    """pairwise sliced-Wasserstein distance between per-dataset embedding clouds.

    a direct, geometry-aware measure of residual batch effect. if the dataset
    adversary scrubbed dataset identity, the clouds overlap and distances sit
    near zero; large distances mean datasets stay separable in latent space.
    computed on full-dim embeddings, where the encoder actually operates, rather
    than on a 2d projection. the full matrix is written to a log file under
    out_dir; only a short summary is printed to the terminal.

    rows only need `dataset_idx` and (optionally) `dataset_name` per point.
    """
    dataset_idx = np.asarray([r["dataset_idx"] for r in rows], dtype=np.int64)
    unique = np.unique(dataset_idx)
    if unique.size < 2:
        print("[wasserstein] only one dataset; skipping batch-effect distance")
        return

    names: dict[int, str] = {}
    for r in rows:
        names.setdefault(
            int(r["dataset_idx"]), r.get("dataset_name") or str(r["dataset_idx"])
        )

    # drop datasets with too few points for stable quantile estimates.
    groups: dict[int, np.ndarray] = {}
    excluded: list[tuple[str, int]] = []
    for d in unique:
        pts = X[dataset_idx == d]
        if pts.shape[0] >= min_per_dataset:
            groups[int(d)] = pts
        else:
            excluded.append((names[int(d)], int(pts.shape[0])))
    keys = sorted(groups)
    if len(keys) < 2:
        print("[wasserstein] fewer than two datasets with enough points; skipping")
        return

    n = len(keys)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dist = sliced_wasserstein_distance(
                groups[keys[i]], groups[keys[j]], n_projections=n_projections, seed=seed
            )
            mat[i, j] = mat[j, i] = dist

    label_list = [names[k] for k in keys]
    counts = [int(groups[k].shape[0]) for k in keys]
    report = _format_wasserstein_report(
        mat, label_list, counts, excluded, n_projections, seed, min_per_dataset
    )

    iu = np.triu_indices(n, k=1)
    vals = mat[iu]
    worst = max(zip(*iu), key=lambda ij: mat[ij])
    best = min(zip(*iu), key=lambda ij: mat[ij])

    if out_dir is not None:
        report_path = Path(out_dir) / f"wasserstein_{time.strftime('%Y%m%d_%H%M%S')}.log"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"[wasserstein] batch-effect report written to {report_path}")
    else:
        # no target folder; fall back to printing the full report.
        print(report)

    print(
        f"[wasserstein] summary over {vals.size} dataset pairs: "
        f"mean={vals.mean():.4f} min={vals.min():.4f} max={vals.max():.4f}"
    )
    print(
        f"[wasserstein] most separated: [{worst[0]}] vs [{worst[1]}]={mat[worst]:.4f}  |  "
        f"best mixed: [{best[0]}] vs [{best[1]}]={mat[best]:.4f}"
    )
