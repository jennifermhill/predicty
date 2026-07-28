"""sweep the latent space left to right along PC1 and render it as a video.

walks the first PCA axis (coord_x in a pca/ metadata.csv) from its minimum to
its maximum, and at each step shows the crop that lives there plus a few of its
nearest neighbors. the result is a video of how the encoder's representation
changes as you move across PC1, with a marker riding the scatter so you can see
where on the map each frame comes from.

this reads an existing pca/metadata.csv written by eval/latent_space.py (no
re-embedding) and reconstructs the raw crops from the zarrs with the same code
path the interactive viewer uses.

usage:
    # pooled sweep over every dataset in the run
    python eval/latent_traversal_video.py \
        --config configs/train.yaml \
        --metadata latent_space/<run>/pca/metadata.csv

    # sweep a single experiment
    python eval/latent_traversal_video.py \
        --config configs/train.yaml \
        --metadata latent_space/<run>/pca/metadata.csv \
        --dataset-idx 3

    # walk only the annotated crops, on a PCA refit to that subset, with the
    # map panel colored by behavior label
    python eval/latent_traversal_video.py \
        --config configs/train.yaml \
        --metadata latent_space/<run>/pca/metadata.csv \
        --labeled-only --fps 4

    # off-cluster, reading the same storage through the mounted sgrolab share
    python eval/latent_traversal_video.py \
        --config configs/train.yaml \
        --metadata latent_space/<run>/pca/metadata.csv \
        --labeled-only --fps 4 --local-mount

this script needs neither a GPU nor torch: it reads the already-computed
embeddings.npz plus the raw zarrs, so with the storage mounted it runs on a
laptop. --zarr-root-map rewrites the cluster paths baked into the config.

point --metadata at the pca/ metadata.csv specifically: PC1 is an ordered,
deterministic axis, so a left-to-right walk is meaningful. a umap/ metadata.csv
has arbitrary axes and the sweep would not mean anything.

--labeled-only restricts the walk to annotated crops and *refits* PCA on just
that subset (from embeddings.npz), which is a much better axis than the
full-cloud PC1. measured on 0720-iBOTPatch4, the fraction of an axis's variance
that sits between behavior classes was:

    full-cloud PC1, restricted to labeled crops   0.052
    PCA refit on the labeled crops only           0.281

so the refit matters: filtering the full-cloud coord_x down to annotated rows
walks an axis that barely knows the labels exist. the refit axis is also more
about behavior than about experiment (0.281 vs 0.160 for zarr volume), unlike
full-cloud UMAP1, which is the reverse (0.258 vs 0.325).

note this is still unsupervised -- PCA never sees the labels, it just gets a
subset that happens to be enriched for behaviorally distinct crops -- so the
alignment is earned rather than fit, but there is no guarantee PC1 orders the
classes in any biologically meaningful sequence. on the run above it does not:
single_cells sits between clumps and aggregate.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

# allow running as `python eval/latent_traversal_video.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

# headless rendering; must be selected before pyplot is imported anywhere,
# including transitively via the interactive viewer import below.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from configs.config import Config
from eval.interactive_latent_viewer import (
    add_zarr_remap_args,
    default_embeddings_path,
    load_raw_crop,
    load_real_embeddings,
    read_metadata,
    require_zarrs_exist,
    resolve_zarr_paths,
    robust_limits,
    unique_crop_indices,
    validate_dataset_mapping,
)

# same palette as latent_space.py's annotations_*_by_label.png, so the video's
# colors match the still plots for the same run.
LABEL_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#ffff33", "#a65628", "#f781bf",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="training yaml used for the run")
    p.add_argument("--metadata", required=True, help="pca/metadata.csv from latent_space.py")
    p.add_argument("--out", default=None, help="output mp4 path; defaults next to the metadata")
    p.add_argument(
        "--dataset-idx",
        type=int,
        default=None,
        help="sweep only this dataset_idx; omit to pool all datasets",
    )
    add_zarr_remap_args(p)
    p.add_argument(
        "--labeled-only",
        action="store_true",
        help=(
            "keep only annotated crops and refit PCA on that subset (needs "
            "embeddings.npz). see the module docstring for why the refit matters."
        ),
    )
    p.add_argument(
        "--embeddings",
        default=None,
        help=(
            "embeddings.npz matching --metadata (default: one directory above "
            "it). only read when --labeled-only is given."
        ),
    )
    p.add_argument(
        "--embedding-preprocess",
        choices=("raw", "l2", "standardize"),
        default="l2",
        help=(
            "must match the --embedding-preprocess used by the latent_space.py "
            "run that wrote --metadata, or the refit axis won't correspond to "
            "the scatter you're used to looking at. l2 is that script's default."
        ),
    )
    p.add_argument(
        "--color-by",
        choices=("auto", "dataset", "label"),
        default="auto",
        help=(
            "map panel coloring. auto = label when --labeled-only, else dataset "
            "(coloring an annotation walk by dataset wastes the one axis that "
            "tracks behavior)."
        ),
    )
    p.add_argument(
        "--bins",
        type=int,
        default=None,
        help=(
            "number of steps across PC1. default 150 for a full-cloud sweep, or "
            "one step per ~5 crops when --labeled-only, since the annotated "
            "subset is far smaller and 150 bins would leave 1-2 crops per step."
        ),
    )
    p.add_argument(
        "--montage",
        type=int,
        default=4,
        help="crops shown per step (the medoid plus its nearest neighbors)",
    )
    p.add_argument("--fps", type=int, default=15, help="frames per second in the output video")
    p.add_argument("--cmap", default="gray", help="matplotlib colormap for the crop images")
    p.add_argument("--point-size", type=float, default=6.0, help="scatter point size on the map")
    return p.parse_args()


def preprocess_embeddings(X: np.ndarray, mode: str) -> np.ndarray:
    """mirror of eval.latent_space.preprocess_embeddings, reimplemented here.

    latent_space.py can't be imported for it: that module calls
    mp.set_start_method("spawn", force=True) at import time and pulls in torch,
    the dataset and the encoder, none of which this script needs.
    """
    X = X.astype(np.float64, copy=False)
    if mode == "raw":
        return X
    if mode == "l2":
        return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    if mode == "standardize":
        return (X - X.mean(axis=0, keepdims=True)) / np.maximum(
            X.std(axis=0, keepdims=True), 1e-6
        )
    raise ValueError(f"unknown embedding preprocess mode: {mode}")


def refit_pca_2d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PC1/PC2 of X, fit on exactly the rows given. returns (coords, explained).

    plain numpy SVD rather than torch.pca_lowrank: the annotated subset is a few
    hundred points by 128 dims, so an exact decomposition is instant and
    deterministic, with no randomized-algorithm sign or ordering wobble between
    runs.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    ev = s ** 2
    explained = ev / max(ev.sum(), 1e-30)
    return Xc @ Vt[:2].T, explained


def label_of(row: dict[str, Any]) -> str:
    """metadata.csv writes unlabeled rows as an empty field, but a csv round-trip
    of a None can also land as the literal string "None"; treat both as absent."""
    value = (row.get("label") or "").strip()
    return "" if value in ("", "None", "nan") else value


def equal_count_bins(x: np.ndarray, n_bins: int) -> list[np.ndarray]:
    """split point indices into n_bins contiguous groups sorted by x (PC1).

    equal-count (quantile) bins guarantee every step has crops to show and the
    sweep stays monotonic in PC1, unlike fixed-width bins which leave the sparse
    tails empty and make the video stutter.
    """
    order = np.argsort(x, kind="stable")
    n_bins = max(1, min(n_bins, len(order)))
    return np.array_split(order, n_bins)


def crop_anchor(row: dict[str, Any]) -> tuple[int, ...]:
    return (
        int(row["dataset_idx"]),
        int(row["t"]),
        int(row["z"]),
        int(row["z_end"]),
        int(row["global_y"]),
        int(row["global_x"]),
    )


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    zarr_paths = resolve_zarr_paths(cfg, args)

    metadata_path = Path(args.metadata)
    rows, coords = read_metadata(metadata_path)
    validate_dataset_mapping(rows, zarr_paths)

    # embeddings must be loaded before any filtering so the npz row order still
    # lines up with the metadata; the same index selection is then applied to
    # both. load_real_embeddings also verifies the npz belongs to this
    # metadata.csv by matching its stored coords.
    embeddings: np.ndarray | None = None
    if args.labeled_only:
        embeddings_path = (
            Path(args.embeddings) if args.embeddings else default_embeddings_path(metadata_path)
        )
        print(f"[traversal] loading embeddings from {embeddings_path}")
        embeddings = load_real_embeddings(embeddings_path, coords)

    keep = np.arange(len(rows))
    if args.dataset_idx is not None:
        keep = keep[[int(rows[i]["dataset_idx"]) == args.dataset_idx for i in keep]]
        if not len(keep):
            raise ValueError(f"no points with dataset_idx={args.dataset_idx} in the metadata")
    if args.labeled_only:
        keep = keep[[bool(label_of(rows[i])) for i in keep]]
        if not len(keep):
            raise ValueError(
                "--labeled-only but no rows in this metadata.csv carry a label; "
                "the latent_space.py run needs an annotation-aware extraction"
            )
        print(f"[traversal] {len(keep)} annotated points before dedup")

    # collapse t1/t2 points that reload the same physical crop, so a step's
    # neighbors are genuinely distinct crops rather than the same one twice.
    keep = unique_crop_indices(keep, rows, cfg)
    rows = [rows[i] for i in keep]
    coords = coords[keep]
    if embeddings is not None:
        embeddings = embeddings[keep]
    print(f"[traversal] {len(rows)} unique crops after dedup")

    axis_prefix = "PC"
    if args.labeled_only:
        assert embeddings is not None
        # refit AFTER dedup so near-identical t1/t2 pairs don't double-weight
        # their crop in the decomposition.
        X = preprocess_embeddings(embeddings, args.embedding_preprocess)
        coords, explained = refit_pca_2d(X)
        axis_prefix = "annotation PC"
        print(
            f"[traversal] refit PCA on {len(rows)} annotated crops "
            f"(preprocess={args.embedding_preprocess}): "
            f"PC1 {explained[0]*100:.1f}%  PC2 {explained[1]*100:.1f}%"
        )

    n_bins = args.bins if args.bins is not None else (
        max(10, len(rows) // 5) if args.labeled_only else 150
    )
    pc1 = coords[:, 0]
    bins = equal_count_bins(pc1, n_bins)
    per_bin = len(rows) / max(len(bins), 1)
    print(
        f"[traversal] {len(bins)} steps across {axis_prefix}1 "
        f"[{pc1.min():.3f}, {pc1.max():.3f}]  ({per_bin:.1f} crops/step, "
        f"{len(bins)/max(args.fps,1):.1f}s at {args.fps} fps)"
    )
    if per_bin < args.montage:
        print(
            f"[traversal] WARNING: fewer than --montage={args.montage} crops per "
            f"step; some montage cells will repeat the bin's few members or sit "
            f"blank. lower --bins or --montage."
        )

    require_zarrs_exist(zarr_paths, rows)

    color_by = args.color_by
    if color_by == "auto":
        color_by = "label" if args.labeled_only else "dataset"

    out_path = Path(args.out) if args.out else metadata_path.parent / (
        f"pca1_traversal{'_labeled' if args.labeled_only else ''}"
        f"{'' if args.dataset_idx is None else f'_ds{args.dataset_idx}'}.mp4"
    )

    # crop reconstruction is the slow part (a dask read per crop), so cache the
    # projected crop by its physical anchor. arrays caches the open zarr handles.
    arrays: dict[int, Any] = {}
    mip_cache: dict[tuple[int, ...], np.ndarray] = {}

    def crop_mip(row: dict[str, Any]) -> np.ndarray:
        anchor = crop_anchor(row)
        cached = mip_cache.get(anchor)
        if cached is not None:
            return cached
        raw = load_raw_crop(arrays, zarr_paths, row, cfg, int(row["t"]))
        mip = raw.max(axis=0)  # (Z, H, W) -> (H, W) maximum-intensity projection
        mip_cache[anchor] = mip
        return mip

    # per step: medoid (nearest the bin centroid) first, then nearest neighbors.
    steps: list[dict[str, Any]] = []
    for members in bins:
        member_coords = coords[members]
        centroid = member_coords.mean(axis=0)
        order = np.argsort(np.linalg.norm(member_coords - centroid, axis=1))
        chosen = members[order[: args.montage]]
        medoid = int(members[order[0]])
        steps.append({"medoid": medoid, "chosen": [int(i) for i in chosen]})

    _render_video(args, cfg, rows, coords, steps, crop_mip, out_path, color_by, axis_prefix)
    print(f"[traversal] wrote {len(steps)} frames to {out_path}")


def _render_video(
    args: argparse.Namespace,
    cfg: Config,
    rows: list[dict[str, Any]],
    coords: np.ndarray,
    steps: list[dict[str, Any]],
    crop_mip: Any,
    out_path: Path,
    color_by: str = "dataset",
    axis_prefix: str = "PC",
) -> None:
    import cv2

    crop_size = cfg.global_crop_size
    cols = math.ceil(math.sqrt(args.montage))
    grid_rows = math.ceil(args.montage / cols)

    fig = plt.figure(figsize=(12, 6), dpi=100)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.15)
    ax_map = fig.add_subplot(outer[0, 0])
    inner = outer[0, 1].subgridspec(grid_rows, cols, wspace=0.05, hspace=0.25)

    # static scatter of the cloud being walked. colored by dataset for a
    # full-cloud sweep, or by behavior label for an annotation walk -- with far
    # fewer points, bigger dots and the annotation palette read better and the
    # label gradient along PC1 is the thing worth seeing.
    if color_by == "label":
        labels = np.asarray([label_of(r) for r in rows])
        values = sorted(v for v in set(labels.tolist()) if v)
        for color_idx, value in enumerate(values):
            mask = labels == value
            ax_map.scatter(
                coords[mask, 0], coords[mask, 1],
                color=LABEL_COLORS[color_idx % len(LABEL_COLORS)],
                s=max(args.point_size * 4.0, 20.0), alpha=0.85, linewidths=0,
                label=f"{value} (n={int(mask.sum())})",
            )
        unlabeled = labels == ""
        if unlabeled.any():
            ax_map.scatter(
                coords[unlabeled, 0], coords[unlabeled, 1],
                color="0.8", s=args.point_size, alpha=0.35, linewidths=0,
                label=f"unlabeled (n={int(unlabeled.sum())})", zorder=0,
            )
        legend_title = "behavior label"
    else:
        dataset_ids = np.asarray([int(r["dataset_idx"]) for r in rows])
        cmap = plt.get_cmap("tab20")
        for color_idx, ds in enumerate(sorted(set(dataset_ids.tolist()))):
            mask = dataset_ids == ds
            ax_map.scatter(
                coords[mask, 0], coords[mask, 1],
                color=cmap(color_idx % 20), s=args.point_size, alpha=0.45, linewidths=0,
                label=f"{ds}: {rows[int(np.flatnonzero(mask)[0])]['dataset_name'][:20]}",
            )
        legend_title = "dataset"
    ax_map.legend(
        fontsize=6, frameon=False, loc="upper left", markerscale=1.6,
        title=legend_title, title_fontsize=6,
    )
    ax_map.set_xlabel(f"{axis_prefix}1")
    ax_map.set_ylabel(f"{axis_prefix}2")
    marker = ax_map.scatter([], [], s=140, facecolor="none", edgecolor="red", linewidths=2.0, zorder=5)

    # crop image axes, initialized with blanks and updated in place each frame.
    blank = np.zeros((crop_size, crop_size), dtype=np.float32)
    img_axes = []
    ims = []
    for k in range(args.montage):
        r, c = divmod(k, cols)
        ax = fig.add_subplot(inner[r, c])
        ax.set_xticks([])
        ax.set_yticks([])
        im = ax.imshow(blank, cmap=args.cmap, aspect="equal")
        img_axes.append(ax)
        ims.append(im)

    suptitle = fig.suptitle("", fontsize=12)

    fig.canvas.draw()
    height, width = fig.canvas.get_width_height()[::-1]
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {out_path}")

    try:
        for step_idx, step in enumerate(steps):
            medoid_row = rows[step["medoid"]]
            marker.set_offsets(coords[step["medoid"]][None, :])

            for k, ax in enumerate(img_axes):
                if k < len(step["chosen"]):
                    idx = step["chosen"][k]
                    row = rows[idx]
                    mip = crop_mip(row)
                    lo, hi = robust_limits(mip)
                    ims[k].set_data(mip)
                    ims[k].set_clim(lo, hi)
                    label = label_of(row) or "-"
                    ax.set_title(f"t={int(row['t'])}  ds{int(row['dataset_idx'])}  {label}", fontsize=8)
                else:
                    ims[k].set_data(blank)
                    ax.set_title("", fontsize=8)
                # medoid cell gets a red border, neighbors a faint gray one.
                is_medoid = k == 0
                for spine in ax.spines.values():
                    spine.set_color("red" if is_medoid else "0.7")
                    spine.set_linewidth(2.0 if is_medoid else 0.8)

            suptitle.set_text(
                f"{axis_prefix}1 sweep  step {step_idx + 1}/{len(steps)}  "
                f"{axis_prefix}1={coords[step['medoid'], 0]:.3f}"
                + (f"  [{label_of(medoid_row)}]" if label_of(medoid_row) else "")
            )

            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            writer.write(bgr)

            if step_idx == 0 or (step_idx + 1) % 25 == 0:
                print(f"  [traversal] rendered {step_idx + 1}/{len(steps)} frames")
    finally:
        writer.release()
        plt.close(fig)


if __name__ == "__main__":
    main()
