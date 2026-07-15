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

point --metadata at the pca/ metadata.csv specifically: PC1 is an ordered,
deterministic axis, so a left-to-right walk is meaningful. a umap/ metadata.csv
has arbitrary axes and the sweep would not mean anything.
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
    load_raw_crop,
    read_metadata,
    robust_limits,
    unique_crop_indices,
    validate_dataset_mapping,
    zarr_paths_from_config,
)


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
    p.add_argument("--bins", type=int, default=150, help="number of steps across PC1")
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


def filter_by_dataset(
    rows: list[dict[str, Any]], coords: np.ndarray, dataset_idx: int | None
) -> tuple[list[dict[str, Any]], np.ndarray]:
    if dataset_idx is None:
        return rows, coords
    keep = [i for i, r in enumerate(rows) if int(r["dataset_idx"]) == dataset_idx]
    if not keep:
        raise ValueError(f"no points with dataset_idx={dataset_idx} in the metadata")
    return [rows[i] for i in keep], coords[keep]


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
    zarr_paths = zarr_paths_from_config(cfg)

    metadata_path = Path(args.metadata)
    rows, coords = read_metadata(metadata_path)
    validate_dataset_mapping(rows, zarr_paths)
    rows, coords = filter_by_dataset(rows, coords, args.dataset_idx)

    # collapse t1/t2 points that reload the same physical crop, so a step's
    # neighbors are genuinely distinct crops rather than the same one twice.
    unique = unique_crop_indices(np.arange(len(rows)), rows, cfg)
    rows = [rows[i] for i in unique]
    coords = coords[unique]
    print(f"[traversal] {len(rows)} unique crops after dedup")

    pc1 = coords[:, 0]
    bins = equal_count_bins(pc1, args.bins)
    print(f"[traversal] {len(bins)} steps across PC1 [{pc1.min():.3f}, {pc1.max():.3f}]")

    out_path = Path(args.out) if args.out else metadata_path.parent / (
        f"pca1_traversal_ds{args.dataset_idx}.mp4"
        if args.dataset_idx is not None
        else "pca1_traversal.mp4"
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

    _render_video(args, cfg, rows, coords, steps, crop_mip, out_path)
    print(f"[traversal] wrote {len(steps)} frames to {out_path}")


def _render_video(
    args: argparse.Namespace,
    cfg: Config,
    rows: list[dict[str, Any]],
    coords: np.ndarray,
    steps: list[dict[str, Any]],
    crop_mip: Any,
    out_path: Path,
) -> None:
    import cv2

    crop_size = cfg.global_crop_size
    cols = math.ceil(math.sqrt(args.montage))
    grid_rows = math.ceil(args.montage / cols)

    fig = plt.figure(figsize=(12, 6), dpi=100)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.15)
    ax_map = fig.add_subplot(outer[0, 0])
    inner = outer[0, 1].subgridspec(grid_rows, cols, wspace=0.05, hspace=0.25)

    # static scatter of the whole cloud, colored by dataset.
    dataset_ids = np.asarray([int(r["dataset_idx"]) for r in rows])
    cmap = plt.get_cmap("tab20")
    for color_idx, ds in enumerate(sorted(set(dataset_ids))):
        mask = dataset_ids == ds
        ax_map.scatter(
            coords[mask, 0], coords[mask, 1],
            color=cmap(color_idx % 20), s=args.point_size, alpha=0.45, linewidths=0,
        )
    ax_map.set_xlabel("PC1")
    ax_map.set_ylabel("PC2")
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
                    label = row.get("label") or "-"
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
                f"PC1 sweep  step {step_idx + 1}/{len(steps)}  "
                f"PC1={coords[step['medoid'], 0]:.3f}"
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
