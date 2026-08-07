"""the script sweeps the latent space left to right along PC1 and renders a video.

warning: nothing guarantees that PC1 orders the behavior classes in a
biologically meaningful way. on 0720-iBOTPatch4, single_cells sits between
clumps and aggregate. PCA is unsupervised and never sees the labels.

the script walks the first PCA axis (coord_x in a pca/ metadata.csv) from its
minimum to its maximum. at each step the script shows the crop that lives there
plus a few of its nearest neighbors. the result is a video of the change in the
encoder's representation as you move across PC1. a marker rides the scatter, so
you can see where on the map each frame comes from.

the script reads an existing pca/metadata.csv from eval/latent_space.py and does
no re-embedding. the script reconstructs the raw crops from the zarrs with the
same code path as the interactive viewer.

usage:
    # pooled sweep over every dataset in the run (add --dataset-idx N for one)
    python eval/latent_traversal_video.py \
        --config configs/train.yaml \
        --metadata latent_space/<run>/pca/metadata.csv

    # THE USUAL ONE: straight line through the annotation PCA, every crop
    # available to the walk, run off the locally mounted share
    python eval/latent_traversal_video.py \
        --config checkpoints/<run>/config.yaml \
        --metadata latent_space/<run>/pca/metadata.csv \
        --labeled-only --include-unannotated \
        --line --line-band-percentile 2 --bins 90 \
        --fps 4 --local-mount

the script needs neither a GPU nor torch. the script reads embeddings.npz plus
the raw zarrs, so it runs on a laptop with the storage mounted (see
--local-mount).

point --metadata at the pca/ metadata.csv specifically. PC1 is an ordered,
deterministic axis, so a left-to-right walk means something. umap/ axes are
arbitrary, and a sweep along them would mean nothing.

--labeled-only refits PCA on the annotated subset. the refit axis is much better
than the full-cloud PC1. on 0720-iBOTPatch4 the between-behavior share of axis
variance was 0.052 for full-cloud PC1 restricted to labeled crops, against 0.281
for the refit. the refit axis is also more about behavior than about experiment
(0.281 against 0.160). full-cloud UMAP1 is the reverse (0.258 against 0.325).

the script fits the refit on every annotated row and applies it to the
deduplicated rows. the fit matches latent_space.py's
annotations_pca_by_label.png. both go through eval/pca_utils.py, so they share
one decomposition and one sign convention.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

# this line lets you run `python eval/latent_traversal_video.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

# the matplotlib.use call must come before any pyplot import, and the transitive
# import below is one of them.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from configs.config import Config

# the same exact-SVD PCA that latent_space.py uses. a refit here and the static
# plots there therefore cannot quietly disagree on the sign or the
# implementation.
from eval.pca_utils import pca_fit, pca_transform, preprocess_embeddings
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

# the same palette as latent_space.py's annotations_*_by_label.png, so that the
# video colors match the still plots for the same run.
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
    p.add_argument(
        "--include-unannotated",
        action="store_true",
        help=(
            "with --labeled-only: still fit the projection on the annotated crops "
            "only, but let every crop be a candidate for the walk. the basis is "
            "unchanged, so the annotated points land exactly where the static plot "
            "puts them, while the walk gets ~20x more points to pick from -- which "
            "decouples frame count from how straight the line is. note most walked "
            "crops will then be unlabeled."
        ),
    )
    p.add_argument(
        "--line",
        action="store_true",
        help=(
            "walk a straight line in the PC1/PC2 plane instead of binning PC1. "
            "keeps only points within a narrow band of the line, so the selection "
            "stays on the line rather than drifting in PC2. the first montage cell "
            "is the point on the line; the rest are its nearest neighbors."
        ),
    )
    p.add_argument(
        "--show-line",
        action="store_true",
        help=(
            "draw the walk line and its band on the map panel. off by default: "
            "it's a debugging aid for confirming the selection really is "
            "collinear, not something you want in a figure."
        ),
    )
    p.add_argument(
        "--line-angle",
        type=float,
        default=0.0,
        help="direction of the line in degrees; 0 = the PC1 axis (default)",
    )
    p.add_argument(
        "--line-offset",
        type=float,
        default=0.0,
        help="shift the line perpendicular to its direction, in coordinate units",
    )
    p.add_argument(
        "--line-band",
        type=float,
        default=None,
        help=(
            "half-width of the band around the line, in coordinate units. "
            "default: whatever --line-band-percentile works out to."
        ),
    )
    p.add_argument(
        "--line-band-percentile",
        type=float,
        default=20.0,
        help=(
            "used when --line-band is unset: keep the points whose perpendicular "
            "distance is within this percentile. 20 keeps the closest fifth, which "
            "reads as a straight line while still leaving enough crops to sample."
        ),
    )
    p.add_argument("--fps", type=int, default=15, help="frames per second in the output video")
    p.add_argument("--cmap", default="gray", help="matplotlib colormap for the crop images")
    p.add_argument("--point-size", type=float, default=6.0, help="scatter point size on the map")
    return p.parse_args()


def label_of(row: dict[str, Any]) -> str:
    """unlabeled rows have an empty field. a csv round-trip of None can land as
    the literal string "None". the function treats both as absent."""
    value = (row.get("label") or "").strip()
    return "" if value in ("", "None", "nan") else value


def equal_count_bins(x: np.ndarray, n_bins: int) -> list[np.ndarray]:
    """the function splits the point indices into n_bins contiguous groups, sorted by x (PC1).

    equal-count bins keep every step populated and the sweep monotonic in PC1.
    fixed-width bins leave the sparse tails empty, and the video then stutters.
    """
    order = np.argsort(x, kind="stable")
    n_bins = max(1, min(n_bins, len(order)))
    return np.array_split(order, n_bins)


def line_basis(angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """unit direction along the walk and unit normal to it, in the PC1/PC2 plane.

    angle 0 is the PC1 axis, so the default walk is horizontal on the scatter.
    """
    theta = np.radians(float(angle_deg))
    direction = np.array([np.cos(theta), np.sin(theta)])
    normal = np.array([-direction[1], direction[0]])
    return direction, normal


def line_walk(
    coords: np.ndarray,
    n_samples: int,
    angle_deg: float,
    offset: float,
    band: float | None,
    band_percentile: float,
    span: tuple[float, float] = (1.0, 99.0),
) -> tuple[np.ndarray, dict[str, Any]]:
    """the function walks a straight line through the 2D projection and picks real points on it.

    the alternative is bin mode. bin mode takes the medoid of each PC1 slab,
    which cannot hold PC2 steady even in principle. the marker wanders
    vertically with the mean PC2 of each slab. a slab that is bimodal in PC2
    puts the centroid in the empty gap, so consecutive frames jump between
    lobes. a fixed line, plus only the points within `band` of that line,
    guarantees what bin mode never could.

    the function collapses consecutive duplicate picks. each distinct crop
    therefore gets exactly one frame, however long the stretch of line that it
    was nearest to.

    the function returns (indices in walk order, info dict for the plot and the
    report).
    """
    if n_samples < 2:
        raise ValueError("need at least 2 samples along the line")
    direction, normal = line_basis(angle_deg)
    centre = coords.mean(axis=0)
    along = (coords - centre) @ direction
    perpendicular = (coords - centre) @ normal - float(offset)

    if band is None:
        if not 0.0 < band_percentile <= 100.0:
            raise ValueError("--line-band-percentile must be in (0, 100]")
        band = float(np.percentile(np.abs(perpendicular), band_percentile))
    band = max(float(band), 1e-9)

    candidates = np.flatnonzero(np.abs(perpendicular) <= band)
    if len(candidates) < 2:
        raise ValueError(
            f"only {len(candidates)} point(s) lie within {band:.4f} of the line; "
            "widen --line-band / --line-band-percentile or drop --line-offset"
        )

    lo, hi = np.percentile(along[candidates], span)
    if not hi > lo:
        raise ValueError("line span collapsed to a point; check --line-angle")
    samples = np.linspace(lo, hi, n_samples)
    # the comparison is 1D on purpose. every candidate is already within band
    # perpendicular, so the along-line nearness decides the order and the
    # progress stays monotonic.
    picks = candidates[np.argmin(np.abs(along[candidates][None, :] - samples[:, None]), axis=1)]

    ordered: list[int] = [int(picks[0])]
    for pick in picks[1:]:
        if int(pick) != ordered[-1]:
            ordered.append(int(pick))
    indices = np.asarray(ordered, dtype=np.int64)

    info = {
        "direction": direction,
        "normal": normal,
        "centre": centre,
        "band": band,
        "offset": float(offset),
        "along": along,
        "perpendicular": perpendicular,
        "lo": float(lo),
        "hi": float(hi),
        "n_candidates": int(len(candidates)),
        "n_samples": int(n_samples),
    }
    return indices, info


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
    if args.include_unannotated and not args.labeled_only:
        raise ValueError(
            "--include-unannotated only means something alongside --labeled-only: "
            "it widens the walk while keeping the annotated-only fit. without "
            "--labeled-only the walk already uses every crop."
        )
    cfg = Config.from_yaml(args.config)
    zarr_paths = resolve_zarr_paths(cfg, args)

    metadata_path = Path(args.metadata)
    rows, coords = read_metadata(metadata_path)
    validate_dataset_mapping(rows, zarr_paths)

    # the code loads before any filter, so that the npz row order still lines up
    # with the metadata. one index selection can then apply to both.
    embeddings: np.ndarray | None = None
    if args.labeled_only:
        embeddings_path = (
            Path(args.embeddings) if args.embeddings else default_embeddings_path(metadata_path)
        )
        print(f"[traversal] loading embeddings from {embeddings_path}")
        embeddings = load_real_embeddings(embeddings_path, coords)

    base = np.arange(len(rows))
    if args.dataset_idx is not None:
        base = base[[int(rows[i]["dataset_idx"]) == args.dataset_idx for i in base]]
        if not len(base):
            raise ValueError(f"no points with dataset_idx={args.dataset_idx} in the metadata")

    # the fit uses the pre-dedup annotated rows, the same rows that
    # annotations_pca_by_label.png uses. a fit on the deduped subset would
    # rotate the plane, and the two figures would then disagree for no reason.
    fit_keep = base
    if args.labeled_only:
        fit_keep = base[[bool(label_of(rows[i])) for i in base]]
        if not len(fit_keep):
            raise ValueError(
                "--labeled-only but no rows in this metadata.csv carry a label; "
                "the latent_space.py run needs an annotation-aware extraction"
            )
        print(f"[traversal] {len(fit_keep)} annotated points define the projection")

    # the set that the walk can pick from. an expansion to every crop leaves the
    # basis untouched, and it gives the line far more real points to snap to.
    walk_source = base if args.include_unannotated else fit_keep

    # the code collapses the latent points that render the *same* crop figure.
    # t1 and t2 are genuinely different images, but each one displays the
    # identical t1/t2 panel pair. both points together therefore waste a frame
    # and inflate the apparent population of a step.
    keep = unique_crop_indices(walk_source, rows, cfg)
    fit_rows = [rows[i] for i in fit_keep]
    rows = [rows[i] for i in keep]

    axis_prefix = "PC"
    if args.labeled_only:
        assert embeddings is not None
        # the code preprocesses every row and *then* takes the subset, the order
        # that latent_space.py uses. under --embedding-preprocess standardize,
        # the column statistics depend on which rows go in. a preprocess of a
        # subset would therefore fit a different basis than the static
        # annotation plot.
        X_all = preprocess_embeddings(embeddings, args.embedding_preprocess)
        components, mean, explained = pca_fit(X_all[fit_keep], 2)
        coords = pca_transform(X_all[keep], components, mean)
        axis_prefix = "annotation PC"
        print(
            f"[traversal] refit PCA on {len(fit_rows)} annotated points "
            f"(preprocess={args.embedding_preprocess}): "
            f"PC1 {explained[0]*100:.1f}%  PC2 {explained[1]*100:.1f}%"
        )
        n_labeled = sum(1 for row in rows if label_of(row))
        print(
            f"[traversal] walking {len(rows)} crops in that basis "
            f"({n_labeled} labeled, {len(rows) - n_labeled} unlabeled)"
        )
    else:
        coords = coords[keep]
        print(f"[traversal] {len(rows)} unique crops after dedup")

    pc1 = coords[:, 0]
    line_info: dict[str, Any] | None = None
    bins: list[np.ndarray] = []
    line_indices: np.ndarray | None = None

    if args.line:
        # the code samples densely and lets the duplicate collapse decide the
        # frame count. every crop then gets the same single frame.
        n_samples = args.bins if args.bins is not None else max(300, 4 * len(rows))
        line_indices, line_info = line_walk(
            coords,
            n_samples,
            args.line_angle,
            args.line_offset,
            args.line_band,
            args.line_band_percentile,
        )
        picked_perp = np.abs(line_info["perpendicular"][line_indices])
        print(
            f"[traversal] line walk at {args.line_angle:g} deg, offset "
            f"{args.line_offset:g}, band +/-{line_info['band']:.4f} "
            f"({line_info['n_candidates']}/{len(rows)} crops in band)"
        )
        print(
            f"[traversal] {n_samples} samples along the line collapsed to "
            f"{len(line_indices)} distinct crops, one frame each "
            f"({len(line_indices)/max(args.fps,1):.1f}s at {args.fps} fps)"
        )
        print(
            f"[traversal] selected points sit {picked_perp.mean():.4f} from the line "
            f"on average (max {picked_perp.max():.4f}); "
            f"{axis_prefix}1 spans [{line_info['lo']:.3f}, {line_info['hi']:.3f}]"
        )
    else:
        n_bins = args.bins if args.bins is not None else (
            max(10, len(rows) // 5) if args.labeled_only else 150
        )
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

    if args.out:
        out_path = Path(args.out)
    else:
        # a line-mode name carries the angle and the band. a sweep of several
        # directions therefore cannot overwrite the previous render, and it
        # cannot collide with a file that is open in a player.
        tag = ""
        if args.line:
            band_tag = (
                f"b{args.line_band:g}" if args.line_band is not None
                else f"p{args.line_band_percentile:g}"
            )
            tag = f"_line_a{args.line_angle:g}_{band_tag}"
        out_path = metadata_path.parent / (
            f"pca1_traversal{tag}"
            f"{'_labeled' if args.labeled_only else ''}"
            f"{'_all' if args.include_unannotated else ''}"
            f"{'' if args.dataset_idx is None else f'_ds{args.dataset_idx}'}.mp4"
        )

    # the crop reconstruction is the slow part, because each crop is one dask
    # read. the code therefore caches the projection by physical anchor. arrays
    # caches the open zarr handles.
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

    steps: list[dict[str, Any]] = []
    if line_indices is not None:
        # the point on the line drives the frame. the rest of the montage is its
        # nearest neighbors, the "medoid plus neighbors" idea of bin mode. a 2x2
        # grid also upscales a 128px crop 2x rather than 4x, so the grid reads
        # sharper than one big panel.
        for idx in line_indices:
            chosen = [int(idx)]
            if args.montage > 1:
                order = np.argsort(np.linalg.norm(coords - coords[idx], axis=1))
                ranked = unique_crop_indices(order, rows, cfg)
                for candidate in ranked:
                    if len(chosen) >= args.montage:
                        break
                    if int(candidate) != int(idx):
                        chosen.append(int(candidate))
            steps.append({"medoid": int(idx), "chosen": chosen})
    else:
        # per step: the medoid first, nearest the bin centroid, then the nearest neighbors.
        for members in bins:
            member_coords = coords[members]
            centroid = member_coords.mean(axis=0)
            order = np.argsort(np.linalg.norm(member_coords - centroid, axis=1))
            chosen = members[order[: args.montage]]
            medoid = int(members[order[0]])
            steps.append({"medoid": medoid, "chosen": [int(i) for i in chosen]})

    _render_video(
        args, cfg, rows, coords, steps, crop_mip, out_path, color_by, axis_prefix, line_info
    )
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
    line_info: dict[str, Any] | None = None,
) -> None:
    import cv2

    crop_size = cfg.global_crop_size
    cols = math.ceil(math.sqrt(args.montage))
    grid_rows = math.ceil(args.montage / cols)

    fig = plt.figure(figsize=(12, 6), dpi=100)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.15)
    ax_map = fig.add_subplot(outer[0, 0])
    inner = outer[0, 1].subgridspec(grid_rows, cols, wspace=0.05, hspace=0.25)

    # static scatter of the cloud that the marker walks. the color comes from
    # the dataset for a full-cloud sweep. the color comes from the behavior
    # label for an annotation walk, where the label gradient along PC1 is the
    # thing worth a look.
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
    if line_info is not None:
        # a debug overlay, off by default. the thing worth a look is the marker
        # on a straight path, not the scaffolding behind it.
        if getattr(args, "show_line", False):
            centre = line_info["centre"]
            direction = line_info["direction"]
            normal = line_info["normal"]
            base = centre + normal * line_info["offset"]
            start = base + direction * line_info["lo"]
            end = base + direction * line_info["hi"]
            ax_map.plot(
                [start[0], end[0]], [start[1], end[1]],
                color="red", linewidth=1.2, alpha=0.9, zorder=4, label="walk line",
            )
            for sign in (-1.0, 1.0):
                edge_start = start + normal * sign * line_info["band"]
                edge_end = end + normal * sign * line_info["band"]
                ax_map.plot(
                    [edge_start[0], edge_end[0]], [edge_start[1], edge_end[1]],
                    color="red", linewidth=0.7, alpha=0.45, linestyle="--", zorder=4,
                )
        # the aspect is equal in every case, because axes with different scales
        # hide the collinearity of the points.
        ax_map.set_aspect("equal", adjustable="datalim")

    ax_map.legend(
        fontsize=6, frameon=False, loc="upper left", markerscale=1.6,
        title=legend_title, title_fontsize=6,
    )
    ax_map.set_xlabel(f"{axis_prefix}1")
    ax_map.set_ylabel(f"{axis_prefix}2")
    marker = ax_map.scatter([], [], s=140, facecolor="none", edgecolor="red", linewidths=2.0, zorder=5)

    # the crop image axes start blank. each frame updates them in place.
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

    # the code renders locally and copies the file into place at the end. a
    # direct mp4 write to the mounted share fails if the previous render is open
    # in a player. Windows denies the write, and cv2 reports only "could not
    # open". a local scratch file also avoids per-frame round trips over SMB.
    tmp_dir = tempfile.mkdtemp(prefix="latent_traversal_")
    tmp_video = Path(tmp_dir) / out_path.name
    writer = cv2.VideoWriter(
        str(tmp_video), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height)
    )
    if not writer.isOpened():
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(
            f"could not open a video writer even locally at {tmp_video}. that points "
            f"at the OpenCV build rather than the destination: check that cv2 has the "
            f"mp4v encoder (frame size {width}x{height}, fps {args.fps})."
        )
    print(f"[traversal] rendering to {tmp_video}, will copy to {out_path}")

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
                # the medoid cell gets a red border. the neighbor cells get a
                # faint gray border.
                is_medoid = k == 0
                for spine in ax.spines.values():
                    spine.set_color("red" if is_medoid else "0.7")
                    spine.set_linewidth(2.0 if is_medoid else 0.8)

            if line_info is not None:
                idx = step["medoid"]
                text = (
                    f"line walk  step {step_idx + 1}/{len(steps)}  "
                    f"{axis_prefix}1={coords[idx, 0]:.3f}  {axis_prefix}2={coords[idx, 1]:.3f}  "
                    f"off-line={line_info['perpendicular'][idx]:+.3f}"
                )
            else:
                text = (
                    f"{axis_prefix}1 sweep  step {step_idx + 1}/{len(steps)}  "
                    f"{axis_prefix}1={coords[step['medoid'], 0]:.3f}"
                )
            suptitle.set_text(
                text + (f"  [{label_of(medoid_row)}]" if label_of(medoid_row) else "")
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

    try:
        # this copy comes after the finally block and not inside it, because
        # release() must happen first.
        shutil.copyfile(tmp_video, out_path)
    except PermissionError as exc:
        raise RuntimeError(
            f"rendered {len(steps)} frames but could not write {out_path}: {exc}. "
            f"the file is almost certainly open somewhere -- close it in your video "
            f"player or Explorer preview pane, or pass a different --out. the "
            f"finished video is still at {tmp_video}."
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"rendered {len(steps)} frames but could not write {out_path}: {exc}. "
            f"the finished video is still at {tmp_video}."
        ) from exc
    shutil.rmtree(tmp_video.parent, ignore_errors=True)


if __name__ == "__main__":
    main()
