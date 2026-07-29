"""interactively inspect the raw crops represented by points in a 2D latent map.

left-click the scatter plot to show the nearest points. right-click and drag to
lasso a region. the selected points are highlighted in the latent map and their
exact temporal crops are rendered in a second window.

if metadata.csv has a "label" column (written by an annotation-aware
eval/latent_space.py run), annotated points are drawn with a colored ring on
the map -- colored by their behavior label, layered over the usual per-dataset
dot color -- and any annotated crop shown in the detail window has its label
(or "not annotated") printed in the title. because annotated points are
usually a small minority of everything plotted, a left-click reserves one
slot for the nearest annotated crop even when it isn't literally one of the
N closest points to the click -- otherwise clicking near a ring almost always
surfaces only unlabeled neighbors.

two checkboxes in the bottom-right corner ("Annotated" / "Unannotated", both
on by default) toggle which points are drawn on the map. hidden points are
also excluded from click and lasso selection, so hiding unannotated crops is
an easy way to browse only the labeled ones.

if --decoder-checkpoint is given (a latent_decoder.pt from eval/decoder.py), a
third "Inspect" / "Generate" radio toggle appears. Generate mode lets you
click *anywhere* on the map, including empty areas with no real point: it
finds real points within a search radius of the click, blends their real
embeddings.npz vectors (gaussian-weighted by distance -- never pads out to a
minimum count with distant, less-relevant points), decodes the blend with the
trained decoder, and shows the generated crop next to its nearest real
neighbors for comparison. clicks far from any real data fall back to the
single nearest point and say so plainly rather than quietly extrapolating.

Example:
    python eval/interactive_latent_viewer.py \
        --config configs/train.yaml \
        --metadata latent_space/latest/umap/metadata.csv

    # with Generate mode enabled:
    python eval/interactive_latent_viewer.py \
        --config configs/train.yaml \
        --metadata latent_space/latest/umap/metadata.csv \
        --decoder-checkpoint latent_space/latest/latent_decoder.pt
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.widgets import Button, CheckButtons, LassoSelector, RadioButtons

from configs.config import Config


REQUIRED_COLUMNS = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Click or lasso a latent map to inspect exact raw crops."
    )
    parser.add_argument("--config", required=True, help="training YAML used for extraction")
    parser.add_argument("--metadata", required=True, help="metadata.csv from latent_space.py")
    parser.add_argument(
        "--nearest",
        type=int,
        default=4,
        help="number of nearest crops displayed after a left-click",
    )
    parser.add_argument(
        "--max-crops",
        type=int,
        default=12,
        help="maximum crops displayed from one lasso selection",
    )
    parser.add_argument("--point-size", type=float, default=7.0)
    parser.add_argument(
        "--decoder-checkpoint",
        default=None,
        help=(
            "optional latent_decoder.pt (from eval/decoder.py). enables a "
            "Generate mode toggle that decodes a blend of nearby real "
            "embeddings for empty areas of the map; without it, Generate mode "
            "is unavailable and the viewer behaves exactly as before."
        ),
    )
    parser.add_argument(
        "--embeddings",
        default=None,
        help=(
            "embeddings.npz matching --metadata (default: one directory above "
            "--metadata, matching eval/latent_space.py's layout). only used "
            "when --decoder-checkpoint is given."
        ),
    )
    parser.add_argument(
        "--gen-radius",
        type=float,
        default=None,
        help=(
            "2D search radius for blending nearby real embeddings in Generate "
            "mode, in the same units as coord_x/coord_y. default: 3x the "
            "median nearest-neighbor spacing among plotted points."
        ),
    )
    add_zarr_remap_args(parser)
    return parser.parse_args()


def read_metadata(path: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"metadata is empty: {path}")
    missing = REQUIRED_COLUMNS.difference(rows[0])
    if missing:
        raise ValueError(f"metadata is missing required columns: {sorted(missing)}")
    coords = np.asarray(
        [[float(row["coord_x"]), float(row["coord_y"])] for row in rows],
        dtype=np.float64,
    )
    if not np.isfinite(coords).all():
        raise ValueError("metadata contains non-finite latent coordinates")
    return rows, coords


def nearest_indices(coords: np.ndarray, target: tuple[float, float], count: int) -> np.ndarray:
    if count < 1:
        raise ValueError("count must be at least 1")
    distances = np.linalg.norm(coords - np.asarray(target)[None, :], axis=1)
    return np.argsort(distances)[: min(count, len(coords))]


def indices_inside_polygon(coords: np.ndarray, vertices: list[tuple[float, float]]) -> np.ndarray:
    if len(vertices) < 3:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(MplPath(vertices, closed=True).contains_points(coords))


def representative_indices(
    coords: np.ndarray, selected: np.ndarray, max_crops: int
) -> np.ndarray:
    """keep crops nearest the selected region's centroid when it is very large."""
    if max_crops < 1:
        raise ValueError("max_crops must be at least 1")
    selected = np.asarray(selected, dtype=np.int64)
    if len(selected) <= max_crops:
        return selected
    centroid = coords[selected].mean(axis=0)
    distance = np.linalg.norm(coords[selected] - centroid[None, :], axis=1)
    return selected[np.argsort(distance)[:max_crops]]


def nearest_annotated_index(ordered_indices: np.ndarray, rows: list[dict[str, Any]]) -> int | None:
    """first labeled row in an already distance-sorted, deduplicated index array.

    annotated points are almost always a small minority of the plotted crops,
    so a click landing near -- but not exactly on -- a ring's data coordinate
    will usually have several unlabeled points as its literal nearest
    neighbors. This walks the ranking to find the closest one that actually
    carries a label, so it can be guaranteed a slot in the click results.
    """
    for idx in ordered_indices:
        if rows[int(idx)].get("label"):
            return int(idx)
    return None


def compute_visible_mask(
    rows: list[dict[str, Any]], show_annotated: bool, show_unannotated: bool
) -> np.ndarray:
    """per-row visibility from the two "Annotated"/"Unannotated" checkboxes."""
    annotated = np.asarray([bool(row.get("label")) for row in rows])
    return np.where(annotated, show_annotated, show_unannotated)


def select_click_indices(
    coords: np.ndarray,
    rows: list[dict[str, Any]],
    cfg: Config,
    target: tuple[float, float],
    nearest: int,
    visible_mask: np.ndarray | None = None,
) -> np.ndarray:
    """nearest crops to a click, with the nearest annotated crop guaranteed a slot.

    pure Euclidean "nearest N" can silently exclude the ring you clicked on:
    annotated points are typically a small fraction of the total, so unless
    the click lands exactly on the point's coordinate, its nearest neighbors
    are usually unlabeled. Reserving the last slot for the nearest annotated
    crop (when one exists) keeps its label visible even when it isn't
    literally one of the N closest points.

    visible_mask restricts candidates to currently-shown points (see the
    show/hide checkboxes); points hidden by a checkbox are never selected.
    """
    ranked = nearest_indices(coords, target, len(coords))
    if visible_mask is not None:
        ranked = ranked[visible_mask[ranked]]
        if not len(ranked):
            return np.empty(0, dtype=np.int64)
    unique_ranked = unique_crop_indices(ranked, rows, cfg)
    indices = unique_ranked[:nearest].tolist()
    annotated_idx = nearest_annotated_index(unique_ranked, rows)
    if annotated_idx is not None and annotated_idx not in indices:
        indices = indices[: max(nearest - 1, 0)] + [annotated_idx]
    return np.asarray(indices, dtype=np.int64)


def unique_crop_indices(
    indices: np.ndarray,
    rows: list[dict[str, Any]],
    cfg: Config,
) -> np.ndarray:
    """remove duplicate t1/t2 latent points that reload the same crop pair."""
    unique: list[int] = []
    seen: set[tuple[int, ...]] = set()
    for raw_idx in indices:
        idx = int(raw_idx)
        row = rows[idx]
        t1, t2 = temporal_pair(row, cfg)
        anchor = (
            int(row["dataset_idx"]),
            t1,
            t2,
            int(row["z"]),
            int(row["z_end"]),
            int(row["global_y"]),
            int(row["global_x"]),
        )
        if anchor not in seen:
            unique.append(idx)
            seen.add(anchor)
    return np.asarray(unique, dtype=np.int64)


def zarr_paths_from_config(cfg: Config) -> list[str]:
    if cfg.zarr_paths:
        return list(cfg.zarr_paths)
    if cfg.zarr_path:
        return [cfg.zarr_path]
    raise ValueError("config must define zarr_path or zarr_paths")


# ---------------------------------------------------------------------------
# zarr path remapping, shared by this viewer, the 3D viewer, and
# eval/latent_traversal_video.py. configs freeze absolute cluster paths at
# training time, so any of these scripts run off a mounted copy of the same
# storage needs those prefixes rewritten.
# ---------------------------------------------------------------------------

# the lab's cluster root and where it lands when the share is mounted on
# Aran's machine. only the default for --local-mount; override by passing a
# different root, or use --zarr-root-map for anything else.
CLUSTER_ZARR_ROOT = "/groups/sgro/sgrolab"
DEFAULT_LOCAL_ZARR_ROOT = "Z:/"


def add_zarr_remap_args(parser: argparse.ArgumentParser) -> None:
    """attach --zarr-root-map / --local-mount to a parser."""
    parser.add_argument(
        "--zarr-root-map",
        action="append",
        default=None,
        metavar="OLD=NEW",
        help=(
            "rewrite a prefix of every zarr path from the config, for running "
            "against a mounted copy of the storage instead of the cluster. "
            "repeatable; longest matching OLD wins."
        ),
    )
    parser.add_argument(
        "--local-mount",
        nargs="?",
        const=DEFAULT_LOCAL_ZARR_ROOT,
        default=None,
        metavar="ROOT",
        help=(
            f"shorthand for --zarr-root-map {CLUSTER_ZARR_ROOT}=ROOT, where "
            f"ROOT defaults to {DEFAULT_LOCAL_ZARR_ROOT}. so a bare "
            "--local-mount is all that's needed on a machine with the sgrolab "
            "share mounted there; pass a different root if the drive letter "
            "differs. explicit --zarr-root-map rules for the same prefix win."
        ),
    )


def parse_root_map(
    mappings: list[str] | None, local_mount: str | None = None
) -> list[tuple[str, str]]:
    """OLD=NEW strings into prefix rewrite rules, longest OLD first.

    longest-first so overlapping rules behave predictably: given both
    /groups/sgro and /groups/sgro/sgrolab, the more specific one applies
    regardless of the order they were typed in. an explicit --zarr-root-map for
    the same prefix takes precedence over --local-mount's implied rule.
    """
    rules: list[tuple[str, str]] = []
    for mapping in mappings or []:
        if "=" not in mapping:
            raise ValueError(
                f"--zarr-root-map expects OLD=NEW, got {mapping!r} "
                f"(e.g. --zarr-root-map {CLUSTER_ZARR_ROOT}={DEFAULT_LOCAL_ZARR_ROOT})"
            )
        old, new = mapping.split("=", 1)
        if not old:
            raise ValueError("--zarr-root-map OLD half must not be empty")
        rules.append((old, new))
    if local_mount is not None and not any(old == CLUSTER_ZARR_ROOT for old, _ in rules):
        rules.append((CLUSTER_ZARR_ROOT, local_mount))
    return sorted(rules, key=lambda rule: len(rule[0]), reverse=True)


def remap_zarr_paths(paths: list[str], rules: list[tuple[str, str]]) -> list[str]:
    """apply the first matching prefix rule to each path.

    matches on path *boundaries* only, so a rule for /groups/sgro/sgrolab does
    not fire on /groups/sgro/sgrolab2. joins with exactly one separator, so
    passing a root with a trailing slash ("Z:/") can't produce a "Z://..."
    double slash -- which Windows would read as the start of a UNC share name.

    only the prefix changes, so each path's basename -- which is what
    validate_dataset_mapping checks against metadata.csv's dataset_name -- is
    untouched and that guard keeps working.
    """
    if not rules:
        return list(paths)
    out: list[str] = []
    for path in paths:
        for old, new in rules:
            trimmed = old.rstrip("/\\")
            if path == trimmed:
                path = new
                break
            if path.startswith(trimmed) and path[len(trimmed)] in "/\\":
                remainder = path[len(trimmed) + 1:]
                path = new.rstrip("/\\") + "/" + remainder
                break
        out.append(path)
    return out


def resolve_zarr_paths(cfg: Config, args: argparse.Namespace) -> list[str]:
    """config zarr paths with any --zarr-root-map / --local-mount applied."""
    zarr_paths = zarr_paths_from_config(cfg)
    rules = parse_root_map(
        getattr(args, "zarr_root_map", None), getattr(args, "local_mount", None)
    )
    if not rules:
        return zarr_paths
    remapped = remap_zarr_paths(zarr_paths, rules)
    for before, after in zip(zarr_paths, remapped):
        if before != after:
            print(f"[zarr] remapped {before}\n            -> {after}")
    unchanged = sum(1 for b, a in zip(zarr_paths, remapped) if b == a)
    if unchanged:
        print(
            f"[zarr] WARNING: {unchanged}/{len(zarr_paths)} zarr paths matched no "
            "remap rule and were left as-is"
        )
    return remapped


def require_zarrs_exist(zarr_paths: list[str], rows: list[dict[str, Any]]) -> None:
    """fail before doing any work if a zarr these rows need isn't readable.

    checks only the datasets actually present in rows, so a filtered subset
    isn't blocked by an unrelated volume being offline. done up front because
    the alternative is discovering a bad remap partway through a render, or on
    the first click into a viewer that took a minute to start.
    """
    needed = sorted({int(row["dataset_idx"]) for row in rows})
    missing = [
        (i, zarr_paths[i])
        for i in needed
        if i < len(zarr_paths) and not Path(zarr_paths[i]).exists()
    ]
    if not missing:
        return
    lines = "\n".join(f"    dataset_idx={i}: {p}" for i, p in missing)
    raise FileNotFoundError(
        f"{len(missing)} of {len(needed)} zarr path(s) needed here do not exist:\n"
        f"{lines}\n"
        "if you're running off a mounted copy rather than the cluster, pass "
        f"--local-mount (or --zarr-root-map OLD=NEW) to rewrite the prefix "
        "baked into the config."
    )


def temporal_pair(row: dict[str, Any], cfg: Config) -> tuple[int, int]:
    t = int(row["t"])
    gap = int(row["frame_gap"])
    if cfg.temporal_direction == "forward":
        sign = 1
    elif cfg.temporal_direction == "backward":
        sign = -1
    else:
        raise ValueError(f"unknown temporal_direction {cfg.temporal_direction!r}")
    if row["view"] == "t1":
        return t, t + sign * gap
    if row["view"] == "t2":
        return t - sign * gap, t
    raise ValueError(f"unknown view {row['view']!r}")


def validate_dataset_mapping(rows: list[dict[str, Any]], zarr_paths: list[str]) -> None:
    """catch reordered config paths, which would silently show the wrong crops."""
    expected: dict[int, str] = {}
    for row in rows:
        idx = int(row["dataset_idx"])
        name = row["dataset_name"]
        if idx in expected and expected[idx] != name:
            raise ValueError(f"dataset_idx={idx} has multiple names in metadata")
        expected[idx] = name
    for idx, metadata_name in expected.items():
        if not 0 <= idx < len(zarr_paths):
            raise ValueError(f"dataset_idx={idx} is outside configured zarr_paths")
        configured_name = Path(zarr_paths[idx]).name
        if metadata_name and metadata_name != configured_name:
            raise ValueError(
                f"dataset_idx={idx} maps to {metadata_name!r} in metadata but "
                f"{configured_name!r} in the config; use the original path ordering"
            )


def load_raw_crop(
    arrays: dict[int, Any],
    zarr_paths: list[str],
    row: dict[str, Any],
    cfg: Config,
    t: int,
) -> np.ndarray:
    import dask.array as da

    dataset_idx = int(row["dataset_idx"])
    if dataset_idx not in arrays:
        arrays[dataset_idx] = da.from_zarr(f"{zarr_paths[dataset_idx]}/0/0")
    array = arrays[dataset_idx]
    if array.ndim != 5:
        raise ValueError(
            f"expected dataset {dataset_idx} to have TCZYX rank 5; got shape {array.shape}"
        )
    z0, z1 = int(row["z"]), int(row["z_end"])
    y0, x0 = int(row["global_y"]), int(row["global_x"])
    size = cfg.global_crop_size
    expected_shape = (z1 - z0, size, size)
    if z1 <= z0:
        raise ValueError(f"invalid Z interval [{z0}, {z1}) for dataset {dataset_idx}")
    if z0 < 0:
        raise IndexError(f"negative Z origin {z0} for dataset {dataset_idx}")
    if not 0 <= t < array.shape[0]:
        raise IndexError(f"time index {t} is outside dataset {dataset_idx} with T={array.shape[0]}")
    if not 0 <= cfg.channel < array.shape[1]:
        raise IndexError(
            f"channel {cfg.channel} is outside dataset {dataset_idx} with C={array.shape[1]}"
        )
    if y0 < 0 or x0 < 0:
        raise IndexError(f"negative crop origin yx=({y0}, {x0})")
    crop = array[
        t, cfg.channel, z0:z1, y0 : y0 + size, x0 : x0 + size
    ]
    result = np.asarray(crop.compute(), dtype=np.float32)
    if result.shape != expected_shape:
        raise IndexError(
            f"metadata requested crop shape {expected_shape} but dataset {dataset_idx} "
            f"returned {result.shape}; check metadata and config compatibility"
        )
    return result


def robust_limits(*images: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([image.reshape(-1) for image in images])
    finite = values[np.isfinite(values)]
    if not len(finite):
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [1.0, 99.5])
    if lo >= hi:
        lo, hi = float(finite.min()), float(finite.max())
    if lo >= hi:
        hi = lo + 1.0
    return float(lo), float(hi)


def default_embeddings_path(metadata_path: Path) -> Path:
    """embeddings.npz sits one directory above the per-method metadata.csv,
    e.g. latent_space/<run>/umap/metadata.csv -> latent_space/<run>/embeddings.npz
    (see eval/latent_space.py's output layout)."""
    return metadata_path.resolve().parent.parent / "embeddings.npz"


def load_real_embeddings(embeddings_path: Path, coords: np.ndarray) -> np.ndarray:
    """load the real high-dim embeddings written alongside metadata.csv.

    confirms it's the same run and row order by checking that whichever
    projection embeddings.npz saved (coords_umap or coords_pca) matches
    --metadata's own coordinates -- catches a mismatched embeddings.npz early
    instead of silently blending nonsense vectors later.
    """
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"embeddings file not found: {embeddings_path}. pass --embeddings "
            "explicitly if it isn't in the default location."
        )
    data = np.load(embeddings_path)
    if "embeddings" not in data:
        raise ValueError(f"{embeddings_path} has no 'embeddings' array")
    embeddings = data["embeddings"]
    if embeddings.shape[0] != len(coords):
        raise ValueError(
            f"{embeddings_path} has {embeddings.shape[0]} embeddings but "
            f"--metadata has {len(coords)} rows -- mismatched run?"
        )
    for key in ("coords_umap", "coords_pca"):
        if key in data and data[key].shape == coords.shape and np.allclose(data[key], coords, atol=1e-3):
            print(f"[viewer] embeddings.npz verified against metadata via '{key}'")
            return embeddings
    raise ValueError(
        f"neither coords_umap nor coords_pca in {embeddings_path} matches "
        "--metadata's coord_x/coord_y -- likely a mismatched embeddings.npz "
        "(wrong run, or metadata.csv regenerated since)."
    )


def estimate_default_radius(coords: np.ndarray, multiple: float = 3.0) -> float:
    """default Generate-mode search radius: a multiple of the typical nearest-
    neighbor spacing among plotted points. UMAP's output scale is arbitrary
    and PCA's depends on the data's variance, so a fixed constant would be
    wrong for a different run; this adapts to whatever's actually on screen.
    """
    if len(coords) < 2:
        return 1.0
    rng = np.random.default_rng(0)
    sample = rng.choice(len(coords), size=min(500, len(coords)), replace=False)
    nn_dists = []
    for i in sample:
        d = np.linalg.norm(coords - coords[i], axis=1)
        d[i] = np.inf
        nn_dists.append(d.min())
    return float(np.median(nn_dists) * multiple)


def gaussian_radius_blend(
    coords: np.ndarray,
    target: tuple[float, float],
    radius: float,
    visible_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """indices + normalized weights of real points to blend for a Generate click.

    only points within `radius` are considered at all; within that cutoff,
    weight falls off as a gaussian with radius as its ~2-sigma width, so nearby
    points dominate the blend. this deliberately never pads out to a minimum
    neighbor count with distant, less-relevant points -- if nothing real is
    within radius, it falls back to the single nearest point and returns a
    confidence string that says so plainly, rather than quietly blending in
    whatever's closest regardless of how far that actually is.

    returns (indices, weights, confidence); weights sum to 1 and are sorted
    descending (closest / highest-weight first).
    """
    distances = np.linalg.norm(coords - np.asarray(target)[None, :], axis=1)
    if visible_mask is not None:
        distances = np.where(visible_mask, distances, np.inf)

    within = np.flatnonzero(distances <= radius)
    if len(within) == 0:
        nearest = int(np.argmin(distances))
        nearest_dist = float(distances[nearest])
        confidence = (
            f"LOW CONFIDENCE -- nearest real point is {nearest_dist:.3f} away, "
            f"outside the search radius ({radius:.3f}). extrapolating from "
            f"that one point only."
        )
        return np.asarray([nearest]), np.asarray([1.0]), confidence

    bandwidth = radius / 2.0
    weights = np.exp(-0.5 * (distances[within] / bandwidth) ** 2)
    weights = weights / weights.sum()
    order = np.argsort(-weights)

    if len(within) <= 2:
        confidence = f"moderate confidence -- only {len(within)} real point(s) within radius {radius:.3f}"
    else:
        confidence = f"{len(within)} real points within radius {radius:.3f} contributed to this blend"
    return within[order], weights[order], confidence


class LatentCropViewer:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        coords: np.ndarray,
        cfg: Config,
        zarr_paths: list[str],
        nearest: int,
        max_crops: int,
        point_size: float,
        embeddings: np.ndarray | None = None,
        decoder_checkpoint: str | None = None,
        gen_radius: float | None = None,
    ) -> None:
        self.rows = rows
        self.coords = coords
        self.cfg = cfg
        self.zarr_paths = zarr_paths
        self.nearest = nearest
        self.max_crops = max_crops
        self.arrays: dict[int, Any] = {}

        # Generate mode: optional, only wired up when a decoder checkpoint is
        # given. torch import is lazy so Inspect-only usage never pays for it.
        self.embeddings = embeddings
        self.gen_radius = gen_radius
        self.mode = "inspect"
        self.decoder = None
        if decoder_checkpoint is not None:
            import torch

            from eval.decoder import LatentDecoder

            ckpt = torch.load(decoder_checkpoint, map_location="cpu", weights_only=True)
            self.decoder = LatentDecoder(
                latent_dim=ckpt["latent_dim"],
                crop_shape=tuple(ckpt["crop_shape"]),
                patch_shape=tuple(ckpt["patch_shape"]),
                hidden_dim=ckpt["hidden_dim"],
                depth=ckpt["depth"],
            )
            self.decoder.load_state_dict(ckpt["decoder_state_dict"])
            self.decoder.eval()
            print(f"[viewer] loaded decoder from {decoder_checkpoint}; Generate mode available")

        self.map_fig, self.map_ax = plt.subplots(figsize=(11, 8))
        self.map_fig.subplots_adjust(bottom=0.2, right=0.76)
        dataset_ids = np.asarray([int(row["dataset_idx"]) for row in rows])
        cmap = plt.get_cmap("tab20")
        self.dataset_artists: list[tuple[Any, np.ndarray]] = []
        for color_idx, dataset_idx in enumerate(sorted(set(dataset_ids))):
            mask = dataset_ids == dataset_idx
            dataset_name = next(
                row["dataset_name"]
                for row in rows
                if int(row["dataset_idx"]) == dataset_idx
            )
            artist = self.map_ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                color=cmap(color_idx % 20),
                label=f"{dataset_idx}: {dataset_name[:28]}",
                s=point_size,
                alpha=0.65,
                linewidths=0,
                picker=False,
                zorder=1,
            )
            self.dataset_artists.append((artist, mask))

        # annotated points get a colored ring on top of their dataset dot,
        # colored by behavior label so both dataset and class are visible
        # at once. "label" is an optional metadata.csv column (written by an
        # annotation-aware latent_space.py run); older metadata.csv files
        # without it just skip this overlay entirely.
        labels = np.asarray([row.get("label") or "" for row in rows])
        label_values = sorted(v for v in set(labels.tolist()) if v)
        label_cmap = plt.get_cmap("Set1")
        self.label_artists: list[tuple[Any, np.ndarray]] = []
        for label_idx, label in enumerate(label_values):
            mask = labels == label
            artist = self.map_ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=point_size * 3.0,
                facecolors="none",
                edgecolors=label_cmap(label_idx % 9),
                linewidths=1.3,
                label=f"[annotated] {label}",
                zorder=2,
            )
            self.label_artists.append((artist, mask))

        self.highlight = self.map_ax.scatter(
            [], [], s=65, facecolors="none", edgecolors="black", linewidths=1.2, zorder=3
        )
        self.map_ax.set(
            title="Latent crop viewer",
            xlabel="UMAP 1",
            ylabel="UMAP 2",
        )
        self.map_ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=7,
            frameon=False,
            markerscale=2,
        )
        status_text = "Left-click: nearest crops | Right-click + drag: lasso | C: clear"
        if label_values:
            status_text += f" | rings = annotated ({', '.join(label_values)})"
        self.status = self.map_fig.text(0.12, 0.035, status_text, fontsize=9)
        clear_ax = self.map_fig.add_axes((0.83, 0.025, 0.11, 0.045))
        self.clear_button = Button(clear_ax, "Clear")
        self.clear_button.on_clicked(self.clear)

        # show/hide checkboxes, on by default; hidden points are also
        # excluded from click and lasso selection (see select_click_indices).
        self.show_annotated = True
        self.show_unannotated = True
        self.visible_mask = compute_visible_mask(rows, True, True)
        visibility_ax = self.map_fig.add_axes((0.83, 0.085, 0.15, 0.095))
        self.visibility_check = CheckButtons(
            visibility_ax, ["Annotated", "Unannotated"], [True, True]
        )
        self.visibility_check.on_clicked(self._on_visibility_toggle)

        # Inspect/Generate toggle -- only shown when a decoder was loaded.
        self.mode_radio = None
        if self.decoder is not None:
            mode_ax = self.map_fig.add_axes((0.83, 0.19, 0.15, 0.09))
            self.mode_radio = RadioButtons(mode_ax, ["Inspect", "Generate"])
            self.mode_radio.on_clicked(self._on_mode_change)

        self.crop_fig = None
        self._create_crop_figure()
        self.lasso = LassoSelector(self.map_ax, self.on_lasso, button=3)
        self.map_fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.map_fig.canvas.mpl_connect("key_press_event", self.on_key)

    def _create_crop_figure(self) -> None:
        """create the crop window and stop retaining it after its GUI is closed."""
        self.crop_fig = plt.figure(figsize=(12, 8))
        self.crop_fig.suptitle("Select points in the latent map")
        self.crop_fig.canvas.mpl_connect("close_event", self._on_crop_figure_closed)

    def _on_crop_figure_closed(self, _event: Any) -> None:
        self.crop_fig = None

    def _ensure_crop_figure(self) -> None:
        if self.crop_fig is None or not plt.fignum_exists(self.crop_fig.number):
            self._create_crop_figure()

    def set_status(self, message: str) -> None:
        self.status.set_text(message)
        self.map_fig.canvas.draw_idle()

    def _on_visibility_toggle(self, label: str) -> None:
        if label == "Annotated":
            self.show_annotated = not self.show_annotated
        elif label == "Unannotated":
            self.show_unannotated = not self.show_unannotated
        self._update_visibility()

    def _update_visibility(self) -> None:
        self.visible_mask = compute_visible_mask(
            self.rows, self.show_annotated, self.show_unannotated
        )
        for artist, dataset_mask in self.dataset_artists:
            idx = np.flatnonzero(dataset_mask & self.visible_mask)
            artist.set_offsets(self.coords[idx] if len(idx) else np.empty((0, 2)))
        for artist, label_mask in self.label_artists:
            idx = np.flatnonzero(label_mask) if self.show_annotated else np.empty(0, dtype=np.int64)
            artist.set_offsets(self.coords[idx] if len(idx) else np.empty((0, 2)))
        self.map_fig.canvas.draw_idle()

    def _on_mode_change(self, label: str) -> None:
        self.mode = label.lower()
        msg = f"Mode: {label}"
        if self.mode == "generate":
            msg += " -- click anywhere to decode a blended crop"
        self.set_status(msg)

    def on_click(self, event: Any) -> None:
        if event.inaxes is not self.map_ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        toolbar = getattr(self.map_fig.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return

        if self.mode == "generate":
            self.handle_generate_click((event.xdata, event.ydata))
            return

        indices = select_click_indices(
            self.coords,
            self.rows,
            self.cfg,
            (event.xdata, event.ydata),
            self.nearest,
            self.visible_mask,
        )
        if not len(indices):
            self.set_status("No visible points near click")
            return
        self.show_selection(indices, f"nearest {len(indices)} to click")

    def on_lasso(self, vertices: list[tuple[float, float]]) -> None:
        selected = indices_inside_polygon(self.coords, vertices)
        selected = selected[self.visible_mask[selected]]
        if not len(selected):
            self.set_status("Lasso contained no visible points")
            return
        unique = unique_crop_indices(selected, self.rows, self.cfg)
        shown = representative_indices(self.coords, unique, self.max_crops)
        detail = (
            f"lasso selected {len(selected)} points / {len(unique)} unique crops; "
            f"displaying {len(shown)}"
        )
        if len(unique) > len(shown):
            detail += " nearest the lasso centroid"
        self.show_selection(shown, detail)

    def on_key(self, event: Any) -> None:
        if event.key and event.key.lower() == "c":
            self.clear()

    def clear(self, _event: Any = None) -> None:
        self.highlight.set_offsets(np.empty((0, 2)))
        if self.crop_fig is not None and plt.fignum_exists(self.crop_fig.number):
            self.crop_fig.clear()
            self.crop_fig.suptitle("Select points in the latent map")
            self.crop_fig.canvas.draw_idle()
        self.set_status("Selection cleared")

    def show_selection(self, indices: np.ndarray, description: str) -> None:
        indices = np.asarray(indices, dtype=np.int64)
        self.highlight.set_offsets(self.coords[indices])
        self.map_fig.canvas.draw_idle()
        self.set_status(f"Loading {description} …")
        try:
            self.render_crops(indices, description)
        except Exception as exc:
            self.set_status(f"Could not load crops: {exc}")
            traceback.print_exc()
            return
        self.set_status(description)

    def render_crops(self, indices: np.ndarray, description: str) -> None:
        self._ensure_crop_figure()
        assert self.crop_fig is not None
        self.crop_fig.clear()
        n = len(indices)
        if n < 1:
            raise ValueError("cannot render an empty selection")
        self.crop_fig.set_size_inches(min(18.0, max(4.0, 2.2 * n)), 9.0, forward=True)
        axes = self.crop_fig.subplots(4, n, squeeze=False)
        row_titles = ["t1 mean-Z", "t2 mean-Z", "t1 max-Z", "t2 max-Z"]

        for col, idx in enumerate(indices):
            row = self.rows[int(idx)]
            t1, t2 = temporal_pair(row, self.cfg)
            crop_t1 = load_raw_crop(self.arrays, self.zarr_paths, row, self.cfg, t1)
            crop_t2 = load_raw_crop(self.arrays, self.zarr_paths, row, self.cfg, t2)
            mean_t1, mean_t2 = crop_t1.mean(axis=0), crop_t2.mean(axis=0)
            max_t1, max_t2 = crop_t1.max(axis=0), crop_t2.max(axis=0)
            images = [mean_t1, mean_t2, max_t1, max_t2]
            mean_limits = robust_limits(mean_t1, mean_t2)
            max_limits = robust_limits(max_t1, max_t2)

            for row_idx, image in enumerate(images):
                ax = axes[row_idx, col]
                limits = mean_limits if row_idx < 2 else max_limits
                ax.imshow(image, cmap="gray", vmin=limits[0], vmax=limits[1])
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(row_titles[row_idx], fontsize=8)
            title = (
                f"point {idx} | ds={row['dataset_idx']}\n"
                f"{row['dataset_name'][:22]}\n"
                f"t=({t1},{t2}) z=[{row['z']},{row['z_end']})\n"
                f"yx=({row['global_y']},{row['global_x']})"
            )
            label = row.get("label")
            title += f"\nANNOTATED: {label}" if label else "\nnot annotated"
            axes[0, col].set_title(title, fontsize=7)

        self.crop_fig.suptitle(description, fontsize=12)
        self.crop_fig.tight_layout(rect=(0, 0, 1, 0.95))
        self.crop_fig.canvas.draw_idle()
        self.crop_fig.show()

    def handle_generate_click(self, target: tuple[float, float]) -> None:
        assert self.decoder is not None and self.embeddings is not None
        indices, weights, confidence = gaussian_radius_blend(
            self.coords, target, self.gen_radius, self.visible_mask
        )
        self.highlight.set_offsets(self.coords[indices])
        self.map_fig.canvas.draw_idle()
        self.set_status("Generating… " + confidence)
        try:
            self.render_generated(indices, weights, confidence)
        except Exception as exc:
            self.set_status(f"Could not generate: {exc}")
            traceback.print_exc()
            return
        self.set_status(confidence)

    def render_generated(
        self, indices: np.ndarray, weights: np.ndarray, confidence: str
    ) -> None:
        import torch

        blended = (weights[:, None] * self.embeddings[indices]).sum(axis=0)
        z = torch.from_numpy(blended).float().unsqueeze(0)
        with torch.no_grad():
            volume = self.decoder(z).squeeze(0).squeeze(0).numpy()  # (Z, H, W)

        self._ensure_crop_figure()
        assert self.crop_fig is not None
        self.crop_fig.clear()

        # show the generated crop plus up to 4 of its highest-weight real
        # neighbors, for a side-by-side sanity check against real data.
        shown = indices[: min(len(indices), 4)]
        shown_weights = weights[: len(shown)]
        n_cols = 1 + len(shown)
        self.crop_fig.set_size_inches(min(18.0, max(4.0, 2.2 * n_cols)), 6.0, forward=True)
        axes = self.crop_fig.subplots(2, n_cols, squeeze=False)

        gen_mean, gen_max = volume.mean(axis=0), volume.max(axis=0)
        mean_lo, mean_hi = robust_limits(gen_mean)
        max_lo, max_hi = robust_limits(gen_max)
        axes[0, 0].imshow(gen_mean, cmap="gray", vmin=mean_lo, vmax=mean_hi)
        axes[1, 0].imshow(gen_max, cmap="gray", vmin=max_lo, vmax=max_hi)
        axes[0, 0].set_ylabel("mean-Z", fontsize=8)
        axes[1, 0].set_ylabel("max-Z", fontsize=8)
        axes[0, 0].set_title("GENERATED\n(decoded blend)", fontsize=8, fontweight="bold")

        for col, (idx, weight) in enumerate(zip(shown, shown_weights), start=1):
            row = self.rows[int(idx)]
            t = int(row["t"])
            crop = load_raw_crop(self.arrays, self.zarr_paths, row, self.cfg, t)
            mean_img, max_img = crop.mean(axis=0), crop.max(axis=0)
            m_lo, m_hi = robust_limits(mean_img)
            x_lo, x_hi = robust_limits(max_img)
            axes[0, col].imshow(mean_img, cmap="gray", vmin=m_lo, vmax=m_hi)
            axes[1, col].imshow(max_img, cmap="gray", vmin=x_lo, vmax=x_hi)
            title = (
                f"real, weight={weight * 100:.0f}%\n"
                f"ds={row['dataset_idx']} {row['dataset_name'][:16]}"
            )
            axes[0, col].set_title(title, fontsize=7)

        for ax_row in axes:
            for ax in ax_row:
                ax.set_xticks([])
                ax.set_yticks([])

        self.crop_fig.suptitle(confidence, fontsize=10)
        self.crop_fig.tight_layout(rect=(0, 0, 1, 0.93))
        self.crop_fig.canvas.draw_idle()
        self.crop_fig.show()


def main() -> None:
    args = parse_args()
    if args.nearest < 1 or args.max_crops < 1:
        raise ValueError("--nearest and --max-crops must be at least 1")
    if args.point_size <= 0:
        raise ValueError("--point-size must be greater than 0")

    cfg = Config.from_yaml(args.config)
    rows, coords = read_metadata(Path(args.metadata))
    zarr_paths = resolve_zarr_paths(cfg, args)
    validate_dataset_mapping(rows, zarr_paths)
    require_zarrs_exist(zarr_paths, rows)

    embeddings = None
    gen_radius = args.gen_radius
    if args.decoder_checkpoint is not None:
        embeddings_path = (
            Path(args.embeddings) if args.embeddings else default_embeddings_path(Path(args.metadata))
        )
        print(f"[viewer] loading real embeddings from {embeddings_path}")
        embeddings = load_real_embeddings(embeddings_path, coords)
        if gen_radius is None:
            gen_radius = estimate_default_radius(coords)
        print(f"[viewer] Generate mode search radius: {gen_radius:.4f}")

    print(f"[viewer] loaded {len(rows)} latent points")
    print("[viewer] left-click for nearest points; right-click-drag to lasso; C to clear")
    viewer = LatentCropViewer(
        rows,
        coords,
        cfg,
        zarr_paths,
        nearest=args.nearest,
        max_crops=args.max_crops,
        point_size=args.point_size,
        embeddings=embeddings,
        decoder_checkpoint=args.decoder_checkpoint,
        gen_radius=gen_radius,
    )
    # Keep the viewer and its weakly-referenced Matplotlib callbacks alive until
    # all GUI windows have closed.
    plt.show()
    del viewer


if __name__ == "__main__":
    main()
