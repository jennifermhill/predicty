"""Interactively inspect the raw crops represented by points in a 2D latent map.

Left-click the scatter plot to show the nearest points. Right-click and drag to
lasso a region. The selected points are highlighted in the latent map and their
exact temporal crops are rendered in a second window.

Example:
    python eval/interactive_latent_viewer.py \
        --config configs/train.yaml \
        --metadata latent_space/latest/metadata.csv
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
from matplotlib.widgets import Button, LassoSelector

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
    """Keep crops nearest the selected region's centroid when it is very large."""
    if max_crops < 1:
        raise ValueError("max_crops must be at least 1")
    selected = np.asarray(selected, dtype=np.int64)
    if len(selected) <= max_crops:
        return selected
    centroid = coords[selected].mean(axis=0)
    distance = np.linalg.norm(coords[selected] - centroid[None, :], axis=1)
    return selected[np.argsort(distance)[:max_crops]]


def unique_crop_indices(
    indices: np.ndarray,
    rows: list[dict[str, Any]],
    cfg: Config,
) -> np.ndarray:
    """Remove duplicate t1/t2 latent points that reload the same crop pair."""
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
    """Catch reordered config paths, which would silently show the wrong crops."""
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
    ) -> None:
        self.rows = rows
        self.coords = coords
        self.cfg = cfg
        self.zarr_paths = zarr_paths
        self.nearest = nearest
        self.max_crops = max_crops
        self.arrays: dict[int, Any] = {}

        self.map_fig, self.map_ax = plt.subplots(figsize=(11, 8))
        self.map_fig.subplots_adjust(bottom=0.13, right=0.76)
        dataset_ids = np.asarray([int(row["dataset_idx"]) for row in rows])
        cmap = plt.get_cmap("tab20")
        for color_idx, dataset_idx in enumerate(sorted(set(dataset_ids))):
            mask = dataset_ids == dataset_idx
            dataset_name = next(
                row["dataset_name"]
                for row in rows
                if int(row["dataset_idx"]) == dataset_idx
            )
            self.map_ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                color=cmap(color_idx % 20),
                label=f"{dataset_idx}: {dataset_name[:28]}",
                s=point_size,
                alpha=0.65,
                linewidths=0,
                picker=False,
            )
        self.highlight = self.map_ax.scatter(
            [], [], s=65, facecolors="none", edgecolors="black", linewidths=1.2
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
        self.status = self.map_fig.text(
            0.12,
            0.035,
            "Left-click: nearest crops | Right-click + drag: lasso | C: clear",
            fontsize=9,
        )
        clear_ax = self.map_fig.add_axes((0.83, 0.025, 0.11, 0.045))
        self.clear_button = Button(clear_ax, "Clear")
        self.clear_button.on_clicked(self.clear)

        self.crop_fig = plt.figure(figsize=(12, 8))
        self.crop_fig.suptitle("Select points in the latent map")
        self.lasso = LassoSelector(self.map_ax, self.on_lasso, button=3)
        self.map_fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.map_fig.canvas.mpl_connect("key_press_event", self.on_key)

    def set_status(self, message: str) -> None:
        self.status.set_text(message)
        self.map_fig.canvas.draw_idle()

    def on_click(self, event: Any) -> None:
        if event.inaxes is not self.map_ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        toolbar = getattr(self.map_fig.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return
        ranked = nearest_indices(self.coords, (event.xdata, event.ydata), len(self.coords))
        indices = unique_crop_indices(ranked, self.rows, self.cfg)[: self.nearest]
        self.show_selection(indices, f"nearest {len(indices)} to click")

    def on_lasso(self, vertices: list[tuple[float, float]]) -> None:
        selected = indices_inside_polygon(self.coords, vertices)
        if not len(selected):
            self.set_status("Lasso contained no points")
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
            axes[0, col].set_title(
                f"point {idx} | ds={row['dataset_idx']}\n"
                f"{row['dataset_name'][:22]}\n"
                f"t=({t1},{t2}) z=[{row['z']},{row['z_end']})\n"
                f"yx=({row['global_y']},{row['global_x']})",
                fontsize=7,
            )

        self.crop_fig.suptitle(description, fontsize=12)
        self.crop_fig.tight_layout(rect=(0, 0, 1, 0.95))
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
    zarr_paths = zarr_paths_from_config(cfg)
    validate_dataset_mapping(rows, zarr_paths)

    print(f"[viewer] loaded {len(rows)} latent points")
    print("[viewer] left-click for nearest points; right-click-drag to lasso; C to clear")
    LatentCropViewer(
        rows,
        coords,
        cfg,
        zarr_paths,
        nearest=args.nearest,
        max_crops=args.max_crops,
        point_size=args.point_size,
    )
    plt.show()


if __name__ == "__main__":
    main()
