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
from matplotlib.widgets import Button, CheckButtons, LassoSelector

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

    def on_click(self, event: Any) -> None:
        if event.inaxes is not self.map_ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        toolbar = getattr(self.map_fig.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
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
    viewer = LatentCropViewer(
        rows,
        coords,
        cfg,
        zarr_paths,
        nearest=args.nearest,
        max_crops=args.max_crops,
        point_size=args.point_size,
    )
    # Keep the viewer and its weakly-referenced Matplotlib callbacks alive until
    # all GUI windows have closed.
    plt.show()
    del viewer


if __name__ == "__main__":
    main()
