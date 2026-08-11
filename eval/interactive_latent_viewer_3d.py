"""the viewer shows the raw crops for the points in a 3D latent map.

the module is the 3D counterpart of eval/interactive_latent_viewer.py. see that
module for the shared workflow, the label rings, the checkboxes and Generate
mode. this module reads the coord_x3/coord_y3/coord_z3 columns from
latent_space.py's 3D fit, and it leaves the 2D coord_x/coord_y alone.

only the mouse behavior differs, because matplotlib's 3D axes already claim the
drags for the camera and there is no 3D lasso:

    left-drag            rotate the camera (matplotlib's own behavior)
    right-drag           zoom (matplotlib's own behavior)
    middle-drag          pan
    left-click           nearest crops to the click
    right-click          every point within --pick-radius pixels of the click
    C                    clear the selection

a "click" is a press and release pair that moved less than --click-tolerance
pixels. a rotation and a zoom therefore never select. the selection works in
*screen space* at the current camera angle. screen space is the only measure of
"near what I clicked" that matches what you see. after a rotation, the same
click picks different points.

in Generate mode a 2D click is a ray and not a point. the viewer therefore
unprojects the click at the depth of the nearest visible real point. you
generate on the visible surface of the cloud, and not at an arbitrary position
along the line of sight. the blend then uses true 3D distances.

Example:
    python eval/interactive_latent_viewer_3d.py \
        --config configs/train.yaml \
        --metadata latent_space/latest/umap/metadata.csv

    # with Generate mode enabled:
    python eval/interactive_latent_viewer_3d.py \
        --config configs/train.yaml \
        --metadata latent_space/latest/umap/metadata.csv \
        --decoder-checkpoint latent_space/latest/latent_decoder.pt
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons, RadioButtons
from mpl_toolkits.mplot3d import proj3d  # noqa: F401  (registers the 3d projection)

from configs.config import Config
from eval.interactive_latent_viewer import (
    LatentCropViewer,
    REQUIRED_COLUMNS,
    add_zarr_remap_args,
    compute_visible_mask,
    default_embeddings_path,
    estimate_default_radius,
    gaussian_radius_blend,
    nearest_annotated_index,
    nearest_indices,
    representative_indices,
    require_zarrs_exist,
    resolve_zarr_paths,
    unique_crop_indices,
    validate_dataset_mapping,
)

COORD_3D_COLUMNS = ("coord_x3", "coord_y3", "coord_z3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Click a 3D latent map to inspect exact raw crops."
    )
    parser.add_argument("--config", required=True, help="training YAML used for extraction")
    parser.add_argument(
        "--metadata",
        required=True,
        help="metadata.csv from latent_space.py (needs coord_x3/coord_y3/coord_z3)",
    )
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
        help="maximum crops displayed from one right-click region selection",
    )
    parser.add_argument("--point-size", type=float, default=7.0)
    parser.add_argument(
        "--pick-radius",
        type=float,
        default=25.0,
        help="right-click region radius in screen pixels",
    )
    parser.add_argument(
        "--click-tolerance",
        type=float,
        default=5.0,
        help=(
            "a press/release pair that moved less than this many pixels counts "
            "as a click rather than a camera rotate/zoom"
        ),
    )
    parser.add_argument(
        "--decoder-checkpoint",
        default=None,
        help=(
            "optional latent_decoder.pt (from eval/decoder.py). enables a "
            "Generate mode toggle that decodes a blend of nearby real "
            "embeddings; without it the viewer is inspect-only."
        ),
    )
    parser.add_argument(
        "--embeddings",
        default=None,
        help=(
            "embeddings.npz matching --metadata (default: one directory above "
            "--metadata). only used when --decoder-checkpoint is given."
        ),
    )
    parser.add_argument(
        "--gen-radius",
        type=float,
        default=None,
        help=(
            "3D search radius for blending nearby real embeddings in Generate "
            "mode, in coord_x3/coord_y3/coord_z3 units. default: 3x the median "
            "nearest-neighbor spacing among plotted points."
        ),
    )
    add_zarr_remap_args(parser)
    return parser.parse_args()


def read_metadata_3d(path: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    """rows plus the (N, 3) coordinate array from the 3D fit.

    the function prefers coord_x3/coord_y3/coord_z3. the function falls back to
    a plain coord_x/coord_y/coord_z trio, so that an externally projected
    metadata.csv works.
    """
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"metadata is empty: {path}")

    # coord_x/coord_y are part of REQUIRED_COLUMNS, but they describe the 2D
    # fit. the code checks the 3D coords separately below.
    missing = (REQUIRED_COLUMNS - {"coord_x", "coord_y"}).difference(rows[0])
    if missing:
        raise ValueError(f"metadata is missing required columns: {sorted(missing)}")

    if all(name in rows[0] for name in COORD_3D_COLUMNS):
        names = COORD_3D_COLUMNS
    elif all(name in rows[0] for name in ("coord_x", "coord_y", "coord_z")):
        names = ("coord_x", "coord_y", "coord_z")
    else:
        raise ValueError(
            f"{path} has no 3D coordinates (looked for {list(COORD_3D_COLUMNS)}). "
            "re-run eval/latent_space.py for this run to add them -- it reuses "
            "the cached embeddings.npz, so only the projection is recomputed."
        )

    coords = np.asarray(
        [[float(row[names[0]]), float(row[names[1]]), float(row[names[2]])] for row in rows],
        dtype=np.float64,
    )
    if not np.isfinite(coords).all():
        raise ValueError("metadata contains non-finite latent coordinates")
    return rows, coords


def load_real_embeddings_3d(embeddings_path: Path, coords: np.ndarray) -> np.ndarray:
    """the function loads the high-dim embeddings and confirms that they belong
    to this metadata.csv.

    the guard is the same as in interactive_latent_viewer.load_real_embeddings.
    the function matches against the coords_<method>_3d arrays instead.
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
    for key in data.files:
        if not key.endswith("_3d"):
            continue
        if data[key].shape == coords.shape and np.allclose(data[key], coords, atol=1e-3):
            print(f"[viewer3d] embeddings.npz verified against metadata via '{key}'")
            return embeddings
    raise ValueError(
        f"no coords_*_3d array in {embeddings_path} matches --metadata's 3D "
        "coordinates -- likely a mismatched embeddings.npz (wrong run, or "
        "metadata.csv regenerated since)."
    )


def axis_labels(metadata_path: Path) -> tuple[str, str, str]:
    """the function labels the axes after the projection method.

    latent_space.py writes <run>/<method>/metadata.csv, so the name of the
    parent directory is the method (umap, pca, tsne).
    """
    method = metadata_path.resolve().parent.name.upper()
    if not method or len(method) > 8:
        method = "LATENT"
    return f"{method} 1", f"{method} 2", f"{method} 3"


def project_to_screen(ax: Any, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N, 2) display-pixel positions and (N,) camera depths for 3D points.

    the function repeats what matplotlib's own 3D artists do: ax.get_proj(),
    then transData. the result is exactly the pixels of the drawn points. a
    screen-space selection therefore agrees with what the user sees.
    """
    M = ax.get_proj()
    xs, ys, zs = proj3d.proj_transform(coords[:, 0], coords[:, 1], coords[:, 2], M)
    display = ax.transData.transform(np.column_stack([xs, ys]))
    return np.asarray(display, dtype=np.float64), np.asarray(zs, dtype=np.float64)


def unproject_from_screen(ax: Any, x: float, y: float, depth: float) -> np.ndarray:
    """3D data point that projects to display pixel (x, y) at camera `depth`.

    a click is only a ray, so the depth must come from somewhere. the caller
    passes the depth of the nearest real point. the code writes the math out and
    does not call proj3d.inv_transform, whose signature moved between matplotlib
    versions.
    """
    xp, yp = ax.transData.inverted().transform((x, y))
    inv_M = np.linalg.inv(ax.get_proj())
    vec = inv_M @ np.array([xp, yp, depth, 1.0], dtype=np.float64)
    if abs(vec[3]) < 1e-12:
        raise ValueError("degenerate projection; try rotating the view slightly")
    return vec[:3] / vec[3]


def set_3d_offsets(artist: Any, points: np.ndarray) -> None:
    """the function moves the points of a 3D scatter, down to zero points.

    Path3DCollection has no public setter for its 3D offsets. the code creates
    every artist with depthshade=False and one scalar color. a change of the
    point count therefore cannot desync a per-point color array.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    artist._offsets3d = (points[:, 0], points[:, 1], points[:, 2])


def select_click_indices_screen(
    display: np.ndarray,
    rows: list[dict[str, Any]],
    cfg: Config,
    target: tuple[float, float],
    nearest: int,
    visible_mask: np.ndarray | None = None,
) -> np.ndarray:
    """screen-space version of interactive_latent_viewer.select_click_indices.
    the logic is identical, but the rank comes from the pixel distance at the
    current camera angle."""
    ranked = nearest_indices(display, target, len(display))
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


def indices_within_pixels(
    display: np.ndarray,
    target: tuple[float, float],
    radius: float,
    visible_mask: np.ndarray | None = None,
) -> np.ndarray:
    """visible points within `radius` screen pixels of a click."""
    distances = np.linalg.norm(display - np.asarray(target)[None, :], axis=1)
    if visible_mask is not None:
        distances = np.where(visible_mask, distances, np.inf)
    return np.flatnonzero(distances <= radius)


class LatentCropViewer3D(LatentCropViewer):
    """3D map window. the crop render and the Generate decode come from the 2D viewer.

    __init__ deliberately skips super().__init__, because super().__init__
    builds the 2D figure. __init__ sets the same attribute names, so the
    inherited render_crops and render_generated plumbing works without a change.
    the class overrides only the methods that touch map coordinates.
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        coords: np.ndarray,
        cfg: Config,
        zarr_paths: list[str],
        nearest: int,
        max_crops: int,
        point_size: float,
        pick_radius: float,
        click_tolerance: float,
        labels_xyz: tuple[str, str, str],
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
        self.pick_radius = pick_radius
        self.click_tolerance = click_tolerance
        self.arrays: dict[int, Any] = {}
        self._press: tuple[int, float, float] | None = None

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
            print(
                f"[viewer3d] loaded decoder from {decoder_checkpoint}; "
                "Generate mode available"
            )

        self.map_fig = plt.figure(figsize=(11, 8))
        self.map_ax = self.map_fig.add_subplot(projection="3d")
        self.map_fig.subplots_adjust(bottom=0.2, right=0.76)

        dataset_ids = np.asarray([int(row["dataset_idx"]) for row in rows])
        cmap = plt.get_cmap("tab20")
        self.dataset_artists: list[tuple[Any, np.ndarray]] = []
        for color_idx, dataset_idx in enumerate(sorted(set(dataset_ids.tolist()))):
            mask = dataset_ids == dataset_idx
            dataset_name = next(
                row["dataset_name"] for row in rows if int(row["dataset_idx"]) == dataset_idx
            )
            artist = self.map_ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                coords[mask, 2],
                color=cmap(color_idx % 20),
                label=f"{dataset_idx}: {dataset_name[:28]}",
                s=point_size,
                alpha=0.5,
                linewidths=0,
                depthshade=False,
            )
            self.dataset_artists.append((artist, mask))

        labels = np.asarray([row.get("label") or "" for row in rows])
        label_values = sorted(v for v in set(labels.tolist()) if v)
        label_cmap = plt.get_cmap("Set1")
        self.label_artists: list[tuple[Any, np.ndarray]] = []
        for label_idx, label in enumerate(label_values):
            mask = labels == label
            artist = self.map_ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                coords[mask, 2],
                s=point_size * 3.0,
                facecolors="none",
                edgecolors=label_cmap(label_idx % 9),
                linewidths=1.3,
                label=f"[annotated] {label}",
                depthshade=False,
            )
            self.label_artists.append((artist, mask))

        self.highlight = self.map_ax.scatter(
            [], [], [], s=65, facecolors="none", edgecolors="black", linewidths=1.2,
            depthshade=False,
        )

        self.map_ax.set_title("Latent crop viewer (3D)")
        self.map_ax.set_xlabel(labels_xyz[0])
        self.map_ax.set_ylabel(labels_xyz[1])
        self.map_ax.set_zlabel(labels_xyz[2])
        self.map_ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=7,
            frameon=False,
            markerscale=2,
        )

        status_text = (
            "Left-drag: rotate | Left-click: nearest crops | "
            "Right-click: region | C: clear"
        )
        if label_values:
            status_text += f" | rings = annotated ({', '.join(label_values)})"
        self.status = self.map_fig.text(0.06, 0.035, status_text, fontsize=9)

        clear_ax = self.map_fig.add_axes((0.83, 0.025, 0.11, 0.045))
        self.clear_button = Button(clear_ax, "Clear")
        self.clear_button.on_clicked(self.clear)

        self.show_annotated = True
        self.show_unannotated = True
        self.visible_mask = compute_visible_mask(rows, True, True)
        visibility_ax = self.map_fig.add_axes((0.83, 0.085, 0.15, 0.095))
        self.visibility_check = CheckButtons(
            visibility_ax, ["Annotated", "Unannotated"], [True, True]
        )
        self.visibility_check.on_clicked(self._on_visibility_toggle)

        self.mode_radio = None
        if self.decoder is not None:
            mode_ax = self.map_fig.add_axes((0.83, 0.19, 0.15, 0.09))
            self.mode_radio = RadioButtons(mode_ax, ["Inspect", "Generate"])
            self.mode_radio.on_clicked(self._on_mode_change)

        self.crop_fig = None
        self._create_crop_figure()
        # there is no LassoSelector in 3D, and the left-drag belongs to the
        # camera. the code therefore pairs press and release manually, to tell a
        # click from a drag.
        self.map_fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.map_fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.map_fig.canvas.mpl_connect("key_press_event", self.on_key)

    # --- map-coordinate overrides -------------------------------------------

    def _update_visibility(self) -> None:
        self.visible_mask = compute_visible_mask(
            self.rows, self.show_annotated, self.show_unannotated
        )
        for artist, dataset_mask in self.dataset_artists:
            idx = np.flatnonzero(dataset_mask & self.visible_mask)
            set_3d_offsets(artist, self.coords[idx])
        for artist, label_mask in self.label_artists:
            idx = np.flatnonzero(label_mask) if self.show_annotated else np.empty(0, dtype=np.int64)
            set_3d_offsets(artist, self.coords[idx])
        self.map_fig.canvas.draw_idle()

    def clear(self, _event: Any = None) -> None:
        set_3d_offsets(self.highlight, np.empty((0, 3)))
        if self.crop_fig is not None and plt.fignum_exists(self.crop_fig.number):
            self.crop_fig.clear()
            self.crop_fig.suptitle("Select points in the latent map")
            self.crop_fig.canvas.draw_idle()
        self.set_status("Selection cleared")

    def show_selection(self, indices: np.ndarray, description: str) -> None:
        import traceback

        indices = np.asarray(indices, dtype=np.int64)
        set_3d_offsets(self.highlight, self.coords[indices])
        self.map_fig.canvas.draw_idle()
        self.set_status(f"Loading {description} …")
        try:
            self.render_crops(indices, description)
        except Exception as exc:
            self.set_status(f"Could not load crops: {exc}")
            traceback.print_exc()
            return
        self.set_status(description)

    # --- mouse events --------------------------------------------------------

    def on_press(self, event: Any) -> None:
        if event.inaxes is not self.map_ax or event.x is None or event.y is None:
            self._press = None
            return
        self._press = (event.button, float(event.x), float(event.y))

    def on_release(self, event: Any) -> None:
        press, self._press = self._press, None
        if press is None or event.x is None or event.y is None:
            return
        button, px, py = press
        if event.button != button:
            return
        moved = float(np.hypot(event.x - px, event.y - py))
        if moved > self.click_tolerance:
            return  # a camera rotate, zoom or pan, and not a click
        toolbar = getattr(self.map_fig.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return

        target = (float(event.x), float(event.y))
        if button == 1:
            self.handle_left_click(target)
        elif button == 3:
            self.handle_region_click(target)

    def handle_left_click(self, target: tuple[float, float]) -> None:
        display, depths = project_to_screen(self.map_ax, self.coords)
        if self.mode == "generate":
            self.handle_generate_click_3d(target, display, depths)
            return
        indices = select_click_indices_screen(
            display, self.rows, self.cfg, target, self.nearest, self.visible_mask
        )
        if not len(indices):
            self.set_status("No visible points near click")
            return
        self.show_selection(indices, f"nearest {len(indices)} to click")

    def handle_region_click(self, target: tuple[float, float]) -> None:
        display, _ = project_to_screen(self.map_ax, self.coords)
        selected = indices_within_pixels(
            display, target, self.pick_radius, self.visible_mask
        )
        if not len(selected):
            self.set_status(
                f"No visible points within {self.pick_radius:.0f} px of the click"
            )
            return
        unique = unique_crop_indices(selected, self.rows, self.cfg)
        # the rank comes from the screen distance, so that a truncated selection
        # keeps the points nearest the actual click.
        shown = representative_indices(display, unique, self.max_crops)
        detail = (
            f"region selected {len(selected)} points / {len(unique)} unique crops "
            f"within {self.pick_radius:.0f} px; displaying {len(shown)}"
        )
        self.show_selection(shown, detail)

    # --- generate mode ------------------------------------------------------

    def handle_generate_click_3d(
        self, target: tuple[float, float], display: np.ndarray, depths: np.ndarray
    ) -> None:
        import traceback

        assert self.decoder is not None and self.embeddings is not None
        distances = np.linalg.norm(display - np.asarray(target)[None, :], axis=1)
        distances = np.where(self.visible_mask, distances, np.inf)
        if not np.isfinite(distances).any():
            self.set_status("No visible points to anchor the click depth")
            return
        anchor = int(np.argmin(distances))
        try:
            point3d = unproject_from_screen(
                self.map_ax, target[0], target[1], float(depths[anchor])
            )
        except Exception as exc:
            self.set_status(f"Could not unproject click: {exc}")
            return

        indices, weights, confidence = gaussian_radius_blend(
            self.coords, tuple(point3d), self.gen_radius, self.visible_mask
        )
        confidence += f" (depth taken from point {anchor})"
        set_3d_offsets(self.highlight, self.coords[indices])
        self.map_fig.canvas.draw_idle()
        self.set_status("Generating… " + confidence)
        try:
            self.render_generated(indices, weights, confidence)
        except Exception as exc:
            self.set_status(f"Could not generate: {exc}")
            traceback.print_exc()
            return
        self.set_status(confidence)


def main() -> None:
    args = parse_args()
    if args.nearest < 1 or args.max_crops < 1:
        raise ValueError("--nearest and --max-crops must be at least 1")
    if args.point_size <= 0:
        raise ValueError("--point-size must be greater than 0")
    if args.pick_radius <= 0:
        raise ValueError("--pick-radius must be greater than 0")
    if args.click_tolerance < 0:
        raise ValueError("--click-tolerance must not be negative")

    cfg = Config.from_yaml(args.config)
    metadata_path = Path(args.metadata)
    rows, coords = read_metadata_3d(metadata_path)
    zarr_paths = resolve_zarr_paths(cfg, args)
    validate_dataset_mapping(rows, zarr_paths)
    require_zarrs_exist(zarr_paths, rows)

    embeddings = None
    gen_radius = args.gen_radius
    if args.decoder_checkpoint is not None:
        embeddings_path = (
            Path(args.embeddings) if args.embeddings else default_embeddings_path(metadata_path)
        )
        print(f"[viewer3d] loading real embeddings from {embeddings_path}")
        embeddings = load_real_embeddings_3d(embeddings_path, coords)
        if gen_radius is None:
            gen_radius = estimate_default_radius(coords)
        print(f"[viewer3d] Generate mode search radius: {gen_radius:.4f}")

    print(f"[viewer3d] loaded {len(rows)} latent points")
    print("[viewer3d] left-drag rotates; left-click picks nearest; right-click picks a region; C clears")
    viewer = LatentCropViewer3D(
        rows,
        coords,
        cfg,
        zarr_paths,
        nearest=args.nearest,
        max_crops=args.max_crops,
        point_size=args.point_size,
        pick_radius=args.pick_radius,
        click_tolerance=args.click_tolerance,
        labels_xyz=axis_labels(metadata_path),
        embeddings=embeddings,
        decoder_checkpoint=args.decoder_checkpoint,
        gen_radius=gen_radius,
    )
    # see interactive_latent_viewer.main for the reason to hold the viewer here.
    plt.show()
    del viewer


if __name__ == "__main__":
    main()
