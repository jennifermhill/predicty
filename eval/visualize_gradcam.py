"""visualization utilities for Grad-CAM heatmaps."""

from __future__ import annotations

import cv2
import matplotlib

# the matplotlib.use call must precede the pyplot import, because tk aborts at
# exit on headless nodes.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional


def apply_colormap(
    heatmap: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """the function applies `colormap` to a grayscale heatmap in [0, 1] or
    [0, 255] and returns RGB."""
    if heatmap.max() <= 1.0:
        heatmap = (heatmap * 255).astype(np.uint8)
    else:
        heatmap = heatmap.astype(np.uint8)

    colored = cv2.applyColorMap(heatmap, colormap)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def overlay_gradcam(
    image: np.ndarray,
    gradcam: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """the function blends a [0, 1] Grad-CAM over a [0, 1] grayscale or RGB image.

    alpha is the weight of the heatmap: 0 gives the image only, 1 gives the
    heatmap only.
    """
    if image.ndim == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    else:
        image_rgb = image

    img_255 = (image_rgb * 255).astype(np.uint8)
    hm_255 = (gradcam * 255).astype(np.uint8)
    heatmap_colored = apply_colormap(hm_255)
    blended = cv2.addWeighted(img_255, 1 - alpha, heatmap_colored, alpha, 0)

    return blended.astype(np.float32) / 255.0


def focus_stats(gradcam: np.ndarray, top_frac: float = 0.2) -> dict[str, float]:
    """peak location, hot-area fraction and the mass in that area, for a Grad-CAM map.

    these numbers let you compare maps quantitatively. a comparison of the
    colors by eye is not necessary.
    """
    hm = gradcam.astype(np.float64)
    total = hm.sum()
    if total <= 0:
        return {"peak_y": 0.0, "peak_x": 0.0, "coverage": 0.0, "concentration": 0.0}

    peak_y, peak_x = np.unravel_index(int(np.argmax(hm)), hm.shape)
    threshold = hm.max() * (1.0 - top_frac)
    hot = hm >= threshold
    coverage = float(hot.mean())
    concentration = float(hm[hot].sum() / total)
    return {
        "peak_y": float(peak_y),
        "peak_x": float(peak_x),
        "coverage": coverage,
        "concentration": concentration,
    }


def _highlight_axis(ax, color: str = "#1565c0", width: float = 3.0) -> None:
    """the function draws a colored border around a subplot, to mark the predicted class."""
    ax.axis("on")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(width)


def _draw_focus_contour(ax, gradcam: np.ndarray, levels=(0.5, 0.75)) -> None:
    """the function outlines the high-activation regions, so that you can read
    the focus against the cell."""
    if gradcam.max() - gradcam.min() < 1e-8:
        return
    ax.contour(
        gradcam,
        levels=[gradcam.max() * lvl for lvl in levels],
        colors=["#00e5ff", "#ffea00"],
        linewidths=1.2,
        alpha=0.9,
    )


def visualize_gradcam_multiclass(
    image: np.ndarray,
    gradcams: dict[int, np.ndarray],
    class_names: list[str],
    title: str = "Grad-CAM by Class",
    save_path: Optional[str] = None,
    figsize: Optional[tuple[int, int]] = None,
    pred_class: Optional[int] = None,
    probs: Optional[dict[int, float]] = None,
) -> None:
    """two-row per-class grid: the overlay on top, the raw heatmap below.

    the leading column is the original crop. the bottom row shares a fixed
    [0, 1] scale and one colorbar, so that the intensities are comparable across
    the classes. the function highlights the column of the predicted class when
    the caller gives `pred_class`.

    Args:
        image: input image [0, 1]
        gradcams: {class_idx: heatmap}
        pred_class: index of the predicted class. the function highlights that
            column.
        probs: {class_idx: probability}. the titles show these values.
    """
    n_classes = len(gradcams)
    n_cols = n_classes + 1  # + 1 for the original-crop column
    if figsize is None:
        figsize = (4 * n_cols, 8)

    if image.ndim == 3:
        image_2d = image.max(axis=0)
    else:
        image_2d = image

    fig, axes = plt.subplots(2, n_cols, figsize=figsize, squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # column 0: the original crop, with no overlay. the bottom cell stays blank,
    # so that the row aligns with the heatmaps to its right.
    axes[0, 0].imshow(image_2d, cmap="gray")
    axes[0, 0].set_title("Original", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")

    hm_im = None
    for offset, (class_idx, gradcam) in enumerate(sorted(gradcams.items())):
        col = offset + 1  # shift past the original-crop column
        stats = focus_stats(gradcam)
        name = class_names[class_idx]
        header = name
        if probs is not None and class_idx in probs:
            header += f"\np={probs[class_idx]:.2f}"
        if pred_class is not None and class_idx == pred_class:
            header += "  (pred)"

        # top row: the overlay and the focus contour
        overlay = overlay_gradcam(image_2d, gradcam, alpha=0.5)
        axes[0, col].imshow(overlay)
        _draw_focus_contour(axes[0, col], gradcam)
        color = "#1565c0" if (pred_class is not None and class_idx == pred_class) else "black"
        axes[0, col].set_title(header, fontsize=11, fontweight="bold", color=color)
        axes[0, col].axis("off")
        if pred_class is not None and class_idx == pred_class:
            _highlight_axis(axes[0, col])

        # bottom row: the raw heatmap on a shared, fixed scale
        hm_im = axes[1, col].imshow(gradcam, cmap="jet", vmin=0.0, vmax=1.0)
        axes[1, col].set_title(
            f"coverage {stats['coverage'] * 100:.0f}%", fontsize=9
        )
        axes[1, col].axis("off")

    if hm_im is not None:
        # the code attaches the colorbar to the heatmap columns only. the
        # colorbar skips the blank cell below the original crop.
        fig.colorbar(
            hm_im, ax=axes[1, 1:].tolist(),
            fraction=0.046, pad=0.02, label="activation",
        )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()


def max_project_3d(
    volume: np.ndarray,
    axis: int = 0,
) -> np.ndarray:
    """the function max-projects a (Z, H, W) volume along `axis` to (H, W)."""
    return np.max(volume, axis=axis)


def visualize_layercam_3d(
    crop: np.ndarray,
    cam3d: np.ndarray,
    title: str = "3D LayerCAM",
    save_path: Optional[str] = None,
    max_slices: int = 12,
    overlay_alpha: float = 0.5,
    z_weights: Optional[np.ndarray] = None,
) -> None:
    """per-z-slice display of a 3D attribution map, so that depth stays visible.

    the raw slices are on top. their LayerCAM overlays are below, on one shared
    [0, 1] scale. a max-projection summary column is at the right.

    the optional `z_weights` bar strip is a *different* signal from the map. the
    per-z variation of the map is feature relevance. the bars are the depth
    weights that ZAttentionPool applied, and those weights are spatially
    uniform.

    Args:
        crop: raw crop volume (Z, H, W).
        cam3d: attribution map in [0, 1], (Z, H, W).
        max_slices: maximum number of slices on display. if Z is larger, the
            function subsamples the slices evenly.
        overlay_alpha: blend weight of the heatmap over the grayscale slice.
        z_weights: optional (Z,) ZAttentionPool weights. the function draws them
            as a bar strip.
    """
    if crop.ndim != 3 or cam3d.ndim != 3:
        raise ValueError(f"expected (Z, H, W) crop and cam; got {crop.shape}, {cam3d.shape}")
    if crop.shape[0] != cam3d.shape[0]:
        raise ValueError(f"z mismatch: crop has {crop.shape[0]} slices, cam has {cam3d.shape[0]}")
    if z_weights is not None and len(z_weights) != crop.shape[0]:
        raise ValueError(f"z_weights has {len(z_weights)} entries, expected {crop.shape[0]}")

    z_total = crop.shape[0]
    # the function subsamples the depth slices if there are too many for a
    # side-by-side display.
    if z_total > max_slices:
        shown = np.linspace(0, z_total - 1, max_slices).round().astype(int)
    else:
        shown = np.arange(z_total)

    # mean attribution per slice, the "which depth matters" signal.
    slice_importance = cam3d.reshape(z_total, -1).mean(axis=1)

    n_cols = len(shown) + 1  # + 1 summary column
    # two image rows, plus a short bar row when the caller supplies the pool weights.
    height_ratios = [1.0, 1.0, 0.5] if z_weights is not None else [1.0, 1.0]
    n_rows = len(height_ratios)
    fig = plt.figure(figsize=(2.2 * n_cols, 2.5 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols, height_ratios=height_ratios)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    hm_im = None
    overlay_axes = []
    for col, z in enumerate(shown):
        ax_raw = fig.add_subplot(gs[0, col])
        ax_raw.imshow(crop[z], cmap="gray")
        ax_raw.set_title(f"z={z}", fontsize=9)
        ax_raw.axis("off")

        ax_ov = fig.add_subplot(gs[1, col])
        ax_ov.imshow(crop[z], cmap="gray")
        hm_im = ax_ov.imshow(cam3d[z], cmap="jet", vmin=0.0, vmax=1.0, alpha=overlay_alpha)
        ax_ov.set_title(f"imp {slice_importance[z]:.2f}", fontsize=8)
        ax_ov.axis("off")
        overlay_axes.append(ax_ov)

    # summary column: the max-projection of the crop and of the attribution.
    ax_sum_raw = fig.add_subplot(gs[0, -1])
    ax_sum_raw.imshow(crop.max(axis=0), cmap="gray")
    ax_sum_raw.set_title("crop max-proj", fontsize=9)
    ax_sum_raw.axis("off")
    ax_sum_ov = fig.add_subplot(gs[1, -1])
    ax_sum_ov.imshow(crop.max(axis=0), cmap="gray")
    ax_sum_ov.imshow(cam3d.max(axis=0), cmap="jet", vmin=0.0, vmax=1.0, alpha=overlay_alpha)
    ax_sum_ov.set_title("cam max-proj", fontsize=9)
    ax_sum_ov.axis("off")
    overlay_axes.append(ax_sum_ov)

    if hm_im is not None:
        fig.colorbar(
            hm_im, ax=overlay_axes,
            fraction=0.02, pad=0.01, label="attribution",
        )

    # bottom strip: ZAttentionPool's literal per-slice depth weights.
    if z_weights is not None:
        ax_bar = fig.add_subplot(gs[2, :])
        zs = np.arange(z_total)
        bars = ax_bar.bar(zs, np.asarray(z_weights), color="#9e9e9e")
        # the code highlights the slices that the montage shows, so that the two
        # views line up.
        for z in shown:
            bars[int(z)].set_color("#1565c0")
        ax_bar.set_xticks(zs)
        ax_bar.set_xlabel("z slice")
        ax_bar.set_ylabel("pool weight")
        ax_bar.set_title(
            "ZAttentionPool depth weighting (model's literal z-attention, spatially uniform)",
            fontsize=9,
        )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()


