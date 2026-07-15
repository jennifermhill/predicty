"""visualization utilities for Grad-CAM heatmaps."""

from __future__ import annotations

import cv2
import matplotlib

# force the non-interactive backend before pyplot loads. on headless compute
# nodes the default tk backend tears its objects down on the wrong thread at
# exit, which aborts with "main thread is not in main loop". we only save pngs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import Optional


def apply_colormap(
    heatmap: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    apply a colormap to a grayscale heatmap.

    Args:
        heatmap: Grayscale image [0, 1] or [0, 255]
        colormap: OpenCV colormap constant

    Returns:
        RGB image with colormap applied
    """
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
    """
    overlay Grad-CAM heatmap on input image.

    Args:
        image: Input image, grayscale [0, 1] or RGB [0, 1]
        gradcam: Grad-CAM heatmap [0, 1]
        alpha: Blending weight (0 = image only, 1 = heatmap only)

    Returns:
        blended RGB image [0, 1]
    """
    # Ensure 3-channel
    if image.ndim == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    else:
        image_rgb = image

    # Normalize to [0, 255]
    img_255 = (image_rgb * 255).astype(np.uint8)
    hm_255 = (gradcam * 255).astype(np.uint8)

    # Apply colormap to heatmap
    heatmap_colored = apply_colormap(hm_255)

    # Blend
    blended = cv2.addWeighted(img_255, 1 - alpha, heatmap_colored, alpha, 0)

    return blended.astype(np.float32) / 255.0


def focus_stats(gradcam: np.ndarray, top_frac: float = 0.2) -> dict[str, float]:
    """summarize where a Grad-CAM heatmap concentrates its attention.

    Returns the peak location, the area fraction above the top-`top_frac`
    intensity threshold, and how concentrated the map is (mass inside that
    region). these numbers make it possible to compare maps quantitatively
    instead of eyeballing colors.
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
    """draw a colored border around a subplot to flag the predicted class."""
    ax.axis("on")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(width)


def _draw_focus_contour(ax, gradcam: np.ndarray, levels=(0.5, 0.75)) -> None:
    """outline high-activation regions so focus is readable against the cell."""
    if gradcam.max() - gradcam.min() < 1e-8:
        return
    ax.contour(
        gradcam,
        levels=[gradcam.max() * lvl for lvl in levels],
        colors=["#00e5ff", "#ffea00"],
        linewidths=1.2,
        alpha=0.9,
    )


def visualize_gradcam_single(
    image: np.ndarray,
    gradcam: np.ndarray,
    title: str = "Grad-CAM",
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (16, 4),
) -> None:
    """
    visualize original, heatmap, overlay, and a focus-contour panel.

    panels: original grayscale; Grad-CAM with an intensity colorbar; blended
    overlay; and the original with the top activation region outlined so the
    focus can be read directly against the cell structure. A caption reports
    peak location, hot-area coverage, and how concentrated the attention is.

    Args:
        image: Input image, grayscale or RGB [0, 1]
        gradcam: Grad-CAM heatmap [0, 1]
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size (width, height)
    """
    # Ensure 2D image for display
    if image.ndim == 3:
        image_2d = image.max(axis=0)
    else:
        image_2d = image

    # Generate overlay
    overlay = overlay_gradcam(image_2d, gradcam, alpha=0.5)
    stats = focus_stats(gradcam)

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].imshow(image_2d, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    hm_im = axes[1].imshow(gradcam, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")
    fig.colorbar(hm_im, ax=axes[1], fraction=0.046, pad=0.04, label="activation")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    axes[3].imshow(image_2d, cmap="gray")
    _draw_focus_contour(axes[3], gradcam)
    axes[3].scatter(
        [stats["peak_x"]], [stats["peak_y"]],
        c="#ff1744", marker="x", s=80, linewidths=2.0,
    )
    axes[3].set_title("Focus region")
    axes[3].axis("off")

    caption = (
        f"peak=({stats['peak_x']:.0f}, {stats['peak_y']:.0f})   "
        f"hot-area coverage={stats['coverage'] * 100:.1f}%   "
        f"concentration={stats['concentration'] * 100:.1f}%"
    )
    fig.text(0.5, 0.01, caption, ha="center", fontsize=10, color="#444444")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()


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
    """
    visualize Grad-CAM for multiple classes in a two-row grid.

    the leading column shows the original crop with no overlay for reference.
    top row overlays each class heatmap on the crop with the top activation
    region outlined; bottom row shows the raw heatmap on a shared, fixed [0, 1]
    color scale (one colorbar) so intensities are comparable across classes.
    per-class titles include hot-area coverage, and the predicted class column
    is highlighted when `pred_class`/`probs` are provided.

    Args:
        image: Input image [0, 1]
        gradcams: Dict mapping class_idx → Grad-CAM heatmap
        class_names: List of class names
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size, defaults to (4*(n_classes+1), 8) for the extra
            original-crop column
        pred_class: Optional predicted class index to highlight
        probs: Optional dict mapping class_idx → probability, shown in titles
    """
    n_classes = len(gradcams)
    # leading column holds the original crop for reference, so the grid has
    # one more column than there are classes.
    n_cols = n_classes + 1
    if figsize is None:
        figsize = (4 * n_cols, 8)

    # Ensure 2D image
    if image.ndim == 3:
        image_2d = image.max(axis=0)
    else:
        image_2d = image

    fig, axes = plt.subplots(2, n_cols, figsize=figsize, squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # column 0: original crop, no overlay. bottom cell left blank so the row
    # stays aligned with the heatmaps to its right.
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

        # top row: overlay + focus contour
        overlay = overlay_gradcam(image_2d, gradcam, alpha=0.5)
        axes[0, col].imshow(overlay)
        _draw_focus_contour(axes[0, col], gradcam)
        color = "#1565c0" if (pred_class is not None and class_idx == pred_class) else "black"
        axes[0, col].set_title(header, fontsize=11, fontweight="bold", color=color)
        axes[0, col].axis("off")
        if pred_class is not None and class_idx == pred_class:
            _highlight_axis(axes[0, col])

        # bottom row: raw heatmap on a shared, fixed scale
        hm_im = axes[1, col].imshow(gradcam, cmap="jet", vmin=0.0, vmax=1.0)
        axes[1, col].set_title(
            f"coverage {stats['coverage'] * 100:.0f}%", fontsize=9
        )
        axes[1, col].axis("off")

    if hm_im is not None:
        # attach the colorbar to the heatmap columns only, skipping the blank
        # cell under the original crop.
        fig.colorbar(
            hm_im, ax=axes[1, 1:].tolist(),
            fraction=0.046, pad=0.02, label="activation",
        )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()


def visualize_gradcam_grid(
    images: np.ndarray,
    gradcams: np.ndarray,
    titles: Optional[list[str]] = None,
    save_path: Optional[str] = None,
    figsize: Optional[tuple[int, int]] = None,
) -> None:
    """
    visualize multiple samples with their Grad-CAM overlays in a grid.

    Args:
        images: Array of images, shape (n_samples, H, W) or (n_samples, H, W, 3)
        gradcams: Array of heatmaps, shape (n_samples, H, W)
        titles: Optional list of titles for each sample
        save_path: Path to save figure
        figsize: Figure size
    """
    n_samples = images.shape[0]

    if figsize is None:
        figsize = (4 * n_samples, 8)

    fig, axes = plt.subplots(2, n_samples, figsize=figsize)
    if n_samples == 1:
        axes = axes.reshape(2, 1)

    for i in range(n_samples):
        image = images[i]
        gradcam = gradcams[i]

        # Ensure 2D image
        if image.ndim == 3:
            image_2d = image.max(axis=0)
        else:
            image_2d = image

        # Top row: original + overlay
        overlay = overlay_gradcam(image_2d, gradcam, alpha=0.5)
        axes[0, i].imshow(overlay)
        axes[0, i].axis("off")
        if titles:
            axes[0, i].set_title(titles[i], fontsize=10)

        # Bottom row: heatmap only
        axes[1, i].imshow(gradcam, cmap="jet")
        axes[1, i].axis("off")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()


def max_project_3d(
    volume: np.ndarray,
    axis: int = 0,
) -> np.ndarray:
    """
    max project a 3D volume to 2D.

    Args:
        volume: 3D array (Z, H, W)
        axis: Axis to project along

    Returns:
        2D projection (H, W)
    """
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
    """per-z-slice display of a 3D attribution map, so depth is not hidden.

    top row shows each raw depth slice; middle row overlays that slice's LayerCAM
    on a shared [0, 1] color scale (one colorbar) so slices are comparable. the
    far-right column summarizes the volume with a max-projection of the crop and
    of the attribution, and each overlay title reports that slice's mean
    attribution so you can read which depth slices the map favors.

    if `z_weights` is given, a bottom bar strip shows ZAttentionPool's literal
    per-slice depth weighting. this is a different signal from the map above:
    the map's spatial-per-z variation reflects feature relevance, whereas the
    bars are the depth weights the model actually applied (spatially uniform).
    uses matplotlib only (no OpenCV dependency).

    Args:
        crop: raw crop volume, shape (Z, H, W).
        cam3d: attribution map in [0, 1], shape (Z, H, W).
        title: plot title.
        save_path: where to save the figure.
        max_slices: cap on displayed slices; evenly subsampled if Z exceeds it.
        overlay_alpha: blend weight of the heatmap over the grayscale slice.
        z_weights: optional (Z,) ZAttentionPool weights, drawn as a bar strip.
    """
    if crop.ndim != 3 or cam3d.ndim != 3:
        raise ValueError(f"expected (Z, H, W) crop and cam; got {crop.shape}, {cam3d.shape}")
    if crop.shape[0] != cam3d.shape[0]:
        raise ValueError(f"z mismatch: crop has {crop.shape[0]} slices, cam has {cam3d.shape[0]}")
    if z_weights is not None and len(z_weights) != crop.shape[0]:
        raise ValueError(f"z_weights has {len(z_weights)} entries, expected {crop.shape[0]}")

    z_total = crop.shape[0]
    # subsample depth slices if there are too many to show side by side.
    if z_total > max_slices:
        shown = np.linspace(0, z_total - 1, max_slices).round().astype(int)
    else:
        shown = np.arange(z_total)

    # mean attribution per slice, the "which depth matters" signal.
    slice_importance = cam3d.reshape(z_total, -1).mean(axis=1)

    n_cols = len(shown) + 1  # + 1 summary column
    # two image rows, plus a short bar row when pool weights are supplied.
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

    # summary column: max-projection of the crop and of the attribution.
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

    # bottom strip: ZAttentionPool's literal per-slice depth weighting.
    if z_weights is not None:
        ax_bar = fig.add_subplot(gs[2, :])
        zs = np.arange(z_total)
        bars = ax_bar.bar(zs, np.asarray(z_weights), color="#9e9e9e")
        # highlight the slices shown in the montage so the two views line up.
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


def save_gradcam_heatmap(
    gradcam: torch.Tensor | np.ndarray,
    save_path: str,
    colormap: int = cv2.COLORMAP_JET,
) -> None:
    """
    save Grad-CAM heatmap as an image file.

    Args:
        gradcam: Heatmap tensor or array [0, 1]
        save_path: Path to save image
        colormap: OpenCV colormap
    """
    if isinstance(gradcam, torch.Tensor):
        gradcam = gradcam.cpu().numpy()

    # Apply colormap and save
    heatmap_colored = apply_colormap(gradcam, colormap)
    heatmap_bgr = cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, heatmap_bgr)
