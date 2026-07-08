"""Visualization utilities for Grad-CAM heatmaps."""

from __future__ import annotations

import cv2
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
    Apply a colormap to a grayscale heatmap.

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
    Overlay Grad-CAM heatmap on input image.

    Args:
        image: Input image, grayscale [0, 1] or RGB [0, 1]
        gradcam: Grad-CAM heatmap [0, 1]
        alpha: Blending weight (0 = image only, 1 = heatmap only)

    Returns:
        Blended RGB image [0, 1]
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
    """Summarize where a Grad-CAM heatmap concentrates its attention.

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
    """Draw a colored border around a subplot to flag the predicted class."""
    ax.axis("on")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(width)


def _draw_focus_contour(ax, gradcam: np.ndarray, levels=(0.5, 0.75)) -> None:
    """Outline high-activation regions so focus is readable against the cell."""
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
    Visualize original, heatmap, overlay, and a focus-contour panel.

    Panels: original grayscale; Grad-CAM with an intensity colorbar; blended
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
    Visualize Grad-CAM for multiple classes in a two-row grid.

    Top row overlays each class heatmap on the crop with the top activation
    region outlined; bottom row shows the raw heatmap on a shared, fixed [0, 1]
    color scale (one colorbar) so intensities are comparable across classes.
    Per-class titles include hot-area coverage, and the predicted class column
    is highlighted when `pred_class`/`probs` are provided.

    Args:
        image: Input image [0, 1]
        gradcams: Dict mapping class_idx → Grad-CAM heatmap
        class_names: List of class names
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size, defaults to (4*n_classes, 8)
        pred_class: Optional predicted class index to highlight
        probs: Optional dict mapping class_idx → probability, shown in titles
    """
    n_classes = len(gradcams)
    if figsize is None:
        figsize = (4 * n_classes, 8)

    # Ensure 2D image
    if image.ndim == 3:
        image_2d = image.max(axis=0)
    else:
        image_2d = image

    fig, axes = plt.subplots(2, n_classes, figsize=figsize, squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    hm_im = None
    for col, (class_idx, gradcam) in enumerate(sorted(gradcams.items())):
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
        fig.colorbar(
            hm_im, ax=axes[1, :].tolist(),
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
    Visualize multiple samples with their Grad-CAM overlays in a grid.

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
    Max project a 3D volume to 2D.

    Args:
        volume: 3D array (Z, H, W)
        axis: Axis to project along

    Returns:
        2D projection (H, W)
    """
    return np.max(volume, axis=axis)


def save_gradcam_heatmap(
    gradcam: torch.Tensor | np.ndarray,
    save_path: str,
    colormap: int = cv2.COLORMAP_JET,
) -> None:
    """
    Save Grad-CAM heatmap as an image file.

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
