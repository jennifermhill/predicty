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


def visualize_gradcam_single(
    image: np.ndarray,
    gradcam: np.ndarray,
    title: str = "Grad-CAM",
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (12, 4),
) -> None:
    """
    Visualize original image, Grad-CAM, and overlay side-by-side.

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

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].imshow(image_2d, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(gradcam, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

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
) -> None:
    """
    Visualize Grad-CAM for multiple classes in a grid.

    Args:
        image: Input image [0, 1]
        gradcams: Dict mapping class_idx → Grad-CAM heatmap
        class_names: List of class names
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size, defaults to (5*n_classes, 5)
    """
    n_classes = len(gradcams)
    if figsize is None:
        figsize = (5 * n_classes, 5)

    # Ensure 2D image
    if image.ndim == 3:
        image_2d = image.max(axis=0)
    else:
        image_2d = image

    fig, axes = plt.subplots(1, n_classes, figsize=figsize)
    if n_classes == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=14, fontweight="bold")

    for idx, (class_idx, gradcam) in enumerate(sorted(gradcams.items())):
        overlay = overlay_gradcam(image_2d, gradcam, alpha=0.5)
        axes[idx].imshow(overlay)
        axes[idx].set_title(f"{class_names[class_idx]}")
        axes[idx].axis("off")

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
