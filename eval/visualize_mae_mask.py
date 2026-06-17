"""Visualize the masked patches used by the MAE auxiliary loss.

This samples one global crop from the configured dataset, applies the same
MaskedPatchAutoencoder patch mask used during training, and saves diagnostic
PNGs showing the original crop, masked crop, and mask layout.

Usage:
    python eval/visualize_mae_mask.py --config configs/train.yaml
    python eval/visualize_mae_mask.py --config configs/train.yaml --view t2 --sample-index 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from configs.config import Config
from data.dataset import MultiZarrDataset, ZarrCellDataset
from models.mae import MaskedPatchAutoencoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to training yaml config")
    p.add_argument("--out-dir", default=None, help="defaults under mae_masks/")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-index", type=int, default=0, help="skip this many samples before plotting")
    p.add_argument("--view", choices=("t1", "t2"), default="t1", help="which global crop to mask")
    p.add_argument(
        "--spatial-region-mode",
        choices=("config", "all", "exclude_holdout", "holdout"),
        default="config",
        help="override cfg.spatial_region_mode for visualization",
    )
    return p.parse_args()


def build_dataset(cfg: Config):
    if cfg.zarr_paths:
        return MultiZarrDataset(cfg)
    if cfg.zarr_path is None:
        raise ValueError("visualization needs cfg.zarr_path or cfg.zarr_paths")
    return ZarrCellDataset(cfg)


def as_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def mean_projection(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=0)


def robust_limits(*images: np.ndarray) -> tuple[float, float]:
    vals = np.concatenate([img.reshape(-1) for img in images])
    lo, hi = np.percentile(vals, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(vals.min()), float(vals.max())
    if lo == hi:
        hi = lo + 1.0
    return float(lo), float(hi)


def add_patch_grid(ax: plt.Axes, h: int, w: int, patch_size: int) -> None:
    for x in range(patch_size, w, patch_size):
        ax.axvline(x - 0.5, color="white", linewidth=0.25, alpha=0.25)
    for y in range(patch_size, h, patch_size):
        ax.axhline(y - 0.5, color="white", linewidth=0.25, alpha=0.25)


def show_gray(
    ax: plt.Axes,
    image: np.ndarray,
    title: str,
    limits: tuple[float, float],
    patch_size: int | None = None,
) -> None:
    ax.imshow(image, cmap="gray", vmin=limits[0], vmax=limits[1])
    if patch_size is not None:
        add_patch_grid(ax, image.shape[0], image.shape[1], patch_size)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def save_overview(
    out_path: Path,
    original: np.ndarray,
    masked: np.ndarray,
    mask_volume: np.ndarray,
    meta_text: str,
    patch_size: int,
) -> None:
    orig_proj = mean_projection(original)
    masked_proj = mean_projection(masked)
    mask_proj = mask_volume.max(axis=0)
    diff_proj = mean_projection(np.abs(original - masked))
    limits = robust_limits(orig_proj, masked_proj)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), dpi=180)
    show_gray(axes[0], orig_proj, "original crop\nmean over z", limits, patch_size)
    show_gray(axes[1], masked_proj, "MAE input\nmasked patches zeroed", limits, patch_size)
    show_gray(axes[2], orig_proj, "mask overlay\nred = hidden", limits, patch_size)
    axes[2].imshow(mask_proj, cmap="Reds", alpha=0.45, vmin=0, vmax=1)
    show_gray(
        axes[3],
        diff_proj,
        "absolute difference\nmean over z",
        robust_limits(diff_proj),
        patch_size,
    )
    fig.suptitle(meta_text, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_z_montage(
    out_path: Path,
    original: np.ndarray,
    masked: np.ndarray,
    mask_volume: np.ndarray,
    patch_size: int,
) -> None:
    z = original.shape[0]
    limits = robust_limits(original, masked)
    fig, axes = plt.subplots(3, z, figsize=(1.8 * z, 5.6), dpi=160)
    for zi in range(z):
        show_gray(axes[0, zi], original[zi], f"orig z{zi}", limits, patch_size)
        show_gray(axes[1, zi], masked[zi], f"masked z{zi}", limits, patch_size)
        show_gray(axes[2, zi], original[zi], f"mask z{zi}", limits, patch_size)
        axes[2, zi].imshow(mask_volume[zi], cmap="Reds", alpha=0.45, vmin=0, vmax=1)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    cfg.seed = args.seed
    if args.spatial_region_mode != "config":
        cfg.spatial_region_mode = args.spatial_region_mode

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path("mae_masks") / time.strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    ds = build_dataset(cfg)
    it = iter(ds)
    sample = None
    for _ in range(args.sample_index + 1):
        sample = next(it)
    if sample is None:
        raise RuntimeError("failed to sample a crop")

    view_idx = 0 if args.view == "t1" else 1
    crop = sample["globals"][view_idx].unsqueeze(0)
    mae = MaskedPatchAutoencoder(
        latent_dim=cfg.latent_dim,
        crop_shape=(cfg.z_thickness, cfg.global_crop_size, cfg.global_crop_size),
        patch_shape=(cfg.mae_patch_z, cfg.mae_patch_size, cfg.mae_patch_size),
        hidden_dim=cfg.mae_decoder_hidden,
        depth=cfg.mae_decoder_depth,
        mask_ratio=cfg.mae_mask_ratio,
        norm_pix_loss=cfg.mae_norm_pix_loss,
    )
    mask = mae.random_mask(batch_size=1, device=crop.device)
    masked_crop = mae.apply_patch_mask(crop, mask)
    mask_patches = mask.float().unsqueeze(-1).expand(-1, -1, mae.patch_dim)
    mask_volume = mae.unpatchify(mask_patches)[0, 0].numpy()

    original_np = crop[0, 0].numpy()
    masked_np = masked_crop[0, 0].numpy()
    meta = sample["meta"]
    dataset_idx = as_int(sample["weak_label"][0])
    t_key = "t1" if view_idx == 0 else "t2"
    meta_text = (
        f"dataset={dataset_idx} view={args.view} t={as_int(meta[t_key])} "
        f"z=[{as_int(meta['z'])},{as_int(meta['z_end'])}) "
        f"yx=({as_int(meta['global_y'])},{as_int(meta['global_x'])}) "
        f"region={as_int(meta['spatial_region_idx'])} "
        f"mask_ratio={float(mask.float().mean()):.3f}"
    )

    save_overview(
        out_dir / "mae_mask_overview.png",
        original_np,
        masked_np,
        mask_volume,
        meta_text,
        cfg.mae_patch_size,
    )
    save_z_montage(
        out_dir / "mae_mask_z_slices.png",
        original_np,
        masked_np,
        mask_volume,
        cfg.mae_patch_size,
    )

    np.savez_compressed(
        out_dir / "mae_mask_sample.npz",
        original=original_np.astype(np.float32),
        masked=masked_np.astype(np.float32),
        mask_volume=mask_volume.astype(np.float32),
        mask=mask.cpu().numpy(),
    )

    print(f"[mae-mask] wrote {out_dir}")
    for path in sorted(out_dir.iterdir()):
        print(f"  - {path}")


if __name__ == "__main__":
    main()
