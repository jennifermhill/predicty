"""Visualize the masked patches used by the MAE auxiliary loss.

This samples one global crop from the configured dataset, applies the same
MaskedPatchAutoencoder patch mask used during training, and saves diagnostic
PNGs showing the original crop, masked crop, and mask layout.
If --checkpoint is provided, it also visualizes the trained MAE reconstruction.

Usage:
    python eval/visualize_mae_mask.py --config configs/train.yaml
    python eval/visualize_mae_mask.py --config configs/train.yaml --view t2 --sample-index 5
    python eval/visualize_mae_mask.py --config configs/train.yaml --checkpoint checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
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
from models.encoder import HybridEncoder
from models.mae import MaskedPatchAutoencoder


MetricValue = float | int | str | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to training yaml config")
    p.add_argument("--checkpoint", default=None, help="optional checkpoint for MAE reconstruction panels")
    p.add_argument("--device", default=None, help="device for reconstruction when --checkpoint is used")
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
    p.add_argument(
        "--save-z-slices",
        action="store_true",
        help="also save the detailed per-Z diagnostic montage",
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


def disable_stochastic_augmentations(cfg: Config) -> None:
    """Keep encoder normalization but make the sampled evaluation crop stable."""
    for name in ("aug_intensity", "aug_gamma", "aug_noise", "aug_blur"):
        getattr(cfg, name).enabled = False
    if cfg.aug_percrop_norm.enabled:
        cfg.aug_percrop_norm.p = 1.0


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


@dataclass(frozen=True)
class ReconstructionResult:
    volume: np.ndarray
    objective_mse: float


def reconstruction_metrics(
    original: np.ndarray,
    reconstruction: np.ndarray,
    mask_volume: np.ndarray,
    objective_mse: float,
) -> dict[str, MetricValue]:
    """Evaluate only hidden voxels; visible voxels are copied and must be excluded."""
    hidden = mask_volume.astype(bool)
    if original.shape != reconstruction.shape or original.shape != hidden.shape:
        raise ValueError("original, reconstruction, and mask_volume must have equal shapes")
    if not hidden.any():
        raise ValueError("mask_volume contains no hidden voxels")

    target = original[hidden].astype(np.float64)
    prediction = reconstruction[hidden].astype(np.float64)
    error = prediction - target
    mse = float(np.mean(error**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))
    data_range = float(original.max() - original.min())
    psnr = float("inf") if rmse == 0 else (
        float(20.0 * np.log10(data_range / rmse)) if data_range > 0 else None
    )

    target_centered = target - target.mean()
    baseline_mse = float(np.mean(target_centered**2))
    r2 = None if baseline_mse <= 0 else float(1.0 - mse / baseline_mse)
    target_std = float(target.std())
    prediction_std = float(prediction.std())
    correlation = None
    if target_std > 0 and prediction_std > 0:
        correlation = float(np.corrcoef(target, prediction)[0, 1])

    return {
        "hidden_voxels": int(hidden.sum()),
        "total_voxels": int(hidden.size),
        "mask_ratio": float(hidden.mean()),
        "mae": mae,
        "rmse": rmse,
        "psnr_db": psnr,
        "psnr_data_range": data_range,
        "pearson_r": correlation,
        "r2": r2,
        "objective_mse": float(objective_mse),
    }


def metric_text(metrics: dict[str, MetricValue]) -> str:
    def value(key: str, suffix: str = "") -> str:
        number = metrics[key]
        if number is None:
            return "N/A"
        if isinstance(number, int):
            return f"{number:,}{suffix}"
        if isinstance(number, str):
            return f"{number}{suffix}"
        if np.isinf(number):
            return f"∞{suffix}"
        return f"{number:.4f}{suffix}"

    return "\n".join(
        [
            "THIS CROP: HIDDEN-VOXEL METRICS",
            "(visible patches excluded)",
            f"Input: {metrics.get('input_space', 'crop-value units')}",
            f"Recon: {metrics.get('reconstruction_space', 'input-value space')}",
            "",
            f"MAE             {value('mae')}",
            f"RMSE            {value('rmse')}",
            f"PSNR            {value('psnr_db', ' dB')}",
            f"  target range  {value('psnr_data_range')}",
            f"Pearson r       {value('pearson_r')}",
            f"R²              {value('r2')}",
            "",
            f"Objective MSE   {value('objective_mse')}",
            f"  {metrics.get('objective_space', 'training target space')}",
            f"Mask ratio      {value('mask_ratio')}",
            f"Hidden voxels   {value('hidden_voxels')}",
        ]
    )


def save_overview(
    out_path: Path,
    original: np.ndarray,
    masked: np.ndarray,
    mask_volume: np.ndarray,
    meta_text: str,
    patch_size: int,
    reconstruction: np.ndarray | None = None,
    metrics: dict[str, MetricValue] | None = None,
) -> None:
    if reconstruction is not None and metrics is None:
        raise ValueError("metrics are required when reconstruction is provided")
    orig_proj = mean_projection(original)
    masked_proj = mean_projection(masked)
    mask_proj = mask_volume.mean(axis=0)
    images = [orig_proj, masked_proj]
    if reconstruction is not None:
        recon_proj = mean_projection(reconstruction)
        hidden_error = np.abs(original - reconstruction) * mask_volume
        hidden_count = mask_volume.sum(axis=0)
        recon_diff_proj = np.divide(
            hidden_error.sum(axis=0),
            hidden_count,
            out=np.zeros_like(hidden_count, dtype=np.float32),
            where=hidden_count > 0,
        )
        images.append(recon_proj)
    else:
        recon_proj = None
        recon_diff_proj = None
    limits = robust_limits(*images)

    n_cols = 5 if reconstruction is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(3.25 * n_cols, 4.2), dpi=180)
    show_gray(axes[0], orig_proj, "Target\nmean-Z", limits, patch_size)
    show_gray(axes[1], masked_proj, "Encoder input\nhidden patches = 0", limits, patch_size)
    axes[1].imshow(mask_proj, cmap="Reds", alpha=0.30 * mask_proj, vmin=0, vmax=1)
    if reconstruction is not None and recon_proj is not None and recon_diff_proj is not None:
        show_gray(
            axes[2],
            recon_proj,
            "Reconstruction\nhidden patches predicted",
            limits,
            patch_size,
        )
        error_limit = float(np.percentile(recon_diff_proj, 99.0))
        if not np.isfinite(error_limit) or error_limit <= 0:
            error_limit = 1.0
        error_image = axes[3].imshow(
            recon_diff_proj, cmap="magma", vmin=0.0, vmax=error_limit
        )
        axes[3].set_title("Absolute error\nmean over hidden Z", fontsize=9)
        axes[3].set_xticks([])
        axes[3].set_yticks([])
        fig.colorbar(error_image, ax=axes[3], fraction=0.046, pad=0.04)
        axes[4].axis("off")
        axes[4].text(
            0.02,
            0.98,
            metric_text(metrics),
            transform=axes[4].transAxes,
            va="top",
            ha="left",
            family="monospace",
            fontsize=8.5,
        )
    else:
        show_gray(
            axes[2],
            orig_proj,
            "Mask layout\nred = fraction hidden over Z",
            limits,
            patch_size,
        )
        axes[2].imshow(mask_proj, cmap="Reds", alpha=0.55 * mask_proj, vmin=0, vmax=1)
    fig.suptitle(meta_text, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


@torch.no_grad()
def reconstruct_with_checkpoint(
    checkpoint_path: str,
    crop: torch.Tensor,
    mask: torch.Tensor,
    mae: MaskedPatchAutoencoder,
    cfg: Config,
    device: torch.device,
) -> ReconstructionResult:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "mae_decoder_state_dict" not in ckpt:
        raise ValueError(f"{checkpoint_path} does not contain mae_decoder_state_dict")

    encoder = HybridEncoder(
        stem_channels=cfg.stem_channels,
        latent_dim=cfg.latent_dim,
        norm_groups=cfg.norm_groups,
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()

    mae = mae.to(device)
    mae.load_state_dict(ckpt["mae_decoder_state_dict"])
    mae.eval()

    crop_d = crop.to(device)
    mask_d = mask.to(device)
    masked_crop = mae.apply_patch_mask(crop_d, mask_d)
    target_raw = mae.patchify(crop_d)

    z = encoder(masked_crop)
    h = mae.latent_proj(z).unsqueeze(1)
    tokens = h + mae.pos_embed + mae.mask_token * mask_d.unsqueeze(-1).to(h.dtype)
    pred = mae.decoder(tokens)

    target_objective = target_raw
    if mae.norm_pix_loss:
        mean = target_raw.mean(dim=-1, keepdim=True)
        var = target_raw.var(dim=-1, keepdim=True, unbiased=False)
        target_objective = (target_raw - mean) / (var + 1e-6).sqrt()
    objective_mse = torch.mean((pred[mask_d] - target_objective[mask_d]) ** 2)
    if mae.norm_pix_loss:
        pred = pred * (var + 1e-6).sqrt() + mean

    recon_patches = target_raw.clone()
    recon_patches[mask_d] = pred[mask_d]
    recon = mae.unpatchify(recon_patches)
    return ReconstructionResult(
        volume=recon[0, 0].detach().cpu().numpy(),
        objective_mse=float(objective_mse.item()),
    )


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
    disable_stochastic_augmentations(cfg)
    device = torch.device(args.device or cfg.device)
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
    reconstruction_np = None
    metrics = None
    if args.checkpoint is not None:
        result = reconstruct_with_checkpoint(
            args.checkpoint,
            crop,
            mask,
            mae,
            cfg,
            device,
        )
        reconstruction_np = result.volume
        metrics = reconstruction_metrics(
            original_np,
            reconstruction_np,
            mask_volume,
            result.objective_mse,
        )
        metrics["input_space"] = (
            "per-crop z-score"
            if cfg.aug_percrop_norm.enabled
            else "[0, 1] scaled intensity"
        )
        metrics["objective_space"] = (
            "per-patch normalized target"
            if cfg.mae_norm_pix_loss
            else "input-value target"
        )
        metrics["reconstruction_space"] = (
            "target patch mean/std restored"
            if cfg.mae_norm_pix_loss
            else "direct input-value prediction"
        )
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
        reconstruction_np,
        metrics,
    )
    if args.save_z_slices:
        save_z_montage(
            out_dir / "mae_mask_z_slices.png",
            original_np,
            masked_np,
            mask_volume,
            cfg.mae_patch_size,
        )

    arrays = {
        "original": original_np.astype(np.float32),
        "masked": masked_np.astype(np.float32),
        "mask_volume": mask_volume.astype(np.float32),
        "mask": mask.cpu().numpy(),
    }
    if reconstruction_np is not None:
        arrays["reconstruction"] = reconstruction_np.astype(np.float32)
    np.savez_compressed(out_dir / "mae_mask_sample.npz", **arrays)
    if metrics is not None:
        serializable_metrics = {
            key: (
                None
                if isinstance(value, float) and not np.isfinite(value)
                else value
            )
            for key, value in metrics.items()
        }
        with (out_dir / "mae_metrics.json").open("w") as handle:
            json.dump(serializable_metrics, handle, indent=2, sort_keys=True, allow_nan=False)
        print("[mae-mask] hidden-voxel reconstruction metrics")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

    print(f"[mae-mask] wrote {out_dir}")
    for path in sorted(out_dir.iterdir()):
        print(f"  - {path}")


if __name__ == "__main__":
    main()
