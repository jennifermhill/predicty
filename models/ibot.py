"""
utils for the masking objective (currently iBOT applied to a Resnet50 backbone): 
patch-level masked reconstruction of feature maps from a shared encoder. 
the encoder is frozen and the decoder is trained to reconstruct
the masked patches from the unmasked ones. the loss is evaluated only on masked patches
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import random


def _trunc_normal_(x: torch.Tensor, std: float = 0.02) -> None:
    nn.init.trunc_normal_(x, std=std, a=-2 * std, b=2 * std)


def _reconstruction_metrics(
    pred: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """scale-free reconstruction quality between two flat voxel tensors.

    returns (r2, pearson_corr, relative_l2). r2 and corr sit in [-1, 1] where
    higher is better; rel_l2 is in [0, inf) where lower is better. falls back to
    finite defaults when the target is constant (undefined variance).
    """
    if pred.numel() == 0:
        zero = pred.new_zeros(())
        return zero, zero, zero

    target_mean = target.mean()
    ss_tot = ((target - target_mean) ** 2).sum()
    ss_res = ((target - pred) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot.clamp(min=1e-8)

    pred_centered = pred - pred.mean()
    target_centered = target - target_mean
    denom = (pred_centered.norm() * target_centered.norm()).clamp(min=1e-8)
    corr = (pred_centered * target_centered).sum() / denom

    rel_l2 = ss_res.sqrt() / target.norm().clamp(min=1e-8)
    return r2, corr, rel_l2

def make_block_mask(
    N: int, grid: int = 8, ratio: float = 0.4, block: int = 2, device: torch.device = "cpu"
) -> torch.Tensor:
    target = round(ratio * grid * grid)
    mask = torch.zeros(N, grid, grid, dtype=torch.bool, device=device)
    for i in range(N):
        while mask[i].sum() < target:
            x = torch.randint(0, grid - block + 1, (1,), device=device)
            y = torch.randint(0, grid - block + 1, (1,), device=device)
            mask[i, x : x + block, y : y + block] = True
    return mask.reshape(N, grid * grid)

def apply_grid_mask(x, mask, grid: int = 8):
    N, _, Z, H, W = x.shape
    cell = H // grid
    m = mask.view(N, grid, grid)
    m = m.repeat_interleave(cell, dim=1).repeat_interleave(cell, dim=2)
    return x.masked_fill(m[:, None, None], 0.0)

def make_random_mask(
    batch_size: int, n_patches: int, mask_ratio: float, device: torch.device
) -> torch.Tensor:
    """generate random binary masks for patchified volumes.

    Args:
        batch_size: number of masks to generate.
        n_patches: total number of patches in a volume.
        mask_ratio: fraction of patches to mask.
        device: device to create mask on.

    Returns:
        boolean tensor of shape (batch_size, n_patches) where True indicates masked patches.
    """
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError(f"mask_ratio must be in (0, 1); got {mask_ratio}")

    n_mask = max(1, min(n_patches - 1, int(round(n_patches * mask_ratio))))
    noise = torch.rand(batch_size, n_patches, device=device)
    ids_shuffle = noise.argsort(dim=1)
    mask = torch.zeros(batch_size, n_patches, dtype=torch.bool, device=device)
    mask.scatter_(1, ids_shuffle[:, :n_mask], True)
    return mask


def apply_mask_to_volume(
    x: torch.Tensor,
    mask: torch.Tensor,
    grid_shape: tuple[int, int, int],
    patch_shape: tuple[int, int, int],
) -> torch.Tensor:
    """apply patch-level masking to a batch of 3D volumes.

    Args:
        x: input tensor of shape (B, 1, Z, H, W).
        mask: binary mask of shape (B, n_patches) where True = masked.
        grid_shape: (n_z, n_h, n_w), number of patches in each dimension.
        patch_shape: (pz, ph, pw), patch size in each dimension.

    Returns:
        masked volume with same shape as x (masked patches set to 0).
    """
    n, c, z, h, w = x.shape
    pz, ph, pw = patch_shape
    gz, gh, gw = grid_shape
    n_patches = math.prod(grid_shape)
    patch_dim = pz * ph * pw

    if c != 1:
        raise ValueError(f"expected single-channel input; got {c} channels")
    if (z, h, w) != tuple(int(g * p) for g, p in zip(grid_shape, patch_shape)):
        raise ValueError(
            f"volume shape {(z, h, w)} does not match grid {grid_shape} * patch {patch_shape}"
        )

    # patchify
    patches = x.reshape(n, c, gz, pz, gh, ph, gw, pw)
    patches = patches.permute(0, 2, 4, 6, 1, 3, 5, 7)
    patches = patches.reshape(n, n_patches, patch_dim)

    # mask
    patches = patches.masked_fill(mask.unsqueeze(-1), 0.0)

    # unpatchify
    patches = patches.reshape(n, gz, gh, gw, c, pz, ph, pw)
    patches = patches.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return patches.reshape(n, c, z, h, w)


class MaskedPatchAutoencoder(nn.Module):
    """patch-level masked reconstruction decoder for 3D crop tensors.

    the shared encoder produces one latent vector per masked crop. the decoder
    conditions that vector on learned patch-position embeddings and predicts a
    flattened voxel patch for every patch location. loss is evaluated only on
    masked patches, matching the MAE objective while avoiding a large dense
    volumetric decoder.
    """

    def __init__(
        self,
        latent_dim: int,
        crop_shape: tuple[int, int, int],
        patch_shape: tuple[int, int, int],
        hidden_dim: int = 512,
        depth: int = 2,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1); got {mask_ratio}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1; got {depth}")

        self.crop_shape = tuple(int(v) for v in crop_shape)
        self.patch_shape = tuple(int(v) for v in patch_shape)
        self.mask_ratio = float(mask_ratio)
        self.norm_pix_loss = bool(norm_pix_loss)

        z, h, w = self.crop_shape
        pz, ph, pw = self.patch_shape
        if z % pz or h % ph or w % pw:
            raise ValueError(
                "crop_shape must be divisible by patch_shape; "
                f"got crop_shape={self.crop_shape}, patch_shape={self.patch_shape}"
            )

        self.grid_shape = (z // pz, h // ph, w // pw)
        self.n_patches = math.prod(self.grid_shape)
        self.patch_dim = pz * ph * pw

        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.empty(1, self.n_patches, hidden_dim))
        self.mask_token = nn.Parameter(torch.empty(1, 1, hidden_dim))

        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.extend([
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            ])
        layers.extend([
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.patch_dim),
        ])
        self.decoder = nn.Sequential(*layers)

        _trunc_normal_(self.pos_embed)
        _trunc_normal_(self.mask_token)

    def forward(self, x: torch.Tensor, encoder: nn.Module) -> dict[str, torch.Tensor]:
        """return masked-patch reconstruction loss and diagnostics.

        Args:
            x: global crops with shape (N, 1, Z, H, W).
            encoder: shared student encoder. it receives the masked crop.
        """
        if x.ndim != 5 or x.size(1) != 1:
            raise ValueError(f"expected x shape (N, 1, Z, H, W); got {tuple(x.shape)}")

        target = self.patchify(x)
        mask = self.random_mask(x.size(0), x.device)
        x_masked = self.apply_patch_mask(x, mask)

        z = encoder(x_masked)
        h = self.latent_proj(z).unsqueeze(1)
        tokens = h + self.pos_embed + self.mask_token * mask.unsqueeze(-1).to(h.dtype)
        pred = self.decoder(tokens)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True, unbiased=False)
            target = (target - mean) / (var + 1e-6).sqrt()

        patch_loss = F.mse_loss(pred, target, reduction="none").mean(dim=-1)
        loss = patch_loss[mask].mean()

        # reconstruction quality on masked patches only. mse alone is hard to
        # read (its scale drifts with norm_pix_loss and input magnitude), so we
        # also report scale-free metrics that say how good the reconstruction
        # actually is: r2 (fraction of target variance explained), pearson
        # correlation, and relative l2 error.
        with torch.no_grad():
            pred_masked = pred[mask].flatten()
            target_masked = target[mask].flatten()
            r2, corr, rel_l2 = _reconstruction_metrics(pred_masked, target_masked)

        return {
            "loss": loss,
            "pred": pred,
            "target": target,
            "mask": mask,
            "mask_ratio": mask.float().mean(),
            "recon_r2": r2,
            "recon_corr": corr,
            "recon_rel_l2": rel_l2,
        }

    def random_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return make_random_mask(batch_size, self.n_patches, self.mask_ratio, device)

    def apply_patch_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return apply_mask_to_volume(x, mask, self.grid_shape, self.patch_shape)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        n, c, z, h, w = x.shape
        pz, ph, pw = self.patch_shape
        gz, gh, gw = self.grid_shape
        if c != 1 or (z, h, w) != self.crop_shape:
            raise ValueError(
                f"expected crop shape (N, 1, {self.crop_shape}); got {tuple(x.shape)}"
            )
        x = x.reshape(n, c, gz, pz, gh, ph, gw, pw)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
        return x.reshape(n, self.n_patches, self.patch_dim)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        n = patches.size(0)
        pz, ph, pw = self.patch_shape
        gz, gh, gw = self.grid_shape
        patches = patches.reshape(n, gz, gh, gw, 1, pz, ph, pw)
        patches = patches.permute(0, 4, 1, 5, 2, 6, 3, 7)
        z, h, w = self.crop_shape
        return patches.reshape(n, 1, z, h, w)
