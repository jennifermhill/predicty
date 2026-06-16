"""Masked patch reconstruction head for the WS-DINO encoder.

This is a training-only auxiliary objective. It masks patchified global crops,
encodes the masked crop with the shared student encoder, and predicts the
original masked patch values from the resulting representation. The decoder is
discarded after training.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _trunc_normal_(x: torch.Tensor, std: float = 0.02) -> None:
    nn.init.trunc_normal_(x, std=std, a=-2 * std, b=2 * std)


class MaskedPatchAutoencoder(nn.Module):
    """Patch-level masked reconstruction decoder for 3D crop tensors.

    The shared encoder produces one latent vector per masked crop. The decoder
    conditions that vector on learned patch-position embeddings and predicts a
    flattened voxel patch for every patch location. Loss is evaluated only on
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
        """Return masked-patch reconstruction loss and diagnostics.

        Args:
            x: Global crops with shape (N, 1, Z, H, W).
            encoder: Shared student encoder. It receives the masked crop.
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

        return {
            "loss": loss,
            "pred": pred,
            "target": target,
            "mask": mask,
            "mask_ratio": mask.float().mean(),
        }

    def random_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        n_mask = max(1, min(self.n_patches - 1, int(round(self.n_patches * self.mask_ratio))))
        noise = torch.rand(batch_size, self.n_patches, device=device)
        ids_shuffle = noise.argsort(dim=1)
        mask = torch.zeros(batch_size, self.n_patches, dtype=torch.bool, device=device)
        mask.scatter_(1, ids_shuffle[:, :n_mask], True)
        return mask

    def apply_patch_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        patches = self.patchify(x)
        patches = patches.masked_fill(mask.unsqueeze(-1), 0.0)
        return self.unpatchify(patches)

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
