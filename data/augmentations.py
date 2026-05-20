"""visual augmentations for single-channel float crops.

transforms work on `[1, H, W]` or `[1, Z, H, W]` float32 tensors.
each transform reads its own `AugCfg` and applies with probability `p`.

pipeline order:

    rescale to [0, 1]
      -> random gamma
      -> random intensity
      -> gaussian noise
      -> gaussian blur
      -> per-crop normalize
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from config import AugCfg, Config


_EPS = 1e-6


class _Aug:
    """apply `_apply` with probability `cfg.p`."""

    def __init__(self, cfg: AugCfg) -> None:
        self.cfg = cfg

    def __call__(self, x: Tensor) -> Tensor:
        if not self.cfg.enabled:
            return x
        if torch.rand(()) >= self.cfg.p:
            return x
        return self._apply(x)

    def _apply(self, x: Tensor) -> Tensor:  # pragma: no cover - overridden
        raise NotImplementedError


class RandomIntensity(_Aug):
    """multiply intensity by a small random scale."""

    def _apply(self, x: Tensor) -> Tensor:
        s = float(self.cfg.strength)
        scale = 1.0 + (torch.rand(()).item() * 2.0 - 1.0) * s
        return x * scale


class RandomGamma(_Aug):
    """apply gamma correction to [0, 1] inputs."""

    def _apply(self, x: Tensor) -> Tensor:
        gmin, gmax = self.cfg.gamma_range
        gamma = torch.rand(()).item() * (gmax - gmin) + gmin
        return x.clamp(min=_EPS).pow(gamma)


class GaussianNoise(_Aug):
    """add gaussian noise."""

    def _apply(self, x: Tensor) -> Tensor:
        smin, smax = self.cfg.sigma_range
        sigma = torch.rand(()).item() * (smax - smin) + smin
        if sigma <= 0.0:
            return x
        return x + torch.randn_like(x) * sigma


class GaussianBlur(_Aug):
    """blur y/x with a sampled sigma."""

    def _apply(self, x: Tensor) -> Tensor:
        from torchvision.transforms.functional import gaussian_blur

        smin, smax = self.cfg.sigma_range
        sigma = torch.rand(()).item() * (smax - smin) + smin
        if sigma <= 0.0:
            return x
        k = self.cfg.kernel_size
        if k % 2 == 0:
            k += 1
        return gaussian_blur(x, kernel_size=[k, k], sigma=[sigma, sigma])


class PerCropNormalize(_Aug):
    """normalize one crop by its own mean and std."""

    def _apply(self, x: Tensor) -> Tensor:
        return (x - x.mean()) / (x.std() + _EPS)


class _Pipeline:
    # module-level so dataloader workers can pickle it.
    def __init__(self, augs: list[_Aug]) -> None:
        self.augs = augs

    def __call__(self, x: Tensor) -> Tensor:
        for t in self.augs:
            x = t(x)
        return x


def build_augmentations(cfg: Config) -> Callable[[Tensor], Tensor]:
    """compose enabled augmentations in pipeline order."""
    ordered: list[_Aug] = []
    if cfg.aug_gamma.enabled:
        ordered.append(RandomGamma(cfg.aug_gamma))
    if cfg.aug_intensity.enabled:
        ordered.append(RandomIntensity(cfg.aug_intensity))
    if cfg.aug_noise.enabled:
        ordered.append(GaussianNoise(cfg.aug_noise))
    if cfg.aug_blur.enabled:
        ordered.append(GaussianBlur(cfg.aug_blur))
    if cfg.aug_percrop_norm.enabled:
        ordered.append(PerCropNormalize(cfg.aug_percrop_norm))

    return _Pipeline(ordered)
