"""Expected Gradients: input-space attribution for the frozen encoder + probe.

expected gradients (Erion et al. 2021), "Improving performance of deep learning
models with axiomatic attribution priors": https://arxiv.org/abs/1906.10670. it
generalizes integrated gradients (Sundararajan et al. 2017,
https://arxiv.org/abs/1703.01365) from a single fixed baseline to an expectation
over a baseline distribution.

unlike grad-cam, which weights a conv feature map and lives at that layer's
coarse resolution, EG attributes the target class logit all the way back to the
raw input voxels (1, Z, H, W). two consequences that matter for this pipeline:
  - attribution is already at input resolution, so there is no upsampling step.
  - the z axis survives. grad-cam hooks the 2d backbone, which sits after
    ZAttentionPool has already collapsed depth, so it has zero z resolution. EG
    differentiates through the pool and recovers per-slice attribution.

attribution is signed: positive voxels push the logit toward the class, negative
voxels push away. that is why the visualizers here use a diverging colormap
rather than the positive-only jet scale the grad-cam viz uses.

use `ExpectedGradients` as the entry point, not the fixed-baseline
`IntegratedGradients` kept below. groupnorm makes this encoder scale invariant,
so a straight-line path from an all-zeros baseline collapses through a
near-discontinuity at the origin and the completeness axiom cannot be satisfied
at any practical step count. sampling real crops as baselines keeps every path
inside the natural, normalized regime and restores a well-behaved estimate.
"""

from __future__ import annotations

import matplotlib

# force the non-interactive backend before pyplot loads. on headless compute
# nodes the default tk backend tears its objects down on the wrong thread at
# exit, which aborts with "main thread is not in main loop". we only save pngs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


class IntegratedGradients:
    """integrated gradients with a single fixed baseline.

    kept as the fixed-reference base case that `ExpectedGradients` generalizes,
    but prefer `ExpectedGradients` on this encoder: a zero baseline fails the
    completeness axiom here because groupnorm is scale invariant (see the module
    docstring). check the returned `completeness` before trusting a fixed-
    baseline map.
    """

    def __init__(
        self,
        encoder: nn.Module,
        classifier: nn.Module,
        device: torch.device,
    ) -> None:
        """
        Args:
            encoder: feature extractor (e.g. HybridEncoder)
            classifier: classification head (e.g. linear probe)
            device: torch device
        """
        self.encoder = encoder
        self.classifier = classifier
        self.device = device

    def _logit(self, x: torch.Tensor, target_class: int) -> torch.Tensor:
        # target class logit summed over the batch, so a single backward gives
        # the per-sample input gradient for every interpolation step at once.
        return self.classifier(self.encoder(x))[:, target_class].sum()

    def attribute(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 32,
        chunk_size: int = 8,
    ) -> dict[str, object]:
        """
        compute integrated gradients for one crop and one target class.

        Args:
            input_tensor: input crop, shape (1, C, Z, H, W)
            target_class: index of the class to explain
            baseline: reference input of "absence"; defaults to zeros (black).
                microscopy background is ~0, so a zero baseline is a reasonable
                default. any near-zero voxel gets ~0 attribution though (IG
                multiplies by input - baseline), so switch to a blurred baseline
                if background suppression looks wrong.
            steps: riemann steps along the baseline -> input path. more steps
                tighten the completeness error; validate with `completeness`.
            chunk_size: interpolation steps per forward/backward. lower this if
                a full-resolution global crop OOMs the backbone.

        Returns:
            dict with:
              attribution: signed (Z, H, W) numpy array
              completeness: relative error of the completeness axiom (see below)
              delta: signed (sum attribution) - (F(x) - F(baseline))
              target_score: target logit at the input
              baseline_score: target logit at the baseline
        """
        self.encoder.eval()
        self.classifier.eval()

        if input_tensor.size(0) != 1:
            raise ValueError(f"expected batch size 1, got {input_tensor.size(0)}")
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")

        x = input_tensor.to(self.device)
        if baseline is None:
            baseline = torch.zeros_like(x)
        else:
            baseline = baseline.to(self.device)
            if baseline.shape != x.shape:
                raise ValueError(
                    f"baseline shape {tuple(baseline.shape)} != input "
                    f"{tuple(x.shape)}"
                )

        delta_input = x - baseline

        # midpoint riemann: alphas at (k + 0.5) / steps, each weighted 1/steps.
        # the midpoint rule converges faster than left/right for the same step
        # count, which keeps the completeness error small at low `steps`.
        alphas = (torch.arange(steps, device=self.device) + 0.5) / steps

        grad_sum = torch.zeros_like(x)
        with torch.enable_grad():
            for start in range(0, steps, chunk_size):
                chunk_alphas = alphas[start : start + chunk_size]
                # interpolate: (n_chunk, C, Z, H, W). broadcast alpha over voxels.
                a = chunk_alphas.view(-1, 1, 1, 1, 1)
                interp = (baseline + a * delta_input).detach().requires_grad_(True)
                score = self._logit(interp, target_class)
                grads = torch.autograd.grad(score, interp)[0]
                # sum over the chunk; the 1/steps averaging happens once below.
                grad_sum += grads.sum(dim=0, keepdim=True)

        avg_grad = grad_sum / steps
        attribution = (delta_input * avg_grad)[0, 0]  # (Z, H, W)

        # completeness axiom: sum of attributions should equal the logit gap
        # between input and baseline. the relative error is a free convergence
        # check grad-cam cannot offer; if it is large, raise `steps`.
        with torch.no_grad():
            target_score = self._logit(x, target_class)
            baseline_score = self._logit(baseline, target_class)
        score_gap = (target_score - baseline_score).item()
        attr_sum = attribution.sum().item()
        delta = attr_sum - score_gap
        completeness = abs(delta) / (abs(score_gap) + 1e-8)

        return {
            "attribution": attribution.detach().cpu().numpy(),
            "completeness": completeness,
            "delta": delta,
            "target_score": target_score.item(),
            "baseline_score": baseline_score.item(),
        }

    def attribute_multiclass(
        self,
        input_tensor: torch.Tensor,
        num_classes: int,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 32,
        chunk_size: int = 8,
    ) -> dict[int, dict[str, object]]:
        """attribute every class for one crop; returns {class_idx: attribute()}."""
        return {
            class_idx: self.attribute(
                input_tensor, class_idx, baseline, steps, chunk_size
            )
            for class_idx in range(num_classes)
        }


class ExpectedGradients:
    """expected gradients (Erion et al. 2021): IG generalized over a baseline
    distribution instead of one fixed reference.

    EG(x)_i = E_{x'~D, a~U(0,1)} [ (x - x')_i * dF_c/dx_i (x' + a (x - x')) ]

    the expectation folds the baseline choice and the path position into one
    monte-carlo estimate. drawing baselines x' from real crops keeps every path
    inside the natural, normalized data regime, which is what makes this the
    right variant for a groupnorm (scale-invariant) encoder where a zero
    baseline is degenerate.
    """

    def __init__(
        self,
        encoder: nn.Module,
        classifier: nn.Module,
        device: torch.device,
    ) -> None:
        """
        Args:
            encoder: feature extractor (e.g. HybridEncoder)
            classifier: classification head (e.g. linear probe)
            device: torch device
        """
        self.encoder = encoder
        self.classifier = classifier
        self.device = device

    def _logit(self, x: torch.Tensor, target_class: int) -> torch.Tensor:
        # target class logit summed over the batch, so one backward gives the
        # per-sample input gradient for every (baseline, alpha) draw at once.
        return self.classifier(self.encoder(x))[:, target_class].sum()

    def _pool_scores(
        self, baselines: torch.Tensor, target_class: int, chunk_size: int
    ) -> torch.Tensor:
        # target logit F(x') for each baseline in the pool, on cpu. used to score
        # the sampled draws for the completeness diagnostic.
        scores = []
        with torch.no_grad():
            for start in range(0, baselines.size(0), chunk_size):
                chunk = baselines[start : start + chunk_size]
                scores.append(self.classifier(self.encoder(chunk))[:, target_class])
        return torch.cat(scores).cpu()

    def attribute(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        baselines: torch.Tensor,
        n_samples: int = 128,
        chunk_size: int = 8,
        seed: Optional[int] = None,
    ) -> dict[str, object]:
        """
        expected-gradients attribution for one crop and one target class.

        Args:
            input_tensor: input crop, shape (1, C, Z, H, W)
            target_class: index of the class to explain
            baselines: pool of reference crops, shape (M, C, Z, H, W), drawn from
                the data distribution. paths run from a sampled baseline to the
                input, so structurally-different crops give the best coverage.
            n_samples: monte-carlo (baseline, alpha) draws. estimator error
                shrinks ~1/sqrt(n_samples); raise it if the map looks noisy.
            chunk_size: draws per forward/backward. lower this if a full-
                resolution global crop OOMs the backbone.
            seed: optional seed for the baseline/alpha sampling, for reproducible
                maps. sampling is done on cpu so it is device-independent.

        Returns:
            dict with:
              attribution: signed (Z, H, W) numpy array
              completeness: relative error of the EG completeness identity
                  E[sum attr] = F(x) - E_{x'}[F(x')]
              delta: signed (sum attribution) - (F(x) - mean baseline score)
              target_score: target logit at the input
              baseline_score: mean target logit over the sampled baselines
              n_samples: draws used
        """
        self.encoder.eval()
        self.classifier.eval()

        if input_tensor.size(0) != 1:
            raise ValueError(f"expected batch size 1, got {input_tensor.size(0)}")
        if baselines.dim() != input_tensor.dim():
            raise ValueError(
                f"baselines must be (M, C, Z, H, W); got {tuple(baselines.shape)}"
            )
        if baselines.size(0) < 1:
            raise ValueError("baselines pool is empty")
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")

        x = input_tensor.to(self.device)
        baselines = baselines.to(self.device)
        n_baselines = baselines.size(0)

        # sample on cpu so the run is reproducible and device-independent (mps
        # lacks a generator). draw baseline indices with replacement and a
        # uniform path position per draw.
        cpu_gen = torch.Generator()
        if seed is not None:
            cpu_gen.manual_seed(seed)
        idx = torch.randint(0, n_baselines, (n_samples,), generator=cpu_gen)
        alphas = torch.rand(n_samples, generator=cpu_gen).to(self.device)

        attr_sum = torch.zeros_like(x)
        with torch.enable_grad():
            for start in range(0, n_samples, chunk_size):
                chunk_idx = idx[start : start + chunk_size]
                a = alphas[start : start + chunk_size].view(-1, 1, 1, 1, 1)
                bl = baselines[chunk_idx]  # (n_chunk, C, Z, H, W)
                diff = x - bl  # broadcast the single input against the baselines
                interp = (bl + a * diff).detach().requires_grad_(True)
                score = self._logit(interp, target_class)
                grads = torch.autograd.grad(score, interp)[0]
                # accumulate (x - x') * grad per draw; the 1/n mean is applied once.
                attr_sum += (diff * grads).sum(dim=0, keepdim=True)

        attribution = (attr_sum / n_samples)[0, 0]  # (Z, H, W)

        # completeness identity for EG: E[sum attr] = F(x) - E_{x'}[F(x')]. score
        # it against the mean over the *sampled* baselines (not the full pool) so
        # it measures path-integration error alone and is not polluted by the
        # baseline-sampling noise between the drawn set and the pool. relative
        # error is the monte-carlo convergence signal; raise n_samples if large.
        with torch.no_grad():
            target_score = self._logit(x, target_class).item()
            pool_scores = self._pool_scores(baselines, target_class, chunk_size)
        baseline_score = pool_scores[idx].mean().item()
        score_gap = target_score - baseline_score
        attr_sum_val = attribution.sum().item()
        delta = attr_sum_val - score_gap
        completeness = abs(delta) / (abs(score_gap) + 1e-8)

        return {
            "attribution": attribution.detach().cpu().numpy(),
            "completeness": completeness,
            "delta": delta,
            "target_score": target_score,
            "baseline_score": baseline_score,
            "n_samples": n_samples,
        }

    def attribute_multiclass(
        self,
        input_tensor: torch.Tensor,
        num_classes: int,
        baselines: torch.Tensor,
        n_samples: int = 128,
        chunk_size: int = 8,
        seed: Optional[int] = None,
    ) -> dict[int, dict[str, object]]:
        """attribute every class for one crop; returns {class_idx: attribute()}.

        a fixed seed is reused across classes so every class map is explained
        with the same sampled paths, keeping them comparable.
        """
        return {
            class_idx: self.attribute(
                input_tensor,
                class_idx,
                baselines,
                n_samples=n_samples,
                chunk_size=chunk_size,
                seed=seed,
            )
            for class_idx in range(num_classes)
        }


def reduce_over_z(attribution: np.ndarray, mode: str = "sum") -> np.ndarray:
    """collapse a signed (Z, H, W) attribution to a (H, W) heatmap.

    "sum" preserves the completeness intuition: summing the reduced map equals
    summing the full volume. "max_abs" keeps, per pixel, the slice with the
    largest magnitude (sign preserved), which is sharper when the informative
    depth varies across the crop.
    """
    if attribution.ndim != 3:
        raise ValueError(f"expected (Z, H, W), got shape {attribution.shape}")
    if mode == "sum":
        return attribution.sum(axis=0)
    if mode == "max_abs":
        z_of_max = np.abs(attribution).argmax(axis=0)
        return np.take_along_axis(
            attribution, z_of_max[None], axis=0
        ).squeeze(0)
    raise ValueError(f"unknown mode {mode!r}; use 'sum' or 'max_abs'")


def _symmetric_vmax(maps: list[np.ndarray], percentile: float = 99.0) -> float:
    """shared symmetric color limit so signed maps are comparable in one figure.

    a high percentile rather than the raw max keeps a few hot voxels from
    flattening the rest of the scale.
    """
    stacked = np.concatenate([np.abs(m).ravel() for m in maps])
    vmax = float(np.percentile(stacked, percentile)) if stacked.size else 0.0
    return vmax if vmax > 1e-12 else 1.0


def _signed_overlay(
    image_2d: np.ndarray,
    heatmap: np.ndarray,
    vmax: float,
    cmap_name: str = "seismic",
    max_alpha: float = 0.7,
) -> np.ndarray:
    """blend a signed heatmap over a grayscale crop.

    alpha scales with attribution magnitude so near-zero regions stay
    transparent and the cell shows through, while strong positive/negative
    voxels read as saturated diverging colors.
    """
    cmap = plt.get_cmap(cmap_name)
    norm = np.clip(heatmap / vmax, -1.0, 1.0)
    heat_rgb = cmap((norm * 0.5 + 0.5))[..., :3]  # diverging, 0 -> center

    gray = image_2d
    if gray.max() > gray.min():
        gray = (gray - gray.min()) / (gray.max() - gray.min())
    image_rgb = np.stack([gray] * 3, axis=-1)

    alpha = (np.abs(norm) * max_alpha)[..., None]
    return image_rgb * (1.0 - alpha) + heat_rgb * alpha


def visualize_eg_multiclass(
    crop_2d: np.ndarray,
    attributions: dict[int, np.ndarray],
    class_names: list[str],
    title: str = "Expected Gradients by Class",
    save_path: Optional[str] = None,
    figsize: Optional[tuple[int, int]] = None,
    pred_class: Optional[int] = None,
    probs: Optional[dict[int, float]] = None,
    completeness: Optional[dict[int, float]] = None,
) -> None:
    """signed EG panel: overlay row + raw diverging-heatmap row per class.

    the leading column is the original crop. top row overlays each class's
    signed attribution on the crop; bottom row shows the raw signed heatmap on a
    shared symmetric red/blue scale (one colorbar) so classes are comparable.
    red pushes the logit toward the class, blue pushes away.

    Args:
        crop_2d: max-projected crop (H, W)
        attributions: {class_idx: z-reduced signed heatmap (H, W)}
        class_names: class name per index
        pred_class: predicted class index to highlight
        probs: {class_idx: probability} shown in titles
        completeness: {class_idx: relative completeness error} shown in titles
    """
    n_classes = len(attributions)
    n_cols = n_classes + 1
    if figsize is None:
        figsize = (4 * n_cols, 8)

    vmax = _symmetric_vmax(list(attributions.values()))

    fig, axes = plt.subplots(2, n_cols, figsize=figsize, squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0, 0].imshow(crop_2d, cmap="gray")
    axes[0, 0].set_title("Original", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")

    hm_im = None
    for offset, (class_idx, heatmap) in enumerate(sorted(attributions.items())):
        col = offset + 1
        name = class_names[class_idx]
        header = name
        if probs is not None and class_idx in probs:
            header += f"\np={probs[class_idx]:.2f}"
        if completeness is not None and class_idx in completeness:
            header += f"\nconv={completeness[class_idx] * 100:.1f}%"
        if pred_class is not None and class_idx == pred_class:
            header += "  (pred)"

        overlay = _signed_overlay(crop_2d, heatmap, vmax)
        axes[0, col].imshow(overlay)
        color = (
            "#c62828"
            if (pred_class is not None and class_idx == pred_class)
            else "black"
        )
        axes[0, col].set_title(header, fontsize=11, fontweight="bold", color=color)
        axes[0, col].axis("off")

        hm_im = axes[1, col].imshow(
            heatmap, cmap="seismic", vmin=-vmax, vmax=vmax
        )
        axes[1, col].set_title("signed attribution", fontsize=9)
        axes[1, col].axis("off")

    if hm_im is not None:
        fig.colorbar(
            hm_im,
            ax=axes[1, 1:].tolist(),
            fraction=0.046,
            pad=0.02,
            label="toward class (+) / away (-)",
        )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_eg_zslices(
    crop: np.ndarray,
    attribution: np.ndarray,
    class_name: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """per-z-slice signed attribution, the detail grad-cam cannot produce.

    one column per z slice: the crop slice on top, its signed EG below on a
    shared symmetric scale. surfaces which depths drove the decision, which
    matters for anisotropic microscopy where z is the low-resolution axis.

    Args:
        crop: input volume (Z, H, W)
        attribution: signed attribution (Z, H, W)
        class_name: class being explained, for the title
    """
    if crop.shape != attribution.shape:
        raise ValueError(
            f"crop {crop.shape} and attribution {attribution.shape} must match"
        )
    n_z = attribution.shape[0]
    vmax = _symmetric_vmax([attribution])

    fig, axes = plt.subplots(2, n_z, figsize=(2.2 * n_z, 4.6), squeeze=False)
    fig.suptitle(
        title or f"Expected Gradients per z-slice - {class_name}",
        fontsize=13,
        fontweight="bold",
    )

    hm_im = None
    for z in range(n_z):
        axes[0, z].imshow(crop[z], cmap="gray")
        axes[0, z].set_title(f"z={z}", fontsize=9)
        axes[0, z].axis("off")

        hm_im = axes[1, z].imshow(
            attribution[z], cmap="seismic", vmin=-vmax, vmax=vmax
        )
        axes[1, z].axis("off")

    if hm_im is not None:
        fig.colorbar(
            hm_im, ax=axes[1, :].tolist(), fraction=0.046, pad=0.02
        )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
