"""Expected Gradients: input-space attribution for the frozen encoder + probe.

warning: plain integrated gradients against a single fixed all-zeros baseline
does NOT work on this encoder. groupnorm makes the encoder scale invariant, so a
straight-line path from zeros collapses through a near-discontinuity at the
origin. no practical step count can then satisfy the completeness axiom. the
fixed-baseline code is gone rather than left as a tempting alternative. real
crops as baselines keep every path inside the natural, normalized regime, and
they restore a well-behaved estimate. if you reintroduce a fixed baseline, check
the reported completeness error before you believe any map from it.

`ExpectedGradients` is the only entry point.

expected gradients (Erion et al. 2021), "Improving performance of deep learning
models with axiomatic attribution priors": https://arxiv.org/abs/1906.10670.
expected gradients generalizes integrated gradients (Sundararajan et al. 2017,
https://arxiv.org/abs/1703.01365) from a single fixed baseline to an expectation
over a baseline distribution.

grad-cam weights a conv feature map and lives at that layer's coarse resolution.
EG instead attributes the target class logit all the way back to the raw input
voxels (1, Z, H, W). two consequences matter for this pipeline:
  - the attribution is already at input resolution, so no upsampling step is
    necessary.
  - the z axis survives. grad-cam hooks the 2d backbone, which sits after
    ZAttentionPool collapses depth, so grad-cam has zero z resolution. EG
    differentiates through the pool and recovers per-slice attribution.

the attribution is signed: positive voxels push the logit toward the class,
negative voxels push the logit away. therefore the visualizers here use a
diverging colormap rather than the positive-only jet scale of the grad-cam
visualizers.
"""

from __future__ import annotations

import matplotlib

# the matplotlib.use call must precede the pyplot import, because tk aborts at
# exit on headless nodes.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


class ExpectedGradients:
    """expected gradients (Erion et al. 2021), see the module docstring.

    EG(x)_i = E_{x'~D, a~U(0,1)} [ (x - x')_i * dF_c/dx_i (x' + a (x - x')) ]

    one monte-carlo estimate folds in both the baseline choice and the path
    position.
    """

    def __init__(
        self,
        encoder: nn.Module,
        classifier: nn.Module,
        device: torch.device,
    ) -> None:
        self.encoder = encoder
        self.classifier = classifier
        self.device = device

    def _logit(self, x: torch.Tensor, target_class: int) -> torch.Tensor:
        # the method sums the target class logit over the batch. one backward
        # pass then gives the per-sample input gradient for every
        # (baseline, alpha) draw at once.
        return self.classifier(self.encoder(x))[:, target_class].sum()

    def _pool_scores(
        self, baselines: torch.Tensor, target_class: int, chunk_size: int
    ) -> torch.Tensor:
        # F(x') per baseline, on cpu, for the completeness diagnostic.
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
            baselines: pool of reference crops (M, C, Z, H, W) from the data
                distribution. structurally-different crops cover best.
            n_samples: number of monte-carlo (baseline, alpha) draws. the error
                shrinks as ~1/sqrt(n_samples). raise n_samples if the map looks
                noisy.
            chunk_size: draws per forward and backward pass. lower chunk_size if
                a full-resolution global crop OOMs the backbone.
            seed: seed for the baseline draws and the alpha draws. a fixed seed
                makes the maps reproducible.

        the method returns a dict of: attribution (signed (Z, H, W) array),
        completeness (relative error of E[sum attr] = F(x) - E_{x'}[F(x')]),
        delta, target_score, baseline_score, n_samples.
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

        # the code draws the samples on cpu, so that the run is
        # device-independent. mps has no generator.
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
                diff = x - bl  # the single input broadcasts against the baselines
                interp = (bl + a * diff).detach().requires_grad_(True)
                score = self._logit(interp, target_class)
                grads = torch.autograd.grad(score, interp)[0]
                # the loop accumulates (x - x') * grad per draw. the code applies
                # the 1/n mean once.
                attr_sum += (diff * grads).sum(dim=0, keepdim=True)

        attribution = (attr_sum / n_samples)[0, 0]  # (Z, H, W)

        # completeness: E[sum attr] = F(x) - E_{x'}[F(x')]. the code scores
        # against the mean over the *sampled* baselines rather than over the
        # full pool. the check therefore measures the path-integration error
        # alone, and not the noise from the baseline draws. a large relative
        # error means that you must raise n_samples.
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
        """the method attributes every class for one crop.

        the method returns {class_idx: attribute()}. the method reuses one fixed
        seed across the classes. every class map therefore uses the same sampled
        paths, and the maps stay comparable.
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
    """the function collapses a signed (Z, H, W) attribution to a (H, W) heatmap.

    "sum" preserves the completeness intuition: the sum of the reduced map
    equals the sum of the full volume. "max_abs" keeps, for each pixel, the
    slice with the largest magnitude, and it keeps the sign. "max_abs" is
    sharper when the informative depth varies across the crop.
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
    """shared symmetric color limit, so that signed maps are comparable in one figure.

    the function uses a high percentile rather than the raw max. a few hot
    voxels therefore cannot flatten the rest of the scale.
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
    """the function blends a signed heatmap over a grayscale crop.

    alpha scales with the attribution magnitude. near-zero regions therefore
    stay transparent and the cell shows through. strong positive voxels and
    strong negative voxels read as saturated diverging colors.
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
    """signed EG panel: one overlay row and one raw diverging-heatmap row per class.

    the leading column is the original crop. the bottom row shares one symmetric
    red and blue scale, so that the classes are comparable. red pushes the logit
    toward the class, blue pushes the logit away.

    Args:
        crop_2d: max-projected crop (H, W)
        attributions: {class_idx: z-reduced signed heatmap (H, W)}
        class_names: class name per index
        pred_class: index of the predicted class. the function highlights that
            column.
        probs: {class_idx: probability}. the titles show these values.
        completeness: {class_idx: relative completeness error}. the titles show
            these values.
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
    """per-z-slice signed attribution, the detail that grad-cam cannot produce.

    there is one column per z slice. the crop is on top and its signed EG is
    below, on a shared symmetric scale. you can then see which depths drove the
    decision.
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
