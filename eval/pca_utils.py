"""exact, deterministic PCA and embedding preprocessing, shared by the eval scripts.

eval/latent_space.py used to compute PCA with torch.pca_lowrank, which is a
*randomized* algorithm, and that script never seeded torch's global RNG (its
cfg.seed only reaches the dataloader workers via worker_init_fn). So two runs
over byte-identical embeddings produced different PC1/PC2, and a static plot
never quite matched anything computed later from the same data.

Two things made that worse than it sounds:

  * it asked for exactly as many basis vectors as components (q = n_components,
    no oversampling), which is the least accurate way to use a randomized SVD;
  * randomized SVD degrades most when the spectrum is flat, and this project's
    is about as flat as it gets -- PC1 ~5.7% vs PC2 ~5.4% of variance, so the
    2D plane is well determined but the axes *within* it are ill-conditioned
    and a small perturbation rotates them a lot.

This module replaces that with an exact float64 SVD plus a fixed sign
convention. At 128 latent dims an exact decomposition is instant, so the
randomization bought nothing in the first place.

numpy-only and torch-free on purpose: eval/latent_traversal_video.py imports
this and is meant to run on a laptop with no torch installed.
"""

from __future__ import annotations

import numpy as np


def preprocess_embeddings(X: np.ndarray, mode: str) -> np.ndarray:
    """prepare raw encoder embeddings for projection.

    l2 is the default everywhere because raw encoder feature norms vary ~2.3x
    across crops and would otherwise dominate PCA/UMAP distances. It also makes
    Euclidean geometry agree exactly with the cosine metric used by
    eval/knn_behavior.py: for unit vectors ||a-b||^2 = 2 - 2*cos(a,b), so
    neighbor orderings are identical.
    """
    X = np.asarray(X, dtype=np.float64)
    if mode == "raw":
        return X
    if mode == "l2":
        return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    if mode == "standardize":
        return (X - X.mean(axis=0, keepdims=True)) / np.maximum(
            X.std(axis=0, keepdims=True), 1e-6
        )
    raise ValueError(f"unknown embedding preprocess mode: {mode}")


def deterministic_sign(components: np.ndarray) -> np.ndarray:
    """flip each component so its largest-magnitude loading is positive.

    an SVD determines each component only up to sign, and neither numpy nor
    torch pins it, so the same data can yield a mirrored plot on consecutive
    runs. keying the sign to the largest-magnitude loading makes orientation a
    property of the data rather than of the linear-algebra backend.

    components: (d, k), one component per column. returns a copy.
    """
    components = np.asarray(components, dtype=np.float64)
    if components.ndim != 2:
        raise ValueError(f"expected a 2D (d, k) array, got shape {components.shape}")
    out = components.copy()
    for j in range(out.shape[1]):
        column = out[:, j]
        pivot = int(np.argmax(np.abs(column)))
        if column[pivot] < 0.0:
            out[:, j] = -column
    return out


def pca_fit(
    X: np.ndarray, n_components: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """exact PCA basis. returns (components (d, q), mean (d,), explained (q,)).

    split out from pca_project so a projection can be *fit* on one set of points
    and *applied* to another -- e.g. fitting on every annotated crop but stepping
    through only the deduplicated ones, which keeps a walk comparable to a static
    plot of the same annotations.

    q is clamped to what the data can support. `explained` is each kept
    component's share of the *total* variance of the centered data -- the same
    definition the previous torch implementation printed, so diagnostics stay
    comparable with older logs.

    components are nested by construction: fitting q=3 and keeping the first two
    columns gives exactly the q=2 fit. that is what lets eval/latent_space.py
    derive its 2D and 3D PCA views from one decomposition.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected a 2D (n, d) array, got shape {X.shape}")
    n, d = X.shape
    q = min(n_components, n - 1, d)
    if q < 1:
        raise ValueError(f"cannot take {n_components} components from shape {X.shape}")
    mean = X.mean(axis=0)
    Xc = X - mean
    _, s, Vh = np.linalg.svd(Xc, full_matrices=False)
    V = deterministic_sign(Vh.T[:, :q])
    total = max(float((Xc ** 2).sum()), 1e-12)
    explained = (s[:q] ** 2) / total
    return V, mean, explained


def pca_transform(X: np.ndarray, components: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """project X onto a basis from pca_fit. returns coords (n, q) float64."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected a 2D (n, d) array, got shape {X.shape}")
    if X.shape[1] != components.shape[0]:
        raise ValueError(
            f"X has {X.shape[1]} features but the basis expects {components.shape[0]}"
        )
    return (X - mean) @ components


def pca_project(X: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """exact PCA fit and applied to the same points. (coords (n, q), explained (q,))."""
    components, mean, explained = pca_fit(X, n_components)
    return pca_transform(X, components, mean), explained
