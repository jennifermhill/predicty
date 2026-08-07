"""exact and deterministic PCA, and the embedding preprocess helpers the eval scripts share.

this module uses an exact float64 SVD and a fixed sign convention. the module
replaces an unseeded torch.pca_lowrank. the randomization of torch.pca_lowrank
gave different PC1/PC2 on byte-identical embeddings. the error was large here,
because the spectrum of this project is nearly flat (PC1 ~5.7% against PC2
~5.4%), and a randomized SVD degrades exactly at a flat spectrum. therefore you
cannot trust the PCA figures in run dirs from before 2026-07-28.

at 128 latent dims the exact decomposition is instant, so the change lost
nothing.

this module uses numpy only and stays torch-free on purpose:
eval/latent_traversal_video.py imports this module, and that script must run on
a laptop with no torch installed.
"""

from __future__ import annotations

import numpy as np


def preprocess_embeddings(X: np.ndarray, mode: str) -> np.ndarray:
    """prepare raw encoder embeddings for projection.

    l2 is the default everywhere. raw feature norms vary ~2.3x across crops, so
    raw norms can dominate PCA/UMAP distances. unit vectors also make Euclidean
    geometry agree exactly with the cosine metric of eval/knn_behavior.py.
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
    """flip each component so that its largest-magnitude loading is positive.

    an SVD determines each component only up to a sign, and neither numpy nor
    torch pins the sign. therefore the same data can give a mirrored plot on two
    consecutive runs. a sign keyed to the largest-magnitude loading makes the
    orientation a property of the data, not of the linear-algebra backend.

    components: (d, k), one component per column. this function returns a copy.
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

    this function is separate from pca_project, so the caller can *fit* a
    projection on one set of points and *apply* it to another set. for example,
    the caller can fit on every annotated crop and then step through only the
    deduplicated crops.

    this function clamps q to what the data can support. `explained` gives the
    share of each kept component in the *total* variance of the centered data,
    which matches what older logs printed.

    the components are nested by construction. a q=3 fit holds the exact q=2 fit
    in its first two columns, so eval/latent_space.py derives its 2D PCA view
    and its 3D PCA view from one decomposition.
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
    """fit an exact PCA and apply it to the same points. (coords (n, q), explained (q,))."""
    components, mean, explained = pca_fit(X, n_components)
    return pca_transform(X, components, mean), explained
