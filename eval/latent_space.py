"""latent-space visualization for a frozen WS-DINO encoder.

this samples normal training crops, embeds them with a checkpointed encoder,
then projects the high-dimensional embeddings to 2D for inspection.

Usage:
    python eval/latent_space.py --config configs/train.yaml --checkpoint checkpoints/latest.pt
    python eval/latent_space.py --config configs/train.yaml --checkpoint checkpoints/latest.pt --n-crops 20000 --method umap

Outputs:
    <out-dir>/embeddings.npz         raw embeddings + 2D coords per method + labels
    <out-dir>/<method>/metadata.csv  crop metadata + that method's coords per point
    <out-dir>/<method>/latent_by_*.png  scatter plots colored by metadata fields

PCA always runs so every report has a deterministic, variance-faithful view.
when umap-learn (or scikit-learn for t-SNE) is installed, that manifold method
runs too, giving both a umap/ and a pca/ subdirectory side by side.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

# allow running as `python eval/latent_space.py` from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

from configs.config import Config
from data.dataset import MultiZarrDataset, ZarrCellDataset, worker_init_fn
from models.encoder import HybridEncoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to the training yaml config")
    p.add_argument("--checkpoint", required=True, help="path to latest.pt/final.pt/step_*.pt")
    p.add_argument("--device", default=None, help="override cfg.device, e.g. cpu or cuda:0")
    p.add_argument("--out-dir", default=None, help="output directory; defaults under latent_space/")
    p.add_argument("--n-crops", type=int, default=10000, help="number of global crops to embed")
    p.add_argument("--batch-size", type=int, default=None, help="override cfg.batch_size for extraction")
    p.add_argument("--num-workers", type=int, default=None, help="override cfg.num_workers for extraction")
    p.add_argument("--seed", type=int, default=42, help="sampling seed")
    p.add_argument(
        "--views",
        choices=("both", "t1", "t2"),
        default="both",
        help="which global crops to include as points",
    )
    p.add_argument(
        "--method",
        choices=("auto", "umap", "tsne", "pca"),
        default="auto",
        help="2D projection method. auto tries UMAP, then t-SNE, then PCA.",
    )
    p.add_argument("--pca-dim", type=int, default=50, help="pre-reduction dim for UMAP/t-SNE")
    p.add_argument("--umap-neighbors", type=int, default=30)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--tsne-perplexity", type=float, default=30.0)
    p.add_argument("--plot-size", type=float, default=8.0)
    p.add_argument(
        "--embedding-preprocess",
        choices=("raw", "l2", "standardize"),
        default="l2",
        help=(
            "preprocess encoder embeddings before projection. l2 is usually "
            "best for comparing representation geometry because raw encoder "
            "feature norms can dominate PCA/UMAP distances."
        ),
    )
    return p.parse_args()


def disable_augmentations(cfg: Config) -> None:
    """turn off stochastic augmentations for deterministic visual diagnostics."""
    for name in ("aug_intensity", "aug_gamma", "aug_noise", "aug_blur"):
        getattr(cfg, name).enabled = False
    # Keep per-crop normalization on because it is the encoder input convention.


def build_dataset(cfg: Config):
    if cfg.zarr_paths:
        return MultiZarrDataset(cfg)
    if cfg.zarr_path is None:
        raise ValueError("latent-space extraction needs cfg.zarr_path or cfg.zarr_paths")
    return ZarrCellDataset(cfg)


def as_numpy_1d(value: Any, batch_size: int) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().reshape(-1)
    arr = np.asarray(value)
    if arr.ndim == 0:
        return np.full(batch_size, arr.item())
    return arr.reshape(-1)


def view_indices(views: str) -> list[int]:
    if views == "t1":
        return [0]
    if views == "t2":
        return [1]
    return [0, 1]


@torch.no_grad()
def extract_embeddings(
    cfg: Config,
    encoder: nn.Module,
    device: torch.device,
    n_crops: int,
    views: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    encoder.eval()
    embeddings: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    selected_views = view_indices(views)

    ds = build_dataset(cfg)
    if isinstance(ds, MultiZarrDataset):
        n_datasets = len(ds.children)
        zarr_paths = [child.cfg.zarr_path or "" for child in ds.children]
        children = ds.children
        assert zarr_paths == list(cfg.zarr_paths), (
            "ds.children order doesn't match cfg.zarr_paths -- annotation "
            "dataset_idx remapping would silently pull the wrong zarr file."
        )
    else:
        n_datasets = 1
        zarr_paths = [cfg.zarr_path or ""]
        children = [ds]


    if cfg.annotation_path is None:
        raise ValueError("cfg.annotation_path must be set for latent space extraction")
    ann_df = pd.read_json(cfg.annotation_path, lines=True)
    ann_df = ann_df[ann_df["label"] != "empty"]

    if cfg.zarr_paths is None:
        raise ValueError("cfg.zarr_paths must be set for latent space extraction")
    ann_df = ann_df[ann_df["zarr_path"].isin(cfg.zarr_paths)]
    ann_df = ann_df.drop(columns=["timestamp"]).reset_index(drop=True)

    zarr_path_to_idx = {p: i for i, p in enumerate(cfg.zarr_paths)}
    ann_df["dataset_idx"] = ann_df["zarr_path"].map(zarr_path_to_idx)

    for col in ann_df.columns:
        try:
            ann_df[col] = pd.to_numeric(ann_df[col], downcast="integer")
        except (ValueError, TypeError):
            pass

    ann_batch_size = cfg.batch_size
    ann_globals: list[torch.Tensor] = []
    ann_metas: list[dict[str, Any]] = []
    ann_labels: list[Any] = []
    ann_uids: list[Any] = []

    def flush_annotation_batch() -> None:
        if not ann_globals:
            return
        globals_ = torch.stack(ann_globals)  # (b, 2, 1, Z, H, W)
        b = globals_.size(0)
        crops = torch.cat([globals_[:, i] for i in selected_views], dim=0).to(
            device, non_blocking=True
        )
        emb = encoder(crops).cpu()
        for offset, view_idx in enumerate(selected_views):
            s, e = offset * b, offset * b + b
            embeddings.append(emb[s:e])
            for i in range(b):
                meta = ann_metas[i]
                rows.append({
                    "point_idx": len(rows),
                    "uid": ann_uids[i],
                    "dataset_idx": int(meta["dataset_idx"]),
                    "dataset_name": Path(zarr_paths[meta["dataset_idx"]]).name,
                    "n_datasets": n_datasets,
                    "view": "t1" if view_idx == 0 else "t2",
                    "view_idx": int(view_idx),
                    "t": int(meta["t1"] if view_idx == 0 else meta["t2"]),
                    "z": int(meta["z"]),
                    "z_end": int(meta["z_end"]),
                    "global_y": int(meta["global_y"]),
                    "global_x": int(meta["global_x"]),
                    "frame_y": int(meta["frame_y"]),
                    "frame_x": int(meta["frame_x"]),
                    "spatial_region_idx": int(meta["spatial_region_idx"]),
                    "frame_gap": int(meta["frame_gap"]),
                    "label": ann_labels[i],
                })
        ann_globals.clear()
        ann_metas.clear()
        ann_labels.clear()
        ann_uids.clear()

    n_skipped = 0
    for _, r in ann_df.iterrows():
        if len(rows) + len(ann_globals) >= n_crops:
            break
        ds_idx = int(r["dataset_idx"])
        child = children[ds_idx]
        sample = child._annotation_tensor(
            t1=int(r["t1"]),
            t2=int(r["t2"]),
            z=int(r["z"]),
            global_x=int(r["global_x"]),
            global_y=int(r["global_y"]),
            dataset_idx=ds_idx,
        )
        ann_globals.append(sample["globals"])
        ann_metas.append(sample["meta"])
        ann_labels.append(r["label"])
        ann_uids.append(r["uid"])
        if len(ann_globals) >= ann_batch_size:
            flush_annotation_batch()
            print(f"  [extract-annotated] {len(rows)}/{n_crops} crops")
    flush_annotation_batch()

    if n_skipped:
        print(f"[extract] WARNING: skipped {n_skipped} annotation rows with dataset_idx/zarr_path mismatch")

    n_annotated = len(rows)
    print(f"[extract] embedded {n_annotated} crops from annotations")

    if len(rows) < n_crops:
        loader = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            worker_init_fn=worker_init_fn,
            persistent_workers=cfg.num_workers > 0,
            pin_memory=device.type == "cuda",
        )

        t0 = time.time()

        for step, batch in enumerate(loader):
            globals_ = batch["globals"]  # (B, 2, 1, Z, H, W)
            weak_label = batch["weak_label"]
            meta = batch["meta"]
            batch_size = globals_.size(0)

            crops = torch.cat([globals_[:, i] for i in selected_views], dim=0).to(
                device, non_blocking=True
            )
            emb = encoder(crops).cpu()

            for offset, view_idx in enumerate(selected_views):
                start = offset * batch_size
                end = start + batch_size
                embeddings.append(emb[start:end])

                dataset_idx = as_numpy_1d(weak_label[:, 0], batch_size)
                frame_gap = as_numpy_1d(weak_label[:, 1], batch_size)
                t = as_numpy_1d(meta["t1" if view_idx == 0 else "t2"], batch_size)
                z = as_numpy_1d(meta["z"], batch_size)
                z_end = as_numpy_1d(meta["z_end"], batch_size)
                y = as_numpy_1d(meta["global_y"], batch_size)
                x = as_numpy_1d(meta["global_x"], batch_size)
                frame_y = as_numpy_1d(meta["frame_y"], batch_size)
                frame_x = as_numpy_1d(meta["frame_x"], batch_size)
                spatial_region_idx = as_numpy_1d(meta["spatial_region_idx"], batch_size)

                for i in range(batch_size):
                    ds_idx = int(dataset_idx[i])
                    rows.append({
                        "point_idx": len(rows),
                        "uid": "",
                        "dataset_idx": ds_idx,
                        "dataset_name": Path(zarr_paths[ds_idx]).name if ds_idx < len(zarr_paths) else "",
                        "n_datasets": n_datasets,
                        "view": "t1" if view_idx == 0 else "t2",
                        "view_idx": int(view_idx),
                        "t": int(t[i]),
                        "z": int(z[i]),
                        "z_end": int(z_end[i]),
                        "global_y": int(y[i]),
                        "global_x": int(x[i]),
                        "frame_y": int(frame_y[i]),
                        "frame_x": int(frame_x[i]),
                        "spatial_region_idx": int(spatial_region_idx[i]),
                        "frame_gap": int(frame_gap[i]),
                        "label": None,
                    })

            seen = len(rows)
            if step == 0 or seen >= n_crops or step % 10 == 0:
                elapsed = time.time() - t0
                rate = (seen - n_annotated) / max(elapsed, 1e-6)
                print(f"  [extract-random] {min(seen, n_crops)}/{n_crops} crops  ({rate:.1f} crops/s)")
            if seen >= n_crops:
                break

    X = torch.cat(embeddings, dim=0).numpy()[:n_crops]
    return X, rows[:n_crops]

def load_cached_embeddings(path: Path, cfg: Config) -> tuple[np.ndarray, list[dict[str, Any]]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cached embeddings file not found at {path}")
    data = np.load(path, allow_pickle=False)
    X = data["embeddings"]
    n = X.shape[0]

    zarr_paths = list(cfg.zarr_paths) if cfg.zarr_paths else [cfg.zarr_path or ""]
    if not zarr_paths or all(p == "" for p in zarr_paths):
        raise ValueError("No zarr paths configured for loading embeddings")

    rows: list[dict[str, Any]] = []
    for i in range(n):
        ds_idx = int(data["dataset_idx"][i])
        view_idx = int(data["view_idx"][i])
        label = str(data["label"][i])
        rows.append({
            "point_idx": i,
            "dataset_idx": ds_idx,
            "dataset_name": Path(zarr_paths[ds_idx]).name if ds_idx < len(zarr_paths) else "",
            "view": "t1" if view_idx == 0 else "t2",
            "view_idx": view_idx,
            "t": int(data["t"][i]),
            "z": int(data["z"][i]),
            "label": label if label else None,
        })
    return X, rows

def pca_project(X: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    X_t = torch.from_numpy(X).float()
    X_t = X_t - X_t.mean(dim=0, keepdim=True)
    q = min(n_components, X_t.shape[0] - 1, X_t.shape[1])
    _, s, v = torch.pca_lowrank(X_t, q=q, center=False)
    coords = X_t @ v[:, :q]
    denom = torch.clamp((X_t ** 2).sum(), min=1e-12)
    explained = (s[:q] ** 2 / denom).cpu().numpy()
    return coords.cpu().numpy(), explained


def preprocess_embeddings(X: np.ndarray, mode: str) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    if mode == "raw":
        return X
    if mode == "l2":
        denom = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(denom, 1e-12)
    if mode == "standardize":
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        return (X - mean) / np.maximum(std, 1e-6)
    raise ValueError(f"unknown embedding preprocess mode: {mode}")


def print_embedding_diagnostics(X_raw: np.ndarray, X_projected: np.ndarray) -> None:
    norms = np.linalg.norm(X_raw, axis=1)
    feature_var = X_raw.var(axis=0)
    _, explained_raw = pca_project(X_raw, min(10, X_raw.shape[0] - 1, X_raw.shape[1]))
    _, explained_projected = pca_project(
        X_projected, min(10, X_projected.shape[0] - 1, X_projected.shape[1])
    )
    print(
        "[diagnostics] raw embedding norm "
        f"mean={norms.mean():.4f} std={norms.std():.4f} "
        f"min={norms.min():.4f} max={norms.max():.4f}"
    )
    print(
        "[diagnostics] raw feature variance "
        f"mean={feature_var.mean():.6f} median={np.median(feature_var):.6f}"
    )
    print(
        "[diagnostics] PCA explained variance raw "
        f"PC1={explained_raw[0]*100:.1f}% "
        f"PC2={explained_raw[1]*100 if explained_raw.size > 1 else 0.0:.1f}% "
        f"PC1-10={explained_raw.sum()*100:.1f}%"
    )
    print(
        "[diagnostics] PCA explained variance after preprocess "
        f"PC1={explained_projected[0]*100:.1f}% "
        f"PC2={explained_projected[1]*100 if explained_projected.size > 1 else 0.0:.1f}% "
        f"PC1-10={explained_projected.sum()*100:.1f}%"
    )


def reduce_for_manifold(X: np.ndarray, pca_dim: int) -> np.ndarray:
    if X.shape[1] <= pca_dim:
        return X
    coords, explained = pca_project(X, min(pca_dim, X.shape[0] - 1, X.shape[1]))
    print(
        f"[project] PCA pre-reduction to {coords.shape[1]} dims "
        f"(explained first 2 = {explained[:2].sum()*100:.1f}%)"
    )
    return coords


def best_available_manifold() -> str:
    """pick the manifold method to pair with PCA: umap if available, else
    t-SNE, else nothing extra (pca only)."""
    try:
        import umap  # type: ignore  # noqa: F401

        return "umap"
    except Exception:
        try:
            from sklearn.manifold import TSNE  # noqa: F401

            return "tsne"
        except Exception:
            return "pca"


def resolve_methods(method: str) -> list[str]:
    """which 2D projections to produce. naming a method explicitly produces
    only that one; the default `auto` pairs the best available manifold method
    with PCA so a run gets both a manifold view and a deterministic,
    variance-faithful reference side by side."""
    if method != "auto":
        return [method]
    manifold = best_available_manifold()
    # dedupe when no manifold lib is installed and auto falls back to pca.
    return ["pca"] if manifold == "pca" else [manifold, "pca"]


def project_single(args: argparse.Namespace, X: np.ndarray, method: str) -> np.ndarray:
    """project preprocessed embeddings X to 2D with one named method."""
    if method == "umap":
        try:
            import umap  # type: ignore
        except Exception as e:
            raise RuntimeError("UMAP requested but umap-learn is not installed") from e
        X_red = reduce_for_manifold(X, args.pca_dim)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            metric="euclidean",
            random_state=args.seed,
        )
        return reducer.fit_transform(X_red)

    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except Exception as e:
            raise RuntimeError("t-SNE requested but scikit-learn is not installed") from e
        X_red = reduce_for_manifold(X, args.pca_dim)
        perplexity = min(args.tsne_perplexity, max(5.0, (X.shape[0] - 1) / 3))
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=args.seed,
        )
        return reducer.fit_transform(X_red)

    if method == "pca":
        coords, explained = pca_project(X, 2)
        print(f"[project] PCA explained variance in 2D: {explained.sum()*100:.1f}%")
        return coords

    raise ValueError(f"unknown projection method: {method}")


def write_metadata_csv(path: Path, rows: list[dict[str, Any]], coords: np.ndarray) -> None:
    fieldnames = ["coord_x", "coord_y"] + list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, xy in zip(rows, coords):
            writer.writerow({"coord_x": float(xy[0]), "coord_y": float(xy[1]), **row})


def save_npz(
    path: Path,
    X: np.ndarray,
    coords_by_method: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
) -> None:
    dataset_idx_arr = np.asarray([r["dataset_idx"] for r in rows], dtype=np.int64)
    t = np.asarray([r["t"] for r in rows], dtype=np.int64)
    z = np.asarray([r["z"] for r in rows], dtype=np.int64)
    view_idx = np.asarray([r["view_idx"] for r in rows], dtype=np.int64)
    # "" for points with no annotation (e.g. randomly-sampled crops in
    # extract_embeddings where row["label"] is None).
    label = np.asarray(["" if r.get("label") is None else str(r["label"]) for r in rows])
    arrays: dict[str, np.ndarray] = {
        "embeddings": X.astype(np.float32),
        "dataset_idx": dataset_idx_arr,
        "t": t,
        "z": z,
        "view_idx": view_idx,
        "label": label,
    }
    # one coords_<method> array per projection so downstream tools can load
    # any view from a single file.
    for method, coords in coords_by_method.items():
        arrays[f"coords_{method}"] = coords.astype(np.float32)
    np.savez_compressed(path, **arrays)


def plot_categorical(
    coords: np.ndarray,
    values: list[Any],
    title: str,
    out_path: Path,
    plot_size: float,
    x_label: str,
    y_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(plot_size, plot_size), dpi=160)
    labels = sorted(set(values), key=lambda x: str(x))
    cmap = plt.get_cmap("tab20")
    for idx, label in enumerate(labels):
        mask = np.asarray([v == label for v in values])
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=5,
            alpha=0.65,
            color=cmap(idx % 20),
            label=str(label)[:36],
            linewidths=0,
        )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(markerscale=3, fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_continuous(
    coords: np.ndarray,
    values: np.ndarray,
    title: str,
    out_path: Path,
    plot_size: float,
    x_label: str,
    y_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(plot_size, plot_size), dpi=160)
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=values,
        s=5,
        alpha=0.75,
        cmap="viridis",
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.colorbar(sc, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_annotations_pca(
    X_projected: np.ndarray,
    rows: list[dict[str, Any]],
    out_dir: Path,
    plot_size: float,
) -> None:
    """PCA fit on just the annotated (labeled) crops, colored by label. this
    is a separate 2D projection from the main pca/ output -- fit only on the
    labeled subset so category structure isn't washed out by the much larger
    random-crop population."""
    ann_idx = [i for i, r in enumerate(rows) if r.get("label")]
    if len(ann_idx) < 3:
        print(f"[annotations] only {len(ann_idx)} labeled points; skipping annotations PCA plot")
        return
    X_ann = X_projected[ann_idx]
    labels = [rows[i]["label"] for i in ann_idx]
    coords, explained = pca_project(X_ann, 2)
    print(
        f"[annotations] PCA on {len(ann_idx)} labeled crops, "
        f"explained variance in 2D: {explained.sum()*100:.1f}%"
    )
    plot_categorical(
        coords,
        labels,
        f"Annotated crops PCA colored by label (n={len(ann_idx)})",
        out_dir / "annotations_pca_by_label.png",
        plot_size,
        "PCA 1",
        "PCA 2",
    )

def save_plots(
    coords: np.ndarray,
    rows: list[dict[str, Any]],
    out_dir: Path,
    plot_size: float,
    method: str,
) -> None:
    dataset_idx = [r["dataset_idx"] for r in rows]
    dataset_name = [r["dataset_name"] for r in rows]
    view = [r["view"] for r in rows]
    t = np.asarray([r["t"] for r in rows])
    z = np.asarray([r["z"] for r in rows])

    if method == "pca":
        x_label, y_label = "PCA 1", "PCA 2"
    else:
        x_label, y_label = f"{method.upper()} 1", f"{method.upper()} 2"

    plot_categorical(coords, dataset_idx, "Latent space colored by dataset_idx", out_dir / "latent_by_dataset_idx.png", plot_size, x_label, y_label)
    plot_categorical(coords, dataset_name, "Latent space colored by dataset name", out_dir / "latent_by_dataset_name.png", plot_size, x_label, y_label)
    plot_categorical(coords, view, "Latent space colored by global view", out_dir / "latent_by_view.png", plot_size, x_label, y_label)
    plot_continuous(coords, t, "Latent space colored by time", out_dir / "latent_by_time.png", plot_size, x_label, y_label)
    plot_continuous(coords, z, "Latent space colored by z", out_dir / "latent_by_z.png", plot_size, x_label, y_label)

    if "frame_gap" in rows[0]:
        frame_gap = np.asarray([r["frame_gap"] for r in rows])
        plot_continuous(coords, frame_gap, "Latent space colored by frame_gap", out_dir / "latent_by_frame_gap.png", plot_size, x_label, y_label)
    else:
        print("[plots] frame_gap not available (loaded from cached embeddings.npz); skipping latent_by_frame_gap.png")


def maybe_print_cluster_metrics(coords: np.ndarray, rows: list[dict[str, Any]]) -> None:
    labels = np.asarray([r["dataset_idx"] for r in rows], dtype=np.int64)
    if len(np.unique(labels)) < 2:
        print("[metrics] only one dataset label; skipping dataset clustering metrics")
        return
    try:
        from sklearn.metrics import silhouette_score
    except Exception:
        print("[metrics] scikit-learn not installed; skipping silhouette score")
        return
    score = silhouette_score(coords, labels)
    print(f"[metrics] silhouette by dataset_idx on 2D projection: {score:.3f}")
    print("[metrics] near 0 means mixed/overlapping datasets; high positive means dataset-separated islands")

def sliced_wasserstein_distance(
    x_a: np.ndarray,
    x_b: np.ndarray,
    n_projections: int = 128,
    n_quantiles: int = 100,
    seed: int = 0,
) -> float:
    """sliced Wasserstein-1 distance between two point clouds.

    projects both clouds onto many random 1d directions, computes the exact 1d
    wasserstein-1 distance on each slice by matching quantiles, and averages. the
    1d case reduces to sorting, so this approximates the full high-dim optimal
    transport distance without a solver or extra dependency. quantile matching
    also handles clouds with different sample counts.
    """
    dim = x_a.shape[1]
    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((dim, n_projections)).astype(np.float64)
    directions /= np.linalg.norm(directions, axis=0, keepdims=True) + 1e-12

    proj_a = x_a.astype(np.float64) @ directions  # (n_a, n_projections)
    proj_b = x_b.astype(np.float64) @ directions  # (n_b, n_projections)

    q = np.linspace(0.0, 1.0, n_quantiles)
    qa = np.quantile(proj_a, q, axis=0)  # (n_quantiles, n_projections)
    qb = np.quantile(proj_b, q, axis=0)
    # mean |Δquantile| per slice IS the 1d wasserstein-1 distance; average slices.
    return float(np.abs(qa - qb).mean())


def print_batch_effect_wasserstein(
    X: np.ndarray,
    rows: list[dict[str, Any]],
    n_projections: int = 128,
    seed: int = 0,
    min_per_dataset: int = 10,
) -> None:
    """pairwise sliced-Wasserstein distance between per-dataset embedding clouds.

    a direct, geometry-aware measure of residual batch effect. if the dataset
    adversary scrubbed dataset identity, the clouds overlap and distances sit
    near zero; large distances mean datasets stay separable in latent space.
    computed on full-dim embeddings, where the encoder actually operates, rather
    than on a 2d projection.
    """
    dataset_idx = np.asarray([r["dataset_idx"] for r in rows], dtype=np.int64)
    unique = np.unique(dataset_idx)
    if unique.size < 2:
        print("[wasserstein] only one dataset; skipping batch-effect distance")
        return

    names: dict[int, str] = {}
    for r in rows:
        names.setdefault(
            int(r["dataset_idx"]), r.get("dataset_name") or str(r["dataset_idx"])
        )

    # drop datasets with too few points for stable quantile estimates.
    groups: dict[int, np.ndarray] = {}
    for d in unique:
        pts = X[dataset_idx == d]
        if pts.shape[0] >= min_per_dataset:
            groups[int(d)] = pts
        else:
            print(
                f"[wasserstein] dataset {names[int(d)]} has {pts.shape[0]} < "
                f"{min_per_dataset} points; excluded"
            )
    keys = sorted(groups)
    if len(keys) < 2:
        print("[wasserstein] fewer than two datasets with enough points; skipping")
        return

    n = len(keys)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dist = sliced_wasserstein_distance(
                groups[keys[i]], groups[keys[j]], n_projections=n_projections, seed=seed
            )
            mat[i, j] = mat[j, i] = dist

    label_list = [names[k] for k in keys]
    width = max(12, max(len(s) for s in label_list) + 2)
    print("[wasserstein] pairwise sliced-Wasserstein distance between dataset embedding clouds")
    print("[wasserstein] lower = better mixing (batch effect removed); higher = datasets separable")
    header = " " * width + "".join(f"{s[: width - 1]:>{width}}" for s in label_list)
    print(header)
    for i, ri in enumerate(label_list):
        cells = "".join(f"{mat[i, j]:>{width}.4f}" for j in range(n))
        print(f"{ri[: width - 1]:>{width}}{cells}")

    iu = np.triu_indices(n, k=1)
    vals = mat[iu]
    print(
        f"[wasserstein] summary over {vals.size} dataset pairs: "
        f"mean={vals.mean():.4f} min={vals.min():.4f} max={vals.max():.4f}"
    )


def find_cached_run(checkpoint: str) -> Path | None:
    base = Path("latent_space")
    if not base.is_dir():
        return None
    prefix = f"{Path(checkpoint).parent.name}_"
    candidates = [
        d for d in base.iterdir()
        if d.is_dir() and d.name.startswith(prefix) and (d / "embeddings.npz").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: (d / "embeddings.npz").stat().st_mtime)

def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    cfg.seed = args.seed
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    disable_augmentations(cfg)

    device = torch.device(args.device or cfg.device)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        cached_dir = find_cached_run(args.checkpoint)
        if cached_dir is not None:
            out_dir = cached_dir
            print(f"[latent] no --out-dir given; found cached run at {out_dir}")
        else:
            out_dir = Path("latent_space") / f"{Path(args.checkpoint).parent.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[latent] device     : {device}")
    print(f"[latent] config     : {args.config}")
    print(f"[latent] checkpoint : {args.checkpoint}")
    print(f"[latent] out_dir    : {out_dir}")
    print(f"[latent] n_crops    : {args.n_crops}")
    print(f"[latent] views      : {args.views}")

    cache_path = out_dir / "embeddings.npz"
    if cache_path.exists():
        print(f"[latent] found existing {cache_path} -- loading cached embeddings instead of re-running the encoder")
        X, rows = load_cached_embeddings(cache_path, cfg)
        print(f"[latent] loaded cached embeddings: {X.shape}")
    else:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        print(f"[latent] checkpoint step: {ckpt.get('step', 'unknown')}")
        encoder = HybridEncoder(
            stem_channels=cfg.stem_channels,
            latent_dim=cfg.latent_dim,
            norm_groups=cfg.norm_groups,
        ).to(device)
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()

        t0 = time.time()
        X, rows = extract_embeddings(cfg, encoder, device, args.n_crops, args.views)
        print(f"[latent] extracted embeddings: {X.shape} in {time.time() - t0:.1f}s")

    X_projected = preprocess_embeddings(X, args.embedding_preprocess)
    print(f"[latent] embedding preprocess: {args.embedding_preprocess}")
    print_embedding_diagnostics(X, X_projected)

    # batch-effect check in the full latent space: how far apart the per-dataset
    # embedding clouds sit. complements the training-time dataset adversary.
    print_batch_effect_wasserstein(X_projected, rows, seed=args.seed)

    methods = resolve_methods(args.method)
    print(f"[latent] projection methods: {', '.join(methods)}")

    coords_by_method: dict[str, np.ndarray] = {}
    for method in methods:
        coords = project_single(args, X_projected, method)
        print(f"[latent] projection {method}: coords={coords.shape}")
        coords_by_method[method] = coords
        method_dir = out_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        write_metadata_csv(method_dir / "metadata.csv", rows, coords)
        save_plots(coords, rows, method_dir, args.plot_size, method)
        print(f"[metrics] {method} projection:")
        maybe_print_cluster_metrics(coords, rows)

    save_npz(out_dir / "embeddings.npz", X, coords_by_method, rows)

    plot_annotations_pca(X_projected, rows, out_dir, args.plot_size)

    print("[latent] wrote:")
    for path in sorted(p for p in out_dir.rglob("*") if p.is_file()):
        print(f"  - {path}")


if __name__ == "__main__":
    main()