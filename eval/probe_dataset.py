"""Linear probe for dataset identity on a frozen WS-DINO encoder.

We freeze the encoder, extract embeddings of t1 global crops over many samples,
and train a single nn.Linear to predict which zarr (dataset_idx) each crop came
from. By default, extraction samples only the configured held-out spatial
regions, which DINO training should exclude, then splits those regions
spatially into probe train/val. The intent is for this classifier to FAIL: low
val accuracy means the encoder has learned representations that don't encode
dataset identity, which is what we want: biology rather than
microscope/illumination signatures.

usage:
    python eval/probe_dataset.py --config configs/train.yaml --checkpoint checkpoints/latest.pt
    python eval/probe_dataset.py --config configs/train.yaml --checkpoint checkpoints/step_001500.pt --n-samples 4000
    python eval/probe_dataset.py --config configs/train.yaml --checkpoint checkpoints/latest.pt --split random

interpretation (3 datasets -> chance = 33.3%):
    val_acc near 33%  : encoder is dataset-invariant. good.
    val_acc ~40-60%   : moderate dataset leakage; encoder partially memorized identity.
    val_acc > 80%     : encoder has clearly memorized dataset identity. bad.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# allow running as `python eval/probe_dataset.py` from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader

from configs.config import Config
from data.dataset import MultiZarrDataset, worker_init_fn
from data.spatial_regions import region_bounds_for_top_left
from models.encoder import HybridEncoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to a yaml config (same one used for training)")
    p.add_argument("--checkpoint", required=True, help="path to a *.pt file (latest.pt, final.pt, step_*.pt)")
    p.add_argument("--device", default=None, help="override cfg.device (e.g. 'cpu')")
    p.add_argument(
        "--n-samples", type=int, default=5000,
        help="number of t1 globals to embed before training the linear head",
    )
    p.add_argument(
        "--probe-steps", type=int, default=300,
        help="optimizer steps for the linear head",
    )
    p.add_argument("--probe-lr", type=float, default=1e-3)
    p.add_argument("--probe-batch-size", type=int, default=512)
    p.add_argument("--probe-weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42, help="seed for sampling + train/val split")
    p.add_argument(
        "--split",
        choices=("spatial-holdout", "random"),
        default="spatial-holdout",
        help=(
            "spatial-holdout samples only the configured held-out regions and "
            "splits them spatially for probe train/val; random keeps the old "
            "80/20 random embedding split"
        ),
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="fraction of the held-out region reserved spatially for probe validation",
    )
    return p.parse_args()

def setup_logging(checkpoint_path: str) -> Path:
    """derive a log directory from the checkpoint's parent folder name and
    tee all stdout to both the terminal and a log file inside it."""
    run_name = Path(checkpoint_path).parent.name
    candidates = [p for p in Path("latent_space").glob(f"{run_name}_*") if p.is_dir()]

    if candidates:
        log_dir = max(candidates, key=lambda p: p.name)
    else:
        log_dir = Path("latent_space") / f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"dataset_probe_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_file = open(log_path, "a")

    class Tee:
        def write(self, msg):
            sys.__stdout__.write(msg)
            log_file.write(msg)
            log_file.flush()
        def flush(self):
            sys.__stdout__.flush()
            log_file.flush()

    sys.stdout = Tee()
    return log_path

def disable_augmentations(cfg: Config) -> None:
    """turn off stochastic augs so embeddings are deterministic given the
    sampled (t, z, y, x). per-crop norm stays on since that's how the encoder
    was trained to receive its input.
    """
    for name in ("aug_intensity", "aug_gamma", "aug_noise", "aug_blur"):
        getattr(cfg, name).enabled = False
    # aug_percrop_norm stays enabled; it's the input convention the encoder learned.


@torch.no_grad()
def extract_embeddings(
    cfg: Config,
    encoder: nn.Module,
    device: torch.device,
    n_samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """run encoder on the t1 global of many samples; return embeddings, labels, metadata."""
    cfg.seed = seed
    disable_augmentations(cfg)

    if not cfg.zarr_paths:
        raise ValueError("probe requires cfg.zarr_paths (a multi-zarr dataset)")
    ds = MultiZarrDataset(cfg)

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=device.type == "cuda",
        # spawn (not fork) so workers start fresh. forking after the parent has
        # initialized cuda + the zarr/blosc threadpool deadlocks on first read.
        multiprocessing_context="spawn" if cfg.num_workers > 0 else None,
    )

    encoder.eval()
    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    seen = 0
    t0 = time.time()
    print_every = max(1, n_samples // (cfg.batch_size * 10))
    iters = 0

    for batch in loader:
        # globals shape: (B, 2, 1, Z, H, W). index 0 is t1.
        g_t1 = batch["globals"][:, 0].to(device, non_blocking=True)
        ds_idx = batch["weak_label"][:, 0]  # (B,) long
        meta = batch["meta"]

        emb = encoder(g_t1)  # (B, latent_dim)
        embeddings.append(emb.cpu())
        labels.append(ds_idx)

        batch_size = g_t1.size(0)
        for i in range(batch_size):
            rows.append({
                "dataset_idx": int(ds_idx[i]),
                "t": int(meta["t1"][i]),
                "z": int(meta["z"][i]),
                "global_y": int(meta["global_y"][i]),
                "global_x": int(meta["global_x"][i]),
                "frame_y": int(meta["frame_y"][i]),
                "frame_x": int(meta["frame_x"][i]),
                "spatial_region_idx": int(meta["spatial_region_idx"][i]),
            })

        seen += g_t1.size(0)
        iters += 1
        if iters % print_every == 0:
            elapsed = time.time() - t0
            rate = seen / max(elapsed, 1e-6)
            print(f"  [extract] {seen}/{n_samples} samples  ({rate:.1f} samples/s)")
        if seen >= n_samples:
            break

    X = torch.cat(embeddings, dim=0)[:n_samples]
    y = torch.cat(labels, dim=0)[:n_samples].long()
    return X, y, rows[:n_samples]


def random_split(
    X: torch.Tensor,
    y: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(X.size(0), generator=gen)
    X, y = X[perm], y[perm]
    n_train = int(0.8 * X.size(0))
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def spatial_holdout_split(
    X: torch.Tensor,
    y: torch.Tensor,
    rows: list[dict[str, Any]],
    cfg: Config,
    val_frac: float,
    n_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"--val-frac must be in (0, 1); got {val_frac}")

    train_idx: list[int] = []
    val_idx: list[int] = []
    for i, row in enumerate(rows):
        y0, y1, x0, x1 = region_bounds_for_top_left(
            int(row["frame_y"]),
            int(row["frame_x"]),
            cfg.global_crop_size,
            cfg.spatial_region_grid_y,
            cfg.spatial_region_grid_x,
            int(row["spatial_region_idx"]),
        )
        split_on_y = (y1 - y0) >= (x1 - x0)
        lo = y0 if split_on_y else x0
        hi = y1 if split_on_y else x1
        coord = int(row["global_y"] if split_on_y else row["global_x"])
        threshold = lo + max(1, int(round((hi - lo) * (1.0 - val_frac))))
        if coord >= threshold:
            val_idx.append(i)
        else:
            train_idx.append(i)

    if not train_idx or not val_idx:
        raise ValueError(
            "spatial-holdout split produced an empty train or val set; "
            "increase --n-samples or adjust --val-frac"
        )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    train_counts = torch.bincount(y_train, minlength=n_classes)
    val_counts = torch.bincount(y_val, minlength=n_classes)
    if bool((train_counts == 0).any()) or bool((val_counts == 0).any()):
        raise ValueError(
            "spatial-holdout split is missing at least one dataset class in "
            f"train or val; train_counts={train_counts.tolist()} "
            f"val_counts={val_counts.tolist()}. Increase --n-samples."
        )

    return X_train, y_train, X_val, y_val


def print_confusion_matrix(cm: torch.Tensor, zarr_paths: list[str]) -> None:
    """print a labeled confusion matrix and per-class accuracy."""
    n = cm.size(0)
    short_names = [Path(p).name[:30] for p in zarr_paths]

    print()
    print("  confusion matrix (rows = true dataset, cols = predicted):")
    header = "    " + " ".join(f" pred{c}" for c in range(n))
    print(header)
    for r in range(n):
        row = " ".join(f"{int(cm[r,c]):6d}" for c in range(n))
        print(f"  true{r}  {row}    {short_names[r]}")

    print()
    print("  per-class val accuracy:")
    for r in range(n):
        total = int(cm[r].sum())
        correct = int(cm[r, r])
        pct = 100.0 * correct / max(total, 1)
        print(f"    ds{r} ({short_names[r]:30s})  {correct:4d}/{total:4d}  ({pct:.1f}%)")


def interpret(val_acc: float, n_classes: int) -> str:
    chance = 1.0 / n_classes
    margin = val_acc - chance
    if margin < 0.10:
        return (
            f"val accuracy is near chance ({chance*100:.1f}%) - encoder appears "
            f"dataset-invariant. this is the desired outcome."
        )
    if val_acc > 0.80:
        return (
            f"val accuracy is well above chance - encoder has clearly memorized "
            f"dataset identity. downstream tasks will likely be contaminated by "
            f"this. consider stronger augmentations, adversarial dataset confusion, "
            f"or more aggressive per-crop normalization."
        )
    return (
        f"val accuracy is moderately above chance ({val_acc*100:.1f}% vs "
        f"{chance*100:.1f}%) - some dataset leakage. worth tracking across "
        f"checkpoints to see whether it grows or stabilizes."
    )


def main() -> None:
    args = parse_args()
    log_path = setup_logging(args.checkpoint)
    print(f"[probe] logging to  : {log_path}")

    cfg = Config.from_yaml(args.config)

    device = torch.device(args.device or cfg.device)
    print(f"[probe] device      : {device}")
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        print(f"[probe] cuda gpu    : {torch.cuda.get_device_name(idx)}")
    print(f"[probe] config      : {args.config}")
    print(f"[probe] checkpoint  : {args.checkpoint}")
    print(f"[probe] n_samples   : {args.n_samples}")
    print(f"[probe] probe_steps : {args.probe_steps}")
    print(f"[probe] seed        : {args.seed}")
    print(f"[probe] split       : {args.split}")
    print()

    # --- load encoder ---
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    print(f"[probe] checkpoint saved at training step {ckpt['step']}")
    encoder = HybridEncoder(
        stem_channels=cfg.stem_channels,
        latent_dim=cfg.latent_dim,
        norm_groups=cfg.norm_groups,
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    n_datasets = len(cfg.zarr_paths) if cfg.zarr_paths else 1
    chance_pct = 100.0 / n_datasets
    print(f"[probe] n_datasets  : {n_datasets}  (chance baseline = {chance_pct:.1f}%)")
    print()

    # --- extract embeddings ---
    if args.split == "spatial-holdout":
        has_manual_holdout = (
            cfg.spatial_holdout_region is not None
            or bool(cfg.spatial_holdout_regions)
        )
        if cfg.spatial_holdout_selection == "manual" and not has_manual_holdout:
            raise ValueError(
                "--split spatial-holdout with manual selection requires "
                "spatial_holdout_regions or spatial_holdout_region in the config"
            )
        cfg.spatial_region_mode = "holdout"
        print(
            "[probe] spatial split: sampling only held-out region(s) "
            "from each dataset"
        )

    print(f"[probe] extracting {args.n_samples} embeddings (this will run the encoder forward, no augmentations)...")
    t0 = time.time()
    X, y, rows = extract_embeddings(cfg, encoder, device, args.n_samples, args.seed)
    print(f"[probe] extracted in {time.time()-t0:.1f}s.  X.shape={tuple(X.shape)}  y.shape={tuple(y.shape)}")
    print(f"[probe] dataset_idx counts: {torch.bincount(y, minlength=n_datasets).tolist()}")
    print()

    # --- train/val split ---
    if args.split == "spatial-holdout":
        X_train, y_train, X_val, y_val = spatial_holdout_split(
            X,
            y,
            rows,
            cfg,
            args.val_frac,
            n_datasets,
        )
        print(
            "[probe] split detail: held-out region was split spatially; "
            "validation uses the high-coordinate strip inside that region"
        )
    else:
        X_train, y_train, X_val, y_val = random_split(X, y, args.seed)
        print("[probe] split detail: random 80/20 embedding split")

    print(f"[probe] train n={X_train.size(0)}  counts={torch.bincount(y_train, minlength=n_datasets).tolist()}")
    print(f"[probe] val   n={X_val.size(0)}  counts={torch.bincount(y_val,   minlength=n_datasets).tolist()}")
    print()

    # --- train probe ---
    print(f"[probe] training linear head (in_dim={cfg.latent_dim}, out_dim={n_datasets})...")

    # Train and save the head
    head = nn.Linear(cfg.latent_dim, n_datasets).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.probe_lr, weight_decay=args.probe_weight_decay)
    ce = nn.CrossEntropyLoss()

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    n = X_train.size(0)
    gen = torch.Generator(device=device).manual_seed(0)

    for step in range(args.probe_steps):
        idx = torch.randint(0, n, (args.probe_batch_size,), device=device, generator=gen)
        logits = head(X_train[idx])
        loss = ce(logits, y_train[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step == 0 or (step + 1) % max(1, args.probe_steps // 10) == 0:
            with torch.no_grad():
                tr_acc = (head(X_train).argmax(1) == y_train).float().mean().item()
                va_acc = (head(X_val).argmax(1) == y_val).float().mean().item()
            print(
                f"  [probe] step {step+1:4d}/{args.probe_steps}  "
                f"loss={loss.item():.4f}  train_acc={tr_acc:.3f}  val_acc={va_acc:.3f}"
            )

    with torch.no_grad():
        tr_acc = (head(X_train).argmax(1) == y_train).float().mean().item()
        preds = head(X_val).argmax(1)
        va_acc = (preds == y_val).float().mean().item()
        cm = torch.zeros(n_datasets, n_datasets, dtype=torch.long)
        for t, p in zip(y_val.cpu().tolist(), preds.cpu().tolist()):
            cm[t, p] += 1

    # --- report ---
    print()
    print("=" * 70)
    print(f"  RESULT @ training step {ckpt['step']}")
    print("=" * 70)
    print(f"  train_acc        : {tr_acc*100:6.2f}%")
    print(f"  val_acc          : {va_acc*100:6.2f}%   (chance = {chance_pct:.1f}%)")
    print(f"  val_acc - chance : {(va_acc - 1.0/n_datasets)*100:+6.2f} percentage points")
    print_confusion_matrix(cm, cfg.zarr_paths)
    print()
    print(f"  interpretation: {interpret(va_acc, n_datasets)}")
    print("=" * 70)

    # Save the trained linear head for Grad-CAM
    log_dir = Path(log_path).parent
    checkpoint = {
        "head_state_dict": head.cpu().state_dict(),
        "input_dim": cfg.latent_dim,
        "output_dim": n_datasets,
        "dataset_names": cfg.zarr_paths,
        "val_acc": va_acc,
        "args": vars(args),
    }

    ckpt_path = log_dir / "dataset_probe.pt"
    torch.save(checkpoint, ckpt_path)
    print()
    print(f"[probe] saved linear head to {ckpt_path}")


if __name__ == "__main__":
    main()
