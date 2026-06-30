"""Linear probe for behavior identity on a frozen WS-DINO encoder.

We freeze the encoder, extract embeddings of t1 global crops over all annotated samples,
and train a single nn.Linear to predict which behavior each crop came from.
The intent is for this classifier to SUCCEED: high val accuracy indicates the encoder
has learned a representation that separates behaviors, which is the desired outcome.

usage:
    python eval/probe_behavior.py --config configs/train.yaml --checkpoint checkpoints/z_filtering/latest.pt

"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any
import statistics
import pandas as pd

# allow running as `python eval/probe_behavior.py` from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
from sklearn.model_selection import StratifiedGroupKFold

from configs.config import Config
from data.dataset import MultiZarrDataset, worker_init_fn
from data.spatial_regions import region_bounds_for_top_left
from models.encoder import HybridEncoder

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

    log_path = log_dir / f"behavior_probe_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_file = open(log_path, "a")

    class Tee:
        def write(self, msg):
            sys.__stdout__.write(msg)
            log_file.write(msg)
        def flush(self):
            sys.__stdout__.flush()
            log_file.flush()

    sys.stdout = Tee()
    return log_path

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to a yaml config (same one used for training)")
    p.add_argument("--checkpoint", required=True, help="path to a *.pt file (latest.pt, final.pt, step_*.pt)")
    p.add_argument("--device", default=None, help="override cfg.device (e.g. 'cpu')")
    p.add_argument(
        "--probe-steps", type=int, default=300,
        help="optimizer steps for the linear head",
    )
    p.add_argument("--probe-lr", type=float, default=1e-3)
    p.add_argument("--probe-batch-size", type=int, default=512)
    p.add_argument("--probe-weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42, help="seed for sampling + train/val split")
    p.add_argument("--n_splits", type=int, default=5, help="Folds for stratified k fold splitting")
    """
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="validation percentage",
    )
    """
    return p.parse_args()


def disable_augmentations(cfg: Config) -> None:
    """turn off stochastic augs so embeddings are deterministic given the
    sampled (t, z, y, x). per-crop norm stays on since that's how the encoder
    was trained to receive its input.
    """
    for name in ("aug_intensity", "aug_gamma", "aug_noise", "aug_blur"):
        getattr(cfg, name).enabled = False
    # aug_percrop_norm stays enabled; it's the input convention the encoder learned.


def load_dataset(cfg: Config) -> MultiZarrDataset:
    df = pd.read_json(cfg.annotation_path, lines=True)
    cfg.zarr_paths = list(df['zarr_path'].unique())
    ds = MultiZarrDataset(cfg)
    return ds


@torch.no_grad()
def extract_embeddings(
    cfg: Config,
    encoder: nn.Module,
    device: torch.device,
    seed: int,
    annotations_df: pd.DataFrame
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """run encoder on the t1 global of many samples; return embeddings, labels, metadata."""
    cfg.seed = seed

    if not cfg.zarr_paths:
        raise ValueError("probe requires cfg.zarr_paths (a multi-zarr dataset)")
    ds = load_dataset(cfg)
    n_samples = len(annotations_df)
    labels_list = sorted(list(annotations_df['label'].unique()))

    encoder.eval()
    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    seen = 0
    t0 = time.time()
    print_every = max(1, n_samples // (cfg.batch_size * 10))
    iters = 0
    
    for i in range(len(cfg.zarr_paths)):
        zarr_df = annotations_df[annotations_df['zarr_path'] == cfg.zarr_paths[i]]
        src = ds.children[i]
        for row in zarr_df.itertuples():
            # globals shape: (B, 2, 1, Z, H, W). index 0 is t1.
            batch = src._annotation_tensor(row.t1, row.t2, row.z, row.global_x, row.global_y)
            g_t1 = batch["globals"][0].to(device, non_blocking=True).unsqueeze(0)
            label_idx = torch.tensor([labels_list.index(row.label)])
            meta = batch["meta"]

            emb = encoder(g_t1)  # (B, latent_dim)
            embeddings.append(emb.cpu())
            labels.append(label_idx)
            rows.append({
                "label_idx": label_idx,
                "t": int(meta["t1"]),
                "z": int(meta["z"]),
                "global_y": int(meta["global_y"]),
                "global_x": int(meta["global_x"]),
                "frame_y": int(meta["frame_y"]),
                "frame_x": int(meta["frame_x"]),
                "spatial_region_idx": int(meta["spatial_region_idx"]),
            })
                

            seen += 1
        iters += 1
        if iters % print_every == 0:
            elapsed = time.time() - t0
            rate = seen / max(elapsed, 1e-6)
            print(f"  [extract] {seen}/{n_samples} samples  ({rate:.1f} samples/s)")

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


def StratifiedGroupKFoldSplit(
    X: torch.Tensor,
    y: torch.Tensor,
    dataset_idx: list,
    seed: int,
    n_splits: int = 5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_index, test_index in sgkf.split(X, y, groups=dataset_idx):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        yield X_train, y_train, X_test, y_test

def train_probe(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    n_classes: int,
    probe_steps: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    device: torch.device,
) -> tuple[float, float, torch.Tensor]:
    """train a single nn.Linear with AdamW + cross-entropy; return train_acc,
    val_acc, and the val-set confusion matrix.
    """
    head = nn.Linear(X_train.size(1), n_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    n = X_train.size(0)
    gen = torch.Generator(device=device).manual_seed(0)

    for step in range(probe_steps):
        idx = torch.randint(0, n, (batch_size,), device=device, generator=gen)
        logits = head(X_train[idx])
        loss = ce(logits, y_train[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step == 0 or (step + 1) % max(1, probe_steps // 10) == 0:
            with torch.no_grad():
                tr_acc = (head(X_train).argmax(1) == y_train).float().mean().item()
                va_acc = (head(X_val).argmax(1) == y_val).float().mean().item()
            print(
                f"  [probe] step {step+1:4d}/{probe_steps}  "
                f"loss={loss.item():.4f}  train_acc={tr_acc:.3f}  val_acc={va_acc:.3f}"
            )

    with torch.no_grad():
        tr_acc = (head(X_train).argmax(1) == y_train).float().mean().item()
        preds = head(X_val).argmax(1)
        va_acc = (preds == y_val).float().mean().item()
        cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
        for t, p in zip(y_val.cpu().tolist(), preds.cpu().tolist()):
            cm[t, p] += 1

    return tr_acc, va_acc, cm


def print_confusion_matrix(cm: torch.Tensor, classes: list[str]) -> None:
    """print a labeled confusion matrix and per-class accuracy."""
    n = cm.size(0)

    print()
    print("  confusion matrix (rows = true dataset, cols = predicted):")
    header = "    " + " ".join(f" pred{c}" for c in range(n))
    print(header)
    for r in range(n):
        row = " ".join(f"{int(cm[r,c]):6d}" for c in range(n))
        print(f"  true{r}  {row}    {classes[r]}")

    print()
    print("  per-class val accuracy:")
    for r in range(n):
        total = int(cm[r].sum())
        correct = int(cm[r, r])
        pct = 100.0 * correct / max(total, 1)
        print(f"    ds{r} ({classes[r]:30s})  {correct:4d}/{total:4d}  ({pct:.1f}%)")


def interpret(val_acc: float, n_classes: int) -> str:
    chance = 1.0 / n_classes
    margin = val_acc - chance
    relative = margin / chance
    if relative < 0.2:
        return (
            f"val accuracy ({val_acc*100:.1f}%) is near chance ({chance*100:.1f}%) — "
            f"encoder does not appear to separate behaviors. the representation may not "
            f"yet be meaningful, or more training steps may be needed."
        )
    if relative < 0.6:
        return (
            f"val accuracy ({val_acc*100:.1f}%) is moderately above chance ({chance*100:.1f}%) — "
            f"encoder is partially separating behaviors. worth tracking across checkpoints "
            f"to see whether it improves with further training."
        )
    return (
        f"val accuracy ({val_acc*100:.1f}%) is well above chance ({chance*100:.1f}%) — "
        f"encoder has learned a representation that separates behaviors well. "
        f"this is the desired outcome."
    )


def main() -> None:
    args = parse_args()
    log_path = setup_logging(args.checkpoint)
    print(f"[probe] logging to  : {log_path}")

    cfg = Config.from_yaml(args.config)
    df = pd.read_json(cfg.annotation_path, lines=True)
    df = df[df['label'] != 'empty']
    df = df.drop(columns = ['timestamp'])
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], downcast="integer")
        except (ValueError, TypeError):
            pass

    cfg.zarr_paths = list(df['zarr_path'].unique())

    device = torch.device(args.device or cfg.device)
    print(f"[probe] device      : {device}")
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        print(f"[probe] cuda gpu    : {torch.cuda.get_device_name(idx)}")
    print(f"[probe] config      : {args.config}")
    print(f"[probe] checkpoint  : {args.checkpoint}")
    print(f"[probe] n_samples   : {len(df)}")
    print(f"[probe] probe_steps : {args.probe_steps}")
    print(f"[probe] seed        : {args.seed}")
    print(f"[probe] split       : Stratified k-fold")
    print()

    # --- load encoder ---
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    print(f"[probe] checkpoint saved at training step {ckpt['step']}")
    encoder = HybridEncoder(
        stem_channels=cfg.stem_channels,
        latent_dim=cfg.latent_dim,
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    n_classes = len(df['label'].unique())
    chance_pct = 100.0 / n_classes
    print(f"[probe] n_datasets  : {n_classes}  (chance baseline = {chance_pct:.1f}%)")
    print()

    # --- extract embeddings ---


    print(f"[probe] extracting {len(df)} embeddings (this will run the encoder forward, no augmentations)...")
    t0 = time.time()
    disable_augmentations(cfg)
    X, y, rows = extract_embeddings(cfg, encoder, device, args.seed, df)
    print(f"[probe] extracted in {time.time()-t0:.1f}s.  X.shape={tuple(X.shape)}  y.shape={tuple(y.shape)}")
    print(f"[probe] label counts: {torch.bincount(y, minlength=n_classes).tolist()}")
    print()

    fold_train_accs, fold_val_accs, fold_cms = [], [], []

    for fold_idx, (X_train, y_train, X_val, y_val) in enumerate(StratifiedGroupKFoldSplit(X, y, list(df['dataset_idx']), args.seed, args.n_splits)):
        print(f"[probe] fold {fold_idx+1}/{args.n_splits}  (grouped by zarr volume, stratified by label)")
        print(f"[probe] train n={X_train.size(0)}  counts={torch.bincount(y_train, minlength=n_classes).tolist()}")
        print(f"[probe] val   n={X_val.size(0)}  counts={torch.bincount(y_val,   minlength=n_classes).tolist()}")
        print()

        # --- train probe ---
        print(f"[probe] training linear head (in_dim={cfg.latent_dim}, out_dim={n_classes})...")
        tr_acc, va_acc, cm = train_probe(
            X_train, y_train, X_val, y_val,
            n_classes=n_classes,
            probe_steps=args.probe_steps,
            lr=args.probe_lr,
            batch_size=args.probe_batch_size,
            weight_decay=args.probe_weight_decay,
            device=device,
        )
        fold_train_accs.append(tr_acc)
        fold_val_accs.append(va_acc)
        fold_cms.append(cm)

        # --- fold-specific report ---
        print()
        print("=" * 70)
        print(f"  FOLD {fold_idx+1}/{args.n_splits} RESULT @ training step {ckpt['step']}")
        print("=" * 70)
        print(f"  train_acc        : {tr_acc*100:6.2f}%")
        print(f"  val_acc          : {va_acc*100:6.2f}%   (chance = {chance_pct:.1f}%)")
        print(f"  val_acc - chance : {(va_acc - 1.0/n_classes)*100:+6.2f} percentage points")
        print_confusion_matrix(cm, sorted(df['label'].unique()))
        print()
        print(f"  interpretation: {interpret(va_acc, n_classes)}")
        print("=" * 70)
        print()
    
    train_mean = statistics.mean(fold_train_accs)
    train_std = statistics.stdev(fold_train_accs) if len(fold_train_accs) > 1 else 0.0
    val_mean = statistics.mean(fold_val_accs)
    val_std = statistics.stdev(fold_val_accs) if len(fold_val_accs) > 1 else 0.0

    cm_total = torch.stack(fold_cms).sum(dim=0)

    print()
    print("#" * 70)
    print(f"  CROSS-VALIDATED SUMMARY ({args.n_splits} folds, grouped by zarr volume)")
    print("#" * 70)
    print(f"  train_acc : {train_mean*100:6.2f}%  ± {train_std*100:.2f}  (per-fold: {[f'{a*100:.1f}' for a in fold_train_accs]})")
    print(f"  val_acc   : {val_mean*100:6.2f}%  ± {val_std*100:.2f}  (per-fold: {[f'{a*100:.1f}' for a in fold_val_accs]})")
    print(f"  val_acc - chance : {(val_mean - 1.0/n_classes)*100:+6.2f} percentage points")
    print()
    print("  pooled confusion matrix (summed across all folds):")
    print_confusion_matrix(cm_total, sorted(df['label'].unique()))
    print()
    print(f"  interpretation: {interpret(val_mean, n_classes)}")
    print("#" * 70)


if __name__ == "__main__":
    main()
