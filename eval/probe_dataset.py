"""linear probe for dataset identity on a frozen WS-DINO encoder.

the probe freezes the encoder, extracts embeddings of t1 global crops over many
samples, and trains a single nn.Linear. the nn.Linear predicts which zarr
(dataset_idx) each crop came from. by default the extraction samples only the
configured held-out spatial regions, which DINO training must exclude. the
extraction then splits those regions spatially into probe train and probe val.
the intent is that this classifier FAILS. a low val accuracy means the encoder
learned representations that do not encode dataset identity, which is the wanted
outcome: biology, not microscope or illumination signatures.

usage:
    python eval/probe_dataset.py --config configs/train.yaml --checkpoint checkpoints/latest.pt
    python eval/probe_dataset.py --config configs/train.yaml --checkpoint checkpoints/step_001500.pt --n-samples 4000
    python eval/probe_dataset.py --config configs/train.yaml --checkpoint checkpoints/latest.pt --split random

interpretation, relative to chance (= 1 / n_classes, printed at startup):
    val_acc near chance : encoder is dataset-invariant. good.
    val_acc ~2x chance  : moderate dataset leakage; encoder partially memorized identity.
    val_acc > 80%       : encoder clearly memorized dataset identity. bad.

n_classes here counts imaging sessions, not zarrs -- see README "Data".
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

# a Field_* path component is a well of one plate, not a separate session.
_FIELD_DIR = re.compile(r"^Field_[A-Z]\d+$", re.I)

# the path insert lets you run `python eval/probe_dataset.py` from the project root.
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
        "--max-contamination", type=float, default=None,
        help=(
            "drop probe crops whose pixels overlap the training regions' "
            "footprint by more than this fraction (0-1). the spatial holdout "
            "partitions crop TOP-LEFT positions, not image area, and every crop "
            "extends global_crop_size-1 px past its top-left -- so a 'held-out' "
            "crop can still be built largely from pixels training saw. the "
            "overlap fraction is always reported; pass this to also filter on "
            "it. note the achievable floor is not 0: with a 3x3 grid, 3 held-out "
            "regions and a 256px crop the uncontaminated area is three 241px "
            "corners, which cannot contain a 256px crop at all (floor ~11%). "
            "compare an unfiltered run against a filtered one to see how much "
            "of the reported leakage is crop memorisation."
        ),
    )
    p.add_argument(
        "--no-merge-sessions", action="store_true",
        help=(
            "keep every zarr as its own class. by default zarrs from one "
            "imaging session (a plate's Field_B2 / Field_B3) are merged, since "
            "they share plate, day and optics and differ only by well and "
            "therefore strain -- separating them measures biology, not batch"
        ),
    )
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
    """derive a log directory from the name of the parent folder of the checkpoint.

    this function tees all stdout to the terminal and to a log file in that
    directory.
    """
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
    """turn off the stochastic augs, so the embeddings stay deterministic for a
    given sampled (t, z, y, x). the per-crop norm stays on, because the encoder
    received its input in that form during training.
    """
    for name in ("aug_intensity", "aug_gamma", "aug_noise", "aug_blur"):
        getattr(cfg, name).enabled = False


@torch.no_grad()
def extract_embeddings(
    cfg: Config,
    encoder: nn.Module,
    device: torch.device,
    n_samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """run the encoder on the t1 global of many samples. return embeddings, labels and metadata."""
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
        # use spawn and not fork, so each worker starts fresh. a fork after the
        # parent initialized cuda and the zarr/blosc threadpool deadlocks on the
        # first read.
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


def training_footprint_overlap(
    rows: list[dict[str, Any]],
    cfg: Config,
    holdout_regions: list[int],
) -> torch.Tensor:
    """the per-crop fraction of pixels that a training-region crop can also cover.

    the holdout partitions the top-left positions of the crops, and not the
    image area. therefore a "held-out" crop can still consist mostly of training
    pixels. see README "How to read the metrics" section 2. this function
    measures the fraction for each crop, so you can read the probe number
    against the fraction, and not assume that the crops are clean.

    returns a float tensor in [0, 1], aligned with `rows`.
    """
    grid_y, grid_x = cfg.spatial_region_grid_y, cfg.spatial_region_grid_x
    crop = cfg.global_crop_size

    # warning: the frame sizes vary by up to 4x in area (README "Data"). you
    # must rebuild the region grid and the footprint for every distinct frame
    # size. the cache key is (h, w).
    sats: dict[tuple[int, int], np.ndarray] = {}

    def sat_for(frame_y: int, frame_x: int) -> np.ndarray:
        key = (frame_y, frame_x)
        if key not in sats:
            train = np.zeros((frame_y, frame_x), dtype=bool)
            for idx in range(grid_y * grid_x):
                if idx in holdout_regions:
                    continue
                y0, y1, x0, x1 = region_bounds_for_top_left(
                    frame_y, frame_x, crop, grid_y, grid_x, idx
                )
                # the bands are top-left ranges. a crop at y1-1 reaches to y1-1+crop.
                train[y0 : y1 - 1 + crop, x0 : x1 - 1 + crop] = True
            # a summed-area table, so the overlap of each crop costs 4 lookups.
            sats[key] = np.pad(train.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        return sats[key]

    out = torch.empty(len(rows), dtype=torch.float32)
    oob = 0
    for i, row in enumerate(rows):
        frame_y, frame_x = int(row["frame_y"]), int(row["frame_x"])
        sat = sat_for(frame_y, frame_x)
        y, x = int(row["global_y"]), int(row["global_x"])
        # clamp (y, x), do not trust them: the overrepresent_classes injection
        # branch in _sample_pair takes (y, x) from index.jsonl without a check,
        # and dask truncates an out-of-range read silently and raises no error.
        y = min(max(y, 0), max(0, frame_y - 1))
        x = min(max(x, 0), max(0, frame_x - 1))
        if y != int(row["global_y"]) or x != int(row["global_x"]):
            oob += 1
        y2, x2 = min(y + crop, frame_y), min(x + crop, frame_x)
        covered = sat[y2, x2] - sat[y, x2] - sat[y2, x] + sat[y, x]
        out[i] = covered / max(1, (y2 - y) * (x2 - x))
    if oob:
        print(f"[probe] WARNING {oob}/{len(rows)} crops had a top-left outside "
              "their own zarr's frame and were clamped for this calculation -- "
              "those crops were read truncated by dask, not as full crops")
    return out


def report_contamination(
    overlap: torch.Tensor, rows: list[dict[str, Any]] | None = None
) -> None:
    q = torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    vals = torch.quantile(overlap, q)
    print("[probe] training-footprint overlap per probe crop "
          "(fraction of each crop's pixels that training could also see):")
    print("        " + "  ".join(f"p{int(p*100):<3d}={v*100:5.1f}%"
                                 for p, v in zip(q.tolist(), vals.tolist())))
    print(f"        mean={overlap.mean()*100:.1f}%   "
          f">99% contaminated: {(overlap > 0.99).float().mean()*100:.1f}% of crops")
    print("        higher overlap inflates this probe (memorised crops are more "
          "identifiable), so it biases the reported leakage UP, not down.")
    # contamination depends strongly on the frame size, because the grid is in
    # absolute pixels and the frames vary by up to 4x in area (README "Data").
    # a pool across the experiments hides the effect, so report it per frame
    # size.
    if rows:
        by_frame: dict[tuple[int, int], list[float]] = {}
        for r, v in zip(rows, overlap.tolist()):
            by_frame.setdefault(
                (int(r["frame_y"]), int(r["frame_x"])), []
            ).append(v)
        print("        by frame size (crop "
              "is a bigger share of a small frame, so contamination is worse):")
        for (h, w), vs in sorted(by_frame.items()):
            a = torch.tensor(vs)
            print(f"          {h:5d}x{w:<5d} n={len(vs):5d}  "
                  f"mean={a.mean()*100:5.1f}%  min={a.min()*100:5.1f}%")


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


def session_key(zarr_path: str) -> str:
    """collapse the zarrs of a single imaging session into one class.

    the Field_* wells of a plate share the plate, the day and the optics, and
    they differ only by strain. therefore a separate score for each well
    measures biology and not batch. see README "Data". this function does NOT
    merge the day subdirectories, for example mixin65 day1 against day2, because
    those are truly different sessions.
    """
    parts = PurePosixPath(zarr_path.replace("\\", "/")).parts
    try:
        i = next(j for j, p in enumerate(parts)
                 if p.startswith("cryolite_mixin_test"))
    except StopIteration:
        return zarr_path
    return "/".join(p for p in parts[i:-1] if not _FIELD_DIR.match(p))


def print_confusion_matrix(cm: torch.Tensor, class_names: list[str]) -> None:
    """print a labeled confusion matrix and per-class accuracy."""
    n = cm.size(0)
    # class_names holds zarr paths in the unmerged case and session keys in the
    # merged case. the constant "cryolite_" prefix goes away, so the experiment
    # and the date stay inside the 30-char truncation.
    short_names = [
        (Path(p).name or p).removeprefix("cryolite_")[:30] for p in class_names
    ]

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

    # set the cap on the config, so the SAMPLER enforces the cap before any
    # read. the test is purely geometric, so a rejected candidate costs nothing.
    # a rejection upstream of balance_datasets also keeps the per-experiment
    # counts even. see README "How to read the metrics" section 2.
    if args.max_contamination is not None:
        cfg.max_train_footprint_overlap = args.max_contamination
        print(f"[probe] max_train_footprint_overlap = {args.max_contamination} "
              "(enforced during sampling, not after)")

    # this probe forces overrepresent_classes off and gives no flag for it. the
    # injection branch of overrepresent_classes bypasses the spatial region
    # filter, so rep_alpha > 0 would feed this leakage probe the most memorised
    # crops available. see README "How to read the metrics" section 1. the
    # numbers from before 2026-08-04 come from runs with the option on.
    if cfg.overrepresent_classes:
        print("[probe] overrepresent_classes: forced OFF "
              "(its injection branch bypasses the spatial region filter)")
        cfg.overrepresent_classes = False

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

    # the class space. by default the zarrs of one imaging session collapse to
    # one class. the code below applies the remap to y directly after the
    # extraction.
    class_names = list(cfg.zarr_paths) if cfg.zarr_paths else ["dataset"]
    session_remap: torch.Tensor | None = None
    if not args.no_merge_sessions and cfg.zarr_paths:
        keys = [session_key(p) for p in cfg.zarr_paths]
        order = sorted(dict.fromkeys(keys), key=keys.index)  # stable, first-seen
        if len(order) < len(keys):
            session_remap = torch.tensor(
                [order.index(k) for k in keys], dtype=torch.long
            )
            print(f"[probe] merged {len(keys)} zarrs -> {len(order)} imaging "
                  "sessions (same plate/day/optics, differing only by well):")
            for k in order:
                members = [Path(p).name[:44]
                           for p, kk in zip(cfg.zarr_paths, keys) if kk == k]
                if len(members) > 1:
                    print(f"    {k.split('/')[0]}")
                    for m in members:
                        print(f"        + {m}")
            class_names = order

    n_datasets = len(class_names)
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
    if session_remap is not None:
        # y indexes cfg.zarr_paths. fold y down to the session class space.
        y = session_remap[y]
    print(f"[probe] dataset_idx counts: {torch.bincount(y, minlength=n_datasets).tolist()}")
    print()

    # --- how much of this "held-out" data did the encoder see? ---
    if args.split == "spatial-holdout" and rows:
        # in holdout mode the sampler visits only the held-out regions, so the
        # regions present in `rows` are the resolved holdout set. read the set
        # from the data, and do not extract it from the dataset object.
        counts: dict[int, int] = {}
        for r in rows:
            k = int(r["spatial_region_idx"])
            counts[k] = counts.get(k, 0) + 1
        holdout_regions = sorted(counts)
        print(f"[probe] regions present in sample: "
              f"{ {k: counts[k] for k in holdout_regions} }")
        n_reg = cfg.spatial_region_grid_y * cfg.spatial_region_grid_x
        if len(holdout_regions) == n_reg:
            print("[probe] WARNING all "
                  f"{n_reg} regions are present, so nothing was held out. "
                  "expected only the resolved holdout regions -- check that "
                  "spatial_region_mode reached the sampler as 'holdout'.")
        # purely diagnostic now. the sampler enforced the cap already, so the
        # code below reports the realised distribution and filters nothing.
        overlap = training_footprint_overlap(rows, cfg, holdout_regions)
        report_contamination(overlap, rows)
        if args.max_contamination is not None:
            worst = float(overlap.max())
            if worst > args.max_contamination + 1e-6:
                print(f"[probe] WARNING realised overlap reached {worst:.3f}, "
                      f"above the requested cap {args.max_contamination} -- the "
                      "sampler-side filter did not apply to every crop (check "
                      "spatial_region_mode reached the sampler as 'holdout')")
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
    print_confusion_matrix(cm, class_names)
    print()
    print(f"  interpretation: {interpret(va_acc, n_datasets)}")
    print("=" * 70)

    # save the trained linear head for grad-cam
    log_dir = Path(log_path).parent
    checkpoint = {
        "head_state_dict": head.cpu().state_dict(),
        "input_dim": cfg.latent_dim,
        "output_dim": n_datasets,
        "dataset_names": class_names,
        "val_acc": va_acc,
        "args": vars(args),
    }

    ckpt_path = log_dir / "dataset_probe.pt"
    torch.save(checkpoint, ckpt_path)
    print()
    print(f"[probe] saved linear head to {ckpt_path}")


if __name__ == "__main__":
    main()
