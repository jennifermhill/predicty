"""linear probe for behavior identity on a frozen WS-DINO encoder.

the probe freezes the encoder, extracts embeddings of t1 global crops over all
annotated samples, and trains a single nn.Linear. the nn.Linear predicts which
behavior each crop came from. the intent is that this classifier SUCCEEDS.

warning: read macro_F1 and balanced_acc on --restrict unseen, and do NOT read
val_acc. this label set is imbalanced, so a high val_acc can sit below the
majority baseline. crops that the encoder saw during pre-training also inflate
the full-set score. see README "How to read the metrics".

usage:
    python eval/probe_behavior.py --config configs/train.yaml --checkpoint checkpoints/z_filtering/latest.pt

"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Iterator
import statistics
import pandas as pd

# the path insert lets you run `python eval/probe_behavior.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
from sklearn.model_selection import StratifiedGroupKFold

from configs.config import Config
from data.dataset import MultiZarrDataset
from models.encoder import HybridEncoder

def setup_logging(checkpoint_path: str, restrict: str = "all") -> Path:
    """derive a log directory from the name of the parent folder of the checkpoint.

    this function tees all stdout to the terminal and to a log file in that
    directory. `restrict` goes into the filename, so the "all" run and the
    "unseen" run of one checkpoint do not give two log files with the same name.
    """
    run_name = Path(checkpoint_path).parent.name
    candidates = [p for p in Path("latent_space").glob(f"{run_name}_*") if p.is_dir()]

    if candidates:
        log_dir = max(candidates, key=lambda p: p.name)
    else:
        log_dir = Path("latent_space") / f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        log_dir.mkdir(parents=True, exist_ok=True)

    tag = "" if restrict == "all" else f"_{restrict}"
    log_path = log_dir / f"behavior_probe{tag}_{time.strftime('%Y%m%d_%H%M%S')}.log"
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
    p.add_argument(
        "--probe-seeds", type=int, default=3,
        help="number of probe restarts per fold (seed, seed+1, ...). metrics are "
        "averaged over restarts and the spread is reported, so run-to-run "
        "differences smaller than the probe's own variance don't read as signal",
    )
    p.add_argument(
        "--standardize", action=argparse.BooleanOptionalAction, default=True,
        help="z-score features using train-fold statistics before fitting the head",
    )
    p.add_argument(
        "--restrict", choices=("all", "seen", "unseen"), default="all",
        help="which experiments to evaluate on, relative to the ones the encoder "
        "was trained on (the zarr_paths in --config). 'unseen' keeps only "
        "annotations from experiments absent from SSL training, so the encoder "
        "provably never saw those pixels -- a stronger guarantee than the spatial "
        "holdout, which partitions crop top-left positions rather than image area "
        "and therefore leaves held-out crops overlapping training crops at the "
        "tile boundaries. 'seen' is the complement. comparing the two says whether "
        "encoder exposure inflates the headline number.",
    )
    p.add_argument(
        "--class-weighted", action=argparse.BooleanOptionalAction, default=True,
        help="weight the probe loss by inverse class frequency so rare classes aren't ignored",
    )
    return p.parse_args()


def disable_augmentations(cfg: Config) -> None:
    """turn off the stochastic augs, so the embeddings stay deterministic for a
    given sampled (t, z, y, x). the per-crop norm stays on, because the encoder
    received its input in that form during training.
    """
    for name in ("aug_intensity", "aug_gamma", "aug_noise", "aug_blur", "aug_flip", "aug_rotation"):
        getattr(cfg, name).enabled = False


def load_dataset(cfg: Config) -> MultiZarrDataset:
    # warning: never overwrite a path list that the caller chose already. an
    # overwrite once replaced the filtered list of main() with the unfiltered
    # list, so the extraction walked a different set, and a different order,
    # than the source of the labels.
    if not cfg.zarr_paths:
        df = pd.read_json(cfg.annotation_path, lines=True)
        cfg.zarr_paths = list(df['zarr_path'].unique())
    ds = MultiZarrDataset(cfg)
    # note: MultiZarrDataset drops without a message any zarr with a blank
    # spreadsheet z_limit, and extract_embeddings indexes the children by
    # position. therefore a drop would attach the labels of one experiment to
    # the crops of another experiment. eval/latent_space.py uses the same guard.
    child_paths = [child.cfg.zarr_path or "" for child in ds.children]
    if child_paths != list(cfg.zarr_paths):
        missing = [p for p in cfg.zarr_paths if p not in child_paths]
        raise RuntimeError(
            "ds.children order doesn't match cfg.zarr_paths -- positional "
            "lookup would read crops from the wrong zarr. dropped/reordered: "
            f"{missing}"
        )
    return ds


@torch.no_grad()
def extract_embeddings(
    cfg: Config,
    encoder: nn.Module,
    device: torch.device,
    seed: int,
    annotations_df: pd.DataFrame
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """run the encoder on the t1 global of many samples. return embeddings, labels and metadata."""
    cfg.seed = seed

    if not cfg.zarr_paths:
        raise ValueError("probe requires cfg.zarr_paths (a multi-zarr dataset)")
    ds = load_dataset(cfg)
    n_samples = len(annotations_df)
    labels_list = sorted(list(annotations_df['label'].unique()))
    label_to_idx = {label: idx for idx, label in enumerate(labels_list)}

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
            # _annotation_tensor returns an UNBATCHED (2, 1, Z, H, W) stack.
            # therefore [0] picks the t1 view, and the unsqueeze below adds the
            # batch axis.
            batch = src._annotation_tensor(row.t1, row.t2, row.z, row.global_x, row.global_y)
            g_t1 = batch["globals"][0].to(device, non_blocking=True).unsqueeze(0)
            label_idx = torch.tensor([label_to_idx[row.label]])
            meta = batch["meta"]

            emb = encoder(g_t1)  # (B, latent_dim)
            embeddings.append(emb.cpu())
            labels.append(label_idx)
            rows.append({
                "label_idx": label_idx,
                # the group id, in X row order. build the fold groups from this
                # field and not from df['dataset_idx']. the code fills X zarr by
                # zarr, although index.jsonl interleaves the zarrs, so the two
                # orders differ.
                "dataset_idx": i,
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


def stratified_group_kfold_split(
    X: torch.Tensor,
    y: torch.Tensor,
    dataset_idx: list,
    seed: int,
    n_splits: int = 5
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_index, test_index in sgkf.split(X, y, groups=dataset_idx):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # WARNING: the yield below is a 5-tuple, and knn_behavior.py unpacks it
        # too. change both files together. the yield includes the held-out
        # groups, because without them you cannot attribute a per-class 0.00 on
        # one fold to an experiment.
        held_out = sorted({dataset_idx[i] for i in test_index}, key=str)
        yield X_train, y_train, X_test, y_test, held_out

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
    class_weighted: bool = True,
    seed: int = 42,
    standardize: bool = True,
    verbose: bool = True,
) -> tuple[float, float, torch.Tensor, nn.Linear]:
    """train a single nn.Linear with AdamW and cross-entropy.

    returns train_acc, val_acc, the confusion matrix and the trained head.

    `seed` controls the head init and the minibatch sampler, so repeated calls
    at different seeds estimate the probe-restart variance. warning: the
    probe-restart variance is NOT the error bar for a comparison of two training
    runs, because it holds the encoder fixed. see README "Claims that were made
    and later retracted".
    """
    # set the seed before the head exists, so the head init is reproducible.
    torch.manual_seed(seed)
    head = nn.Linear(X_train.size(1), n_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    if standardize:
        # use the train-fold statistics only. the raw feature scales differ by
        # orders of magnitude across the dimensions. without the z-score, a
        # fixed lr and a fixed step budget mix up "no signal in the embedding"
        # with "the head never converged". kNN gets the same effect free from
        # the cosine similarity.
        mu = X_train.mean(dim=0, keepdim=True)
        sd = X_train.std(dim=0, keepdim=True).clamp(min=1e-6)
        X_train = (X_train - mu) / sd
        X_val = (X_val - mu) / sd

    if class_weighted:
        # the inverse class frequency, normalized to mean 1, so the head cannot
        # ignore a rare class.
        counts = torch.bincount(y_train, minlength=n_classes).float()
        class_w = counts.sum() / (counts.clamp(min=1) * n_classes)
        ce = nn.CrossEntropyLoss(weight=class_w)
    else:
        ce = nn.CrossEntropyLoss()

    n = X_train.size(0)
    gen = torch.Generator(device=device).manual_seed(seed)

    for step in range(probe_steps):
        idx = torch.randint(0, n, (batch_size,), device=device, generator=gen)
        logits = head(X_train[idx])
        loss = ce(logits, y_train[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

        if verbose and (step == 0 or (step + 1) % max(1, probe_steps // 10) == 0):
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

    return tr_acc, va_acc, cm, head


def macro_f1_from_cm(cm: torch.Tensor) -> tuple[float, list[float]]:
    """macro-averaged F1 from a confusion matrix (rows=true, cols=pred).

    this function skips a class that is absent from val, so that class does not
    drag the mean to 0.
    """
    per_class: list[float] = []
    present: list[float] = []
    for c in range(cm.size(0)):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum()) - tp
        fn = int(cm[c, :].sum()) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class.append(f1)
        if cm[c, :].sum() > 0:
            present.append(f1)
    return (statistics.mean(present) if present else 0.0), per_class


def balanced_acc_from_cm(cm: torch.Tensor) -> float:
    """the mean per-class recall over the classes present in val, that is, the balanced accuracy."""
    recalls: list[float] = []
    for c in range(cm.size(0)):
        total = int(cm[c, :].sum())
        if total > 0:
            recalls.append(int(cm[c, c]) / total)
    return statistics.mean(recalls) if recalls else 0.0


def print_confusion_matrix(cm: torch.Tensor, classes: list[str]) -> None:
    """print a labeled confusion matrix and per-class accuracy."""
    n = cm.size(0)

    print()
    print("  confusion matrix (rows = true class, cols = predicted):")
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
        print(f"    {r} ({classes[r]:30s})  {correct:4d}/{total:4d}  ({pct:.1f}%)")


# two-sided 95% t critical values, keyed by the degrees of freedom. a flat 2.0
# is the large-sample limit and is wrong at 5 folds. see README "How to read the
# metrics" section 5.
_T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
              7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 20: 2.086, 30: 2.042}


def _t_crit_95(df: int) -> float:
    """two-sided 95% t critical value. the value stays conservative between two table entries."""
    if df < 1:
        return float("inf")
    if df >= 60:
        return 2.000   # never go below the large-sample limit
    return _T_CRIT_95[max(k for k in _T_CRIT_95 if k <= df)]


def interpret(val_acc: float, majority: float) -> str:
    # the baseline is the majority class rate, and not uniform chance. see
    # README "How to read the metrics" section 4.
    baseline = majority
    margin = val_acc - baseline
    if margin <= 0.0:
        return (
            f"val accuracy ({val_acc*100:.1f}%) is at or BELOW the baseline "
            f"({baseline*100:.1f}%) — a constant predictor does as well or better. "
            f"read macro_F1 / balanced_acc instead."
        )
    relative = margin / baseline
    if relative < 0.2:
        return (
            f"val accuracy ({val_acc*100:.1f}%) is near the baseline ({baseline*100:.1f}%) — "
            f"encoder does not appear to separate behaviors. the representation may not "
            f"yet be meaningful, or more training steps may be needed."
        )
    if relative < 0.6:
        return (
            f"val accuracy ({val_acc*100:.1f}%) is moderately above the baseline ({baseline*100:.1f}%) — "
            f"encoder is partially separating behaviors. worth tracking across checkpoints "
            f"to see whether it improves with further training."
        )
    return (
        f"val accuracy ({val_acc*100:.1f}%) is well above the baseline ({baseline*100:.1f}%) — "
        f"encoder has learned a representation that separates behaviors well. "
        f"this is the desired outcome."
    )


def main() -> None:
    args = parse_args()
    log_path = setup_logging(args.checkpoint, args.restrict)
    print(f"[probe] logging to  : {log_path}")

    cfg = Config.from_yaml(args.config)
    # snapshot the SSL training set before the probe overwrites cfg.zarr_paths
    # with the larger path list of the annotation file. this line is the last
    # point where the two lists are still separate, and --restrict needs the
    # difference.
    ssl_train_paths = set(cfg.zarr_paths or [])

    df = pd.read_json(cfg.annotation_path, lines=True)
    df = df[df['label'] != 'empty']
    df = df.drop(columns = ['timestamp'])
    df = df[df['label'] != 'slug']
    # warning: eval/knn_behavior.py hardcodes the same literal path. change both
    # files together, or the two evals no longer see identical folds. mixin49 is
    # a bad acquisition and is not part of the data of this project. no run
    # trains on mixin49. the line below drops its 31 annotations, so they cannot
    # leak into the --restrict unseen pool.
    df = df[df["zarr_path"] != "/groups/sgro/sgrolab/jennifer/cryolite/cryolite_mixin_test49_2024-02-08/mixin49_wellB2_24hr_ERH Red FarRed.zarr"]
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass

    # apply the restrict filter before the code derives cfg.zarr_paths and
    # dataset_idx. then the extraction loop opens only the retained experiments,
    # and the fold groups stay aligned.
    if args.restrict != "all":
        if not ssl_train_paths:
            raise ValueError(
                f"--restrict {args.restrict} needs the encoder's training set, but "
                f"{args.config} has no zarr_paths. pass a run's frozen "
                "checkpoints/<run>/config.yaml, not a bare config."
            )
        was_seen = df["zarr_path"].isin(ssl_train_paths)
        n_before = len(df)
        df = df[was_seen] if args.restrict == "seen" else df[~was_seen]
        if df.empty:
            raise ValueError(
                f"--restrict {args.restrict} left no annotations "
                f"(of {n_before}); check that --config matches this encoder."
            )
        print(
            f"[probe] restrict    : {args.restrict}  "
            f"({len(df)}/{n_before} annotations, "
            f"{df['zarr_path'].nunique()} experiments)"
        )
        print(f"[probe] label counts: {df['label'].value_counts().to_dict()}")
        if args.restrict == "unseen":
            print("[probe] these experiments were absent from SSL training, so the "
                  "encoder has never seen these pixels in any form.")

    cfg.zarr_paths = list(df['zarr_path'].unique())
    zarr_path_to_idx = {p: i for i, p in enumerate(cfg.zarr_paths)}
    df['dataset_idx'] = df['zarr_path'].map(zarr_path_to_idx)

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
    step = ckpt.get("step", "unknown")
    print(f"[probe] checkpoint saved at training step {step}")
    encoder = HybridEncoder(
        stem_channels=cfg.stem_channels,
        latent_dim=cfg.latent_dim,
        norm_groups=cfg.norm_groups,
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    n_classes = len(df['label'].unique())
    chance_pct = 100.0 / n_classes
    print(f"[probe] n_classes  : {n_classes}  (chance baseline = {chance_pct:.1f}%)")
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
    fold_macro_f1s, fold_bal_accs = [], []
    fold_heads = []  # store the trained heads for later
    # the per-fold spread across the probe restarts. the spread measures the
    # noise of the head fit on a fixed encoder. the spread is not the encoder
    # run-to-run variance. see README "Claims that were made and later
    # retracted".
    fold_restart_sds = []

    # warning: the groups must come from `rows`, which is in X order, and not
    # from df['dataset_idx'], which is in annotation-file order. df once made
    # StratifiedGroupKFold group the wrong rows, and the leave-experiment-out
    # split became near-random.
    groups = [r["dataset_idx"] for r in rows]

    probe_seeds = [args.seed + s for s in range(max(1, args.probe_seeds))]
    print(f"[probe] probe restarts per fold: {len(probe_seeds)}  (seeds {probe_seeds})")
    print()

    class_names_sorted = sorted(df['label'].unique())
    for fold_idx, (X_train, y_train, X_val, y_val, held_out) in enumerate(stratified_group_kfold_split(X, y, groups, args.seed, args.n_splits)):
        val_counts = torch.bincount(y_val, minlength=n_classes).tolist()
        print(f"[probe] fold {fold_idx+1}/{args.n_splits}  (grouped by zarr volume, stratified by label)")
        print(f"[probe] train n={X_train.size(0)}  counts={torch.bincount(y_train, minlength=n_classes).tolist()}")
        print(f"[probe] val   n={X_val.size(0)}  counts={val_counts}  held out={held_out}")
        # you cannot interpret a per-class F1 at val n<=2. see README "How to
        # read the metrics" section 5.
        thin = [f"{class_names_sorted[c]} n={val_counts[c]}"
                for c in range(n_classes) if val_counts[c] <= 2]
        if thin:
            print(f"[probe]   WARNING thin val support: {', '.join(thin)} "
                  f"-- per-class F1 not interpretable on this fold")
        print()

        # --- train the probe once for each restart seed, then take the mean of the metrics ---
        print(f"[probe] training linear head (in_dim={cfg.latent_dim}, out_dim={n_classes})...")
        seed_tr, seed_va, seed_macro, seed_bal = [], [], [], []
        cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
        head = None
        for s_i, probe_seed in enumerate(probe_seeds):
            tr_s, va_s, cm_s, head_s = train_probe(
                X_train, y_train, X_val, y_val,
                n_classes=n_classes,
                probe_steps=args.probe_steps,
                lr=args.probe_lr,
                batch_size=args.probe_batch_size,
                weight_decay=args.probe_weight_decay,
                device=device,
                class_weighted=args.class_weighted,
                seed=probe_seed,
                standardize=args.standardize,
                verbose=(s_i == 0),
            )
            macro_s, _ = macro_f1_from_cm(cm_s)
            bal_s = balanced_acc_from_cm(cm_s)
            seed_tr.append(tr_s)
            seed_va.append(va_s)
            seed_macro.append(macro_s)
            seed_bal.append(bal_s)
            cm = cm + cm_s
            if s_i == 0:
                head = head_s
            if len(probe_seeds) > 1:
                print(
                    f"  [probe] restart seed={probe_seed}  val_acc={va_s:.3f}  "
                    f"macro_F1={macro_s:.3f}"
                )

        tr_acc = statistics.mean(seed_tr)
        va_acc = statistics.mean(seed_va)
        macro = statistics.mean(seed_macro)
        bal = statistics.mean(seed_bal)
        va_sd = statistics.stdev(seed_va) if len(seed_va) > 1 else 0.0
        macro_sd = statistics.stdev(seed_macro) if len(seed_macro) > 1 else 0.0

        fold_train_accs.append(tr_acc)
        fold_val_accs.append(va_acc)
        fold_cms.append(cm)
        fold_heads.append((head.cpu(), va_acc))
        fold_macro_f1s.append(macro)
        fold_bal_accs.append(bal)
        fold_restart_sds.append(macro_sd)

        # --- fold-specific report ---
        print()
        print("=" * 70)
        print(f"  FOLD {fold_idx+1}/{args.n_splits} RESULT @ training step {step}")
        print(f"  (mean over {len(probe_seeds)} probe restart(s))")
        print("=" * 70)
        print(f"  train_acc        : {tr_acc*100:6.2f}%")
        print(f"  val_acc          : {va_acc*100:6.2f}%  ± {va_sd*100:.2f}   (chance = {chance_pct:.1f}%)")
        print(f"  balanced_acc     : {bal*100:6.2f}%   (mean per-class recall)")
        print(f"  macro_F1         : {macro:6.3f}  ± {macro_sd:.3f}")
        fold_majority = max(val_counts) / max(sum(val_counts), 1)
        print(f"  val_acc - chance : {(va_acc - 1.0/n_classes)*100:+6.2f} percentage points")
        print(f"  val_acc - majority ({fold_majority*100:.1f}%) : {(va_acc - fold_majority)*100:+6.2f} percentage points")
        print_confusion_matrix(cm, sorted(df['label'].unique()))
        print()
        print(f"  interpretation: {interpret(va_acc, fold_majority)}")
        print("=" * 70)
        print()
    
    train_mean = statistics.mean(fold_train_accs)
    train_std = statistics.stdev(fold_train_accs) if len(fold_train_accs) > 1 else 0.0
    val_mean = statistics.mean(fold_val_accs)
    val_std = statistics.stdev(fold_val_accs) if len(fold_val_accs) > 1 else 0.0
    macro_mean = statistics.mean(fold_macro_f1s)
    macro_std = statistics.stdev(fold_macro_f1s) if len(fold_macro_f1s) > 1 else 0.0
    bal_mean = statistics.mean(fold_bal_accs)
    bal_std = statistics.stdev(fold_bal_accs) if len(fold_bal_accs) > 1 else 0.0

    cm_total = torch.stack(fold_cms).sum(dim=0)
    pooled_macro, per_class_f1 = macro_f1_from_cm(cm_total)

    print()
    print("#" * 70)
    print(f"  CROSS-VALIDATED SUMMARY ({args.n_splits} folds, grouped by zarr volume)")
    print(f"  class_weighted={args.class_weighted}")
    print("#" * 70)
    print(f"  train_acc    : {train_mean*100:6.2f}%  ± {train_std*100:.2f}  (per-fold: {[f'{a*100:.1f}' for a in fold_train_accs]})")
    print(f"  val_acc      : {val_mean*100:6.2f}%  ± {val_std*100:.2f}  (per-fold: {[f'{a*100:.1f}' for a in fold_val_accs]})")
    print(f"  balanced_acc : {bal_mean*100:6.2f}%  ± {bal_std*100:.2f}  (mean per-class recall)")
    print(f"  macro_F1     : {macro_mean:6.3f}  ± {macro_std:.3f}  (per-fold: {[f'{m:.3f}' for m in fold_macro_f1s]})")
    # the probe prints both baselines. read the majority baseline. see README
    # "How to read the metrics" section 4.
    label_counts = df['label'].value_counts()
    majority_pct = 100.0 * label_counts.max() / label_counts.sum()
    print(f"  val_acc - uniform chance ({chance_pct:.1f}%)   : {(val_mean - 1.0/n_classes)*100:+6.2f} percentage points")
    print(f"  val_acc - majority class ({majority_pct:.1f}%) : {(val_mean*100 - majority_pct):+6.2f} percentage points")
    print(f"    (majority baseline = always predict '{label_counts.idxmax()}'; macro_F1 and")
    print(f"     balanced_acc are the metrics to compare across runs, not val_acc)")
    if any(sd > 0 for sd in fold_restart_sds):
        # warning: an old version of this block told the reader that a gap
        # bigger than 2x the restart spread meant that one run was better. that
        # rule is WRONG, and somebody retracted the conclusions from it. the
        # restart spread is the noise of a new head fit on a FIXED encoder, so
        # it says nothing about the difference between two training runs of the
        # same config. nobody measured the encoder run-to-run variance on this
        # project. see README "Claims that were made and later retracted".
        restart_sd = statistics.mean(fold_restart_sds)
        print()
        print("  " + "-" * 66)
        print("  COMPARING THIS RUN TO ANOTHER ONE")
        print("  " + "-" * 66)
        print(f"  use macro_F1 ({macro_mean:.3f}) on --restrict unseen, and keep --seed {args.seed}")
        print("  the same in both runs.")
        print()
        print(f"    probe-restart spread on THIS fixed encoder : +- {restart_sd:.3f}")
        print(f"    fold spread (which experiments are held out): +- {macro_std:.3f}")
        print()
        print("  neither is the error bar for a run-vs-run comparison. the restart figure")
        print("  holds the encoder fixed, and encoder run-to-run variance has never been")
        print("  measured here -- so a gap between two runs of a few hundredths is NOT")
        print("  resolvable. train one config twice before trusting any such ranking.")
        print("  " + "-" * 66)
    print()
    print(f"  pooled confusion matrix (summed across all folds x {len(probe_seeds)} restart(s)):")
    print_confusion_matrix(cm_total, sorted(df['label'].unique()))
    print()
    # WARNING: the val n column below sums over the restart seeds, so it shows
    # 3x the true support on a 3-restart run. the README notes this known bug.
    # the probe reports the per-class F1 with its fold-level spread, because the
    # pooled figure alone cannot rank the classes at this support. see README
    # "How to read the metrics" section 6.
    class_names = sorted(df['label'].unique())
    per_fold_class_f1: list[list[float]] = [[] for _ in class_names]
    per_class_support: list[int] = [0] * len(class_names)
    for fold_cm in fold_cms:
        _, fold_pc = macro_f1_from_cm(fold_cm)
        for c in range(len(class_names)):
            support = int(fold_cm[c, :].sum())
            per_class_support[c] += support
            # a class that is absent from the val of this fold has no estimate.
            # a mean that includes its placeholder 0.0 would invent a low score.
            if support > 0:
                per_fold_class_f1[c].append(fold_pc[c])

    print(f"  pooled macro-F1: {pooled_macro:.3f}   per-class F1:")
    print(f"    {'class':30s} {'pooled':>7s} {'mean':>7s} {'sd':>7s} {'val n':>7s}   per-fold")
    for c, (name, f1) in enumerate(zip(class_names, per_class_f1)):
        vals = per_fold_class_f1[c]
        mean_f1 = statistics.mean(vals) if vals else float("nan")
        sd_f1 = statistics.stdev(vals) if len(vals) > 1 else 0.0
        per_fold = "[" + ", ".join(f"{v:.2f}" for v in vals) + "]"
        print(f"    {name:30s} {f1:7.3f} {mean_f1:7.3f} {sd_f1:7.3f} "
              f"{per_class_support[c]:7d}   {per_fold}")
    # flag a ranking that the data cannot support.
    ranked = sorted(
        (c for c in range(len(class_names)) if per_fold_class_f1[c]),
        key=lambda c: statistics.mean(per_fold_class_f1[c]),
    )
    if len(ranked) > 1:
        lo, hi = ranked[0], ranked[-1]
        gap = (statistics.mean(per_fold_class_f1[hi])
               - statistics.mean(per_fold_class_f1[lo]))

        def _se(vals: list[float]) -> float:
            if len(vals) < 2:
                return 0.0
            return statistics.stdev(vals) / (len(vals) ** 0.5)

        # use the se of the DIFFERENCE between the two compared classes, and not
        # a mean sd across all classes. use a t critical value, and not a flat
        # 2x. see README "How to read the metrics" section 5.
        se_diff = (_se(per_fold_class_f1[lo]) ** 2
                   + _se(per_fold_class_f1[hi]) ** 2) ** 0.5
        # WARNING: this variable is dof, and not df. `df` is the annotations
        # DataFrame in this scope, and a shadow of `df` breaks the checkpoint
        # save below.
        dof = min(len(per_fold_class_f1[lo]), len(per_fold_class_f1[hi])) - 1
        t_crit = _t_crit_95(dof)
        print()
        print(f"    worst-to-best per-class gap: {gap:.3f} "
              f"({class_names[lo]} -> {class_names[hi]}), "
              f"se of the difference {se_diff:.3f}, "
              f"t(df={dof})={t_crit:.2f}")
        if se_diff <= 0 or gap < t_crit * se_diff:
            print(f"    -> within {t_crit:.2f} se: the per-class RANKING is not "
                  "resolvable at this sample size. do not tune against it, and "
                  "do not read a story into which class is worst.")
        else:
            print(f"    -> exceeds {t_crit:.2f} se, so the ordering of these two "
                  "is real. adjacent classes may still be indistinguishable.")
        print("    (folds are grouped by experiment, so these are correlated and "
              "the se is optimistic; treat it as a floor on the uncertainty.)")
    print()
    print(f"  interpretation: {interpret(val_mean, majority_pct / 100.0)}")
    print("#" * 70)

    # save the linear head of the best fold for grad-cam
    best_fold_idx = fold_val_accs.index(max(fold_val_accs))
    best_head, best_va_acc = fold_heads[best_fold_idx]

    log_dir = Path(log_path).parent
    checkpoint = {
        "head_state_dict": best_head.state_dict(),
        "input_dim": cfg.latent_dim,
        "output_dim": n_classes,
        # the same order that the confusion matrices and the per-class tables use.
        "class_names": class_names_sorted,
        "val_acc": best_va_acc,
        "fold_idx": best_fold_idx,
        "args": vars(args),
    }

    # warning: only the full-set run can own the name "behavior_probe.pt",
    # because run_gradcam.py and run_eg.py load that name. a --restrict run
    # writes a file with a suffix.
    ckpt_path = log_dir / f"behavior_probe{'' if args.restrict == 'all' else '_' + args.restrict}.pt"
    torch.save(checkpoint, ckpt_path)
    print()
    print(f"[probe] saved best fold (fold {best_fold_idx+1}) linear head to {ckpt_path}")


if __name__ == "__main__":
    main()
