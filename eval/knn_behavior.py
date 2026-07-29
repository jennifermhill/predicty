"""non-parametric k-NN evaluation of behavior identity on a frozen encoder.

this mirrors eval/probe_behavior.py but replaces the trainable nn.Linear head
with a parameter-free k-nearest-neighbor vote. the encoder is frozen, we extract
t1 global-crop embeddings over all annotated samples, split with the same
StratifiedGroupKFold (grouped by zarr volume, stratified by label), and for each
val embedding predict its label by a cosine vote over its k nearest neighbors in
the fold's train embeddings.

k-NN is the standard frozen-feature SSL benchmark (DINO/iBOT), so the number is
comparable to the literature and has no lr/steps/weight-decay to tune.

--weighted gives closer neighbors a bigger say (softmax over cosine sim),
which helps the rare classes without ignoring larger k.

alongside accuracy this reports two purity metrics off the same k-NN graph:

  label purity  fraction of a val point's neighbors sharing its label, folded
                exactly like the accuracy. the continuous quantity underneath
                accuracy -- it moves when neighborhoods improve without any
                prediction flipping, so it ranks runs more stably on a small
                annotated set.
  batch purity  fraction of a point's neighbors from its own zarr volume,
                pooled over everything rather than folded (see batch_purity).
                high values mean the local geometry encodes which experiment a
                crop came from, so a behavior gain arriving together with a
                batch-purity gain is probably not a behavior gain.

both are reported normalized against their chance baseline, because raw purity
is dominated by the class prior (and by volume sizes) and is not comparable
between runs, folds, or datasets without it.

usage:
    python eval/knn_behavior.py --config configs/train.yaml --checkpoint checkpoints/z_filtering/latest.pt
    python eval/knn_behavior.py --config configs/train.yaml --checkpoint checkpoints/z_filtering/latest.pt --k-values 1 5 10 --weighted
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import pandas as pd

# allow running as `python eval/knn_behavior.py` from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from configs.config import Config
from models.encoder import HybridEncoder

# reuse the probe's extraction + splitting + reporting verbatim so the two evals
# see identical embeddings, folds, and confusion-matrix formatting.
from eval.probe_behavior import (
    disable_augmentations,
    extract_embeddings,
    print_confusion_matrix,
    stratified_group_kfold_split,
)


def setup_logging(checkpoint_path: str) -> Path:
    """derive a log directory from the checkpoint's parent folder name and tee
    stdout to both the terminal and a knn_eval_*.log file inside it. mirrors
    probe_behavior.setup_logging but with a knn-specific filename."""
    run_name = Path(checkpoint_path).parent.name
    candidates = [p for p in Path("latent_space").glob(f"{run_name}_*") if p.is_dir()]

    if candidates:
        log_dir = max(candidates, key=lambda p: p.name)
    else:
        log_dir = Path("latent_space") / f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"knn_eval_{time.strftime('%Y%m%d_%H%M%S')}.log"
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
    p.add_argument("--k-values", type=int, nargs="+", default=[1, 5, 10, 25, 50], help="k values to sweep")
    p.add_argument("--weighted", action="store_true", help="softmax-weighted vote instead of a plain majority")
    p.add_argument("--temperature", type=float, default=0.07, help="softmax temperature for --weighted (DINO default)")
    p.add_argument("--class-balanced", action="store_true", help="divide neighbor votes by train class frequency so rare classes aren't outvoted just for being scarce")
    p.add_argument("--seed", type=int, default=42, help="seed for the fold split")
    p.add_argument("--n_splits", type=int, default=5, help="folds for stratified k-fold splitting")
    return p.parse_args()


def knn_predict(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    k: int,
    n_classes: int,
    weighted: bool,
    temperature: float,
    class_balanced: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """predict a label for every row of X_val by a cosine vote over its k nearest
    neighbors in X_train. returns (preds (n_val,) long, neighbor_labels (n_val, k)).

    k is clamped to the train size so tiny folds don't crash. each neighbor
    contributes to its label's score: a plain vote gives every neighbor weight 1;
    --weighted gives closer neighbors more via softmax over cosine similarity.

    neighbor_labels is returned so label_purity can reduce the same neighbor
    matrix a different way without paying for a second cosine product.
    """
    k = min(k, X_train.size(0))
    sim = F.normalize(X_val, dim=1) @ F.normalize(X_train, dim=1).t()  # higher = closer
    top_sim, top_idx = sim.topk(k, dim=1)
    neighbor_labels = y_train[top_idx]  # (n_val, k)

    weights = torch.softmax(top_sim / temperature, dim=1) if weighted else torch.ones_like(top_sim)
    scores = torch.zeros(X_val.size(0), n_classes)
    scores.scatter_add_(1, neighbor_labels, weights)
    if class_balanced:
        # prior correction: divide each class's vote by its train frequency so a
        # rare class isn't outvoted just for being scarce among the neighbors.
        counts = torch.bincount(y_train, minlength=n_classes).float().clamp(min=1)
        scores = scores / counts
    return scores.argmax(dim=1), neighbor_labels


def confusion(y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: int) -> torch.Tensor:
    cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[t, p] += 1
    return cm


def macro_f1(cm: torch.Tensor) -> tuple[float, list[float]]:
    """macro-averaged F1 from a confusion matrix (rows=true, cols=pred). classes
    absent from val (empty row) are skipped so they don't drag the mean down."""
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


def balanced_acc(cm: torch.Tensor) -> float:
    """mean per-class recall over classes present in val (balanced accuracy)."""
    recs = [int(cm[c, c]) / int(cm[c, :].sum()) for c in range(cm.size(0)) if cm[c, :].sum() > 0]
    return statistics.mean(recs) if recs else 0.0


def label_purity(
    neighbor_labels: torch.Tensor,
    y_val: torch.Tensor,
    y_train: torch.Tensor,
    n_classes: int,
) -> tuple[float, float, float, list[float]]:
    """fraction of each val point's k neighbors that share its label.

    this is the continuous quantity underneath k-NN accuracy: accuracy asks
    whether the vote came out right, purity asks how lopsided the vote was. a
    run can improve from 3-of-10 matching neighbors to 5-of-10 without a single
    prediction flipping, so purity separates runs that accuracy scores as tied
    -- useful when the annotated set is small enough that fold-to-fold accuracy
    noise rivals the effect being measured.

    deliberately unweighted even when --weighted is set: purity describes the
    neighborhood, not the voting rule, so it stays comparable across flag
    settings.

    returns (purity, normalized, baseline, per_class_purity).
    """
    matches = (neighbor_labels == y_val[:, None]).float()
    per_point = matches.mean(dim=1)
    purity = per_point.mean().item()

    # chance baseline: a random train neighbor matches a val point of class c
    # with probability p_train[c], so the baseline is the class-frequency
    # weighted mean over the val set. raw purity is NOT comparable without
    # this -- on an imbalanced 4-class split a useless embedding still scores
    # far above 25%, and the baseline itself moves fold to fold because the
    # grouped split changes the train class mix.
    p_train = torch.bincount(y_train, minlength=n_classes).float() / y_train.numel()
    baseline = p_train[y_val].mean().item()
    normalized = (purity - baseline) / (1.0 - baseline) if baseline < 1.0 else 0.0

    per_class: list[float] = []
    for c in range(n_classes):
        mask = y_val == c
        per_class.append(per_point[mask].mean().item() if bool(mask.any()) else float("nan"))
    return purity, normalized, baseline, per_class


def batch_purity(X: torch.Tensor, groups: list[int], k: int) -> tuple[float, float, float]:
    """fraction of each point's k neighbors drawn from its own zarr volume.

    high values mean the local geometry encodes which experiment a crop came
    from rather than what the animal was doing -- the local counterpart to the
    Wasserstein cloud-separation check and to the training-time dataset
    adversary, and more sensitive than either: two volumes can overlap almost
    perfectly in aggregate while still being locally segregated into
    interleaved single-volume filaments that no global metric will flag.

    pooled over every annotated embedding on purpose. the cross-validation
    folds are grouped by zarr volume, so inside a fold a val point's train
    neighbors can never come from its own volume and same-volume purity is
    structurally zero -- it would report perfect integration no matter how bad
    the batch effect actually is. pooling with self excluded is what kBET/LISI
    do and is the only framing where the number means anything.

    returns (purity, normalized, baseline).
    """
    g = torch.as_tensor(groups, dtype=torch.long)
    if g.numel() != X.size(0):
        raise ValueError(
            f"groups has {g.numel()} entries but X has {X.size(0)} rows; "
            "groups must be in X order (from `rows`, not the annotation df)"
        )
    n = X.size(0)
    if n < 2:
        return 0.0, 0.0, 0.0
    k = min(k, n - 1)
    normed = F.normalize(X, dim=1)
    sim = normed @ normed.t()
    sim.fill_diagonal_(float("-inf"))  # a point is never its own neighbor
    top_idx = sim.topk(k, dim=1).indices
    per_point = (g[top_idx] == g[:, None]).float().mean(dim=1)

    # chance baseline: for a point in a volume holding n_g annotated crops, a
    # random *other* point shares its volume with probability
    # (n_g - 1) / (n - 1). volumes differ a lot in annotation count, so this is
    # averaged per point rather than assumed uniform.
    counts = torch.bincount(g).float()
    baseline = ((counts[g] - 1.0) / (n - 1)).mean().item()
    purity = per_point.mean().item()
    normalized = (purity - baseline) / (1.0 - baseline) if baseline < 1.0 else 0.0
    return purity, normalized, baseline


def interpret_batch_purity(normalized: float) -> str:
    if normalized < 0.1:
        return (
            "volumes are locally well mixed -- neighborhoods are not organized by "
            "experiment, so behavior signal is unlikely to be batch effect in disguise."
        )
    if normalized < 0.3:
        return (
            "partial local batch structure -- some neighborhoods lean toward their own "
            "volume. compare against behavior purity: if both moved the same direction "
            "between runs, the behavior gain is suspect."
        )
    return (
        "strong local batch structure -- neighborhoods largely reflect which volume a "
        "crop came from. k-NN behavior numbers should be treated as contaminated by "
        "experiment identity until this comes down."
    )


def interpret(val_acc: float, n_classes: int) -> str:
    chance = 1.0 / n_classes
    relative = (val_acc - chance) / chance
    if relative < 0.2:
        return (
            f"k-NN accuracy ({val_acc*100:.1f}%) is near chance ({chance*100:.1f}%) - embedding "
            f"neighborhoods are not label-coherent; local structure does not separate behaviors."
        )
    if relative < 0.6:
        return (
            f"k-NN accuracy ({val_acc*100:.1f}%) is moderately above chance ({chance*100:.1f}%) - "
            f"neighborhoods are partially label-coherent. if the linear probe is much higher, "
            f"classes are linearly separable but locally tangled."
        )
    return (
        f"k-NN accuracy ({val_acc*100:.1f}%) is well above chance ({chance*100:.1f}%) - the local "
        f"geometry of the representation separates behaviors, not just a hyperplane."
    )


def main() -> None:
    args = parse_args()
    log_path = setup_logging(args.checkpoint)
    print(f"[knn] logging to  : {log_path}")

    cfg = Config.from_yaml(args.config)

    # identical df preprocessing to probe_behavior so the folds line up exactly.
    df = pd.read_json(cfg.annotation_path, lines=True)
    df = df[df["label"] != "empty"]
    df = df.drop(columns=["timestamp"])
    df = df[df["label"] != "slug"]
    df = df[df["zarr_path"] != "/groups/sgro/sgrolab/jennifer/cryolite/cryolite_mixin_test49_2024-02-08/mixin49_wellB2_24hr_ERH Red FarRed.zarr"]
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass
    cfg.zarr_paths = list(df["zarr_path"].unique())
    zarr_path_to_idx = {p: i for i, p in enumerate(cfg.zarr_paths)}
    df["dataset_idx"] = df["zarr_path"].map(zarr_path_to_idx)

    device = torch.device(args.device or cfg.device)
    print(f"[knn] device      : {device}")
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        print(f"[knn] cuda gpu    : {torch.cuda.get_device_name(idx)}")
    print(f"[knn] config      : {args.config}")
    print(f"[knn] checkpoint  : {args.checkpoint}")
    print(f"[knn] n_samples   : {len(df)}")
    print(f"[knn] k-values    : {args.k_values}  (weighted={args.weighted})")
    print(f"[knn] seed        : {args.seed}")
    print()

    # --- load encoder (frozen) ---
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    step = ckpt.get("step", "unknown")
    print(f"[knn] checkpoint saved at training step {step}")
    encoder = HybridEncoder(
        stem_channels=cfg.stem_channels,
        latent_dim=cfg.latent_dim,
        norm_groups=cfg.norm_groups,
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    classes = sorted(df["label"].unique())
    n_classes = len(classes)
    chance_pct = 100.0 / n_classes
    print(f"[knn] n_classes   : {n_classes}  (chance baseline = {chance_pct:.1f}%)")
    print()

    # --- extract embeddings once (shared across all folds and k values) ---
    print(f"[knn] extracting {len(df)} embeddings (encoder forward, no augmentations)...")
    t0 = time.time()
    disable_augmentations(cfg)
    X, y, rows = extract_embeddings(cfg, encoder, device, args.seed, df)
    print(f"[knn] extracted in {time.time()-t0:.1f}s.  X.shape={tuple(X.shape)}  y.shape={tuple(y.shape)}")
    print(f"[knn] label counts: {torch.bincount(y, minlength=n_classes).tolist()}  ({classes})")
    print()

    # per-k accumulators across folds
    accs = {k: [] for k in args.k_values}
    macros = {k: [] for k in args.k_values}
    bals = {k: [] for k in args.k_values}
    purs = {k: [] for k in args.k_values}
    pur_norms = {k: [] for k in args.k_values}
    # per-class purity per fold; entries are nan for classes absent from that
    # fold's val split, which the pooled report skips rather than counts as 0.
    pur_classes: dict[int, list[list[float]]] = {k: [] for k in args.k_values}
    cms = {k: torch.zeros(n_classes, n_classes, dtype=torch.long) for k in args.k_values}

    # groups come from `rows` (X order), not df["dataset_idx"] (annotation-file
    # row order). extract_embeddings fills X zarr-by-zarr while index.jsonl
    # interleaves zarrs, so the df ordering grouped the wrong rows and the
    # leave-experiment-out split degenerated into a near-random one.
    groups = [r["dataset_idx"] for r in rows]

    for fold_idx, (X_train, y_train, X_val, y_val) in enumerate(
        stratified_group_kfold_split(X, y, groups, args.seed, args.n_splits)
    ):
        print(f"[knn] fold {fold_idx+1}/{args.n_splits}  (grouped by zarr volume, stratified by label)")
        print(f"[knn] train n={X_train.size(0)}  val n={X_val.size(0)}")
        for k in args.k_values:
            preds, neighbor_labels = knn_predict(X_train, y_train, X_val, k, n_classes, args.weighted, args.temperature, args.class_balanced)
            cm = confusion(y_val, preds, n_classes)
            acc = (preds == y_val).float().mean().item()
            macro, _ = macro_f1(cm)
            bal = balanced_acc(cm)
            pur, pur_norm, pur_base, pur_per_class = label_purity(
                neighbor_labels, y_val, y_train, n_classes
            )
            accs[k].append(acc)
            macros[k].append(macro)
            bals[k].append(bal)
            purs[k].append(pur)
            pur_norms[k].append(pur_norm)
            pur_classes[k].append(pur_per_class)
            cms[k] += cm
            print(
                f"    k={k:2d}  acc={acc*100:6.2f}%  bal_acc={bal*100:6.2f}%  "
                f"macro-F1={macro:.3f}  purity={pur*100:6.2f}% "
                f"(chance {pur_base*100:5.2f}%, norm {pur_norm:+.3f})"
            )
        print()

    # --- cross-validated summary: one row per k ---
    print("#" * 70)
    print(f"  k-NN CROSS-VALIDATED SUMMARY ({args.n_splits} folds, grouped by zarr volume)")
    print(f"  weighted={args.weighted}  class_balanced={args.class_balanced}  @ training step {step}  (chance = {chance_pct:.1f}%)")
    print("#" * 70)
    print(
        f"  {'k':>3}  {'acc':>7}  {'acc std':>7}  {'bal_acc':>8}  {'macroF1':>8}  "
        f"{'mF1 std':>7}  {'purity':>7}  {'pur norm':>8}  {'pn std':>7}"
    )
    for k in args.k_values:
        a_mean, a_std = statistics.mean(accs[k]), statistics.stdev(accs[k]) if len(accs[k]) > 1 else 0.0
        b_mean = statistics.mean(bals[k])
        m_mean, m_std = statistics.mean(macros[k]), statistics.stdev(macros[k]) if len(macros[k]) > 1 else 0.0
        p_mean = statistics.mean(purs[k])
        pn_mean = statistics.mean(pur_norms[k])
        pn_std = statistics.stdev(pur_norms[k]) if len(pur_norms[k]) > 1 else 0.0
        print(
            f"  {k:>3}  {a_mean*100:6.2f}%  {a_std*100:7.2f}  {b_mean*100:7.2f}%  "
            f"{m_mean:8.3f}  {m_std:7.3f}  {p_mean*100:6.2f}%  {pn_mean:+8.3f}  {pn_std:7.3f}"
        )
    print()
    print("  pur norm = purity normalized against the class prior: +1.000 is perfect")
    print("  label-coherent neighborhoods, 0.000 is chance, negative means neighbors")
    print("  are anti-correlated with the label. compare runs on this, not raw purity.")
    print()

    # --- pooled batch purity: is the local geometry organized by experiment? ---
    # not folded, and not optional reading: a behavior gain that arrives together
    # with a batch-purity gain is probably the encoder getting better at telling
    # experiments apart. see batch_purity's docstring for why this must be pooled.
    print("=" * 70)
    print("  BATCH PURITY by zarr volume (pooled over all annotated crops, self excluded)")
    print("=" * 70)
    print(f"  {'k':>3}  {'same-volume':>12}  {'chance':>7}  {'normalized':>10}")
    batch_norms: dict[int, float] = {}
    for k in args.k_values:
        bp, bp_norm, bp_base = batch_purity(X, groups, k)
        batch_norms[k] = bp_norm
        print(f"  {k:>3}  {bp*100:11.2f}%  {bp_base*100:6.2f}%  {bp_norm:+10.3f}")
    print()
    worst_k = max(args.k_values, key=lambda k: batch_norms[k])
    print(f"  interpretation (worst k={worst_k}): {interpret_batch_purity(batch_norms[worst_k])}")
    print()

    # --- detailed pooled report for the best-accuracy k ---
    best_k = max(args.k_values, key=lambda k: statistics.mean(accs[k]))
    best_acc = statistics.mean(accs[best_k])
    _, per_class_f1 = macro_f1(cms[best_k])
    print("=" * 70)
    print(f"  BEST k = {best_k}   (acc {best_acc*100:.2f}%, macro-F1 {statistics.mean(macros[best_k]):.3f})")
    print("=" * 70)
    print("  pooled confusion matrix (summed across all folds):")
    print_confusion_matrix(cms[best_k], classes)
    print()
    print("  per-class F1 and neighborhood purity (pooled):")
    for c, (name, f1) in enumerate(zip(classes, per_class_f1)):
        # average each class's purity over the folds where it actually appeared
        # in val; a class missing from every fold prints "n/a" instead of 0.
        seen = [fold[c] for fold in pur_classes[best_k] if fold[c] == fold[c]]
        purity_str = f"purity={statistics.mean(seen)*100:6.2f}%" if seen else "purity=   n/a"
        print(f"    {name:30s}  F1={f1:.3f}  {purity_str}")
    print()
    print(f"  interpretation: {interpret(best_acc, n_classes)}")
    print("#" * 70)


if __name__ == "__main__":
    main()
