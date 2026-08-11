"""non-parametric k-NN evaluation of behavior identity on a frozen encoder.

this module mirrors eval/probe_behavior.py, but it replaces the trainable
nn.Linear head with a parameter-free k-nearest-neighbor vote. the encoder stays
frozen. the module extracts t1 global-crop embeddings over all annotated
samples, then splits them with the same StratifiedGroupKFold: the groups are
zarr volumes, and the split is stratified by label. for each val embedding, a
cosine vote over the k nearest neighbors in the train embeddings of the fold
gives the predicted label.

k-NN is the standard frozen-feature SSL benchmark (DINO/iBOT). therefore the
number compares to the literature, and it has no lr, no steps and no
weight-decay to tune.

--weighted gives a closer neighbor a bigger vote (softmax over cosine sim).
--weighted helps the rare classes and keeps a larger k useful.

next to accuracy, this module reports two purity metrics from the same k-NN
graph:

  label purity  the fraction of the neighbors of a val point that share its
                label, folded like the accuracy. label purity is the continuous
                quantity under accuracy, so it ranks runs more stably on a
                small annotated set.
  batch purity  the fraction of the neighbors of a point that come from its own
                zarr volume, pooled and not folded (see batch_purity). a
                behavior gain that arrives together with a batch-purity gain is
                probably not a behavior gain.

this module normalizes both metrics against their chance baseline. the class
prior and the volume sizes dominate raw purity, so raw purity alone does not
compare across runs.

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

# the path insert lets you run `python eval/knn_behavior.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from configs.config import Config
from models.encoder import HybridEncoder

# reuse the extraction, the split and the report of the probe without a change,
# so the two evals see identical embeddings, identical folds and an identical
# confusion-matrix layout.
from eval.probe_behavior import (
    disable_augmentations,
    extract_embeddings,
    print_confusion_matrix,
    stratified_group_kfold_split,
)


def setup_logging(checkpoint_path: str) -> Path:
    """derive a log directory from the name of the parent folder of the checkpoint.

    this function tees stdout to the terminal and to a knn_eval_*.log file in
    that directory. the function mirrors probe_behavior.setup_logging, but the
    filename is knn-specific.
    """
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

    this function clamps k to the train size, so a tiny fold does not crash it.
    the function also returns neighbor_labels, so label_purity can reduce the
    same neighbor matrix and not pay for a second cosine product.
    """
    k = min(k, X_train.size(0))
    sim = F.normalize(X_val, dim=1) @ F.normalize(X_train, dim=1).t()  # higher = closer
    top_sim, top_idx = sim.topk(k, dim=1)
    neighbor_labels = y_train[top_idx]  # (n_val, k)

    weights = torch.softmax(top_sim / temperature, dim=1) if weighted else torch.ones_like(top_sim)
    scores = torch.zeros(X_val.size(0), n_classes)
    scores.scatter_add_(1, neighbor_labels, weights)
    if class_balanced:
        # a prior correction, so the other classes do not outvote a rare class
        # only because the rare class is scarce.
        counts = torch.bincount(y_train, minlength=n_classes).float().clamp(min=1)
        scores = scores / counts
    return scores.argmax(dim=1), neighbor_labels


def confusion(y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: int) -> torch.Tensor:
    cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[t, p] += 1
    return cm


def macro_f1(cm: torch.Tensor) -> tuple[float, list[float]]:
    """macro-averaged F1 from a confusion matrix (rows=true, cols=pred).

    this function skips a class that is absent from val (an empty row), so that
    class does not drag the mean down.
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


def balanced_acc(cm: torch.Tensor) -> float:
    """the mean per-class recall over the classes present in val (balanced accuracy)."""
    recs = [int(cm[c, c]) / int(cm[c, :].sum()) for c in range(cm.size(0)) if cm[c, :].sum() > 0]
    return statistics.mean(recs) if recs else 0.0


def label_purity(
    neighbor_labels: torch.Tensor,
    y_val: torch.Tensor,
    y_train: torch.Tensor,
    n_classes: int,
) -> tuple[float, float, float, list[float]]:
    """the fraction of the k neighbors of each val point that share its label.

    accuracy asks whether the vote came out right. purity asks how lopsided the
    vote was. therefore purity separates two runs that accuracy scores as tied,
    which matters at this annotated-set size. see README "How to read the
    metrics" section 5.

    purity stays unweighted on purpose, even when --weighted is set. purity
    describes the neighborhood and not the vote rule, so purity stays comparable
    across the flags.

    returns (purity, normalized, baseline, per_class_purity).
    """
    matches = (neighbor_labels == y_val[:, None]).float()
    per_point = matches.mean(dim=1)
    purity = per_point.mean().item()

    # chance baseline: a random train neighbor matches a class-c val point with
    # probability p_train[c]. therefore the baseline is the class-frequency
    # weighted mean over val. the baseline moves from fold to fold, because the
    # grouped split changes the train class mix. compute the baseline for every
    # fold. do not assume 1/n.
    p_train = torch.bincount(y_train, minlength=n_classes).float() / y_train.numel()
    baseline = p_train[y_val].mean().item()
    normalized = (purity - baseline) / (1.0 - baseline) if baseline < 1.0 else 0.0

    per_class: list[float] = []
    for c in range(n_classes):
        mask = y_val == c
        per_class.append(per_point[mask].mean().item() if bool(mask.any()) else float("nan"))
    return purity, normalized, baseline, per_class


def batch_purity(X: torch.Tensor, groups: list[int], k: int) -> tuple[float, float, float]:
    """the fraction of the k neighbors of each point that come from its own zarr volume.

    a high value means that the local geometry encodes which experiment a crop
    came from. batch purity is more sensitive than the Wasserstein cloud check.
    two volumes can overlap almost perfectly in total, and stay locally separate
    in interleaved single-volume filaments.

    this metric pools every annotated embedding and excludes the point itself,
    on purpose. the folds group by zarr volume, so inside a fold the train
    neighbors of a val point never share its volume. a folded same-volume purity
    is therefore zero by construction, and it would report perfect integration
    for any batch effect, however bad. kBET and LISI also pool.

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

    # the chance baseline per point is (n_g - 1) / (n - 1). take the mean of the
    # per-point values. do not assume a uniform value, because the volumes
    # differ a lot in annotation count.
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

    # this module prepares df exactly as probe_behavior does, so the folds line
    # up exactly.
    df = pd.read_json(cfg.annotation_path, lines=True)
    df = df[df["label"] != "empty"]
    df = df.drop(columns=["timestamp"])
    df = df[df["label"] != "slug"]
    # warning: eval/probe_behavior.py hardcodes the same literal path. change
    # both files together, or the two evals no longer see identical folds.
    # mixin49 is a bad acquisition and is not part of the data of this project.
    # no run trains on mixin49. the line below drops its 31 annotations, so they
    # cannot leak into the --restrict unseen pool.
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

    # the per-k accumulators across the folds
    accs = {k: [] for k in args.k_values}
    macros = {k: [] for k in args.k_values}
    bals = {k: [] for k in args.k_values}
    purs = {k: [] for k in args.k_values}
    pur_norms = {k: [] for k in args.k_values}
    # the per-class purity for each fold. an entry is nan for a class that is
    # absent from the val split of that fold. the pooled report skips a nan
    # entry and does not count it as 0.
    pur_classes: dict[int, list[list[float]]] = {k: [] for k in args.k_values}
    cms = {k: torch.zeros(n_classes, n_classes, dtype=torch.long) for k in args.k_values}

    # warning: the groups must come from `rows`, which is in X order, and not
    # from df["dataset_idx"], which is in annotation-file order. the trap is the
    # same as in probe_behavior.py: df grouped the wrong rows once, and the
    # leave-experiment-out split became near-random.
    groups = [r["dataset_idx"] for r in rows]

    for fold_idx, (X_train, y_train, X_val, y_val, held_out) in enumerate(
        stratified_group_kfold_split(X, y, groups, args.seed, args.n_splits)
    ):
        print(f"[knn] fold {fold_idx+1}/{args.n_splits}  (grouped by zarr volume, stratified by label)")
        print(f"[knn] train n={X_train.size(0)}  val n={X_val.size(0)}  held out: {held_out}")
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

    # --- pooled batch purity: does the local geometry organize by experiment? ---
    # this block is not folded, and you must read it. a behavior gain that
    # arrives together with a batch-purity gain probably means only that the
    # encoder tells the experiments apart better. the docstring of batch_purity
    # explains why the metric must be pooled.
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
        # take the mean purity of each class over the folds where the class
        # appeared in val. a class that is absent from every fold prints "n/a"
        # and not 0.
        seen = [fold[c] for fold in pur_classes[best_k] if fold[c] == fold[c]]
        purity_str = f"purity={statistics.mean(seen)*100:6.2f}%" if seen else "purity=   n/a"
        print(f"    {name:30s}  F1={f1:.3f}  {purity_str}")
    print()
    print(f"  interpretation: {interpret(best_acc, n_classes)}")
    print("#" * 70)


if __name__ == "__main__":
    main()
