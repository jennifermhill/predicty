from __future__ import annotations

"""Save/resume helpers for the interactive crop-annotation notebook.

Two append-only logs live under `out_dir`, sharing one pixel store:

    <out_dir>/index.jsonl      labeled crops, one JSON record per line
    <out_dir>/skipped.jsonl    crops the annotator skipped (to revisit later)
    <out_dir>/crops/<uid>.npy  the full t1 global crop, shape (1, Z, H, W)
    <out_dir>/mips/<uid>.png   single-panel max-intensity-projection preview

`uid` is derived deterministically from the crop's identity (experiment +
temporal + spatial anchor) so re-running the notebook never relabels or
re-shows a crop already labeled or skipped. Skips persist their pixels so they
can be reviewed in bulk later (e.g. they turn out to be a new class) without
relying on the random sampler reproducing them. "Pending" skips are those in
`skipped.jsonl` not yet present in `index.jsonl`. See `notebooks/annotate.ipynb`.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# provenance fields copied verbatim from a sample's meta into every record.
_PROVENANCE_KEYS = (
    "uid",
    "t1",
    "t2",
    "z",
    "z_end",
    "dataset_idx",
    "frame_gap",
    "global_y",
    "global_x",
    "global_crop_size",
    "zarr_path",
    "crop_path",
    "mip_path",
)


def _crops_dir(out_dir: str | Path) -> Path:
    return Path(out_dir) / "crops"


def _mips_dir(out_dir: str | Path) -> Path:
    return Path(out_dir) / "mips"


def index_path(out_dir: str | Path) -> Path:
    return Path(out_dir) / "index.jsonl"


def skipped_path(out_dir: str | Path) -> Path:
    return Path(out_dir) / "skipped.jsonl"


def make_uid(meta: dict[str, Any], crop_size: int) -> str:
    """Deterministic id from the crop's anchor.

    Two draws with the same experiment, timepoints, z slab, (y, x) offset and
    crop size produce the same uid, which is how resume/dedupe works. `t2` is a
    fixed function of `t1` per dataset, so it never splits one t1 crop into two
    uids; it is kept in the id only for readability.
    """
    return (
        f"ds{int(meta['dataset_idx'])}"
        f"_t{int(meta['t1'])}-{int(meta['t2'])}"
        f"_z{int(meta['z'])}"
        f"_y{int(meta['global_y'])}_x{int(meta['global_x'])}"
        f"_s{int(crop_size)}"
    )


def _read_uids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.add(json.loads(line)["uid"])
            except (json.JSONDecodeError, KeyError):
                continue  # tolerate a partially written trailing line
    return out


def _read_records(path: Path) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    if not path.exists():
        return recs
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


def load_seen(out_dir: str | Path) -> set[str]:
    """uids already labeled (present in index.jsonl)."""
    return _read_uids(index_path(out_dir))


def load_skipped(out_dir: str | Path) -> set[str]:
    """uids already skipped (present in skipped.jsonl)."""
    return _read_uids(skipped_path(out_dir))


def load_suppressed(out_dir: str | Path) -> set[str]:
    """uids that live annotation should not show again (labeled OR skipped)."""
    return load_seen(out_dir) | load_skipped(out_dir)


def load_pending_skips(out_dir: str | Path) -> list[dict[str, Any]]:
    """Skipped records not yet labeled, in skip order, for review mode.

    Deduplicated by uid; a skip that has since been labeled drops out.
    """
    labeled = load_seen(out_dir)
    seen_uids: set[str] = set()
    pending: list[dict[str, Any]] = []
    for rec in _read_records(skipped_path(out_dir)):
        uid = rec.get("uid")
        if uid is None or uid in labeled or uid in seen_uids:
            continue
        seen_uids.add(uid)
        pending.append(rec)
    return pending


def load_counts(out_dir: str | Path) -> dict[int, dict[str, int]]:
    """Per dataset_idx, a count of each label in index.jsonl.

    Returns e.g. {0: {"class_a": 12, "unsure": 1}, 3: {"class_b": 7}}.
    """
    counts: dict[int, dict[str, int]] = {}
    for rec in _read_records(index_path(out_dir)):
        try:
            idx = int(rec["dataset_idx"])
        except (KeyError, TypeError, ValueError):
            continue
        label = rec.get("label")
        if label is None:
            continue
        counts.setdefault(idx, {})
        counts[idx][label] = counts[idx].get(label, 0) + 1
    return counts


def pending_skip_counts(out_dir: str | Path) -> dict[int, int]:
    """Per dataset_idx count of skips not yet labeled (for the counts table)."""
    counts: dict[int, int] = {}
    for rec in load_pending_skips(out_dir):
        try:
            idx = int(rec["dataset_idx"])
        except (KeyError, TypeError, ValueError):
            continue
        counts[idx] = counts.get(idx, 0) + 1
    return counts


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()


def _save_crop_files(out_dir: str | Path, uid: str, sample: dict[str, Any]) -> tuple[int, str, str]:
    """Write the t1 crop .npy and MIP .png for `uid` if not already present.

    Returns (crop_size, crop_path, mip_path).
    """
    crop = sample["globals"][0]  # (1, Z, H, W); index 0 of globals is t1
    crop_np = crop.detach().cpu().numpy() if hasattr(crop, "detach") else np.asarray(crop)
    crop_np = np.asarray(crop_np, dtype=np.float32)

    _crops_dir(out_dir).mkdir(parents=True, exist_ok=True)
    _mips_dir(out_dir).mkdir(parents=True, exist_ok=True)

    crop_path = _crops_dir(out_dir) / f"{uid}.npy"
    if not crop_path.exists():
        np.save(crop_path, crop_np)
    mip_path = _mips_dir(out_dir) / f"{uid}.png"
    if not mip_path.exists():
        _save_mip_png(crop_np[0], mip_path)  # (Z, H, W)
    return int(crop_np.shape[-1]), str(crop_path), str(mip_path)


def _base_record(
    meta: dict[str, Any],
    uid: str,
    crop_size: int,
    zarr_path: str | None,
    crop_path: str,
    mip_path: str,
) -> dict[str, Any]:
    return {
        "uid": uid,
        "t1": int(meta["t1"]),
        "t2": int(meta["t2"]),
        "z": int(meta["z"]),
        "z_end": int(meta["z_end"]),
        "dataset_idx": int(meta["dataset_idx"]),
        "frame_gap": int(meta["frame_gap"]),
        "global_y": int(meta["global_y"]),
        "global_x": int(meta["global_x"]),
        "global_crop_size": int(crop_size),
        "zarr_path": zarr_path,
        "crop_path": crop_path,
        "mip_path": mip_path,
    }


def save_annotation(
    out_dir: str | Path,
    sample: dict[str, Any],
    label: str,
    zarr_path: str | None = None,
) -> dict[str, Any]:
    """Persist one t1 crop + its class label; append a record to index.jsonl."""
    meta = sample["meta"]
    uid = make_uid(meta, int(sample["globals"].shape[-1]))
    crop_size, crop_path, mip_path = _save_crop_files(out_dir, uid, sample)
    rec = _base_record(meta, uid, crop_size, zarr_path, crop_path, mip_path)
    rec["label"] = label
    rec["timestamp"] = _now()
    _append(index_path(out_dir), rec)
    return rec


def save_skip(
    out_dir: str | Path,
    sample: dict[str, Any],
    zarr_path: str | None = None,
) -> dict[str, Any]:
    """Persist a skipped t1 crop (pixels too) and append to skipped.jsonl.

    Saving the pixels lets review mode revisit skips in bulk without the
    sampler having to reproduce them.
    """
    meta = sample["meta"]
    uid = make_uid(meta, int(sample["globals"].shape[-1]))
    crop_size, crop_path, mip_path = _save_crop_files(out_dir, uid, sample)
    rec = _base_record(meta, uid, crop_size, zarr_path, crop_path, mip_path)
    rec["label"] = None
    rec["timestamp"] = _now()
    _append(skipped_path(out_dir), rec)
    return rec


def relabel_skip(
    out_dir: str | Path,
    skip_record: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    """Promote a previously skipped crop to a labeled one in index.jsonl.

    Reuses the crop/MIP already on disk (no sample needed); the crop drops out
    of `load_pending_skips` once its uid appears in index.jsonl.
    """
    rec = {k: skip_record.get(k) for k in _PROVENANCE_KEYS}
    rec["label"] = label
    rec["timestamp"] = _now()
    _append(index_path(out_dir), rec)
    return rec


def _save_mip_png(volume: np.ndarray, path: str | Path) -> None:
    """Write a grayscale max-intensity projection of a (Z, H, W) volume.

    Uses the object-oriented ``Figure`` API (not ``pyplot``) and never calls
    ``matplotlib.use(...)``, so saving a preview does not touch the notebook's
    active backend — otherwise the first label would switch it to Agg and break
    inline display.
    """
    from matplotlib.figure import Figure

    mip = volume.max(axis=0)
    lo, hi = np.percentile(mip, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(mip.min()), float(mip.max())
    if hi <= lo:
        hi = lo + 1e-6

    fig = Figure(figsize=(4, 4))
    ax = fig.subplots()
    ax.imshow(mip, cmap="gray", vmin=lo, vmax=hi)
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", pad_inches=0, dpi=100)
