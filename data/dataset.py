"""positive-pair sampler over one or more ome-multiscale zarrs.

a positive pair is two zyx crops at the same anchor but different timepoints.
the time gap is a fixed number of frames derived from the desired wall-clock
gap and the experiment's interval_minutes (looked up via data/metadata.py).
the z range and t1 range are fractions of each dataset by default.

two dataset shapes are exposed:
- `ZarrCellDataset` samples pairs from a single zarr.
- `MultiZarrDataset` holds N `ZarrCellDataset` children and, per sample, picks
  one child weighted by its usable t1 range, then yields a pair from it. each
  batch therefore mixes pairs across experiments; pairs themselves never cross
  experiment boundaries.

the dataset is iterable because the crop space is too large to index usefully.

zarr handles are reopened in each worker so dask handles stay process-local.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any, Iterator

import dask.array as da
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from configs.config import Config
from data.augmentations import build_augmentations
from data.metadata import get_interval_minutes


class ZarrCellDataset(IterableDataset):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        if cfg.zarr_path is None:
            raise ValueError("cfg.zarr_path must be set before constructing ZarrCellDataset")
        self.cfg = cfg

        # open once here to read shape and resolve ranges.
        self._open_zarr()
        self._resolve_ranges()
        self._log_open()

        # augmentations are stateless across workers.
        self._augment = build_augmentations(cfg)

    def _open_zarr(self) -> None:
        # cells are already low resolution, so always read the native pyramid level.
        self.arr = da.from_zarr(f"{self.cfg.zarr_path}/0/0")

        # shape varies across datasets.
        # expected layout is (t, c, z, y, x).
        if self.arr.ndim != 5:
            raise ValueError(
                f"Expected 5D zarr (T, C, Z, Y, X); got shape {self.arr.shape}"
            )
        self.T, self.C, self.Z, self.Y, self.X = self.arr.shape
        if self.T < 2:
            raise ValueError(f"need at least 2 timepoints; got T={self.T}")

    def _resolve_ranges(self) -> None:
        cfg = self.cfg

        self._check_fraction("z_frac_min", cfg.z_frac_min)
        self._check_fraction("z_frac_max", cfg.z_frac_max)
        self._check_fraction("t1_frac_min", cfg.t1_frac_min)
        self._check_fraction("t1_frac_max", cfg.t1_frac_max)
        self._check_order("z_frac", cfg.z_frac_min, cfg.z_frac_max)
        self._check_order("t1_frac", cfg.t1_frac_min, cfg.t1_frac_max)

        if cfg.z_thickness < 1:
            raise ValueError(f"z_thickness must be >= 1; got {cfg.z_thickness}")

        if cfg.target_gap_minutes <= 0:
            raise ValueError(
                f"target_gap_minutes must be > 0; got {cfg.target_gap_minutes}"
            )
        if cfg.temporal_direction not in ("forward", "backward"):
            raise ValueError(
                f"temporal_direction must be 'forward' or 'backward'; "
                f"got {cfg.temporal_direction!r}"
            )

        # interval_minutes comes from the spreadsheet, keyed by mixin id.
        self.interval_minutes = get_interval_minutes(
            cfg.zarr_path, cfg.training_data_index_path
        )
        if self.interval_minutes <= 0:
            raise ValueError(
                f"interval_minutes from spreadsheet must be > 0; "
                f"got {self.interval_minutes}"
            )

        # deterministic frame gap derived from the requested wall-clock gap.
        self.frame_gap = int(round(cfg.target_gap_minutes / self.interval_minutes))
        if self.frame_gap < 1:
            raise ValueError(
                f"target_gap_minutes={cfg.target_gap_minutes} with "
                f"interval_minutes={self.interval_minutes} rounds to "
                f"frame_gap={self.frame_gap}; need at least 1 frame of spacing"
            )
        if self.frame_gap >= self.T:
            raise ValueError(
                f"dataset has T={self.T} timepoints but frame_gap={self.frame_gap}; "
                f"no valid (t1, t2) pair exists"
            )

        # absolute z bounds are only for pinning a known dataset.
        if cfg.z_abs_min is not None and cfg.z_abs_max is not None:
            self.z_min = cfg.z_abs_min
            self.z_max = cfg.z_abs_max
        else:
            self.z_min = int(round(self.Z * cfg.z_frac_min))
            self.z_max = max(self.z_min + 1, int(round(self.Z * cfg.z_frac_max)))
        self.z_min = max(0, min(self.z_min, self.Z - 1))
        self.z_max = max(self.z_min + 1, min(self.z_max, self.Z))

        if self.z_max - self.z_min < cfg.z_thickness:
            raise ValueError(
                f"z range [{self.z_min}, {self.z_max}) has "
                f"{self.z_max - self.z_min} slices, but z_thickness={cfg.z_thickness}"
            )

        # absolute t bounds are only for pinning a known dataset.
        if cfg.t1_abs_min is not None and cfg.t1_abs_max is not None:
            self.t1_min = cfg.t1_abs_min
            self.t1_max = cfg.t1_abs_max
        else:
            self.t1_min = int(round(self.T * cfg.t1_frac_min))
            self.t1_max = max(self.t1_min + 1, int(round(self.T * cfg.t1_frac_max)))
        self.t1_min = max(0, min(self.t1_min, self.T - 1))
        self.t1_max = max(self.t1_min + 1, min(self.t1_max, self.T))

        # restrict t1 so that t2 = t1 +/- frame_gap is always in bounds.
        if cfg.temporal_direction == "forward":
            self.t1_max = min(self.t1_max, self.T - self.frame_gap)
        else:
            self.t1_min = max(self.t1_min, self.frame_gap)
        if self.t1_min >= self.t1_max:
            raise ValueError(
                f"t1 window collapsed after applying frame_gap={self.frame_gap} "
                f"in direction {cfg.temporal_direction!r}; "
                f"resulting range was [{self.t1_min}, {self.t1_max})"
            )

    @staticmethod
    def _check_fraction(name: str, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]; got {value}")

    @staticmethod
    def _check_order(name: str, low: float, high: float) -> None:
        if low > high:
            raise ValueError(f"{name}_min must be <= {name}_max; got {low} > {high}")

    def _log_open(self) -> None:
        print(
            f"[ZarrCellDataset] opened {self.cfg.zarr_path} "
            f"shape={tuple(self.arr.shape)} chunks={self.arr.chunks}"
        )
        print(
            f"[ZarrCellDataset] T={self.T}; interval_minutes={self.interval_minutes}; "
            f"target_gap_minutes={self.cfg.target_gap_minutes} -> "
            f"frame_gap={self.frame_gap} ({self.cfg.temporal_direction})"
        )
        print(
            f"[ZarrCellDataset] Z={self.Z}; z range resolved to "
            f"[{self.z_min}, {self.z_max}) slices, thickness={self.cfg.z_thickness}"
        )
        print(
            f"[ZarrCellDataset] t1 range resolved to "
            f"[{self.t1_min}, {self.t1_max}) frames"
        )

    def _reopen_zarrs(self) -> None:
        """uniform hook used by `worker_init_fn` across dataset types."""
        self._open_zarr()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        info = get_worker_info()
        if info is None:
            rng = random.Random(self.cfg.seed)
        else:
            rng = random.Random(self.cfg.seed + info.id)

        while True:
            yield self._sample_pair(rng)

    def _sample_pair(self, rng: random.Random) -> dict[str, Any]:
        cfg = self.cfg
        crop_h = crop_w = cfg.crop_size

        if crop_h > self.Y or crop_w > self.X:
            raise ValueError(
                f"crop_size={cfg.crop_size} exceeds frame size {self.Y}x{self.X}"
            )

        # sample one z stack and reuse it for both views.
        z_start_max = self.z_max - cfg.z_thickness
        z = rng.randrange(self.z_min, z_start_max + 1)
        t1 = rng.randrange(self.t1_min, self.t1_max)

        # fixed gap in the configured direction. t1 was bounded so no clipping is needed.
        sign = 1 if cfg.temporal_direction == "forward" else -1
        t2 = t1 + sign * self.frame_gap

        # sample one spatial crop and reuse it for both views.
        y_off = rng.randrange(0, self.Y - crop_h + 1)
        x_off = rng.randrange(0, self.X - crop_w + 1)

        view1 = self._load_crop(t1, z, y_off, x_off, crop_h, crop_w)
        view2 = self._load_crop(t2, z, y_off, x_off, crop_h, crop_w)

        view1 = self._augment(view1)
        view2 = self._augment(view2)

        return {
            "view1": view1,
            "view2": view2,
            "meta": {
                "t1": t1,
                "t2": t2,
                "z": z,
                "z_end": z + cfg.z_thickness,
                "x": x_off,
                "y": y_off,
            },
        }

    def _load_crop(
        self,
        t: int,
        z: int,
        y_off: int,
        x_off: int,
        crop_h: int,
        crop_w: int,
    ) -> torch.Tensor:
        cfg = self.cfg
        # index the smallest useful dask slice.
        sub = self.arr[
            t,
            cfg.channel,
            z : z + cfg.z_thickness,
            y_off : y_off + crop_h,
            x_off : x_off + crop_w,
        ]
        np_arr = np.asarray(sub.compute(), dtype=np.float32)

        # rescale to [0, 1] before augmentations.
        np_arr = np.clip(np_arr, 0.0, cfg.max_intensity) / cfg.max_intensity

        # shape is [1, z, h, w].
        return torch.from_numpy(np_arr).unsqueeze(0)


class MultiZarrDataset(IterableDataset):
    """mixes pairs from several zarrs in every batch.

    each child is a fully-initialized `ZarrCellDataset` built from a copy of
    the parent config with `zarr_path` set to one of `cfg.zarr_paths`. on
    every draw, this wrapper picks a child weighted by its usable t1 range
    and asks it for one pair. pairs never cross experiment boundaries; only
    the batch does.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        if not cfg.zarr_paths:
            raise ValueError(
                "cfg.zarr_paths must be a non-empty list to build a MultiZarrDataset"
            )
        self.cfg = cfg

        # build one fully-initialized child per zarr. each child reads its own
        # shape, looks up its own interval_minutes, and computes its own
        # frame_gap from the shared target_gap_minutes.
        self.children: list[ZarrCellDataset] = []
        for path in cfg.zarr_paths:
            child_cfg = replace(cfg, zarr_path=path, zarr_paths=None)
            self.children.append(ZarrCellDataset(child_cfg))

        # weight each child by the number of valid t1 anchors it exposes.
        # this proportionally favors longer experiments without starving
        # short ones to zero.
        sizes = [max(1, c.t1_max - c.t1_min) for c in self.children]
        total = float(sum(sizes))
        self.weights = [s / total for s in sizes]
        self._log_open()

    def _log_open(self) -> None:
        print(f"[MultiZarrDataset] {len(self.children)} children:")
        for i, (child, w) in enumerate(zip(self.children, self.weights)):
            print(
                f"  [{i}] {child.cfg.zarr_path} "
                f"T={child.T} frame_gap={child.frame_gap} weight={w:.3f}"
            )

    def _reopen_zarrs(self) -> None:
        """reopen every child's zarr handle inside this worker process."""
        for child in self.children:
            child._open_zarr()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        info = get_worker_info()
        if info is None:
            rng = random.Random(self.cfg.seed)
        else:
            rng = random.Random(self.cfg.seed + info.id)

        indices = list(range(len(self.children)))
        while True:
            idx = rng.choices(indices, weights=self.weights, k=1)[0]
            sample = self.children[idx]._sample_pair(rng)
            # tag the sample so downstream code can trace which experiment it came from.
            sample["meta"]["dataset_idx"] = idx
            yield sample


def worker_init_fn(worker_id: int) -> None:
    """re-open every zarr handle in each worker process."""
    info = get_worker_info()
    if info is None:
        return
    ds = info.dataset
    ds._reopen_zarrs()
    # seed numpy/torch per worker so augmentations stay reproducible-but-distinct.
    seed = ds.cfg.seed + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)
