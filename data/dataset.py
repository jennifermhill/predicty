"""WS-DINO multi-crop sampler over one or more ome-multiscale zarrs.

each sample yields a set of crops at one anchor:
- `n_global_crops` larger views, each paired across (t1, t2). the WS-DINO
  weak label is the frame gap between t1 and t2.
- `n_local_crops` smaller views at t1, each spatially nested inside one of
  the global crops.

the time gap is a fixed number of frames derived from the desired wall-clock
gap and the experiment's interval_minutes (looked up via data/metadata.py).
the z range and t1 range are fractions of each dataset by default. all crops
in a sample share the same z slab.

global crops are intensity-gated against the mean of the active z slab at t1
(not the full z window) so that the threshold and crop means live in the same
units. the slab mean is computed once per (t1, z) and cached per worker process.

two dataset shapes are exposed:
- `ZarrCellDataset` samples from a single zarr.
- `MultiZarrDataset` holds N `ZarrCellDataset` children and, per sample, picks
  one child weighted by its usable t1 range, then yields a sample from it.
  each batch therefore mixes crops across experiments; pairs themselves
  never cross experiment boundaries.

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

        # per-(t, z_slab) mean cache for intensity gating. keyed by the actual
        # z slab the crops will compare against, not the full z window, so the
        # threshold lives in the same units as the crop means.
        self._frame_mean_cache: dict[tuple[int, int], float] = {}

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

        if cfg.n_global_crops < 1:
            raise ValueError(f"n_global_crops must be >= 1; got {cfg.n_global_crops}")
        if cfg.n_local_crops < 0:
            raise ValueError(f"n_local_crops must be >= 0; got {cfg.n_local_crops}")
        if cfg.global_crop_size < 1:
            raise ValueError(
                f"global_crop_size must be >= 1; got {cfg.global_crop_size}"
            )
        if cfg.local_crop_size < 1:
            raise ValueError(
                f"local_crop_size must be >= 1; got {cfg.local_crop_size}"
            )
        if cfg.local_crop_size > cfg.global_crop_size:
            raise ValueError(
                f"local_crop_size ({cfg.local_crop_size}) must be <= "
                f"global_crop_size ({cfg.global_crop_size}) so locals can nest"
            )
        if cfg.global_crop_size > self.Y or cfg.global_crop_size > self.X:
            raise ValueError(
                f"global_crop_size={cfg.global_crop_size} exceeds frame size "
                f"{self.Y}x{self.X}"
            )
        if not 0.0 <= cfg.intensity_min_frac:
            raise ValueError(
                f"intensity_min_frac must be >= 0; got {cfg.intensity_min_frac}"
            )

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

    def _sample_pair(
        self, rng: random.Random, dataset_idx: int = 0
    ) -> dict[str, Any]:
        cfg = self.cfg
        g_size = cfg.global_crop_size
        l_size = cfg.local_crop_size

        # one z slab is reused across all crops in this sample.
        z_start_max = self.z_max - cfg.z_thickness
        z = rng.randrange(self.z_min, z_start_max + 1)
        t1 = rng.randrange(self.t1_min, self.t1_max)
        sign = 1 if cfg.temporal_direction == "forward" else -1
        t2 = t1 + sign * self.frame_gap

        # gate global crops against the per-(t1, z_slab) baseline so the
        # threshold and the crop means are computed over the same z range.
        threshold = cfg.intensity_min_frac * self._get_frame_mean(t1, z)

        global_offsets: list[tuple[int, int]] = []
        globals_t1: list[torch.Tensor] = []
        globals_t2: list[torch.Tensor] = []
        for _ in range(cfg.n_global_crops):
            # resample (y, x) until the t1 crop clears the intensity gate.
            while True:
                y_off = rng.randrange(0, self.Y - g_size + 1)
                x_off = rng.randrange(0, self.X - g_size + 1)
                raw_t1 = self._load_crop_raw(t1, z, y_off, x_off, g_size, g_size)
                if float(raw_t1.mean()) >= threshold:
                    break
            raw_t2 = self._load_crop_raw(t2, z, y_off, x_off, g_size, g_size)
            global_offsets.append((y_off, x_off))
            globals_t1.append(self._augment(self._to_tensor(raw_t1)))
            globals_t2.append(self._augment(self._to_tensor(raw_t2)))

        # locals are at t1 only, nested inside a randomly chosen global.
        local_parents: list[int] = []
        local_offsets: list[tuple[int, int]] = []
        locals_: list[torch.Tensor] = []
        for _ in range(cfg.n_local_crops):
            parent = rng.randrange(cfg.n_global_crops)
            gy, gx = global_offsets[parent]
            ly = rng.randrange(gy, gy + g_size - l_size + 1)
            lx = rng.randrange(gx, gx + g_size - l_size + 1)
            raw_l = self._load_crop_raw(t1, z, ly, lx, l_size, l_size)
            local_parents.append(parent)
            local_offsets.append((ly, lx))
            locals_.append(self._augment(self._to_tensor(raw_l)))

        sample: dict[str, Any] = {
            "globals_t1": torch.stack(globals_t1),
            "globals_t2": torch.stack(globals_t2),
            # WS-DINO weak label: (dataset_idx, frame_gap). dataset_idx is 0 for
            # a single-zarr dataset; MultiZarrDataset passes the child index it
            # drew. frame_gap is the per-dataset integer frame spacing.
            "weak_label": torch.tensor(
                [dataset_idx, self.frame_gap], dtype=torch.long
            ),
            "meta": {
                "t1": t1,
                "t2": t2,
                "z": z,
                "z_end": z + cfg.z_thickness,
                "dataset_idx": dataset_idx,
                "frame_gap": self.frame_gap,
                "global_y": torch.tensor([y for y, _ in global_offsets]),
                "global_x": torch.tensor([x for _, x in global_offsets]),
            },
        }
        if locals_:
            sample["locals"] = torch.stack(locals_)
            sample["meta"]["local_parent"] = torch.tensor(local_parents)
            sample["meta"]["local_y"] = torch.tensor([y for y, _ in local_offsets])
            sample["meta"]["local_x"] = torch.tensor([x for _, x in local_offsets])
        return sample

    def _get_frame_mean(self, t: int, z: int) -> float:
        """mean intensity over the active z slab [z, z+z_thickness) at frame t.

        cached per worker, keyed by (t, z) so identical anchors reuse the read.
        averaging over the slab that crops are drawn from (rather than the full
        z window) keeps the gate threshold in the same units as crop means.
        """
        key = (t, z)
        cached = self._frame_mean_cache.get(key)
        if cached is not None:
            return cached
        frame = np.asarray(
            self.arr[
                t,
                self.cfg.channel,
                z : z + self.cfg.z_thickness,
                :,
                :,
            ].compute(),
            dtype=np.float32,
        )
        mean = float(frame.mean())
        self._frame_mean_cache[key] = mean
        return mean

    def _load_crop_raw(
        self,
        t: int,
        z: int,
        y_off: int,
        x_off: int,
        crop_h: int,
        crop_w: int,
    ) -> np.ndarray:
        # raw float32 zyx slab. left unnormalized so the intensity gate can
        # compare crop and frame means in the same units.
        sub = self.arr[
            t,
            self.cfg.channel,
            z : z + self.cfg.z_thickness,
            y_off : y_off + crop_h,
            x_off : x_off + crop_w,
        ]
        return np.asarray(sub.compute(), dtype=np.float32)

    def _to_tensor(self, np_arr: np.ndarray) -> torch.Tensor:
        # rescale to [0, 1] and add a channel dim -> [1, z, h, w].
        max_i = self.cfg.max_intensity
        np_arr = np.clip(np_arr, 0.0, max_i) / max_i
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
            yield self.children[idx]._sample_pair(rng, dataset_idx=idx)


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
