"""WS-DINO multi-crop sampler over one or more ome-multiscale zarrs.

each sample gives a set of crops at one anchor:
- exactly 2 globals: one at frame t1 and one at frame t2. both globals use the
  same (y, x) offset and the same z slab, which is `z_thickness` deep. only the
  timepoint differs. the WS-DINO weak label is `(dataset_idx, frame_gap)`.
- `n_local_crops` smaller views. the code splits the locals evenly between the
  two globals, which are the spatial parents. each local takes a contiguous
  `local_z_thickness` sub-slab inside the global z window. each local also
  nests in (y, x) inside the spatial window of its parent global. the timepoint
  of a local comes from {t1, t2} when `local_time_endpoints` is true, which is
  the default. otherwise the timepoint comes uniformly from [t1, t2].

the time gap is a fixed number of frames. the code derives the gap from the
wall-clock gap that you want and from the interval_minutes of the experiment.
data/metadata.py looks up interval_minutes. by default, the z range and the t1
range are fractions of each dataset.

an intensity gate tests each global against the mean of the active z slab at
t1, so the threshold and the crop means share units. the code samples only one
(y, x) per pair, so the gate runs once per sample. each worker caches the slab
means. see README "How to read the metrics" section 3 for the
reference-population rules.

this module gives two dataset shapes:
- `ZarrCellDataset` samples from a single zarr.
- `MultiZarrDataset` holds N `ZarrCellDataset` children. for each sample,
  `MultiZarrDataset` picks a child from a batch-aware schedule. by default,
  each batch gives every child equal representation. you can disable the
  default and sample the children with weights from the usable t1 range. a
  pair never crosses an experiment boundary.

the dataset is iterable because the crop space is too large to index usefully.

each worker reopens the zarr handles, so the dask handles stay process-local.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any, Iterator

import dask.array as da
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

import pandas as pd

from configs.config import Config
from data.augmentations import build_augmentations
from data.balancing import balanced_batch_indices
from data.metadata import get_interval_minutes, get_z_limit
from data.spatial_regions import (
    region_bounds_for_top_left,
    region_index_for_top_left,
    select_nonadjacent_regions,
)


class ZarrCellDataset(IterableDataset):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        if cfg.zarr_path is None:
            raise ValueError("cfg.zarr_path must be set before constructing ZarrCellDataset")
        self.cfg = cfg

        # the allowed top-left positions under cfg.max_train_footprint_overlap.
        # see README "How to read the metrics" section 2. the code builds the
        # set late, so each dataloader worker computes its own copy and does not
        # inherit a large array across a fork. a late build also lets the code
        # invalidate the set when MultiZarrDataset rebuilds the region set.
        self._allowed_offsets: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._allowed_offsets_built = False
        # the pixel window that the allowed offsets reach. the intensity gate
        # uses the window as its reference when the allowed set is restricted.
        # the value is None in training.
        self._allowed_bbox: tuple[int, int, int, int] | None = None

        # open the zarr once here to read the shape and to resolve the ranges.
        self._open_zarr()
        # the code makes the cache before it resolves the ranges, because
        # content-aware holdout selection scores the crops against cached
        # frame means.
        self._frame_mean_cache: dict[tuple[int, int], float] = {}
        self._resolve_ranges()
        self._log_open()

        if cfg.overrepresent_classes:
            self._load_annotations(self.cfg.zarr_path)

        # the augmentations hold no state across the workers.
        self._augment = build_augmentations(cfg)

    def _open_zarr(self) -> None:
        # the cells are already low resolution, so always read the native
        # pyramid level.
        self.arr = da.from_zarr(f"{self.cfg.zarr_path}/0/0")

        # the expected layout is (t, c, z, y, x). the shape varies across the
        # datasets. see README "Data" / "Frame sizes vary by up to 4x in area".
        if self.arr.ndim != 5:
            raise ValueError(
                f"Expected 5D zarr (T, C, Z, Y, X); got shape {self.arr.shape}"
            )
        self.T, self.C, self.Z, self.Y, self.X = self.arr.shape
        if self.T < 2:
            raise ValueError(f"need at least 2 timepoints; got T={self.T}")
    
    def _load_annotations(self, zarr_path: str) -> None:
        cfg = self.cfg
        if cfg.annotation_path is None:
            raise ValueError("cfg.annotation_path must be set to load annotations.")
        df = pd.read_json(cfg.annotation_path, lines=True)

        df = df[df['label'] != 'empty']
        if df.empty:
            raise ValueError(
                f"No annotations found for usable zarrs in {cfg.annotation_path}"
            )
        df = df[df['zarr_path'].isin([zarr_path])]

        excluded = {"single_cells", "clumps"}
        all_classes = set(df["label"].unique())
        self.minority_classes = list(all_classes - excluded)
        df = df[df['label'].isin(self.minority_classes)]
        if df.empty:
            print(f"No minority class annotations found for zarrs in {zarr_path}. Proceeding without annotations.")
        else:
            df = df.drop(columns = ['timestamp'])
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass
        self.by_label = {}
        for label in df["label"].unique():
            self.by_label[label] = df[df["label"] == label]
        
    def _resolve_ranges(self) -> None:
        cfg = self.cfg

        self._check_fraction("z_frac_min", cfg.z_frac_min)
        self._check_fraction("t1_frac_min", cfg.t1_frac_min)
        self._check_fraction("t1_frac_max", cfg.t1_frac_max)
        self._check_fraction("representation_alpha", cfg.rep_alpha)
        self._check_order("t1_frac", cfg.t1_frac_min, cfg.t1_frac_max)
        self._check_spatial_region_config()

        if cfg.z_thickness < 1:
            raise ValueError(f"z_thickness must be >= 1; got {cfg.z_thickness}")
        if cfg.local_z_thickness < 1:
            raise ValueError(
                f"local_z_thickness must be >= 1; got {cfg.local_z_thickness}"
            )
        if cfg.local_z_thickness > cfg.z_thickness:
            raise ValueError(
                f"local_z_thickness ({cfg.local_z_thickness}) must be <= "
                f"z_thickness ({cfg.z_thickness}) so locals nest in the global slab"
            )

        if cfg.n_local_crops < 0:
            raise ValueError(f"n_local_crops must be >= 0; got {cfg.n_local_crops}")
        if cfg.n_local_crops % 2 != 0:
            raise ValueError(
                f"n_local_crops must be even (split evenly between the two "
                f"global parents); got {cfg.n_local_crops}"
            )
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
        if cfg.intensity_max_attempts < 1:
            raise ValueError(
                "intensity_max_attempts must be >= 1; "
                f"got {cfg.intensity_max_attempts}"
            )
        if not 0 <= cfg.channel < self.C:
            raise ValueError(
                f"channel must be in [0, {self.C}); got {cfg.channel}"
            )
        if not np.isfinite(cfg.max_intensity) or cfg.max_intensity <= 0:
            raise ValueError(
                f"max_intensity must be finite and > 0; got {cfg.max_intensity}"
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

        # interval_minutes comes from the spreadsheet. the mixin id is the key.
        self.interval_minutes = get_interval_minutes(
            cfg.zarr_path, cfg.training_data_index_path
        )
        if self.interval_minutes <= 0:
            raise ValueError(
                f"interval_minutes from spreadsheet must be > 0; "
                f"got {self.interval_minutes}"
            )

        # the code derives a deterministic frame gap from the requested
        # wall-clock gap.
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

        # the per-experiment spreadsheet cell gives the z upper bound. a blank
        # z_limit means that the experiment is not-for-training. in single-zarr
        # construction a blank z_limit is a hard error. MultiZarrDataset filters
        # these experiments out before they reach this path.
        z_limit = get_z_limit(cfg.zarr_path, cfg.training_data_index_path)
        if z_limit is None:
            raise ValueError(
                f"{cfg.zarr_path} has a blank z_limit in the training_data_index "
                f"spreadsheet; this experiment is marked not-for-training"
            )

        # z_min: the absolute override wins, otherwise z_frac_min gives z_min.
        if cfg.z_abs_min is not None:
            self.z_min = cfg.z_abs_min
        else:
            self.z_min = int(round(self.Z * cfg.z_frac_min))
        # z_max: the spreadsheet z_limit is the ceiling. z_abs_max can only
        # narrow the range.
        if cfg.z_abs_max is not None:
            self.z_max = min(cfg.z_abs_max, z_limit)
        else:
            self.z_max = z_limit
        self.z_min = max(0, min(self.z_min, self.Z - 1))
        self.z_max = max(self.z_min + 1, min(self.z_max, self.Z))

        if self.z_max - self.z_min < cfg.z_thickness:
            raise ValueError(
                f"z range [{self.z_min}, {self.z_max}) has "
                f"{self.z_max - self.z_min} slices, but z_thickness={cfg.z_thickness}"
            )

        if not 0.0 <= cfg.z_edge_bias <= 1.0:
            raise ValueError(
                f"z_edge_bias must be in [0, 1]; got {cfg.z_edge_bias}"
            )
        if not 0.0 <= cfg.z_edge_band_frac <= 1.0:
            raise ValueError(
                f"z_edge_band_frac must be in [0, 1]; got {cfg.z_edge_band_frac}"
            )

        # use the absolute t bounds only when you pin a known dataset.
        if cfg.t1_abs_min is not None and cfg.t1_abs_max is not None:
            self.t1_min = cfg.t1_abs_min
            self.t1_max = cfg.t1_abs_max
        else:
            self.t1_min = int(round(self.T * cfg.t1_frac_min))
            self.t1_max = max(self.t1_min + 1, int(round(self.T * cfg.t1_frac_max)))
        self.t1_min = max(0, min(self.t1_min, self.T - 1))
        self.t1_max = max(self.t1_min + 1, min(self.t1_max, self.T))

        # restrict t1 so that t2 = t1 +/- frame_gap always stays in bounds.
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

        self._spatial_holdout_regions = self._resolve_spatial_holdout_regions()
        self._spatial_regions = self._build_spatial_regions()

    @staticmethod
    def _check_fraction(name: str, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]; got {value}")

    @staticmethod
    def _check_order(name: str, low: float, high: float) -> None:
        if low > high:
            raise ValueError(f"{name}_min must be <= {name}_max; got {low} > {high}")

    def _check_spatial_region_config(self) -> None:
        cfg = self.cfg
        if cfg.spatial_region_grid_y < 1 or cfg.spatial_region_grid_x < 1:
            raise ValueError(
                "spatial_region_grid_y and spatial_region_grid_x must be >= 1; "
                f"got {cfg.spatial_region_grid_y}x{cfg.spatial_region_grid_x}"
            )
        if cfg.spatial_region_mode not in ("all", "exclude_holdout", "holdout"):
            raise ValueError(
                "spatial_region_mode must be one of "
                "'all', 'exclude_holdout', or 'holdout'; "
                f"got {cfg.spatial_region_mode!r}"
            )
        if cfg.spatial_holdout_selection not in ("manual", "content"):
            raise ValueError(
                "spatial_holdout_selection must be 'manual' or 'content'; "
                f"got {cfg.spatial_holdout_selection!r}"
            )
        if cfg.spatial_holdout_region_count < 1:
            raise ValueError(
                "spatial_holdout_region_count must be >= 1; "
                f"got {cfg.spatial_holdout_region_count}"
            )
        if cfg.spatial_holdout_nonadjacent_gap < 0:
            raise ValueError(
                "spatial_holdout_nonadjacent_gap must be >= 0; "
                f"got {cfg.spatial_holdout_nonadjacent_gap}"
            )
        if cfg.spatial_holdout_content_samples_per_region < 1:
            raise ValueError(
                "spatial_holdout_content_samples_per_region must be >= 1; "
                f"got {cfg.spatial_holdout_content_samples_per_region}"
            )
        n_regions = cfg.spatial_region_grid_y * cfg.spatial_region_grid_x
        resolved_regions = self._resolved_spatial_holdout_regions()
        for idx in resolved_regions:
            if not 0 <= idx < n_regions:
                raise ValueError(
                    f"resolved spatial holdout region must be in [0, {n_regions}); got {idx}"
                )
        manual_regions = self._manual_spatial_holdout_regions()
        for idx in manual_regions:
            if not 0 <= idx < n_regions:
                raise ValueError(
                    f"spatial holdout region must be in [0, {n_regions}); got {idx}"
                )
        if cfg.spatial_region_mode != "all" and cfg.spatial_holdout_selection == "manual":
            if not manual_regions:
                raise ValueError(
                    f"spatial_region_mode={cfg.spatial_region_mode!r} requires "
                    "spatial_holdout_regions or spatial_holdout_region"
                )

    def _manual_spatial_holdout_regions(self) -> list[int]:
        cfg = self.cfg
        regions: list[int] = []
        if cfg.spatial_holdout_regions is not None:
            regions.extend(int(idx) for idx in cfg.spatial_holdout_regions)
        if cfg.spatial_holdout_region is not None:
            regions.append(int(cfg.spatial_holdout_region))
        return sorted(set(regions))

    def _resolved_spatial_holdout_regions(self) -> list[int]:
        if self.cfg.spatial_holdout_resolved_regions is None:
            return []
        return sorted(set(int(idx) for idx in self.cfg.spatial_holdout_resolved_regions))

    def _resolve_spatial_holdout_regions(self) -> list[int]:
        cfg = self.cfg
        if cfg.spatial_region_mode == "all":
            return []
        resolved = self._resolved_spatial_holdout_regions()
        if resolved:
            return resolved
        if cfg.spatial_holdout_selection == "manual":
            return self._manual_spatial_holdout_regions()
        return self._select_content_holdout_regions()

    def _score_spatial_regions_by_content(self) -> dict[int, float]:
        cfg = self.cfg
        rng = random.Random(cfg.spatial_holdout_seed + 7919)
        n_regions = cfg.spatial_region_grid_y * cfg.spatial_region_grid_x
        scores: dict[int, float] = {}
        for idx in range(n_regions):
            y0, y1, x0, x1 = region_bounds_for_top_left(
                self.Y,
                self.X,
                cfg.global_crop_size,
                cfg.spatial_region_grid_y,
                cfg.spatial_region_grid_x,
                idx,
            )
            ratios: list[float] = []
            for _ in range(cfg.spatial_holdout_content_samples_per_region):
                z = rng.randrange(self.z_min, self.z_max - cfg.z_thickness + 1)
                t = rng.randrange(self.t1_min, self.t1_max)
                y_off = rng.randrange(y0, y1)
                x_off = rng.randrange(x0, x1)
                raw = self._load_crop_raw(
                    t,
                    z,
                    y_off,
                    x_off,
                    cfg.global_crop_size,
                    cfg.global_crop_size,
                )
                frame_mean = max(self._get_frame_mean(t, z), 1e-12)
                ratios.append(float(raw.mean()) / frame_mean)
            scores[idx] = float(np.median(ratios))
        return scores

    def _select_content_holdout_regions(self) -> list[int]:
        cfg = self.cfg
        scores = self._score_spatial_regions_by_content()
        selected = select_nonadjacent_regions(
            scores,
            cfg.spatial_holdout_region_count,
            cfg.spatial_region_grid_x,
            cfg.spatial_holdout_nonadjacent_gap,
            cfg.spatial_holdout_min_content,
        )
        score_text = ", ".join(f"{idx}:{scores[idx]:.3f}" for idx in sorted(scores))
        print(
            f"[ZarrCellDataset] content scores by spatial region: {score_text}; "
            f"selected holdout regions={selected}"
        )
        return selected

    def _build_spatial_regions(self) -> list[tuple[int, int, int, int, int]]:
        cfg = self.cfg
        n_regions = cfg.spatial_region_grid_y * cfg.spatial_region_grid_x
        holdout_regions = set(self._spatial_holdout_regions)
        regions: list[tuple[int, int, int, int, int]] = []
        for idx in range(n_regions):
            y0, y1, x0, x1 = region_bounds_for_top_left(
                self.Y,
                self.X,
                cfg.global_crop_size,
                cfg.spatial_region_grid_y,
                cfg.spatial_region_grid_x,
                idx,
            )
            if cfg.spatial_region_mode == "holdout" and idx not in holdout_regions:
                continue
            if cfg.spatial_region_mode == "exclude_holdout" and idx in holdout_regions:
                continue
            regions.append((idx, y0, y1, x0, x1))
        if not regions:
            raise ValueError(
                f"spatial_region_mode={cfg.spatial_region_mode!r} left no "
                "sampleable regions"
            )
        return regions

    def _build_allowed_offsets(self) -> None:
        """precompute the top-left positions that pass max_train_footprint_overlap.

        see README "How to read the metrics" section 2. the method is exact and
        uses no rejection loop. the method scores the whole candidate grid in
        one vectorised pass. the reference is a summed-area table of the pixel
        footprint of the training regions. therefore the sampler is O(1)
        afterwards and needs no retry loop.
        """
        cfg = self.cfg
        self._allowed_offsets_built = True
        self._allowed_offsets = None
        self._allowed_bbox = None
        thr = cfg.max_train_footprint_overlap
        # the field is meaningful only when the code samples the held-out
        # regions. training leaves the field None and never reaches this code.
        if thr is None or cfg.spatial_region_mode != "holdout":
            return

        crop = cfg.global_crop_size
        gy, gx = cfg.spatial_region_grid_y, cfg.spatial_region_grid_x
        holdout = set(self._spatial_holdout_regions)
        train = np.zeros((self.Y, self.X), dtype=bool)
        for idx in range(gy * gx):
            if idx in holdout:
                continue
            y0, y1, x0, x1 = region_bounds_for_top_left(
                self.Y, self.X, crop, gy, gx, idx
            )
            # the bands are top-left ranges. a crop at y1-1 reaches y1-1+crop.
            train[y0 : y1 - 1 + crop, x0 : x1 - 1 + crop] = True
        sat = np.pad(train.cumsum(0).cumsum(1), ((1, 0), (1, 0)))

        ys_keep, xs_keep, regs_keep = [], [], []
        for idx, y0, y1, x0, x1 in self._spatial_regions:
            yy, xx = np.meshgrid(
                np.arange(y0, y1), np.arange(x0, x1), indexing="ij"
            )
            yy, xx = yy.ravel(), xx.ravel()
            overlap = (
                sat[yy + crop, xx + crop]
                - sat[yy, xx + crop]
                - sat[yy + crop, xx]
                + sat[yy, xx]
            ) / float(crop * crop)
            keep = overlap <= thr
            ys_keep.append(yy[keep].astype(np.int32))
            xs_keep.append(xx[keep].astype(np.int32))
            regs_keep.append(np.full(int(keep.sum()), idx, dtype=np.int32))

        ys = np.concatenate(ys_keep)
        if ys.size == 0:
            raise ValueError(
                f"max_train_footprint_overlap={thr} leaves no legal crop "
                f"position in {cfg.zarr_path!r} (frame {self.Y}x{self.X}, crop "
                f"{crop}, holdout {sorted(holdout)}). raise the threshold: with "
                "a 3x3 grid and a 256px crop the achievable floor is ~11% on a "
                "978px frame and 0% only on the larger frames."
            )
        xs = np.concatenate(xs_keep)
        self._allowed_offsets = (ys, xs, np.concatenate(regs_keep))
        # the pixel window that the allowed crops can reach. the window is the
        # reference population of the intensity gate, not the whole frame. see
        # README "How to read the metrics" section 3. a crop at ys.max()
        # reaches ys.max() + crop.
        self._allowed_bbox = (
            int(ys.min()), int(ys.max()) + crop,
            int(xs.min()), int(xs.max()) + crop,
        )

    def _sample_global_offset(self, rng: random.Random) -> tuple[int, int, int]:
        cfg = self.cfg
        # overlap-capped sampling. see README "How to read the metrics"
        # section 2. the code rejects a position here, before it calls
        # _load_crop_raw, because the test is purely geometric. therefore a
        # filtered run reads only the crops that it keeps.
        if cfg.max_train_footprint_overlap is not None:
            if not self._allowed_offsets_built:
                self._build_allowed_offsets()
            if self._allowed_offsets is not None:
                ys, xs, regs = self._allowed_offsets
                i = rng.randrange(len(ys))
                return int(ys[i]), int(xs[i]), int(regs[i])

        if cfg.spatial_region_mode == "all":
            y_off = rng.randrange(0, self.Y - cfg.global_crop_size + 1)
            x_off = rng.randrange(0, self.X - cfg.global_crop_size + 1)
            region_idx = region_index_for_top_left(
                y_off,
                x_off,
                self.Y,
                self.X,
                cfg.global_crop_size,
                cfg.spatial_region_grid_y,
                cfg.spatial_region_grid_x,
            )
            return y_off, x_off, region_idx

        weights = [(y1 - y0) * (x1 - x0) for _, y0, y1, x0, x1 in self._spatial_regions]
        region_idx, y0, y1, x0, x1 = rng.choices(
            self._spatial_regions,
            weights=weights,
            k=1,
        )[0]
        return rng.randrange(y0, y1), rng.randrange(x0, x1), region_idx

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
        edge = ""
        if self.cfg.z_edge_bias > 0.0:
            edge = (
                f"; z_edge_bias={self.cfg.z_edge_bias} "
                f"(band_frac={self.cfg.z_edge_band_frac})"
            )
        print(
            f"[ZarrCellDataset] Z={self.Z}; z range resolved to "
            f"[{self.z_min}, {self.z_max}) slices, thickness={self.cfg.z_thickness}{edge}"
        )
        print(
            f"[ZarrCellDataset] t1 range resolved to "
            f"[{self.t1_min}, {self.t1_max}) frames"
        )
        if self.cfg.spatial_region_mode != "all":
            regions = [r[0] for r in self._spatial_regions]
            print(
                f"[ZarrCellDataset] spatial regions sampled: {regions} "
                f"(mode={self.cfg.spatial_region_mode}, "
                f"holdout={self._spatial_holdout_regions})"
            )

    def _reopen_zarrs(self) -> None:
        """a uniform hook. `worker_init_fn` calls the hook for every dataset type."""
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
        self, rng: random.Random, dataset_idx: int = 0,
    ) -> dict[str, Any]:
        cfg = self.cfg
        g_size = cfg.global_crop_size
        l_size = cfg.local_crop_size
        g_z = cfg.z_thickness
        l_z = cfg.local_z_thickness

        raw_t1, raw_t2 = None, None
        z_start_max = self.z_max - g_z

        # both globals share one z slab, which also bounds the locals.
        # z_edge_bias pushes the slab start toward the edges of the z range.
        # the behaviors of interest usually sit near those edges.

        if cfg.overrepresent_classes and rng.random() < cfg.rep_alpha and self.minority_classes: 
            # rep_alpha injects annotated probe crops into pre-training. see
            # README "How to read the metrics" section 1.
            label = self.minority_classes[rng.randrange(len(self.minority_classes))]
            df = self.by_label[label]
            row = df.iloc[rng.randrange(len(df))]
            t1 = row['t1']
            t2 = row['t2']
            z = row['z']
            y_off = row['global_y']
            x_off = row['global_x']

            raw_t1 = self._load_crop_raw(t1, z, y_off, x_off, g_size, g_size)
            raw_t2 = self._load_crop_raw(t2, z, y_off, x_off, g_size, g_size)

            region_idx = region_index_for_top_left(
                y_off,
                x_off,
                self.Y,
                self.X,
                cfg.global_crop_size,
                cfg.spatial_region_grid_y,
                cfg.spatial_region_grid_x,
            )

        else:
            # the gate is relative to the (t1, z) slab mean. therefore the
            # anchor controls whether any legal position can pass. a new draw
            # of (y, x) alone cannot escape a bad anchor, because the inner
            # loop draws from the same fixed offset set every time. thus the
            # outer loop retries the anchor. see README "How to read the
            # metrics" section 3.
            best_mean = float("-inf")
            worst_anchor: tuple[int, int, float] | None = None
            for _anchor_attempt in range(cfg.intensity_anchor_attempts):
                if cfg.z_edge_bias > 0.0 and rng.random() < cfg.z_edge_bias:
                    band = int(round(cfg.z_edge_band_frac * (z_start_max - self.z_min)))
                    if rng.random() < 0.5:  # low edge (near z_min)
                        z = rng.randrange(self.z_min, self.z_min + band + 1)
                    else:  # high edge (near z_max)
                        z = rng.randrange(z_start_max - band, z_start_max + 1)
                else:
                    z = rng.randrange(self.z_min, z_start_max + 1)
                t1 = rng.randrange(self.t1_min, self.t1_max)
                sign = 1 if cfg.temporal_direction == "forward" else -1
                t2 = t1 + sign * self.frame_gap

                # gate against the per-(t1, z_slab) baseline. one (y, x) serves
                # both globals. _allowed_bbox is None in training, which means
                # a whole-frame reference. _allowed_bbox narrows to the allowed
                # window when the overlap cap applies. see README "How to read
                # the metrics" section 3.
                if not self._allowed_offsets_built:
                    self._build_allowed_offsets()
                threshold = cfg.intensity_min_frac * self._get_frame_mean(
                    t1, z, self._allowed_bbox
                )
                passed = False
                for _ in range(cfg.intensity_max_attempts):
                    y_off, x_off, region_idx = self._sample_global_offset(rng)
                    raw_t1 = self._load_crop_raw(t1, z, y_off, x_off, g_size, g_size)
                    crop_mean = float(raw_t1.mean())
                    best_mean = max(best_mean, crop_mean)
                    if crop_mean >= threshold:
                        passed = True
                        break
                if passed:
                    break
                worst_anchor = (t1, z, threshold)
            else:
                t1_l, z_l, thr_l = worst_anchor  # type: ignore[misc]
                raise RuntimeError(
                    "could not sample a global crop that passes the intensity gate "
                    f"after {cfg.intensity_anchor_attempts} anchor draws x "
                    f"{cfg.intensity_max_attempts} offsets: dataset={cfg.zarr_path!r}, "
                    f"last anchor t1={t1_l}, z={z_l}, threshold={thr_l:.6g}, "
                    f"best_crop_mean over all attempts={best_mean:.6g}. every legal "
                    "position is dimmer than intensity_min_frac x the slab mean. if "
                    "max_train_footprint_overlap is set, the allowed set is one frame "
                    "corner and this is expected -- raise the cap or lower "
                    "intensity_min_frac. otherwise inspect this dataset for empty regions."
                )
            raw_t2 = self._load_crop_raw(t2, z, y_off, x_off, g_size, g_size)

        globals_tensor = torch.stack([
            self._augment(self._to_tensor(raw_t1)),
            self._augment(self._to_tensor(raw_t2)),
        ])

        # the code splits the locals evenly between the two global parents.
        # because both globals share (y, x), the spatial nest collapses to the
        # same window. the code still records the parent index, so the loss can
        # pair each local with the correct teacher target.
        n_local = cfg.n_local_crops
        n_per = n_local // 2
        local_parents: list[int] = [0] * n_per + [1] * (n_local - n_per)

        t_lo, t_hi = (t1, t2) if t1 <= t2 else (t2, t1)
        lz_start_max = z + g_z - l_z  # inclusive upper bound

        local_ts: list[int] = []
        local_zs: list[int] = []
        local_offsets: list[tuple[int, int]] = []
        locals_: list[torch.Tensor] = []
        for _parent in local_parents:
            if cfg.local_time_endpoints:
                t_local = t1 if rng.random() < 0.5 else t2
            else:
                t_local = rng.randrange(t_lo, t_hi + 1)
            lz = rng.randrange(z, lz_start_max + 1)
            ly = rng.randrange(y_off, y_off + g_size - l_size + 1)
            lx = rng.randrange(x_off, x_off + g_size - l_size + 1)
            if cfg.local_time_endpoints:
                # the local already nests in its parent global in z, y and x.
                # therefore an endpoint timepoint makes the local a slice of an
                # array that is already in memory. the code does no zarr read
                # here, which matters because the throughput is NFS-I/O bound.
                # see README "Training". the parent origin is (z, y_off, x_off).
                # the t1 test compares values, so the code also handles
                # temporal_direction "backward".
                src = raw_t1 if t_local == t1 else raw_t2
                raw_l = np.ascontiguousarray(src[
                    lz - z : lz - z + l_z,
                    ly - y_off : ly - y_off + l_size,
                    lx - x_off : lx - x_off + l_size,
                ])
            else:
                raw_l = self._load_crop_raw(
                    t_local, lz, ly, lx, l_size, l_size, z_thickness=l_z
                )
            local_ts.append(t_local)
            local_zs.append(lz)
            local_offsets.append((ly, lx))
            locals_.append(self._augment(self._to_tensor(raw_l)))

        sample: dict[str, Any] = {
            # (2, 1, z_thickness, g_size, g_size). index 0 is t1 and index 1 is t2.
            "globals": globals_tensor,
            # WS-DINO weak label: (dataset_idx, frame_gap). dataset_idx is 0 for
            # a single-zarr dataset. MultiZarrDataset passes the child index
            # that it drew. see README "How to read the metrics" section 10.
            "weak_label": torch.tensor(
                [dataset_idx, self.frame_gap], dtype=torch.long
            ),
            "meta": {
                "t1": t1,
                "t2": t2,
                "z": z,
                "z_end": z + g_z,
                "dataset_idx": dataset_idx,
                "frame_gap": self.frame_gap,
                "global_t": torch.tensor([t1, t2]),
                "global_y": y_off,
                "global_x": x_off,
                "frame_y": self.Y,
                "frame_x": self.X,
                "spatial_region_idx": region_idx,
            },
        }
        if locals_:
            sample["locals"] = torch.stack(locals_)
            sample["meta"]["local_parent"] = torch.tensor(local_parents)
            sample["meta"]["local_t"] = torch.tensor(local_ts)
            sample["meta"]["local_z"] = torch.tensor(local_zs)
            sample["meta"]["local_z_end"] = torch.tensor([lz + l_z for lz in local_zs])
            sample["meta"]["local_y"] = torch.tensor([y for y, _ in local_offsets])
            sample["meta"]["local_x"] = torch.tensor([x for _, x in local_offsets])
        return sample

    def _annotation_tensor(
        self, t1: int, t2: int, z: int, global_x: int, global_y: int, dataset_idx: int = 0
    ) -> dict[str, Any]:
        """load a pair of global crops from annotation coordinates. no augmentation runs."""
        cfg = self.cfg
        g_size = cfg.global_crop_size
        g_z = cfg.z_thickness

        raw_t1 = self._load_crop_raw(t1, z, global_y, global_x, g_size, g_size)
        raw_t2 = self._load_crop_raw(t2, z, global_y, global_x, g_size, g_size)
        globals_tensor = torch.stack([
            self._to_tensor(raw_t1),
            self._to_tensor(raw_t2),
        ])

        return {
            "globals": globals_tensor,
            "meta": {
                "t1": t1,
                "t2": t2,
                "z": z,
                "z_end": z + g_z,
                "global_y": global_y,
                "global_x": global_x,
                "frame_y": self.Y,
                "frame_x": self.X,
                "spatial_region_idx": 0,
                "dataset_idx": dataset_idx,
                "frame_gap": self.frame_gap,
            },
        }

    def _get_frame_mean(
        self, t: int, z: int, bbox: tuple[int, int, int, int] | None = None
    ) -> float:
        """return the mean intensity over the active z slab [z, z+z_thickness) at frame t.

        each worker caches the mean with the key (t, z, bbox), so identical
        anchors reuse the read. the mean covers the slab that gives the crops,
        not the full z window. therefore the threshold keeps the same units as
        the crop means.

        `bbox` restricts the mean to a (y0, y1, x0, x1) pixel window. `bbox` is
        None in training, which means the whole frame. the caller sets `bbox`
        when max_train_footprint_overlap confines the sampler to a sub-window.
        see README "How to read the metrics" section 3 for the reason why the
        reference population must be the allowed bbox.
        """
        key = (t, z, bbox)
        cached = self._frame_mean_cache.get(key)
        if cached is not None:
            return cached
        ys = slice(None) if bbox is None else slice(bbox[0], bbox[1])
        xs = slice(None) if bbox is None else slice(bbox[2], bbox[3])
        frame = np.asarray(
            self.arr[
                t,
                self.cfg.channel,
                z : z + self.cfg.z_thickness,
                ys,
                xs,
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
        z_thickness: int | None = None,
    ) -> np.ndarray:
        # a raw float32 zyx slab. the code does not normalize the slab, so the
        # intensity gate can compare the crop mean and the frame mean in the
        # same units. you can override `z_thickness`, so a local can take a
        # thinner sub-slab than a global.
        if z_thickness is None:
            z_thickness = self.cfg.z_thickness
        sub = self.arr[
            t,
            self.cfg.channel,
            z : z + z_thickness,
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
    """mix pairs from several zarrs in every batch.

    each child is a fully-initialized `ZarrCellDataset`. the code builds each
    child from a copy of the parent config and sets `zarr_path` to one of
    `cfg.zarr_paths`.

    on every balanced batch, the wrapper schedules equal child counts. the
    counts differ by one sample when the division is uneven. the wrapper
    rotates the remainder slots across the batches. a pair never crosses an
    experiment boundary. only the batch crosses one.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        if not cfg.zarr_paths:
            raise ValueError(
                "cfg.zarr_paths must be a non-empty list to build a MultiZarrDataset"
            )
        if cfg.batch_size < 1:
            raise ValueError(f"cfg.batch_size must be >= 1; got {cfg.batch_size}")
        self.cfg = cfg

        # filter out the experiments that are not-for-training, which have a
        # blank z_limit in the spreadsheet. training excludes these experiments
        # silently. the code logs which experiments it skips, so you can see a
        # spreadsheet row with a bad configuration.
        usable_paths: list[str] = []
        skipped_paths: list[str] = []
        for path in cfg.zarr_paths:
            if get_z_limit(path, cfg.training_data_index_path) is None:
                skipped_paths.append(path)
            else:
                usable_paths.append(path)
        if skipped_paths:
            print(
                f"[MultiZarrDataset] skipped {len(skipped_paths)} zarr(s) "
                f"with blank z_limit:"
            )
            for p in skipped_paths:
                print(f"  - {p}")
        if not usable_paths:
            raise ValueError(
                "no usable zarrs after filtering by spreadsheet z_limit; "
                "every entry in cfg.zarr_paths has a blank z_limit"
            )

        # one child per usable zarr. each child resolves its own shape,
        # interval_minutes and frame_gap from the shared target_gap_minutes.
        common_content_holdout = (
            cfg.spatial_region_mode != "all"
            and cfg.spatial_holdout_selection == "content"
            and cfg.spatial_holdout_resolved_regions is None
            and len(usable_paths) > 1
        )
        self.children: list[ZarrCellDataset] = []
        for path in usable_paths:
            child_cfg = replace(cfg, zarr_path=path, zarr_paths=None)
            if common_content_holdout:
                child_cfg = replace(child_cfg, spatial_region_mode="all")
            self.children.append(ZarrCellDataset(child_cfg))
        self._apply_common_content_holdout()

        # weight each child by the number of valid t1 anchors that it gives.
        # the weight favors the longer experiments in proportion, but it does
        # not starve the short experiments to zero.
        sizes = [max(1, c.t1_max - c.t1_min) for c in self.children]
        total = float(sum(sizes))
        self.weights = [s / total for s in sizes]
        self._log_open()

    def _apply_common_content_holdout(self) -> None:
        cfg = self.cfg
        if cfg.spatial_region_mode == "all":
            return
        if cfg.spatial_holdout_selection != "content":
            return
        if cfg.spatial_holdout_resolved_regions is not None:
            return
        if len(self.children) < 2:
            return

        child_scores = [child._score_spatial_regions_by_content() for child in self.children]
        region_ids = sorted(child_scores[0])
        aggregate_scores = {
            idx: min(scores[idx] for scores in child_scores)
            for idx in region_ids
        }
        selected = select_nonadjacent_regions(
            aggregate_scores,
            cfg.spatial_holdout_region_count,
            cfg.spatial_region_grid_x,
            cfg.spatial_holdout_nonadjacent_gap,
            cfg.spatial_holdout_min_content,
        )
        score_text = ", ".join(
            f"{idx}:{aggregate_scores[idx]:.3f}" for idx in region_ids
        )
        print(
            "[MultiZarrDataset] common content holdout scores "
            f"(min across datasets): {score_text}; selected={selected}"
        )

        self.cfg = replace(cfg, spatial_holdout_resolved_regions=selected)
        for child in self.children:
            child.cfg = replace(
                child.cfg,
                spatial_region_mode=cfg.spatial_region_mode,
                spatial_holdout_resolved_regions=selected,
            )
            child._spatial_holdout_regions = selected
            child._spatial_regions = child._build_spatial_regions()
            # the allowed-offset set depends on which regions are held out. the
            # code must recompute the set now, because the regions are resolved.
            child._allowed_offsets = None
            child._allowed_bbox = None
            child._allowed_offsets_built = False

    def _log_open(self) -> None:
        mode = (
            "balanced batches"
            if self.cfg.balance_datasets
            else "weighted sampling"
        )
        print(f"[MultiZarrDataset] {len(self.children)} children ({mode}):")
        for i, (child, w) in enumerate(zip(self.children, self.weights)):
            print(
                f"  [{i}] {child.cfg.zarr_path} "
                f"T={child.T} frame_gap={child.frame_gap} weight={w:.3f}"
            )

    def _reopen_zarrs(self) -> None:
        """reopen the zarr handle of every child in this worker process."""
        for child in self.children:
            child._open_zarr()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        info = get_worker_info()
        if info is None:
            rng = random.Random(self.cfg.seed)
        else:
            rng = random.Random(self.cfg.seed + info.id)

        indices = list(range(len(self.children)))
        batch_number = info.id if info is not None else 0
        while True:
            if self.cfg.balance_datasets:
                batch_indices = balanced_batch_indices(
                    len(self.children),
                    self.cfg.batch_size,
                    batch_number,
                    rng,
                )
                batch_number += 1
                for idx in batch_indices:
                    yield self.children[idx]._sample_pair(rng, dataset_idx=idx)
            else:
                idx = rng.choices(indices, weights=self.weights, k=1)[0]
                yield self.children[idx]._sample_pair(rng, dataset_idx=idx)


def worker_init_fn(worker_id: int) -> None:
    """re-open every zarr handle in each worker process."""
    info = get_worker_info()
    if info is None:
        return
    ds = info.dataset
    ds._reopen_zarrs()
    # seed numpy and torch per worker, so the augmentations stay reproducible
    # but distinct.
    seed = ds.cfg.seed + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)
