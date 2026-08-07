from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

# repo root. the config resolves relative paths against this root. this file is
# at <repo>/configs/config.py, so the root is two parents up.
_REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class AugCfg:
    enabled: bool = True
    p: float = 0.5
    # shared fields keep the aug config compact.
    strength: float = 0.0
    gamma_range: tuple[float, float] = (1.0, 1.0)
    sigma_range: tuple[float, float] = (0.0, 0.0)
    kernel_size: int = 5


@dataclass
class Config:
    # data
    # single-zarr training. set either zarr_path or `zarr_paths`, not both.
    # the default is null, so the file picker or the --zarr-path flag can fill
    # zarr_path at runtime.
    zarr_path: str | None = None

    # multi-zarr training. when zarr_paths is non-null, the dataloader mixes
    # pairs from every zarr in the list in each batch. the dataloader then
    # ignores `zarr_path`.
    zarr_paths: list[str] | None = None

    # give every usable zarr equal representation in each batch. the counts
    # differ by one at most when batch_size is not divisible by the dataset
    # count.
    balance_datasets: bool = True

    # path to the jsonl file with behavior/dataset annotations for the probes.
    # __post_init__ resolves a relative path against the repo root, not against
    # the working directory. therefore "index.jsonl" works from any directory
    # and the config stays portable between checkouts. __post_init__ does not
    # change an absolute path.
    annotation_path: str | None = None

    # draw a fraction of PRE-TRAINING samples straight from annotated crop
    # coordinates, to enrich rare behaviors. warning: the fraction applies to
    # training, not to a probe fit. the branch skips spatial-region selection
    # completely, so the encoder sees probe crops at their exact positions. the
    # full-set behavior probe is too high as a result. eval/probe_dataset.py
    # forces overrepresent_classes off. see README "How to read the metrics"
    # section 1.
    overrepresent_classes: bool = False

    # strength of that oversampling. rep_alpha is the fraction of training
    # samples that come from the annotated set, so rep_alpha is also the leakage
    # knob. 0720 ran 0.15, and current runs use 0.06. a higher value gives more
    # rare-class exposure and lifts the full-set probe higher.
    rep_alpha: float = 0.5

    channel: int = 0
    z_thickness: int = 10

    # local crops use a thinner, contiguous z sub-slab from inside the global z
    # window. local_z_thickness must be <= z_thickness.
    local_z_thickness: int = 5

    # multi-crop: exactly 2 globals (t1 and t2, same (y, x) and z slab) plus
    # n_local_crops locals. the sampler splits the locals evenly between the two
    # globals as the spatial parent. each local sits inside that parent in z, y
    # and x. n_local_crops must be even.
    n_local_crops: int = 6
    global_crop_size: int = 256
    local_crop_size: int = 96

    # draw each local's timepoint from {t1, t2}, not uniformly over [t1, t2].
    # each local is then a slice of an array already in memory, not a fresh
    # zarr read (50 -> 20 plane reads per sample; see README "Training",
    # "Throughput is NFS-I/O bound, not compute bound"). the change does not
    # affect cross-time invariance, because the sampler sets local_parent
    # independently of the timepoint. false restores the old uniform draw for
    # an ablation.
    local_time_endpoints: bool = True

    # accept a global crop only if the crop mean is at least this fraction of
    # the mean over the active z slab at t1. a lower value is more permissive.
    # the reference population matters: see README "How to read the metrics"
    # section 3.
    intensity_min_frac: float = 0.95
    # fail with context instead of a DataLoader that hangs on an impossible gate.
    intensity_max_attempts: int = 1000
    # number of redraws of the (t1, z) anchor when no allowed (y, x) offset
    # passes the gate. feasibility is a property of the anchor, not of the
    # offset, so the inner offset loop alone cannot escape a dim slab. keep the
    # value small, because each anchor that fails costs a full
    # intensity_max_attempts sweep.
    intensity_anchor_attempts: int = 8

    # spatial region holdout for leakage-controlled probes. the regions form a
    # row-major grid over legal global-crop top-left positions. DINO training
    # must use "exclude_holdout". the dataset-ID probe can set the mode to
    # "holdout" instead, so the probe sees only regions that the encoder did not
    # see. see README "How to read the metrics" section 2.
    spatial_region_grid_y: int = 3
    spatial_region_grid_x: int = 3
    # manual holdout regions. the singular form is the older, backward-compatible
    # name.
    spatial_holdout_region: int | None = None
    spatial_holdout_regions: list[int] | None = None
    spatial_region_mode: str = "all"
    # "manual" uses spatial_holdout_regions/spatial_holdout_region. "content"
    # scores the tiles per dataset, then selects high-content, non-adjacent
    # regions.
    spatial_holdout_selection: str = "manual"
    spatial_holdout_region_count: int = 3
    spatial_holdout_nonadjacent_gap: int = 1
    spatial_holdout_min_content: float = 0.95
    spatial_holdout_content_samples_per_region: int = 8
    spatial_holdout_seed: int = 0

    # permanent, and necessary together with the 2x2 grid. the grid decides
    # whether a clean crop position exists, and this cap selects the position.
    # when you set the cap (0-1), the sampler keeps only global crops that
    # overlap the training footprint by this fraction at most, and the sampler
    # tests the overlap geometrically before any read. the cap applies to eval
    # only, and None is a no-op for training. see README "How to read the
    # metrics" section 2.
    max_train_footprint_overlap: float | None = None
    # internal override. MultiZarrDataset sets this field after it aggregates
    # the content scores, so every dataset holds out the same tile indices.
    spatial_holdout_resolved_regions: list[int] | None = None

    # spreadsheet that maps experiment id -> interval_minutes.
    training_data_index_path: str = "training_data_index.xlsx"

    # wall-clock gap that you want between paired frames, in minutes.
    # the dataset converts the gap to a frame_gap with the experiment's interval.
    # the default 50 gives 5 frames at a 10-minute interval, the WS-DINO weak
    # label.
    target_gap_minutes: float = 50.0

    # "forward" -> t2 = t1 + frame_gap; "backward" -> t2 = t1 - frame_gap.
    temporal_direction: str = "forward"

    # z lower bound as a fraction of each dataset. the dataset takes the upper
    # bound from the per-experiment `z_limit` column in training_data_index.xlsx
    # (1-indexed inclusive, equivalent to a 0-indexed exclusive cap).
    z_frac_min: float = 0.0

    # optional absolute z overrides. the dataset clamps z_abs_max to the
    # spreadsheet z_limit, so z_abs_max can only narrow the window and can never
    # extend past z_limit.
    z_abs_min: int | None = None
    z_abs_max: int | None = None

    # bias the z-slab draw toward the top or the bottom of the z range. with
    # probability `z_edge_bias` the sampler draws a sample's z slab from an edge
    # band (low or high, 50/50) instead of uniformly across the whole z range.
    # 0.0 (the default) keeps a uniform draw. `z_edge_band_frac` is the width of
    # each band as a fraction of the usable z-start span.
    z_edge_bias: float = 0.0
    z_edge_band_frac: float = 0.15

    # t1 window as a fraction of each dataset.
    t1_frac_min: float = 0.0
    t1_frac_max: float = 1.0

    # absolute t1 override for one known dataset.
    t1_abs_min: int | None = None
    t1_abs_max: int | None = None

    # augmentations
    aug_intensity: AugCfg = field(default_factory=lambda: AugCfg(
        enabled=True, p=0.8, strength=0.4,
    ))
    aug_gamma: AugCfg = field(default_factory=lambda: AugCfg(
        enabled=True, p=0.8, gamma_range=(0.5, 2.0),
    ))
    aug_noise: AugCfg = field(default_factory=lambda: AugCfg(
        enabled=True, p=0.5, sigma_range=(0.0, 0.05),
    ))
    aug_blur: AugCfg = field(default_factory=lambda: AugCfg(
        enabled=True, p=0.5, sigma_range=(0.1, 2.0), kernel_size=5,
    ))
    aug_percrop_norm: AugCfg = field(default_factory=lambda: AugCfg(
        enabled=True, p=1.0,
    ))
    aug_flip: AugCfg = field(default_factory=lambda: AugCfg(
        enabled=False, p=0.5,
    ))
    aug_rotation: AugCfg = field(default_factory=lambda: AugCfg(
        enabled=False, p=0.5,
    ))

    # runtime
    max_intensity: float = 10000.0
    num_workers: int = 4
    seed: int = 0

    # ------------------------------------------------------------------
    # model
    # ------------------------------------------------------------------
    # hybrid encoder: a small 3D stem collapses z, then a 2D ResNet-50 runs.
    # stem_channels is the channel count that the stem gives to the ResNet's
    # first conv. the encoder replaces the ResNet's default 3-channel conv.
    stem_channels: int = 64
    # ResNet-50's pre-classifier output dim, the encoder's representation size.
    latent_dim: int = 2048
    # groupnorm group count. groupnorm replaces batchnorm, so normalization is
    # per-sample. no running stats exist for the masked/multi-scale ssl crops to
    # pollute, and the frozen encoder has no train/eval mismatch. the encoder
    # clamps the count to the largest divisor of each layer's channel count that
    # is <= this value.
    norm_groups: int = 32
    # crops per pass through the 3D stem. the dino wrapper flattens every crop
    # into one batch, and the stem's full-resolution activations dominate gpu
    # memory (~20 GiB for 128 globals). the chunk split is exact and is the main
    # lever against oom. None disables the chunk split. lower the value on oom,
    # and raise the value for speed.
    encoder_chunk_size: int | None = 32
    # gradient checkpointing, split by half of the encoder. the stem's
    # activations dominate memory and are cheap to recompute, so keep
    # checkpoint_stem on. the backbone instead holds most of the flops, so
    # gradient checkpointing on the backbone costs ~25% throughput for a modest
    # reduction in memory. turn checkpoint_backbone on only for a real oom.
    checkpoint_stem: bool = True
    checkpoint_backbone: bool = False

    # the training discards the projector (DINO head) at the end. K is the
    # prototype count that the cross-entropy operates over.
    projector_hidden: int = 512
    projector_bottleneck: int = 256
    projector_k: int = 512

    # ------------------------------------------------------------------
    # DINO
    # ------------------------------------------------------------------
    # teacher softmax temperature (sharp). a lower value gives more confident
    # targets.
    teacher_temp_start: float = 0.04
    teacher_temp_end: float = 0.07
    temp_warmup_frac: float = 0.3
    # student softmax temperature (softer). the asymmetry against the teacher
    # drives the training.
    student_temp: float = 0.1
    # EMA momentum schedule for the teacher weights. cosine from start to end.
    ema_momentum_start: float = 0.996
    ema_momentum_end: float = 1.0
    # center buffer momentum. the buffer prevents dominance by one prototype.
    center_momentum: float = 0.9

    # ------------------------------------------------------------------
    # adversarial dataset classifier
    # ------------------------------------------------------------------
    # the classifier predicts dataset_idx from encoder features. at the same
    # time a gradient reversal layer sends the opposite gradient into the
    # encoder, which discourages leakage of the dataset identity. see README
    # "Training", "What gets trained".
    dataset_adversary_enabled: bool = False
    dataset_adversary_weight: float = 0.1
    dataset_adversary_grl_start: float = 0.0
    dataset_adversary_grl_scale: float = 1.0
    dataset_adversary_hidden: int = 256
    dataset_adversary_dropout: float = 0.1

    # ------------------------------------------------------------------
    # masked patch autoencoder
    # ------------------------------------------------------------------
    # auxiliary reconstruction loss on global crops. the decoder applies to
    # training only, and the training discards the decoder at the end. see
    # README "Training", "What gets trained".
    mae_enabled: bool = False
    mae_weight: float = 0.05
    mae_mask_ratio: float = 0.75
    mae_patch_z: int = 2
    mae_patch_size: int = 16
    mae_decoder_hidden: int = 512
    mae_decoder_depth: int = 2
    mae_norm_pix_loss: bool = True


    # ---------------------------------------------------------------------------
    # Koleo auxiliary loss
    # ---------------------------------------------------------------------------

    koleo_enabled: bool = False
    koleo_weight: float = 0.1     # DINOv2's value

    # ------------------------------------------------------------------
    # iBOT masking-based self-supervised objective
    # ------------------------------------------------------------------
    # the student predicts the teacher's soft token distribution on masked
    # global crops. the objective reuses DINO's teacher, student and center.
    # iBOT and MAE are mutually exclusive (__post_init__ resolves the conflict).
    # see README "Training", "What gets trained".
    ibot_enabled: bool = False
    ibot_p: float = 0.5
    ibot_weight: float = 1.0
    # mask fraction. the sampler draws the fraction uniformly from this range at
    # each step.
    ibot_mask_ratio: tuple = (0.1, 0.5)
    ibot_layer: str = "layer4"
    ibot_block_size: int = 2

    # ------------------------------------------------------------------
    # optimizer / schedule
    # ------------------------------------------------------------------
    base_lr: float = 2.5e-4
    weight_decay_start: float = 0.04
    weight_decay_end: float = 0.4
    grad_clip_norm: float = 1.0
    # fraction of total_steps in linear warmup. the rest is cosine decay.
    warmup_frac: float = 0.15

    # ------------------------------------------------------------------
    # training loop
    # ------------------------------------------------------------------
    batch_size: int = 64
    total_steps: int = 100_000
    log_interval: int = 50
    ckpt_interval: int = 2000
    device: str = "cuda"
    use_amp: bool = True
    # autocast dtype: "fp16" or "bf16". the code falls back to fp16 on hardware
    # without bf16 support. keep fp16. the division of the DINO head's [-1, 1]
    # logits by teacher_temp amplifies bf16's coarser mantissa by ~14-25x, and
    # bf16 collapsed the effective prototype count in 0727-ThruTuning3. see
    # README "Results: where the models stand".
    amp_dtype: str = "fp16"

    # ------------------------------------------------------------------
    # io
    # ------------------------------------------------------------------
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # serve the root log_dir (all runs) together with the training. the server
    # binds to all interfaces, so the compute node can reach the server. the
    # code tears the server down at exit.
    tensorboard_enabled: bool = True
    tensorboard_port: int = 6006

    def __post_init__(self) -> None:
        # only one masking objective can run. iBOT wins, because iBOT is the
        # stronger signal. the code resolves the conflict here, so the
        # serialized config records the effective state. see README "Training",
        # "What gets trained".
        if self.ibot_enabled and self.mae_enabled:
            print(
                "[config] ibot_enabled and mae_enabled both set; disabling MAE "
                "(iBOT takes precedence as the primary masking objective)"
            )
            self.mae_enabled = False

        # resolve a relative annotation_path against the repo root, not against
        # the working directory. this file lives in <repo>/configs/, so the root
        # is two levels up. the code resolves the path here, so every consumer
        # gets an absolute path. the consumers are the sampler and all six eval
        # scripts, and no consumer needs to know its own working directory.
        #
        # archived checkpoints/<run>/config.yaml snapshots hold an absolute
        # posix path from before this change. the code must pass those paths
        # through unchanged.
        #
        # warning: the startswith check is necessary on windows. pathlib
        # treats "/groups/sgro/..." as RELATIVE on windows (no drive letter).
        # is_absolute() alone therefore rewrites every archived config's path to
        # Z:\groups\sgro\... and breaks all the eval scripts off the mounted
        # drive.
        if self.annotation_path is not None:
            looks_absolute = (
                Path(self.annotation_path).is_absolute()
                or self.annotation_path.startswith(("/", "\\"))
            )
            if not looks_absolute:
                self.annotation_path = str(_REPO_ROOT / self.annotation_path)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """build a Config from a yaml file.

        keys that the yaml file does not set keep their python defaults.
        therefore the yaml file does not need to list every knob.
        """
        import yaml

        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"top-level yaml in {path} must be a mapping")
        return _build_dataclass(cls, raw)


# these field names existed before and still appear in archived config
# snapshots under checkpoints/<run>/config.yaml. every eval script runs with one
# of those frozen files. warning: if you delete a field and you do not name the
# field here, every past run becomes permanently unloadable. the loader ignores
# a key from this set and prints a warning instead of an error.
#
# add a name when you delete a field. never reuse a retired name for another
# purpose, because an old snapshot then feeds the old value to the new field
# silently. a name in this set that is still a live field has no effect (see
# below). therefore you can add the entry in the same commit that removes the
# field, or earlier.
_DEPRECATED_KEYS = {
    # both names became obsolete before 2026-08-06, when an old config still
    # raised an error. both names appear in pre-0715 snapshots: weight_decay in
    # 55 of them, and teacher_temp in 20. a later change split each name into a
    # start/end pair (weight_decay_start/_end, teacher_temp_start/_end), so no
    # single value exists to map them onto. one number cannot give back an old
    # run's schedule endpoints. the two entries here make those 55 archived runs
    # loadable again.
    "weight_decay",
    "teacher_temp",
}


def _build_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """build a dataclass from a plain dict. the function recurses into nested dataclasses."""
    field_names = {f.name for f in fields(cls)}
    unknown = set(data) - field_names
    # the loader drops only the keys that are BOTH unrecognized and retired. the
    # code subtracts from `unknown`, not from `data`, so the loader still honors
    # a deprecated name that is a live field.
    deprecated = unknown & _DEPRECATED_KEYS
    if deprecated:
        print(
            f"[config] ignoring key(s) from an older config that no longer "
            f"exist in {cls.__name__}: {sorted(deprecated)}"
        )
        data = {k: v for k, v in data.items() if k not in deprecated}
        unknown -= deprecated
    # anything still unrecognized is a typo, and still an error.
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    # resolve string annotations (from `from __future__ import annotations`) to real types.
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        ftype = hints.get(name)
        if isinstance(value, dict) and is_dataclass(ftype):
            kwargs[name] = _build_dataclass(ftype, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)
