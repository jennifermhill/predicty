from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints


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
    # single-zarr training. set this OR `zarr_paths`, not both.
    # left null so the file picker / --zarr-path flag can fill it at runtime.
    zarr_path: str | None = None

    # multi-zarr training. when non-null, the dataloader mixes pairs from
    # every listed zarr in each batch and `zarr_path` is ignored.
    zarr_paths: list[str] | None = None

    # give every usable zarr equal representation within each batch. counts
    # differ by at most one when batch_size is not divisible by dataset count.
    balance_datasets: bool = True

    # path to jsonl file with behavior/dataset annotations for probes.
    annotation_path: str | None = None

    # oversample minority behavior classes during probe training.
    overrepresent_classes: bool = False

    # strength of class oversampling when overrepresent_classes is true.
    # higher = stronger oversampling of rare classes.
    rep_alpha: float = 0.5

    channel: int = 0
    z_thickness: int = 10

    # local crops use a thinner, contiguous z sub-slab drawn from inside the
    # global z window. must be <= z_thickness.
    local_z_thickness: int = 5

    # multi-crop architecture: exactly 2 globals (one at t1, one at t2,
    # sharing the same (y, x) and z slab) and n_local_crops locals. each local
    # is split evenly between the two globals as spatial parent, samples its
    # time uniformly from [t1, t2], picks a contiguous local_z_thickness
    # sub-slab inside the global z window, and is spatially nested inside its
    # parent global's (y, x) window. n_local_crops must be even.
    n_local_crops: int = 6
    global_crop_size: int = 256
    local_crop_size: int = 96

    # accept a global crop only if its mean is at least this fraction of the
    # mean intensity over the active z slab at t1. lower values are more permissive.
    intensity_min_frac: float = 0.95
    # Stop retrying an impossible intensity gate forever. A bad threshold or
    # empty field should fail with context instead of hanging a DataLoader.
    intensity_max_attempts: int = 1000

    # Spatial region holdout for leakage-controlled probes. Regions are a
    # row-major grid over legal global-crop top-left positions. DINO training
    # should use "exclude_holdout"; the dataset-ID probe can override this to
    # "holdout" so it only sees regions the encoder did not.
    spatial_region_grid_y: int = 3
    spatial_region_grid_x: int = 3
    # Backward-compatible single manual holdout region.
    spatial_holdout_region: int | None = None
    # Preferred manual form for multiple held-out regions.
    spatial_holdout_regions: list[int] | None = None
    spatial_region_mode: str = "all"
    # "manual" uses spatial_holdout_regions/spatial_holdout_region. "content"
    # scores tiles per dataset and selects high-content, non-adjacent regions.
    spatial_holdout_selection: str = "manual"
    spatial_holdout_region_count: int = 3
    spatial_holdout_nonadjacent_gap: int = 1
    spatial_holdout_min_content: float = 0.95
    spatial_holdout_content_samples_per_region: int = 8
    spatial_holdout_seed: int = 0
    # Internal override used by MultiZarrDataset after aggregating content
    # scores across datasets, so every dataset holds out the same tile indices.
    spatial_holdout_resolved_regions: list[int] | None = None

    # spreadsheet that maps experiment id -> interval_minutes.
    training_data_index_path: str = "training_data_index.xlsx"

    # desired wall-clock gap between paired frames, in minutes.
    # the dataset converts this to a frame_gap using the experiment's interval.
    # default 50 -> 5 frames at a 10-minute interval, the WS-DINO weak label.
    target_gap_minutes: float = 50.0

    # "forward" -> t2 = t1 + frame_gap; "backward" -> t2 = t1 - frame_gap.
    temporal_direction: str = "forward"

    # z lower bound as a fraction of each dataset. the upper bound is taken
    # from the per-experiment `z_limit` column in training_data_index.xlsx
    # (1-indexed inclusive, equivalent to a 0-indexed exclusive cap).
    z_frac_min: float = 0.0

    # optional absolute z overrides. z_abs_max is clamped to the spreadsheet
    # z_limit (it can only narrow the window, never extend past z_limit).
    z_abs_min: int | None = None
    z_abs_max: int | None = None

    # bias z-slab sampling toward the top/bottom of the z range. with
    # probability `z_edge_bias` a sample's z slab is drawn from an edge band
    # (low or high, 50/50) instead of uniformly across the whole z range;
    # 0.0 (default) keeps uniform sampling. `z_edge_band_frac` is each band's
    # width as a fraction of the usable z-start span.
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
    # hybrid encoder: small 3D stem collapses z, then 2D ResNet-50.
    # stem_channels is the channel count handed to the ResNet's first conv
    # (ResNet's default 3-channel conv is replaced).
    stem_channels: int = 64
    # ResNet-50's pre-classifier output dim, the encoder's representation size.
    latent_dim: int = 2048
    # groupnorm group count for the encoder. groupnorm replaces batchnorm so
    # normalization is per-sample: no running-stat buffers to be polluted by the
    # masked/multi-scale crops seen during ssl, and no train/eval mismatch in the
    # frozen encoder. the effective group count is clamped to the largest divisor
    # of each layer's channel count that is <= this value.
    norm_groups: int = 32
    # cap on how many crops flow through the encoder's 3D stem at once. the dino
    # wrapper flattens all crops into one batch (batch_size * 2 globals, plus the
    # locals), and the stem's full-resolution activations dominate gpu memory
    # (groupnorm runs in fp32 under amp, ~20 GiB for 128 globals). chunking the
    # stem forward is exact (each crop encodes independently) and, with the
    # encoder's whole-forward checkpointing, is the main lever against oom. None
    # disables chunking. lower this if you still oom, raise it for speed.
    encoder_chunk_size: int | None = 32

    # projector (DINO head) discarded after training. K is the prototype count
    # the cross-entropy operates over.
    projector_hidden: int = 512
    projector_bottleneck: int = 256
    projector_k: int = 512

    # ------------------------------------------------------------------
    # DINO
    # ------------------------------------------------------------------
    # teacher softmax temperature (sharp). lower -> more confident targets.
    teacher_temp_start: float = 0.04
    teacher_temp_end: float = 0.07
    temp_warmup_frac: float = 0.3
    # student softmax temperature (softer). asymmetry vs teacher drives learning.
    student_temp: float = 0.1
    # EMA momentum schedule for teacher weights. cosine from start to end.
    ema_momentum_start: float = 0.996
    ema_momentum_end: float = 1.0
    # center buffer momentum. prevents one prototype from dominating.
    center_momentum: float = 0.9

    # ------------------------------------------------------------------
    # adversarial dataset classifier
    # ------------------------------------------------------------------
    # Trains a small classifier to predict dataset_idx from encoder features,
    # while a gradient reversal layer sends the opposite gradient into the
    # encoder. This discourages dataset identity leakage in the representation.
    dataset_adversary_enabled: bool = False
    dataset_adversary_weight: float = 0.1
    dataset_adversary_grl_start: float = 0.0
    dataset_adversary_grl_scale: float = 1.0
    dataset_adversary_hidden: int = 256
    dataset_adversary_dropout: float = 0.1

    # ------------------------------------------------------------------
    # masked patch autoencoder
    # ------------------------------------------------------------------
    # Auxiliary reconstruction loss on global crops. The student encoder sees
    # a patch-masked crop; the training-only decoder predicts only masked
    # patch contents. The encoder is still what ships downstream.
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
    # Distillation-based masking objective: student predicts teacher's soft
    # token distribution on masked global crops. Uses same teacher/student
    # architecture and centering mechanism as DINO but with masked input and
    # cross-entropy loss. iBOT and MAE are both masking objectives and are
    # mutually exclusive: when iBOT is on, MAE is force-disabled (see
    # __post_init__) so only one masked encoder forward runs per step.
    ibot_enabled: bool = False
    ibot_p: float = 0.5
    ibot_weight: float = 1.0
    ibot_mask_ratio: tuple = (0.1, 0.5)
    # samples uniformly from this range each step. 0.1 = 10% of tokens masked, 0.5 = 50%.
    ibot_layer: str = "layer4"
    ibot_block_size: int = 2

    # ------------------------------------------------------------------
    # optimizer / schedule
    # ------------------------------------------------------------------
    base_lr: float = 2.5e-4
    weight_decay_start: float = 0.04
    weight_decay_end: float = 0.4
    grad_clip_norm: float = 1.0
    # fraction of total_steps spent in linear warmup. rest is cosine decay.
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

    # ------------------------------------------------------------------
    # io
    # ------------------------------------------------------------------
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # launch a tensorboard server alongside training so a run is viewable
    # without a separate `tensorboard --logdir ...` invocation. it serves the
    # root log_dir (all runs), binds all interfaces so it's reachable from the
    # compute node, and is torn down when training exits.
    tensorboard_enabled: bool = True
    tensorboard_port: int = 6006

    def __post_init__(self) -> None:
        # iBOT and MAE are redundant masking objectives that both add a masked
        # encoder forward and compete on the encoder. keep only one: iBOT wins
        # since it's the stronger representation signal. resolving it here means
        # the serialized config and all downstream reads see the effective state.
        if self.ibot_enabled and self.mae_enabled:
            print(
                "[config] ibot_enabled and mae_enabled both set; disabling MAE "
                "(iBOT takes precedence as the primary masking objective)"
            )
            self.mae_enabled = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """build a Config from a yaml file.

        unspecified keys keep their python defaults so this file does not
        need to enumerate every knob.
        """
        import yaml

        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"top-level yaml in {path} must be a mapping")
        return _build_dataclass(cls, raw)

    def to_yaml(self, path: str | Path) -> None:
        """serialize this config to yaml. tuples are written as lists."""
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(_to_plain(self), f, sort_keys=False)


def _to_plain(obj: Any) -> Any:
    """recursively turn dataclasses/tuples into yaml-safe primitives."""
    if is_dataclass(obj):
        return {k: _to_plain(v) for k, v in asdict(obj).items()}
    if isinstance(obj, tuple):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    return obj


def _build_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """instantiate a dataclass from a plain dict, recursing into nested dataclasses."""
    field_names = {f.name for f in fields(cls)}
    unknown = set(data) - field_names
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
