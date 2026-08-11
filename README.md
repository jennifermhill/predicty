# predicty

This repository does self-supervised representation learning for cell-movement
behavior in 3D microscopy videos of *Dictyostelium*.

The goal is a single artifact: an **encoder**. The encoder maps a
high-dimensional cell crop to a compact embedding. A cell crop is a small 3D
sub-volume from a microscopy video. The encoder trains **without labels**, so
that crops with similar behavior land near each other. You then train a small
linear classifier on top of the frozen encoder. The classifier recognizes
behaviors: streaming, clumping, aggregation and single cells. Everything else in
this repo is scaffolding that trains and evaluates that encoder.

The training method is **WS-DINO** (weakly-supervised DINO). WS-DINO is a
student/teacher self-supervised setup. The two "views" of a scene are the same
field of view at two nearby timepoints (`t1`, `t2`), plus augmentations. The
teacher is an EMA of the student. Centering and sharpening prevent collapse. An
auxiliary **iBOT** masked-image-modeling loss and a **dataset adversary** run
with WS-DINO.

**New here?** Read [Setup](#setup) and [Training](#training) to run something.
Then read [How to read the metrics](#how-to-read-the-metrics) before you believe
any number you produce. That section is the most important part of this
document. Several measurements on this dataset look natural but mislead you. The
reasons are not obvious.

---

## Contents

- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Data](#data)
- [Training](#training)
- [Evaluation](#evaluation)
- [How to read the metrics](#how-to-read-the-metrics)
- [Open questions and suggested next steps](#open-questions-and-suggested-next-steps)
- [Claims that were made and later retracted](#claims-that-were-made-and-later-retracted)
- [Environment and cluster notes](#environment-and-cluster-notes)
- [Tests](#tests)
- [Conventions](#conventions)

---

## Repository layout

```
predicty/
├── README.md                 you are here
├── train.py                  ← training entry point
├── train_utils.py            schedules, checkpoint I/O, device helpers used by train.py
├── benchmark_dataloader.py   dataloader smoke test + throughput benchmark (NOT training)
│
├── configs/
│   ├── config.py             Config dataclass (every knob + its default) and yaml loader
│   └── train.yaml            the config you actually edit for a run
│
├── data/                     the data pipeline (zarr -> augmented crop pairs)
│   ├── dataset.py            ZarrCellDataset / MultiZarrDataset (the core loader)
│   ├── picker.py             interactive/CLI zarr chooser for single-zarr runs
│   ├── metadata.py           reads training_data_index.xlsx (per-experiment quirks)
│   ├── balancing.py          equal per-dataset representation within a batch
│   ├── spatial_regions.py    spatial-region holdout for leakage-controlled probes
│   └── augmentations.py      semantically-preserving crop augmentations
│
├── models/
│   ├── encoder.py            HybridEncoder: 3D stem collapses z, then 2D ResNet-50
│   ├── dino.py               DINO head, student/teacher wrapper, DINO + iBOT losses
│   └── ibot.py               iBOT masking helpers + MaskedPatchAutoencoder (MAE decoder)
│
├── eval/                     everything for inspecting a trained encoder (see below)
├── tests/                    pytest suite for the non-GPU pieces
├── notebooks/visualize.ipynb scratch visualization notebook (stale, see below)
│
├── launch.sh                 submit a training run to the LSF cluster
├── launch_eval.sh            submit the evaluation pipeline to the LSF cluster
├── submit_train.sh           LSF job body: fresh run
├── submit_resume.sh          LSF job body: resume from checkpoint
├── submit_eval.sh            LSF job body: evaluation pipeline
│
├── index.jsonl               behavior/dataset annotations (one JSON object per line)
├── training_data_index.xlsx  per-experiment metadata (frame interval, z limits)
├── environment.yaml          full conda export (Linux; exact pinned build strings)
└── requirements.txt          portable pip requirements
```

Note that the name `models/ibot.py` is wrong. Only ~20 lines of the file are
iBOT mask helpers, and the rest is the MAE `MaskedPatchAutoencoder`.
`Config.__post_init__` force-disables MAE whenever iBOT is on, so that decoder
never trains in the current configuration. The module must still exist, because
`models/dino.py` imports the iBOT mask helpers from it, and `eval/decoder.py`
imports `_reconstruction_metrics` and `_trunc_normal_`. (`eval/decoder.py`
re-implements patchify and unpatchify as free functions instead of an import.)

---

## Setup

The project targets Python 3.11 with PyTorch and CUDA on a Linux GPU node.

**Option A -- conda (an exact match to the training cluster):**

```bash
conda env create -f environment.yaml
conda activate predicty
```

`environment.yaml` is a full `conda env export` with pinned Linux build strings.
The file reproduces the cluster environment, but the file is not portable to
macOS or Windows.

**Option B -- pip (portable):**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Training requires a CUDA-capable GPU. Evaluation of a saved checkpoint can run
on a CPU, but a CPU is slow.

---

## Data

- The inputs are **zarr** arrays of microscopy video, with the index
  `(t, c, z, y, x)`. The data is not uniform across experiments.
  `training_data_index.xlsx` holds the per-experiment quirks: the frame interval
  in minutes and the usable `z` range. `data/metadata.py` reads the spreadsheet.
  Check that spreadsheet before you change loader code.
- The z axis has a lower resolution than x and y (anisotropic). The loader and
  the crop geometry account for the difference.
- **The temporal pair:** each training sample is one field of view at two
  timepoints, `t1` and `t2`. The gap between the two timepoints is
  `target_gap_minutes`. The loader rounds the gap to a whole frame gap with the
  per-experiment interval. The temporal pair is the weak label.
- **Annotations** (`index.jsonl`, 500 records) serve the downstream *probes*.
  Through `rep_alpha`, the annotations also serve a fraction of the pre-training
  samples.
  See [rep_alpha injects probe crops into pre-training](#1-rep_alpha-injects-probe-crops-into-pre-training).

  Raw counts across all 500: `single_cells` 171, `clumps` 155, `streams` 66,
  `aggregate` 64, `empty` 37, `slug` 7. After the exclusions below, the probe
  sees **429**: `single_cells` 161, `clumps` 148, `streams` 64, `aggregate` 56.

### What gets dropped from the annotation set, and why

`index.jsonl` spans **18** distinct zarr volumes, but training uses only the
**9** volumes in `configs/train.yaml`. The other 9 volumes are the pool for
`--restrict unseen`. The code applies three exclusions before any probe runs:

| Dropped | Where | Why |
| --- | --- | --- |
| `label == "empty"` | both behavior evals | not a behavior class |
| `label == "slug"` | both behavior evals | n=7 across the whole set. Too few to fit or to score |
| **all of `mixin49`** | `probe_behavior.py`, `knn_behavior.py` | **a bad acquisition, not part of the project's data.** Training never used it. The code drops its 31 annotations, so they cannot leak into the unseen pool. |

The mixin49 path is a **hardcoded string literal in both files**. If you change
one file but not the other, the two evals silently use different folds. Then
their numbers are not comparable. If you add or retire a bad volume, grep for
`mixin49` and fix every hit.

### Frame sizes vary by more than 10× in area

```
512x512   mixin57        977x977   mixin41-B2     1442x1441 mixin54
977x1439  mixin44        978x975/976/977 mixin41/42   1905-1909 x 1438-1443  mixin45/52/62/65/66
980x976   mixin43        1024x1024 mixin64 (1x1 binning!)
```

512×512 is 0.26 MP and 1909×1443 is 2.75 MP: **10.5× in area, 3.7× linear.**
Without the one 512 px frame, the area ratio is still 2.9×.

**Warning: code that computes region geometry must compute it per frame, not
once from `rows[0]`.** That mistake caused a real bug before.

### The 9 "datasets" are really 7 imaging sessions

`mixin41` (B2=NC28.1, B3=NC4) and `mixin42` (B2=NC4, B3=NC105.1) are each **one
plate, one day, one session, two wells**. The two wells differ by *strain*,
which is biology, not a batch effect. `balance_datasets: true` therefore gives
those two plates double weight in every batch.

The merge key that works is the path from the `cryolite_mixin_test*` directory
down to the `.zarr`, without the `Field_*` components. The key collapses 9 → 7
and keeps mixin65 day1 and day2 apart. `eval/probe_dataset.py` implements the
key.

---

## Training

All settings live in `configs/train.yaml`, which has inline documentation. The
`Config` dataclass in `configs/config.py` is the authoritative list of every
field and its default. Pick **either** `zarr_path` (a single dataset) **or**
`zarr_paths` (a mix of datasets).

`configs/train.yaml` in the repository is the current reference configuration. A
new experiment must change **one line** from that configuration, and you must
compare the experiment back to it. Multi-lever runs here produced
uninterpretable results more than once.

Note that `num_workers` changes the sample *order*, because each worker seeds
from `seed + worker_id`. If you change `num_workers`, a rerun is not
bit-identical, although the objective does not change.

**On the LSF cluster (recommended):**

```bash
./launch.sh my_run             # fresh run from step 0
./launch.sh my_run --resume    # resume from checkpoints/my_run/latest.pt
```

`launch.sh` gives each run a name. At submit time `launch.sh` **snapshots
`configs/train.yaml`** into `checkpoints/<run>/config.yaml`. An edit to the yaml
for the next run therefore never disturbs a run already in the queue.
`launch.sh` also keeps all outputs in per-run directories:

```
checkpoints/<run>/   checkpoints + the frozen config snapshot
logs/<run>/          tensorboard event files
logs/lsf/<run>/      LSF stdout/stderr + a periodic nvidia-smi GPU sampler
```

**Directly (local smoke test or non-LSF machine):**

```bash
python train.py --config configs/train.yaml --run-name my_run
python train.py --config configs/train.yaml --device cpu   # tiny CPU smoke test
```

**Monitor:**

```bash
tensorboard --logdir logs
```

Watch `collapse/embedding_var`, which must stay well off the floor. Watch
`dataset_adversary/acc`, which must trend toward chance. Also watch
`teacher_entropy / ln(projector_k)`. Healthy runs sit near **0.71**. A climb
toward 1.0 is the uniform-collapse signature. The signature appears by ~step
1000, so stop the run at that step and save the wall time.

**Before you commit a GPU,** check the data path and its throughput:

```bash
python benchmark_dataloader.py --config configs/train.yaml
```

### What gets trained

The training optimizes the encoder against a DINO self-distillation loss. The
config also toggles these auxiliary objectives:

- **DINO** (always): student/teacher self-distillation over the crop views.
- **iBOT** (`ibot_enabled`): block-wise masked feature reconstruction. On by
  default. **MAE** (`mae_enabled`) is the mutually-exclusive alternative. The
  code force-disables MAE whenever iBOT is on, so MAE is dead in the current
  configuration.
- **Dataset adversary** (`dataset_adversary_enabled`, multi-zarr only): a
  gradient-reversal classifier that pushes the encoder to drop dataset identity
  (batch-effect removal). Keep its pressure weak until DINO is stable.
- **KoLeo** (`koleo_enabled`): a uniformity regularizer on the embedding batch.

After training the code discards the projector, the decoder and the adversary.
Only the encoder ships.

### Throughput is NFS-I/O bound, not compute bound

Zarr chunks are full XY planes `(1,1,1,978,976)`. A 256²×10z global slices 10
whole planes (~19 MB) but uses only ~1.3 MB. Janelia has no parallel filesystem,
only plain NFS, so the read cost dominates.

The largest win is already in the code: `local_time_endpoints` (default true)
draws the timepoint of each local crop from `{t1, t2}`, not uniformly over
`[t1, t2]`. Every local is then a numpy slice of an array already in memory, not
a separate zarr read. **50 → 20 plane reads per sample.** Cross-time invariance
does not change, because the code assigns `local_parent` independently of the
timepoint of the local. The cost is the loss of intermediate timepoints. To
ablate, set the flag false.

Measurements: a single run alone gives 81% GPU util, 0.198 steps/s and 41 GB of
80 GB VRAM. The 41 GB is why the 24 GB `gpu_l4` queue is not an option. Four
concurrent runs under the *old* sampler give ~35% util and ~0.10 steps/s each.
Under the current sampler, two concurrent runs reach **0.26 steps/s each**. The
aggregate is ~2.6× the old rate.

Remaining ideas, cheapest first: (1) a per-worker LRU cache with the key
`(t, z-plane)`, which removes the re-reads of the intensity gate on rejected
attempts; (2) a one-off rechunk of channel 0 to `(1,1,10,256,256)` on `/nrs`
(~1 TB). **The old "cap concurrency at 2" rule is now unmeasured, not wrong.**
Someone set the rule when a sample cost 50 plane reads. Re-measure the rule
before you assume that it binds.

---

## Evaluation

Run these scripts against a checkpoint after training. Most scripts take
`--config <path> --checkpoint checkpoints/<run>/latest.pt`. Always pass the
**frozen config of the run**, not `configs/train.yaml`.

| Script | What it does |
| --- | --- |
| `latent_space.py` | Embeds sampled crops, applies PCA, UMAP and t-SNE to them, and caches the results under `latent_space/<run>_*/`. **Run this script first.** Several tools reuse its cache. Note that `embeddings.npz` lands at the run-dir root, but the script writes `metadata.csv` **per method**, at `<run_dir>/<method>/metadata.csv`. The `--metadata` flag on the viewers and on `latent_traversal_video.py` wants the `pca/` copy specifically. |
| `probe_behavior.py` | Linear probe for behavior identity on the frozen encoder. The probe must **succeed**. `--restrict seen\|unseen\|all` limits the probe to experiments that the encoder did or did not train on. |
| `probe_dataset.py` | Linear probe for dataset identity. The probe must **fail**. `--max-contamination` filters probe crops by training-pixel overlap. |
| `knn_behavior.py` | Non-parametric k-NN behavior eval, plus label purity and batch purity. |
| `wasserstein.py` / `probe_wasserstein.py` | Sliced-Wasserstein batch-effect metric between per-dataset embedding clouds. The first file is the reusable numpy-only module. The second file runs the module over a cached `embeddings.npz`. |
| `gradcam.py` + `run_gradcam.py` + `visualize_gradcam.py` | Grad-CAM / 3D LayerCAM attribution (library, CLI runner, plots). |
| `expected_gradients.py` + `run_eg.py` | Expected-Gradients input-space attribution (library + CLI runner). Signed, unlike Grad-CAM. |
| `decoder.py` | Trains a post-hoc latent→voxel decoder on a frozen encoder. The decoder can then turn latents back into images. |
| `interactive_latent_viewer.py` | Click or lasso points in a 2D latent map to see the raw crops behind them. The viewer has a Generate mode, which decodes interpolated latents with `decoder.py`. |
| `interactive_latent_viewer_3d.py` | 3D counterpart, reads `coord_x3/y3/z3` from `metadata.csv`. |
| `latent_traversal_video.py` | Sweeps PC1 and renders a video of the crops along that axis. |
| `visualize_mae_mask.py` | Visualizes the masked patches that the MAE loss reconstructs. Only meaningful for an MAE-trained checkpoint. |
| `pca_utils.py` | Exact float64 PCA, shared by the tools above. Numpy only. |
| `evaluation_pipeline.py` | **Orchestrator.** Runs latent space → dataset probe → behavior probe (twice) → kNN → Grad-CAM → EG. Skips every run that already has an evaluation. |

**Use the pipeline.** The pipeline applies the leakage-controlled settings by
default. Those settings make the numbers comparable across runs:

```bash
./launch_eval.sh my_run other_run          # on the cluster
python eval/evaluation_pipeline.py --config checkpoints/my_run/config.yaml --run-names my_run
```

Manual end-to-end for one run:

```bash
python eval/latent_space.py   --config checkpoints/my_run/config.yaml --checkpoint checkpoints/my_run/latest.pt
python eval/probe_dataset.py  --config checkpoints/my_run/config.yaml --checkpoint checkpoints/my_run/latest.pt --max-contamination 0.0
python eval/probe_behavior.py --config checkpoints/my_run/config.yaml --checkpoint checkpoints/my_run/latest.pt --restrict unseen
```

---

## How to read the metrics

Most of the work of the final month went here, not into new models. The short
version: several measurements on this dataset look obvious but mislead you. The
reasons are structural, not statistical.

### 1. `rep_alpha` injects probe crops into pre-training

With `overrepresent_classes: true` and `rep_alpha: 0.15`, the *training* sampler
pulls ~15% of the samples straight from annotated crop coordinates. That branch
also **bypasses `_spatial_regions` entirely**, because the branch never calls
`_sample_global_offset`. The encoder therefore sees probe crops at their exact
positions. The behavior probe then scores the encoder on those same crops.

`rep_alpha` is now 0.06, and `eval/probe_dataset.py` forces `overrepresent_classes`
off. The injection branch draws only annotations that belong to the nine
*training* zarrs. `--restrict unseen` is therefore immune to the confound, but
the full-set probe is not.

### 2. The spatial holdout partitions TOP-LEFT positions, not image area

`region_bounds_for_top_left` splits the `frame − crop + 1` legal top-left
positions into a grid. But a crop occupies `global_crop_size` px *beyond* its
top-left. The **pixel footprint** of each region is therefore its band plus
`crop − 1` px, and adjacent bands overlap by exactly that much.

On a 978 px frame with crop 256: bands `[0,241)/[241,482)/[482,723)`, footprints
`[0,496)/[241,737)/[482,978)`. Under the old 3×3 holdout, the footprints of the 6
training regions covered **81.8% of the frame**. Training could also reach a mean
**77.7%** of the pixels of each "held-out" probe crop. With ~71k crops per
experiment, a reachable pixel is a seen pixel.

The contamination is **frame-size dependent, not a constant offset**: 78% at
978 px, 68% at 977×1439, 48% at 1906 px, 97% at 512 px.

**The fix has two halves, and you need both halves:**

- **A 2×2 grid that holds out one tile** (`spatial_region_grid_y/x: 2`,
  `spatial_holdout_region_count: 1`) makes a zero-contamination crop *possible*.
  A 3×3 grid gives 240 px tiles against a 256 px crop. Even position 0 spills
  over, so the clean count is exactly **zero**. A 2×2 grid gives 361 px tiles.
  Positions 0–105 per axis clear the boundary: 106² = 11,236 clean positions per
  977 px frame.
- **`--max-contamination 0.0`** *selects* those positions. If you run 2×2
  without the filter, you get the same contaminated crops as before, with only a
  different mean.

The count **MUST** be 1 on a 2×2. All four tiles are pairwise
Chebyshev-adjacent, so `count: 2` raises out of `select_nonadjacent_regions`.

| | clean probe positions | training positions retained |
|---|---|---|
| 3×3, hold [0,2,8] | **373,906**, but **0** on six of nine frames | 66.7% |
| 2×2, hold [0] | **670,578**, ≥**11,130** on *every* frame | **75.0%** |

Trade-off: all probe crops now come from one ~361×361 corner, so a location bias
exists. A holdout of the diagonal pair `[0,3]` gives two corners, but the
holdout drops training to 50% of the positions.

**`max_train_footprint_overlap` is permanent, not scaffolding.** An earlier note
in the code called the field temporary. The note said that the end goal was a
repartition by area and a deletion of the field. That is backwards. No pure
partition of top-left space can ever do better, because the boundary is a hard
edge and crops straddle it by construction.

The only alternative pushes the *training* top-lefts back from the boundary, and
that alternative costs far more. Training occupies three of four tiles, and a
256 px gutter cuts the adjacent training band from 361 to ~105 positions per
axis. A filter on the probe side throws away ~92% of one tile, and you need only
a few thousand crops from that tile. A filter on the training side throws away
the data that the encoder learns from.

**Still unfixable:** a 512×512 frame (mixin57) has legal top-lefts of only
257 px. A 2×2 tile is then ~128 px, and the 256 px crop swallows the tile whole.
mixin57 is not in `zarr_paths` today. If you ever add mixin57, it needs a smaller
`global_crop_size`, or you must exclude it from the probe.

### 3. The intensity gate needs the right reference population

At first, `--max-contamination 0.0` crashed the dataset probe outright. Two
independent defects caused the crash. `data/dataset.py` now fixes both defects:

1. **The retry loop could not escape a bad anchor.** The code drew `(t1, z)`
   once. The loop then redrew only `(y, x)`, from the same fixed allowed-offset
   set every time. If the best allowed position fell short of the threshold, all
   1000 attempts *had* to fail. The failure was deterministic, not bad luck. The
   loop now redraws the anchor too, up to `intensity_anchor_attempts`
   (default 8).
2. **The gate compared against the wrong population.** The threshold was
   `intensity_min_frac × mean(whole frame slab)`. But the cap confines the
   sampler to one ~106×106-position corner, and corners are dimmer. A dim corner
   fails that cross-population comparison at *every* anchor. The margin of the
   failure also differs per dataset, and a gate with that defect silently biases
   the per-dataset probe counts. `_get_frame_mean` now takes an optional bbox,
   so the gate judges "unusually empty" inside the population that the sampler
   can reach.

Training does not change: the bbox is `None` whenever
`max_train_footprint_overlap` has no value.

### 4. Compare against the majority baseline, not uniform chance

The label sets here are imbalanced. The probe printed "well above 1/n_classes"
for runs that **lose to a constant predictor**. A real example: a
`--restrict unseen` probe scored val_acc 46.54% against a 47.6% majority
baseline. The score is 1.06 points *below* a model that always guesses the same
class. Every fold except one sat at or below its own baseline. Every log before
2026-08-04 called that result "well above chance -- this is the desired outcome."

The imbalance does not affect `macro_F1` and `balanced_acc`, which is the
argument for those two as the comparison metrics. `probe_behavior.py` now
reports against the majority baseline. **`probe_dataset.py` and
`knn_behavior.py` still carry their own uniform-chance `interpret()`, and nobody
fixed them.** See the open items.

### 5. Resolvability: use a t critical value, and be honest about what varies

A flat 2× standard error is the large-sample limit. At 5 folds the difference has
~4 degrees of freedom, where the two-sided 95% value is **2.776**. Several older
logs concluded that "the ordering is real". Those logs used 2× against an
observed ratio of 2.22, so the ordering was not resolvable.

More important: **the code groups the folds by experiment, so the folds
correlate and the standard error is optimistic.** Treat the standard error as a
floor on the uncertainty.

### 6. The dominant variance is *which experiments are held out*

A representative `--restrict unseen` probe, per class:

| class | pooled F1 | fold mean | fold sd | val n | per-fold |
|---|---|---|---|---|---|
| aggregate | 0.325 | 0.310 | 0.059 | 23 | [0.32, 0.33, 0.39, 0.28, 0.23] |
| clumps | 0.434 | 0.405 | 0.128 | 59 | [0.33, 0.60, 0.25, 0.42, 0.42] |
| single_cells | 0.607 | 0.562 | 0.210 | 99 | [0.78, 0.43, 0.78, 0.51, 0.31] |
| streams | 0.328 | 0.256 | 0.226 | 27 | [0.00, 0.62, 0.25, 0.21, 0.20] |

> **Known reporting bug:** `probe_behavior.py` sums the `val n` column over the
> restart seeds. A 3-restart run therefore shows 3× the real support (69/177/
> 297/81, a total of 624 = 3 × 208). The counts above are the corrected counts,
> and they total 208, the actual unseen set. Until someone fixes the bug, divide
> the printed column by the restart count. If you do not divide, the
> thin-support warning reads as if the data were three times larger.

Per-fold macro_F1 runs 0.290–0.495. **The main open scientific question is this:
why does the representation generalise so unevenly across sessions?** The
question dwarfs every per-class story.

That table has two traps. `streams` = 0.00 on fold 1 looks like a finding, but it
is not a finding. The val counts of fold 1 are `[3, 11, 22, 1]`, so the 0.00
comes from **one** misclassified crop. The probe now prints the held-out
experiments of each fold, and the probe warns on any class with val n ≤ 2. A low
fold sd also does not mean stability. `aggregate` sits near its floor, and a
metric that cannot move cannot vary.

### 7. `--restrict` matches by file path, and one experiment appears on both sides

`mixin65` appears on both sides of the seen/unseen split.
`test65_2024-04-16/NC28.1_overnight_day1` trains, and
`.../WS205_overnight_day2` is "unseen". `--restrict` must group by experiment,
not by file. **The defect is still present. Nobody fixed it.**

Also note that the split changed over time. An older log reports 223 seen and
240 unseen. A newer log reports 208 unseen of 429 across 8 experiments. A
comparison that mixes those two numbers is not a matched contrast.

### 8. `--restrict unseen` and the spatial holdout are not substitutes

The two methods measure different things. The spatial holdout gives fresh
*pixels* from the *same* experiments. The spatial holdout is the only way to run
a dataset-ID probe at all, because the label space of that probe **is** the
training experiments. `--restrict unseen` gives fresh *experiments*. Neither
method replaces the other.

### 9. The behavior probe only embeds the t1 frame

`eval/probe_behavior.py` uses `batch["globals"][0]`. `_annotation_tensor` loads
both frames, and the code then discards t2. A comparison of the `t1`, `both` and
`delta` readouts costs no extra I/O. The comparison says whether the
temporal-invariance objective discards usable information. **Do not motivate the
comparison with `streams`.** See the retracted claims below.

### 10. `weak_label` is consumed in exactly one place

`models/dino.py` consumes `weak_label`. `weak_label[:,0]` (dataset_idx) is the
**target of the dataset adversary**, through the gradient reversal.
`weak_label` plays no part in the positive-pair groups, and the forward pass
never reads `frame_gap` (column 1).

Consequence: with 9 classes the adversary must separate mixin42-B2 from
mixin42-B3. Those two share a plate, a day and the optics. Through the GRL the
encoder therefore takes an **active penalty for a representation of strain**,
and strain is biology that you want. A 7-class session-merged adversary is a
well-motivated single-lever experiment.

---

## Open questions and suggested next steps

### The one that matters most

**Nobody knows the run-to-run encoder variance on this project.** The effects
that this project chases are ~0.05 on unseen macro-F1. If the encoder noise is
±0.06, no single-lever comparison in the pipeline has an interpretation.
Everything below is worth less than that measurement.

> **Suggested first batch: two concurrent runs from the current baseline,
> compared on `--restrict unseen`.** One run changes a single lever that you
> care about. The other run changes **only `seed`** and gives the error bar that
> every other comparison in this project needs.

If you do the attribution first and the calibration later, you get an answer
faster. But the answer has no interpretation until the calibration arrives.

### Levers to attribute

The current baseline differs from the older configuration with the better score
in three ways. Nobody knows which difference costs what:

1. **`local_time_endpoints`**: the flag removes the intermediate timepoints. The
   loss of information is real. The flag is the prime suspect.
2. **`rep_alpha` 0.15 → 0.06**: 2.5× less oversampling of the annotated,
   behaviour-rich regions. The change is not leakage on the unseen metric, but
   the change is genuine for what the encoder sees. Someone once dropped an A/B,
   because nobody could settle it cleanly. That reasoning was wrong *for the
   unseen metric specifically*, where the confound vanishes. `rep_alpha` is a
   live suspect.
3. **2×2 holdout**: the holdout gives *more* training positions (75% vs 66.7%).
   The holdout probably does not hurt.

### Free, no cluster needed

- **`--restrict seen`**, ~5 min per checkpoint. People usually infer an overfit
  to the training experiments from a seen/unseen reversal. Nobody measures the
  overfit. `--restrict seen` measures it.
- `eval/probe_dataset.py` and `eval/knn_behavior.py` still compare to uniform
  chance, not to the majority baseline. Decide what to report.
- `eval/probe_behavior.py` sums the per-class `val n` over the restart seeds, so
  a 3-restart run reads 3× the true support. Divide the number, or fix the
  script.
- Stratification of the dataset-probe train/val split: the val counts run 155 to
  774. The "high-coordinate strip" split interacts with the capped positions,
  which cluster at the outer corner of each region. That interaction is *why*
  the majority baseline is so high. The interaction and the item above are one
  problem from two sides.
- Re-measure the GPU utilisation and the steps/s, because a sample now costs 20
  plane reads, not 50. Then examine the concurrency cap of 2 again.
- Fix `--restrict` to group by experiment, not by `zarr_path`, so that mixin65
  appears on one side only.

### Needs a training run

- Re-run the best older configuration under the current sampler. Every
  measurement before 2026-08-04 used a different sampler, so those measurements
  do not compare to a measurement from now.
- 7-class session-merged dataset adversary.
- Re-baseline the no-adversary ablation under the current dataset-probe
  protocol.

### Unswept levers

`total_steps: 25000` **in fp16**: the one long run used bf16 and collapsed. The
step count itself is therefore untested, not disproven. `projector_k: 4096`: the
one measurement in that direction says go up, never down. And
`target_gap_minutes` at 25 or 100: **nobody swept it at all**, although it is the
core temporal signal of the entire method.

Do not use bf16 here. bf16 has 8 mantissa bits against the 10 bits of fp16. A
`teacher_temp` of 0.04–0.07 divides the cosine-scale logits and amplifies the
error 14–25×. The teacher then collapses to ~16 effective prototypes against a
healthy ~224.

### Annotations

429 of 500 are usable (see [Data](#data)), and the rarest class is at n=56. More
crops of the rare classes help. But the binding constraint looks like the fold
heterogeneity, not the support, so measure the heterogeneity first.

---

## Claims that were made and later retracted

The list exists so that nobody rediscovers these claims the hard way. Someone
believed each claim, wrote it down, and then disproved it.

- **"All 9 spatial regions appeared in a run's crops because of
  `overrepresent_classes`."** The `overrepresent` bypass is real, but the bypass
  was not the cause. The replacement explanation said that someone disabled the
  spatial holdout in the frozen config of that run. The replacement explanation
  was **also** wrong, because the frozen snapshot configures the holdout in full.
  **The claim is therefore still unexplained.** Anyone who re-opens the claim
  must start from the frozen config, not from either theory.
- **"`streams` is the weak class, because motion can't be seen in one frame."**
  Wrong. `aggregate` is worse in 5 of 6 runs, and the ranking is not resolvable
  anyway.
- **"Rare classes will show larger fold variance."** The opposite is true.
- **"Contamination is roughly a constant offset across runs."** The contamination
  is frame-size dependent, 48%–97%.
- **"`--restrict unseen` is a stronger check that covers the spatial holdout."**
  The two measure different things. Neither one substitutes for the other.
- **"`max_train_footprint_overlap` is temporary scaffolding."** Backwards. A
  repartition cannot remove the need for the field.
- **"2×2 fixes the contamination."** Too strong. The 2×2 grid makes the clean set
  non-empty. The `--max-contamination 0.0` filter selects the clean set.
- **"The worst-to-best per-class gap exceeds 2 se, so the ordering is real."**
  Every log before 2026-08-04 printed that claim. 2× is the large-sample limit.
  At 4 df the value is 2.776, and the observed ratio was 2.22.
- **"val accuracy is well above chance -- this is the desired outcome."** Every
  log before 08-04 also printed that claim, even for folds that scored *below*
  their own majority baseline.
- **"`ibot_weight: 0.2` looks safe and mildly positive."** The weight won
  in-distribution, because the full-set probe and the kNN both went up. But the
  weight lost on held-out experiments *and* on invariance. That combination is an
  overfit signature. The combination is also the clearest demonstration in this
  project of why the unseen probe is the metric that counts. Keep `ibot_weight`
  at 0.1, and see the note on the field in `configs/train.yaml`.
- **"A gap between two runs is established because it exceeds twice the
  probe-restart spread."** Wrong yardstick, and several rankings rested on it.
  The restart spread is the noise from a refit of a linear head on a **fixed**
  encoder. The run-to-run encoder variance comes from two training runs of the
  same config. Nobody measured that variance here, and the effects that this
  project chases can be the same size as that variance.
- **A one-mask-for-all-frames bug** in `training_footprint_overlap`, which used
  the dimensions of `rows[0]`. Someone fixed the bug, but see the frame-size
  warning above.

---

## Environment and cluster notes

- **Cluster is LSF, not Slurm.** Queue `gpu_a100`, `-gpu "num=1"`, 12 slots/GPU.
  Always set `-W`. The 41 GB peak excludes `gpu_l4` (24 GB VRAM).
- `submit_train.sh` requests `-n 24` for `num_workers: 22`. That request is 2×
  the natural 12-slots-per-GPU ratio of the queue. One job therefore holds the
  slots of two GPUs on a 4-GPU node, but the job uses one GPU. The request is
  acceptable for a short sweep. For a long run, drop to `-n 12` and
  `num_workers: 10`.
- **Cluster-specific paths.** The `zarr_paths` field of `configs/train.yaml`, the
  conda activation path in `submit_*.sh`, and the LSF directives are specific to
  the Janelia Sgro-lab setup. Before you run elsewhere, point them at your own
  data, env and scheduler. `launch.sh` assumes `bsub`.
- **Probe extraction rate is dataset-dependent, not a constant.** A behavior
  probe on the `--restrict unseen` set ran at **1.4 samples/s** (208 crops in
  546 s), because those frames are mostly ~1906 px. The ~4.5–6 crops/s figure
  holds for the smaller training zarrs. Budget from the frame sizes, not from a
  remembered rate.
- **Git ignores the run artifacts**, and nobody commits them: `checkpoints/`,
  `logs/`, `latent_space/`, `outputs/`, `mae_masks/`, `annotations/`, `plans/`.
  The artifacts live only on the machine that produced them. ~86 archived runs
  live on the original workstation.
- **Archived configs must stay loadable.** Every eval command takes
  `checkpoints/<run>/config.yaml`. `configs/config.py` raises on unknown keys. If
  you delete a `Config` field, no past run is evaluatable. Add the name to
  `_DEPRECATED_KEYS` in the same change, and never reuse a retired name.
- **`benchmark_dataloader.py` is not a training entry point**, although the file
  sits at the repo root. `train.py` is the training entry point.
- **Two dependency files, two purposes.** `environment.yaml` is an exact conda
  export with Linux build strings, and the export is not portable.
  `requirements.txt` is a portable pip list that a person maintains by hand.
  **Update both files when you change a dependency.** The two files drift apart.
  The symptom is a manual `pip install` after you follow the setup instructions.
- **Regenerate `environment.yaml` on the cluster, never on Windows.** The export
  comes from the Linux GPU node, and its pinned build strings are
  Linux-specific. A re-export from a Windows conda env silently replaces them
  with Windows builds and breaks the cluster reproduction path. On the cluster,
  run: `conda env export | grep -v '^prefix:' > environment.yaml`. The command
  strips the `prefix:` line, because that line records the absolute path of one
  machine. conda ignores the path on create, and the path misleads readers.
- **`notebooks/visualize.ipynb` is stale scratch work.** The notebook cannot run
  top-to-bottom, because nothing assigns `primary`, `is_multi` and
  `sample_iter`. The stored outputs reflect 388 annotations against the 500 of
  today. The notebook also hardcodes a Windows drive-letter path rewrite. Treat
  the notebook as a reference for plot idioms, not as working code.

---

## Tests

```bash
pytest            # or: python -m pytest
```

The tests cover the non-GPU pieces: batches, spatial regions, MAE-mask metrics
and the latent viewer. The tests still import the training stack, so run them
inside the `predicty` environment. There is no `conftest.py` and no pytest
config, so run them from the repo root.

---

## Conventions

- Lowercase comments. No em dashes.
- `snake_case` for functions and variables, `PascalCase` for classes.
- Hyphenated CLI flags (`--run-name`, not `--run_name`).
- Checkpoints always save **both** the encoder state and the full model state,
  so that you can extract the encoder independently of the training scaffolding.
- Do not commit secrets or scratch notes. Git ignores `notes.txt` for that
  reason.
- `df` is the annotations DataFrame in `probe_behavior.main()`. Do not reuse the
  name. A shadow of the name once broke the checkpoint save at the very end of a
  40-minute probe run.
