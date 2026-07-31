#!/bin/bash
# LSF job body for a WS-DINO training run from scratch (step 0).
#
# do not submit this directly. launch it through launch.sh, which assigns the
# per-run name, snapshots the config, and overrides the -J/-oo/-eo directives
# so concurrent runs stay separate:
#
#   cd /groups/sgro/sgrolab/Ankit/predicty
#   ./launch.sh <run_name>
#
# once the first checkpoint exists (after ckpt_interval steps), resume with
#   ./launch.sh <run_name> --resume

# ---------------------------------------------------------------------------
# LSF directives
# ---------------------------------------------------------------------------
# -J/-oo/-eo are overridden per-run by launch.sh; the values here are only the
# fallback for a raw `bsub < submit_train.sh` (run name defaults to 'default').
#BSUB -J predicty_train              # job name (shows in bjobs)
#BSUB -q gpu_a100                   # A100 queue: 12 slots/GPU, 40GB ram/slot, 141GB VRAM
#BSUB -gpu "num=1"                   # request 1 GPU
#BSUB -n 12                           # 8 slots = 8 cores + 320 GB ram (slot-implied)
#BSUB -W 72:00                       # 72 hour walltime
#BSUB -oo logs/lsf/default/train.out # stdout log (overwritten on each submit; -oo not -o)
#BSUB -eo logs/lsf/default/train.err # stderr log (overwritten on each submit)

# ---------------------------------------------------------------------------
# script body
# ---------------------------------------------------------------------------
set -euo pipefail

# run name drives every output path. launch.sh exports it; a raw bsub falls
# back to 'default' so an un-namespaced submit still works (but two of them
# would collide, so prefer launch.sh).
RUN_NAME="${RUN_NAME:-default}"
# each run reads the config snapshot launch.sh froze at submit time, not the
# live configs/train.yaml. this is what lets you edit train.yaml and launch the
# next run while this one is still queued.
config_path="checkpoints/${RUN_NAME}/config.yaml"
lsf_dir="logs/lsf/${RUN_NAME}"

# LSF needs the log directory to exist before it can write into it.
mkdir -p "${lsf_dir}"

if [ ! -f "${config_path}" ]; then
    echo "ERROR: ${config_path} not found. launch via ./launch.sh ${RUN_NAME}" >&2
    exit 1
fi

# guard against oversubscribing the slot allocation. every dataloader worker is
# its own process and the main process needs a core too, so num_workers must be
# strictly below the slot count. LSB_DJOB_NUMPROC is what LSF actually granted,
# counted in physical cores -- do not size against /proc/cpuinfo or
# os.cpu_count(), which report the node's logical count (2x, hyperthreading) and
# would silently let e.g. 22 workers land on a 16-slot allocation.
# awk (not grep) so a config without the key yields empty instead of tripping
# `set -o pipefail`.
n_workers=$(awk '/^[[:space:]]*num_workers:/ {print $2; exit}' "${config_path}")
if [ -n "${n_workers:-}" ] && [ -n "${LSB_DJOB_NUMPROC:-}" ]; then
    if [ "${n_workers}" -ge "${LSB_DJOB_NUMPROC}" ]; then
        echo "ERROR: num_workers=${n_workers} in ${config_path} needs at least" >&2
        echo "       $((n_workers + 1)) slots, but LSF granted ${LSB_DJOB_NUMPROC}." >&2
        echo "       lower num_workers or raise '#BSUB -n' before resubmitting." >&2
        exit 1
    fi
    echo "[submit] slots=${LSB_DJOB_NUMPROC}  num_workers=${n_workers}  (ok)"
fi

# cap per-process threading at 1. without these, every numerical library detects
# the *node's* core count (48-128 here, and 2x that in logical CPUs because
# hyperthreading is on) rather than the slots LSF granted, and spawns that many
# threads per process -- which then multiplies by num_workers. these workers do
# nfs reads and blosc decompression, no BLAS, so 1 apiece is right.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TBB_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# let the caching allocator grow segments instead of fragmenting into fixed-size
# blocks it can't reuse. matters on a 24GB L4, where the 3D stem's transient
# activations (~84MB per crop for one 64ch x 10z x 256^2 map, x encoder_chunk_size
# crops at once) leave little slack -- fragmentation alone can be the difference
# between fitting and an OOM. harmless on an 80GB A100.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# node-local scratch for any temp files (matplotlib cache, dask spills, etc.)
# never use /tmp on compute nodes per cluster policy.
export TMPDIR=/scratch/$USER/$LSB_JOBID
mkdir -p $TMPDIR

# activate the conda env. verify the source path with `which conda` on a
# login node if this fails (path may differ from ~/miniconda3 on the cluster).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate predicty

# spin up a background nvidia-smi sampler. polls every 30s and writes CSV
# rows to the run's gpu.out so we can confirm the GPU is actually doing work.
# the EXIT trap kills it whenever this shell exits (normal end, error, bkill).
nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv \
    --loop=30 \
    > "${lsf_dir}/gpu.out" 2>&1 &
GPU_SAMPLER_PID=$!
trap 'kill ${GPU_SAMPLER_PID} 2>/dev/null || true' EXIT

# launch training. --run-name namespaces checkpoints (-> checkpoints/<run>/)
# and tensorboard logs (-> logs/<run>/).
python train.py --config "${config_path}" --run-name "${RUN_NAME}"

# clean up node-local scratch on the way out.
rm -rf $TMPDIR