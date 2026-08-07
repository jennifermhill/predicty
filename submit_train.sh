#!/bin/bash
# LSF job body for a WS-DINO training run from scratch (step 0).
#
# do not submit this file directly. launch the file through launch.sh.
# launch.sh assigns the per-run name, snapshots the config, and overrides the
# -J/-oo/-eo directives, so concurrent runs stay separate:
#
#   cd /groups/sgro/sgrolab/Aran/predicty      # the checkout location of this repo
#   ./launch.sh <run_name>
#
# after the first checkpoint exists (after ckpt_interval steps), resume with
#   ./launch.sh <run_name> --resume

# ---------------------------------------------------------------------------
# LSF directives
# ---------------------------------------------------------------------------
# launch.sh overrides -J/-oo/-eo per run. the values here are only the fallback
# for a raw `bsub < submit_train.sh` (the run name defaults to 'default').
#BSUB -J predicty_train              # job name. bjobs shows this name.
#BSUB -q gpu_a100                    # A100 queue: 12 slots/GPU, 40GB ram/slot, 80GB VRAM
#BSUB -gpu "num=1"                   # request 1 GPU
# 24 slots = 24 cores + 960 GB ram (slot-implied), for num_workers: 22 + main.
# NOTE the count is 2x the queue's natural 12 slots/GPU ratio. one job
# therefore holds the slots of two GPUs on a 4-GPU/48-slot A100 node, but uses
# one GPU. two concurrent runs then take the whole node and strand 2 A100s.
# the cost is acceptable for a short sweep. drop to -n 12 / num_workers 10 for
# a long job.
#BSUB -n 24
#BSUB -W 72:00                       # 72 hour walltime
#BSUB -oo logs/lsf/default/train.out # stdout log. -oo, not -o, so LSF overwrites the log on each submit.
#BSUB -eo logs/lsf/default/train.err # stderr log. LSF overwrites the log on each submit.

# ---------------------------------------------------------------------------
# script body
# ---------------------------------------------------------------------------
set -euo pipefail

# the run name drives every output path. launch.sh exports the name, and a raw
# bsub falls back to 'default'. an un-namespaced submit therefore still works,
# but two such submits collide. prefer launch.sh.
RUN_NAME="${RUN_NAME:-default}"
# each run reads the config snapshot that launch.sh froze at submit time, not
# the live configs/train.yaml. the snapshot lets you edit train.yaml and launch
# the next run although this run is still in the queue.
config_path="checkpoints/${RUN_NAME}/config.yaml"
lsf_dir="logs/lsf/${RUN_NAME}"

# LSF needs the log directory before LSF can write into the directory.
mkdir -p "${lsf_dir}"

if [ ! -f "${config_path}" ]; then
    echo "ERROR: ${config_path} not found. launch via ./launch.sh ${RUN_NAME}" >&2
    exit 1
fi

# guard against an oversubscription of the slot allocation. every dataloader
# worker is its own process, and the main process also needs a core, so
# num_workers must stay strictly below the slot count. LSB_DJOB_NUMPROC is the
# count that LSF granted, in physical cores. warning: do not size against
# /proc/cpuinfo or os.cpu_count(), because they report the node's logical count
# (2x, hyperthreading) and silently permit, for example, 22 workers on a
# 16-slot allocation.
# use awk, not grep, so a config without the key gives an empty string and does
# not trip `set -o pipefail`.
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

# cap the thread count per process at 1. without these variables, every
# numerical library detects the *node's* core count (48-128 here, and 2x that
# in logical CPUs because hyperthreading is on) instead of the slots that LSF
# granted. the library then spawns that many threads per process, and the total
# multiplies by num_workers. these workers do nfs reads and blosc
# decompression, not BLAS, so 1 thread each is correct.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TBB_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# let the caching allocator grow its segments. without the variable the
# allocator fragments into fixed-size blocks that it cannot reuse. the setting
# matters on a 24GB L4, where the 3D stem's transient activations leave little
# slack (~84MB per crop for one 64ch x 10z x 256^2 map, x encoder_chunk_size
# crops at once). fragmentation alone can decide between a fit and an OOM. the
# setting is harmless on an 80GB A100.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# warning: cluster policy prohibits /tmp on a compute node.
# node-local scratch for any temp file (matplotlib cache, dask spills, and so
# on).
export TMPDIR=/scratch/$USER/$LSB_JOBID
mkdir -p $TMPDIR

# activate the conda env. if the activation fails, verify the source path with
# `which conda` on a login node. the path can differ from ~/miniconda3 on the
# cluster.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate predicty

# start a background nvidia-smi sampler. the sampler polls every 30s and writes
# CSV rows to the run's gpu.out, so you can confirm that the GPU does real
# work. the EXIT trap kills the sampler when this shell exits (normal end,
# error, bkill).
nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv \
    --loop=30 \
    > "${lsf_dir}/gpu.out" 2>&1 &
GPU_SAMPLER_PID=$!
trap 'kill ${GPU_SAMPLER_PID} 2>/dev/null || true' EXIT

# launch the training. --run-name namespaces the checkpoints
# (-> checkpoints/<run>/) and the tensorboard logs (-> logs/<run>/).
python train.py --config "${config_path}" --run-name "${RUN_NAME}"

# clean up the node-local scratch at the end.
rm -rf $TMPDIR