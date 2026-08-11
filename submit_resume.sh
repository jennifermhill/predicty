#!/bin/bash
# LSF job body that resumes a WS-DINO run from its most recent checkpoint.
#
# do not submit this file directly. launch the file through launch.sh:
#
#   cd /groups/sgro/sgrolab/Aran/predicty      # the checkout location of this repo
#   ./launch.sh <run_name> --resume
#
# checkpoints/<run_name>/latest.pt must exist. train.py saves the file every
# ckpt_interval steps. the job fails loudly if the file is absent. launch a
# fresh run first.

# ---------------------------------------------------------------------------
# LSF directives (the same shape as submit_train.sh)
# ---------------------------------------------------------------------------
# launch.sh overrides -J/-oo/-eo per run. the values here are only the fallback
# for a raw `bsub < submit_resume.sh` (the run name defaults to 'default').
#BSUB -J predicty_resume              # job name. bjobs shows this name.
#BSUB -q gpu_a100                    # A100 queue: 12 slots/GPU, 40GB ram/slot, 80GB VRAM
#BSUB -gpu "num=1"                   # request 1 GPU
# the count must match submit_train.sh, and must exceed the num_workers value
# of the frozen config.
#BSUB -n 24
#BSUB -W 72:00                        # 72 hour walltime
#BSUB -oo logs/lsf/default/resume.out # stdout log. -oo, not -o, so LSF overwrites the log on each submit.
#BSUB -eo logs/lsf/default/resume.err # stderr log. LSF overwrites the log on each submit.

# ---------------------------------------------------------------------------
# script body
# ---------------------------------------------------------------------------
set -euo pipefail

# the run name drives every output path. launch.sh exports the name, and a raw
# bsub falls back to 'default'.
RUN_NAME="${RUN_NAME:-default}"
config_path="checkpoints/${RUN_NAME}/config.yaml"
ckpt_path="checkpoints/${RUN_NAME}/latest.pt"
lsf_dir="logs/lsf/${RUN_NAME}"

# LSF needs the log directory before LSF can write into the directory.
mkdir -p "${lsf_dir}"

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

# fail with a clear error if no checkpoint exists for a resume.
if [ ! -f "${ckpt_path}" ]; then
    echo "ERROR: ${ckpt_path} not found." >&2
    echo "launch a fresh run first: ./launch.sh ${RUN_NAME}" >&2
    exit 1
fi

# guard against an oversubscription of the slot allocation. every dataloader
# worker is its own process, and the main process also needs a core, so
# num_workers must stay strictly below the slot count. LSB_DJOB_NUMPROC is the
# count that LSF granted, in physical cores. warning: do not size against
# /proc/cpuinfo or os.cpu_count(), because they report the node's logical count
# (2x, hyperthreading) and silently permit, for example, 22 workers on a
# 16-slot allocation.
# the guard matters most on a resume. the frozen config keeps the original
# num_workers, so a resubmit to a smaller queue oversubscribes silently.
if [ -f "${config_path}" ]; then
    n_workers=$(awk '/^[[:space:]]*num_workers:/ {print $2; exit}' "${config_path}")
    if [ -n "${n_workers:-}" ] && [ -n "${LSB_DJOB_NUMPROC:-}" ]; then
        if [ "${n_workers}" -ge "${LSB_DJOB_NUMPROC}" ]; then
            echo "ERROR: num_workers=${n_workers} in ${config_path} needs at least" >&2
            echo "       $((n_workers + 1)) slots, but LSF granted ${LSB_DJOB_NUMPROC}." >&2
            echo "       raise '#BSUB -n' to match the frozen config." >&2
            exit 1
        fi
        echo "[submit] slots=${LSB_DJOB_NUMPROC}  num_workers=${n_workers}  (ok)"
    fi
fi

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

# resume from the most recent checkpoint. train.py loads the student, teacher,
# optimizer, scaler, center and step counter. train.py then continues from
# saved_step + 1 until cfg.total_steps. the job uses the same frozen config as
# the original run.
python train.py --config "${config_path}" --run-name "${RUN_NAME}" --resume "${ckpt_path}"

# clean up the node-local scratch at the end.
rm -rf $TMPDIR
