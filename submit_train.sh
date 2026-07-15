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
#BSUB -q gpu_a100                   # H200 queue: 12 slots/GPU, 40GB ram/slot, 141GB VRAM
#BSUB -gpu "num=1"                   # request 1 GPU
#BSUB -n 20                           # 8 slots = 8 cores + 320 GB ram (slot-implied)
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

# cap per-process threading. without these, every numerical library detects
# the full node's core count and spawns that many threads per process. with
# 4 dataloader workers + 1 main process, value of 2 keeps total threads at
# ~10 across the 8 slots we requested.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

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