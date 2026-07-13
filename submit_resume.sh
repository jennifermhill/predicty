#!/bin/bash
# LSF job body for resuming a WS-DINO run from its most recent checkpoint.
#
# do not submit this directly. launch it through launch.sh:
#
#   cd /groups/sgro/sgrolab/Ankit/predicty
#   ./launch.sh <run_name> --resume
#
# requires checkpoints/<run_name>/latest.pt to exist (saved every ckpt_interval
# steps by train.py). will fail loudly if it doesn't. launch a fresh run first.

# ---------------------------------------------------------------------------
# LSF directives (same shape as submit_train.sh)
# ---------------------------------------------------------------------------
# -J/-oo/-eo are overridden per-run by launch.sh; the values here are only the
# fallback for a raw `bsub < submit_resume.sh` (run name defaults to 'default').
#BSUB -J predicty_resume              # job name (shows in bjobs)
#BSUB -q gpu_h200                     # H200 queue: 12 slots/GPU, 40GB ram/slot, 141GB VRAM
#BSUB -gpu "num=1"                    # request 1 GPU
#BSUB -n 20                            # 8 slots = 8 cores + 320 GB ram (slot-implied)
#BSUB -W 72:00                        # 72 hour walltime
#BSUB -oo logs/lsf/default/resume.out # stdout log (overwritten on each submit; -oo not -o)
#BSUB -eo logs/lsf/default/resume.err # stderr log (overwritten on each submit)

# ---------------------------------------------------------------------------
# script body
# ---------------------------------------------------------------------------
set -euo pipefail

# run name drives every output path. launch.sh exports it; a raw bsub falls
# back to 'default'.
RUN_NAME="${RUN_NAME:-default}"
config_path="checkpoints/${RUN_NAME}/config.yaml"
ckpt_path="checkpoints/${RUN_NAME}/latest.pt"
lsf_dir="logs/lsf/${RUN_NAME}"

# LSF needs the log directory to exist before it can write into it.
mkdir -p "${lsf_dir}"

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

# fail with a clear error if there's nothing to resume from.
if [ ! -f "${ckpt_path}" ]; then
    echo "ERROR: ${ckpt_path} not found." >&2
    echo "launch a fresh run first: ./launch.sh ${RUN_NAME}" >&2
    exit 1
fi

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

# resume from the most recent checkpoint. train.py loads student + teacher
# + optimizer + scaler + center + step counter, then continues from
# saved_step + 1 until cfg.total_steps. uses the same frozen config the run
# started with.
python train.py --config "${config_path}" --run-name "${RUN_NAME}" --resume "${ckpt_path}"

# clean up node-local scratch on the way out.
rm -rf $TMPDIR
