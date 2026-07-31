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
#BSUB -q gpu_a100                 # A100 queue: 12 slots/GPU, 40GB ram/slot, 141GB VRAM
#BSUB -gpu "num=1"                   # request 1 GPU
#BSUB -n 12                           # 8 slots = 8 cores + 320 GB ram (slot-implied)
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

# fail with a clear error if there's nothing to resume from.
if [ ! -f "${ckpt_path}" ]; then
    echo "ERROR: ${ckpt_path} not found." >&2
    echo "launch a fresh run first: ./launch.sh ${RUN_NAME}" >&2
    exit 1
fi

# guard against oversubscribing the slot allocation. every dataloader worker is
# its own process and the main process needs a core too, so num_workers must be
# strictly below the slot count. LSB_DJOB_NUMPROC is what LSF actually granted,
# counted in physical cores -- do not size against /proc/cpuinfo or
# os.cpu_count(), which report the node's logical count (2x, hyperthreading) and
# would silently let e.g. 22 workers land on a 16-slot allocation.
# matters especially on resume: the frozen config keeps the original
# num_workers, so resubmitting to a smaller queue silently oversubscribes.
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
