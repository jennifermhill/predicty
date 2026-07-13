#!/bin/bash
# entry point for launching the evaluation pipeline (latent space viz +
# dataset/behavior probes + grad-cam) on the LSF cluster.
#
# usage:
#   ./launch_eval.sh <run_name> [<run_name2> ...]
#
# each run must already exist from training (checkpoints/<run>/config.yaml +
# checkpoints/<run>/latest.pt, both written by ./launch.sh). evaluation_pipeline.py
# skips any step it finds already-completed output for, so re-running this on
# runs that were already evaluated is cheap.
#
# slot count (-n) is derived from num_workers in each run's frozen config.yaml
# (the max across the listed runs, if evaluating several at once) plus 2, to
# cover the dataloader workers plus the main process and gpu sampler.

set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: ./launch_eval.sh <run_name> [<run_name2> ...]" >&2
    exit 1
fi

run_names=("$@")
# joins run names with underscores to namespace this submission's logs, e.g.
# two runs "foo" and "bar" -> logs/lsf/eval_foo_bar/
eval_name=$(IFS=_; echo "${run_names[*]}")
lsf_dir="logs/lsf/eval_${eval_name}"
mkdir -p "${lsf_dir}"

# read num_workers straight out of each run's frozen config.yaml (plain
# `key: value` line, no need to pull in a yaml parser) and take the max
# across runs so one job comfortably covers every run's dataloader workers.
max_workers=0
for run in "${run_names[@]}"; do
    config_path="checkpoints/${run}/config.yaml"
    if [ ! -f "${config_path}" ]; then
        echo "ERROR: ${config_path} not found. train this run first (./launch.sh ${run})." >&2
        exit 1
    fi
    workers=$(grep -E '^num_workers:' "${config_path}" | head -1 | sed -E 's/^num_workers:[[:space:]]*([0-9]+).*/\1/')
    if [ -z "${workers}" ]; then
        echo "ERROR: could not find num_workers in ${config_path}." >&2
        exit 1
    fi
    if [ "${workers}" -gt "${max_workers}" ]; then
        max_workers=${workers}
    fi
done
n_slots=$((max_workers + 2))

echo "[launch_eval] runs      : ${run_names[*]}"
echo "[launch_eval] slots (-n): ${n_slots}  (max num_workers ${max_workers} + 2)"
echo "[launch_eval] lsf logs  : ${lsf_dir}/eval.out , ${lsf_dir}/eval.err"

# command-line -J/-n/-oo/-eo override the static directives in submit_eval.sh.
# -env exports EVAL_NAME (for log namespacing) and RUN_NAMES (space-separated,
# forwarded as-is to evaluation_pipeline.py's --run_names). "all" keeps the
# rest of the submission environment too.
bsub -J "eval_${eval_name}" -n "${n_slots}" \
    -oo "${lsf_dir}/eval.out" -eo "${lsf_dir}/eval.err" \
    -env "all, EVAL_NAME=${eval_name}, RUN_NAMES=${run_names[*]}" \
    < submit_eval.sh
