#!/bin/bash
# entry point that launches the evaluation pipeline (latent space viz +
# dataset/behavior probes + grad-cam) on the LSF cluster.
#
# usage:
#   ./launch_eval.sh <run_name> [<run_name2> ...]
#
# each run must already exist from training. ./launch.sh writes both
# checkpoints/<run>/config.yaml and checkpoints/<run>/latest.pt.
# evaluation_pipeline.py skips a step when it finds complete output for that
# step, so a second run of this script on an evaluated run is cheap.
#
# the script derives the slot count (-n) from num_workers in each run's frozen
# config.yaml, plus 2. for several runs at once the script takes the maximum
# num_workers. the 2 extra slots cover the main process and the gpu sampler.

set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: ./launch_eval.sh <run_name> [<run_name2> ...]" >&2
    exit 1
fi

run_names=("$@")
# join the run names with underscores, to namespace the logs of this
# submission. for example, two runs "foo" and "bar" -> logs/lsf/eval_foo_bar/
eval_name=$(IFS=_; echo "${run_names[*]}")
lsf_dir="logs/lsf/eval_${eval_name}"
mkdir -p "${lsf_dir}"

# read num_workers straight out of each run's frozen config.yaml. the line is a
# plain `key: value` line, so the script needs no yaml parser. take the maximum
# across the runs, so one job covers the dataloader workers of every run.
max_workers=0
for run in "${run_names[@]}"; do
    config_path="checkpoints/${run}/config.yaml"
    if [ ! -f "${config_path}" ]; then
        echo "ERROR: ${config_path} not found. train this run first (./launch.sh ${run})." >&2
        exit 1
    fi
    # use awk, not grep | head | sed, so a config without the key gives an empty
    # string and does not trip `set -o pipefail`. under pipefail a grep with no
    # match exits 1, and set -e then kills the script here silently. the error
    # message below could never print. submit_train.sh uses the same idiom.
    workers=$(awk '/^[[:space:]]*num_workers:/ {print $2; exit}' "${config_path}")
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

# the command-line -J/-n/-oo/-eo override the static directives in
# submit_eval.sh. -env exports EVAL_NAME, which namespaces the logs, and
# RUN_NAMES, which is space-separated. the script forwards RUN_NAMES unchanged
# to evaluation_pipeline.py's --run-names. "all" also keeps the rest of the
# submission environment.
bsub -J "eval_${eval_name}" -n "${n_slots}" \
    -oo "${lsf_dir}/eval.out" -eo "${lsf_dir}/eval.err" \
    -env "all, EVAL_NAME=${eval_name}, RUN_NAMES=${run_names[*]}" \
    < submit_eval.sh
