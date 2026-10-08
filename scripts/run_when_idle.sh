#!/usr/bin/env bash
# Wait for the GPU to be genuinely free, then run every measurement that has
# to be trustworthy, in order of how soon its answer is wanted.
#
# Timings taken while something else is using the card are worthless: this
# project measured one solver configuration at 10s and at 77s on consecutive
# runs before noticing an unrelated workload was pinning the GPU at 98%. So
# nothing starts until the card has been quiet for a while.
#
#   bash scripts/run_when_idle.sh [max_wait_minutes]

set -uo pipefail
cd "$HOME/bond-cvar-cuopt"

MAX_WAIT_MIN="${1:-180}"
IDLE_PCT=15          # utilisation below this counts as free
NEED_CONSECUTIVE=6   # ~1 minute of continuous quiet
SLEEP_S=10

echo "Waiting for the GPU (idle = <${IDLE_PCT}% for ${NEED_CONSECUTIVE} samples)"
deadline=$(( $(date +%s) + MAX_WAIT_MIN * 60 ))
quiet=0
while :; do
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
    if [ "${util:-100}" -lt "$IDLE_PCT" ]; then
        quiet=$(( quiet + 1 ))
    else
        [ "$quiet" -gt 0 ] && echo "  ...busy again (${util}%), resetting"
        quiet=0
    fi
    [ "$quiet" -ge "$NEED_CONSECUTIVE" ] && break
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "GAVE UP: GPU still busy after ${MAX_WAIT_MIN} minutes."
        exit 2
    fi
    sleep "$SLEEP_S"
done
echo "GPU is free at $(date -Is). Starting."
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv

echo
echo "=============================================================="
echo "[1/3] M2 re-measurement on a clean GPU (credit pipeline)"
echo "=============================================================="
.venv/bin/python scripts/run_credit.py --n-bonds 500 --n-scenarios 20000 \
    --runs 3 --skip-cpu-scenarios \
    --out results/m2_credit_500x20k.csv 2>&1 | grep -v RuntimeWarning
echo "M2 exit=$?"

echo
echo "=============================================================="
echo "[2/3] M3 scaling sweep (this is the long one)"
echo "=============================================================="
.venv/bin/python scripts/run_sweep.py --grid full --runs 3 \
    --time-limit 300 --out results/m3_sweep.csv 2>&1 | grep -v RuntimeWarning
echo "sweep exit=$?"

echo
echo "=============================================================="
echo "[3/3] Chart"
echo "=============================================================="
.venv/bin/python scripts/plot_sweep.py --csv results/m3_sweep.csv \
    --out results/m3_scaling.png
echo "plot exit=$?"

echo
echo "GPU state at finish:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv
echo "ALL_DONE"
