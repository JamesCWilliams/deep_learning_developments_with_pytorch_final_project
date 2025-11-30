#!/usr/bin/env bash
# Run genetic algorithm training across a fixed set of environments, algorithms, and seeds.
# Outputs the best actor parameters into a structured directory tree under the provided root.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <output_root> [additional train_ga.py args]" >&2
    exit 1
fi

OUTPUT_ROOT=${1%/}
shift

# Additional arguments are forwarded directly to train_ga.py
EXTRA_ARGS=()
if [[ $# -gt 0 ]]; then
    EXTRA_ARGS=("$@")
fi

ENV_IDS=("Hopper-v5" "HalfCheetah-v5" "Pusher-v5" "Ant-v5")
ALGOS=("ppo" "sac" "td3")
SEEDS=$(seq 1 10)

for ENV_ID in "${ENV_IDS[@]}"; do
    for ALGO in "${ALGOS[@]}"; do
        ALGO_DIR="${OUTPUT_ROOT}/${ENV_ID}/${ALGO^^}"
        mkdir -p "${ALGO_DIR}"

        for SEED in ${SEEDS}; do
            BEST_PATH="${ALGO_DIR}/weights${SEED}.pt"
            echo "Running env=${ENV_ID}, algo=${ALGO}, seed=${SEED} -> ${BEST_PATH}" >&2
            python train_ga.py \
                --env_id "${ENV_ID}" \
                --flavor "${ALGO}" \
                --seed "${SEED}" \
                --best_actor_path "${BEST_PATH}" \
                --population_size 50 --generations 20 --num_workers 4 --max_steps 1000 \
                "${EXTRA_ARGS[@]}"
        done
    done
done
