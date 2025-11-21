#!/usr/bin/env bash

set -euo pipefail

# config
PROJECT="${PROJECT:-baseline}"
ENTITY="${ENTITY:-}"
TIMESTEPS="${TIMESTEPS:-1_000_000}"

# sweep
SEEDS=(${SEEDS:-1 2 3 4 5 6 7 8 9 10})
ENVS=(${ENVS:-Hopper-v5 Ant-v5 HalfCheetah-v5 Pusher-v5})

# how many runs in parallel
CONCURRENCY="${CONCURRENCY:-1}"

# performance tuning
NUM_ENVS="${NUM_ENVS:=8}"
BATCH_SIZE="${BATCH_SIZE:-512}"

# whether to capture videos
CAPTURE_VIDEO="${CAPTURE_VIDEO:-1}"

# args to pass straight through
EXTRA_ARGS="${EXTRA_ARGS:-}"

TRACK_FLAG="--track"
WPN_FLAG=(--wandb-project-name "${PROJECT}")
WE_FLAG=()
[[ -n "${ENTITY}" ]] && WE_FLAG=(--wandb-entity "${ENTITY}")

[[ "${CAPTURE_VIDEO}" == "1" ]] && EXTRA_ARGS+=" --capture-video"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"   # headless rendering

# semaphore
sem() {
  local max="$1"
  while (( $(jobs -rp | wc -l) >= max )); do sleep 1; done
}

# run one algorithm over ENVS x SEEDS
run_algo () {
  local ALGO="$1" SCRIPT="$2" GROUP="$3"

  for ENV in "${ENVS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      sem "${CONCURRENCY}"
      (
        export WANDB_RUN_GROUP="${GROUP}"

        # per-run extra args (so we can conditionally add video)
        local run_extra_args="${EXTRA_ARGS}"

        # only capture video for seed 1 if CAPTURE_VIDEO=1
        if [[ "${CAPTURE_VIDEO}" == "1" ]]; then
          run_extra_args+=" --capture-video"
        fi

        python -m "cleanrl.${SCRIPT}" \
          --env-id "${ENV}" \
          --total-timesteps "${TIMESTEPS}" \
          --seed "${SEED}" \
          ${TRACK_FLAG} \
          "${WPN_FLAG[@]}" "${WE_FLAG[@]}" \
          --exp-name "${ALGO}" \
          --num-envs "${NUM_ENVS}" \
          --batch-size "${BATCH_SIZE}" \
          ${run_extra_args}
      ) &
    done
  done
}

# launch all baselines
run_algo "ppo" "ppo_continuous_action" "ppo_baseline"
run_algo "sac" "sac_continuous_action" "sac_baseline"
run_algo "td3" "td3_continuous_action" "td3_baseline"

wait
echo "All baselines finished."
