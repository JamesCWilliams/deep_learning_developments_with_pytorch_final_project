#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_ppo_with_weights.sh ENV_ID SEED WEIGHTS_FILE
#
# Example:
#   ./scripts/run_ppo_with_weights.sh HalfCheetah-v4 1 /path/to/ppo_run.pt

ENV_ID="${1:?need ENV_ID (e.g. HalfCheetah-v4)}"
SEED="${2:?need SEED (e.g. 1)}"
ACTOR_INIT_FILE="${3:?need path to PPO weights file}"

PROJECT="${PROJECT:-ppo_pretrain_runs}"   # can be overridden by orchestrator
ENTITY="${ENTITY:-}"                      # optional W&B entity
TIMESTEPS="${TIMESTEPS:-1000000}"
NUM_ENVS="${NUM_ENVS:-1}"
NUM_STEPS="${NUM_STEPS:-2048}"            # match CleanRL default
CAPTURE_VIDEO="${CAPTURE_VIDEO:-0}"
CUDA="${CUDA:-True}"                      # default CUDA
EXP_NAME="${EXP_NAME:-ppo_preinit}"       # allow override

# Build a run id & directories
TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
RUN_ID="${EXP_NAME}_${ENV_ID}_seed${SEED}_${TIMESTAMP}"

RUN_DIR="runs/${RUN_ID}"
DATA_DIR="data/${RUN_ID}"

mkdir -p "${RUN_DIR}" "${DATA_DIR}"

# optional: make W&B run name/id nice & deterministic
export WANDB_RUN_ID="${RUN_ID}"
export WANDB_RUN_NAME="${RUN_ID}"

EXTRA_ARGS=()
if [[ "${CAPTURE_VIDEO}" == "1" ]]; then
  EXTRA_ARGS+=(--capture_video)
fi

ENTITY_ARGS=()
if [[ -n "${ENTITY}" ]]; then
  ENTITY_ARGS=(--wandb_entity "${ENTITY}")
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

python scripts/ppo_runner.py \
  --env_id "${ENV_ID}" \
  --seed "${SEED}" \
  --total_timesteps "${TIMESTEPS}" \
  --num_envs "${NUM_ENVS}" \
  --exp_name "${EXP_NAME}" \
  --cuda "${CUDA}" \
  --track \
  --wandb_project_name "${PROJECT}" \
  "${ENTITY_ARGS[@]}" \
  --actor-init-file "${ACTOR_INIT_FILE}" \
  --run-dir "${RUN_DIR}" \
  --data-dir "${DATA_DIR}" \
  "${EXTRA_ARGS[@]}"
