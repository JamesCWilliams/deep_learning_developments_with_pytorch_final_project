#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_sac_with_weights.sh ENV_ID SEED WEIGHTS_FILE
#
# Env vars (optional):
#   PROJECT, ENTITY, TIMESTEPS, NUM_ENVS, CAPTURE_VIDEO, CUDA, EXP_NAME

ENV_ID="${1:?need ENV_ID (e.g. Hopper-v4)}"
SEED="${2:?need SEED (e.g. 1)}"
ACTOR_INIT_FILE="${3:?need path to SAC weights file}"

PROJECT="${PROJECT:-sac_pretrain_runs}"
ENTITY="${ENTITY:-}"
TIMESTEPS="${TIMESTEPS:-1000000}"
NUM_ENVS="${NUM_ENVS:-1}"
CAPTURE_VIDEO="${CAPTURE_VIDEO:-0}"
CUDA="${CUDA:-False}"
EXP_NAME="${EXP_NAME:-sac_preinit}"

# Build a run id & directories
TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
RUN_ID="${EXP_NAME}_${ENV_ID}_seed${SEED}_${TIMESTAMP}"

RUN_DIR="runs/${RUN_ID}"
DATA_DIR="data/${RUN_ID}"

mkdir -p "${RUN_DIR}" "${DATA_DIR}"

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

# Optionally help wandb naming
export WANDB_RUN_ID="${RUN_ID}"
export WANDB_RUN_NAME="${RUN_ID}"

python scripts/sac_runner.py \
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
