#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_td3_with_weights.sh ENV_ID SEED WEIGHTS_FILE

ENV_ID="${1:?need ENV_ID (e.g. Hopper-v4)}"
SEED="${2:?need SEED (e.g. 1)}"
ACTOR_INIT_FILE="${3:?need path to TD3 weights file}"

PROJECT="${PROJECT:-td3_pretrain_runs}"
ENTITY="${ENTITY:-}"
TIMESTEPS="${TIMESTEPS:-1000000}"
NUM_ENVS="${NUM_ENVS:-1}"
CAPTURE_VIDEO="${CAPTURE_VIDEO:-0}"
CUDA="${CUDA:-False}"
EXP_NAME="${EXP_NAME:-td3_preinit}"

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

python scripts/td3_runner.py \
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
  "${EXTRA_ARGS[@]}"
