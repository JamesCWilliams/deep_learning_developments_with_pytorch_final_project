#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_sac_baseline.sh Hopper-v5 3
#
# Runs a cold-start SAC baseline training run:
#   - NO --actor-init-file
#   - full wandb logging + checkpoints via sac_runner.py

ENV_ID="${1:?Need ENV_ID (e.g. Hopper-v5)}"
SEED="${2:?Need SEED}"

PROJECT="${PROJECT:-sac_baseline_runs}"   # wandb project
ENTITY="${ENTITY:-}"                      # optional
TIMESTEPS="${TIMESTEPS:-1000000}"
NUM_ENVS="${NUM_ENVS:-1}"
CAPTURE_VIDEO="${CAPTURE_VIDEO:-0}"
CUDA="${CUDA:-True}"
EXP_NAME="${EXP_NAME:-sac_baseline}"

# Unique run ID
TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
RUN_ID="${EXP_NAME}_${ENV_ID}_seed${SEED}_${TIMESTAMP}"

RUN_DIR="runs/${RUN_ID}"
DATA_DIR="data/${RUN_ID}"

mkdir -p "${RUN_DIR}" "${DATA_DIR}"

# Make wandb run names deterministic
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
  --run-dir "${RUN_DIR}" \
  --data-dir "${DATA_DIR}" \
  "${EXTRA_ARGS[@]}"
