#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_preinit_sweep.sh ALGO ENV_ID WEIGHTS_DIR
#
# Example:
#   ./scripts/run_preinit_sweep.sh ppo Hopper-v5 ../ga/gen-alg-rl/hopper_ppo/
#
# Env vars:
#   PROJECT       - wandb project name (default: "<ALGO>_<ENV_ID>_pretrain")
#   SEEDS         - space-separated seeds (default: "1 2 3 4 5 6 7 8 9 10")
#   ENTITY        - wandb entity (optional)
#   CUDA          - "True"/"False" (default: False)
#   CAPTURE_VIDEO - "0" or "1" (default: 0)
#   CONCURRENCY   - max number of runs in parallel (default: 1)

ALGO="${1:?need ALGO (ppo|sac|td3)}"
ENV_ID="${2:?need ENV_ID (e.g. Hopper-v5)}"
WEIGHTS_DIR="${3:?need WEIGHTS_DIR containing .pt files}"

# normalize algo
ALGO="$(echo "${ALGO}" | tr '[:upper:]' '[:lower:]')"

case "${ALGO}" in
  ppo) RUN_SCRIPT="run_ppo_pretrain.sh" ;;
  sac) RUN_SCRIPT="run_sac_pretrain.sh" ;;
  td3) RUN_SCRIPT="run_td3_pretrain.sh" ;;
  *)
    echo "Unsupported ALGO '${ALGO}'. Use one of: ppo, sac, td3." >&2
    exit 1
    ;;
esac

SEEDS_DEFAULT="1 2 3 4 5 6 7 8 9 10"
SEEDS="${SEEDS:-$SEEDS_DEFAULT}"

# project name for all runs in this sweep
PROJECT="${PROJECT:-${ALGO}_${ENV_ID}_pretrain}"
CUDA="${CUDA:-False}"
CAPTURE_VIDEO="${CAPTURE_VIDEO:-0}"
ENTITY="${ENTITY:-}"
CONCURRENCY="${CONCURRENCY:-1}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# simple semaphore: wait until number of running jobs < CONCURRENCY
sem() {
  local max="$1"
  while (( $(jobs -rp | wc -l) >= max )); do
    sleep 1
  done
}

# collect weight files (sorted for determinism)
mapfile -t WEIGHT_FILES < <(find "${WEIGHTS_DIR}" -maxdepth 1 -type f -name '*.pt' | sort)

if [[ ${#WEIGHT_FILES[@]} -eq 0 ]]; then
  echo "No .pt files found in ${WEIGHTS_DIR}" >&2
  exit 1
fi

# pair seeds with weight files
read -r -a SEED_ARR <<< "${SEEDS}"

if [[ ${#SEED_ARR[@]} -lt ${#WEIGHT_FILES[@]} ]]; then
  echo "Warning: fewer seeds (${#SEED_ARR[@]}) than weight files (${#WEIGHT_FILES[@]}). Seeds will wrap around." >&2
fi

echo "Running sweep:"
echo "  ALGO          = ${ALGO}"
echo "  ENV_ID        = ${ENV_ID}"
echo "  PROJECT       = ${PROJECT}"
echo "  WEIGHTS_DIR   = ${WEIGHTS_DIR}"
echo "  CUDA          = ${CUDA}"
echo "  ENTITY        = ${ENTITY}"
echo "  CAPTURE_VIDEO = ${CAPTURE_VIDEO}"
echo "  CONCURRENCY   = ${CONCURRENCY}"
echo

idx=0
for WEIGHT_PATH in "${WEIGHT_FILES[@]}"; do
  # throttle parallelism
  sem "${CONCURRENCY}"

  SEED="${SEED_ARR[$((idx % ${#SEED_ARR[@]}))]}"

  BASENAME="$(basename "${WEIGHT_PATH}")"
  BASENAME_NOEXT="${BASENAME%.*}"

  # This ends up as args.exp_name -> part of run_name -> W&B run name
  EXP_NAME="${ALGO}_preinit_${BASENAME_NOEXT}"

  echo "==> [${ALGO}] Env=${ENV_ID}, Seed=${SEED}, Weight=${WEIGHT_PATH}, ExpName=${EXP_NAME}"

  PROJECT="${PROJECT}" \
  ENTITY="${ENTITY}" \
  CUDA="${CUDA}" \
  CAPTURE_VIDEO="${CAPTURE_VIDEO}" \
  EXP_NAME="${EXP_NAME}" \
    "./scripts/${RUN_SCRIPT}" "${ENV_ID}" "${SEED}" "${WEIGHT_PATH}" &

  idx=$((idx + 1))
done

# wait for all background jobs to finish
wait
echo "All runs for ${ALGO} on ${ENV_ID} finished."
