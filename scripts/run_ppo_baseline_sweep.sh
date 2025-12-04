#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_ppo_baseline_sweep.sh ENV_ID
#
# Environment variables (optional):
#   PROJECT       - W&B project name (default: "ppo_<ENV_ID>_baseline")
#   SEEDS         - space-separated list of seeds (default: "1 2 3 4 5 6 7 8 9 10")
#   ENTITY        - W&B entity (optional)
#   CUDA          - "True"/"False" (default: "True")
#   CAPTURE_VIDEO - "0" or "1" (default: "0")
#   CONCURRENCY   - maximum parallel workers (default: 1)
#
# This script launches multiple PPO baseline runs in parallel using run_ppo_baseline.sh.

ENV_ID="${1:?need ENV_ID (e.g. Hopper-v5)}"

SEEDS_DEFAULT="2 3 4 5 6 7 8 9 10"
SEEDS="${SEEDS:-$SEEDS_DEFAULT}"

PROJECT="${PROJECT:-ppo_${ENV_ID}_baseline}"
ENTITY="${ENTITY:-}"
CUDA="${CUDA:-True}"
CAPTURE_VIDEO="${CAPTURE_VIDEO:-0}"
CONCURRENCY="${CONCURRENCY:-1}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "Launching PPO baseline sweep:"
echo "  ENV_ID        = ${ENV_ID}"
echo "  PROJECT       = ${PROJECT}"
echo "  SEEDS         = ${SEEDS}"
echo "  ENTITY        = ${ENTITY}"
echo "  CUDA          = ${CUDA}"
echo "  CAPTURE_VIDEO = ${CAPTURE_VIDEO}"
echo "  CONCURRENCY   = ${CONCURRENCY}"
echo

# simple job throttle (same as your pretrain script)
sem() {
  local max="$1"
  while (( $(jobs -rp | wc -l) >= max )); do
    sleep 1
  done
}

for SEED in ${SEEDS}; do
  sem "${CONCURRENCY}"

  echo "==> PPO baseline | Env=${ENV_ID} | Seed=${SEED}"

  PROJECT="${PROJECT}" \
  ENTITY="${ENTITY}" \
  CUDA="${CUDA}" \
  CAPTURE_VIDEO="${CAPTURE_VIDEO}" \
    ./scripts/run_ppo_baseline.sh "${ENV_ID}" "${SEED}" &
done

wait
echo "All PPO baseline runs for ${ENV_ID} finished."
