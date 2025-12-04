#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_sac_baseline_sweep.sh Hopper-v5
#
# Optional environment variables:
#   PROJECT       - wandb project name (default: "sac_<ENV_ID>_baseline")
#   SEEDS         - list of seeds (default: "1 2 3 4 5 6 7 8 9 10")
#   ENTITY        - wandb entity
#   CUDA          - True/False (default: True)
#   CAPTURE_VIDEO - 0/1 (default: 0)
#   CONCURRENCY   - number of parallel workers (default: 1)

ENV_ID="${1:?Need ENV_ID}"

SEEDS_DEFAULT="1 2 3 4 5 6 7 8 9 10"
SEEDS="${SEEDS:-$SEEDS_DEFAULT}"

PROJECT="${PROJECT:-sac_${ENV_ID}_baseline}"
ENTITY="${ENTITY:-}"
CUDA="${CUDA:-True}"
CAPTURE_VIDEO="${CAPTURE_VIDEO:-0}"
CONCURRENCY="${CONCURRENCY:-1}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "Launching SAC baseline sweep:"
echo "  ENV_ID        = ${ENV_ID}"
echo "  PROJECT       = ${PROJECT}"
echo "  SEEDS         = ${SEEDS}"
echo "  ENTITY        = ${ENTITY}"
echo "  CUDA          = ${CUDA}"
echo "  CAPTURE_VIDEO = ${CAPTURE_VIDEO}"
echo "  CONCURRENCY   = ${CONCURRENCY}"
echo

# job throttle
sem() {
  local max="$1"
  while (( $(jobs -rp | wc -l) >= max )); do
    sleep 1
  done
}

for SEED in ${SEEDS}; do
  sem "${CONCURRENCY}"

  echo "==> SAC baseline | Env=${ENV_ID} | Seed=${SEED}"

  PROJECT="${PROJECT}" \
  ENTITY="${ENTITY}" \
  CUDA="${CUDA}" \
  CAPTURE_VIDEO="${CAPTURE_VIDEO}" \
    ./scripts/run_sac_baseline.sh "${ENV_ID}" "${SEED}" &
done

wait
echo "All SAC baseline runs for ${ENV_ID} finished."