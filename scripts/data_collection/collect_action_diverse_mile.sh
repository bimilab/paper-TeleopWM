#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/data_collection/collect_action_diverse_mile.sh CARLA_SH DATASET_ROOT PORT TEST_SUITE

Example:
  scripts/data_collection/collect_action_diverse_mile.sh \
    /opt/carla-0.9.11/CarlaUE4.sh \
    /path/to/mile_action_diverse/train \
    2000 \
    config/test_suites/lb_town01.yaml

Notes:
  - This wrapper does not start unless all arguments are explicit.
  - It delegates to official external/mile/run/data_collect.sh.
  - CARLA 0.9.11 and the MILE conda environment must already be installed.
  - The official script may remove MILE resume files under external/mile/outputs/.
EOF
}

if [[ $# -ne 4 ]]; then
  usage
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MILE_DIR="${REPO_ROOT}/external/mile"
CARLA_SH="$1"
DATASET_ROOT="$2"
PORT="$3"
TEST_SUITE="$4"

if [[ ! -d "${MILE_DIR}" ]]; then
  echo "Missing external/mile. Run scripts/data_collection/setup_mile.sh first." >&2
  exit 2
fi

if [[ ! -x "${CARLA_SH}" ]]; then
  echo "CARLA executable does not exist or is not executable: ${CARLA_SH}" >&2
  exit 2
fi

mkdir -p "${DATASET_ROOT}"

cat <<EOF
About to run MILE data collection:
  MILE_DIR: ${MILE_DIR}
  CARLA_SH: ${CARLA_SH}
  DATASET_ROOT: ${DATASET_ROOT}
  PORT: ${PORT}
  TEST_SUITE: ${TEST_SUITE}

Press Ctrl-C now if this is not intended.
EOF
sleep 5

cd "${MILE_DIR}"
bash run/data_collect.sh "${CARLA_SH}" "${DATASET_ROOT}" "${PORT}" "${TEST_SUITE}"
