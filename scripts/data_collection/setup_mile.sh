#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MILE_DIR="${REPO_ROOT}/external/mile"
MILE_URL="https://github.com/wayveai/mile"

if [[ -d "${MILE_DIR}/.git" ]]; then
  echo "MILE already exists: ${MILE_DIR}"
else
  mkdir -p "${REPO_ROOT}/external"
  git clone "${MILE_URL}" "${MILE_DIR}"
fi

echo "MILE repo:"
git -C "${MILE_DIR}" remote -v
echo "branch: $(git -C "${MILE_DIR}" branch --show-current)"
echo "commit: $(git -C "${MILE_DIR}" rev-parse HEAD)"

cat > "${REPO_ROOT}/external/mile_info.json" <<EOF
{
  "branch": "$(git -C "${MILE_DIR}" branch --show-current)",
  "commit": "$(git -C "${MILE_DIR}" rev-parse HEAD)",
  "remote": "$(git -C "${MILE_DIR}" remote get-url origin)",
  "repo": "${MILE_URL}"
}
EOF

cat <<'EOF'

Manual next steps:
1. Install CARLA 0.9.11 separately.
2. Create the MILE conda environment from external/mile/environment.yml.
3. Install the CARLA Python egg and set CARLA_ROOT/PYTHONPATH.
4. Confirm W&B access for the Roach PPO checkpoint configured in config/agent/ppo.yaml.
5. Run scripts/data_collection/collect_action_diverse_mile.sh with explicit arguments.

This setup script does not install dependencies or download checkpoints.
EOF
