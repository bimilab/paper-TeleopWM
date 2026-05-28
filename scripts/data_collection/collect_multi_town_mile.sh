#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Collect controlled MILE runs one town at a time.

Required:
  --carla-sh PATH
  --dataset-root PATH

Optional:
  --port PORT                 Default: 2000
  --runs-per-town N           Default: 3
  --towns Town01 Town02 ...   Default: Town01 Town02 Town03 Town04 Town05
  --mile-dir PATH             Default: external/mile
  --python PATH               Default: python
  --min-frames N              Default: 18

Example:
  scripts/data_collection/collect_multi_town_mile.sh \
    --carla-sh /path/to/CARLA_0.9.11/CarlaUE4.sh \
    --dataset-root /path/to/mile_action_diverse/train \
    --port 2000 \
    --runs-per-town 3 \
    --towns Town01 Town02 Town03 Town04 Town05
EOF
}

CARLA_SH=""
DATASET_ROOT=""
PORT="2000"
RUNS_PER_TOWN="3"
MILE_DIR="external/mile"
PYTHON_BIN="python"
MIN_FRAMES="18"
TOWNS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --carla-sh)
      CARLA_SH="$2"
      shift 2
      ;;
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --runs-per-town)
      RUNS_PER_TOWN="$2"
      shift 2
      ;;
    --mile-dir)
      MILE_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --min-frames)
      MIN_FRAMES="$2"
      shift 2
      ;;
    --towns)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        TOWNS+=("$1")
        shift
      done
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$CARLA_SH" || -z "$DATASET_ROOT" ]]; then
  usage >&2
  exit 2
fi

if [[ ${#TOWNS[@]} -eq 0 ]]; then
  TOWNS=(Town01 Town02 Town03 Town04 Town05)
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MILE_DIR_ABS="$(cd "$ROOT_DIR/$MILE_DIR" 2>/dev/null || cd "$MILE_DIR"; pwd)"
DATASET_ROOT_ABS="$(mkdir -p "$DATASET_ROOT" && cd "$DATASET_ROOT"; pwd)"
STAGING_BASE="$DATASET_ROOT_ABS/_mile_staging"

complete_count() {
  local town="$1"
  "$PYTHON_BIN" - "$DATASET_ROOT_ABS" "$town" "$MIN_FRAMES" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
town = sys.argv[2]
min_frames = int(sys.argv[3])
count = 0
town_dir = root / town
if town_dir.is_dir():
    for run_dir in sorted(p for p in town_dir.iterdir() if p.is_dir()):
        image_dir = run_dir / "image"
        if not (run_dir / "pd_dataframe.pkl").is_file() or not image_dir.is_dir():
            continue
        n_images = sum(1 for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if n_images >= min_frames:
            count += 1
print(count)
PY
}

next_run_id() {
  local town="$1"
  "$PYTHON_BIN" - "$DATASET_ROOT_ABS" "$town" <<'PY'
import sys
from pathlib import Path

town_dir = Path(sys.argv[1]) / sys.argv[2]
max_id = -1
if town_dir.is_dir():
    for path in town_dir.iterdir():
        if path.is_dir() and path.name.isdigit():
            max_id = max(max_id, int(path.name))
print(f"{max_id + 1:04d}")
PY
}

archive_existing_staging() {
  local staging_root="$1"
  if [[ -d "$staging_root" ]]; then
    local archive="${staging_root}_incomplete_$(date +%Y%m%d_%H%M%S)"
    echo "Archiving leftover staging directory: $staging_root -> $archive"
    mv "$staging_root" "$archive"
  fi
}

move_completed_from_staging() {
  local town="$1"
  local staging_root="$2"
  local src_town="$staging_root/$town"
  local dst_town="$DATASET_ROOT_ABS/$town"
  mkdir -p "$dst_town"

  if [[ ! -d "$src_town" ]]; then
    return 0
  fi

  local moved=0
  for run_dir in "$src_town"/*; do
    [[ -d "$run_dir" ]] || continue
    local image_dir="$run_dir/image"
    [[ -f "$run_dir/pd_dataframe.pkl" && -d "$image_dir" ]] || continue
    local image_count
    image_count="$(find "$image_dir" -maxdepth 1 -type f -name '*.png' | wc -l)"
    if [[ "$image_count" -lt "$MIN_FRAMES" ]]; then
      continue
    fi
    local new_id
    new_id="$(next_run_id "$town")"
    echo "Moving completed $town run $(basename "$run_dir") -> $town/$new_id ($image_count frames)"
    mv "$run_dir" "$dst_town/$new_id"
    moved=$((moved + 1))
  done

  if [[ "$moved" -gt 0 ]]; then
    echo "Moved $moved completed staged run(s) for $town."
  fi
}

kill_carla_port() {
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
  fi
}

echo "MILE directory: $MILE_DIR_ABS"
echo "Dataset root:   $DATASET_ROOT_ABS"
echo "CARLA shell:    $CARLA_SH"
echo "Port:           $PORT"
echo "Runs per town:  $RUNS_PER_TOWN"
echo "Towns:          ${TOWNS[*]}"
echo ""
echo "MILE server_utils launches CARLA through CarlaServerManager."
echo "Expected local settings: CARLA_FPS=15 and -quality-level=Low; RenderOffScreen is not used."
echo ""

for town in "${TOWNS[@]}"; do
  suite="lb_${town,,}_small"
  suite_file="$MILE_DIR_ABS/config/test_suites/${suite}.yaml"
  if [[ ! -f "$suite_file" ]]; then
    echo "Missing suite for $town: $suite_file" >&2
    exit 1
  fi

  staging_root="$STAGING_BASE/$town"
  move_completed_from_staging "$town" "$staging_root"
  archive_existing_staging "$staging_root"

  current="$(complete_count "$town")"
  if [[ "$current" -ge "$RUNS_PER_TOWN" ]]; then
    echo "[$town] already has $current completed run(s); target is $RUNS_PER_TOWN. Skipping."
    continue
  fi

  while [[ "$current" -lt "$RUNS_PER_TOWN" ]]; do
    needed=$((RUNS_PER_TOWN - current))
    echo "[$town] completed=$current target=$RUNS_PER_TOWN; collecting $needed staged run(s)."
    mkdir -p "$staging_root"

    (
      cd "$MILE_DIR_ABS"
      "$PYTHON_BIN" -u data_collect.py \
        --config-name data_collect \
        "carla_sh_path=$CARLA_SH" \
        "dataset_root=$staging_root" \
        "port=$PORT" \
        "test_suites=$suite" \
        "n_episodes=$needed" \
        "resume=false" \
        "kill_running=true"
    )

    move_completed_from_staging "$town" "$staging_root"
    archive_existing_staging "$staging_root"
    kill_carla_port

    current="$(complete_count "$town")"
    echo "[$town] now has $current completed run(s)."
  done
done

echo ""
echo "Final run counts:"
"$ROOT_DIR/scripts/data_collection/count_mile_runs.py" \
  --dataset-root "$DATASET_ROOT_ABS" \
  --towns "${TOWNS[@]}" \
  --min-frames "$MIN_FRAMES"

echo ""
echo "Controlled multi-town MILE collection complete."
