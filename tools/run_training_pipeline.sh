#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 RUN CONFIG [plump train options...]" >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_name="$1"
config_path="$2"
shift 2

train_session="plump-${run_name}-train"
monitor_session="plump-${run_name}-monitor"
run_path="${PLUMP_RUNS_DIR:-${project_root}/runs}/${run_name}"
mkdir -p "$run_path"

if screen -ls | grep -Fq ".${train_session}"; then
  echo "screen session already exists: ${train_session}" >&2
  exit 2
fi
if screen -ls | grep -Fq ".${monitor_session}"; then
  echo "screen session already exists: ${monitor_session}" >&2
  exit 2
fi

train_command=(
  uv run plump train "$run_name" --config "$config_path" "$@"
)
monitor_command=(
  "$project_root/tools/watch_evaluation_dashboard.sh" "$run_name"
)

printf -v train_shell '%q ' "${train_command[@]}"
printf -v monitor_shell '%q ' "${monitor_command[@]}"
screen -dmS "$train_session" \
  bash -lc "cd $(printf %q "$project_root") && exec ${train_shell} >>$(printf %q "$run_path/training-screen.log") 2>&1"
screen -dmS "$monitor_session" \
  bash -lc "cd $(printf %q "$project_root") && exec ${monitor_shell} >>$(printf %q "$run_path/monitor-screen.log") 2>&1"

echo "training: screen -r ${train_session}"
echo "monitor:  screen -r ${monitor_session}"
echo "dashboard: ${run_path}/dashboard.png"
