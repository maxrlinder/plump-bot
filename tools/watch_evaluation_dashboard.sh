#!/usr/bin/env bash
set -u

if [[ $# -lt 1 ]]; then
  echo "usage: $0 RUN [plump monitor options...]" >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_name="$1"
shift
poll_seconds="${PLUMP_MONITOR_POLL_SECONDS:-10}"
log_path="${PLUMP_RUNS_DIR:-${project_root}/runs}/${run_name}/evaluation.log"
lock_path="${PLUMP_RUNS_DIR:-${project_root}/runs}/${run_name}/evaluations/watcher.lock"

cd "$project_root"
mkdir -p "$(dirname "$log_path")"
mkdir -p "$(dirname "$lock_path")"
if ! mkdir "$lock_path" 2>/dev/null; then
  existing_pid="$(<"$lock_path/pid" 2>/dev/null || true)"
  if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "evaluation/dashboard watcher already running as PID ${existing_pid}" >&2
    exit 2
  fi
  rm -f "$lock_path/pid"
  rmdir "$lock_path" 2>/dev/null || true
  mkdir "$lock_path"
fi
printf '%s\n' "$$" >"$lock_path/pid"
cleanup_lock() {
  rm -f "$lock_path/pid"
  rmdir "$lock_path" 2>/dev/null || true
}
trap cleanup_lock EXIT
trap 'exit 0' INT TERM

while true; do
  # This is intentionally a fresh process each pass: evaluation and dashboard
  # code/config edits take effect without touching the training process.
  uv run plump monitor "$run_name" "$@" >>"$log_path" 2>&1
  status=$?
  if [[ $status -ne 0 ]]; then
    printf '[%s] monitor pass failed with exit %s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >>"$log_path"
  fi
  sleep "$poll_seconds"
done
