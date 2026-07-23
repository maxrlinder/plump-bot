#!/bin/zsh
set -u

repo_dir="${0:A:h:h}"
run_dir="${PLUMP_LAPTOP_RUN_DIR:-$repo_dir/checkpoints/local/v9_8m_laptop_seed1}"
supervisor="$repo_dir/scripts/supervise_training_v9_8m_laptop_seed1.zsh"
target_iteration="${PLUMP_LAPTOP_ITERATIONS:-18000}"
poll_seconds="${PLUMP_LAPTOP_WATCHDOG_POLL_SECONDS:-15}"

target_reached() {
  local latest="$run_dir/latest.json"
  local iteration
  [[ -f "$latest" ]] || return 1
  iteration=$(
    "$repo_dir/.venv/bin/python" -c \
      'import json, sys; print(int(json.load(open(sys.argv[1]))["iteration"]))' \
      "$latest" 2>/dev/null
  ) || return 1
  (( iteration >= target_iteration ))
}

wait_for_expected_process() {
  local pid_file="$1"
  local command_pattern="$2"
  local pid command
  [[ -f "$pid_file" ]] || return 0
  pid="$(<"$pid_file")"
  command="$(ps -p "$pid" -o command= 2>/dev/null)" || return 0
  [[ "$command" == *"$command_pattern"* ]] || return 0
  print -r -- \
    "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] watchdog observing pid=$pid command=$command_pattern" \
    >> "$run_dir/supervisor.log"
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$poll_seconds"
  done
}

mkdir -p "$run_dir"
if target_reached; then
  exit 0
fi

# A launchd-managed watchdog can coexist with the interactive supervisor:
# it only takes over after both that supervisor and any surviving trainer
# have exited, preventing duplicate MPS workloads during handoff.
wait_for_expected_process \
  "$run_dir/supervisor.pid" \
  "supervise_training_v9_8m_laptop_seed1.zsh"
wait_for_expected_process \
  "$run_dir/train.pid" \
  "examples/train_ppo.py"

if target_reached; then
  exit 0
fi

print -r -- \
  "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] watchdog starting replacement supervisor" \
  >> "$run_dir/supervisor.log"
exec zsh "$supervisor"
