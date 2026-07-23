#!/bin/zsh
set -u

repo_dir="${0:A:h:h}"
run_dir="${PLUMP_LAPTOP_RUN_DIR:-$repo_dir/checkpoints/local/v9_8m_laptop_seed1}"
runner="$repo_dir/scripts/run_training_v9_8m_laptop_seed1.zsh"
max_restarts="${PLUMP_LAPTOP_MAX_RESTARTS:-50}"

mkdir -p "$run_dir"
print -r -- "$$" > "$run_dir/supervisor.pid"
for (( attempt = 1; attempt <= max_restarts; attempt++ )); do
  print -r -- "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] supervisor attempt $attempt/$max_restarts" \
    >> "$run_dir/supervisor.log"
  if zsh "$runner"; then
    print -r -- "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] training completed" \
      >> "$run_dir/supervisor.log"
    exit 0
  fi
  print -r -- "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] training exited; restarting in 15s" \
    >> "$run_dir/supervisor.log"
  sleep 15
done
print -r -- "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] restart budget exhausted" \
  >> "$run_dir/supervisor.log"
exit 1
