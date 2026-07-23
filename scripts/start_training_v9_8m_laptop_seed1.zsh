#!/bin/zsh
set -euo pipefail

repo_dir="${0:A:h:h}"
run_dir="${PLUMP_LAPTOP_RUN_DIR:-$repo_dir/checkpoints/local/v9_8m_laptop_seed1}"
watchdog="$repo_dir/scripts/watch_training_v9_8m_laptop_seed1.zsh"
metrics_starter="$repo_dir/scripts/start_combined_training_metrics.zsh"
screen_name="plump_v9_8m_laptop_seed1"

mkdir -p "$run_dir"
screens="$(/usr/bin/screen -ls 2>&1 || true)"
if [[ "$screens" != *".plump_metrics_combined"* ]]; then
  zsh "$metrics_starter"
  screens="$(/usr/bin/screen -ls 2>&1 || true)"
fi
if [[ "$screens" == *".$screen_name"* ]]; then
  print -u2 -r -- "Laptop training watchdog is already running in screen $screen_name"
  exit 1
fi

/usr/bin/screen -dmS "$screen_name" /bin/zsh "$watchdog"
print -r -- "started laptop training watchdog screen=$screen_name"
print -r -- "logs: $run_dir/train.log"
