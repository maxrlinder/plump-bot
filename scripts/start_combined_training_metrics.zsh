#!/bin/zsh
set -euo pipefail

repo_dir="${0:A:h:h}"
screen_name="plump_metrics_combined"
supervisor="$repo_dir/scripts/supervise_combined_training_metrics.py"

screens="$(/usr/bin/screen -ls 2>&1 || true)"
if [[ "$screens" == *".$screen_name"* ]]; then
  print -u2 -r -- "Metrics updater is already running in screen $screen_name"
  exit 1
fi

/usr/bin/screen -dmS "$screen_name" "$repo_dir/.venv/bin/python" "$supervisor"
print -r -- "started continuously supervised metrics updater screen=$screen_name interval=15s"
