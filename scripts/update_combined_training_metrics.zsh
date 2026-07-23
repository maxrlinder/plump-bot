#!/bin/zsh
set -u

repo_dir="${0:A:h:h}"
modal_dir="${PLUMP_MODAL_METRICS_DIR:-$repo_dir/checkpoints/modal/v9_8m_wideppo_seed1}"
local_dir="${PLUMP_LOCAL_METRICS_DIR:-$repo_dir/checkpoints/local/v9_8m_laptop_seed1}"
interval_seconds="${PLOT_INTERVAL_SECONDS:-15}"
output="$modal_dir/metrics_new.png"
combined_csv="$modal_dir/metrics_combined.csv"
log_file="$modal_dir/metrics-combined-updater.log"

cd "$repo_dir" || exit 1
mkdir -p "$modal_dir"

print -r -- \
  "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] combined metrics updater started" \
  >> "$log_file"
while true; do
  timestamp="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  if "$repo_dir/.venv/bin/python" \
      "$repo_dir/scripts/render_combined_training_metrics.py" \
      --modal-dir "$modal_dir" \
      --local-dir "$local_dir" \
      --output "$output" \
      --combined-csv "$combined_csv" \
      --smooth 50 \
      --diagnostic-smooth 50 \
      >> "$log_file" 2>&1; then
    print -r -- "[$timestamp] refreshed" >> "$log_file"
  else
    status=$?
    print -r -- \
      "[$timestamp] refresh failed with status $status" \
      >> "$log_file"
  fi

  sleep "$interval_seconds"
done
