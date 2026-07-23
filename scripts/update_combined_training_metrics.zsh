#!/bin/zsh
set -u

repo_dir="${0:A:h:h}"
modal_dir="${PLUMP_MODAL_METRICS_DIR:-$repo_dir/checkpoints/modal/v9_8m_wideppo_seed1}"
local_dir="${PLUMP_LOCAL_METRICS_DIR:-$repo_dir/checkpoints/local/v9_8m_laptop_seed1}"
interval_seconds="${PLOT_INTERVAL_SECONDS:-15}"
output="$modal_dir/metrics_new.png"
combined_csv="$modal_dir/metrics_combined.csv"
log_file="$modal_dir/metrics-combined-updater.log"
modal_metrics="$modal_dir/metrics.csv"
local_metrics="$local_dir/metrics.csv"

cd "$repo_dir" || exit 1
mkdir -p "$modal_dir"

print -r -- \
  "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] combined metrics updater started" \
  >> "$log_file"
last_signature=""
while true; do
  timestamp="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  signature="$(
    "$repo_dir/.venv/bin/python" -c \
      'import os, sys; print("|".join(f"{os.stat(path).st_mtime_ns}:{os.stat(path).st_size}" if os.path.exists(path) else "missing" for path in sys.argv[1:]))' \
      "$modal_metrics" "$local_metrics"
  )"
  if [[ "$signature" == "$last_signature" && -s "$output" ]]; then
    sleep "$interval_seconds"
    continue
  fi
  if /usr/bin/nice -n 10 "$repo_dir/.venv/bin/python" \
      "$repo_dir/scripts/render_combined_training_metrics.py" \
      --modal-dir "$modal_dir" \
      --local-dir "$local_dir" \
      --output "$output" \
      --combined-csv "$combined_csv" \
      --smooth 50 \
      --diagnostic-smooth 50 \
      >> "$log_file" 2>&1; then
    last_signature="$signature"
    print -r -- "[$timestamp] refreshed" >> "$log_file"
  else
    status=$?
    print -r -- \
      "[$timestamp] refresh failed with status $status" \
      >> "$log_file"
  fi

  sleep "$interval_seconds"
done
