#!/bin/zsh
# Launch the play GUI against the newest checkpoint saved by laptop training.
# Re-run to pick up a newer checkpoint.
set -euo pipefail

repo_dir="/Users/erehmax/Documents/Projects/Personal/plump-bot"
run_name="${1:-v9_8m_laptop_seed1}"
port="${2:-8765}"
run_dir="${PLUMP_LOCAL_RUN_DIR:-$repo_dir/checkpoints/local/$run_name}"

close_listening_port() {
  local selected_port="$1"
  local listeners remaining
  local -a pids

  listeners="$(lsof -nP -tiTCP:"$selected_port" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -z "$listeners" ]]; then
    return
  fi

  pids=(${(f)listeners})
  print -r -- "Closing existing listener on port $selected_port (PID ${pids[*]})"
  kill -TERM -- $pids 2>/dev/null || true

  for _ in {1..20}; do
    remaining="$(lsof -nP -tiTCP:"$selected_port" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -z "$remaining" ]]; then
      return
    fi
    sleep 0.1
  done

  pids=(${(f)remaining})
  print -r -- "Listener did not exit cleanly; force-closing PID ${pids[*]}"
  kill -KILL -- $pids 2>/dev/null || true
}

checkpoints=("$run_dir"/plump_v4_iter_*.pt(N))
if (( ${#checkpoints} == 0 )); then
  print -r -- "No local checkpoints found in '$run_dir'" >&2
  exit 1
fi
newest="${checkpoints[-1]}"
print -r -- "Using newest local checkpoint: $newest"

close_listening_port "$port"

exec "$repo_dir/.venv/bin/python" "$repo_dir/examples/play_gui.py" \
  --checkpoint "$newest" --port "$port"
