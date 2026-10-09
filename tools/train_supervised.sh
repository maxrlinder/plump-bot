#!/usr/bin/env bash
# Run `plump train` under a restart loop that recycles its process.
#
# usage: tools/train_supervised.sh RUN CONFIG [plump train options...]
#
# The trainer is started with --recycle-after (default 1000 iterations, or
# PLUMP_RECYCLE_AFTER). At the first checkpoint past that many iterations it
# exits with status 75 and this loop resumes it from that checkpoint in a
# fresh process. Any other exit status -- 0 when the run reaches its target,
# or a genuine failure -- ends the loop and is passed through unchanged.
#
# Why: native heap that PyTorch's MPS backend leaks per operation is only
# returned to the OS when the process exits. plump.seq.autorelease removes
# ~83% of it in-process; recycling bounds the remainder. The restart costs
# one model load and one warm-up iteration (~40 s) and loses no iterations,
# because the exit happens directly after a checkpoint is written.
set -u

if [[ $# -lt 2 ]]; then
  echo "usage: $0 RUN CONFIG [plump train options...]" >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_name="$1"
config_path="$2"
shift 2
recycle_after="${PLUMP_RECYCLE_AFTER:-1000}"
recycle_status=75

cd "$project_root"
while true; do
  # Not `uv run`: the venv carries stale console-script shebangs.
  "${project_root}/.venv/bin/python" -m plump.cli train "$run_name" \
    --config "$config_path" --recycle-after "$recycle_after" "$@"
  status=$?
  if [[ $status -ne $recycle_status ]]; then
    exit "$status"
  fi
  printf '[%s] recycled %s; resuming from its latest checkpoint\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$run_name"
done
