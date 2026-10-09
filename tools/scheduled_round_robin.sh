#!/usr/bin/env bash
# Wait a fixed delay, then play a lineage round robin that includes whatever
# checkpoints exist by then.
#
# The roster is a stride over the lineage plus the newest checkpoint, so this
# picks up every checkpoint training writes between now and the launch time
# without anyone having to name them. Runs on CPU at nice, so training keeps
# the accelerator and the two can overlap.
#
#   tools/scheduled_round_robin.sh [run] [delay] [from] [stride] [deals] [waves]
#
#   delay accepts 38h, 90m, or plain seconds. Everything after it is optional.
#
# Launch detached so it survives the terminal:
#
#   screen -dmS plump-rr tools/scheduled_round_robin.sh ppo-oracle-mps-768-v4-5 38h
#
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$project_root"

run="${1:-ppo-oracle-mps-768-v4-5}"
delay="${2:-38h}"
from_iteration="${3:-38700}"
stride="${4:-1600}"
deals_per_match="${5:-512}"
waves="${6:-2}"
threads="${OMP_NUM_THREADS:-8}"

python_bin="$project_root/.venv/bin/python"
[ -x "$python_bin" ] || { echo "no interpreter at $python_bin" >&2; exit 1; }

# Deliberately not `uv run`: the venv carries stale shebangs and uv resolves
# a different interpreter, which is how a watcher silently stopped once.

case "$delay" in
    *h) delay_seconds=$(( ${delay%h} * 3600 )) ;;
    *m) delay_seconds=$(( ${delay%m} * 60 )) ;;
    *s) delay_seconds=$(( ${delay%s} )) ;;
    *)  delay_seconds=$(( delay )) ;;
esac

target=$(( $(date +%s) + delay_seconds ))
out_dir="$project_root/runs/$run/tournaments"
mkdir -p "$out_dir"
log="$out_dir/scheduled-round-robin.log"

say() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$log"; }

say "scheduled round robin for $run"
say "  waiting ${delay} (${delay_seconds}s); launches at $(date -r "$target" 2>/dev/null || date -d "@$target")"
say "  roster: from ${from_iteration} stride ${stride}, plus the newest checkpoint at launch"
say "  ${deals_per_match} deals/match, ${waves} waves, no heuristic, cpu, ${threads} threads"

# Poll rather than one long sleep: survives a laptop suspend, and the log
# shows it is still alive.
while [ "$(date +%s)" -lt "$target" ]; do
    remaining=$(( target - $(date +%s) ))
    if [ $(( remaining % 3600 )) -lt 300 ]; then
        say "  ${remaining}s to launch (newest checkpoint: $(
            "$python_bin" -c "
import json,sys
p='runs/$run/checkpoints/latest.json'
try: print(json.load(open(p))['iteration'])
except Exception: print('?')" ))"
    fi
    sleep 300
done

latest=$("$python_bin" -c "
import json
print(json.load(open('runs/$run/checkpoints/latest.json'))['iteration'])")
name="rr-${from_iteration}-to-${latest}-stride${stride}"

say "launching: newest checkpoint is ${latest}, output stem ${name}"

common=(
    "$python_bin" -m plump.cli tournament "$run"
    --from-iteration "$from_iteration"
    --to-iteration "$latest"
    --stride "$stride"
    --deals-per-match "$deals_per_match"
    --waves "$waves"
    --no-heuristic
    --device cpu
    --name "$name"
)

say "--- plan ---"
"${common[@]}" --dry-run 2>&1 | tee -a "$log"

say "--- playing ---"
OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads" \
    nice -n 10 "${common[@]}" 2>&1 | tee -a "$log"

say "done: $out_dir/$name.{csv,json,png}"
