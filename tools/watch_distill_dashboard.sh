#!/usr/bin/env bash
# Keep a distillation run's dashboard.png fresh while training runs.
#
# `plump train` never renders a dashboard: dashboard_every is read only by
# `plump monitor`, which also wants to evaluate checkpoints against the
# heuristic. A distillation bootstrap has nothing to evaluate yet -- the
# question is how close the student is to the teacher, which is entirely in
# metrics.csv -- so this just re-renders on an interval and does nothing else.
#
#   tools/watch_distill_dashboard.sh distill-v5 [interval_seconds] [smooth]
set -euo pipefail

RUN="${1:?usage: watch_distill_dashboard.sh RUN [interval] [smooth]}"
INTERVAL="${2:-60}"
SMOOTH="${3:-10}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"

echo "Refreshing runs/${RUN}/dashboard.png every ${INTERVAL}s (smooth=${SMOOTH}). Ctrl-C to stop."
while true; do
    if [ -s "${ROOT}/runs/${RUN}/metrics.csv" ]; then
        # A run mid-write can present a partial final row; skip that tick
        # rather than dying and leaving the dashboard stale forever.
        "${PY}" -m plump.cli dashboard "${RUN}" --smooth "${SMOOTH}" 2>&1 \
            | tail -1 || echo "refresh deferred"
    else
        echo "waiting for runs/${RUN}/metrics.csv"
    fi
    sleep "${INTERVAL}"
done
