#!/usr/bin/env python3
"""Keep the combined training-metrics updater alive in a detached session."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
import signal
import subprocess
import time


REPO_DIR = Path(__file__).resolve().parent.parent
UPDATER = REPO_DIR / "scripts" / "update_combined_training_metrics.zsh"
MODAL_DIR = Path(
    os.environ.get(
        "PLUMP_MODAL_METRICS_DIR",
        REPO_DIR / "checkpoints" / "modal" / "v9_8m_wideppo_seed1",
    )
)
SUPERVISOR_LOG = MODAL_DIR / "metrics-combined-supervisor.log"
RESTART_DELAY = float(os.environ.get("PLOT_RESTART_DELAY_SECONDS", "5"))

stopping = False
child: subprocess.Popen[bytes] | None = None


def timestamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str) -> None:
    MODAL_DIR.mkdir(parents=True, exist_ok=True)
    with SUPERVISOR_LOG.open("a") as file:
        file.write(f"[{timestamp()}] {message}\n")


def stop(_signum: int, _frame: object) -> None:
    global stopping
    stopping = True
    if child is not None and child.poll() is None:
        child.terminate()


signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGHUP, stop)

while not stopping:
    environment = os.environ.copy()
    environment.setdefault("PLOT_INTERVAL_SECONDS", "15")
    log("metrics supervisor starting updater")
    child = subprocess.Popen(
        ["/bin/zsh", str(UPDATER)],
        cwd=REPO_DIR,
        env=environment,
    )
    status = child.wait()
    child = None
    if stopping:
        break
    log(f"updater exited status={status}; restarting in {RESTART_DELAY:g}s")
    deadline = time.monotonic() + RESTART_DELAY
    while not stopping and time.monotonic() < deadline:
        time.sleep(min(0.1, deadline - time.monotonic()))
