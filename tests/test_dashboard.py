"""Static dashboard rendering tolerates sparse and resumed metrics."""

from __future__ import annotations

import csv

import numpy as np

from plump.dashboard import _duration, _has_signal, render_dashboard


def test_dashboard_renders_sparse_partial_and_resumed_rows(tmp_path):
    metrics = tmp_path / "metrics.csv"
    with metrics.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "iteration",
                "elapsed_sec",
                "loss_policy",
                "eval_reward_vs_heuristic",
                "reward_focal",
                "reward_non_focal",
                "bid_hit_focal",
                "bid_hit_non_focal",
                "rolled_back",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "iteration": 1,
                "elapsed_sec": 2.0,
                "loss_policy": 0.5,
                "eval_reward_vs_heuristic": "",
                "reward_focal": -0.2,
                "reward_non_focal": 0.1,
                "bid_hit_focal": 0.25,
                "bid_hit_non_focal": 0.4,
                "rolled_back": 0,
            }
        )
        writer.writerow(
            {
                "iteration": 5,
                "elapsed_sec": 11.0,
                "loss_policy": "",
                "eval_reward_vs_heuristic": 0.2,
                "reward_focal": 0.3,
                "reward_non_focal": -0.15,
                "bid_hit_focal": 0.5,
                "bid_hit_non_focal": 0.45,
                "rolled_back": 1,
            }
        )

    output = tmp_path / "dashboard.png"
    assert render_dashboard(metrics, output, smooth=2, dpi=40) == 2
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not list(tmp_path.glob(".*.tmp.png"))


def test_dashboard_omits_disabled_zero_only_metrics():
    disabled = np.asarray([0.0, 0.0, np.nan])
    active = np.asarray([0.0, 0.1, np.nan])

    assert not _has_signal(disabled, omit_zero=True)
    assert _has_signal(disabled, omit_zero=False)
    assert _has_signal(active, omit_zero=True)


def test_dashboard_formats_elapsed_time_compactly():
    assert _duration(42.0) == "42s"
    assert _duration(90.0) == "1.5m"
    assert _duration(7200.0) == "2.0h"
