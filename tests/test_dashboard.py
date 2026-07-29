"""Static dashboard rendering tolerates sparse and resumed metrics."""

from __future__ import annotations

import csv
import json

import numpy as np

from plump.dashboard import (
    _duration,
    _evaluation_points,
    _has_signal,
    render_dashboard,
)


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
                "policy_kl",
                "policy_kl_p99",
                "policy_kl_max",
                "proposed_policy_kl",
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
                "policy_kl": 0.001,
                "policy_kl_p99": 0.004,
                "policy_kl_max": 0.006,
                "proposed_policy_kl": 0.08,
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
                "policy_kl": 0.002,
                "policy_kl_p99": 0.007,
                "policy_kl_max": 0.009,
                "proposed_policy_kl": 0.12,
                "rolled_back": 1,
            }
        )

    output = tmp_path / "dashboard.png"
    assert render_dashboard(metrics, output, smooth=2, dpi=40) == 2
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with_lr = tmp_path / "dashboard-with-lr.png"
    assert (
        render_dashboard(
            metrics,
            with_lr,
            smooth=2,
            dpi=40,
            include_learning_rate=True,
        )
        == 2
    )
    assert with_lr.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
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


def test_sidecar_evaluation_overrides_inline_checkpoint_score(tmp_path):
    evaluations = tmp_path / "evaluations"
    output = evaluations / "iter_000005" / "heuristic.json"
    output.parent.mkdir(parents=True)
    output.write_text(
        json.dumps(
            {
                "format_version": 1,
                "iteration": 5,
                "report": {
                    "macro_relative_reward": 0.35,
                    "macro_bid_hit_rate": 0.42,
                    "relative_reward_ci_low": 0.1,
                    "relative_reward_ci_high": 0.6,
                },
            }
        )
    )
    rows = [
        {
            "iteration": "5",
            "eval_reward_vs_heuristic": "-0.5",
            "eval_bid_hit": "0.2",
        }
    ]

    points = _evaluation_points(rows, evaluations)

    assert points == [
        {
            "iteration": 5.0,
            "reward": 0.35,
            "bid_hit": 0.42,
            "ci_low": 0.1,
            "ci_high": 0.6,
        }
    ]
