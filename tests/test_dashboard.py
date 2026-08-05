"""Static dashboard rendering tolerates sparse and resumed metrics."""

from __future__ import annotations

import csv
import json

import numpy as np

from plump.dashboard import (
    DEFAULT_SMOOTH_WINDOW,
    _duration,
    _evaluation_points,
    _has_signal,
    _smooth,
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
                "value_rmse",
                "value_zero_rmse",
                "value_correlation",
                "loss_suit",
                "loss_trick",
                "loss_oracle_trick",
                "suit_accuracy_10c_0",
                "suit_accuracy_10c_4",
                "suit_accuracy_10c_8",
                "trick_accuracy_10c_0",
                "trick_accuracy_10c_4",
                "trick_accuracy_10c_8",
                "oracle_trick_accuracy",
                "core_grad_norm",
                "auxiliary_grad_norm",
                "critic_grad_norm",
                "critic_all_player_rmse",
                "critic_all_player_correlation",
                "critic_loss_reduction",
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
                "value_rmse": 4.8,
                "value_zero_rmse": 5.0,
                "value_correlation": 0.1,
                "loss_suit": 0.65,
                "loss_trick": 1.4,
                "loss_oracle_trick": 1.2,
                "suit_accuracy_10c_0": 0.55,
                "suit_accuracy_10c_4": 0.65,
                "suit_accuracy_10c_8": 0.75,
                "trick_accuracy_10c_0": 0.15,
                "trick_accuracy_10c_4": 0.25,
                "trick_accuracy_10c_8": 0.45,
                "oracle_trick_accuracy": 0.5,
                "core_grad_norm": 80.0,
                "auxiliary_grad_norm": 0.08,
                "critic_grad_norm": 1.8,
                "critic_all_player_rmse": 4.7,
                "critic_all_player_correlation": 0.12,
                "critic_loss_reduction": 0.08,
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
                "value_rmse": 4.5,
                "value_zero_rmse": 5.1,
                "value_correlation": 0.2,
                "loss_suit": 0.55,
                "loss_trick": 1.3,
                "loss_oracle_trick": 1.0,
                "suit_accuracy_10c_0": 0.56,
                "suit_accuracy_10c_4": 0.67,
                "suit_accuracy_10c_8": 0.78,
                "trick_accuracy_10c_0": 0.16,
                "trick_accuracy_10c_4": 0.28,
                "trick_accuracy_10c_8": 0.5,
                "oracle_trick_accuracy": 0.58,
                "core_grad_norm": 60.0,
                "auxiliary_grad_norm": 0.06,
                "critic_grad_norm": 1.4,
                "critic_all_player_rmse": 4.3,
                "critic_all_player_correlation": 0.24,
                "critic_loss_reduction": 0.17,
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


def test_dashboard_uses_trailing_fifty_iteration_means_by_default():
    values = np.arange(1.0, 61.0)
    smoothed = _smooth(values, DEFAULT_SMOOTH_WINDOW)

    assert DEFAULT_SMOOTH_WINDOW == 50
    assert smoothed[0] == 1.0
    assert smoothed[49] == np.mean(values[:50])
    assert smoothed[59] == np.mean(values[10:60])


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
            "mode": "argmax",
            "reward": 0.35,
            "bid_hit": 0.42,
            "ci_low": 0.1,
            "ci_high": 0.6,
        }
    ]


def test_dashboard_loads_argmax_and_sample_checkpoint_evaluations(tmp_path):
    evaluations = tmp_path / "evaluations"
    output = evaluations / "iter_000050"
    output.mkdir(parents=True)
    report = {
        "macro_relative_reward": 0.2,
        "macro_bid_hit_rate": 0.4,
        "relative_reward_ci_low": -0.1,
        "relative_reward_ci_high": 0.5,
    }
    (output / "heuristic.json").write_text(
        json.dumps({"iteration": 50, "protocol": {"greedy": True}, "report": report})
    )
    (output / "heuristic_sample.json").write_text(
        json.dumps(
            {
                "iteration": 50,
                "protocol": {"greedy": False},
                "report": {**report, "macro_relative_reward": -0.3},
            }
        )
    )

    points = _evaluation_points([], evaluations)

    assert [(point["mode"], point["reward"]) for point in points] == [
        ("argmax", 0.2),
        ("sample", -0.3),
    ]
