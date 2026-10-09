"""Static dashboard rendering tolerates sparse and resumed metrics."""

from __future__ import annotations

import csv
import json

import numpy as np

from plump.dashboard import (
    DEFAULT_SMOOTH_WINDOW,
    EVALUATION_SMOOTH_WINDOW,
    _checkpoint_evaluation,
    _current_entropy_targets,
    _current_kl_caps,
    _duration,
    _evaluation_points,
    _actor_versus_oracle_value,
    _has_signal,
    _next_evaluation_eta,
    _next_trick_winner_learning,
    _smooth,
    render_dashboard,
)
import matplotlib.pyplot as plt


def test_checkpoint_evaluation_is_marker_free_rolling_five_lines():
    points = [
        {
            "iteration": float(index * 100),
            "mode": "argmax",
            "reward": float(index),
            "bid_hit": float(index) / 10.0,
            "ci_low": float(index) - 0.5,
            "ci_high": float(index) + 0.5,
        }
        for index in range(1, 7)
    ]
    figure, axis = plt.subplots()
    _checkpoint_evaluation(axis, points)
    reward = next(line for line in axis.lines if line.get_label() == "reward · argmax")
    bid = next(
        line
        for line in figure.axes[1].lines
        if line.get_label() == "bid accuracy · argmax"
    )
    assert EVALUATION_SMOOTH_WINDOW == 5
    assert reward.get_marker() == "None"
    assert bid.get_marker() == "None"
    assert np.allclose(reward.get_ydata(), [1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    assert np.allclose(bid.get_ydata(), [0.1, 0.15, 0.2, 0.25, 0.3, 0.4])
    assert not axis.collections
    plt.close(figure)


def test_dashboard_omits_disabled_p99_cap_but_keeps_mean_cap(tmp_path):
    metrics = tmp_path / "metrics.csv"
    metrics.touch()
    (tmp_path / "config.toml").write_text(
        "[training]\npolicy_kl_cap = 0.004\npolicy_kl_p99_cap = 0\n"
    )

    assert _current_kl_caps(metrics) == (0.004, None)


def test_dashboard_reads_current_entropy_targets(tmp_path):
    metrics = tmp_path / "metrics.csv"
    metrics.touch()
    (tmp_path / "config.toml").write_text(
        "[training]\n"
        "ppo_bid_entropy_target = 0.5\n"
        "ppo_play_entropy_target = 0.4\n"
    )

    assert _current_entropy_targets(metrics) == (0.5, 0.4)


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
                "loss_next_winner",
                "loss_oracle_next_winner",
                "loss_oracle_card_order",
                "suit_accuracy_10c_0",
                "suit_accuracy_10c_4",
                "suit_accuracy_10c_8",
                "trick_accuracy_10c_0",
                "trick_accuracy_10c_4",
                "trick_accuracy_10c_8",
                "oracle_trick_accuracy",
                "next_winner_accuracy",
                "oracle_next_winner_accuracy",
                "oracle_card_order_top1_accuracy",
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
                "loss_next_winner": 0.8,
                "loss_oracle_next_winner": 1.5,
                "loss_oracle_card_order": 1.8,
                "suit_accuracy_10c_0": 0.55,
                "suit_accuracy_10c_4": 0.65,
                "suit_accuracy_10c_8": 0.75,
                "trick_accuracy_10c_0": 0.15,
                "trick_accuracy_10c_4": 0.25,
                "trick_accuracy_10c_8": 0.45,
                "oracle_trick_accuracy": 0.5,
                "next_winner_accuracy": 0.45,
                "oracle_next_winner_accuracy": 0.6,
                "oracle_card_order_top1_accuracy": 0.3,
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
                "loss_next_winner": 0.7,
                "loss_oracle_next_winner": 1.3,
                "loss_oracle_card_order": 1.5,
                "suit_accuracy_10c_0": 0.56,
                "suit_accuracy_10c_4": 0.67,
                "suit_accuracy_10c_8": 0.78,
                "trick_accuracy_10c_0": 0.16,
                "trick_accuracy_10c_4": 0.28,
                "trick_accuracy_10c_8": 0.5,
                "oracle_trick_accuracy": 0.58,
                "next_winner_accuracy": 0.5,
                "oracle_next_winner_accuracy": 0.68,
                "oracle_card_order_top1_accuracy": 0.4,
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


def test_next_trick_winner_has_dedicated_actor_oracle_panels():
    rows = [
        {
            "iteration": "1",
            "next_winner_accuracy": "0.5",
            "oracle_next_winner_accuracy": "0.7",
            "loss_next_winner": "0.8",
            "loss_oracle_next_winner": "1.4",
        },
        {
            "iteration": "2",
            "next_winner_accuracy": "0.6",
            "oracle_next_winner_accuracy": "0.75",
            "loss_next_winner": "0.7",
            "loss_oracle_next_winner": "1.2",
        },
    ]
    figure, (accuracy_ax, loss_ax) = plt.subplots(1, 2)
    _next_trick_winner_learning(
        accuracy_ax,
        loss_ax,
        np.asarray([1.0, 2.0]),
        rows,
        smooth=1,
    )

    assert accuracy_ax.get_title() == "Next-trick-winner accuracy"
    assert loss_ax.get_title() == "Next-trick-winner loss"
    assert [line.get_label() for line in accuracy_ax.lines] == ["actor", "oracle"]
    assert [line.get_label() for line in loss_ax.lines] == [
        "actor CE",
        "oracle CE · epoch sum",
    ]
    assert accuracy_ax.get_ylim() == (0.0, 1.0)
    plt.close(figure)


def test_dashboard_formats_elapsed_time_compactly():
    assert _duration(42.0) == "42s"
    assert _duration(90.0) == "1.5m"
    assert _duration(7200.0) == "2.0h"


def test_dashboard_estimates_next_eval_from_pace_since_checkpoint(tmp_path):
    metrics = tmp_path / "metrics.csv"
    (tmp_path / "config.toml").write_text(
        "[run]\niterations = 500\ncheckpoint_every = 100\n"
        "[evaluation]\nevery = 200\n"
    )
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "iter_000100.pt").touch()
    rows = [
        {"iteration": "100", "total_sec": "10"},
        {"iteration": "101", "total_sec": "20"},
        {"iteration": "102", "total_sec": "30"},
    ]

    assert _next_evaluation_eta(rows, metrics) == (200, 98, 25.0, 100)


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


def test_value_comparison_panel_starts_where_the_actor_probe_does():
    """The oracle has tens of thousands of iterations the comparison cannot use."""

    rows = []
    for iteration in range(1, 501):
        started = iteration >= 450
        rows.append(
            {
                "iteration": str(iteration),
                "critic_all_player_rmse": "6.0",
                "actor_all_player_rmse": (
                    f"{6.5 - 0.001 * (iteration - 450)}" if started else ""
                ),
                "actor_value_zero_rmse": "6.6",
                "oracle_value_loss": "0.73",
                "actor_value_loss": "0.9" if started else "",
            }
        )
    iteration = np.asarray([float(row["iteration"]) for row in rows])

    figure, axis = plt.subplots()
    _actor_versus_oracle_value(axis, iteration, rows, smooth=1)
    left, right = axis.get_xlim()
    assert 400 < left < 450, "the panel is still showing pre-probe history"
    assert right >= 500
    plt.close(figure)


def test_value_comparison_panel_keeps_full_range_without_a_probe():
    rows = [
        {"iteration": str(index), "critic_all_player_rmse": "6.0"}
        for index in range(1, 21)
    ]
    iteration = np.asarray([float(row["iteration"]) for row in rows])

    figure, axis = plt.subplots()
    _actor_versus_oracle_value(axis, iteration, rows, smooth=1)
    left, _ = axis.get_xlim()
    assert left <= 1.0
    plt.close(figure)
