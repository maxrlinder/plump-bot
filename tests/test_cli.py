"""Unified CLI smoke coverage using an isolated one-iteration CPU run."""

from __future__ import annotations

import csv
import json

import torch

from plump.cli import (
    METRIC_COLUMNS,
    _apply_completed_evaluations,
    _discard_reporting_after,
    _ensure_metrics_header,
    _truncate_metrics_after,
    main,
)
from plump.runs import RunDirectory

TINY_OVERRIDES = (
    "run.iterations=1",
    "run.checkpoint_every=1",
    "run.dashboard_every=0",
    'run.device="cpu"',
    "model.d_model=16",
    "model.n_layers=1",
    "model.n_heads=2",
    "model.n_kv_heads=1",
    "model.d_ff=32",
    "training.hand_sizes=[3]",
    "training.player_counts=[3]",
    "training.player_count_weights=[1.0]",
    "training.games_per_cell=1",
    "training.reference_rate=0.01",
    "training.exhaustive_until=2",
    # The tiny run wants no branching at all, including the preset's explicit
    # multi-arm bid exploration.
    'training.bid_mode="sample_k"',
    "training.bid_top_k=1",
    'training.play_mode="none"',
    "training.play_top_k=1",
    "training.microbatch_positions=256",
    "training.suit_coef=0.0",
    "training.bid_hit_coef=0.0",
    'rollout.opponent_mode="off"',
    "rollout.opponent_fraction=0.0",
    "rollout.auto_deals_per_batch=false",
    "rollout.max_cache_rows=256",
    "evaluation.every=0",
    'evaluation.training_action_mode="argmax"',
    "evaluation.deals=1",
)


def _train_args(name: str) -> list[str]:
    args = ["train", name]
    for override in TINY_OVERRIDES:
        args.extend(("--set", override))
    return args


def test_legacy_metrics_header_is_upgraded_for_reporting_fields(tmp_path):
    metrics = tmp_path / "metrics.csv"
    new_fields = {
        "bid_hit_focal",
        "bid_hit_non_focal",
        "reward_focal",
        "reward_non_focal",
        "loss_value_zero",
        "auxiliary_learning_rate",
        "value_rmse",
        "value_zero_rmse",
        "value_correlation",
        "value_prediction_std",
        "value_rows",
        "proposed_policy_kl",
        "proposed_policy_kl_p95",
        "proposed_policy_kl_p99",
        "proposed_policy_kl_max",
        "proposed_mean_exceeded",
        "proposed_p99_exceeded",
        "core_grad_norm",
        "auxiliary_grad_norm",
        "trees_self",
        "trees_heuristic",
        "trees_historical",
        "reward_heuristic",
        "opponent_phase",
        "heuristic_eval_win_streak",
        "critic_all_player_rmse",
        "critic_all_player_correlation",
        "critic_loss_first_epoch",
        "critic_loss_last_epoch",
        "critic_loss_reduction",
    }
    legacy = tuple(column for column in METRIC_COLUMNS if column not in new_fields)
    with metrics.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=legacy)
        writer.writeheader()
        writer.writerow({"iteration": 1, "bid_hit_rate": 0.25})

    _ensure_metrics_header(metrics)

    with metrics.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert tuple(reader.fieldnames) == METRIC_COLUMNS
    assert rows[0]["bid_hit_rate"] == "0.25"
    assert all(rows[0][field] == "" for field in new_fields)


def test_resume_discards_only_post_checkpoint_metric_rows(tmp_path):
    metrics = tmp_path / "metrics.csv"
    with metrics.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("iteration", "loss_policy"))
        writer.writeheader()
        writer.writerows(
            (
                {"iteration": 49, "loss_policy": 1.0},
                {"iteration": 50, "loss_policy": 0.9},
                {"iteration": 51, "loss_policy": 0.8},
            )
        )

    assert _truncate_metrics_after(metrics, 50) == 1
    with metrics.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["iteration"] for row in rows] == ["49", "50"]
    assert _truncate_metrics_after(metrics, 50) == 0


def test_resume_discards_stale_evaluation_rows_and_derived_state(tmp_path):
    run = RunDirectory("stale", root=tmp_path)
    run.evaluations.mkdir(parents=True)
    run.checkpoints.mkdir()
    with run.metrics.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("iteration",))
        writer.writeheader()
        writer.writerows({"iteration": value} for value in (1, 2, 3))
    (run.evaluations / "iter_000001").mkdir()
    (run.evaluations / "iter_000003").mkdir()
    run.evaluation_state.write_text("{}")
    (run.evaluations / "monitor.json").write_text("{}")
    (run.checkpoints / "best.json").write_text(
        json.dumps({"iteration": 3})
    )
    (run.checkpoints / "best.pt").write_bytes(b"stale")

    rows, directories = _discard_reporting_after(run, 2)

    assert (rows, directories) == (1, 1)
    assert (run.evaluations / "iter_000001").is_dir()
    assert not (run.evaluations / "iter_000003").exists()
    assert not run.evaluation_state.exists()
    assert not (run.evaluations / "monitor.json").exists()
    assert not (run.checkpoints / "best.json").exists()
    assert not (run.checkpoints / "best.pt").exists()


def test_checkpoint_selection_and_curriculum_gate_use_argmax(tmp_path):
    run = RunDirectory("argmax-state", root=tmp_path)
    run.evaluations.mkdir(parents=True)
    run.checkpoints.mkdir()
    for iteration, sample_reward, argmax_reward in (
        (1, 1.0, -0.2),
        (2, 0.9, -0.1),
    ):
        checkpoint = run.interval_checkpoint(iteration)
        checkpoint.write_bytes(f"checkpoint-{iteration}".encode())
        directory = run.evaluations / f"iter_{iteration:06d}"
        directory.mkdir()
        (directory / "heuristic_sample.json").write_text(
            json.dumps(
                {
                    "report": {
                        "macro_relative_reward": sample_reward,
                        "macro_bid_hit_rate": 0.5,
                    }
                }
            )
        )
        (directory / "heuristic.json").write_text(
            json.dumps(
                {
                    "report": {
                        "macro_relative_reward": argmax_reward,
                        "macro_bid_hit_rate": 0.4,
                    }
                }
            )
        )

    state = _apply_completed_evaluations(
        run,
        opponent_mode="heuristic_then_historical",
        switch_reward=0.0,
        switch_consecutive=2,
        maximum_iteration=2,
    )

    best = json.loads((run.checkpoints / "best.json").read_text())
    assert best["iteration"] == 2
    assert best["action_mode"] == "argmax"
    assert state["opponent_phase"] == "heuristic"
    assert state["win_streak"] == 0

    fixed_anchor_state = _apply_completed_evaluations(
        run,
        opponent_mode="heuristic",
        switch_reward=0.0,
        switch_consecutive=1,
        maximum_iteration=2,
        gate_action_mode="sample",
    )
    assert fixed_anchor_state["opponent_phase"] == "heuristic"
    assert fixed_anchor_state["win_streak"] == 0


def test_cli_tiny_run_resume_and_mismatch(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PLUMP_RUNS_DIR", str(tmp_path))

    assert main(_train_args("tiny")) == 0
    run = tmp_path / "tiny"
    assert (run / "checkpoints" / "iter_000001.pt").is_file()
    assert (run / "checkpoints" / "latest.json").is_file()
    assert (run / "metrics.csv").read_text().count("\n") == 2
    metadata = json.loads((run / "metadata.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["device"] == "cpu"
    assert metadata["seed"] == 0
    assert metadata["command"] == ["plump", *_train_args("tiny")]
    payload = torch.load(
        run / "checkpoints" / "iter_000001.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert payload["resolved_config"]["run"]["iterations"] == 1

    fork_args = _train_args("forked")
    fork_args.extend(
        (
            "--set",
            "training.suit_coef=0.1",
            "--from-checkpoint",
            str(run / "checkpoints" / "iter_000001.pt"),
        )
    )
    assert main(fork_args) == 0
    forked_checkpoint = tmp_path / "forked" / "checkpoints" / "iter_000001.pt"
    forked = torch.load(
        forked_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    assert forked["iteration"] == 1
    assert forked["training_config"]["suit_coef"] == 0.1
    assert forked["resolved_config"]["training"]["suit_coef"] == 0.1

    assert main(_train_args("tiny")) == 0
    assert (run / "metrics.csv").read_text().count("\n") == 2
    assert "Resumed tiny" in capsys.readouterr().out

    changed = _train_args("tiny")
    changed.extend(("--set", "training.learning_rate=0.1"))
    assert main(changed) == 2
    assert "training.learning_rate" in capsys.readouterr().err


def test_cli_prepare_only_forks_for_unavailable_device_and_resets_league(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PLUMP_RUNS_DIR", str(tmp_path))
    assert main(_train_args("prepare-source")) == 0
    source = tmp_path / "prepare-source" / "checkpoints" / "iter_000001.pt"

    prepared_args = _train_args("prepared-cuda")
    prepared_args.extend(
        (
            "--set",
            "run.iterations=3",
            "--set",
            'run.device="cuda"',
            "--from-checkpoint",
            str(source),
            "--reset-league",
            "--prepare-only",
        )
    )
    assert main(prepared_args) == 0

    run = tmp_path / "prepared-cuda"
    metadata = json.loads((run / "metadata.json").read_text())
    assert metadata["status"] == "prepared"
    assert metadata["prepared_iteration"] == 1
    assert metadata["target_device"] == "cuda"
    assert metadata["target_iterations"] == 3
    assert (run / "metrics.csv").read_text().count("\n") == 1
    latest = json.loads((run / "checkpoints" / "latest.json").read_text())
    payload = torch.load(
        run / "checkpoints" / latest["path"],
        map_location="cpu",
        weights_only=False,
    )
    assert payload["iteration"] == 1
    assert payload["resolved_config"]["run"]["device"] == "cuda"
    assert payload["resolved_config"]["run"]["iterations"] == 3
    assert payload["league"] == []


def test_monitor_evaluates_initial_checkpoint_outside_training(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("PLUMP_RUNS_DIR", str(tmp_path))
    args = _train_args("initial-eval")
    args.extend(
        (
            "--set",
            "evaluation.every=1",
            "--set",
            'evaluation.training_action_mode="sample"',
        )
    )

    assert main(args) == 0
    training_output = capsys.readouterr().out
    assert "eval" not in next(
        line for line in training_output.splitlines() if "iter     1" in line
    )
    assert main(["monitor", "initial-eval", "--device", "cpu"]) == 0

    run = tmp_path / "initial-eval"
    baseline = (
        run
        / "evaluations"
        / "iter_000000"
        / "heuristic_sample.json"
    )
    payload = json.loads(baseline.read_text())
    assert payload["iteration"] == 0
    assert payload["protocol"]["greedy"] is False
    assert payload["report"]["rounds"] == 9
    argmax = run / "evaluations" / "iter_000000" / "heuristic.json"
    assert json.loads(argmax.read_text())["protocol"]["greedy"] is True
    with (run / "metrics.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    # Scheduled inference is checkpoint-scoped and asynchronous; exact points
    # live in sidecars instead of delaying and then rewriting the update row.
    assert not row["eval_reward_vs_heuristic_sample"]
    assert not row["eval_bid_hit_sample"]
    assert not row["eval_reward_vs_heuristic_argmax"]
    assert not row["eval_bid_hit_argmax"]
    best = json.loads((run / "checkpoints" / "best.json").read_text())
    assert best["iteration"] in (0, 1)
    assert best["action_mode"] == "argmax"
    output = capsys.readouterr().out
    assert "iter_000000.pt [sample] evaluated" in output
    assert "iter_000000.pt [argmax] evaluated" in output
    assert "Applied paired evals through 1" in output


def test_cli_reconfigure_writes_resume_checkpoint_and_config_audit(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PLUMP_RUNS_DIR", str(tmp_path))
    assert main(_train_args("tiny-reconfigure")) == 0

    changed = _train_args("tiny-reconfigure")
    changed.extend(
        (
            "--set",
            "training.learning_rate=0.0003",
            "--reconfigure",
            "--reconfigure-reason",
            "test curriculum migration",
        )
    )
    assert main(changed) == 0

    run = tmp_path / "tiny-reconfigure"
    latest = json.loads((run / "checkpoints" / "latest.json").read_text())
    assert latest["path"].startswith("resume_000001_reconfigured")
    assert (run / "checkpoints" / latest["path"]).is_file()
    assert list((run / "config-history").glob("iter_000001*.toml"))
    metadata = json.loads((run / "metadata.json").read_text())
    migration = metadata["config_migrations"][-1]
    assert migration["reason"] == "test curriculum migration"
    assert any("training.learning_rate" in line for line in migration["changes"])
