"""Unified CLI smoke coverage using an isolated one-iteration CPU run."""

from __future__ import annotations

import json

import torch

from plump.cli import main

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
    'rollout.historical_arm="off"',
    "rollout.auto_deals_per_batch=false",
    "rollout.max_cache_rows=256",
    "evaluation.every=0",
    "evaluation.deals=1",
)


def _train_args(name: str) -> list[str]:
    args = ["train", name]
    for override in TINY_OVERRIDES:
        args.extend(("--set", override))
    return args


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
