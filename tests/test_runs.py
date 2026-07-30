"""Run creation, manifests, and full schema-v6 checkpoint persistence."""

from __future__ import annotations

import json
import random
import shutil

import numpy as np
import pytest
import torch

from plump.run_config import (
    CHECKOUT_CONFIG_PATH,
    PACKAGED_CONFIG_PATH,
    config_diff,
    load_training_config,
)
from plump.runs import RunDirectory
from plump.seq.config import (
    BranchBudgetConfig,
    GameScheduleCell,
    SeqModelConfig,
    SeqTrainingConfig,
)
from plump.seq.model import SeqPlumpModel
from plump.seq.trainer import SeqTrainer

MODEL = SeqModelConfig(
    d_model=16,
    n_layers=1,
    n_heads=2,
    n_kv_heads=1,
    d_ff=32,
)
TRAIN = SeqTrainingConfig(
    schedule_cells=(GameScheduleCell(hand_size=3, num_players=3),),
    branch_budget=BranchBudgetConfig(branch_rate=0.5),
    microbatch_positions=256,
    seed=13,
)


def _trainer() -> SeqTrainer:
    torch.manual_seed(2)
    return SeqTrainer(SeqPlumpModel(MODEL), TRAIN, device="cpu")


def test_run_creation_records_config_and_rejects_field_changes(tmp_path):
    assert CHECKOUT_CONFIG_PATH.read_bytes() == PACKAGED_CONFIG_PATH.read_bytes()
    resolved = load_training_config()
    assert resolved.training.policy_objective == "neurd"
    assert resolved.training.branch_depth_exponent == -0.5
    assert resolved.training.value_objective == "mse"
    assert resolved.training.value_positions == "policy"
    assert resolved.training.value_reward_scale == 5.0
    assert resolved.training.core_lr == 2.5e-5
    assert resolved.training.auxiliary_lr == 2e-4
    assert resolved.training.kl_backtrack_attempts == 8
    assert resolved.training.branch_rule.bid_rule() == (
        "stratified",
        5,
    )
    assert resolved.training.branch_rule.play_rule_for_trick(0) == ("stratified", 4)
    assert resolved.training.checkpoint_every == 50
    assert resolved.training.suit_coef == 0.05
    assert resolved.training.trick_coef == 0.05
    assert resolved.training.rollout.historical_arm == "off"
    run = RunDirectory("unit-run", root=tmp_path)

    with run.acquire_lock():
        run.create(resolved.raw, ["plump", "train", "unit-run"])

    assert run.recorded_config() == resolved.raw
    metadata = json.loads(run.metadata.read_text())
    assert metadata["command"] == ["plump", "train", "unit-run"]
    assert metadata["git"]["commit"]
    assert metadata["python"]
    assert metadata["torch"]

    changed = load_training_config(overrides=["training.learning_rate=0.125"])
    assert config_diff(run.recorded_config(), changed.raw) == [
        "training.learning_rate: recorded=0.0002 requested=0.125"
    ]


def test_checkpoint_roundtrip_restores_rng_collector_and_relocates_league(
    tmp_path,
):
    original = tmp_path / "original" / "checkpoints"
    original.mkdir(parents=True)
    trainer = _trainer()
    trainer.iteration = 7
    trainer.optimizer_steps = 3
    trainer.collector._peak_rows = {("self", 3, 3): 19}
    trainer.collector._rows_per_deal = {(3, 3): 4.5}
    trainer.collector._seat_cursor = {3: 2}

    reference = original / "iter_000006.pt"
    trainer.save_checkpoint(reference)
    trainer.league.add("iter_6", str(reference), 6)

    trainer.rng.seed(23)
    expected_trainer = random.Random()
    expected_trainer.setstate(trainer.rng.getstate())
    expected_trainer_value = expected_trainer.random()

    random.seed(29)
    python_state = random.getstate()
    expected_python_value = random.random()
    random.setstate(python_state)

    np.random.seed(31)
    numpy_state = np.random.get_state()
    expected_numpy_value = np.random.random()
    np.random.set_state(numpy_state)

    torch.manual_seed(37)
    torch_state = torch.get_rng_state()
    expected_torch_value = torch.rand(3)
    torch.set_rng_state(torch_state)

    checkpoint = original / "iter_000007.pt"
    trainer.save_checkpoint(checkpoint)

    relocated_root = tmp_path / "relocated"
    shutil.copytree(original.parent, relocated_root)
    relocated = relocated_root / "checkpoints" / checkpoint.name

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    fresh = _trainer()
    fresh.load_checkpoint(relocated)

    assert fresh.iteration == 7
    assert fresh.optimizer_steps == 3
    assert fresh.collector._peak_rows == {("self", 3, 3): 19}
    assert fresh.collector._rows_per_deal == {(3, 3): 4.5}
    assert fresh.collector._seat_cursor == {3: 2}
    assert fresh.rng.random() == expected_trainer_value
    assert random.random() == expected_python_value
    assert np.random.random() == expected_numpy_value
    torch.testing.assert_close(torch.rand(3), expected_torch_value)
    assert fresh.league.snapshots[0].path == str(
        relocated_root / "checkpoints" / reference.name
    )


def test_atomic_checkpoint_validation_keeps_existing_file(tmp_path, monkeypatch):
    trainer = _trainer()
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"keep-me")

    monkeypatch.setattr(
        "plump.seq.trainer.torch.load",
        lambda *args, **kwargs: {"schema_version": -1},
    )
    with pytest.raises(RuntimeError, match="validation"):
        trainer.save_checkpoint(path)

    assert path.read_bytes() == b"keep-me"
    assert not list(tmp_path.glob(".*.tmp"))


def test_latest_best_resolution_and_corruption_detection(tmp_path):
    run = RunDirectory("manifest-run", root=tmp_path)
    resolved = load_training_config()
    with run.acquire_lock():
        run.create(resolved.raw, ["plump", "train", "manifest-run"])

    first = run.interval_checkpoint(1)
    second = run.interval_checkpoint(2)
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    run.record_latest(second, 2)
    run.record_best(first, 1, 0.25)

    assert run.resolve_checkpoint("latest") == second.resolve()
    assert run.resolve_checkpoint("best") == first.resolve()
    assert run.resolve_checkpoint(1) == first.resolve()
    assert first.exists() and second.exists()

    second.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="size mismatch"):
        run.resolve_checkpoint("latest")
