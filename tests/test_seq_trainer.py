"""Trainer: loss normalization, KL-cap rollback, checkpoints, smoke update."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from plump.seq.config import (
    BranchBudgetConfig,
    BranchRuleConfig,
    GameScheduleCell,
    SeqModelConfig,
    SeqTrainingConfig,
)
from plump.seq.model import SeqPlumpModel
from plump.seq.policy import SeqModelPolicy
from plump.seq.trainer import (
    SeqTrainer,
    build_training_groups,
    summarize_trees,
)

MODEL_CONFIG = SeqModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128)


def make_trainer(tmp_path=None, **overrides):
    torch.manual_seed(0)
    defaults = dict(
        schedule_cells=(
            GameScheduleCell(hand_size=3, num_players=3),
            GameScheduleCell(hand_size=4, num_players=3),
        ),
        branch_rule=BranchRuleConfig(bid_top_k=3),
        branch_budget=BranchBudgetConfig(branch_rate=0.5),
        microbatch_positions=512,
        snapshot_every=1,
    )
    defaults.update(overrides)
    train = SeqTrainingConfig(**defaults)
    model = SeqPlumpModel(MODEL_CONFIG)
    return SeqTrainer(model, train, device="cpu")


def test_group_weights_normalize_to_one():
    trainer = make_trainer()
    trees, _ = trainer.collect()
    groups = build_training_groups(trees, MODEL_CONFIG, trainer.train)
    assert groups

    seq_weight_total = 0.0
    spine_weight_total = 0.0
    branch_trees = set()
    for group in groups:
        batch, length = group.tokens.shape[:2]
        owned_counts = group.owned.sum(axis=1)
        seq_weight_total += float((group.seq_weight * owned_counts).sum())
        for rows in group.spine.values():
            spine_weight_total += float(np.sum(rows.weight))
        for rows in group.branch.values():
            if rows.weight:
                branch_trees.add(id(group))
    # Each loss family's weights sum to exactly 1, whatever the exponent and
    # however many trees happen to have rows of that kind. A tree that branched
    # everywhere has no spine rows at all, and the spine loss must not quietly
    # change scale -- an effective learning-rate change -- because a deal came
    # out that way.
    assert seq_weight_total == pytest.approx(1.0)
    assert spine_weight_total == pytest.approx(1.0)


def test_update_produces_finite_losses_and_changes_weights():
    trainer = make_trainer()
    trees, summary = trainer.collect()
    assert summary.trees == len(trees)
    before = [p.detach().clone() for p in trainer.model.parameters()]
    stats = trainer.update(trees)
    assert np.isfinite(stats.loss_value)
    assert np.isfinite(stats.loss_owner)
    assert np.isfinite(stats.loss_trick)
    assert np.isfinite(stats.loss_suit)
    assert np.isfinite(stats.branch_kl)
    assert stats.spine_rows > 0
    assert stats.positions > 0
    if not stats.rolled_back:
        changed = any(
            not torch.equal(a, b)
            for a, b in zip(before, trainer.model.parameters())
        )
        assert changed


def test_kl_cap_rolls_back_the_epoch():
    trainer = make_trainer(
        learning_rate=1.0,  # force a destructive step
        branch_kl_cap=1e-6,
    )
    trees, _ = trainer.collect()
    before = [p.detach().clone() for p in trainer.model.parameters()]
    stats = trainer.update(trees)
    if stats.branch_rows > 0:
        assert stats.rolled_back
        for a, b in zip(before, trainer.model.parameters()):
            torch.testing.assert_close(a, b.detach())


def test_ppo_branch_objective_runs():
    trainer = make_trainer(branch_policy_objective="ppo")
    trees, _ = trainer.collect()
    stats = trainer.update(trees)
    assert np.isfinite(stats.loss_branch) or stats.branch_rows == 0


def test_checkpoint_roundtrip_and_league(tmp_path):
    trainer = make_trainer()
    trees, _ = trainer.collect()
    trainer.update(trees)
    trainer.iteration = 1
    path = trainer.maybe_snapshot(tmp_path)
    assert path is not None
    assert trainer.league.has_snapshots()

    policy = SeqModelPolicy.from_checkpoint(path, device="cpu")
    for a, b in zip(policy.model.parameters(), trainer.model.parameters()):
        torch.testing.assert_close(a, b.detach())

    fresh = make_trainer()
    fresh.load_checkpoint(path)
    assert fresh.iteration == 1
    for a, b in zip(fresh.model.parameters(), trainer.model.parameters()):
        torch.testing.assert_close(a.detach(), b.detach())


def test_league_opponents_join_collection(tmp_path):
    trainer = make_trainer(tmp_path)
    trees, _ = trainer.collect()
    trainer.update(trees)
    trainer.iteration = 1
    trainer.maybe_snapshot(tmp_path)
    trees, summary = trainer.collect()
    arms = {tree.arm for tree in trees}
    assert arms == {"self", "historical"}
    assert summary.reward_historical != 0.0 or summary.reward_self != 0.0


@pytest.mark.parametrize(
    "exponent, expect_ratio", [(0.0, 1.0), (0.5, None), (1.0, None)]
)
def test_tree_weight_exponent_shifts_weight_toward_larger_trees(
    exponent, expect_ratio
):
    """The exponent decides how much of the gradient follows compute."""

    trainer = make_trainer(
        schedule_cells=(
            GameScheduleCell(hand_size=3, num_players=3),
            GameScheduleCell(hand_size=8, num_players=5),
        ),
        branch_budget=BranchBudgetConfig(branch_rate=1.0),
        tree_weight_exponent=exponent,
    )
    trees, _ = trainer.collect()
    groups = build_training_groups(trees, MODEL_CONFIG, trainer.train)

    # Total weight each tree receives on the aux/value losses.
    totals = {}
    rows = {}
    for group in groups:
        owned_counts = group.owned.sum(axis=1)
        for row, weight in enumerate(group.seq_weight):
            key = (group.num_players, group.hand_size)
            totals[key] = totals.get(key, 0.0) + float(weight * owned_counts[row])
            rows[key] = rows.get(key, 0) + int(owned_counts[row])
    assert sum(totals.values()) == pytest.approx(1.0)

    small, big = (3, 3), (5, 8)
    assert rows[big] > rows[small]
    ratio = totals[big] / totals[small]
    if expect_ratio is not None:
        # Exponent 0: size is irrelevant, both trees weigh the same.
        assert ratio == pytest.approx(expect_ratio)
    else:
        # Otherwise the big tree's share grows as (rows ratio) ** exponent.
        assert ratio == pytest.approx((rows[big] / rows[small]) ** exponent, rel=1e-4)


def _shape_weight_totals(trainer, trees):
    groups = build_training_groups(trees, MODEL_CONFIG, trainer.train)
    totals = {}
    for group in groups:
        owned_counts = group.owned.sum(axis=1)
        key = (group.num_players, group.hand_size)
        totals[key] = totals.get(key, 0.0) + float(
            (group.seq_weight * owned_counts).sum()
        )
    return totals


def test_shape_importance_is_independent_of_how_many_rows_a_shape_produced():
    """Equal points per round means equal weight, whatever the tree size."""

    cells = (
        GameScheduleCell(hand_size=3, num_players=3),
        GameScheduleCell(hand_size=9, num_players=5),
    )
    trainer = make_trainer(
        schedule_cells=cells,
        branch_budget=BranchBudgetConfig(branch_rate=1.0),
        tree_weight_exponent=0.0,
    )
    trees, _ = trainer.collect()
    totals = _shape_weight_totals(trainer, trees)
    assert totals[(5, 9)] == pytest.approx(totals[(3, 3)])

    tilted = make_trainer(
        schedule_cells=cells,
        branch_budget=BranchBudgetConfig(branch_rate=1.0),
        tree_weight_exponent=0.0,
        shape_importance_exponent=1.0,
    )
    trees, _ = tilted.collect()
    totals = _shape_weight_totals(tilted, trees)
    # Weight now follows hand size directly, not row count.
    assert totals[(5, 9)] / totals[(3, 3)] == pytest.approx(9 / 3, rel=1e-4)


def test_branch_depth_exponent_moves_weight_without_changing_tree_totals():
    """Depth weighting decides where in a tree the gradient lands, not which
    tree gets more of it."""

    cells = (GameScheduleCell(hand_size=8, num_players=4),)
    budget = BranchBudgetConfig(branch_rate=0.5)

    def branch_rows(exponent):
        trainer = make_trainer(
            schedule_cells=cells,
            branch_budget=budget,
            branch_depth_exponent=exponent,
        )
        trees, _ = trainer.collect()
        groups = build_training_groups(trees, MODEL_CONFIG, trainer.train)
        weights, shallow, deep = [], 0.0, 0.0
        for group in groups:
            for rows in group.branch.values():
                for i, weight in enumerate(rows.weight):
                    weights.append(weight)
                    # Row order follows leaf/decision order, so use the
                    # candidate count as a stand-in for nothing -- instead
                    # split by position within the sequence.
                    if rows.position[i] <= 10:
                        shallow += weight
                    else:
                        deep += weight
        return sum(weights), shallow, deep

    flat_total, flat_shallow, _ = branch_rows(0.0)
    early_total, early_shallow, _ = branch_rows(-1.0)
    # The tree's total branch weight is preserved exactly.
    assert early_total == pytest.approx(flat_total, rel=1e-4)
    # But early positions now carry more of it.
    assert early_shallow > flat_shallow
