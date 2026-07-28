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
from plump.seq.tokens import IGNORE_LABEL
from plump.seq.trainer import (
    SeqTrainer,
    _rows_by_chunk,
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
    policy_weight_total = 0.0
    for group in groups:
        owned_counts = group.owned.sum(axis=1)
        seq_weight_total += float((group.seq_weight * owned_counts).sum())
        for rows in group.policy.values():
            policy_weight_total += float(np.sum(rows.weight))
    # Each loss family's weights sum to exactly 1, whatever the exponent and
    # however many trees happen to have rows of that kind, so the effective
    # learning rate cannot drift with how the deals happened to come out.
    assert seq_weight_total == pytest.approx(1.0)
    assert policy_weight_total == pytest.approx(1.0)


def test_update_produces_finite_losses_and_changes_weights():
    # Trick count on as well, so every label path stays exercised even though
    # the default objective leaves it at zero.
    trainer = make_trainer(trick_coef=0.25)
    trees, summary = trainer.collect()
    assert summary.trees == len(trees)
    before = [p.detach().clone() for p in trainer.model.parameters()]
    stats = trainer.update(trees)
    assert np.isfinite(stats.loss_value)
    assert np.isfinite(stats.loss_trick)
    assert np.isfinite(stats.loss_suit)
    assert np.isfinite(stats.loss_bid_hit)
    assert stats.loss_trick > 0.0
    assert stats.loss_suit > 0.0
    assert stats.loss_bid_hit > 0.0
    assert np.isfinite(stats.policy_kl)
    assert stats.policy_rows > 0
    assert stats.policy_rows == stats.branched_rows + stats.unbranched_rows
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
        policy_kl_cap=1e-6,
    )
    trees, _ = trainer.collect()
    before = [p.detach().clone() for p in trainer.model.parameters()]
    stats = trainer.update(trees)
    if stats.policy_rows > 0:
        assert stats.rolled_back
        for a, b in zip(before, trainer.model.parameters()):
            torch.testing.assert_close(a, b.detach())


def test_update_with_belief_losses_disabled_is_policy_value_only():
    """The initial objective: PPO + NeuRD + value, aux heads never run."""

    trainer = make_trainer(
        suit_coef=0.0, trick_coef=0.0, bid_hit_coef=0.0
    )
    trees, _ = trainer.collect()
    before = [p.detach().clone() for p in trainer.model.parameters()]
    stats = trainer.update(trees)
    assert stats.loss_suit == 0.0
    assert stats.loss_trick == 0.0
    assert stats.loss_bid_hit == 0.0
    assert np.isfinite(stats.loss_value)
    assert np.isfinite(stats.loss_policy)
    if not stats.rolled_back:
        assert any(
            not torch.equal(a, b)
            for a, b in zip(before, trainer.model.parameters())
        )


def test_belief_gradients_reach_only_the_heads_the_loss_weights():
    """The default objective: suit presence + bid hit, one forward, one backward.

    Trick count is weighted at zero, so its head must not even run.
    """

    trainer = make_trainer()
    trees, _ = trainer.collect()
    groups = build_training_groups(trees, MODEL_CONFIG, trainer.train)
    group = groups[0]

    # Empty policy rows: this isolates the auxiliary losses, so any gradient
    # below came from a belief head and not from NeuRD reaching the trunk.
    loss, terms = trainer._chunk_loss(group, 0, group.tokens.shape[0], {})
    trainer.model.zero_grad(set_to_none=True)
    loss.backward()

    model = trainer.model
    assert terms["suit"] > 0.0
    assert terms["bid_hit"] > 0.0
    assert "trick" not in terms
    assert model.suit_presence_head.weight.grad.abs().sum() > 0.0
    assert all(
        p.grad is not None and p.grad.abs().sum() > 0.0
        for p in model.bid_hit_head.parameters()
    )
    assert model.trick_count_head.weight.grad is None


def test_belief_labels_cover_every_seat_and_stop_at_the_table_size():
    """Both beliefs label the observer's own seat; padding seats stay masked."""

    trainer = make_trainer()
    trees, _ = trainer.collect()
    groups = build_training_groups(trees, MODEL_CONFIG, trainer.train)
    for group in groups:
        players = group.num_players
        owned = group.owned

        assert (group.bid_hit_targets[:, :players] != IGNORE_LABEL).all()
        assert (group.bid_hit_targets[:, players:] == IGNORE_LABEL).all()
        assert np.isin(group.bid_hit_targets[:, :players], (0, 1)).all()

        # Suit presence is per position, so check it only where a leaf owns the
        # position -- unowned rows are never read by the loss.
        real = group.suit_targets[:, :, :players, :][owned]
        padding = group.suit_targets[:, :, players:, :][owned]
        assert (real != IGNORE_LABEL).all()
        assert (padding == IGNORE_LABEL).all()


def test_entropy_bonus_and_advantage_clip_run():
    trainer = make_trainer(entropy_coef=0.01, neurd_advantage_clip=1.0)
    trees, _ = trainer.collect()
    stats = trainer.update(trees)
    assert np.isfinite(stats.loss_policy) or stats.policy_rows == 0
    assert np.isfinite(stats.entropy)


def test_lr_warmup_ramps_and_survives_the_kl_cap():
    """A cold start at full LR is an Adam sign step that trips the cap;
    the warmup ramp is what lets the first updates survive."""

    trainer = make_trainer(lr_warmup_updates=10)
    base = trainer.train.learning_rate
    trees, _ = trainer.collect()
    stats = trainer.update(trees)
    assert not stats.rolled_back
    assert trainer.optimizer_steps == trainer.train.epochs
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(
        base * trainer.train.epochs / 10
    )
    trees, _ = trainer.collect()
    trainer.update(trees)
    assert trainer.optimizer_steps == 2 * trainer.train.epochs


def test_optimizer_steps_survive_a_checkpoint_roundtrip(tmp_path):
    trainer = make_trainer(lr_warmup_updates=10)
    trees, _ = trainer.collect()
    trainer.update(trees)
    steps = trainer.optimizer_steps
    assert steps > 0
    path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(path)
    fresh = make_trainer(lr_warmup_updates=10)
    fresh.load_checkpoint(path)
    assert fresh.optimizer_steps == steps


def test_inclusion_exponent_changes_the_gradient_it_does_not_break_it():
    """1/q is the difference between having NeuRD and thinking you have it."""

    trees = None
    grads = {}
    for exponent in (0.0, 1.0):
        trainer = make_trainer(neurd_inclusion_exponent=exponent)
        if trees is None:
            trees = trainer.collect()[0]
        stats = trainer.update(trees)
        assert np.isfinite(stats.loss_policy)
        grads[exponent] = stats.loss_policy
    assert grads[0.0] != grads[1.0]


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

    def policy_rows(exponent):
        trainer = make_trainer(
            schedule_cells=cells,
            branch_budget=budget,
            branch_depth_exponent=exponent,
        )
        trees, _ = trainer.collect()
        groups = build_training_groups(trees, MODEL_CONFIG, trainer.train)
        weights, shallow, deep = [], 0.0, 0.0
        for group in groups:
            for rows in group.policy.values():
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

    flat_total, flat_shallow, _ = policy_rows(0.0)
    early_total, early_shallow, _ = policy_rows(-1.0)
    # The tree's total branch weight is preserved exactly.
    assert early_total == pytest.approx(flat_total, rel=1e-4)
    # But early positions now carry more of it.
    assert early_shallow > flat_shallow


def test_unbranched_rows_produce_a_nonzero_policy_gradient():
    """The k=1 trap: centering A over the candidate set, or dividing the row
    loss by the candidate count, silently zeroes every unbranched decision --
    half the rows in a typical iteration, gone with no error."""

    import torch as _torch

    trainer = make_trainer(branch_budget=BranchBudgetConfig(branch_rate=0.0))
    trees, _ = trainer.collect()
    groups = build_training_groups(trees, MODEL_CONFIG, trainer.train)
    # The focal's own bid is expanded whatever the rate -- it is the root of
    # the tree -- so only the play rows are unbranched here.
    play = [group.policy["play"] for group in groups if group.policy["play"].weight]
    assert play, "expected play decisions with branching switched off"
    assert all(
        len(candidates) == 1 for r in play for candidates in r.candidates
    )
    assert all(not any(r.branched) for r in play)

    trainer.model.zero_grad(set_to_none=True)
    total = 0.0
    for group in groups:
        by_chunk = _rows_by_chunk(group, trainer._microbatches(group))
        for (start, stop), policy_rows in zip(
            trainer._microbatches(group), by_chunk
        ):
            loss, terms = trainer._chunk_loss(group, start, stop, policy_rows)
            if loss is not None:
                loss.backward()
            total += terms.get("policy", 0.0)

    head_grad = trainer.model.card_head.weight.grad
    assert head_grad is not None
    assert _torch.linalg.vector_norm(head_grad).item() > 0.0
