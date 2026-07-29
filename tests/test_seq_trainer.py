"""Trainer: loss normalization, KL-cap rollback, checkpoints, smoke update."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from plump.rewards import compute_relative_rewards
from plump.seq.config import (
    BranchBudgetConfig,
    BranchRuleConfig,
    GameScheduleCell,
    RolloutOptions,
    SeqModelConfig,
    SeqTrainingConfig,
)
from plump.seq.model import SeqPlumpModel
from plump.seq.policy import SeqLeague, SeqModelPolicy
from plump.seq.tokens import IGNORE_LABEL
from plump.seq.trainer import (
    SeqTrainer,
    _rows_by_chunk,
    build_training_groups,
    control_variate_action_advantages,
    sampled_mirror_target,
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

    position_weight_total = 0.0
    policy_weight_total = 0.0
    for group in groups:
        position_weight_total += float(group.position_weight.sum())
        for rows in group.policy.values():
            policy_weight_total += float(np.sum(rows.weight))
    # Each loss family's weights sum to exactly 1, whatever the exponent and
    # however many trees happen to have rows of that kind, so the effective
    # learning rate cannot drift with how the deals happened to come out.
    assert position_weight_total == pytest.approx(1.0)
    assert policy_weight_total == pytest.approx(1.0)


def test_rollout_summary_separates_focal_and_non_focal_outcomes():
    trainer = make_trainer()
    trees, summary = trainer.collect()
    focal_hits = []
    non_focal_hits = []
    focal_rewards = []
    non_focal_rewards = []
    for tree in trees:
        spine = next(leaf for leaf in tree.leaves if leaf.on_policy_spine)
        round_state = spine.env.state.current_round
        relative = compute_relative_rewards(round_state.round_scores)
        focal_rewards.append(relative[tree.focal])
        non_focal_rewards.extend(
            reward for player, reward in relative.items() if player != tree.focal
        )
        for bid in round_state.bids:
            destination = focal_hits if bid.player == tree.focal else non_focal_hits
            destination.append(round_state.tricks_won[bid.player] == bid.value)

    assert summary.bid_hit_focal == pytest.approx(np.mean(focal_hits))
    assert summary.bid_hit_non_focal == pytest.approx(np.mean(non_focal_hits))
    assert summary.bid_hit_rate == pytest.approx(
        np.mean([*focal_hits, *non_focal_hits])
    )
    assert summary.reward_focal == pytest.approx(np.mean(focal_rewards))
    assert summary.reward_non_focal == pytest.approx(np.mean(non_focal_rewards))


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
            not torch.equal(a, b) for a, b in zip(before, trainer.model.parameters())
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
    """NeuRD plus value only: disabled auxiliary heads never run."""

    trainer = make_trainer(suit_coef=0.0, trick_coef=0.0, bid_hit_coef=0.0)
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
            not torch.equal(a, b) for a, b in zip(before, trainer.model.parameters())
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


def test_belief_labels_cover_the_right_seats_and_stop_at_the_table_size():
    """Outcome beliefs label the observer too; suit presence labels opponents only.

    The two seat axes differ on purpose, so this pins both: bid hit and trick
    count run over all ``num_players`` seats including relative seat 0, while
    suit presence starts at relative seat 1 -- the observer's own suits are in
    its prefix verbatim, so a column for them would supervise an identity.
    """

    trainer = make_trainer()
    trees, _ = trainer.collect()
    groups = build_training_groups(trees, MODEL_CONFIG, trainer.train)
    for group in groups:
        players = group.num_players
        owned = group.owned

        for outcome in (group.bid_hit_targets, group.trick_targets):
            assert (outcome[:, :players] != IGNORE_LABEL).all()
            assert (outcome[:, players:] == IGNORE_LABEL).all()
        assert np.isin(group.bid_hit_targets[:, :players], (0, 1)).all()

        # Suit presence is per position, so check it only where a leaf owns the
        # position -- unowned rows are never read by the loss.
        opponents = players - 1
        real = group.suit_targets[:, :, :opponents, :][owned]
        padding = group.suit_targets[:, :, opponents:, :][owned]
        assert (real != IGNORE_LABEL).all()
        assert (padding == IGNORE_LABEL).all()


def test_entropy_bonus_and_advantage_clip_run():
    trainer = make_trainer(entropy_coef=0.01, neurd_advantage_clip=1.0)
    trees, _ = trainer.collect()
    stats = trainer.update(trees)
    assert np.isfinite(stats.loss_policy) or stats.policy_rows == 0
    assert np.isfinite(stats.entropy)


def test_sampled_mirror_target_matches_closed_form_exponentiated_update():
    old = torch.tensor([[0.25, 0.75, 0.0]])
    advantages = torch.tensor([[2.0, -1.0, 99.0]])
    legal = old > 0
    observed = sampled_mirror_target(
        old,
        advantages,
        legal,
        step_size=0.5,
        uniform_mix=0.0,
        target_kl=0.0,
    )
    expected = torch.softmax(
        torch.tensor(
            [[np.log(0.25) + 1.0, np.log(0.75) - 0.5]],
            dtype=old.dtype,
        ),
        dim=-1,
    )
    torch.testing.assert_close(observed[:, :2], expected)
    assert observed[0, 2] == 0


def test_sampled_mirror_target_respects_kl_bound_and_legal_support():
    old = torch.tensor([[0.97, 0.02, 0.01, 0.0]])
    advantages = torch.tensor([[0.0, 4.0, 10.0, 100.0]])
    legal = old > 0
    cap = 0.003
    target = sampled_mirror_target(
        old,
        advantages,
        legal,
        step_size=2.0,
        uniform_mix=0.02,
        target_kl=cap,
        bisection_steps=24,
    )
    kl = (
        old
        * (torch.log(old.clamp_min(1e-12)) - torch.log(target.clamp_min(1e-12)))
        * legal.float()
    ).sum()
    assert kl <= cap + 1e-6
    assert target.sum() == pytest.approx(1.0)
    assert target[0, 2] > old[0, 2]
    assert target[0, 3] == 0


def test_sampled_neurd_advantage_has_the_full_information_expectation():
    """A tabular bandit pins the expected update direction action by action."""

    torch.manual_seed(91)
    draws = 500_000
    old = torch.tensor([[0.5, 0.3, 0.15, 0.05]], dtype=torch.float32)
    q = torch.tensor([-2.0, 0.5, 3.0, 8.0], dtype=torch.float32)
    candidates = torch.multinomial(old[0], draws, replacement=True).unsqueeze(1)
    old_batch = old.expand(draws, -1)
    observed = control_variate_action_advantages(
        old_batch,
        candidates,
        q[candidates],
        old[0, candidates],
        torch.ones_like(candidates, dtype=torch.bool),
        torch.full((draws, 1), 0.7),
    )
    expected = q - (old[0] * q).sum()
    torch.testing.assert_close(observed.mean(dim=0), expected, atol=0.08, rtol=0.0)
    torch.testing.assert_close(
        (old_batch * observed).sum(dim=1),
        torch.zeros(draws),
        atol=2e-6,
        rtol=0.0,
    )


def test_sampled_mirror_update_is_finite_and_changes_weights():
    trainer = make_trainer(
        policy_objective="sampled_mirror",
        sampled_mirror_uniform_mix=0.02,
        sampled_mirror_target_kl=0.003,
    )
    trees, _ = trainer.collect()
    before = [parameter.detach().clone() for parameter in trainer.model.parameters()]
    stats = trainer.update(trees)
    assert np.isfinite(stats.loss_policy)
    assert np.isfinite(stats.policy_kl)
    assert not stats.rolled_back
    assert any(
        not torch.equal(left, right)
        for left, right in zip(before, trainer.model.parameters())
    )


def test_unknown_policy_objective_is_rejected():
    with pytest.raises(ValueError, match="policy_objective"):
        make_trainer(policy_objective="ppo")


def test_kl_backtracking_accepts_a_smaller_adam_step():
    trainer = make_trainer(
        policy_objective="sampled_mirror",
        learning_rate=0.1,
        lr_warmup_updates=0,
        policy_kl_cap=1e-4,
        policy_kl_p99_cap=4e-4,
        sampled_mirror_target_kl=5e-5,
        kl_backtrack_attempts=16,
        kl_backtrack_factor=0.5,
        suit_coef=0.0,
        trick_coef=0.0,
        bid_hit_coef=0.0,
    )
    trees, _ = trainer.collect()
    stats = trainer.update(trees)
    assert not stats.rolled_back
    assert stats.backtracks > 0
    assert 0.0 < stats.step_scale < 1.0
    assert stats.policy_kl <= trainer.train.policy_kl_cap
    assert stats.policy_kl_p99 <= trainer.train.policy_kl_p99_cap
    assert trainer.optimizer_steps == 1


def test_proven_early_p99_rejection_matches_full_backtracking(monkeypatch):
    """Skipping the remainder of a provably failed guard pass is transparent."""

    options = dict(
        policy_objective="sampled_mirror",
        learning_rate=0.1,
        lr_warmup_updates=0,
        policy_kl_cap=1e-4,
        policy_kl_p99_cap=4e-4,
        sampled_mirror_target_kl=5e-5,
        kl_backtrack_attempts=16,
        kl_backtrack_factor=0.5,
        suit_coef=0.0,
        trick_coef=0.0,
        bid_hit_coef=0.0,
    )
    early = make_trainer(**options)
    complete = make_trainer(**options)
    trees, _ = early.collect()

    early_calls = 0
    complete_calls = 0
    early_forward = early.model.forward_full
    complete_forward = complete.model.forward_full

    def count_early(*args, **kwargs):
        nonlocal early_calls
        early_calls += 1
        return early_forward(*args, **kwargs)

    def count_complete(*args, **kwargs):
        nonlocal complete_calls
        complete_calls += 1
        return complete_forward(*args, **kwargs)

    monkeypatch.setattr(early.model, "forward_full", count_early)
    monkeypatch.setattr(complete.model, "forward_full", count_complete)
    full_guard = complete._evaluate_policy_kl

    def disable_early_return(groups, **_kwargs):
        return full_guard(groups)

    monkeypatch.setattr(complete, "_evaluate_policy_kl", disable_early_return)

    early_stats = early.update(trees)
    complete_stats = complete.update(trees)
    early_payload = dataclasses.asdict(early_stats)
    complete_payload = dataclasses.asdict(complete_stats)
    for key in ("update_sec", "build_sec"):
        early_payload.pop(key)
        complete_payload.pop(key)
    assert early_payload == complete_payload
    for left, right in zip(early.model.parameters(), complete.model.parameters()):
        assert torch.equal(left, right)
    assert early_calls < complete_calls


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


def test_league_uses_only_the_newest_half_of_training_history():
    league = SeqLeague(max_snapshots=20)
    for iteration in (10, 25, 49, 50, 75, 100, 101):
        league.add(f"iter_{iteration}", f"/tmp/{iteration}.pt", iteration)
    assert [snapshot.iteration for snapshot in league.eligible_snapshots(100)] == [
        50,
        75,
        100,
    ]
    assert [snapshot.iteration for snapshot in league.eligible_snapshots(101)] == [
        75,
        100,
        101,
    ]


def test_league_opponents_join_collection(tmp_path):
    trainer = make_trainer(
        tmp_path,
        rollout=RolloutOptions(historical_arm="concurrent"),
    )
    trees, _ = trainer.collect()
    trainer.update(trees)
    trainer.iteration = 1
    trainer.maybe_snapshot(tmp_path)
    trees, summary = trainer.collect()
    arms = {tree.arm for tree in trees}
    assert arms == {"self", "historical"}
    self_tree = next(tree for tree in trees if tree.arm == "self")
    historical_tree = next(tree for tree in trees if tree.arm == "historical")
    assert self_tree.initial_hands != historical_tree.initial_hands
    assert summary.reward_historical != 0.0 or summary.reward_self != 0.0


@pytest.mark.parametrize(
    "exponent, expect_ratio", [(0.0, 1.0), (0.5, None), (1.0, None)]
)
def test_tree_weight_exponent_shifts_weight_toward_larger_trees(exponent, expect_ratio):
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
    rows = {
        (tree.num_players, tree.hand_size): sum(
            sum(leaf.position_reach_weights().values()) for leaf in tree.leaves
        )
        for tree in trees
    }
    for group in groups:
        for row in range(group.position_weight.shape[0]):
            key = (group.num_players, group.hand_size)
            totals[key] = totals.get(key, 0.0) + float(group.position_weight[row].sum())
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
        key = (group.num_players, group.hand_size)
        totals[key] = totals.get(key, 0.0) + float(group.position_weight.sum())
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
    assert all(len(candidates) == 1 for r in play for candidates in r.candidates)
    assert all(not any(r.branched) for r in play)

    trainer.model.zero_grad(set_to_none=True)
    total = 0.0
    for group in groups:
        by_chunk = _rows_by_chunk(group, trainer._microbatches(group))
        for (start, stop), policy_rows in zip(trainer._microbatches(group), by_chunk):
            loss, terms = trainer._chunk_loss(group, start, stop, policy_rows)
            if loss is not None:
                loss.backward()
            total += terms.get("policy", 0.0)

    head_grad = trainer.model.card_head.weight.grad
    assert head_grad is not None
    assert _torch.linalg.vector_norm(head_grad).item() > 0.0
