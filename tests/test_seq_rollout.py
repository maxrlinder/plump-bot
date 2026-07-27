"""Rollout engine invariants: budgets, ownership, backups, KV-path parity."""

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np
import pytest
import torch

from plump.seq.config import (
    NEXT_BID,
    BranchBudgetConfig,
    BranchRuleConfig,
    GameScheduleCell,
    RolloutOptions,
    SeqModelConfig,
    SeqTrainingConfig,
    ShapeBranchRate,
    build_branch_rate_table,
    build_game_schedule,
    seq_len,
)
from plump.seq.model import SeqPlumpModel
from plump.seq.policy import masked_probabilities
from plump.seq.rollout import SeqRolloutCollector
from plump.seq.tokens import build_seat_tokens
from plump.training.common import compute_relative_rewards

MODEL_CONFIG = SeqModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128)


def make_collector(
    *,
    cells,
    branch_rate: float = 1.0,
    seed=0,
    bid_top_k=4,
    play_mode="all_legal",
    play_top_k=4,
):
    torch.manual_seed(seed)
    model = SeqPlumpModel(MODEL_CONFIG).eval()
    train = SeqTrainingConfig(
        schedule_cells=tuple(cells),
        branch_rule=BranchRuleConfig(
            bid_top_k=bid_top_k, play_mode=play_mode, play_top_k=play_top_k
        ),
        branch_budget=BranchBudgetConfig(branch_rate=branch_rate),
    )
    return SeqRolloutCollector(model, train, device="cpu")


def collect_trees(collector, seed=0):
    return collector.collect(None, random.Random(seed))


def test_unbranched_single_leaf_tree():
    cells = [GameScheduleCell(hand_size=4, num_players=3)]
    collector = make_collector(cells=cells, branch_rate=0.0, bid_top_k=1)
    trees = collect_trees(collector)
    assert len(trees) == 1
    tree = trees[0]
    assert tree.leaf_total == 1
    assert len(tree.leaves) == 1
    leaf = tree.leaves[0]
    # 1 bid + (N-1) plays; the forced final trick is played without decisions.
    assert len(leaf.decisions) == 4
    assert all(record.branch is None for record in leaf.decisions)
    scores = leaf.env.state.current_round.round_scores
    assert leaf.terminal_value == compute_relative_rewards(scores)[tree.focal]
    targets = leaf.value_targets()
    total_len = seq_len(tree.num_players, tree.hand_size)
    assert sorted(targets) == list(range(total_len))
    assert all(value == leaf.terminal_value for value in targets.values())


def test_branching_partitions_positions_across_leaves():
    cells = [GameScheduleCell(hand_size=3, num_players=3)]
    collector = make_collector(cells=cells)
    trees = collect_trees(collector, seed=1)
    tree = trees[0]
    assert tree.leaf_total > 1
    assert len(tree.leaves) == tree.leaf_total
    total_len = seq_len(tree.num_players, tree.hand_size)

    spine_count = sum(leaf.on_policy_spine for leaf in tree.leaves)
    assert spine_count == 1

    for leaf in tree.leaves:
        targets = leaf.value_targets()
        assert sorted(targets) == list(range(leaf.owned_from, total_len))
        for record in leaf.decisions:
            assert record.position >= leaf.owned_from
            if record.branch is not None:
                assert record.branch.backed_value is not None
                assert targets[record.position] == record.branch.backed_value

    # Every position of every leaf path is owned exactly once across the tree:
    # leaves partition path positions as prefix-sharing suffixes.
    owned_total = sum(total_len - leaf.owned_from for leaf in tree.leaves)
    branch_children = sum(
        len(record.branch.candidate_indices) - 1
        for leaf in tree.leaves
        for record in leaf.decisions
        if record.branch is not None
    )
    assert owned_total == total_len + sum(
        total_len - leaf.owned_from
        for leaf in tree.leaves
        if not leaf.on_policy_spine
    )
    assert branch_children == tree.leaf_total - 1


def test_backed_values_match_backup_formulas():
    cells = [GameScheduleCell(hand_size=4, num_players=3)]
    collector = make_collector(cells=cells, bid_top_k=3)
    trees = collect_trees(collector, seed=2)
    checked_exact = 0
    checked_capped = 0
    for tree in trees:
        for leaf in tree.leaves:
            for record in leaf.decisions:
                branch = record.branch
                if branch is None:
                    continue
                assert set(branch.child_values) == set(branch.candidate_indices)
                if math.isclose(branch.candidate_mass, 1.0, abs_tol=1e-6):
                    expected = sum(
                        p * branch.child_values[i]
                        for p, i in zip(
                            branch.prior_probs, branch.candidate_indices
                        )
                    )
                    checked_exact += 1
                else:
                    deterministic = set(
                        branch.candidate_indices[: branch.deterministic_count]
                    )
                    expected = sum(
                        branch.raw_probs[i] * branch.child_values[i]
                        for i in deterministic
                    )
                    if branch.sampled_index not in deterministic:
                        expected += branch.child_values[branch.sampled_index]
                    checked_capped += 1
                assert branch.backed_value == pytest.approx(expected)
    assert checked_exact > 0


def test_rollout_probs_match_full_forward_replay():
    """The KV-cached rollout path must equal a from-scratch causal forward."""

    cells = [GameScheduleCell(hand_size=4, num_players=3)]
    collector = make_collector(cells=cells)
    trees = collect_trees(collector, seed=3)
    model = collector.model
    checked_branch_leaf = False
    for tree in trees:
        for leaf in tree.leaves:
            round_state = leaf.env.state.current_round
            tokens = build_seat_tokens(
                MODEL_CONFIG,
                leaf.env.state.event_log,
                tree.focal,
                tree.num_players,
                tree.hand_size,
                tree.initial_hands[tree.focal],
                tree.bidding_start_player,
            )
            with torch.inference_mode():
                output = model.forward_full(
                    torch.from_numpy(tokens[None]), aux_heads=False
                )
            for record in leaf.decisions:
                position = record.position
                mask = record.old_probs > 0
                if record.phase == NEXT_BID:
                    logits = output.bid_logits[0, position].numpy()
                else:
                    logits = output.card_logits[0, position].numpy()
                expected = masked_probabilities(logits, mask)
                np.testing.assert_allclose(
                    record.old_probs, expected, atol=1e-4, rtol=0
                )
            if not leaf.on_policy_spine:
                checked_branch_leaf = True
            assert round_state.round_scores
    assert checked_branch_leaf


def test_sample_k_uses_monte_carlo_weights():
    """Sampled candidates weight by empirical frequency, summing to one."""

    cells = [GameScheduleCell(hand_size=6, num_players=3)]
    collector = make_collector(
        cells=cells, play_mode="sample_k", play_top_k=3
    )
    trees = collect_trees(collector, seed=21)
    checked = 0
    for tree in trees:
        for leaf in tree.leaves:
            for record in leaf.decisions:
                branch = record.branch
                if branch is None or record.phase == NEXT_BID:
                    continue
                # Weights are multiples of 1/k and form a distribution.
                assert branch.candidate_mass == pytest.approx(1.0)
                assert sum(branch.prior_probs) == pytest.approx(1.0)
                for weight in branch.prior_probs:
                    assert weight * 3 == pytest.approx(round(weight * 3))
                # Candidates must be actions the policy could actually draw.
                for index in branch.candidate_indices:
                    assert record.old_probs[index] > 0
                # The realized action is always among the draws.
                assert branch.sampled_index in branch.candidate_indices
                expected = sum(
                    w * branch.child_values[i]
                    for w, i in zip(branch.prior_probs, branch.candidate_indices)
                )
                assert branch.backed_value == pytest.approx(expected)
                checked += 1
    assert checked > 0


def test_uniform_arm_is_explored_but_carries_no_backup_weight():
    """The exploration arm gets a Q value without moving the parent's value."""

    cells = [GameScheduleCell(hand_size=7, num_players=4)]
    collector = make_collector(
        cells=cells,
        play_mode="sample_k_plus_uniform",
        play_top_k=3,
    )
    trees = collect_trees(collector, seed=11)
    zero_weight_arms = 0
    for tree in trees:
        for leaf in tree.leaves:
            for record in leaf.decisions:
                branch = record.branch
                if branch is None or record.phase == NEXT_BID:
                    continue
                weights = dict(zip(branch.candidate_indices, branch.prior_probs))
                # Still a distribution: the k policy draws carry all the mass.
                assert sum(weights.values()) == pytest.approx(1.0)
                assert branch.candidate_mass == pytest.approx(1.0)
                for index, weight in weights.items():
                    assert weight * 3 == pytest.approx(round(weight * 3))
                    # A zero-weight arm is still fully evaluated: it has a
                    # child Q value, which is the whole point of exploring it.
                    assert index in branch.child_values
                    if weight == 0.0:
                        zero_weight_arms += 1
                assert branch.sampled_index in branch.candidate_indices
                assert weights[branch.sampled_index] > 0.0
                # The backup ignores the explored arm entirely.
                expected = sum(
                    weight * branch.child_values[index]
                    for index, weight in weights.items()
                )
                assert branch.backed_value == pytest.approx(expected)
    assert zero_weight_arms > 0


def test_uniform_arm_matches_sample_k_backups_on_shared_actions():
    """Adding the arm must not perturb the on-policy draws or their values."""

    cells = [GameScheduleCell(hand_size=5, num_players=3)]
    plain = make_collector(
        cells=cells, play_mode="sample_k", play_top_k=3
    )
    explored = make_collector(
        cells=cells,
        play_mode="sample_k_plus_uniform",
        play_top_k=3,
    )
    explored.model.load_state_dict(plain.model.state_dict())

    def first_play_branch(collector):
        # The spine leaf of the first tree: everything up to its first play
        # branch is identical between the two runs, because the uniform draw
        # only happens at a play branch and so cannot perturb the tape before
        # the first one.
        tree = collect_trees(collector, seed=3)[0]
        spine = [leaf for leaf in tree.leaves if leaf.on_policy_spine]
        assert len(spine) == 1
        for record in sorted(spine[0].decisions, key=lambda r: r.position):
            if record.branch is not None and record.phase != NEXT_BID:
                return record.branch
        raise AssertionError("No play branch was created.")

    a, b = first_play_branch(plain), first_play_branch(explored)
    # The policy draws are made before the uniform arm, off the same tape, so
    # the weighted part of the candidate set must be identical.
    weights_a = dict(zip(a.candidate_indices, a.prior_probs))
    weights_b = {
        index: weight
        for index, weight in zip(b.candidate_indices, b.prior_probs)
        if weight > 0.0
    }
    assert weights_a == pytest.approx(weights_b)


def test_sample_k_can_pick_outside_the_top_actions():
    """Sampling must be able to reach actions top-k would never evaluate."""

    cells = [GameScheduleCell(hand_size=8, num_players=4)]
    collector = make_collector(
        cells=cells, play_mode="sample_k", play_top_k=3
    )
    trees = collect_trees(collector, seed=5)
    reached_outside_top3 = False
    for tree in trees:
        for leaf in tree.leaves:
            for record in leaf.decisions:
                branch = record.branch
                if branch is None or record.phase == NEXT_BID:
                    continue
                order = np.argsort(-record.old_probs)
                top3 = set(order[:3].tolist())
                if set(branch.candidate_indices) - top3:
                    reached_outside_top3 = True
    assert reached_outside_top3


def test_cache_free_mode_matches_cached_rollout():
    """The no-cache path must reproduce the cached rollout exactly."""

    cells = [GameScheduleCell(hand_size=4, num_players=3)]
    cached = make_collector(cells=cells)
    uncached = make_collector(cells=cells)
    uncached.model.load_state_dict(cached.model.state_dict())
    uncached.use_cache = False

    cached_trees = collect_trees(cached, seed=6)
    uncached_trees = collect_trees(uncached, seed=6)

    assert len(cached_trees) == len(uncached_trees)
    for left, right in zip(cached_trees, uncached_trees):
        assert left.leaf_total == right.leaf_total
        assert left.decision_total == right.decision_total
        assert len(left.leaves) == len(right.leaves)
        for a, b in zip(left.leaves, right.leaves):
            assert a.terminal_value == pytest.approx(b.terminal_value)
            assert a.owned_from == b.owned_from
            assert len(a.decisions) == len(b.decisions)
            for ra, rb in zip(a.decisions, b.decisions):
                assert ra.position == rb.position
                assert ra.action_index == rb.action_index
                np.testing.assert_allclose(
                    ra.old_probs, rb.old_probs, atol=1e-4, rtol=0
                )


def test_stats_and_forward_accounting():
    cells = [GameScheduleCell(hand_size=3, num_players=3)]
    collector = make_collector(cells=cells)
    trees = collect_trees(collector, seed=4)
    stats = collector.stats
    assert stats.games == 1
    assert stats.trees == len(trees) == 1
    assert stats.leaves == trees[0].leaf_total
    assert stats.decisions == trees[0].decision_total
    assert stats.forward_rows > 0
    assert stats.collect_sec > 0


# --------------------------------------------------------------------- #
# Rollout options: deal batching, historical arm modes, bid splitting     #
# --------------------------------------------------------------------- #


def options_collector(
    *,
    cells,
    options,
    branch_rate: float = 1.0,
    play_mode="all_legal",
    seed=0,
):
    torch.manual_seed(seed)
    model = SeqPlumpModel(MODEL_CONFIG).eval()
    train = SeqTrainingConfig(
        schedule_cells=tuple(cells),
        branch_rule=BranchRuleConfig(bid_top_k=4, play_mode=play_mode),
        branch_budget=BranchBudgetConfig(branch_rate=branch_rate),
        rollout=options,
    )
    return SeqRolloutCollector(model, train, device="cpu")


def test_deal_batching_matches_sequential_collection():
    """Batching deals into one wave loop must not change the trees."""

    cells = [GameScheduleCell(hand_size=4, num_players=3, games=3)]
    sequential = options_collector(
        cells=cells, options=RolloutOptions(deals_per_batch=1)
    )
    batched = options_collector(
        cells=cells, options=RolloutOptions(deals_per_batch=3)
    )
    batched.model.load_state_dict(sequential.model.state_dict())

    left = collect_trees(sequential, seed=4)
    right = collect_trees(batched, seed=4)
    assert len(left) == len(right) == 3
    for a, b in zip(left, right):
        assert a.hand_size == b.hand_size
        assert a.focal == b.focal
        assert a.initial_hands == b.initial_hands
        assert a.leaf_total == b.leaf_total
        assert a.decision_total == b.decision_total
        # Batching only widens the forwards: every training row is the same
        # one the deal would have produced on its own (values to tolerance,
        # since batched matmul is not bitwise associative across batch sizes).
        sequential_rows = training_rows(a)
        batched_rows = training_rows(b)
        assert set(sequential_rows) == set(batched_rows)
        for key, expected in sequential_rows.items():
            for want, got in zip(expected, batched_rows[key]):
                if want is None:
                    assert got is None
                else:
                    assert got == pytest.approx(want, abs=1e-6)


def test_historical_arm_off_produces_only_self_trees():
    cells = [GameScheduleCell(hand_size=4, num_players=3, games=2)]
    collector = options_collector(
        cells=cells, options=RolloutOptions(historical_arm="off")
    )
    trees = collect_trees(collector, seed=2)
    assert {tree.arm for tree in trees} == {"self"}
    assert len(trees) == 2


def test_bid_split_matches_unsplit_tree():
    """Splitting the root bid must reproduce the unsplit tree exactly."""

    cells = [GameScheduleCell(hand_size=5, num_players=3)]
    whole = options_collector(
        cells=cells, options=RolloutOptions(bid_split_groups=1)
    )
    split = options_collector(
        cells=cells, options=RolloutOptions(bid_split_groups=2)
    )
    split.model.load_state_dict(whole.model.state_dict())

    a = collect_trees(whole, seed=8)[0]
    b = collect_trees(split, seed=8)[0]

    assert a.leaf_total == b.leaf_total
    assert len(a.leaves) == len(b.leaves)
    # Exactly one on-policy spine survives the split.
    assert sum(l.on_policy_spine for l in a.leaves) == 1
    assert sum(l.on_policy_spine for l in b.leaves) == 1
    # The same terminal values are reached by the same set of leaves.
    assert sorted(l.terminal_value for l in a.leaves) == pytest.approx(
        sorted(l.terminal_value for l in b.leaves)
    )
    # The root bid backup is identical, so training targets are unchanged.
    def root_target(tree):
        owners = [leaf for leaf in tree.leaves if leaf.owned_from == 0]
        assert len(owners) == 1
        return owners[0].value_targets()[0]

    assert root_target(b) == pytest.approx(root_target(a))


def test_bid_split_owns_each_position_exactly_once():
    """No training position may be claimed by two split passes."""

    cells = [GameScheduleCell(hand_size=5, num_players=3)]
    collector = options_collector(
        cells=cells, options=RolloutOptions(bid_split_groups=2)
    )
    tree = collect_trees(collector, seed=8)[0]
    total_len = seq_len(tree.num_players, tree.hand_size)
    for leaf in tree.leaves:
        targets = leaf.value_targets()
        assert sorted(targets) == list(range(leaf.owned_from, total_len))
    # The shared prefix is owned by exactly one leaf.
    owners = [leaf for leaf in tree.leaves if leaf.owned_from == 0]
    assert len(owners) == 1


def training_rows(tree) -> dict:
    """Every training row a tree emits, keyed by its exact causal context.

    A position's training signal depends only on the tokens up to and
    including it, so ``tokens[:p + 1]`` identifies the row. Two trees with the
    same key set emit exactly the same training data regardless of how leaves
    were split or ordered, or which leaf owns the shared prefix.
    """

    rows: dict = {}
    for leaf in tree.leaves:
        tokens = build_seat_tokens(
            MODEL_CONFIG,
            leaf.env.state.event_log,
            tree.focal,
            tree.num_players,
            tree.hand_size,
            tree.initial_hands[tree.focal],
            tree.bidding_start_player,
        )
        context = [tokens[: position + 1].tobytes() for position in range(len(tokens))]
        decisions = {record.position: record for record in leaf.decisions}
        for position, target in leaf.value_targets().items():
            record = decisions.get(position)
            branch = None if record is None else record.branch
            key = (
                position,
                context[position],
                None if record is None else (record.phase, record.action_index),
                None if branch is None else branch.candidate_indices,
            )
            payload = (
                float(target),
                None if record is None else record.old_probs.copy(),
                None if branch is None else np.asarray(branch.prior_probs),
                None if branch is None else float(branch.backed_value),
            )
            if key in rows:
                raise AssertionError(f"Position {position} emitted twice: {key[:1]}")
            rows[key] = payload
    return rows


@pytest.mark.parametrize("splits", [2, 3, 4])
def test_bid_split_emits_identical_training_rows(splits):
    """Splitting must change only *when* rows are produced, never *what*.

    Row identity is exact; the numbers on them are compared to tolerance
    because a split pass runs smaller batches through the model, and batched
    matmul is not bitwise associative across batch sizes.
    """

    cells = [GameScheduleCell(hand_size=6, num_players=4)]
    whole = options_collector(
        cells=cells,
        options=RolloutOptions(bid_split_groups=1),
    )
    split = options_collector(
        cells=cells,
        options=RolloutOptions(bid_split_groups=splits),
    )
    split.model.load_state_dict(whole.model.state_dict())

    unsplit_tree = collect_trees(whole, seed=8)[0]
    split_tree = collect_trees(split, seed=8)[0]

    # The bid expansion must be wide enough for the split to be meaningful.
    assert len(split_tree.split_bid_candidates) >= 2
    assert unsplit_tree.leaf_total == split_tree.leaf_total > 1
    assert unsplit_tree.leaves_by_bid_group == split_tree.leaves_by_bid_group

    unsplit_rows = training_rows(unsplit_tree)
    split_rows = training_rows(split_tree)
    assert set(unsplit_rows) == set(split_rows)
    for key, expected in unsplit_rows.items():
        actual = split_rows[key]
        for left, right in zip(expected, actual):
            if left is None:
                assert right is None
            else:
                assert right == pytest.approx(left, abs=1e-6)


def test_bid_split_lowers_peak_cache_rows():
    """The point of splitting: fewer rows live at once for the same tree."""

    cells = [GameScheduleCell(hand_size=6, num_players=4)]
    whole = options_collector(
        cells=cells, options=RolloutOptions(bid_split_groups=1)
    )
    split = options_collector(
        cells=cells, options=RolloutOptions(bid_split_groups=4)
    )
    split.model.load_state_dict(whole.model.state_dict())

    unsplit_tree = collect_trees(whole, seed=8)[0]
    split_tree = collect_trees(split, seed=8)[0]

    assert unsplit_tree.leaf_total == split_tree.leaf_total
    assert split.stats.peak_cache_rows < whole.stats.peak_cache_rows


@pytest.mark.parametrize("splits", [1, 2])
def test_no_cache_row_is_read_before_it_is_written(splits, monkeypatch):
    """Every row the dense read path touches must have been written first.

    The wave loop reads whole row ranges rather than gathering live rows, so
    a row that is allocated but never prefilled or branch-copied would be
    attended to as zeros and quietly corrupt a leaf's policy. Filling the
    pool with NaN turns that into a hard failure.
    """

    import plump.seq.rollout as rollout_module

    original = rollout_module.KVCache

    def poisoned(*args, **kwargs):
        return original(*args, **kwargs, poison=True)

    monkeypatch.setattr(rollout_module, "KVCache", poisoned)

    cells = [GameScheduleCell(hand_size=6, num_players=4, games=2)]
    collector = options_collector(
        cells=cells,
        options=RolloutOptions(deals_per_batch=2, bid_split_groups=splits),
    )
    trees = collect_trees(collector, seed=8)
    assert sum(tree.leaf_total for tree in trees) > 2
    for tree in trees:
        for leaf in tree.leaves:
            assert not math.isnan(leaf.terminal_value)
            for record in leaf.decisions:
                assert not np.isnan(record.old_probs).any()


def test_bid_split_min_hand_size_gates_splitting():
    options = RolloutOptions(bid_split_groups=2, bid_split_min_hand_size=8)
    assert options.splits_for(5) == 1
    assert options.splits_for(9) == 2


def test_schedule_apportions_the_mix_exactly():
    """The realized (players, cards) mix must not be a random variable."""

    schedule = build_game_schedule(games_total=240, hand_size_tilt=1.0)
    assert sum(cell.games for cell in schedule) == 240
    # Every cell of the grid is present and named explicitly.
    shapes = {(cell.num_players, cell.hand_size) for cell in schedule}
    assert shapes == {(p, n) for p in (3, 4, 5) for n in range(3, 11)}

    by_hand = {}
    for cell in schedule:
        by_hand[cell.hand_size] = by_hand.get(cell.hand_size, 0) + cell.games
    # Linear tilt: a 10-card game gets 10/3 the deals of a 3-card game.
    assert by_hand[10] / by_hand[3] == pytest.approx(10 / 3, rel=0.05)
    # Player counts are balanced by default.
    by_players = {}
    for cell in schedule:
        by_players[cell.num_players] = by_players.get(cell.num_players, 0) + cell.games
    assert len(set(by_players.values())) == 1


def test_schedule_tilt_zero_is_uniform_over_hand_sizes():
    schedule = build_game_schedule(games_total=240, hand_size_tilt=0.0)
    by_hand = {}
    for cell in schedule:
        by_hand[cell.hand_size] = by_hand.get(cell.hand_size, 0) + cell.games
    assert len(set(by_hand.values())) == 1


def test_auto_batching_scales_deals_to_the_shape():
    """One global deals_per_batch cannot serve a 3-card and an 8-card shape."""

    options = RolloutOptions(
        auto_deals_per_batch=True,
        auto_target_rows=4096,
        auto_deals_headroom=0.0,
        max_deals_per_batch=64,
    )
    cells = [GameScheduleCell(hand_size=3, num_players=3, games=2)]
    collector = options_collector(cells=cells, options=options)
    small_cell = cells[0]
    # Nothing bounds a tree ahead of time, so an unmeasured shape probes with
    # a single deal rather than guessing.
    assert collector._deals_for(small_cell, 3, remaining=32) == 1
    collect_trees(collector, seed=1)

    small = collector._deals_for(small_cell, 3, remaining=1000)
    big_cell = GameScheduleCell(hand_size=8, num_players=5)
    collector._rows_per_deal[(5, 8)] = 3000.0
    big = collector._deals_for(big_cell, 5, remaining=1000)
    assert big == 1
    assert small > 8 * big
    assert small <= 64


def test_cycle_balances_bidding_position_not_absolute_seat():
    """Absolute seat is a relabeling; bidding position is what the model sees."""

    cells = [
        GameScheduleCell(hand_size=3, num_players=5, games=5),
        GameScheduleCell(hand_size=4, num_players=5, games=5),
    ]
    collector = options_collector(
        cells=cells,
        options=RolloutOptions(deals_per_batch=5, bid_position_mode="cycle"),
    )
    trees = collect_trees(collector, seed=2)
    positions = Counter(
        (tree.focal - tree.bidding_start_player) % tree.num_players
        for tree in trees
    )
    assert positions == {0: 2, 1: 2, 2: 2, 3: 2, 4: 2}
    # The cursor is per player count and must not restart on a new cell, or a
    # five-player game would never reach its late bidding positions.
    first_cell = trees[:5]
    assert len({
        (tree.focal - tree.bidding_start_player) % 5 for tree in first_cell
    }) == 5


def test_branch_points_reach_the_endgame():
    """Branching must fund the late tricks, not pile up at the opening.

    The leaf-budget floor this replaced branched every layer until the limit
    was crossed and then stopped, which produced *zero* branch decisions past
    the middle of a 9-card game -- the endgame, where counterfactuals are most
    decidable, was never funded.
    """

    torch.manual_seed(0)
    model = SeqPlumpModel(MODEL_CONFIG).eval()
    train = SeqTrainingConfig(
        schedule_cells=(GameScheduleCell(hand_size=9, num_players=4, games=4),),
        branch_rule=BranchRuleConfig(
            bid_top_k=4, play_mode="sample_k_plus_uniform", play_top_k=3
        ),
        branch_budget=BranchBudgetConfig(branch_rate=0.5),
        rollout=RolloutOptions(deals_per_batch=4),
    )
    collector = SeqRolloutCollector(model, train, device="cpu")
    trees = collector.collect(None, random.Random(0))
    counts = Counter()
    for tree in trees:
        for stage, n in tree.branch_decisions_by_stage.items():
            counts[stage] += n

    assert counts[-1] > 0                      # the bid is always expanded
    assert max(counts) >= 6                    # and the endgame is reached
    late = sum(counts[stage] for stage in range(6, 9))
    assert late > 0


def test_branch_rate_controls_tree_size_and_decay_moves_it_early():
    """The rate is the only thing deciding how much of a tree gets built."""

    def collect(**budget_kwargs):
        torch.manual_seed(0)
        model = SeqPlumpModel(MODEL_CONFIG).eval()
        train = SeqTrainingConfig(
            schedule_cells=(
                GameScheduleCell(hand_size=9, num_players=4, games=4),
            ),
            branch_rule=BranchRuleConfig(
                bid_top_k=4, play_mode="sample_k_plus_uniform", play_top_k=3
            ),
            branch_budget=BranchBudgetConfig(
                **budget_kwargs
            ),
            rollout=RolloutOptions(deals_per_batch=4),
        )
        collector = SeqRolloutCollector(model, train, device="cpu")
        trees = collector.collect(None, random.Random(0))
        return trees, collector.stats

    # A rate below 1 must actually be skipping eligible decisions.
    _, stats = collect(branch_rate=0.4)
    assert stats.skipped_by_placement > 0
    assert stats.blocked_by_cache == 0

    # A higher rate must produce a bigger tree.
    sparse, _ = collect(branch_rate=0.2)
    dense, _ = collect(branch_rate=0.8)
    assert sum(t.leaf_total for t in dense) > sum(t.leaf_total for t in sparse)

    # Decay pushes branch points away from the late tricks.
    flat, _ = collect(branch_rate=0.6)
    decayed, _ = collect(branch_rate=0.6, branch_rate_decay=2.0)
    deepest = lambda trees: max(
        (max(t.branched_tricks) for t in trees if t.branched_tricks), default=-1
    )
    assert deepest(decayed) <= deepest(flat)
    assert sum(t.leaf_total for t in decayed) < sum(t.leaf_total for t in flat)


def test_shape_rate_table_overrides_the_global_rate():
    """A per-shape rate must reach the gate, and the most specific one wins."""

    budget = BranchBudgetConfig(
        branch_rate=0.1,
        branch_rate_by_shape=(
            ShapeBranchRate(rate=0.5, hand_size=9),
            ShapeBranchRate(rate=0.9, num_players=4, hand_size=9),
            ShapeBranchRate(rate=0.3, num_players=5),
        ),
    )
    assert budget.rate_for_shape(3, 3) == 0.1  # nothing matches -> global
    assert budget.rate_for_shape(3, 9) == 0.5  # hand size only
    assert budget.rate_for_shape(4, 9) == 0.9  # both slots beat one
    assert budget.rate_for_shape(5, 4) == 0.3  # player count only
    # Two matches of equal specificity: the shape rule beats the global, and
    # neither of the one-slot rules is silently preferred by ordering.
    assert budget.rate_for_shape(5, 9) in (0.5, 0.3)

    def leaves(table):
        torch.manual_seed(0)
        model = SeqPlumpModel(MODEL_CONFIG).eval()
        train = SeqTrainingConfig(
            schedule_cells=(GameScheduleCell(hand_size=9, num_players=4, games=3),),
            branch_rule=BranchRuleConfig(
                bid_top_k=4, play_mode="sample_k_plus_uniform", play_top_k=3
            ),
            branch_budget=BranchBudgetConfig(
                    branch_rate=0.2,
                branch_rate_by_shape=table,
            ),
            rollout=RolloutOptions(deals_per_batch=3),
        )
        collector = SeqRolloutCollector(model, train, device="cpu")
        return sum(t.leaf_total for t in collector.collect(None, random.Random(0)))

    # The override must actually drive the gate, not just the accessor.
    assert leaves(()) < leaves((ShapeBranchRate(rate=0.9, hand_size=9),))
    # A rule for another shape must leave this one on the global rate.
    assert leaves(()) == leaves((ShapeBranchRate(rate=0.9, hand_size=5),))


def test_derived_rate_table_is_exhaustive_then_tapers():
    """Rate 1.0 while branching is cheap, then geometric down to the anchor."""

    table = build_branch_rate_table(0.5, exhaustive_until=7)
    rates = {
        (rule.num_players, rule.hand_size): rule.rate for rule in table
    }
    # One rule per scheduled shape, so no shape falls back to a global rate.
    assert len(rates) == 3 * 8
    # Short games branch every eligible decision; long games taper to the
    # anchor in equal multiplicative steps.
    for hand_size in range(3, 8):
        assert rates[(5, hand_size)] == 1.0
    assert rates[(5, 8)] == pytest.approx(0.5 ** (1 / 3))
    assert rates[(5, 9)] == pytest.approx(0.5 ** (2 / 3))
    assert rates[(5, 10)] == pytest.approx(0.5)
    assert all(0.0 < rate <= 1.0 for rate in rates.values())
    # The taper is steeper than equal-tree-size (rate * (N - 1) constant)
    # would give, because past ~8 cards time and memory bind super-linearly.
    assert rates[(5, 10)] < rates[(5, 7)] * 6 / 9
    # A wider exhaustive band must not push any rate above 1.
    wide = build_branch_rate_table(0.5, exhaustive_until=9)
    assert all(0.0 < rule.rate <= 1.0 for rule in wide)
    with pytest.raises(ValueError, match="exhaustive_until"):
        build_branch_rate_table(0.5, exhaustive_until=10)
    # Player count is left alone by default; the exponent is what moves it.
    assert rates[(3, 10)] == rates[(5, 10)]
    tilted = {
        (r.num_players, r.hand_size): r.rate
        for r in build_branch_rate_table(0.5, player_exponent=1.0)
    }
    assert tilted[(3, 10)] > tilted[(5, 10)]


def test_validate_requires_a_rate_for_every_scheduled_shape():
    """A partial table would fail deep in the wave loop; fail at validate()."""

    def config(table):
        return SeqTrainingConfig(
            schedule_cells=(
                GameScheduleCell(hand_size=5, num_players=3, games=1),
                GameScheduleCell(hand_size=9, num_players=4, games=1),
            ),
            branch_budget=BranchBudgetConfig(
                    branch_rate=None,
                branch_rate_by_shape=table,
            ),
        )

    with pytest.raises(ValueError, match=r"uncovered shapes.*\(4, 9\)"):
        config((ShapeBranchRate(rate=0.5, hand_size=5),)).validate()
    config(build_branch_rate_table(0.5)).validate()


def test_rate_skips_are_not_cache_blocks():
    """A rate skip is the mechanism working; a cache block is a misconfigured run."""

    torch.manual_seed(0)
    model = SeqPlumpModel(MODEL_CONFIG).eval()
    train = SeqTrainingConfig(
        schedule_cells=(GameScheduleCell(hand_size=8, num_players=4, games=2),),
        branch_rule=BranchRuleConfig(
            bid_top_k=4, play_mode="sample_k_plus_uniform", play_top_k=3
        ),
        branch_budget=BranchBudgetConfig(
            branch_rate=0.3,
        ),
        rollout=RolloutOptions(deals_per_batch=2),
    )
    collector = SeqRolloutCollector(model, train, device="cpu")
    collector.collect(None, random.Random(0))
    assert collector.stats.skipped_by_placement > 0
    assert collector.stats.blocked_by_cache == 0
