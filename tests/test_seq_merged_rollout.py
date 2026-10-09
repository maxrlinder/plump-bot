"""Merged waves: one continuously-batched loop over the whole schedule.

The lockstep loop runs one cell at a time because every leaf in it must share
a token position. Merging drops that constraint, so the thing to prove is that
nothing else changed: the same deals, the same trajectories, the same training
positions, and a pool whose live rows stay dense as games retire.
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest
import torch

from plump.seq.config import (
    BranchBudgetConfig,
    GameScheduleCell,
    RolloutOptions,
    SeqModelConfig,
    SeqTrainingConfig,
)
from plump.seq.kv import KVCache
from plump.seq.model import SeqPlumpModel
from plump.seq.rollout import SeqRolloutCollector


MODEL_CONFIG = SeqModelConfig(d_model=32, n_layers=2, n_heads=4, d_ff=64)

# Deliberately ragged: three player counts and four hand sizes, so one merged
# wave holds rows at different lengths, different table sizes and different
# token counts per wave, and the shapes retire at four different times.
CELLS = (
    GameScheduleCell(hand_size=3, num_players=3, games=2),
    GameScheduleCell(hand_size=6, num_players=4, games=2),
    GameScheduleCell(hand_size=4, num_players=5, games=2),
    GameScheduleCell(hand_size=5, num_players=3, games=3),
)


def make_collector(*, merged: bool, cells=CELLS, merged_max_rows=None, seed=0):
    torch.manual_seed(seed)
    model = SeqPlumpModel(MODEL_CONFIG).eval()
    train = SeqTrainingConfig(
        schedule_cells=tuple(cells),
        policy_objective="ppo",
        branch_budget=BranchBudgetConfig(branch_rate=0.0),
        rollout=RolloutOptions(
            merged_waves=merged, merged_max_rows=merged_max_rows
        ),
    )
    return SeqRolloutCollector(model, train, device="cpu")


def summarize(trees):
    """Everything downstream of collect() reads off a tree."""

    return sorted(
        (
            tree.num_players,
            tree.hand_size,
            tree.bidding_start_player,
            tree.focal,
            tuple(sorted((seat, tuple(hand)) for seat, hand in tree.initial_hands.items())),
            tuple(
                (record.position, record.phase, record.seat, record.action_index)
                for leaf in tree.leaves
                for record in leaf.decisions
            ),
            tuple(sorted(leaf.value_targets().items()) for leaf in tree.leaves)[0],
        )
        for tree in trees
    )


def test_merged_reproduces_the_lockstep_trees_exactly():
    """Same seed, same deals, same trajectories -- only the batching differs."""

    lockstep = make_collector(merged=False).collect(None, random.Random(7))
    merged = make_collector(merged=True).collect(None, random.Random(7))

    assert len(merged) == len(lockstep) == 9
    assert summarize(merged) == summarize(lockstep)


def test_merged_matches_when_the_pool_is_too_small_to_hold_the_schedule():
    """Chunking the schedule must not change any game, only the batching."""

    reference = make_collector(merged=True).collect(None, random.Random(7))
    chunked = make_collector(merged=True, merged_max_rows=8).collect(
        None, random.Random(7)
    )

    assert summarize(chunked) == summarize(reference)


def test_every_training_position_is_owned_exactly_once():
    """The positions a merged leaf claims must tile its own sequence."""

    collector = make_collector(merged=True)
    for tree in collector.collect(None, random.Random(11)):
        length = collector.model_config.seq_len(tree.num_players, tree.hand_size)
        for leaf in tree.leaves:
            owned = [
                position for positions, _, _ in leaf.segments for position in positions
            ]
            assert sorted(owned) == list(range(length))
            assert leaf.cache_length <= length


def count_row_moves(monkeypatch) -> list[int]:
    moves: list[int] = []
    original = KVCache.branch_copy

    def spy(self, parents, children, length):
        moves.append(int(parents.numel()))
        return original(self, parents, children, length)

    monkeypatch.setattr(KVCache, "branch_copy", spy)
    return moves


def test_retirement_order_keeps_the_live_rows_dense(monkeypatch):
    """The fast path: nothing is ever moved, so the read stays a plain slice."""

    moves = count_row_moves(monkeypatch)
    make_collector(merged=True).collect(None, random.Random(3))
    assert moves == []


def test_a_shape_that_retires_out_of_turn_is_compacted(monkeypatch):
    """The dense-prefix invariant is repaired, not merely relied upon.

    Retirement order normally makes compaction a no-op, so force the pool into
    the layout it protects against: deal the shortest-lived games first, which
    leaves dead rows in the middle of the pool, and check the trees are still
    the ones the lockstep loop produces.
    """

    reversed_cells = tuple(reversed(CELLS))
    lockstep = make_collector(merged=False, cells=reversed_cells).collect(
        None, random.Random(3)
    )
    moves = count_row_moves(monkeypatch)
    collector = make_collector(merged=True, cells=reversed_cells)
    # A constant key makes the sort a no-op, so the pool is laid out in dealing
    # order -- which for these cells retires a middle block first.
    collector._decision_waves = staticmethod(lambda players, hand_size: 0)
    merged = collector.collect(None, random.Random(3))

    assert sum(moves) > 0
    assert summarize(merged) == summarize(lockstep)


def test_merging_turns_itself_off_when_an_arm_cannot_merge():
    """A config the merged loop cannot honour falls back instead of failing."""

    collector = make_collector(merged=True)
    assert collector.merged_waves_available([0] * len(CELLS))
    assert not collector.merged_waves_available([0, 1, 0, 0])

    neurd = make_collector(merged=True)
    neurd.train = replace(neurd.train, policy_objective="neurd")
    assert not neurd.merged_waves_available([0] * len(CELLS))


def test_merged_runs_fewer_and_wider_forwards():
    """The point of the exercise: the same work in far fewer launches."""

    lockstep = make_collector(merged=False)
    lockstep.collect(None, random.Random(5))
    merged = make_collector(merged=True)
    merged.collect(None, random.Random(5))

    # Same rows pushed through the trunk, in one loop instead of four.
    assert merged.stats.decisions == lockstep.stats.decisions
    assert merged.stats.peak_cache_rows > lockstep.stats.peak_cache_rows


def test_mass_strata_stay_nonempty_for_a_sharp_policy():
    """A converged policy underflows its tail to exactly zero.

    Placing a zero-probability action does not raise its stratum's mass, so a
    naive lightest-bin rule lets bin 0 swallow every zero and leaves the rest
    empty -- and the caller then has no representative to draw from them.
    """

    import numpy as np

    from plump.seq.rollout import SeqRolloutCollector

    legal = list(range(11))
    probs = np.zeros(11, dtype=np.float32)
    probs[3] = 1.0  # everything else underflowed to exactly 0.0
    groups = SeqRolloutCollector._policy_mass_strata(legal, probs, 5)
    assert len(groups) == 5
    assert all(group for group in groups), f"empty stratum: {groups}"
    flat = [action for group in groups for action in group]
    assert sorted(flat) == legal, "strata must partition the legal set"


def test_mass_strata_unchanged_for_an_ordinary_policy():
    """The healthy path must behave exactly as before the fix."""

    import numpy as np

    from plump.seq.rollout import SeqRolloutCollector

    legal = list(range(6))
    probs = np.array([0.30, 0.25, 0.20, 0.13, 0.07, 0.05], dtype=np.float32)
    groups = SeqRolloutCollector._policy_mass_strata(legal, probs, 3)
    assert all(group for group in groups)
    assert sorted(a for g in groups for a in g) == legal
    # Highest-probability actions seed distinct strata, lightest-bin after.
    assert [g[0] for g in groups] == [0, 1, 2]
