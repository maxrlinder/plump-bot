"""Fork-aware roster construction and seat-symmetric tournament play."""

from __future__ import annotations

import dataclasses
import json
from collections import Counter, defaultdict

import pytest
import torch

from plump.policies import HeuristicPolicy, RandomPolicy
from plump.runs import RunDirectory, atomic_write_json
from plump.seq.config import SEQ_SCHEMA_VERSION, SeqModelConfig
from plump.seq.model import SEQ_MODEL_FORMAT_VERSION, SeqPlumpModel
from plump.tournament import (
    HEURISTIC_KEY,
    RosterEntry,
    TournamentProtocol,
    build_plan,
    build_roster,
    estimate_seconds,
    lineage_checkpoints,
    load_policies,
    resolve_run_chain,
    run_tournament,
    summarize,
    tournament_payload,
    write_tournament_csv,
)
from plump.tournament_dashboard import render_tournament_dashboard


def _make_run(root, name, iterations, *, parent=None, parent_iteration=None):
    run = RunDirectory(name, root=root)
    run.checkpoints.mkdir(parents=True)
    for iteration in iterations:
        (run.checkpoints / f"iter_{iteration:06d}.pt").touch()
    metadata = {"run": name}
    if parent is not None:
        metadata["parent_checkpoint"] = str(
            parent.checkpoints / f"iter_{parent_iteration:06d}.pt"
        )
        metadata["parent_iteration"] = parent_iteration
    run.path.mkdir(parents=True, exist_ok=True)
    run.metadata.write_text(json.dumps(metadata))
    return run


def _protocol(**overrides) -> TournamentProtocol:
    values = dict(
        player_counts=(3, 4),
        hand_sizes=(3,),
        deals_per_match=2,
        waves=1,
        deal_seed=1234,
        action_seed=17,
        schedule_seed=7,
        bootstrap_samples=64,
        greedy=True,
    )
    values.update(overrides)
    return TournamentProtocol(**values)


def test_lineage_stitches_a_fork_without_duplicating_the_seed(tmp_path):
    parent = _make_run(tmp_path, "base", range(0, 1001, 100))
    child = _make_run(
        tmp_path,
        "fork",
        range(1000, 1401, 100),
        parent=parent,
        parent_iteration=1000,
    )

    assert [run.name for run, _ in resolve_run_chain(child)] == ["base", "fork"]
    lineage = lineage_checkpoints(child)
    iterations = [iteration for iteration, _, _ in lineage]
    assert iterations == sorted(iterations)
    assert len(iterations) == len(set(iterations))
    # The fork's seeded copy of iteration 1000 belongs to the parent that made it.
    owner = {iteration: run.name for iteration, run, _ in lineage}
    assert owner[1000] == "base"
    assert owner[1100] == "fork"


def test_stride_is_anchored_at_the_start_so_it_spans_the_fork(tmp_path):
    parent = _make_run(tmp_path, "base", range(0, 1001, 100))
    child = _make_run(
        tmp_path,
        "fork",
        range(1000, 1401, 100),
        parent=parent,
        parent_iteration=1000,
    )

    roster = build_roster(
        child,
        from_iteration=500,
        stride=200,
        include_latest=False,
        include_heuristic=False,
    )
    assert [entry.iteration for entry in roster] == [500, 700, 900, 1100, 1300]
    assert [entry.run for entry in roster] == ["base"] * 3 + ["fork"] * 2


def test_latest_checkpoint_is_included_off_stride(tmp_path):
    run = _make_run(tmp_path, "base", range(0, 1001, 100))

    roster = build_roster(
        run,
        from_iteration=0,
        stride=300,
        include_latest=True,
        include_heuristic=True,
    )
    assert [entry.iteration for entry in roster if not entry.is_heuristic] == [
        0,
        300,
        600,
        900,
        1000,
    ]
    assert roster[-1].key == HEURISTIC_KEY


def test_exact_checkpoint_roster_ignores_stride_and_latest(tmp_path):
    run = _make_run(tmp_path, "base", range(0, 1001, 100))

    roster = build_roster(
        run,
        from_iteration=0,
        stride=100,
        include_latest=True,
        include_heuristic=False,
        iterations=(0, 300, 700),
    )

    assert [entry.iteration for entry in roster] == [0, 300, 700]


def test_exact_checkpoint_roster_rejects_missing_iteration(tmp_path):
    run = _make_run(tmp_path, "base", (0, 100, 200))

    with pytest.raises(FileNotFoundError, match="150"):
        build_roster(run, iterations=(0, 150), include_heuristic=False)


def test_every_participant_holds_every_seat_and_bidding_position():
    entries = [
        RosterEntry(key=f"p{index}", iteration=index, run="r", path=None)
        for index in range(4)
    ]
    plan = build_plan(entries, _protocol(player_counts=(4,), hand_sizes=(3,)))
    policies = [RandomPolicy(seed=index) for index in range(len(entries))]
    outcomes = run_tournament(plan, policies)

    per_entry_seats = defaultdict(Counter)
    per_entry_hands = defaultdict(Counter)
    for outcome in outcomes:
        per_entry_seats[outcome.entry][outcome.bidding_position] += 1
        per_entry_hands[outcome.entry][(outcome.match, outcome.deal, outcome.rotation)] += 1

    assert len(per_entry_seats) == len(entries)
    for counts in per_entry_seats.values():
        # Each participant occupies each bidding position exactly as often.
        assert len(set(counts.values())) == 1
        assert sorted(counts) == [0, 1, 2, 3]


def test_relative_rewards_are_zero_sum_within_each_round():
    entries = [
        RosterEntry(key=f"p{index}", iteration=index, run="r", path=None)
        for index in range(3)
    ]
    plan = build_plan(entries, _protocol(player_counts=(3,), hand_sizes=(3,)))
    policies = [RandomPolicy(seed=index) for index in range(len(entries))]
    outcomes = run_tournament(plan, policies)

    totals = defaultdict(float)
    for outcome in outcomes:
        totals[(outcome.match, outcome.deal, outcome.rotation)] += outcome.relative_reward
    assert totals
    for total in totals.values():
        assert total == pytest.approx(0.0, abs=1e-9)


def test_identical_policies_score_alike_and_the_field_sums_to_zero():
    entries = [
        RosterEntry(key=f"p{index}", iteration=index, run="r", path=None)
        for index in range(4)
    ]
    plan = build_plan(
        entries,
        _protocol(player_counts=(4,), hand_sizes=(3, 4), deals_per_match=8),
    )
    # One shared heuristic instance in every seat: any spread is pure seat bias.
    shared = HeuristicPolicy()
    outcomes = run_tournament(plan, [shared] * len(entries))
    report = summarize(plan, outcomes)

    rewards = [entry.relative_reward for entry in report.entries]
    assert len(rewards) == len(entries)
    assert sum(rewards) == pytest.approx(0.0, abs=1e-9)
    # Full seat rotation over shared deals leaves nothing to separate clones.
    assert max(rewards) - min(rewards) == pytest.approx(0.0, abs=1e-9)


def test_summary_and_csv_round_trip(tmp_path):
    entries = [
        RosterEntry(key=f"p{index}", iteration=100 * index, run="r", path=None)
        for index in range(3)
    ]
    plan = build_plan(entries, _protocol(player_counts=(3,), hand_sizes=(3,)))
    policies = [RandomPolicy(seed=index) for index in range(len(entries))]
    report = summarize(plan, run_tournament(plan, policies))

    assert [entry.iteration for entry in report.entries] == [0, 100, 200]
    for entry in report.entries:
        assert entry.rounds == plan.rounds_per_entry[entry.key]
        assert entry.relative_reward_ci_low <= entry.relative_reward_ci_high

    path = tmp_path / "t.csv"
    write_tournament_csv(path, report)
    lines = path.read_text().strip().splitlines()
    assert lines[0].startswith("iteration,run,key,rounds,relative_reward")
    assert len(lines) == 1 + len(entries)


def test_plan_costs_match_the_played_tournament():
    entries = [
        RosterEntry(key=f"p{index}", iteration=index, run="r", path=None)
        for index in range(5)
    ]
    plan = build_plan(entries, _protocol(player_counts=(3, 4), hand_sizes=(3, 4)))
    outcomes = run_tournament(plan, [RandomPolicy(seed=i) for i in range(5)])

    assert sum(plan.rounds_per_entry.values()) == len(outcomes)
    # Each round-seat contributes one bid plus one play per card.
    assert plan.decisions == sum(1 + outcome.hand_size for outcome in outcomes)
    assert plan.round_seats == len(outcomes)


def test_tournament_emits_monotonic_complete_chunk_snapshots():
    entries = [
        RosterEntry(key=f"p{index}", iteration=index, run="r", path=None)
        for index in range(4)
    ]
    plan = build_plan(entries, _protocol(player_counts=(3,), hand_sizes=(3, 4)))
    snapshots = []

    outcomes = run_tournament(
        plan,
        [RandomPolicy(seed=index) for index in range(4)],
        max_concurrent_rounds=1,
        snapshot=lambda progress, rows: snapshots.append((progress, len(rows))),
    )

    assert len(snapshots) > 1
    assert [progress.completed_matches for progress, _ in snapshots] == sorted(
        progress.completed_matches for progress, _ in snapshots
    )
    assert [rows for _, rows in snapshots] == sorted(rows for _, rows in snapshots)
    final, rows = snapshots[-1]
    assert final.completed_matches == len(plan.matches)
    assert final.completed_decisions == plan.decisions
    assert final.completed_round_seats == plan.round_seats == rows == len(outcomes)
    assert final.fraction == pytest.approx(1.0)


def _write_model_checkpoint(path, config: SeqModelConfig, iteration: int) -> None:
    model = SeqPlumpModel(config)
    torch.save(
        {
            "iteration": iteration,
            "schema_version": SEQ_SCHEMA_VERSION,
            "model_format_version": SEQ_MODEL_FORMAT_VERSION,
            "model_config": dataclasses.asdict(config),
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def test_mixed_checkpoint_architectures_play_at_the_same_table(tmp_path):
    narrow_path = tmp_path / "iter_000100.pt"
    wide_path = tmp_path / "iter_000200.pt"
    _write_model_checkpoint(
        narrow_path,
        SeqModelConfig(
            max_players=3,
            max_hand_size=3,
            d_model=32,
            n_layers=1,
            n_heads=4,
            d_ff=64,
            turn_token="off",
        ),
        100,
    )
    _write_model_checkpoint(
        wide_path,
        SeqModelConfig(
            max_players=3,
            max_hand_size=3,
            d_model=48,
            n_layers=2,
            n_heads=6,
            d_ff=96,
            trick_win_token=False,
            turn_token="bid",
            embedding_rms_norm=True,
        ),
        200,
    )
    roster = [
        RosterEntry("narrow", 100, "v2", narrow_path),
        RosterEntry("wide", 200, "v4", wide_path),
        RosterEntry("heuristic", -1, "", None),
    ]
    plan = build_plan(
        roster,
        _protocol(player_counts=(3,), hand_sizes=(3,), deals_per_match=1),
    )

    policies = load_policies(roster, device="cpu")
    outcomes = run_tournament(plan, policies)

    assert policies[0].config.d_model == 32
    assert policies[1].config.d_model == 48
    assert policies[0].config.turn_token == "off"
    assert policies[1].config.turn_token == "bid"
    assert len(outcomes) == plan.round_seats


def test_running_dashboard_renders_with_only_the_first_shape_complete(tmp_path):
    entries = [
        RosterEntry(key=f"p{index}", iteration=100 * index, run="v2", path=None)
        for index in range(4)
    ]
    plan = build_plan(entries, _protocol(player_counts=(3,), hand_sizes=(3, 4)))
    snapshots = []
    run_tournament(
        plan,
        [RandomPolicy(seed=index) for index in range(4)],
        max_concurrent_rounds=1,
        snapshot=lambda progress, rows: snapshots.append((progress, list(rows))),
    )
    progress, partial = snapshots[0]
    run = RunDirectory("live", root=tmp_path)
    run.path.mkdir()
    payload_path = run.tournaments / "live.json"
    payload = tournament_payload(
        run,
        plan,
        summarize(plan, partial, bootstrap_samples=0),
        device="cpu",
        elapsed_sec=1.0,
        status="running",
        progress=progress,
    )
    atomic_write_json(payload_path, payload)

    output = render_tournament_dashboard(payload_path, run.tournament_dashboard, dpi=40)

    assert output == run.tournament_dashboard
    assert output.stat().st_size > 0


def test_batching_groups_rows_by_acting_participant():
    class _Recorder(RandomPolicy):
        def __init__(self, seed):
            super().__init__(seed=seed)
            self.batches = []

        def act_many(self, envs, *, rngs=None):
            self.batches.append(len(envs))
            rngs = rngs or [None] * len(envs)
            return [self.act(env, rng=rng) for env, rng in zip(envs, rngs)]

    entries = [
        RosterEntry(key=f"p{index}", iteration=index, run="r", path=None)
        for index in range(4)
    ]
    plan = build_plan(
        entries,
        _protocol(player_counts=(4,), hand_sizes=(3,), deals_per_match=16),
    )
    policies = [_Recorder(index) for index in range(4)]
    run_tournament(plan, policies)

    # Play order follows trick winners, so a participant is on turn in more
    # rows at some steps than others; what throughput depends on is the mean,
    # which the rotation layout pins to one row per shared deal.
    deals = plan.protocol.deals_per_match
    for policy in policies:
        assert policy.batches
        mean_batch = sum(policy.batches) / len(policy.batches)
        assert 0.5 * deals <= mean_batch <= 2.0 * deals


def test_cost_estimate_scales_with_the_planned_work():
    entries = [
        RosterEntry(key=f"p{index}", iteration=index, run="r", path=None)
        for index in range(8)
    ]
    small = build_plan(entries, _protocol(player_counts=(4,), hand_sizes=(3,)))
    large = build_plan(entries, _protocol(player_counts=(4,), hand_sizes=(3, 10)))

    assert estimate_seconds(small, device="mps") > 0
    assert estimate_seconds(large, device="mps") > estimate_seconds(small, device="mps")
    # Longer hands cost more per decision, not just more decisions.
    assert (
        estimate_seconds(small, device="cpu") > estimate_seconds(small, device="mps")
    )


def test_rates_are_interpolated_over_sequence_length():
    from plump.tournament import MEASURED_RATES, _interpolate_rate, sequence_scale

    # Both axes feed one scale: five players at 6 cards costs more per decision
    # than three players at 9, despite the shorter hand.
    assert sequence_scale(5, 6) > sequence_scale(3, 9)

    table = MEASURED_RATES["cpu"]
    assert _interpolate_rate(table, 12) == table[12]
    assert _interpolate_rate(table, 55) == table[55]
    assert table[36] < _interpolate_rate(table, 32) < table[28]
    # Out-of-range shapes clamp rather than extrapolate to nonsense.
    assert _interpolate_rate(table, 1) == table[12]
    assert _interpolate_rate(table, 500) == table[55]
