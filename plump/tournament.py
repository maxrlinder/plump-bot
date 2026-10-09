"""Seat-symmetric round-robin tournaments across a checkpoint lineage.

`plump evaluate` scores one focal checkpoint against a table of heuristic
clones, which carries a systematic lone-seat advantage. A tournament instead
seats several checkpoints at the same table and replays every deal once per
cyclic seat rotation, so each participant holds every hand and every bidding
position exactly the same number of times. Relative rewards sum to zero within
a rotation, so no seat, hand, or bidding-order effect survives aggregation.
"""

from __future__ import annotations

import csv
import dataclasses
import io
import json
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from plump.env import PlumpEnv
from plump.evaluation import (
    BaseDeal,
    DealBank,
    _bootstrap_mean_ci,
    _mean,
    _policy_actions,
    _relative_rewards,
)
from plump.policies import ActionPolicy, HeuristicPolicy
from plump.rounds import RoundSpec, round_game_config, rules_fingerprint
from plump.runs import RunDirectory, atomic_write_text
from plump.seq.policy import SeqModelPolicy

TOURNAMENT_FORMAT_VERSION = 2
HEURISTIC_KEY = "heuristic"
_INTERVAL_CHECKPOINT = re.compile(r"iter_(\d+)\.pt")


@dataclass(frozen=True)
class RosterEntry:
    """One tournament participant: a lineage checkpoint or the heuristic."""

    key: str
    iteration: int
    run: str
    path: Path | None

    @property
    def is_heuristic(self) -> bool:
        return self.path is None

    def as_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "iteration": self.iteration,
            "run": self.run,
            "path": None if self.path is None else str(self.path),
        }


@dataclass(frozen=True)
class TournamentProtocol:
    """Inputs that make one tournament reproducible and comparable."""

    player_counts: tuple[int, ...]
    hand_sizes: tuple[int, ...]
    deals_per_match: int
    waves: int
    deal_seed: int
    action_seed: int
    schedule_seed: int
    bootstrap_samples: int
    greedy: bool

    def as_json(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["player_counts"] = list(self.player_counts)
        value["hand_sizes"] = list(self.hand_sizes)
        return value


@dataclass(frozen=True)
class Match:
    """A fixed set of participants playing a shared deal bank at one shape."""

    index: int
    spec: RoundSpec
    entries: tuple[int, ...]
    bidding_start: int


@dataclass(frozen=True)
class RoundOutcome:
    """One participant's result in one seat rotation of one deal."""

    entry: int
    match: int
    deal: int
    rotation: int
    num_players: int
    hand_size: int
    bidding_position: int
    raw_score: float
    relative_reward: float
    bid_hit: float
    bid_abs_error: float


@dataclass
class EntryReport:
    key: str
    iteration: int
    run: str
    rounds: int
    relative_reward: float
    relative_reward_ci_low: float
    relative_reward_ci_high: float
    raw_score: float
    bid_hit_rate: float
    bid_abs_error: float
    cells: dict[str, float]


@dataclass
class TournamentReport:
    rounds: int
    matches: int
    rules_hash: str
    entries: list[EntryReport]


@dataclass
class TournamentPlan:
    """Everything needed to run — and to cost — a tournament before playing."""

    roster: list[RosterEntry]
    protocol: TournamentProtocol
    matches: list[Match]
    deals: dict[tuple[int, int], list[BaseDeal]]

    @property
    def rounds(self) -> int:
        return sum(
            self.protocol.deals_per_match * match.spec.num_players
            for match in self.matches
        )

    @property
    def decisions(self) -> int:
        """Model forward rows: every seat of every round acts each decision."""

        return sum(
            self.protocol.deals_per_match
            * match.spec.num_players
            * match.spec.num_players
            * (1 + match.spec.hand_size)
            for match in self.matches
        )

    @property
    def round_seats(self) -> int:
        """Participant result rows produced by the complete tournament."""

        return sum(
            self.protocol.deals_per_match * match.spec.num_players**2
            for match in self.matches
        )

    @property
    def rounds_per_entry(self) -> dict[str, int]:
        counts: dict[str, int] = {entry.key: 0 for entry in self.roster}
        for match in self.matches:
            played = self.protocol.deals_per_match * match.spec.num_players
            for index in match.entries:
                counts[self.roster[index].key] += played
        return counts


# --- roster construction -------------------------------------------------


def resolve_run_chain(run: RunDirectory) -> list[tuple[RunDirectory, int]]:
    """Walk `parent_checkpoint` back to the root, oldest run first.

    Each element pairs a run with the iteration it was forked at; the root run
    reports -1 so that its own iteration 0 is still in range.
    """

    chain: list[tuple[RunDirectory, int]] = []
    seen: set[str] = set()
    current = run
    while True:
        if current.name in seen:
            raise ValueError(f"Run lineage for {run.name!r} contains a cycle.")
        seen.add(current.name)
        try:
            metadata = json.loads(current.metadata.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            metadata = {}
        parent_path = metadata.get("parent_checkpoint")
        parent_iteration = metadata.get("parent_iteration")
        if not parent_path:
            chain.append((current, -1))
            break
        chain.append((current, int(parent_iteration or 0)))
        parent_run = RunDirectory(Path(parent_path).parent.parent.name, root=current.root)
        if not parent_run.exists:
            raise FileNotFoundError(
                f"Parent run directory is missing: {parent_run.path}"
            )
        current = parent_run
    chain.reverse()
    return chain


def _run_checkpoints(run: RunDirectory) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for path in run.checkpoints.glob("iter_*.pt"):
        match = _INTERVAL_CHECKPOINT.fullmatch(path.name)
        if match is not None and path.is_file():
            found[int(match.group(1))] = path.resolve()
    return found


def lineage_checkpoints(run: RunDirectory) -> list[tuple[int, RunDirectory, Path]]:
    """Return the single training history behind `run`, oldest iteration first.

    A fork contributes only the iterations it actually trained: everything at
    or below its fork point belongs to its parent, so the seeded copy a fork
    starts from never shadows the ancestor that produced it.
    """

    chain = resolve_run_chain(run)
    forks = [parent_iteration for _, parent_iteration in chain]
    lineage: list[tuple[int, RunDirectory, Path]] = []
    for position, (member, fork_point) in enumerate(chain):
        upper = forks[position + 1] if position + 1 < len(chain) else None
        for iteration, path in _run_checkpoints(member).items():
            if iteration <= fork_point:
                continue
            if upper is not None and iteration > upper:
                continue
            lineage.append((iteration, member, path))
    lineage.sort(key=lambda row: row[0])
    return lineage


def build_roster(
    run: RunDirectory,
    *,
    from_iteration: int = 0,
    to_iteration: int | None = None,
    stride: int = 200,
    include_latest: bool = True,
    include_heuristic: bool = True,
    iterations: Sequence[int] | None = None,
) -> list[RosterEntry]:
    """Select exact lineage iterations or a stride anchored at `from_iteration`.

    Anchoring the stride at the start rather than at each run keeps the spacing
    uniform across a fork, where the child's iteration counter continues the
    parent's. Supplying ``iterations`` bypasses stride/latest selection and is
    useful for small, controlled comparison fields.
    """

    lineage = lineage_checkpoints(run)
    if not lineage:
        raise FileNotFoundError(f"No interval checkpoints in the lineage of {run.name!r}.")
    if iterations is not None:
        requested = tuple(dict.fromkeys(int(value) for value in iterations))
        if not requested:
            raise ValueError("iterations must contain at least one checkpoint.")
        by_iteration = {row[0]: row for row in lineage}
        missing = [iteration for iteration in requested if iteration not in by_iteration]
        if missing:
            raise FileNotFoundError(
                "Requested lineage checkpoints are missing: "
                + ", ".join(str(iteration) for iteration in missing)
            )
        selected = [by_iteration[iteration] for iteration in requested]
        selected.sort(key=lambda row: row[0])
    else:
        if stride < 1:
            raise ValueError("stride must be positive.")
        ceiling = lineage[-1][0] if to_iteration is None else to_iteration
        selected = [
            row
            for row in lineage
            if from_iteration <= row[0] <= ceiling
            and (row[0] - from_iteration) % stride == 0
        ]
        if include_latest:
            newest = max(
                (row for row in lineage if row[0] <= ceiling),
                key=lambda row: row[0],
            )
            if not selected or selected[-1][0] != newest[0]:
                selected.append(newest)
        if not selected:
            raise FileNotFoundError(
                f"No checkpoints in [{from_iteration}, {ceiling}] on stride {stride}."
            )
    roster = [
        RosterEntry(
            key=f"iter_{iteration:06d}",
            iteration=iteration,
            run=member.name,
            path=path,
        )
        for iteration, member, path in selected
    ]
    if include_heuristic:
        roster.append(RosterEntry(key=HEURISTIC_KEY, iteration=-1, run="", path=None))
    return roster


# --- scheduling ----------------------------------------------------------


def _partition(indices: Sequence[int], size: int, rng: random.Random) -> list[tuple[int, ...]]:
    """Split a shuffled roster into distinct-membership tables of `size`."""

    order = list(indices)
    rng.shuffle(order)
    tables: list[tuple[int, ...]] = []
    for start in range(0, len(order), size):
        table = order[start : start + size]
        if len(table) < size:
            spare = [index for index in indices if index not in table]
            if len(spare) < size - len(table):
                raise ValueError("Roster is too small for the requested table size.")
            table = table + rng.sample(spare, size - len(table))
        tables.append(tuple(table))
    return tables


def build_plan(
    roster: Sequence[RosterEntry],
    protocol: TournamentProtocol,
) -> TournamentPlan:
    """Lay out every match and its shared deal bank without playing anything."""

    if protocol.deals_per_match < 1:
        raise ValueError("deals_per_match must be positive.")
    if protocol.waves < 1:
        raise ValueError("waves must be positive.")
    smallest = min(protocol.player_counts)
    if len(roster) < smallest:
        raise ValueError(
            f"A {smallest}-player table needs at least {smallest} roster entries."
        )
    rng = random.Random(protocol.schedule_seed)
    indices = list(range(len(roster)))
    matches: list[Match] = []
    deals: dict[tuple[int, int], list[BaseDeal]] = {}
    for num_players in protocol.player_counts:
        for hand_size in protocol.hand_sizes:
            spec = RoundSpec(num_players, hand_size)
            spec.validate()
            # One deal bank per shape, shared by every match at that shape, so
            # participants are compared on common random deals.
            bank = DealBank.generate(
                player_counts=(num_players,),
                hand_sizes=(hand_size,),
                deals_per_configuration=protocol.deals_per_match,
                seed=protocol.deal_seed + num_players * 1_009 + hand_size,
            )
            deals[(num_players, hand_size)] = list(bank.deals)
            for _ in range(protocol.waves):
                for table in _partition(indices, num_players, rng):
                    matches.append(
                        Match(
                            index=len(matches),
                            spec=spec,
                            entries=table,
                            bidding_start=rng.randrange(num_players),
                        )
                    )
    return TournamentPlan(
        roster=list(roster),
        protocol=protocol,
        matches=matches,
        deals=deals,
    )


# --- cost model ----------------------------------------------------------

# Decisions/sec measured on this machine at --deals-per-match 64, with the full
# 84-policy roster resident and a training run competing for the same device.
# Cost keys on the sequence length each decision attends over, which is about
# num_players * (1 + hand_size); rates from the whole 24-shape grid collapse
# onto that one axis, so player count matters as much as hand size.
MEASURED_RATES = {
    "mps": {12: 3620.0, 20: 3600.0, 28: 2600.0, 33: 2400.0, 55: 2130.0},
    "cpu": {12: 3205.0, 20: 2340.0, 28: 1790.0, 36: 1420.0, 44: 1150.0, 55: 995.0},
}


def sequence_scale(num_players: int, hand_size: int) -> int:
    """Tokens a decision attends over: every seat's bid plus every card played."""

    return num_players * (1 + hand_size)


def _interpolate_rate(table: dict[int, float], scale: int) -> float:
    points = sorted(table)
    if scale <= points[0]:
        return table[points[0]]
    if scale >= points[-1]:
        return table[points[-1]]
    for low, high in zip(points, points[1:]):
        if low <= scale <= high:
            weight = (scale - low) / (high - low)
            return table[low] + weight * (table[high] - table[low])
    return table[points[-1]]


def estimate_seconds(plan: TournamentPlan, *, device: str) -> float:
    """Predict wall time from per-shape decision rates measured on this box.

    Rates were taken at the default batch width; a much smaller
    --deals-per-match makes each forward less efficient and the real run
    slower than this estimate. Long-hand, many-player shapes dominate the
    total, so the blended figure sits well below the headline rate.
    """

    table = MEASURED_RATES.get(str(device).split(":")[0], MEASURED_RATES["cpu"])
    total = 0.0
    for match in plan.matches:
        num_players = match.spec.num_players
        scale = sequence_scale(num_players, match.spec.hand_size)
        decisions = plan.protocol.deals_per_match * num_players * num_players * (
            1 + match.spec.hand_size
        )
        total += decisions / _interpolate_rate(table, scale)
    return total


# --- execution -----------------------------------------------------------


@dataclass
class _Row:
    match: Match
    deal: BaseDeal
    deal_index: int
    rotation: int
    env: PlumpEnv
    rng: random.Random
    seat_entries: tuple[int, ...]


@dataclass(frozen=True)
class TournamentProgress:
    """Durable progress counters emitted after each completed match chunk."""

    current_shape: tuple[int, int]
    completed_matches: int
    total_matches: int
    completed_decisions: int
    total_decisions: int
    completed_round_seats: int
    total_round_seats: int

    @property
    def fraction(self) -> float:
        return self.completed_decisions / max(self.total_decisions, 1)

    def as_json(self) -> dict[str, Any]:
        return {
            "current_shape": list(self.current_shape),
            "completed_matches": self.completed_matches,
            "total_matches": self.total_matches,
            "completed_decisions": self.completed_decisions,
            "total_decisions": self.total_decisions,
            "completed_round_seats": self.completed_round_seats,
            "total_round_seats": self.total_round_seats,
            "fraction": self.fraction,
        }


def _new_row(
    match: Match,
    deal: BaseDeal,
    deal_index: int,
    rotation: int,
    action_seed: int,
) -> _Row:
    n = match.spec.num_players
    # Participant j sits at seat (j + rotation) % n and is dealt that seat's
    # hand, so across n rotations every participant holds every hand once and
    # occupies every bidding position once.
    seat_entries = tuple(match.entries[(seat - rotation) % n] for seat in range(n))
    env = PlumpEnv(
        round_game_config(
            match.spec,
            bidding_start_player=match.bidding_start,
            manual_hands={seat: list(deal.hands[seat]) for seat in range(n)},
        )
    )
    env.reset()
    return _Row(
        match=match,
        deal=deal,
        deal_index=deal_index,
        rotation=rotation,
        env=env,
        rng=random.Random(
            action_seed * 1_000_003 + match.index * 9_176 + deal_index * 97 + rotation
        ),
        seat_entries=seat_entries,
    )


def _row_outcomes(row: _Row) -> list[RoundOutcome]:
    round_state = row.env.state.current_round
    relative = _relative_rewards(round_state.round_scores)
    bids = {item.player: item.value for item in round_state.bids}
    n = row.match.spec.num_players
    outcomes = []
    for seat in range(n):
        bid = bids[seat]
        outcomes.append(
            RoundOutcome(
                entry=row.seat_entries[seat],
                match=row.match.index,
                deal=row.deal_index,
                rotation=row.rotation,
                num_players=n,
                hand_size=row.match.spec.hand_size,
                bidding_position=(seat - row.match.bidding_start) % n,
                raw_score=float(round_state.round_scores[seat]),
                relative_reward=relative[seat],
                bid_hit=float(round_state.tricks_won[seat] == bid),
                bid_abs_error=float(abs(round_state.tricks_won[seat] - bid)),
            )
        )
    return outcomes


def load_policies(
    roster: Sequence[RosterEntry],
    *,
    device: str | torch.device,
    greedy: bool = True,
    progress=None,
) -> list[ActionPolicy]:
    """Load every participant once, using each checkpoint's own architecture.

    Sequence schema compatibility is checked by ``from_checkpoint``, while
    model width, conditioning paths, normalization and auxiliary-head presence
    come from that checkpoint's embedded ``model_config``.  A mixed lineage
    therefore does not get coerced into the newest architecture.
    """

    policies: list[ActionPolicy] = []
    for position, entry in enumerate(roster):
        if entry.is_heuristic:
            policies.append(HeuristicPolicy())
        else:
            policies.append(
                SeqModelPolicy.from_checkpoint(
                    entry.path,
                    device=device,
                    greedy=greedy,
                    name=entry.key,
                )
            )
        if progress is not None:
            progress(position + 1, len(roster), entry)
    return policies


def play_matches(
    matches: Iterable[Match],
    plan: TournamentPlan,
    policies: Sequence[ActionPolicy],
) -> list[RoundOutcome]:
    """Play a group of matches concurrently, batching by acting participant.

    Every match in the group holds the same shape and advances in lockstep, so
    grouping active rows by which participant is on turn yields one forward per
    participant of `deals_per_match` rows.
    """

    active: list[_Row] = []
    for match in matches:
        bank = plan.deals[(match.spec.num_players, match.spec.hand_size)]
        for deal_index, deal in enumerate(bank):
            for rotation in range(match.spec.num_players):
                active.append(
                    _new_row(
                        match,
                        deal,
                        deal_index,
                        rotation,
                        plan.protocol.action_seed,
                    )
                )
    outcomes: list[RoundOutcome] = []
    while active:
        by_entry: dict[int, list[_Row]] = defaultdict(list)
        for row in active:
            by_entry[row.seat_entries[row.env.current_player()]].append(row)
        actions: dict[int, object] = {}
        for entry_index, rows in by_entry.items():
            # _policy_actions batches through act_many where the policy offers
            # it and falls back to per-row act otherwise.
            chosen, _ = _policy_actions(policies[entry_index], rows)
            for row, action in zip(rows, chosen):
                actions[id(row)] = action
        remaining = []
        for row in active:
            row.env.step(actions[id(row)])
            if row.env.is_done():
                outcomes.extend(_row_outcomes(row))
            else:
                remaining.append(row)
        active = remaining
    return outcomes


def run_tournament(
    plan: TournamentPlan,
    policies: Sequence[ActionPolicy],
    *,
    max_concurrent_rounds: int = 8192,
    progress=None,
    snapshot=None,
) -> list[RoundOutcome]:
    """Play every match, grouped by shape so batches stay rectangular.

    ``snapshot`` is called synchronously after each completed chunk with a
    :class:`TournamentProgress` value and all outcomes accumulated so far.
    Callers can atomically persist aggregates without waiting for the complete
    tournament, and a process interruption leaves the last finished chunk
    visible rather than losing all observable progress.
    """

    if max_concurrent_rounds < 1:
        raise ValueError("max_concurrent_rounds must be positive.")
    by_shape: dict[tuple[int, int], list[Match]] = defaultdict(list)
    for match in plan.matches:
        by_shape[(match.spec.num_players, match.spec.hand_size)].append(match)

    outcomes: list[RoundOutcome] = []
    done = 0
    completed_decisions = 0
    for shape in sorted(by_shape):
        num_players, hand_size = shape
        rounds_per_match = plan.protocol.deals_per_match * num_players
        group_size = max(1, max_concurrent_rounds // rounds_per_match)
        shape_matches = by_shape[shape]
        for start in range(0, len(shape_matches), group_size):
            group = shape_matches[start : start + group_size]
            started = time.perf_counter()
            played = play_matches(group, plan, policies)
            outcomes.extend(played)
            done += len(group)
            decisions = len(played) * (1 + hand_size)
            completed_decisions += decisions
            if progress is not None:
                progress(
                    shape,
                    done,
                    len(plan.matches),
                    decisions,
                    time.perf_counter() - started,
                    len(outcomes),
                )
            if snapshot is not None:
                snapshot(
                    TournamentProgress(
                        current_shape=shape,
                        completed_matches=done,
                        total_matches=len(plan.matches),
                        completed_decisions=completed_decisions,
                        total_decisions=plan.decisions,
                        completed_round_seats=len(outcomes),
                        total_round_seats=plan.round_seats,
                    ),
                    outcomes,
                )
    return outcomes


# --- aggregation ---------------------------------------------------------


def _macro_mean(values_by_shape: dict[tuple[int, int], list[float]]) -> float:
    """Average within each shape first so no shape dominates by round count."""

    return _mean([_mean(values) for values in values_by_shape.values()])


def summarize(
    plan: TournamentPlan,
    outcomes: Sequence[RoundOutcome],
    *,
    bootstrap_samples: int | None = None,
) -> TournamentReport:
    """Aggregate all currently available outcomes.

    Live snapshots pass ``bootstrap_samples=0`` so frequent dashboard refreshes
    do not repeatedly run the final expensive resampling calculation.
    """

    ci_samples = (
        plan.protocol.bootstrap_samples
        if bootstrap_samples is None
        else bootstrap_samples
    )
    by_entry: dict[int, list[RoundOutcome]] = defaultdict(list)
    for outcome in outcomes:
        by_entry[outcome.entry].append(outcome)

    reports: list[EntryReport] = []
    for index, entry in enumerate(plan.roster):
        rows = by_entry.get(index, [])
        if not rows:
            continue
        shaped: dict[tuple[int, int], list[float]] = defaultdict(list)
        raw: dict[tuple[int, int], list[float]] = defaultdict(list)
        hit: dict[tuple[int, int], list[float]] = defaultdict(list)
        error: dict[tuple[int, int], list[float]] = defaultdict(list)
        # A deal within a match is the exchangeable unit: its n rotations are
        # the same cards reshuffled between the same players.
        blocks: dict[tuple[int, int], list[float]] = defaultdict(list)
        for row in rows:
            shape = (row.num_players, row.hand_size)
            shaped[shape].append(row.relative_reward)
            raw[shape].append(row.raw_score)
            hit[shape].append(row.bid_hit)
            error[shape].append(row.bid_abs_error)
            blocks[(row.match, row.deal)].append(row.relative_reward)
        ci_low, ci_high = _bootstrap_mean_ci(
            [_mean(values) for values in blocks.values()],
            ci_samples,
            plan.protocol.action_seed + index,
        )
        reports.append(
            EntryReport(
                key=entry.key,
                iteration=entry.iteration,
                run=entry.run,
                rounds=len(rows),
                relative_reward=_macro_mean(shaped),
                relative_reward_ci_low=ci_low,
                relative_reward_ci_high=ci_high,
                raw_score=_macro_mean(raw),
                bid_hit_rate=_macro_mean(hit),
                bid_abs_error=_macro_mean(error),
                cells={
                    f"{n}p-{h}c": _mean(values)
                    for (n, h), values in sorted(shaped.items())
                },
            )
        )
    reports.sort(key=lambda report: (report.iteration, report.key))
    return TournamentReport(
        rounds=len(outcomes),
        matches=len(plan.matches),
        rules_hash=rules_fingerprint(),
        entries=reports,
    )


def tournament_payload(
    run: RunDirectory,
    plan: TournamentPlan,
    report: TournamentReport,
    *,
    device: str,
    elapsed_sec: float,
    status: str = "complete",
    progress: TournamentProgress | dict[str, Any] | None = None,
    started_at: str | None = None,
) -> dict[str, Any]:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = {
        "format_version": TOURNAMENT_FORMAT_VERSION,
        "run": run.name,
        "device": device,
        "status": status,
        "protocol": plan.protocol.as_json(),
        "roster": [entry.as_json() for entry in plan.roster],
        "evaluated_at": now,
        "updated_at": now,
        "elapsed_sec": elapsed_sec,
        "report": dataclasses.asdict(report),
    }
    if started_at is not None:
        payload["started_at"] = started_at
    if progress is not None:
        payload["progress"] = (
            progress.as_json()
            if isinstance(progress, TournamentProgress)
            else dict(progress)
        )
    return payload


TOURNAMENT_CSV_COLUMNS = (
    "iteration",
    "run",
    "key",
    "rounds",
    "relative_reward",
    "relative_reward_ci_low",
    "relative_reward_ci_high",
    "raw_score",
    "bid_hit_rate",
    "bid_abs_error",
)


def write_tournament_csv(path: Path, report: TournamentReport) -> None:
    """Atomically replace the current aggregate table for dashboard readers."""

    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=TOURNAMENT_CSV_COLUMNS)
    writer.writeheader()
    for entry in report.entries:
        writer.writerow(
            {column: getattr(entry, column) for column in TOURNAMENT_CSV_COLUMNS}
        )
    atomic_write_text(path, handle.getvalue())
