"""Position-controlled deal banks and paired policy evaluation."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field

from plump.cards import Card, make_deck
from plump.env import PlumpEnv
from plump.policies import ActionPolicy, BatchedActionPolicy
from plump.rounds import (
    RoundSpec,
    descending_ascending_schedule,
    round_game_config,
    rules_fingerprint,
)
from plump.state import GameConfig


@dataclass(frozen=True)
class BaseDeal:
    deal_id: int
    spec: RoundSpec
    hands: tuple[tuple[Card, ...], ...]


@dataclass
class DealBank:
    deals: list[BaseDeal]
    seed: int
    rules_hash: str = field(default_factory=rules_fingerprint)

    @classmethod
    def generate(
        cls,
        *,
        player_counts: tuple[int, ...] = (3, 4, 5),
        hand_sizes: tuple[int, ...] = tuple(range(3, 11)),
        deals_per_configuration: int = 64,
        seed: int = 1,
    ) -> "DealBank":
        if deals_per_configuration < 1:
            raise ValueError("deals_per_configuration must be positive.")
        rng = random.Random(seed)
        deals: list[BaseDeal] = []
        deal_id = 0
        for num_players in player_counts:
            for hand_size in hand_sizes:
                spec = RoundSpec(num_players, hand_size)
                spec.validate()
                for _ in range(deals_per_configuration):
                    deck = make_deck()
                    rng.shuffle(deck)
                    hands = [[] for _ in range(num_players)]
                    for card_index in range(hand_size * num_players):
                        hands[card_index % num_players].append(deck[card_index])
                    deals.append(
                        BaseDeal(
                            deal_id=deal_id,
                            spec=spec,
                            hands=tuple(tuple(hand) for hand in hands),
                        )
                    )
                    deal_id += 1
        return cls(deals=deals, seed=seed)

    @property
    def specs(self) -> tuple[RoundSpec, ...]:
        return tuple(sorted({deal.spec for deal in self.deals}))


@dataclass(frozen=True)
class ScenarioResult:
    deal_id: int
    spec: RoundSpec
    focal_hand: int
    bidding_position: int
    raw_score: float
    relative_reward: float
    bid_hit: float
    bid_abs_error: float
    bid_value: float
    first_leader: float
    forward_passes: int


@dataclass
class EvaluationCell:
    num_players: int
    hand_size: int
    bidding_position: int
    rounds: int
    raw_score: float
    relative_reward: float
    bid_hit_rate: float
    bid_abs_error: float
    first_leader_rate: float
    forward_passes: float


@dataclass
class EvaluationReport:
    policy_name: str
    opponent_name: str
    rounds: int
    macro_raw_score: float
    macro_relative_reward: float
    macro_bid_hit_rate: float
    macro_bid_abs_error: float
    macro_first_leader_rate: float
    mean_forward_passes: float
    relative_reward_ci_low: float
    relative_reward_ci_high: float
    elo_delta: float
    rules_hash: str
    cells: list[EvaluationCell]
    results: list[ScenarioResult] = field(repr=False)


@dataclass
class PairedEvaluationReport:
    candidate_name: str
    baseline_name: str
    macro_relative_reward_delta: float
    ci_low: float
    ci_high: float
    worst_cell_delta: float
    cell_deltas: dict[str, float]
    cell_confidence_intervals: dict[str, tuple[float, float]]
    candidate: EvaluationReport
    baseline: EvaluationReport

    def passes_gate(self, *, max_cell_regression: float = 0.5) -> bool:
        confirmed_regression = any(
            high < -max_cell_regression
            for _, high in self.cell_confidence_intervals.values()
        )
        return self.ci_low > 0.0 and not confirmed_regression


@dataclass
class FullGameReport:
    games: int
    average_cumulative_relative_score: float
    score_std: float
    win_or_tie_rate: float
    average_final_rank: float
    schedule_relative_score: dict[str, float]
    focal_seat_relative_score: dict[str, float]
    initial_bidding_position_relative_score: dict[str, float]


@dataclass
class PairedFullGameReport:
    games: int
    average_relative_score_delta: float
    ci_low: float
    ci_high: float
    candidate: FullGameReport
    baseline: FullGameReport


@dataclass
class CompatibilityReport:
    completed: bool
    rounds: int
    actions: int
    schema_context_enabled: bool | None


@dataclass
class _ActiveScenario:
    deal: BaseDeal
    focal_hand: int
    bidding_position: int
    env: PlumpEnv
    rng: random.Random
    candidate_forward_passes: int = 0


def check_round_policy_compatibility(
    candidate: ActionPolicy,
    opponent: ActionPolicy,
    *,
    num_players: int = 4,
    hand_sizes: tuple[int, ...] = (3, 4, 3),
    seed: int = 71,
) -> CompatibilityReport:
    """Verify that a round policy executes across round boundaries without scoring it."""

    env = PlumpEnv(GameConfig(num_players=num_players, hand_sizes=list(hand_sizes)), seed=seed)
    env.reset()
    rng = random.Random(seed + 1)
    actions = 0
    while not env.is_done():
        policy = candidate if env.current_player() == 0 else opponent
        env.step(policy.act(env, rng=rng))
        actions += 1
    return CompatibilityReport(
        completed=True,
        rounds=len(env.state.rounds),
        actions=actions,
        schema_context_enabled=getattr(candidate, "include_game_context", None),
    )


def evaluate_policy(
    candidate: ActionPolicy,
    opponent: ActionPolicy,
    deal_bank: DealBank,
    *,
    bootstrap_samples: int = 2_000,
    seed: int = 17,
    batch_size: int = 128,
) -> EvaluationReport:
    """Evaluate one focal policy over every hand and bidding-position rotation."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    scenarios = [
        (deal, focal_hand, bidding_position)
        for deal in deal_bank.deals
        for focal_hand in range(deal.spec.num_players)
        for bidding_position in range(deal.spec.num_players)
    ]
    results: list[ScenarioResult] = []
    candidate.reset_counters()
    opponent.reset_counters()
    for start in range(0, len(scenarios), batch_size):
        active = [
            _new_active_scenario(deal, focal_hand, bidding_position, seed)
            for deal, focal_hand, bidding_position in scenarios[start : start + batch_size]
        ]
        while active:
            candidate_rows = [
                row for row in active if row.env.current_player() == 0
            ]
            opponent_rows = [
                row for row in active if row.env.current_player() != 0
            ]
            actions: dict[int, object] = {}
            candidate_actions, candidate_costs = _policy_actions(candidate, candidate_rows)
            for row, action, cost in zip(
                candidate_rows,
                candidate_actions,
                candidate_costs,
            ):
                actions[id(row)] = action
                row.candidate_forward_passes += cost
            opponent_actions, _ = _policy_actions(opponent, opponent_rows)
            for row, action in zip(opponent_rows, opponent_actions):
                actions[id(row)] = action

            next_active = []
            for row in active:
                row.env.step(actions[id(row)])
                if row.env.is_done():
                    results.append(_scenario_result(row))
                else:
                    next_active.append(row)
            active = next_active

    cells = _evaluation_cells(results)
    deal_means = _means_by_deal(results, "relative_reward")
    ci_low, ci_high = _bootstrap_mean_ci(list(deal_means.values()), bootstrap_samples, seed)
    positive_rate = _mean(
        [1.0 if result.relative_reward > 0 else 0.5 if result.relative_reward == 0 else 0.0 for result in results]
    )
    return EvaluationReport(
        policy_name=candidate.name,
        opponent_name=opponent.name,
        rounds=len(results),
        macro_raw_score=_uniform_configuration_cell_mean(cells, "raw_score"),
        macro_relative_reward=_uniform_configuration_cell_mean(
            cells,
            "relative_reward",
        ),
        macro_bid_hit_rate=_uniform_configuration_cell_mean(
            cells,
            "bid_hit_rate",
        ),
        macro_bid_abs_error=_uniform_configuration_cell_mean(
            cells,
            "bid_abs_error",
        ),
        macro_first_leader_rate=_uniform_configuration_cell_mean(
            cells,
            "first_leader_rate",
        ),
        mean_forward_passes=_uniform_configuration_cell_mean(
            cells,
            "forward_passes",
        ),
        relative_reward_ci_low=ci_low,
        relative_reward_ci_high=ci_high,
        elo_delta=_elo_delta(positive_rate),
        rules_hash=deal_bank.rules_hash,
        cells=cells,
        results=results,
    )


def evaluate_paired(
    candidate: ActionPolicy,
    baseline: ActionPolicy,
    opponent: ActionPolicy,
    deal_bank: DealBank,
    *,
    bootstrap_samples: int = 2_000,
    seed: int = 17,
    batch_size: int = 128,
) -> PairedEvaluationReport:
    candidate_report = evaluate_policy(
        candidate,
        opponent,
        deal_bank,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        batch_size=batch_size,
    )
    baseline_report = evaluate_policy(
        baseline,
        opponent,
        deal_bank,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        batch_size=batch_size,
    )
    candidate_by_key = {_result_key(result): result for result in candidate_report.results}
    baseline_by_key = {_result_key(result): result for result in baseline_report.results}
    if candidate_by_key.keys() != baseline_by_key.keys():
        raise RuntimeError("Paired evaluation produced different scenario sets.")

    deltas = {
        key: candidate_by_key[key].relative_reward - baseline_by_key[key].relative_reward
        for key in candidate_by_key
    }
    by_deal: dict[int, list[float]] = defaultdict(list)
    by_cell: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    by_cell_deal: dict[tuple[int, int, int], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for key, delta in deltas.items():
        result = candidate_by_key[key]
        by_deal[result.deal_id].append(delta)
        cell = (result.spec.num_players, result.spec.hand_size, result.bidding_position)
        by_cell[cell].append(delta)
        by_cell_deal[cell][result.deal_id].append(delta)

    deal_deltas = [_mean(values) for values in by_deal.values()]
    ci_low, ci_high = _bootstrap_mean_ci(deal_deltas, bootstrap_samples, seed + 1)
    cell_deltas = {
        f"{n}p-{hand}c-pos{position}": _mean(values)
        for (n, hand, position), values in sorted(by_cell.items())
    }
    cell_confidence_intervals = {
        f"{n}p-{hand}c-pos{position}": _bootstrap_mean_ci(
            [
                _mean(deal_values)
                for deal_values in by_cell_deal[(n, hand, position)].values()
            ],
            bootstrap_samples,
            seed + 10_000 + n * 1_000 + hand * 10 + position,
        )
        for (n, hand, position), values in sorted(by_cell.items())
    }
    return PairedEvaluationReport(
        candidate_name=candidate.name,
        baseline_name=baseline.name,
        macro_relative_reward_delta=_uniform_configuration_delta(cell_deltas),
        ci_low=ci_low,
        ci_high=ci_high,
        worst_cell_delta=min(cell_deltas.values()) if cell_deltas else 0.0,
        cell_deltas=cell_deltas,
        cell_confidence_intervals=cell_confidence_intervals,
        candidate=candidate_report,
        baseline=baseline_report,
    )


def evaluate_full_games(
    candidate: ActionPolicy,
    opponent: ActionPolicy,
    *,
    num_players: int,
    games: int = 32,
    hand_sizes: tuple[int, ...] = tuple(descending_ascending_schedule()),
    seed: int = 91,
) -> FullGameReport:
    scores: list[float] = []
    win_or_ties: list[float] = []
    ranks: list[float] = []
    schedule_scores: dict[int, list[float]] = defaultdict(list)
    seat_scores: dict[int, list[float]] = defaultdict(list)
    position_scores: dict[int, list[float]] = defaultdict(list)
    for game_index in range(games):
        focal_player = game_index % num_players
        initial_bidding_position = (
            game_index // num_players
        ) % num_players
        initial_start = (
            focal_player - initial_bidding_position
        ) % num_players
        bidding_starts = [
            (initial_start + round_index) % num_players
            for round_index in range(len(hand_sizes))
        ]
        env = PlumpEnv(
            GameConfig(
                num_players=num_players,
                hand_sizes=list(hand_sizes),
                bidding_start_players=bidding_starts,
            ),
            seed=seed + game_index,
        )
        env.reset()
        rng = random.Random(seed * 10_000 + game_index)
        while not env.is_done():
            policy = (
                candidate
                if env.current_player() == focal_player
                else opponent
            )
            env.step(policy.act(env, rng=rng))
        final_scores = env.state.cumulative_scores
        relative_score = _relative_rewards(final_scores)[focal_player]
        scores.append(relative_score)
        seat_scores[focal_player].append(relative_score)
        position_scores[initial_bidding_position].append(relative_score)
        focal_score = final_scores[focal_player]
        win_or_ties.append(float(focal_score == max(final_scores.values())))
        ranks.append(
            1.0 + sum(score > focal_score for score in final_scores.values())
        )
        for round_state in env.state.rounds:
            schedule_scores[round_state.round_index].append(
                _relative_rewards(round_state.round_scores)[focal_player]
            )
    return FullGameReport(
        games=games,
        average_cumulative_relative_score=_mean(scores),
        score_std=_std(scores),
        win_or_tie_rate=_mean(win_or_ties),
        average_final_rank=_mean(ranks),
        schedule_relative_score={
            str(index): _mean(values)
            for index, values in sorted(schedule_scores.items())
        },
        focal_seat_relative_score={
            str(index): _mean(values)
            for index, values in sorted(seat_scores.items())
        },
        initial_bidding_position_relative_score={
            str(index): _mean(values)
            for index, values in sorted(position_scores.items())
        },
    )


def evaluate_full_games_paired(
    candidate: ActionPolicy,
    baseline: ActionPolicy,
    opponent: ActionPolicy,
    *,
    num_players: int,
    games: int = 32,
    hand_sizes: tuple[int, ...] = tuple(descending_ascending_schedule()),
    seed: int = 91,
    bootstrap_samples: int = 2_000,
) -> PairedFullGameReport:
    candidate_scores = _full_game_scores(
        candidate,
        opponent,
        num_players=num_players,
        games=games,
        hand_sizes=hand_sizes,
        seed=seed,
    )
    baseline_scores = _full_game_scores(
        baseline,
        opponent,
        num_players=num_players,
        games=games,
        hand_sizes=hand_sizes,
        seed=seed,
    )
    deltas = [
        candidate_score - baseline_score
        for candidate_score, baseline_score in zip(
            candidate_scores,
            baseline_scores,
        )
    ]
    low, high = _bootstrap_mean_ci(deltas, bootstrap_samples, seed + 7_000)
    return PairedFullGameReport(
        games=games,
        average_relative_score_delta=_mean(deltas),
        ci_low=low,
        ci_high=high,
        candidate=evaluate_full_games(
            candidate,
            opponent,
            num_players=num_players,
            games=games,
            hand_sizes=hand_sizes,
            seed=seed,
        ),
        baseline=evaluate_full_games(
            baseline,
            opponent,
            num_players=num_players,
            games=games,
            hand_sizes=hand_sizes,
            seed=seed,
        ),
    )


def _new_active_scenario(
    deal: BaseDeal,
    focal_hand: int,
    bidding_position: int,
    seed: int,
) -> _ActiveScenario:
    n = deal.spec.num_players
    manual_hands = {
        player: list(deal.hands[(focal_hand + player) % n])
        for player in range(n)
    }
    env = PlumpEnv(
        round_game_config(
            deal.spec,
            bidding_start_player=(-bidding_position) % n,
            manual_hands=manual_hands,
        )
    )
    env.reset()
    return _ActiveScenario(
        deal=deal,
        focal_hand=focal_hand,
        bidding_position=bidding_position,
        env=env,
        rng=random.Random(
            _scenario_seed(seed, deal.deal_id, focal_hand, bidding_position)
        ),
    )


def _policy_actions(
    policy: ActionPolicy,
    rows: list[_ActiveScenario],
) -> tuple[list[object], list[int]]:
    if not rows:
        return [], []
    if isinstance(policy, BatchedActionPolicy):
        return (
            policy.act_many(
                [row.env for row in rows],
                rngs=[row.rng for row in rows],
            ),
            [1] * len(rows),
        )

    actions = []
    costs = []
    for row in rows:
        before = policy.forward_passes
        actions.append(policy.act(row.env, rng=row.rng))
        costs.append(policy.forward_passes - before)
    return actions, costs


def _scenario_result(row: _ActiveScenario) -> ScenarioResult:
    round_state = row.env.state.current_round
    relative = _relative_rewards(round_state.round_scores)
    bid = next(item.value for item in round_state.bids if item.player == 0)
    return ScenarioResult(
        deal_id=row.deal.deal_id,
        spec=row.deal.spec,
        focal_hand=row.focal_hand,
        bidding_position=row.bidding_position,
        raw_score=float(round_state.round_scores[0]),
        relative_reward=relative[0],
        bid_hit=float(round_state.tricks_won[0] == bid),
        bid_abs_error=float(abs(round_state.tricks_won[0] - bid)),
        bid_value=float(bid),
        first_leader=float(round_state.play_start_player == 0),
        forward_passes=row.candidate_forward_passes,
    )


def _evaluation_cells(results: list[ScenarioResult]) -> list[EvaluationCell]:
    grouped: dict[tuple[int, int, int], list[ScenarioResult]] = defaultdict(list)
    for result in results:
        grouped[(result.spec.num_players, result.spec.hand_size, result.bidding_position)].append(result)
    cells = []
    for (num_players, hand_size, bidding_position), rows in sorted(grouped.items()):
        cells.append(
            EvaluationCell(
                num_players=num_players,
                hand_size=hand_size,
                bidding_position=bidding_position,
                rounds=len(rows),
                raw_score=_mean([row.raw_score for row in rows]),
                relative_reward=_mean([row.relative_reward for row in rows]),
                bid_hit_rate=_mean([row.bid_hit for row in rows]),
                bid_abs_error=_mean([row.bid_abs_error for row in rows]),
                first_leader_rate=_mean([row.first_leader for row in rows]),
                forward_passes=_mean([float(row.forward_passes) for row in rows]),
            )
        )
    return cells


def _uniform_configuration_cell_mean(
    cells: list[EvaluationCell],
    field_name: str,
) -> float:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for cell in cells:
        grouped[(cell.num_players, cell.hand_size)].append(
            float(getattr(cell, field_name))
        )
    return _mean([_mean(values) for values in grouped.values()])


def _uniform_configuration_delta(cell_deltas: dict[str, float]) -> float:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for key, value in cell_deltas.items():
        player_part, hand_part, _ = key.split("-")
        grouped[(int(player_part[:-1]), int(hand_part[:-1]))].append(value)
    return _mean([_mean(values) for values in grouped.values()])


def _full_game_scores(
    candidate: ActionPolicy,
    opponent: ActionPolicy,
    *,
    num_players: int,
    games: int,
    hand_sizes: tuple[int, ...],
    seed: int,
) -> list[float]:
    scores = []
    for game_index in range(games):
        focal_player = game_index % num_players
        initial_bidding_position = (
            game_index // num_players
        ) % num_players
        initial_start = (
            focal_player - initial_bidding_position
        ) % num_players
        env = PlumpEnv(
            GameConfig(
                num_players=num_players,
                hand_sizes=list(hand_sizes),
                bidding_start_players=[
                    (initial_start + round_index) % num_players
                    for round_index in range(len(hand_sizes))
                ],
            ),
            seed=seed + game_index,
        )
        env.reset()
        rng = random.Random(seed * 10_000 + game_index)
        while not env.is_done():
            policy = (
                candidate
                if env.current_player() == focal_player
                else opponent
            )
            env.step(policy.act(env, rng=rng))
        scores.append(
            _relative_rewards(env.state.cumulative_scores)[focal_player]
        )
    return scores


def _relative_rewards(scores: dict[int, int]) -> dict[int, float]:
    total = sum(scores.values())
    opponents = len(scores) - 1
    return {
        player: float(score - ((total - score) / opponents))
        for player, score in scores.items()
    }


def _means_by_deal(results: list[ScenarioResult], field_name: str) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for result in results:
        grouped[result.deal_id].append(float(getattr(result, field_name)))
    return {deal_id: _mean(values) for deal_id, values in grouped.items()}


def _bootstrap_mean_ci(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if samples <= 0:
        mean = _mean(values)
        return mean, mean
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        estimates.append(_mean([rng.choice(values) for _ in values]))
    estimates.sort()
    low = estimates[max(int(0.025 * samples) - 1, 0)]
    high = estimates[min(int(0.975 * samples), samples - 1)]
    return low, high


def _result_key(result: ScenarioResult) -> tuple[int, int, int]:
    return result.deal_id, result.focal_hand, result.bidding_position


def _scenario_seed(seed: int, deal_id: int, focal_hand: int, position: int) -> int:
    return seed * 1_000_003 + deal_id * 97 + focal_hand * 11 + position


def _elo_delta(score: float) -> float:
    score = min(max(score, 1e-6), 1.0 - 1e-6)
    return 400.0 * math.log10(score / (1.0 - score))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return math.sqrt(_mean([(value - mean) ** 2 for value in values]))
