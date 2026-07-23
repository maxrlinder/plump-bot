"""Constrained determinizations and root-only information-set lookahead."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from plump.cards import Card, Suit, make_deck
from plump.env import PlumpEnv
from plump.modeling import ModelConfig, card_from_id, card_id
from plump.modeling.encoding import NUM_CARDS
from plump.policies import ActionPolicy, HeuristicPolicy, ModelPolicy, RandomPolicy
from plump.state import (
    BidAction,
    EventType,
    GameConfig,
    Observation,
    Phase,
    PlayCardAction,
    ScoringConfig,
    TrumpPolicy,
)


@dataclass(frozen=True)
class Determinization:
    current_hands: dict[int, tuple[Card, ...]]
    undealt_cards: tuple[Card, ...]


@dataclass(frozen=True)
class SearchConfig:
    min_determinizations: int = 4
    max_determinizations: int = 32
    batch_determinizations: int = 4
    forward_pass_budget: int = 2_000
    close_action_stderr: float = 1.5
    regret_temperature: float = 2.0
    maximum_target_js: float = 0.05
    exact_tricks_remaining: int = 3
    exact_node_budget: int = 4_096
    use_owner_beliefs: bool = True
    interactive_wall_clock_seconds: float | None = None
    seed: int = 1


@dataclass
class SearchDecision:
    action: BidAction | PlayCardAction
    action_values: dict[str, float]
    action_probabilities: dict[str, float]
    action_regrets: dict[str, float]
    action_stderr: dict[str, float]
    prior_probabilities: dict[str, float]
    split_half_argmax_agreement: bool
    target_js_divergence: float
    accepted: bool
    samples_per_action: int
    forward_passes: int


class ConstrainedDeterminizationSampler:
    """Sample exact opponent hands while respecting all public constraints."""

    def __init__(self) -> None:
        self.draws_attempted = 0
        self.draws_succeeded = 0
        self.draws_failed = 0
        self.candidate_attempts = 0
        self.infeasible_candidate_rejections = 0

    def counters(self) -> tuple[int, int, int, int, int]:
        return (
            self.draws_attempted,
            self.draws_succeeded,
            self.draws_failed,
            self.candidate_attempts,
            self.infeasible_candidate_rejections,
        )

    def sample(
        self,
        observation: Observation,
        *,
        owner_probs: list[list[float]] | None,
        model_config: ModelConfig,
        rng: random.Random,
    ) -> Determinization:
        self.draws_attempted += 1
        num_players = len(observation.scores)
        own_cards = set(observation.my_hand)
        played_cards = set(observation.played_cards_total)
        available = set(make_deck()) - own_cards - played_cards
        capacities = {
            player: observation.hand_size - len(observation.played_cards_by_player.get(player, []))
            for player in range(num_players)
            if player != observation.player_id
        }
        legal_cards = {
            player: {
                card
                for card in available
                if not observation.voids.get(player, {}).get(card.suit, False)
            }
            for player in capacities
        }
        if sum(capacities.values()) > len(available):
            self.draws_failed += 1
            raise ValueError("Public state requires more opponent cards than remain unseen.")
        if not _can_complete(capacities, available, legal_cards):
            self.draws_failed += 1
            raise ValueError("Public void constraints admit no legal hidden-card assignment.")

        assignments = {player: [] for player in capacities}
        remaining = set(available)
        remaining_capacities = dict(capacities)
        while any(value > 0 for value in remaining_capacities.values()):
            candidates_by_player = {
                player: legal_cards[player] & remaining
                for player, capacity in remaining_capacities.items()
                if capacity > 0
            }
            player = min(
                candidates_by_player,
                key=lambda item: (
                    len(candidates_by_player[item]) / remaining_capacities[item],
                    item,
                ),
            )
            candidates = list(candidates_by_player[player])
            if not candidates:
                raise ValueError("A public-consistent opponent hand could not be completed.")
            relative = (player - observation.player_id) % num_players
            owner_class = relative - 1
            ordered = _weighted_order(
                candidates,
                [
                    _owner_weight(owner_probs, card, owner_class)
                    for card in candidates
                ],
                rng,
            )
            selected = None
            for card in ordered:
                self.candidate_attempts += 1
                next_remaining = remaining - {card}
                next_capacities = dict(remaining_capacities)
                next_capacities[player] -= 1
                if _can_complete(next_capacities, next_remaining, legal_cards):
                    selected = card
                    remaining = next_remaining
                    remaining_capacities = next_capacities
                    assignments[player].append(card)
                    break
                self.infeasible_candidate_rejections += 1
            if selected is None:
                self.draws_failed += 1
                raise ValueError("Weighted assignment exhausted every feasible card.")

        expected_kitty = NUM_CARDS - observation.hand_size * num_players
        if len(remaining) != expected_kitty:
            self.draws_failed += 1
            raise AssertionError(
                f"Determinization kitty has {len(remaining)} cards; expected {expected_kitty}."
            )
        self.draws_succeeded += 1
        return Determinization(
            current_hands={
                player: tuple(sorted(cards, key=card_id))
                for player, cards in assignments.items()
            },
            undealt_cards=tuple(sorted(remaining, key=card_id)),
        )


class RootSearchPolicy:
    """Root-action Monte Carlo lookahead with observation-only continuations."""

    def __init__(
        self,
        base_policy: ActionPolicy,
        opponent_policy: ActionPolicy,
        *,
        config: SearchConfig | None = None,
        scoring: ScoringConfig | None = None,
        name: str | None = None,
    ) -> None:
        self.base_policy = base_policy
        self.opponent_policy = opponent_policy
        self.config = config or SearchConfig()
        self.scoring = scoring or ScoringConfig()
        self.sampler = ConstrainedDeterminizationSampler()
        self.name = name or f"search:{base_policy.name}"
        self.forward_passes = 0
        self.last_decision: SearchDecision | None = None

    def reset_counters(self) -> None:
        self.forward_passes = 0
        self.base_policy.reset_counters()
        self.opponent_policy.reset_counters()

    def act(self, env: PlumpEnv, *, rng: random.Random | None = None) -> BidAction | PlayCardAction:
        legal_actions = env.legal_actions()
        if len(legal_actions) == 1:
            key = _action_key(legal_actions[0])
            self.last_decision = SearchDecision(
                legal_actions[0],
                {key: 0.0},
                {key: 1.0},
                {key: 0.0},
                {key: 0.0},
                {key: 1.0},
                True,
                0.0,
                True,
                0,
                0,
            )
            return legal_actions[0]
        rng = rng or random.Random(self.config.seed)
        decision = self.search(
            env.get_observation(env.current_player()),
            legal_actions=legal_actions,
            rng=rng,
        )
        self.last_decision = decision
        self.forward_passes += decision.forward_passes
        return decision.action

    def search(
        self,
        observation: Observation,
        *,
        legal_actions: list[BidAction | PlayCardAction],
        rng: random.Random,
    ) -> SearchDecision:
        root_prediction = None
        if isinstance(self.base_policy, ModelPolicy):
            encoded, output = self.base_policy.predict_observation(
                observation,
                need_owner=self.config.use_owner_beliefs,
            )
            root_prediction = self._root_prediction(
                observation,
                legal_actions,
                encoded.observer_player,
                (
                    output.owner_probs[0].detach().cpu().tolist()
                    if output.owner_probs is not None
                    else None
                ),
                (
                    output.masked_bid_logits[0].float()
                    if observation.phase == Phase.BIDDING
                    else output.masked_card_logits[0].float()
                ).softmax(dim=-1).detach().cpu().tolist(),
            )
        return self._search(
            observation,
            legal_actions=legal_actions,
            rng=rng,
            root_prediction=root_prediction,
            root_forward_passes=int(root_prediction is not None),
        )

    def search_many(
        self,
        requests: list[
            tuple[
                Observation,
                list[BidAction | PlayCardAction],
                random.Random,
            ]
        ],
    ) -> list[SearchDecision]:
        """Batch root model inference while preserving serial search semantics."""

        if not requests:
            return []
        if not isinstance(self.base_policy, ModelPolicy):
            return [
                self.search(
                    observation,
                    legal_actions=legal_actions,
                    rng=rng,
                )
                for observation, legal_actions, rng in requests
            ]
        observations = [request[0] for request in requests]
        encoded, output = self.base_policy.predict_observations(
            observations,
            need_owner=self.config.use_owner_beliefs,
        )
        predictions = []
        for index, (observation, legal_actions, _) in enumerate(requests):
            logits = (
                output.masked_bid_logits[index].float()
                if observation.phase == Phase.BIDDING
                else output.masked_card_logits[index].float()
            )
            predictions.append(
                self._root_prediction(
                    observation,
                    legal_actions,
                    encoded[index].observer_player,
                    (
                        output.owner_probs[index].detach().cpu().tolist()
                        if output.owner_probs is not None
                        else None
                    ),
                    logits.softmax(dim=-1).detach().cpu().tolist(),
                )
            )
        return [
            self._search(
                observation,
                legal_actions=legal_actions,
                rng=rng,
                root_prediction=prediction,
                root_forward_passes=1,
            )
            for (observation, legal_actions, rng), prediction in zip(
                requests,
                predictions,
            )
        ]

    def _search(
        self,
        observation: Observation,
        *,
        legal_actions: list[BidAction | PlayCardAction],
        rng: random.Random,
        root_prediction: tuple[
            list[list[float]] | None,
            dict[BidAction | PlayCardAction, float],
            ModelConfig,
        ]
        | None,
        root_forward_passes: int,
    ) -> SearchDecision:
        root_player = observation.player_id
        started_at = time.monotonic()
        owner_probs = None
        prior = {action: 1.0 / len(legal_actions) for action in legal_actions}
        model_config = ModelConfig()
        before_forwards = self._policy_forward_passes()

        if root_prediction is not None:
            owner_probs, prior, model_config = root_prediction

        entropy = _mean_owner_entropy(owner_probs, observation, model_config)
        breadths = []
        breadth = self.config.min_determinizations
        while breadth < self.config.max_determinizations:
            breadths.append(breadth)
            breadth = min(
                self.config.max_determinizations,
                max(
                    breadth * 2,
                    breadth + self.config.batch_determinizations,
                ),
            )
        breadths.append(self.config.max_determinizations)
        breadth_index = min(
            round(entropy * (len(breadths) - 1)),
            len(breadths) - 1,
        )
        adaptive_max = breadths[breadth_index]

        values = {action: [] for action in legal_actions}
        determinizations: list[Determinization] = []
        rollout_seeds: list[int] = []
        target = self.config.min_determinizations
        while len(determinizations) < adaptive_max:
            while len(determinizations) < target:
                determinizations.append(
                    self.sampler.sample(
                        observation,
                        owner_probs=owner_probs,
                        model_config=model_config,
                        rng=rng,
                    )
                )
                rollout_seeds.append(rng.randrange(2**31))

            completed = min(len(values[next(iter(values))]), target)
            continuations: list[tuple[BidAction | PlayCardAction, PlumpEnv, int]] = []
            for index in range(completed, target):
                determinization = determinizations[index]
                seed = rollout_seeds[index]
                for action in legal_actions:
                    rollout_env = reconstruct_public_state(
                        observation,
                        determinization,
                        scoring=self.scoring,
                    )
                    rollout_env.step(action)
                    continuations.append((action, rollout_env, seed))
            if (
                continuations
                and self.base_policy is self.opponent_policy
                and isinstance(self.base_policy, ModelPolicy)
            ):
                rewards = self._evaluate_self_play_continuations(
                    [row[1] for row in continuations],
                    root_player=root_player,
                    seeds=[row[2] for row in continuations],
                )
                for (action, _, _), reward in zip(continuations, rewards):
                    values[action].append(reward)
            else:
                for action, rollout_env, seed in continuations:
                    values[action].append(
                        self._evaluate_continuation(
                            rollout_env,
                            root_player=root_player,
                            seed=seed,
                        )
                    )

            used = self._policy_forward_passes() - before_forwards
            if target >= adaptive_max or used >= self.config.forward_pass_budget:
                break
            if (
                self.config.interactive_wall_clock_seconds is not None
                and time.monotonic() - started_at
                >= self.config.interactive_wall_clock_seconds
            ):
                break
            if not _actions_are_close(values, self.config.close_action_stderr):
                break
            target = min(
                max(
                    target * 2,
                    target + self.config.batch_determinizations,
                ),
                adaptive_max,
            )

        means = {action: sum(samples) / len(samples) for action, samples in values.items()}
        baseline = sum(prior[action] * means[action] for action in legal_actions)
        regrets = {
            action: means[action] - baseline
            for action in legal_actions
        }
        target_probabilities = _policy_anchored_target(
            prior,
            regrets,
            temperature=self.config.regret_temperature,
        )
        agreement, target_js = _split_half_stability(
            values,
            prior,
            temperature=self.config.regret_temperature,
        )
        best_action = max(
            legal_actions,
            key=lambda action: (means[action], -legal_actions.index(action)),
        )
        used = root_forward_passes + (
            self._policy_forward_passes() - before_forwards
        )
        return SearchDecision(
            action=best_action,
            action_values={_action_key(action): value for action, value in means.items()},
            action_probabilities={
                _action_key(action): probability
                for action, probability in target_probabilities.items()
            },
            action_regrets={
                _action_key(action): value
                for action, value in regrets.items()
            },
            action_stderr={
                _action_key(action): _stderr(values[action])
                for action in legal_actions
            },
            prior_probabilities={
                _action_key(action): probability
                for action, probability in prior.items()
            },
            split_half_argmax_agreement=agreement,
            target_js_divergence=target_js,
            accepted=(
                agreement
                and target_js <= self.config.maximum_target_js
            ),
            samples_per_action=len(values[best_action]),
            forward_passes=used,
        )

    def _root_prediction(
        self,
        observation: Observation,
        legal_actions: list[BidAction | PlayCardAction],
        observer_player: int,
        owner_probs: list[list[float]] | None,
        probabilities: list[float],
    ) -> tuple[
        list[list[float]] | None,
        dict[BidAction | PlayCardAction, float],
        ModelConfig,
    ]:
        if observer_player != observation.player_id:
            raise AssertionError("Root belief prediction used the wrong observer.")
        prior = {
            action: probabilities[
                action.bid
                if isinstance(action, BidAction)
                else card_id(action.card)
            ]
            for action in legal_actions
        }
        return owner_probs, prior, self.base_policy.model_config

    def _policy_forward_passes(self) -> int:
        """Count each distinct policy wrapper once when roles share it."""

        policies = {id(self.base_policy): self.base_policy}
        policies[id(self.opponent_policy)] = self.opponent_policy
        return sum(policy.forward_passes for policy in policies.values())

    def _evaluate_self_play_continuations(
        self,
        envs: list[PlumpEnv],
        *,
        root_player: int,
        seeds: list[int],
    ) -> list[float]:
        """Roll out same-policy continuations in batched decision waves.

        All candidate root actions use matched determinization and RNG seeds.
        Batching the active environments preserves that common-random-number
        comparison while avoiding thousands of batch-size-one MPS forwards.
        """

        if not isinstance(self.base_policy, ModelPolicy):
            raise TypeError("Batched self-play search requires ModelPolicy.")
        if len(envs) != len(seeds):
            raise ValueError("envs and seeds must have equal length.")
        rngs = [random.Random(seed) for seed in seeds]
        pending = list(range(len(envs)))
        while pending:
            active = [index for index in pending if not envs[index].is_done()]
            if not active:
                break
            actions = self.base_policy.act_many(
                [envs[index] for index in active],
                rngs=[rngs[index] for index in active],
            )
            for index, action in zip(active, actions):
                envs[index].step(action)
            pending = active
        return [
            _relative_rewards(env.state.current_round.round_scores)[root_player]
            for env in envs
        ]

    def _evaluate_continuation(
        self,
        env: PlumpEnv,
        *,
        root_player: int,
        seed: int,
    ) -> float:
        remaining = (
            env.state.current_round.hand_size
            - len(
                [
                    trick
                    for trick in env.state.current_round.tricks
                    if trick.winner is not None
                ]
            )
        )
        if remaining <= self.config.exact_tricks_remaining:
            try:
                return _exact_policy_expectation(
                    env,
                    root_player=root_player,
                    base_policy=self.base_policy,
                    opponent_policy=self.opponent_policy,
                    node_budget=self.config.exact_node_budget,
                )
            except _NodeBudgetExceeded:
                pass

        rollout_env = env.clone()
        rollout_rng = random.Random(seed)
        while not rollout_env.is_done():
            policy = (
                self.base_policy
                if rollout_env.current_player() == root_player
                else self.opponent_policy
            )
            rollout_env.step(policy.act(rollout_env, rng=rollout_rng))
        return _relative_rewards(
            rollout_env.state.current_round.round_scores
        )[root_player]


def reconstruct_public_state(
    observation: Observation,
    determinization: Determinization,
    *,
    scoring: ScoringConfig | None = None,
) -> PlumpEnv:
    """Rebuild a sampled state solely from public history and sampled hidden cards."""

    num_players = len(observation.scores)
    played = observation.played_cards_by_player
    initial_hands: dict[int, list[Card]] = {}
    for player in range(num_players):
        if player == observation.player_id:
            current = list(observation.my_hand)
        else:
            current = list(determinization.current_hands[player])
        initial_hands[player] = current + list(played.get(player, []))
        if len(initial_hands[player]) != observation.hand_size:
            raise ValueError("Reconstructed initial hand has the wrong size.")

    config = GameConfig(
        num_players=num_players,
        hand_sizes=[observation.hand_size],
        forbid_total_bid_equals_hand_size=True,
        scoring=scoring or ScoringConfig(),
        trump_policy=TrumpPolicy.NONE,
        manual_hands=initial_hands,
        manual_trump_suit=observation.trump_suit,
        bidding_start_players=[observation.bidding_start_player],
    )
    env = PlumpEnv(config)
    env.reset()
    for event in observation.event_log:
        if event.round_index != observation.round_index:
            continue
        if event.type == EventType.BID:
            if event.player is None or event.bid is None:
                raise ValueError("Malformed public bid event.")
            env.step(BidAction(event.player, event.bid))
        elif event.type == EventType.PLAY:
            if event.player is None or event.card is None:
                raise ValueError("Malformed public play event.")
            env.step(PlayCardAction(event.player, event.card))

    if env.phase() != observation.phase or env.state.current_player != observation.current_player:
        raise AssertionError("Reconstructed state does not match the public decision point.")
    return env


def _can_complete(
    capacities: dict[int, int],
    available: set[Card],
    legal_cards: dict[int, set[Card]],
) -> bool:
    slots = [
        player
        for player, capacity in capacities.items()
        for _ in range(max(capacity, 0))
    ]
    if len(slots) > len(available):
        return False
    slots.sort(key=lambda player: len(legal_cards[player] & available))
    matched_card_to_slot: dict[Card, int] = {}

    def assign(slot_index: int, seen: set[Card]) -> bool:
        player = slots[slot_index]
        for card in legal_cards[player] & available:
            if card in seen:
                continue
            seen.add(card)
            previous = matched_card_to_slot.get(card)
            if previous is None or assign(previous, seen):
                matched_card_to_slot[card] = slot_index
                return True
        return False

    for slot_index in range(len(slots)):
        if not assign(slot_index, set()):
            return False
    return True


def _weighted_order(
    candidates: list[Card],
    weights: list[float],
    rng: random.Random,
) -> list[Card]:
    keyed = []
    for card, weight in zip(candidates, weights):
        positive = max(float(weight), 1e-6)
        keyed.append((-math.log(max(rng.random(), 1e-12)) / positive, card))
    keyed.sort(key=lambda item: item[0])
    return [card for _, card in keyed]


def _owner_weight(
    owner_probs: list[list[float]] | None,
    card: Card,
    owner_class: int,
) -> float:
    if owner_probs is None:
        return 1.0
    return owner_probs[card_id(card)][owner_class]


def _mean_owner_entropy(
    owner_probs: list[list[float]] | None,
    observation: Observation,
    model_config: ModelConfig,
) -> float:
    if owner_probs is None:
        return 1.0
    entropies = []
    for index, probabilities in enumerate(owner_probs):
        card = card_from_id(index)
        if card in observation.my_hand or card in observation.played_cards_total:
            continue
        active = [probability for probability in probabilities if probability > 0.0]
        if len(active) <= 1:
            continue
        entropy = -sum(value * math.log(max(value, 1e-12)) for value in active)
        entropies.append(entropy / math.log(len(active)))
    return sum(entropies) / len(entropies) if entropies else 0.0


def _actions_are_close(
    values: dict[BidAction | PlayCardAction, list[float]],
    multiplier: float,
) -> bool:
    ranked = sorted(
        (
            (sum(samples) / len(samples), _stderr(samples))
            for samples in values.values()
            if samples
        ),
        reverse=True,
    )
    if len(ranked) < 2:
        return False
    return ranked[0][0] - ranked[1][0] <= multiplier * (ranked[0][1] + ranked[1][1])


def _stderr(values: list[float]) -> float:
    if len(values) < 2:
        return float("inf")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def _action_key(action: BidAction | PlayCardAction) -> str:
    if isinstance(action, BidAction):
        return f"bid:{action.bid}"
    return f"card:{card_id(action.card)}"


def _relative_rewards(scores: dict[int, int]) -> dict[int, float]:
    total = sum(scores.values())
    opponents = len(scores) - 1
    return {
        player: float(score - ((total - score) / opponents))
        for player, score in scores.items()
    }


class _NodeBudgetExceeded(RuntimeError):
    pass


def _exact_policy_expectation(
    env: PlumpEnv,
    *,
    root_player: int,
    base_policy: ActionPolicy,
    opponent_policy: ActionPolicy,
    node_budget: int,
) -> float:
    remaining = [node_budget]

    def visit(node: PlumpEnv) -> float:
        remaining[0] -= 1
        if remaining[0] < 0:
            raise _NodeBudgetExceeded
        if node.is_done():
            return _relative_rewards(
                node.state.current_round.round_scores
            )[root_player]
        policy = (
            base_policy
            if node.current_player() == root_player
            else opponent_policy
        )
        distribution = _policy_distribution(policy, node)
        expectation = 0.0
        for action, probability in distribution:
            if probability <= 0.0:
                continue
            child = node.clone()
            child.step(action)
            expectation += probability * visit(child)
        return expectation

    return visit(env.clone())


def _policy_distribution(
    policy: ActionPolicy,
    env: PlumpEnv,
) -> list[tuple[BidAction | PlayCardAction, float]]:
    legal_actions = env.legal_actions()
    if len(legal_actions) == 1:
        return [(legal_actions[0], 1.0)]
    if isinstance(policy, ModelPolicy):
        _, _, output = policy.predict(env)
        logits = (
            output.masked_bid_logits[0].float()
            if env.phase() == Phase.BIDDING
            else output.masked_card_logits[0].float()
        )
        probabilities = logits.softmax(dim=-1).detach().cpu().tolist()
        return [
            (
                action,
                probabilities[
                    action.bid
                    if isinstance(action, BidAction)
                    else card_id(action.card)
                ],
            )
            for action in legal_actions
        ]
    if isinstance(policy, RandomPolicy):
        probability = 1.0 / len(legal_actions)
        return [(action, probability) for action in legal_actions]
    if isinstance(policy, HeuristicPolicy):
        selected = policy.act(env, rng=random.Random(0))
        return [
            (action, float(action == selected))
            for action in legal_actions
        ]
    selected = policy.act(env, rng=random.Random(0))
    return [
        (action, float(action == selected))
        for action in legal_actions
    ]


def _policy_anchored_target(
    prior: dict[BidAction | PlayCardAction, float],
    regrets: dict[BidAction | PlayCardAction, float],
    *,
    temperature: float,
) -> dict[BidAction | PlayCardAction, float]:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    peak = max(regrets.values())
    weights = {
        action: max(prior[action], 1e-4)
        * math.exp((regrets[action] - peak) / temperature)
        for action in regrets
    }
    total = sum(weights.values())
    return {
        action: weight / total
        for action, weight in weights.items()
    }


def _positive_regret_target(
    regrets: dict[BidAction | PlayCardAction, float],
) -> dict[BidAction | PlayCardAction, float]:
    weights = {
        action: max(regret, 0.0)
        for action, regret in regrets.items()
    }
    total = sum(weights.values())
    if total <= 1e-12:
        best = max(regrets, key=regrets.get)
        return {
            action: float(action == best)
            for action in regrets
        }
    return {
        action: weight / total
        for action, weight in weights.items()
    }


def _split_half_stability(
    values: dict[BidAction | PlayCardAction, list[float]],
    prior: dict[BidAction | PlayCardAction, float],
    *,
    temperature: float,
) -> tuple[bool, float]:
    sample_count = min(len(samples) for samples in values.values())
    if sample_count < 4:
        return False, float("inf")
    midpoint = sample_count // 2
    targets = []
    best_actions = []
    for rows in (
        {action: samples[:midpoint] for action, samples in values.items()},
        {action: samples[midpoint:] for action, samples in values.items()},
    ):
        means = {
            action: sum(samples) / len(samples)
            for action, samples in rows.items()
        }
        baseline = sum(prior[action] * means[action] for action in means)
        regrets = {
            action: means[action] - baseline
            for action in means
        }
        targets.append(
            _policy_anchored_target(
                prior,
                regrets,
                temperature=temperature,
            )
        )
        best_actions.append(max(means, key=means.get))
    return (
        best_actions[0] == best_actions[1],
        _jensen_shannon(targets[0], targets[1]),
    )


def _jensen_shannon(
    first: dict[BidAction | PlayCardAction, float],
    second: dict[BidAction | PlayCardAction, float],
) -> float:
    midpoint = {
        action: 0.5 * (first[action] + second[action])
        for action in first
    }

    def divergence(
        source: dict[BidAction | PlayCardAction, float],
    ) -> float:
        return sum(
            probability
            * math.log(
                max(probability, 1e-12)
                / max(midpoint[action], 1e-12)
            )
            for action, probability in source.items()
            if probability > 0.0
        )

    return 0.5 * (divergence(first) + divergence(second))


def _softmax_action_values(
    values: dict[BidAction | PlayCardAction, float],
    *,
    temperature: float = 1.0,
) -> dict[str, float]:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    peak = max(values.values())
    weights = {
        action: math.exp((value - peak) / temperature)
        for action, value in values.items()
    }
    total = sum(weights.values())
    return {
        _action_key(action): weight / total
        for action, weight in weights.items()
    }
