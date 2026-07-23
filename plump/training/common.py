"""Training primitives shared by PPO and schema-v5 expert iteration."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Literal

from plump.modeling import ModelConfig, card_from_id
from plump.modeling.encoding import NUM_CARDS, SUITS
from plump.rounds import RoundSpec
from plump.state import Bid


OpponentArm = Literal[
    "self",
    "heuristic",
    "mixed",
    "historical",
    "explore_self",
    "explore_historical",
    "tempered_self",
    "tempered_historical",
    "capped_self",
    "capped_historical",
    "epsilon_self",
    "epsilon_historical",
]
PositionKey = tuple[int, int, int]


def compute_relative_rewards(scores: dict[int, int]) -> dict[int, float]:
    if len(scores) < 2:
        raise ValueError("Relative rewards require at least two players.")
    total = sum(scores.values())
    opponents = len(scores) - 1
    return {
        player: float(score - ((total - score) / opponents))
        for player, score in scores.items()
    }


class PositionBaseline:
    """Lagged EMA intercept keyed by configuration and bidding position."""

    def __init__(self, decay: float = 0.98):
        self.decay = decay
        self.values: dict[PositionKey, float] = {}

    def get(self, key: PositionKey) -> float:
        return self.values.get(key, 0.0)

    def update_many(self, observations: Iterable[tuple[PositionKey, float]]) -> None:
        grouped: dict[PositionKey, list[float]] = defaultdict(list)
        for key, value in observations:
            grouped[key].append(value)
        for key, values in grouped.items():
            batch_mean = sum(values) / len(values)
            previous = self.values.get(key, batch_mean)
            self.values[key] = (
                self.decay * previous
                + (1.0 - self.decay) * batch_mean
            )

    def state_dict(self) -> dict[str, float]:
        return {
            "|".join(map(str, key)): value
            for key, value in self.values.items()
        }

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.values = {
            tuple(int(part) for part in key.split("|")): float(value)
            for key, value in state.items()
        }


@dataclass(frozen=True)
class OpponentMix:
    self_play: float = 0.3
    heuristic: float = 0.3
    mixed: float = 0.3
    historical: float = 0.1

    def effective(self, *, has_history: bool) -> dict[OpponentArm, float]:
        fractions: dict[OpponentArm, float] = {
            "self": self.self_play,
            "heuristic": self.heuristic,
            "mixed": self.mixed,
            "historical": self.historical,
        }
        if not has_history:
            fractions["self"] += fractions["historical"]
            fractions["historical"] = 0.0
        return fractions

    def validate(self) -> None:
        values = (
            self.self_play,
            self.heuristic,
            self.mixed,
            self.historical,
        )
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("Opponent-arm fractions must be in [0, 1].")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-9):
            raise ValueError("Opponent-arm fractions must sum to 1.")


def allocate_opponent_arms(
    rounds: int,
    fractions: dict[OpponentArm, float],
    rng: random.Random,
) -> list[OpponentArm]:
    order: tuple[OpponentArm, ...] = (
        "self",
        "heuristic",
        "mixed",
        "historical",
        "explore_self",
        "explore_historical",
        "tempered_self",
        "tempered_historical",
        "capped_self",
        "capped_historical",
        "epsilon_self",
        "epsilon_historical",
    )
    raw = {arm: rounds * fractions.get(arm, 0.0) for arm in order}
    counts = {arm: int(math.floor(raw[arm])) for arm in order}
    remaining = rounds - sum(counts.values())
    ranked = sorted(
        order,
        key=lambda arm: (raw[arm] - counts[arm], fractions.get(arm, 0.0)),
        reverse=True,
    )
    for arm in ranked[:remaining]:
        counts[arm] += 1
    arms = [arm for arm in order for _ in range(counts[arm])]
    rng.shuffle(arms)
    return arms


def final_tricks_relative(
    tricks_won: dict[int, int],
    observer_player: int,
    num_players: int,
    model_config: ModelConfig,
) -> list[int]:
    targets = [-100] * model_config.max_players
    for relative in range(num_players):
        player = (observer_player + relative) % num_players
        targets[relative] = tricks_won.get(player, 0)
    return targets


def final_bids_relative(
    bids: list[Bid],
    observer_player: int,
    num_players: int,
    model_config: ModelConfig,
) -> list[int]:
    bids_by_player = {bid.player: bid.value for bid in bids}
    targets = [-100] * model_config.max_players
    for relative in range(num_players):
        player = (observer_player + relative) % num_players
        targets[relative] = bids_by_player[player]
    return targets


def owner_targets_relative(
    env,
    observer_player: int,
    owner_valid_mask: list[list[bool]],
    model_config: ModelConfig,
) -> list[int]:
    round_state = env.state.current_round
    num_players = env.config.num_players
    dealt_cards = {
        card
        for hand in round_state.initial_hands.values()
        for card in hand
    }
    current_owner = {
        card: player
        for player, hand in round_state.current_hands.items()
        for card in hand
    }
    targets = [-100] * NUM_CARDS
    for index, valid_classes in enumerate(owner_valid_mask):
        if not any(valid_classes):
            continue
        card = card_from_id(index)
        owner = current_owner.get(card)
        if owner is None:
            if card in dealt_cards:
                raise AssertionError(
                    "A dealt, unplayed card has no current owner."
                )
            target = model_config.undealt_owner_class
        else:
            relative = (owner - observer_player) % num_players
            if relative == 0:
                raise AssertionError(
                    "Observer-owned cards must be excluded from owner targets."
                )
            target = relative - 1
        if not valid_classes[target]:
            raise AssertionError(
                "Ground-truth owner is excluded by the public owner mask."
            )
        targets[index] = target
    return targets


def suit_presence_targets_relative(
    env,
    observer_player: int,
    model_config: ModelConfig,
) -> list[list[int]]:
    """Per relative opponent and suit: does their current hand hold the suit?

    Relative slot 0 is the observer and is masked; padding players are masked.
    """

    round_state = env.state.current_round
    num_players = env.config.num_players
    targets = [
        [-100] * len(SUITS)
        for _ in range(model_config.max_players)
    ]
    for relative in range(1, num_players):
        player = (observer_player + relative) % num_players
        held_suits = {card.suit for card in round_state.current_hands[player]}
        for suit_index, suit in enumerate(SUITS):
            targets[relative][suit_index] = int(suit in held_suits)
    return targets


def position_key(spec: RoundSpec, bidding_position: int) -> PositionKey:
    return (spec.num_players, spec.hand_size, bidding_position)
