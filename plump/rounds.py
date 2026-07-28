"""Round configuration primitives shared across training and evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from plump.state import GameConfig, ScoringConfig, TrumpPolicy


@dataclass(frozen=True, order=True)
class RoundSpec:
    num_players: int
    hand_size: int

    def validate(self) -> None:
        if self.num_players not in (3, 4, 5):
            raise ValueError("Training supports 3 to 5 players.")
        if not 3 <= self.hand_size <= 10:
            raise ValueError("Training supports hand sizes 3 to 10.")
        if self.hand_size > 52 // self.num_players:
            raise ValueError("Hand size does not fit the player count.")


def descending_ascending_schedule(
    *,
    min_cards: int = 3,
    max_cards: int = 10,
) -> list[int]:
    """Return max..min..max without duplicating the turning point."""

    if min_cards < 1:
        raise ValueError("min_cards must be positive.")
    if max_cards < min_cards:
        raise ValueError("max_cards must be at least min_cards.")
    descending = list(range(max_cards, min_cards - 1, -1))
    ascending = list(range(min_cards + 1, max_cards + 1))
    return descending + ascending


def round_game_config(
    spec: RoundSpec,
    *,
    bidding_start_player: int = 0,
    manual_hands=None,
    scoring: ScoringConfig | None = None,
) -> GameConfig:
    spec.validate()
    return GameConfig(
        num_players=spec.num_players,
        hand_sizes=[spec.hand_size],
        forbid_total_bid_equals_hand_size=True,
        scoring=scoring or ScoringConfig(),
        trump_policy=TrumpPolicy.NONE,
        manual_hands=manual_hands,
        bidding_start_players=[bidding_start_player],
    )


def rules_fingerprint(scoring: ScoringConfig | None = None) -> str:
    """Stable fingerprint for the rules used by training and evaluation."""

    payload = {
        "deck": "standard-52",
        "follow_suit": True,
        "first_leader": "highest-bid-earliest-tie",
        "forbid_total_bid_equals_hand_size": True,
        "scoring": asdict(scoring or ScoringConfig()),
        "trump_policy": TrumpPolicy.NONE.value,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(serialized).hexdigest()
