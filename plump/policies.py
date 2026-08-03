"""Reusable action-policy interfaces for training, evaluation, search, and GUI play."""

from __future__ import annotations

import random
from collections import defaultdict
from math import comb, exp
from typing import Protocol, runtime_checkable

from plump.cards import Card, Rank, Suit
from plump.env import PlumpEnv
from plump.rules import determine_trick_winner
from plump.state import BidAction, Observation, Phase, PlayCardAction, Trick, TrickPlay


class ActionPolicy(Protocol):
    name: str
    forward_passes: int

    def act(self, env: PlumpEnv, *, rng: random.Random | None = None) -> BidAction | PlayCardAction:
        ...

    def reset_counters(self) -> None:
        ...


@runtime_checkable
class BatchedActionPolicy(ActionPolicy, Protocol):
    """Policy that can act on a heterogeneous environment batch."""

    def act_many(
        self,
        envs: list[PlumpEnv],
        *,
        rngs: list[random.Random] | None = None,
    ) -> list[BidAction | PlayCardAction]:
        ...


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.forward_passes = 0

    def act(self, env: PlumpEnv, *, rng: random.Random | None = None) -> BidAction | PlayCardAction:
        chooser = rng or self.rng
        return chooser.choice(env.legal_actions())

    def reset_counters(self) -> None:
        self.forward_passes = 0


class HeuristicPolicy:
    """Small deterministic baseline that bids strength and plays toward its bid."""

    name = "heuristic"
    bid_signal_strength = 0.12
    max_bid_signal = 2.0

    def __init__(self) -> None:
        self.forward_passes = 0

    def act(self, env: PlumpEnv, *, rng: random.Random | None = None) -> BidAction | PlayCardAction:
        player = env.current_player()
        observation = env.get_observation(player)
        if env.phase() == Phase.BIDDING:
            card_distribution = _rank_only_trick_distribution(
                observation.my_hand,
                num_players=env.config.num_players,
            )
            card_only_bid = _select_expected_score_bid(
                card_distribution,
                observation.legal_bids,
            )
            adjusted_distribution = _adjust_distribution_for_prior_bids(
                card_distribution,
                prior_bids=[bid.value for bid in observation.bids],
                hand_size=observation.hand_size,
                num_players=env.config.num_players,
                strength=self.bid_signal_strength,
                max_signal=self.max_bid_signal,
            )
            nearby_legal_bids = [
                value
                for value in observation.legal_bids
                if abs(value - card_only_bid) <= 1
            ]
            bid = _select_expected_score_bid(adjusted_distribution, nearby_legal_bids)
            return BidAction(player, bid)
        if env.phase() == Phase.PLAYING:
            card = _select_heuristic_play(
                observation,
                num_players=env.config.num_players,
            )
            return PlayCardAction(player, card)
        raise RuntimeError(f"Cannot act in phase {env.phase().value}.")

    def reset_counters(self) -> None:
        self.forward_passes = 0

    def act_many(
        self,
        envs: list[PlumpEnv],
        *,
        rngs: list[random.Random] | None = None,
    ) -> list[BidAction | PlayCardAction]:
        """Batch interface used by rollout/evaluation wave schedulers.

        The heuristic is CPU-only and deterministic, but accepting a whole
        wave lets callers keep it inside their existing batched orchestration
        without allocating model/cache rows for heuristic seats.
        """

        if rngs is None:
            rngs = [None] * len(envs)
        if len(rngs) != len(envs):
            raise ValueError("rngs must match envs.")
        return [self.act(env, rng=rng) for env, rng in zip(envs, rngs)]


def _rank_only_trick_distribution(
    hand: list[Card],
    *,
    num_players: int,
) -> dict[int, float]:
    """Exact trick distribution when only same-suit rank strength matters."""

    hand_size = len(hand)
    unknown_count = 52 - hand_size
    opponent_card_count = (num_players - 1) * hand_size
    if opponent_card_count > unknown_count:
        raise ValueError("The hand does not fit the requested player count.")

    hand_ranks = {
        suit: {int(card.rank) for card in hand if card.suit == suit}
        for suit in Suit
    }
    # State is (opponent cards assigned, unbeaten cards in our hand) -> deal count.
    distribution_counts: dict[tuple[int, int], int] = {(0, 0): 1}
    for suit in Suit:
        own = hand_ranks[suit]
        unknown = [int(rank) for rank in Rank if int(rank) not in own]
        local_counts: dict[tuple[int, int], int] = defaultdict(int)
        for selected in range(min(len(unknown), opponent_card_count) + 1):
            if selected == 0:
                local_counts[(0, len(own))] = 1
                continue
            for max_index, opponent_highest in enumerate(unknown):
                if max_index < selected - 1:
                    continue
                winners = sum(rank > opponent_highest for rank in own)
                local_counts[(selected, winners)] += comb(max_index, selected - 1)

        next_counts: dict[tuple[int, int], int] = defaultdict(int)
        for (assigned, winners), ways in distribution_counts.items():
            for (local_assigned, local_winners), local_ways in local_counts.items():
                total_assigned = assigned + local_assigned
                if total_assigned <= opponent_card_count:
                    next_counts[(total_assigned, winners + local_winners)] += ways * local_ways
        distribution_counts = next_counts

    total_deals = comb(unknown_count, opponent_card_count)
    probabilities: dict[int, float] = defaultdict(float)
    for (assigned, winners), ways in distribution_counts.items():
        if assigned == opponent_card_count:
            probabilities[winners] += ways / total_deals
    return dict(probabilities)


def _adjust_distribution_for_prior_bids(
    distribution: dict[int, float],
    *,
    prior_bids: list[int],
    hand_size: int,
    num_players: int,
    strength: float,
    max_signal: float,
) -> dict[int, float]:
    """Weakly condition rank strength on whether earlier bids look high or low."""

    if not prior_bids:
        return distribution
    expected_prior_total = len(prior_bids) * hand_size / num_players
    signal = sum(prior_bids) - expected_prior_total
    signal = max(-max_signal, min(max_signal, signal))
    if abs(signal) <= 1e-12:
        return distribution

    weighted = {
        tricks: probability * exp(-strength * signal * tricks)
        for tricks, probability in distribution.items()
    }
    normalizer = sum(weighted.values())
    return {
        tricks: probability / normalizer
        for tricks, probability in weighted.items()
    }


def _select_expected_score_bid(
    distribution: dict[int, float],
    legal_bids: list[int],
) -> int:
    """Choose the legal bid with the highest expected exact-hit score."""

    expected_tricks = sum(tricks * probability for tricks, probability in distribution.items())
    return max(
        legal_bids,
        key=lambda value: (
            distribution.get(value, 0.0) * (5 if value == 0 else 10 + value),
            -abs(value - expected_tricks),
            -value,
        ),
    )


def _select_heuristic_play(
    observation: Observation,
    *,
    num_players: int,
) -> Card:
    """Play toward the bid, or disrupt opponents once our bid is lost."""

    player = observation.player_id
    bid_by_player = {bid.player: bid.value for bid in observation.bids}
    own_bid = bid_by_player[player]
    tricks_won = observation.tricks_won.get(player, 0)
    remaining_tricks = len(observation.my_hand)
    gone_over = tricks_won > own_bid
    cannot_reach = tricks_won + remaining_tricks < own_bid

    if gone_over or cannot_reach:
        total_bid = sum(bid_by_player.values())
        if gone_over and total_bid > observation.hand_size:
            wants_trick = True
        elif cannot_reach and total_bid < observation.hand_size:
            wants_trick = False
        else:
            current_winner = _current_trick_winner(
                observation.current_trick,
                observation.trump_suit,
            )
            wants_trick = _lost_player_should_take(
                player=player,
                bids=bid_by_player,
                tricks_won=observation.tricks_won,
                remaining_tricks=remaining_tricks,
                current_winner=current_winner,
                num_players=num_players,
                total_round_tricks=observation.hand_size,
            )
    else:
        wants_trick = tricks_won < own_bid

    return _select_card_for_intent(
        observation.legal_cards,
        player=player,
        current_trick=observation.current_trick,
        trump_suit=observation.trump_suit,
        wants_trick=wants_trick,
    )


def _select_card_for_intent(
    legal_cards: list[Card],
    *,
    player: int,
    current_trick: Trick | None,
    trump_suit: Suit | None,
    wants_trick: bool,
) -> Card:
    """Take with the highest winner, or shed the highest card that still loses."""

    if not legal_cards:
        raise ValueError("Heuristic play requires at least one legal card.")
    ordered = sorted(legal_cards, key=lambda card: (int(card.rank), card.suit.value))
    winners = [
        card
        for card in ordered
        if _card_is_current_winner(
            card,
            player=player,
            current_trick=current_trick,
            trump_suit=trump_suit,
        )
    ]
    if wants_trick:
        return winners[-1] if winners else ordered[0]

    winner_set = set(winners)
    losing_cards = [card for card in ordered if card not in winner_set]
    return losing_cards[-1] if losing_cards else ordered[0]


def _card_is_current_winner(
    card: Card,
    *,
    player: int,
    current_trick: Trick | None,
    trump_suit: Suit | None,
) -> bool:
    if current_trick is None:
        return True
    led_suit = current_trick.led_suit
    if not current_trick.plays:
        led_suit = card.suit
    trial = Trick(
        trick_index=current_trick.trick_index,
        leader=current_trick.leader,
        led_suit=led_suit,
        plays=list(current_trick.plays)
        + [TrickPlay(player=player, card=card, position=len(current_trick.plays))],
    )
    return determine_trick_winner(trial, trump_suit) == player


def _current_trick_winner(
    current_trick: Trick | None,
    trump_suit: Suit | None,
) -> int | None:
    if current_trick is None or not current_trick.plays:
        return None
    return determine_trick_winner(current_trick, trump_suit)


def _lost_player_should_take(
    *,
    player: int,
    bids: dict[int, int],
    tricks_won: dict[int, int],
    remaining_tricks: int,
    current_winner: int | None,
    num_players: int,
    total_round_tricks: int,
) -> bool:
    """Choose the current outcome that minimizes opponents' expected hit points."""

    take_score = _expected_opponent_hit_points(
        player=player,
        bids=bids,
        tricks_won=tricks_won,
        future_tricks=max(remaining_tricks - 1, 0),
        current_recipient=player,
        num_players=num_players,
    )
    if current_winner is not None and current_winner != player:
        offload_score = _expected_opponent_hit_points(
            player=player,
            bids=bids,
            tricks_won=tricks_won,
            future_tricks=max(remaining_tricks - 1, 0),
            current_recipient=current_winner,
            num_players=num_players,
        )
    else:
        opponent_scores = [
            _expected_opponent_hit_points(
                player=player,
                bids=bids,
                tricks_won=tricks_won,
                future_tricks=max(remaining_tricks - 1, 0),
                current_recipient=opponent,
                num_players=num_players,
            )
            for opponent in bids
            if opponent != player
        ]
        offload_score = sum(opponent_scores) / len(opponent_scores)

    if abs(take_score - offload_score) > 1e-12:
        return take_score < offload_score
    return sum(bids.values()) >= total_round_tricks


def _expected_opponent_hit_points(
    *,
    player: int,
    bids: dict[int, int],
    tricks_won: dict[int, int],
    future_tricks: int,
    current_recipient: int,
    num_players: int,
) -> float:
    probability_per_trick = 1.0 / num_players
    expected_points = 0.0
    for opponent, bid in bids.items():
        if opponent == player:
            continue
        current_delta = int(current_recipient == opponent)
        needed = bid - tricks_won.get(opponent, 0) - current_delta
        probability = _binomial_probability(
            trials=future_tricks,
            successes=needed,
            probability=probability_per_trick,
        )
        hit_points = 5 if bid == 0 else 10 + bid
        expected_points += probability * hit_points
    return expected_points


def _binomial_probability(
    *,
    trials: int,
    successes: int,
    probability: float,
) -> float:
    if successes < 0 or successes > trials:
        return 0.0
    return (
        comb(trials, successes)
        * probability**successes
        * (1.0 - probability) ** (trials - successes)
    )
