"""Schema-v6 per-seat token sequences and replay-derived training labels.

A seat sequence is the game from one observer's point of view:

    [GAME] [HAND x N] [BID x P] { [PLAY x P] [TRICK_WIN] } x N

All player references are observer-relative. The sequence contains exactly the
information the observer may see: their own dealt hand plus the public event
stream. Per-position auxiliary labels (card ownership, suit presence, final
trick counts) are reconstructed by replaying the event log against the true
deal; they are training targets only and never model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plump.cards import Card, Rank, Suit, sort_cards
from plump.state import EventType, GameEvent

from .config import (
    NEXT_BID,
    NEXT_NONE,
    NEXT_PLAY,
    NUM_CARDS,
    SLOT_BID,
    SLOT_CARD,
    SLOT_HAND_SIZE,
    SLOT_NEXT_ACTOR,
    SLOT_NEXT_PHASE,
    SLOT_NUM_PLAYERS,
    SLOT_POS_IN_TRICK,
    SLOT_RANK,
    SLOT_REL_PLAYER,
    SLOT_SUIT,
    SLOT_TRICK,
    SLOT_TYPE,
    TOKEN_BID,
    TOKEN_GAME,
    TOKEN_HAND,
    TOKEN_PLAY,
    TOKEN_TRICK_WIN,
    TOKEN_WIDTH,
    SeqModelConfig,
    seq_len,
)

SUITS: tuple[Suit, ...] = (Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS)
RANKS: tuple[Rank, ...] = tuple(Rank)

IGNORE_LABEL = -100


def card_id(card: Card) -> int:
    return SUITS.index(card.suit) * len(RANKS) + RANKS.index(card.rank)


def card_from_id(index: int) -> Card:
    if index < 0 or index >= NUM_CARDS:
        raise ValueError(f"Card id must be in 0..{NUM_CARDS - 1}.")
    return Card(SUITS[index // len(RANKS)], RANKS[index % len(RANKS)])


def _relative(player: int, observer: int, num_players: int) -> int:
    return (player - observer) % num_players


def _base_token(config: SeqModelConfig, num_players: int, hand_size: int) -> list[int]:
    token = [0] * TOKEN_WIDTH
    token[SLOT_REL_PLAYER] = config.player_na_id
    token[SLOT_RANK] = config.rank_na_id
    token[SLOT_SUIT] = config.suit_na_id
    token[SLOT_CARD] = config.card_na_id
    token[SLOT_BID] = config.bid_na_id
    token[SLOT_TRICK] = config.trick_na_id
    token[SLOT_POS_IN_TRICK] = config.pos_na_id
    token[SLOT_HAND_SIZE] = hand_size
    token[SLOT_NUM_PLAYERS] = num_players
    token[SLOT_NEXT_ACTOR] = config.player_na_id
    token[SLOT_NEXT_PHASE] = NEXT_NONE
    return token


def game_token(
    config: SeqModelConfig,
    num_players: int,
    hand_size: int,
    bidding_position: int,
) -> list[int]:
    token = _base_token(config, num_players, hand_size)
    token[SLOT_TYPE] = TOKEN_GAME
    token[SLOT_POS_IN_TRICK] = bidding_position
    return token


def hand_token(
    config: SeqModelConfig,
    num_players: int,
    hand_size: int,
    card: Card,
) -> list[int]:
    token = _base_token(config, num_players, hand_size)
    token[SLOT_TYPE] = TOKEN_HAND
    token[SLOT_REL_PLAYER] = 0
    token[SLOT_RANK] = RANKS.index(card.rank)
    token[SLOT_SUIT] = SUITS.index(card.suit)
    token[SLOT_CARD] = card_id(card)
    return token


def event_token(
    config: SeqModelConfig,
    event: GameEvent,
    observer: int,
    num_players: int,
    hand_size: int,
) -> list[int] | None:
    """Map one public game event to a token row; None for non-token events."""

    if event.type == EventType.BID:
        token = _base_token(config, num_players, hand_size)
        token[SLOT_TYPE] = TOKEN_BID
        token[SLOT_REL_PLAYER] = _relative(event.player, observer, num_players)
        token[SLOT_BID] = event.bid
        return token
    if event.type == EventType.PLAY:
        token = _base_token(config, num_players, hand_size)
        token[SLOT_TYPE] = TOKEN_PLAY
        token[SLOT_REL_PLAYER] = _relative(event.player, observer, num_players)
        token[SLOT_RANK] = RANKS.index(event.card.rank)
        token[SLOT_SUIT] = SUITS.index(event.card.suit)
        token[SLOT_CARD] = card_id(event.card)
        token[SLOT_TRICK] = event.trick_index
        token[SLOT_POS_IN_TRICK] = event.position_in_trick
        return token
    if event.type == EventType.TRICK_WIN:
        token = _base_token(config, num_players, hand_size)
        token[SLOT_TYPE] = TOKEN_TRICK_WIN
        token[SLOT_REL_PLAYER] = _relative(event.player, observer, num_players)
        token[SLOT_TRICK] = event.trick_index
        return token
    return None


def prefix_tokens(
    config: SeqModelConfig,
    observer: int,
    num_players: int,
    hand_size: int,
    initial_hand: list[Card],
    bidding_start_player: int,
) -> list[list[int]]:
    """[GAME] + [HAND x N] with next-actor fields set for the first bid."""

    if len(initial_hand) != hand_size:
        raise ValueError("Observer hand does not match the hand size.")
    bidding_position = _relative(observer, bidding_start_player, num_players)
    tokens = [game_token(config, num_players, hand_size, bidding_position)]
    for card in sort_cards(initial_hand):
        tokens.append(hand_token(config, num_players, hand_size, card))
    tokens[-1][SLOT_NEXT_ACTOR] = _relative(bidding_start_player, observer, num_players)
    tokens[-1][SLOT_NEXT_PHASE] = NEXT_BID
    return tokens


def _round_events(events: list[GameEvent]) -> list[GameEvent]:
    return [
        event
        for event in events
        if event.type in (EventType.BID, EventType.PLAY, EventType.TRICK_WIN)
    ]


def build_seat_tokens(
    config: SeqModelConfig,
    events: list[GameEvent],
    observer: int,
    num_players: int,
    hand_size: int,
    initial_hand: list[Card],
    bidding_start_player: int,
    pending_actor: int | None = None,
    pending_phase: int = NEXT_NONE,
) -> np.ndarray:
    """Token rows for one observer. ``pending_actor`` (absolute id) sets the
    next-actor fields of the final token for in-progress games."""

    round_events = _round_events(events)
    tokens = prefix_tokens(
        config, observer, num_players, hand_size, initial_hand, bidding_start_player
    )
    for event in round_events:
        token = event_token(config, event, observer, num_players, hand_size)
        if token is None:
            raise AssertionError("Round events must map to tokens.")
        tokens.append(token)

    # Look-ahead next-actor annotation: token t announces the actor of the
    # action event at t+1 (TRICK_WIN and HAND interiors announce nothing).
    prefix_len = 1 + hand_size
    for position in range(prefix_len - 1, len(tokens) - 1):
        upcoming = tokens[position + 1]
        if upcoming[SLOT_TYPE] in (TOKEN_BID, TOKEN_PLAY):
            tokens[position][SLOT_NEXT_ACTOR] = upcoming[SLOT_REL_PLAYER]
            tokens[position][SLOT_NEXT_PHASE] = (
                NEXT_BID if upcoming[SLOT_TYPE] == TOKEN_BID else NEXT_PLAY
            )
        else:
            tokens[position][SLOT_NEXT_ACTOR] = config.player_na_id
            tokens[position][SLOT_NEXT_PHASE] = NEXT_NONE
    if pending_actor is not None:
        tokens[-1][SLOT_NEXT_ACTOR] = _relative(pending_actor, observer, num_players)
        tokens[-1][SLOT_NEXT_PHASE] = pending_phase
    return np.asarray(tokens, dtype=np.int64)


@dataclass
class ReplayArrays:
    """Tokens plus per-position labels for one focal seat of one leaf."""

    tokens: np.ndarray            # [L, TOKEN_WIDTH] int64
    decision_positions: np.ndarray  # [D] int64 — h_t predicts the observer's action
    decision_phases: np.ndarray   # [D] int64 — NEXT_BID or NEXT_PLAY
    legal_bid_masks: np.ndarray   # [D, bid_count] bool
    legal_card_masks: np.ndarray  # [D, 52] bool
    action_targets: np.ndarray    # [D] int64 — bid value or card id actually taken
    owner_targets: np.ndarray     # [L, 52] int64, IGNORE_LABEL outside hidden cards
    owner_valid: np.ndarray       # [L, 52, owner_classes] bool
    owner_capacities: np.ndarray  # [L, owner_classes] int64
    trick_targets: np.ndarray     # [max_players] int64, IGNORE_LABEL padding
    trick_masks: np.ndarray       # [L, max_players, bid_count] bool
    suit_targets: np.ndarray      # [L, max_players, 4] int64, IGNORE_LABEL masked
    final_tricks_won: dict[int, int]


def build_replay_arrays(
    config: SeqModelConfig,
    initial_hands: dict[int, list[Card]],
    events: list[GameEvent],
    observer: int,
    num_players: int,
    hand_size: int,
    bidding_start_player: int,
    label_from: int = 0,
) -> ReplayArrays:
    """Replay a completed round and emit tokens plus per-position labels.

    Labels at position t describe the game state after the event at t (prefix
    positions use the post-deal state). Truth (opponents' hands) is used only
    for targets; the token stream itself stays observer-visible.

    ``label_from`` restricts label filling and decision records to positions
    >= label_from — a leaf that owns only its post-branch suffix skips the
    expensive label work for prefix positions owned by another leaf.
    """

    if set(initial_hands) != set(range(num_players)):
        raise ValueError("initial_hands must cover every player.")
    for player, hand in initial_hands.items():
        if len(hand) != hand_size:
            raise ValueError(f"Player {player} hand does not match hand size.")

    tokens = build_seat_tokens(
        config,
        events,
        observer,
        num_players,
        hand_size,
        initial_hands[observer],
        bidding_start_player,
    )
    total_len = seq_len(num_players, hand_size)
    if tokens.shape[0] != total_len:
        raise ValueError(
            f"Replay expects a completed round: {tokens.shape[0]} tokens != {total_len}."
        )

    classes = config.owner_class_count
    bid_count = config.bid_count
    owner_targets = np.full((total_len, NUM_CARDS), IGNORE_LABEL, dtype=np.int64)
    owner_valid = np.zeros((total_len, NUM_CARDS, classes), dtype=bool)
    owner_capacities = np.zeros((total_len, classes), dtype=np.int64)
    trick_masks = np.zeros((total_len, config.max_players, bid_count), dtype=bool)
    suit_targets = np.full(
        (total_len, config.max_players, len(SUITS)), IGNORE_LABEL, dtype=np.int64
    )

    # Vectorized replay state (rel-indexed): holder_rel[card] is the current
    # holder's relative seat, ``num_players`` for the undealt kitty, -1 once
    # publicly played.
    suit_of_card = np.arange(NUM_CARDS) // len(RANKS)
    holder_rel = np.full(NUM_CARDS, num_players, dtype=np.int64)
    suit_count_rel = np.zeros((num_players, len(SUITS)), dtype=np.int64)
    for player, hand in initial_hands.items():
        rel = (player - observer) % num_players
        for card in hand:
            holder_rel[card_id(card)] = rel
            suit_count_rel[rel, SUITS.index(card.suit)] += 1
    void_rel = np.zeros((num_players, len(SUITS)), dtype=bool)
    played_count_rel = np.zeros(num_players, dtype=np.int64)
    tricks_won_rel = np.zeros(num_players, dtype=np.int64)
    observer_hand: set[Card] = set(initial_hands[observer])
    bid_values: list[int] = []
    completed_tricks = 0
    current_led_suit: Suit | None = None
    current_trick_size = 0
    kitty_size = NUM_CARDS - hand_size * num_players
    undealt = config.undealt_owner_class
    count_range = np.arange(bid_count)
    card_range = np.arange(NUM_CARDS)

    def fill_labels(position: int) -> None:
        hidden = holder_rel > 0
        valid = owner_valid[position]
        for rel in range(1, num_players):
            valid[:, rel - 1] = hidden & ~void_rel[rel, suit_of_card]
        if kitty_size > 0:
            valid[:, undealt] = hidden
        targets = np.where(holder_rel == num_players, undealt, holder_rel - 1)
        owner_targets[position] = np.where(hidden, targets, IGNORE_LABEL)
        if not valid[card_range[hidden], targets[hidden]].all():
            raise AssertionError(
                "Ground-truth owner is excluded by the public owner mask."
            )
        owner_capacities[position, : num_players - 1] = (
            hand_size - played_count_rel[1:num_players]
        )
        owner_capacities[position, undealt] = kitty_size
        unresolved = hand_size - completed_tricks
        upper = np.minimum(tricks_won_rel + unresolved, config.max_hand_size)
        trick_masks[position, :num_players] = (
            count_range[None, :] >= tricks_won_rel[:, None]
        ) & (count_range[None, :] <= upper[:, None])
        suit_targets[position, 1:num_players] = (
            suit_count_rel[1:num_players] > 0
        ).astype(np.int64)

    prefix_len = 1 + hand_size
    if label_from < prefix_len:
        first = max(label_from, 0)
        fill_labels(first)
        for position in range(first + 1, prefix_len):
            owner_targets[position] = owner_targets[first]
            owner_valid[position] = owner_valid[first]
            owner_capacities[position] = owner_capacities[first]
            trick_masks[position] = trick_masks[first]
            suit_targets[position] = suit_targets[first]

    decision_positions: list[int] = []
    decision_phases: list[int] = []
    legal_bid_masks: list[np.ndarray] = []
    legal_card_masks: list[np.ndarray] = []
    action_targets: list[int] = []

    def record_decision(position: int, upcoming: GameEvent) -> None:
        if upcoming.type not in (EventType.BID, EventType.PLAY):
            return
        if upcoming.player != observer or position < label_from:
            return
        phase = NEXT_BID if upcoming.type == EventType.BID else NEXT_PLAY
        bid_mask = np.zeros(bid_count, dtype=bool)
        card_mask = np.zeros(NUM_CARDS, dtype=bool)
        if phase == NEXT_BID:
            values = list(range(hand_size + 1))
            if len(bid_values) == num_players - 1:
                forbidden = hand_size - sum(bid_values)
                values = [value for value in values if value != forbidden]
            bid_mask[values] = True
            target = upcoming.bid
        else:
            hand = observer_hand
            if current_trick_size > 0 and current_led_suit is not None:
                suited = [card for card in hand if card.suit == current_led_suit]
                legal = suited if suited else list(hand)
            else:
                legal = list(hand)
            for card in legal:
                card_mask[card_id(card)] = True
            target = card_id(upcoming.card)
        decision_positions.append(position)
        decision_phases.append(phase)
        legal_bid_masks.append(bid_mask)
        legal_card_masks.append(card_mask)
        action_targets.append(target)

    round_events = _round_events(events)
    if round_events:
        record_decision(prefix_len - 1, round_events[0])
    for index, event in enumerate(round_events):
        position = prefix_len + index
        if event.type == EventType.BID:
            bid_values.append(event.bid)
        elif event.type == EventType.PLAY:
            rel = (event.player - observer) % num_players
            if current_trick_size == 0:
                current_led_suit = event.card.suit
            elif (
                current_led_suit is not None
                and event.card.suit != current_led_suit
            ):
                void_rel[rel, SUITS.index(current_led_suit)] = True
            index_of_card = card_id(event.card)
            if holder_rel[index_of_card] != rel:
                raise AssertionError("Replay saw a play from a non-holder.")
            holder_rel[index_of_card] = -1
            suit_count_rel[rel, SUITS.index(event.card.suit)] -= 1
            played_count_rel[rel] += 1
            if event.player == observer:
                observer_hand.remove(event.card)
            current_trick_size += 1
        elif event.type == EventType.TRICK_WIN:
            tricks_won_rel[(event.player - observer) % num_players] += 1
            completed_tricks += 1
            current_trick_size = 0
            current_led_suit = None
        if position >= label_from:
            fill_labels(position)
        if index + 1 < len(round_events):
            record_decision(position, round_events[index + 1])

    if completed_tricks != hand_size:
        raise ValueError("Replay expects a completed round.")

    trick_targets = np.full(config.max_players, IGNORE_LABEL, dtype=np.int64)
    trick_targets[:num_players] = tricks_won_rel

    return ReplayArrays(
        tokens=tokens,
        decision_positions=np.asarray(decision_positions, dtype=np.int64),
        decision_phases=np.asarray(decision_phases, dtype=np.int64),
        legal_bid_masks=(
            np.stack(legal_bid_masks)
            if legal_bid_masks
            else np.zeros((0, bid_count), dtype=bool)
        ),
        legal_card_masks=(
            np.stack(legal_card_masks)
            if legal_card_masks
            else np.zeros((0, NUM_CARDS), dtype=bool)
        ),
        action_targets=np.asarray(action_targets, dtype=np.int64),
        owner_targets=owner_targets,
        owner_valid=owner_valid,
        owner_capacities=owner_capacities,
        trick_targets=trick_targets,
        trick_masks=trick_masks,
        suit_targets=suit_targets,
        final_tricks_won={
            (observer + rel) % num_players: int(tricks_won_rel[rel])
            for rel in range(num_players)
        },
    )
