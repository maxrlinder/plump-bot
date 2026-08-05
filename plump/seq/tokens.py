"""Schema-v6 per-seat token sequences and replay-derived training labels.

A seat sequence is the game from one observer's point of view:

    [GAME] [HAND x N] [BID x P] { [PLAY x P] [TRICK_WIN] } x N

All player references are observer-relative. The sequence contains exactly the
information the observer may see: their own dealt hand plus the public event
stream. Per-position auxiliary labels (suit presence, bid hit, final trick
counts) are reconstructed by replaying the event log against the true deal; they
are training targets only and never model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from plump.cards import Card, Rank, Suit, sort_cards
from plump.state import EventType, GameEvent

from .config import (
    BASE_TOKEN_WIDTH,
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
    SLOT_REMAINING_HAND_START,
    SLOT_SUIT,
    SLOT_TRICK,
    SLOT_TYPE,
    TOKEN_BID,
    TOKEN_GAME,
    TOKEN_HAND,
    TOKEN_PLAY,
    TOKEN_TRICK_WIN,
    TOKEN_TURN,
    TOKEN_WIDTH,
    SeqModelConfig,
)

SUITS: tuple[Suit, ...] = (Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS)
RANKS: tuple[Rank, ...] = tuple(Rank)

IGNORE_LABEL = -100


# There are 52 cards and this is called millions of times per update, so the
# two tuple.index() linear scans are worth trading for one dict hit.
_CARD_IDS: dict[Card, int] = {
    Card(suit, rank): suit_index * len(RANKS) + rank_index
    for suit_index, suit in enumerate(SUITS)
    for rank_index, rank in enumerate(RANKS)
}


def card_id(card: Card) -> int:
    return _CARD_IDS[card]


def card_from_id(index: int) -> Card:
    if index < 0 or index >= NUM_CARDS:
        raise ValueError(f"Card id must be in 0..{NUM_CARDS - 1}.")
    return Card(SUITS[index // len(RANKS)], RANKS[index % len(RANKS)])


def _relative(player: int, observer: int, num_players: int) -> int:
    return (player - observer) % num_players


@lru_cache(maxsize=None)
def _base_token_template(
    config: SeqModelConfig, num_players: int, hand_size: int
) -> tuple[int, ...]:
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
    token[SLOT_REMAINING_HAND_START:] = [config.card_na_id] * (
        TOKEN_WIDTH - BASE_TOKEN_WIDTH
    )
    return tuple(token)


def _base_token(config: SeqModelConfig, num_players: int, hand_size: int) -> list[int]:
    # The template depends only on (config, players, cards), all constant across
    # a shape group, so the eleven field writes happen once per shape instead of
    # once per token -- and list(tuple) is a single C-level copy.
    return list(_base_token_template(config, num_players, hand_size))


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


def oracle_hand_token(
    config: SeqModelConfig,
    num_players: int,
    hand_size: int,
    owner: int,
    card: Card,
) -> list[int]:
    """A critic-only HAND token whose player field is the absolute owner."""

    if owner < 0 or owner >= num_players:
        raise ValueError("Oracle card owner must be an active absolute seat.")
    token = hand_token(config, num_players, hand_size, card)
    token[SLOT_REL_PLAYER] = owner
    return token


def turn_token(
    config: SeqModelConfig,
    num_players: int,
    hand_size: int,
    rel_actor: int,
    phase: int,
    trick_index: int | None = None,
    pos_in_trick: int | None = None,
) -> list[int]:
    """A pause token announcing whose turn it is. ``rel_actor == 0`` is "mine".

    Deliberately near-empty: rank/suit/card/bid stay NA so the only thing this
    position contributes to the residual stream is "an action is due, from this
    seat, at this point of the game". Everything else has to come through
    attention, which is the point -- the head reads a state summary rather than
    a state summary tangled up with the last event's card.
    """

    token = _base_token(config, num_players, hand_size)
    token[SLOT_TYPE] = TOKEN_TURN
    token[SLOT_REL_PLAYER] = rel_actor
    if trick_index is not None:
        token[SLOT_TRICK] = trick_index
    if pos_in_trick is not None:
        token[SLOT_POS_IN_TRICK] = pos_in_trick
    token[SLOT_NEXT_ACTOR] = rel_actor
    token[SLOT_NEXT_PHASE] = phase
    return token


def turn_token_for_phase(config: SeqModelConfig, phase: int) -> bool:
    """Does an upcoming action in ``phase`` (NEXT_BID/NEXT_PLAY) get a TURN?"""

    if config.turn_token == "off" or phase == NEXT_NONE:
        return False
    return phase == NEXT_BID or config.turn_token == "all"


def turn_token_precedes(config: SeqModelConfig, event: GameEvent) -> bool:
    """Does a TURN token sit immediately before this event's token?"""

    if event.type == EventType.BID:
        return turn_token_for_phase(config, NEXT_BID)
    if event.type == EventType.PLAY:
        return turn_token_for_phase(config, NEXT_PLAY)
    return False


def emits_token(config: SeqModelConfig, event: GameEvent) -> bool:
    if event.type == EventType.TRICK_WIN:
        return config.trick_win_token
    return event.type in (EventType.BID, EventType.PLAY)


def event_token(
    config: SeqModelConfig,
    event: GameEvent,
    observer: int,
    num_players: int,
    hand_size: int,
    remaining_hand: list[Card] | None = None,
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
        if not config.trick_win_token:
            return None
        token = _base_token(config, num_players, hand_size)
        token[SLOT_TYPE] = TOKEN_TRICK_WIN
        token[SLOT_REL_PLAYER] = _relative(event.player, observer, num_players)
        token[SLOT_TRICK] = event.trick_index
        if remaining_hand is not None:
            set_remaining_hand(token, remaining_hand, hand_size)
        return token
    return None


def set_remaining_hand(
    token: list[int] | np.ndarray,
    cards: list[Card],
    hand_size: int,
) -> None:
    """Write a sorted observer-visible hand into a TRICK_WIN token."""

    if len(cards) > hand_size:
        raise ValueError("Remaining hand cannot exceed the round hand size.")
    values = [card_id(card) for card in sort_cards(cards)]
    width = TOKEN_WIDTH - SLOT_REMAINING_HAND_START
    if len(values) > width:
        raise ValueError("Remaining hand exceeds the token's card capacity.")
    token[SLOT_REMAINING_HAND_START:] = values + [NUM_CARDS] * (
        width - len(values)
    )


def prefix_tokens(
    config: SeqModelConfig,
    observer: int,
    num_players: int,
    hand_size: int,
    initial_hand: list[Card],
    bidding_start_player: int,
) -> list[list[int]]:
    """[GAME] + [HAND x N] with next-actor fields set for the first bid.

    With ``turn_token`` enabled the opening bid's TURN token belongs here: it
    precedes the first event, so it is part of the prefill rather than of any
    wave, and the first bidder's readout comes off it.
    """

    if len(initial_hand) != hand_size:
        raise ValueError("Observer hand does not match the hand size.")
    bidding_position = _relative(observer, bidding_start_player, num_players)
    tokens = [game_token(config, num_players, hand_size, bidding_position)]
    for card in sort_cards(initial_hand):
        tokens.append(hand_token(config, num_players, hand_size, card))
    rel_first = _relative(bidding_start_player, observer, num_players)
    if config.turn_token != "off":
        tokens.append(
            turn_token(config, num_players, hand_size, rel_first, NEXT_BID)
        )
    else:
        tokens[-1][SLOT_NEXT_ACTOR] = rel_first
        tokens[-1][SLOT_NEXT_PHASE] = NEXT_BID
    return tokens


def _round_events(events: list[GameEvent]) -> list[GameEvent]:
    return [
        event
        for event in events
        if event.type in (EventType.BID, EventType.PLAY, EventType.TRICK_WIN)
    ]


def token_layout(
    config: SeqModelConfig, round_events: list[GameEvent]
) -> list[tuple[str, int]]:
    """Token stream after the prefix, as ("turn"|"event", event index) pairs.

    One layout serves both the rollout wave loop and the replay labeller, so
    the schema flags cannot make the two disagree about which token sits where.
    Event 0 is always the opening bid, whose TURN token lives in the prefix.
    """

    layout: list[tuple[str, int]] = []
    for index, event in enumerate(round_events):
        if index > 0 and turn_token_precedes(config, event):
            layout.append(("turn", index))
        if emits_token(config, event):
            layout.append(("event", index))
    return layout


def _turn_row(
    config: SeqModelConfig,
    event: GameEvent,
    observer: int,
    num_players: int,
    hand_size: int,
) -> list[int]:
    """The TURN token that precedes ``event``."""

    is_bid = event.type == EventType.BID
    return turn_token(
        config,
        num_players,
        hand_size,
        _relative(event.player, observer, num_players),
        NEXT_BID if is_bid else NEXT_PLAY,
        trick_index=None if is_bid else event.trick_index,
        pos_in_trick=None if is_bid else event.position_in_trick,
    )


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
    token_prefix: np.ndarray | None = None,
) -> np.ndarray:
    """Token rows for one observer. ``pending_actor`` (absolute id) sets the
    next-actor fields of the final token for in-progress games.

    ``token_prefix`` is an already-built ``[n, TOKEN_WIDTH]`` block for the first
    ``n`` positions, supplied by a caller that knows this sequence shares them
    with one it has already built (a branch child and its parent, whose event
    logs agree up to the branch). Those positions are copied rather than
    replayed, which is where most of the work is: in a branching tree the great
    majority of every leaf's tokens are prefix a sibling already produced.
    """

    round_events = _round_events(events)
    prefix_len = config.prefix_len(hand_size)
    skip = 0 if token_prefix is None else int(token_prefix.shape[0])
    if skip >= prefix_len:
        tokens: list = [None] * prefix_len
    else:
        tokens = prefix_tokens(
            config, observer, num_players, hand_size, initial_hand, bidding_start_player
        )
    remaining_by_event: dict[int, list[Card]] = {}
    remaining = set(initial_hand)
    for index, event in enumerate(round_events):
        if event.type == EventType.PLAY and event.player == observer:
            if event.card not in remaining:
                raise ValueError("Observer played a card absent from its hand.")
            remaining.remove(event.card)
        elif event.type == EventType.TRICK_WIN:
            remaining_by_event[index] = list(remaining)

    for offset, (kind, index) in enumerate(token_layout(config, round_events)):
        if prefix_len + offset < skip:
            tokens.append(None)      # comes from token_prefix; never read below
            continue
        event = round_events[index]
        if kind == "turn":
            tokens.append(
                _turn_row(config, event, observer, num_players, hand_size)
            )
            continue
        token = event_token(
            config,
            event,
            observer,
            num_players,
            hand_size,
            remaining_hand=remaining_by_event.get(index),
        )
        if token is None:
            raise AssertionError("Layout emitted an event with no token.")
        tokens.append(token)

    # Look-ahead next-actor annotation: token t announces the actor of the
    # action event at t+1 (TRICK_WIN and HAND interiors announce nothing).
    # With TURN tokens on, the token before an action is the TURN token, which
    # already carries the same annotation -- this loop just re-derives it.
    #
    # Starts at ``skip`` when a prefix was supplied: position skip-1 is the last
    # copied token, and its annotation is already correct because it describes
    # the *actor and phase* of the event at skip, which a branch child shares
    # with its parent -- only the action taken differs, not who takes it.
    for position in range(max(prefix_len - 1, skip), len(tokens) - 1):
        upcoming = tokens[position + 1]
        if upcoming[SLOT_TYPE] in (TOKEN_BID, TOKEN_PLAY):
            tokens[position][SLOT_NEXT_ACTOR] = upcoming[SLOT_REL_PLAYER]
            tokens[position][SLOT_NEXT_PHASE] = (
                NEXT_BID if upcoming[SLOT_TYPE] == TOKEN_BID else NEXT_PLAY
            )
        elif upcoming[SLOT_TYPE] != TOKEN_TURN:
            tokens[position][SLOT_NEXT_ACTOR] = config.player_na_id
            tokens[position][SLOT_NEXT_PHASE] = NEXT_NONE
    if pending_actor is not None:
        rel = _relative(pending_actor, observer, num_players)
        if turn_token_for_phase(config, pending_phase):
            # An in-progress game stops one token short of the decision: the
            # actor's TURN token is the position the policy head reads. Its
            # trick fields come off the event tail so this matches byte for
            # byte what the rollout wave loop appends.
            trick = pos = None
            if pending_phase == NEXT_PLAY:
                trick = sum(
                    1 for e in round_events if e.type == EventType.TRICK_WIN
                )
                pos = 0
                for event in reversed(round_events):
                    if event.type == EventType.TRICK_WIN:
                        break
                    if event.type == EventType.PLAY:
                        pos += 1
            tokens.append(
                turn_token(
                    config, num_players, hand_size, rel, pending_phase, trick, pos
                )
            )
        else:
            tokens[-1][SLOT_NEXT_ACTOR] = rel
            tokens[-1][SLOT_NEXT_PHASE] = pending_phase
    if skip == 0:
        return np.asarray(tokens, dtype=np.int64)
    array = np.empty((len(tokens), TOKEN_WIDTH), dtype=np.int64)
    array[:skip] = token_prefix
    if skip < len(tokens):
        # Guarded: an empty tail is a list, which numpy reads as shape (0,) and
        # refuses to broadcast into (0, TOKEN_WIDTH).
        array[skip:] = tokens[skip:]
    return array


def build_oracle_tokens(
    config: SeqModelConfig,
    events: list[GameEvent],
    num_players: int,
    hand_size: int,
    initial_hands: dict[int, list[Card]],
    bidding_start_player: int,
) -> np.ndarray:
    """One canonical, perfect-information critic sequence for a whole game.

    Seat ids are absolute within the environment: input owner/actor id ``s``
    and value output column ``s`` therefore refer to exactly the same player.
    Every dealt card remains a separate prefix token; no hand or deal pooling
    occurs. The public suffix is the ordinary seat-0 stream, whose relative
    player ids equal these canonical absolute ids.
    """

    if set(initial_hands) != set(range(num_players)):
        raise ValueError("initial_hands must cover every active oracle seat.")
    for owner, cards in initial_hands.items():
        if len(cards) != hand_size:
            raise ValueError(f"Player {owner} hand does not match hand size.")

    canonical = build_seat_tokens(
        config,
        events,
        observer=0,
        num_players=num_players,
        hand_size=hand_size,
        initial_hand=initial_hands[0],
        bidding_start_player=bidding_start_player,
    )
    cards = [
        oracle_hand_token(config, num_players, hand_size, owner, card)
        for owner in range(num_players)
        for card in sort_cards(initial_hands[owner])
    ]
    # Drop seat 0's ordinary N-card prefix and insert the complete P*N prefix.
    oracle = np.concatenate(
        (
            canonical[:1],
            np.asarray(cards, dtype=np.int64),
            canonical[1 + hand_size :],
        ),
        axis=0,
    )
    # The actor's TRICK_WIN row repeats its observer's remaining hand. That is
    # redundant in a full-information stream and would privilege canonical
    # seat 0, so the oracle derives current hands from initial cards + plays.
    oracle[:, SLOT_REMAINING_HAND_START:] = NUM_CARDS
    if config.turn_token == "off":
        # In the actor stream this annotation lived on seat 0's last HAND row,
        # which was replaced above. Restore it on the final oracle card token.
        oracle[num_players * hand_size, SLOT_NEXT_ACTOR] = bidding_start_player
        oracle[num_players * hand_size, SLOT_NEXT_PHASE] = NEXT_BID
    expected = config.oracle_seq_len(num_players, hand_size)
    if oracle.shape != (expected, TOKEN_WIDTH):
        raise AssertionError(
            f"Oracle sequence has shape {oracle.shape}, expected "
            f"{(expected, TOKEN_WIDTH)}."
        )
    return oracle


@dataclass
class ReplayArrays:
    """Tokens plus per-position labels for one focal seat of one leaf."""

    tokens: np.ndarray            # [L, TOKEN_WIDTH] int64
    decision_positions: np.ndarray  # [D] int64 — h_t predicts the observer's action
    decision_phases: np.ndarray   # [D] int64 — NEXT_BID or NEXT_PLAY
    legal_bid_masks: np.ndarray   # [D, bid_count] bool
    legal_card_masks: np.ndarray  # [D, 52] bool
    action_targets: np.ndarray    # [D] int64 — bid value or card id actually taken
    trick_targets: np.ndarray     # [max_players] int64, IGNORE_LABEL padding
    trick_masks: np.ndarray       # [L, max_players, bid_count] bool
    suit_targets: np.ndarray      # [L, belief_opponents, 4], column j is rel seat j+1
    bid_hit_targets: np.ndarray   # [max_players] int64 in {0,1}, IGNORE_LABEL padding
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
    tokens: np.ndarray | None = None,
    suit_labels: bool = True,
    trick_labels: bool = True,
) -> ReplayArrays:
    """Replay a completed round and emit tokens plus per-position labels.

    Labels at position t describe the game state after the event at t (prefix
    positions use the post-deal state). Truth (opponents' hands) is used only
    for targets; the token stream itself stays observer-visible.

    ``label_from`` restricts label filling and decision records to positions
    >= label_from — a leaf that owns only its post-branch suffix skips the
    expensive label work for prefix positions owned by another leaf.

    ``tokens`` supplies an already-built token array, for callers that built it
    themselves to reuse a sibling's prefix (see ``build_seat_tokens``). The
    event walk below still runs in full either way: the replay state it carries
    (holders, suit counts, tricks won) has to reach ``label_from`` before any
    label can be written, so only the token construction is skippable.

    ``suit_labels`` / ``trick_labels`` both default to on. A caller whose loss
    weights one of these at zero can turn it off to skip building targets
    nothing will read. The arrays are still returned at full shape, left at
    IGNORE_LABEL / False, so a loss enabled against labels that were not built
    sees no labeled positions rather than wrong ones. Bid hit has no gate: it is
    one comparison per seat at the end of the walk, with no per-position work to
    skip.
    """

    if set(initial_hands) != set(range(num_players)):
        raise ValueError("initial_hands must cover every player.")
    for player, hand in initial_hands.items():
        if len(hand) != hand_size:
            raise ValueError(f"Player {player} hand does not match hand size.")

    if tokens is None:
        tokens = build_seat_tokens(
            config,
            events,
            observer,
            num_players,
            hand_size,
            initial_hands[observer],
            bidding_start_player,
        )
    total_len = config.seq_len(num_players, hand_size)
    if tokens.shape[0] != total_len:
        raise ValueError(
            f"Replay expects a completed round: {tokens.shape[0]} tokens != {total_len}."
        )

    bid_count = config.bid_count
    trick_masks = np.zeros((total_len, config.max_players, bid_count), dtype=bool)
    suit_targets = np.full(
        (total_len, config.belief_opponents, len(SUITS)), IGNORE_LABEL, dtype=np.int64
    )

    # Vectorized replay state (rel-indexed): holder_rel[card] is the current
    # holder's relative seat, ``num_players`` for the undealt kitty, -1 once
    # publicly played. It exists to catch a replay that has drifted from the
    # real game -- see the non-holder assertion below.
    holder_rel = np.full(NUM_CARDS, num_players, dtype=np.int64)
    suit_count_rel = np.zeros((num_players, len(SUITS)), dtype=np.int64)
    for player, hand in initial_hands.items():
        rel = (player - observer) % num_players
        for card in hand:
            holder_rel[card_id(card)] = rel
            suit_count_rel[rel, SUITS.index(card.suit)] += 1
    tricks_won_rel = np.zeros(num_players, dtype=np.int64)
    observer_hand: set[Card] = set(initial_hands[observer])
    bid_values: list[int] = []
    # bid_values is in bidding order (the forbidden-bid rule needs that); this is
    # the same bids keyed by relative seat, for the end-of-round hit comparison.
    bid_rel = np.full(num_players, -1, dtype=np.int64)
    completed_tricks = 0
    current_led_suit: Suit | None = None
    current_trick_size = 0
    count_range = np.arange(bid_count)

    def fill_labels(position: int) -> None:
        if trick_labels:
            unresolved = hand_size - completed_tricks
            upper = np.minimum(tricks_won_rel + unresolved, config.max_hand_size)
            trick_masks[position, :num_players] = (
                count_range[None, :] >= tricks_won_rel[:, None]
            ) & (count_range[None, :] <= upper[:, None])
        if suit_labels:
            # Opponents only, so column j is relative seat j + 1. The observer's
            # own suits (rel 0) are not a belief: its dealt hand and every card
            # it has played are both in its own token stream, so the answer is
            # already in the prefix verbatim.
            suit_targets[position, : num_players - 1] = (
                suit_count_rel[1:num_players] > 0
            ).astype(np.int64)

    prefix_len = config.prefix_len(hand_size)
    if label_from < prefix_len:
        first = max(label_from, 0)
        fill_labels(first)
        for position in range(first + 1, prefix_len):
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
    # Walk events and token positions together. The two can drift apart in both
    # directions: a TURN token is a position with no event, and a TRICK_WIN
    # with the token disabled is an event with no position (its state change
    # still has to land, and shows up in the next token's labels).
    position = prefix_len - 1
    for index, event in enumerate(round_events):
        if index > 0 and turn_token_precedes(config, event):
            position += 1
            if position >= label_from:
                fill_labels(position)
        tokenized = emits_token(config, event)
        if tokenized:
            # The readout for an action sits on the token immediately before
            # it -- the TURN token when TURN tokens are on. Record before
            # applying the event so the legal mask is the pre-action one.
            record_decision(position, event)
            position += 1
        if event.type == EventType.BID:
            bid_values.append(event.bid)
            bid_rel[(event.player - observer) % num_players] = event.bid
        elif event.type == EventType.PLAY:
            rel = (event.player - observer) % num_players
            if current_trick_size == 0:
                current_led_suit = event.card.suit
            index_of_card = card_id(event.card)
            if holder_rel[index_of_card] != rel:
                raise AssertionError("Replay saw a play from a non-holder.")
            holder_rel[index_of_card] = -1
            suit_count_rel[rel, SUITS.index(event.card.suit)] -= 1
            if event.player == observer:
                observer_hand.remove(event.card)
            current_trick_size += 1
        elif event.type == EventType.TRICK_WIN:
            tricks_won_rel[(event.player - observer) % num_players] += 1
            completed_tricks += 1
            current_trick_size = 0
            current_led_suit = None
        if tokenized and position >= label_from:
            fill_labels(position)

    if position + 1 != total_len:
        raise AssertionError(
            f"Replay walked {position + 1} token positions, expected {total_len}."
        )
    if completed_tricks != hand_size:
        raise ValueError("Replay expects a completed round.")

    trick_targets = np.full(config.max_players, IGNORE_LABEL, dtype=np.int64)
    trick_targets[:num_players] = tricks_won_rel

    if (bid_rel < 0).any():
        raise ValueError("Replay expects every player to have bid.")
    # One boolean per seat for the whole round: the label is the outcome, so it
    # is the same at every position. Positions before a seat has bid are labeled
    # too -- there the question is "will this seat hit whatever it ends up
    # bidding", which is a genuine forecast, not leakage.
    bid_hit_targets = np.full(config.max_players, IGNORE_LABEL, dtype=np.int64)
    bid_hit_targets[:num_players] = (tricks_won_rel == bid_rel).astype(np.int64)

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
        trick_targets=trick_targets,
        trick_masks=trick_masks,
        suit_targets=suit_targets,
        bid_hit_targets=bid_hit_targets,
        final_tricks_won={
            (observer + rel) % num_players: int(tricks_won_rel[rel])
            for rel in range(num_players)
        },
    )
