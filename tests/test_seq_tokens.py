"""Schema-v6 token/label builder tests, cross-checked against the v4 encoder."""

from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest

from plump.env import PlumpEnv
from plump.modeling.encoding import ModelConfig
from plump.modeling.encoding import card_id as v4_card_id
from plump.modeling.encoding import encode_observation
from plump.rounds import RoundSpec, round_game_config
from plump.seq import tokens as seq_tokens
from plump.seq.config import (
    NEXT_BID,
    NEXT_NONE,
    NEXT_PLAY,
    SLOT_BID,
    SLOT_CARD,
    SLOT_NEXT_ACTOR,
    SLOT_NEXT_PHASE,
    SLOT_NUM_PLAYERS,
    SLOT_HAND_SIZE,
    SLOT_REL_PLAYER,
    SLOT_TYPE,
    TOKEN_BID,
    TOKEN_GAME,
    TOKEN_HAND,
    TOKEN_PLAY,
    TOKEN_TRICK_WIN,
    TOKEN_TURN,
    SeqModelConfig,
    seq_len,
)
from plump.state import EventType
from plump.training.common import (
    suit_presence_targets_relative,
)

CONFIG = SeqModelConfig()
V4_CONFIG = ModelConfig()


def play_random_round(num_players, hand_size, seed, observer):
    """Play one full round; capture v4-encoded oracles at observer decisions."""

    spec = RoundSpec(num_players, hand_size)
    start = random.Random(seed).randrange(num_players)
    env = PlumpEnv(round_game_config(spec, bidding_start_player=start), seed=seed)
    env.reset(seed=seed)
    rng = random.Random(seed + 1)
    oracles = []
    while not env.is_done():
        if env.current_player() == observer:
            observation = env.get_observation(observer)
            encoded = encode_observation(observation)
            oracles.append(
                {
                    "encoded": encoded,
                    "suit_targets": suit_presence_targets_relative(
                        env, observer, V4_CONFIG
                    ),
                    # The v4 oracle masks the observer's own row; seq labels it.
                    "own_suit_targets": [
                        int(
                            any(
                                card.suit == suit
                                for card in env.state.current_round.current_hands[
                                    observer
                                ]
                            )
                        )
                        for suit in seq_tokens.SUITS
                    ],
                }
            )
        env.step(rng.choice(env.legal_actions()))
    return env, oracles, start


def replay_arrays_for(env, observer, start, label_from=0, config=None):
    round_state = env.state.current_round
    return seq_tokens.build_replay_arrays(
        config or CONFIG,
        round_state.initial_hands,
        env.state.event_log,
        observer,
        env.config.num_players,
        round_state.hand_size,
        start,
        label_from=label_from,
    )


CASES = [(3, 3, 0), (3, 10, 1), (4, 5, 2), (5, 8, 3), (5, 10, 4)]


def test_card_id_matches_v4_encoding():
    for index in range(52):
        card = seq_tokens.card_from_id(index)
        assert seq_tokens.card_id(card) == index
        assert v4_card_id(card) == index


@pytest.mark.parametrize("num_players,hand_size,seed", CASES)
def test_sequence_structure(num_players, hand_size, seed):
    env, _, start = play_random_round(num_players, hand_size, seed, observer=0)
    for observer in range(num_players):
        arrays = replay_arrays_for(env, observer, start)
        tokens = arrays.tokens
        assert tokens.shape == (seq_len(num_players, hand_size), seq_tokens.TOKEN_WIDTH)

        vocab = CONFIG.slot_vocab_sizes
        for slot in range(seq_tokens.TOKEN_WIDTH):
            assert tokens[:, slot].min() >= 0
            assert tokens[:, slot].max() < vocab[slot]
        assert (tokens[:, SLOT_HAND_SIZE] == hand_size).all()
        assert (tokens[:, SLOT_NUM_PLAYERS] == num_players).all()

        assert tokens[0, SLOT_TYPE] == TOKEN_GAME
        assert (tokens[1 : 1 + hand_size, SLOT_TYPE] == TOKEN_HAND).all()
        bid_rows = tokens[1 + hand_size : 1 + hand_size + num_players]
        assert (bid_rows[:, SLOT_TYPE] == TOKEN_BID).all()
        for trick in range(hand_size):
            base = 1 + hand_size + num_players + trick * (num_players + 1)
            trick_rows = tokens[base : base + num_players + 1]
            assert (trick_rows[:num_players, SLOT_TYPE] == TOKEN_PLAY).all()
            assert trick_rows[num_players, SLOT_TYPE] == TOKEN_TRICK_WIN


@pytest.mark.parametrize("num_players,hand_size,seed", CASES)
def test_decision_positions_and_next_actor(num_players, hand_size, seed):
    for observer in range(num_players):
        env, _, start = play_random_round(num_players, hand_size, seed, observer)
        arrays = replay_arrays_for(env, observer, start)
        tokens = arrays.tokens

        assert len(arrays.decision_positions) == hand_size + 1
        annotated = np.flatnonzero(
            (tokens[:, SLOT_NEXT_ACTOR] == 0)
            & (tokens[:, SLOT_NEXT_PHASE] != NEXT_NONE)
        )
        assert np.array_equal(annotated, arrays.decision_positions)
        for position, phase in zip(arrays.decision_positions, arrays.decision_phases):
            upcoming = tokens[position + 1]
            assert upcoming[SLOT_REL_PLAYER] == 0
            if phase == NEXT_BID:
                assert upcoming[SLOT_TYPE] == TOKEN_BID
            else:
                assert phase == NEXT_PLAY
                assert upcoming[SLOT_TYPE] == TOKEN_PLAY


@pytest.mark.parametrize("num_players,hand_size,seed", CASES)
def test_action_targets_match_taken_actions(num_players, hand_size, seed):
    for observer in range(num_players):
        env, _, start = play_random_round(num_players, hand_size, seed, observer)
        arrays = replay_arrays_for(env, observer, start)
        tokens = arrays.tokens
        for index, position in enumerate(arrays.decision_positions):
            action_row = tokens[position + 1]
            if arrays.decision_phases[index] == NEXT_BID:
                assert arrays.action_targets[index] == action_row[SLOT_BID]
                assert arrays.legal_bid_masks[index][arrays.action_targets[index]]
            else:
                assert arrays.action_targets[index] == action_row[SLOT_CARD]
                assert arrays.legal_card_masks[index][arrays.action_targets[index]]


@pytest.mark.parametrize("num_players,hand_size,seed", CASES)
def test_labels_match_v4_encoder_at_decisions(num_players, hand_size, seed):
    for observer in range(num_players):
        env, oracles, start = play_random_round(num_players, hand_size, seed, observer)
        arrays = replay_arrays_for(env, observer, start)
        assert len(oracles) == len(arrays.decision_positions)
        for index, oracle in enumerate(oracles):
            position = arrays.decision_positions[index]
            encoded = oracle["encoded"]
            assert np.array_equal(
                arrays.legal_bid_masks[index], np.asarray(encoded.legal_bid_mask)
            )
            assert np.array_equal(
                arrays.legal_card_masks[index], np.asarray(encoded.legal_card_mask)
            )
            assert np.array_equal(
                arrays.trick_masks[position],
                np.asarray(encoded.final_trick_count_mask),
            )
            assert np.array_equal(
                arrays.suit_targets[position, 1:],
                np.asarray(oracle["suit_targets"])[1:],
            )
            assert np.array_equal(
                arrays.suit_targets[position, 0],
                np.asarray(oracle["own_suit_targets"], dtype=np.int64),
            )


@pytest.mark.parametrize("num_players,hand_size,seed", CASES)
def test_label_invariants_at_every_position(num_players, hand_size, seed):
    observer = seed % num_players
    env, _, start = play_random_round(num_players, hand_size, seed, observer)
    arrays = replay_arrays_for(env, observer, start)
    round_state = env.state.current_round

    bid_of = {bid.player: bid.value for bid in round_state.bids}
    for rel in range(num_players):
        player = (observer + rel) % num_players
        assert arrays.trick_targets[rel] == round_state.tricks_won[player]
        assert arrays.bid_hit_targets[rel] == int(
            round_state.tricks_won[player] == bid_of[player]
        )
    assert (arrays.trick_targets[num_players:] == seq_tokens.IGNORE_LABEL).all()
    assert (arrays.bid_hit_targets[num_players:] == seq_tokens.IGNORE_LABEL).all()

    total_len = arrays.tokens.shape[0]
    for position in range(total_len):
        for rel in range(num_players):
            # The realized final trick count is always inside the mask.
            assert arrays.trick_masks[position, rel, arrays.trick_targets[rel]]


@pytest.mark.parametrize("label_from", [0, 10, 25])
def test_label_from_matches_full_build(label_from):
    env, _, start = play_random_round(4, 6, seed=5, observer=2)
    full = replay_arrays_for(env, 2, start)
    partial = replay_arrays_for(env, 2, start, label_from=label_from)
    assert np.array_equal(full.tokens, partial.tokens)
    total_len = full.tokens.shape[0]
    for position in range(label_from, total_len):
        assert np.array_equal(full.trick_masks[position], partial.trick_masks[position])
        assert np.array_equal(
            full.suit_targets[position], partial.suit_targets[position]
        )
    kept = full.decision_positions >= label_from
    assert np.array_equal(full.decision_positions[kept], partial.decision_positions)
    assert np.array_equal(full.action_targets[kept], partial.action_targets)


def test_tokens_do_not_depend_on_hidden_hands():
    env, _, start = play_random_round(4, 5, seed=11, observer=0)
    round_state = env.state.current_round
    arrays = replay_arrays_for(env, 0, start)
    direct = seq_tokens.build_seat_tokens(
        CONFIG,
        env.state.event_log,
        0,
        4,
        5,
        round_state.initial_hands[0],
        start,
    )
    assert np.array_equal(arrays.tokens, direct)


def test_incomplete_round_raises():
    env = PlumpEnv(round_game_config(RoundSpec(4, 5), bidding_start_player=0), seed=3)
    env.reset(seed=3)
    rng = random.Random(4)
    for _ in range(6):
        env.step(rng.choice(env.legal_actions()))
    round_state = env.state.current_round
    with pytest.raises(ValueError):
        seq_tokens.build_replay_arrays(
            CONFIG,
            round_state.initial_hands,
            env.state.event_log,
            0,
            4,
            5,
            0,
        )


SCHEMAS = [
    (trick_win, turn)
    for trick_win in (True, False)
    for turn in ("off", "bid", "all")
]


@pytest.mark.parametrize("trick_win_token,turn_token", SCHEMAS)
@pytest.mark.parametrize("num_players,hand_size,seed", CASES)
def test_schema_flags_keep_the_stream_and_the_labels_aligned(
    trick_win_token, turn_token, num_players, hand_size, seed
):
    """seq_len, the token stream and the decision positions must agree.

    The flags move every token after the first bid, so an off-by-one between
    the layout the stream is built from and the one the labeller walks would
    hand the policy head a row belonging to a different position.
    """

    config = replace(
        CONFIG, trick_win_token=trick_win_token, turn_token=turn_token
    )
    env, _, start = play_random_round(num_players, hand_size, seed, observer=0)
    for observer in range(num_players):
        arrays = replay_arrays_for(env, observer, start, config=config)
        tokens = arrays.tokens
        length = config.seq_len(num_players, hand_size)
        assert tokens.shape == (length, seq_tokens.TOKEN_WIDTH)

        vocab = config.slot_vocab_sizes
        for slot in range(seq_tokens.TOKEN_WIDTH):
            assert tokens[:, slot].min() >= 0
            assert tokens[:, slot].max() < vocab[slot]

        types = tokens[:, SLOT_TYPE]
        assert (types == TOKEN_TRICK_WIN).sum() == (
            hand_size if trick_win_token else 0
        )
        expected_turns = {
            "off": 0, "bid": num_players, "all": num_players * (hand_size + 1)
        }[turn_token]
        assert (types == TOKEN_TURN).sum() == expected_turns

        # The observer decides once per bid and once per trick, and each
        # decision is read off the token immediately before its action.
        assert len(arrays.decision_positions) == hand_size + 1
        for position, target, phase in zip(
            arrays.decision_positions,
            arrays.action_targets,
            arrays.decision_phases,
        ):
            action = tokens[position + 1]
            assert action[SLOT_REL_PLAYER] == 0
            if phase == NEXT_BID:
                assert action[SLOT_TYPE] == TOKEN_BID
                assert action[SLOT_BID] == target
            else:
                assert action[SLOT_TYPE] == TOKEN_PLAY
                assert action[SLOT_CARD] == target
            if turn_token == "all" or (turn_token == "bid" and phase == NEXT_BID):
                assert tokens[position, SLOT_TYPE] == TOKEN_TURN
                assert tokens[position, SLOT_REL_PLAYER] == 0
        assert (tokens[arrays.decision_positions, SLOT_NEXT_ACTOR] == 0).all()


@pytest.mark.parametrize("num_players,hand_size,seed", CASES)
def test_token_prefix_reuse_matches_a_full_build(num_players, hand_size, seed):
    """A supplied prefix must reproduce the from-scratch tokens exactly.

    This is the invariant the update relies on when a branch child copies its
    parent's tokens instead of replaying the shared events: everything before
    the branch is identical, including the next-actor annotation on the last
    copied token, which describes who acts at the branch rather than what they
    do there.
    """

    env, _, start = play_random_round(num_players, hand_size, seed, observer=0)
    round_state = env.state.current_round
    args = (
        CONFIG,
        env.state.event_log,
        0,
        num_players,
        hand_size,
        round_state.initial_hands[0],
        start,
    )
    scratch = seq_tokens.build_seat_tokens(*args)
    for split in range(0, len(scratch) + 1):
        reused = seq_tokens.build_seat_tokens(
            *args, token_prefix=scratch[:split]
        )
        assert np.array_equal(reused, scratch), f"prefix reuse differs at {split}"


@pytest.mark.parametrize("num_players,hand_size,seed", CASES)
def test_disabled_label_groups_stay_unlabelled_and_do_not_disturb_the_rest(
    num_players, hand_size, seed
):
    """Turning a label group off must leave it at IGNORE_LABEL/False and change
    nothing else -- a loss enabled against it then sees no labelled positions
    rather than wrong ones."""

    env, _, start = play_random_round(num_players, hand_size, seed, observer=0)
    full = replay_arrays_for(env, 0, start)
    round_state = env.state.current_round
    off = seq_tokens.build_replay_arrays(
        CONFIG,
        round_state.initial_hands,
        env.state.event_log,
        0,
        num_players,
        round_state.hand_size,
        start,
        suit_labels=False,
        trick_labels=False,
    )
    assert (off.suit_targets == seq_tokens.IGNORE_LABEL).all()
    assert not off.trick_masks.any()
    # Everything not switched off is untouched. Bid hit has no gate, so it is
    # built either way and must still agree.
    assert np.array_equal(off.tokens, full.tokens)
    assert np.array_equal(off.trick_targets, full.trick_targets)
    assert np.array_equal(off.bid_hit_targets, full.bid_hit_targets)
    assert np.array_equal(off.decision_positions, full.decision_positions)
    assert np.array_equal(off.action_targets, full.action_targets)
    assert np.array_equal(off.legal_card_masks, full.legal_card_masks)
