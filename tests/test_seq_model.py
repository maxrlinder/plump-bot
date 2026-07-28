"""KV-cached decoding must reproduce the full causal forward exactly."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from plump.env import PlumpEnv
from plump.rounds import RoundSpec, round_game_config
from plump.seq import tokens as seq_tokens
from plump.seq.config import SeqModelConfig, seq_len
from plump.seq.model import SeqPlumpModel

ATOL = 1e-4


def small_config(**overrides) -> SeqModelConfig:
    defaults = dict(d_model=64, n_layers=2, n_heads=4, d_ff=128)
    defaults.update(overrides)
    return SeqModelConfig(**defaults)


def seat_tokens(num_players, hand_size, seed, observer) -> np.ndarray:
    start = random.Random(seed).randrange(num_players)
    env = PlumpEnv(
        round_game_config(RoundSpec(num_players, hand_size), bidding_start_player=start),
        seed=seed,
    )
    env.reset(seed=seed)
    rng = random.Random(seed + 1)
    while not env.is_done():
        env.step(rng.choice(env.legal_actions()))
    round_state = env.state.current_round
    return seq_tokens.build_seat_tokens(
        SeqModelConfig(),
        env.state.event_log,
        observer,
        num_players,
        hand_size,
        round_state.initial_hands[observer],
        start,
    )


def token_batch(num_players, hand_size, seeds) -> torch.Tensor:
    rows = [
        seat_tokens(num_players, hand_size, seed, observer=seed % num_players)
        for seed in seeds
    ]
    return torch.from_numpy(np.stack(rows))


@pytest.mark.parametrize("kv_heads", [None, 2])
def test_cached_decode_matches_full_forward(kv_heads):
    torch.manual_seed(0)
    config = small_config(n_kv_heads=kv_heads)
    model = SeqPlumpModel(config).eval()
    num_players, hand_size = 4, 5
    tokens = token_batch(num_players, hand_size, seeds=[0, 1, 2])
    length = seq_len(num_players, hand_size)
    assert tokens.shape[1] == length

    with torch.no_grad():
        full = model.forward_full(tokens, aux_heads=False)
        cache = model.new_cache(capacity=4)
        slots = torch.tensor(cache.alloc(3), dtype=torch.long)
        prefix_len = 1 + hand_size
        step = model.forward_prefill(tokens[:, :prefix_len], cache, slots)
        torch.testing.assert_close(
            step.bid_logits, full.bid_logits[:, prefix_len - 1], atol=ATOL, rtol=0
        )
        for position in range(prefix_len, length):
            step = model.forward_step(tokens[:, position], position, cache, slots)
            torch.testing.assert_close(
                step.bid_logits, full.bid_logits[:, position], atol=ATOL, rtol=0
            )
            torch.testing.assert_close(
                step.card_logits, full.card_logits[:, position], atol=ATOL, rtol=0
            )
            torch.testing.assert_close(
                step.value, full.value[:, position], atol=ATOL, rtol=0
            )


@pytest.mark.parametrize("kv_heads", [None, 2])
@pytest.mark.parametrize("run", [2, 3, 6])
def test_multi_token_append_matches_stepping_one_at_a_time(kv_heads, run):
    """Appending a run of events in one call must equal appending them singly.

    The wave loop merges a trick's completing play with its TRICK_WIN token,
    because only the last event of a run produces a readout. The queries then
    sit at the end of a non-empty prefix, so the mask is rectangular -- query i
    sees the prefix plus the first i tokens of the run, and nothing after.
    """

    torch.manual_seed(0)
    config = small_config(n_kv_heads=kv_heads)
    model = SeqPlumpModel(config).eval()
    num_players, hand_size = 4, 5
    tokens = token_batch(num_players, hand_size, seeds=[0, 1, 2])
    length = seq_len(num_players, hand_size)
    prefix_len = 1 + hand_size

    with torch.no_grad():
        full = model.forward_full(tokens, aux_heads=False)

        single = model.new_cache(capacity=4)
        single_slots = torch.tensor(single.alloc(3), dtype=torch.long)
        model.forward_prefill(tokens[:, :prefix_len], single, single_slots)
        merged = model.new_cache(capacity=4)
        merged_slots = torch.tensor(merged.alloc(3), dtype=torch.long)
        model.forward_prefill(tokens[:, :prefix_len], merged, merged_slots)

        for start in range(prefix_len, length - run + 1, run):
            stop = start + run
            for position in range(start, stop):
                step = model.forward_step(
                    tokens[:, position], position, single, single_slots
                )
            block = model.forward_step(
                tokens[:, start:stop], start, merged, merged_slots
            )
            # The merged call reads the heads at the last position of the run,
            # which is the only readout the wave loop consumes.
            torch.testing.assert_close(
                block.card_logits, step.card_logits, atol=ATOL, rtol=0
            )
            torch.testing.assert_close(
                block.card_logits, full.card_logits[:, stop - 1], atol=ATOL, rtol=0
            )
            torch.testing.assert_close(block.value, step.value, atol=ATOL, rtol=0)
        # And the caches must agree everywhere, not just at the readout: the
        # skipped positions' K/V are what later steps attend to.
        for layer in range(config.n_layers):
            torch.testing.assert_close(
                merged.k[layer], single.k[layer], atol=ATOL, rtol=0
            )
            torch.testing.assert_close(
                merged.v[layer], single.v[layer], atol=ATOL, rtol=0
            )


def test_branch_copy_matches_independent_full_forwards():
    torch.manual_seed(1)
    config = small_config()
    model = SeqPlumpModel(config).eval()
    num_players, hand_size = 3, 4
    length = seq_len(num_players, hand_size)
    prefix_len = 1 + hand_size

    parent_tokens = token_batch(num_players, hand_size, seeds=[7])
    child_tokens = parent_tokens.clone()
    split = prefix_len + 4
    # Divergent but vocabulary-valid suffix for the child branch.
    child_tokens[0, split:, seq_tokens.SLOT_CARD] = (
        child_tokens[0, split:, seq_tokens.SLOT_CARD].flip(0)
    )

    with torch.no_grad():
        full_parent = model.forward_full(parent_tokens, aux_heads=False)
        full_child = model.forward_full(child_tokens, aux_heads=False)

        cache = model.new_cache(capacity=4)
        parent_slot = torch.tensor(cache.alloc(1), dtype=torch.long)
        model.forward_prefill(parent_tokens[:, :prefix_len], cache, parent_slot)
        for position in range(prefix_len, split):
            model.forward_step(parent_tokens[:, position], position, cache, parent_slot)

        child_slot = torch.tensor(cache.alloc(1), dtype=torch.long)
        cache.branch_copy(parent_slot, child_slot, length=split)

        for position in range(split, length):
            parent_step = model.forward_step(
                parent_tokens[:, position], position, cache, parent_slot
            )
            child_step = model.forward_step(
                child_tokens[:, position], position, cache, child_slot
            )
            torch.testing.assert_close(
                parent_step.card_logits,
                full_parent.card_logits[:, position],
                atol=ATOL,
                rtol=0,
            )
            torch.testing.assert_close(
                child_step.card_logits,
                full_child.card_logits[:, position],
                atol=ATOL,
                rtol=0,
            )


def test_aux_head_shapes():
    torch.manual_seed(2)
    config = small_config()
    model = SeqPlumpModel(config).eval()
    tokens = token_batch(4, 3, seeds=[3, 4])
    with torch.no_grad():
        output = model.forward_full(tokens)
    batch, length = tokens.shape[:2]
    assert output.bid_logits.shape == (batch, length, config.bid_count)
    assert output.card_logits.shape == (batch, length, 52)
    assert output.value.shape == (batch, length)
    assert output.trick_logits.shape == (
        batch, length, config.max_players, config.bid_count
    )
    assert output.suit_logits.shape == (batch, length, config.belief_opponents, 4)
    assert output.bid_hit_logits.shape == (batch, length, config.max_players)


def test_aux_heads_are_individually_selectable():
    torch.manual_seed(2)
    model = SeqPlumpModel(small_config()).eval()
    tokens = token_batch(4, 3, seeds=[3, 4])
    with torch.no_grad():
        subset = model.forward_full(tokens, aux_heads={"suit", "bid_hit"})
        none = model.forward_full(tokens, aux_heads=False)
    # Unrequested heads come back None, not zeros, so a loss reading one it did
    # not ask for raises rather than silently training on nothing.
    assert subset.suit_logits is not None
    assert subset.bid_hit_logits is not None
    assert subset.trick_logits is None
    assert none.suit_logits is none.bid_hit_logits is None
    with pytest.raises(ValueError):
        model.forward_full(tokens, aux_heads={"beliefs"})


def test_cache_alloc_free_bookkeeping():
    config = small_config()
    model = SeqPlumpModel(config)
    cache = model.new_cache(capacity=4)
    slots = cache.alloc(3)
    assert cache.free_count == 1
    with pytest.raises(RuntimeError):
        cache.alloc(2)
    cache.free(slots[:2])
    assert cache.free_count == 3
    cache.reset()
    assert cache.free_count == 4
