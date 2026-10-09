"""KV-cached decoding must reproduce the full causal forward exactly."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from plump.env import PlumpEnv
from plump.rounds import RoundSpec, round_game_config
from plump.seq import tokens as seq_tokens
from plump.seq.config import (
    BASE_TOKEN_WIDTH,
    CORE_TOKEN_WIDTH,
    SLOT_REMAINING_HAND_START,
    SLOT_NA_PATTERN,
    SLOT_TYPE,
    TOKEN_TRICK_WIN,
    TOKEN_WIDTH,
    SeqModelConfig,
    seq_len,
)
from plump.seq.model import (
    RANK_BOUNDARY_HEAD_KEYS,
    STRUCTURED_CARD_OUTPUT_KEYS,
    SeqPlumpModel,
    SeqPPOOracleCritic,
    load_seq_model_state_dict,
)

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


def test_structured_card_output_starts_as_the_exact_former_head():
    torch.manual_seed(0)
    model = SeqPlumpModel(small_config())
    tokens = token_batch(4, 3, seeds=[0, 1])

    assert torch.count_nonzero(model.card_rank_output_embedding.weight) == 0
    assert torch.count_nonzero(model.card_suit_output_embedding.weight) == 0
    output = model.forward_full(tokens, aux_heads=False)
    former_logits = torch.nn.functional.linear(
        output.hidden,
        model.card_head.weight,
        model.card_head.bias,
    )
    torch.testing.assert_close(output.card_logits, former_logits)


def test_effective_card_output_adds_exact_rank_and_suit_rows():
    torch.manual_seed(0)
    model = SeqPlumpModel(small_config())
    card = 2 * 13 + 3
    with torch.no_grad():
        model.card_rank_output_embedding.weight[3].fill_(0.25)
        model.card_suit_output_embedding.weight[2].fill_(-0.10)

    effective = model.effective_card_output_weight()
    torch.testing.assert_close(
        effective[card],
        model.card_head.weight[card] + 0.15,
    )
    torch.testing.assert_close(
        effective[card + 1],
        model.card_head.weight[card + 1] - 0.10,
    )
    torch.testing.assert_close(
        effective[card - 13],
        model.card_head.weight[card - 13] + 0.25,
    )


def test_trick_win_embedding_adds_each_remaining_card_input_direction():
    torch.manual_seed(0)
    model = SeqPlumpModel(small_config())
    empty = torch.zeros((1, 1, TOKEN_WIDTH), dtype=torch.long)
    empty[..., SLOT_TYPE] = TOKEN_TRICK_WIN
    empty[..., SLOT_REMAINING_HAND_START:SLOT_NA_PATTERN] = (
        model.config.card_na_id
    )
    with_cards = empty.clone()
    with_cards[0, 0, SLOT_REMAINING_HAND_START : SLOT_REMAINING_HAND_START + 2] = (
        torch.tensor([0, 27])
    )

    baseline = model.embed(empty)
    actual = model.embed(with_cards)
    expected = (
        baseline
        + model.effective_card_input_weight()[[0, 27]].sum(dim=0)
    )
    torch.testing.assert_close(actual, expected)


def test_fused_embedding_bag_matches_explicit_masked_gather_and_gradient():
    torch.manual_seed(3)
    indices = torch.randint(17, (4, 7, 12))
    valid = torch.rand((4, 7, 12)) > 0.25
    explicit_weight = torch.randn(17, 23, requires_grad=True)
    fused_weight = explicit_weight.detach().clone().requires_grad_(True)

    explicit = (
        torch.nn.functional.embedding(indices, explicit_weight)
        * valid.unsqueeze(-1)
    ).sum(dim=-2)
    fused = SeqPlumpModel._embedding_bag_sum(
        indices,
        fused_weight,
        valid=valid,
    )
    torch.testing.assert_close(fused, explicit, rtol=1e-5, atol=1e-6)

    probe = torch.randn_like(explicit)
    (explicit * probe).sum().backward()
    (fused * probe).sum().backward()
    torch.testing.assert_close(
        fused_weight.grad,
        explicit_weight.grad,
        rtol=1e-5,
        atol=1e-6,
    )


def test_na_pattern_initialization_reproduces_legacy_sentinel_sum():
    torch.manual_seed(7)
    model = SeqPlumpModel(small_config())
    tokens = token_batch(4, 3, seeds=[0, 1])[:, : model.config.prefix_len(3)]
    core = tokens[..., :CORE_TOKEN_WIDTH]
    pressure = tokens[..., CORE_TOKEN_WIDTH:BASE_TOKEN_WIDTH]
    legacy = model._embedding_bag_sum(
        core + model.slot_offsets, model.slot_embedding.weight
    ) + model._embedding_bag_sum(
        pressure + model.bid_pressure_offsets,
        model.bid_pressure_embedding.weight,
    )
    current = model.embed(tokens) - model.pos_embedding(
        torch.arange(tokens.shape[1])
    )
    torch.testing.assert_close(current, legacy, rtol=1e-5, atol=1e-6)


def test_individual_na_rows_receive_no_gradient_after_pattern_refactor():
    torch.manual_seed(8)
    model = SeqPlumpModel(
        small_config(per_layer_embedding_dim=16, value_embedding_mode="all")
    )
    tokens = token_batch(4, 3, seeds=[0, 1])
    model.forward_full(tokens, aux_heads=False).bid_logits.sum().backward()

    for table, offsets in (
        (model.slot_embedding, model.slot_offsets),
        (model.bid_pressure_embedding, model.bid_pressure_offsets),
        (model.per_layer_slot_embedding, model.conditioning_slot_offsets),
        (model.value_slot_embedding, model.conditioning_slot_offsets),
    ):
        assert table is not None
        start = 0 if table is not model.bid_pressure_embedding else CORE_TOKEN_WIDTH
        for local_slot, offset in enumerate(offsets):
            slot = start + local_slot
            sentinel = model.config.base_slot_na_ids[slot]
            if sentinel is not None:
                assert torch.count_nonzero(table.weight.grad[int(offset) + sentinel]) == 0

    assert model.na_pattern_embedding.weight.grad.abs().sum() > 0
    assert model.per_layer_na_pattern_embedding.weight.grad.abs().sum() > 0
    assert model.value_na_pattern_embedding.weight.grad.abs().sum() > 0


def test_pre_structured_model_state_loads_with_zero_shared_output_rows():
    torch.manual_seed(0)
    original = SeqPlumpModel(small_config())
    old_state = {
        key: value
        for key, value in original.state_dict().items()
        if key not in STRUCTURED_CARD_OUTPUT_KEYS
    }

    torch.manual_seed(99)
    loaded = SeqPlumpModel(small_config())
    assert load_seq_model_state_dict(loaded, old_state)
    assert torch.count_nonzero(loaded.card_rank_output_embedding.weight) == 0
    assert torch.count_nonzero(loaded.card_suit_output_embedding.weight) == 0
    for key, expected in old_state.items():
        torch.testing.assert_close(loaded.state_dict()[key], expected)


def test_pre_rank_boundary_state_keeps_new_head_randomly_initialized():
    torch.manual_seed(0)
    original = SeqPlumpModel(small_config())
    old_state = {
        key: value
        for key, value in original.state_dict().items()
        if key not in RANK_BOUNDARY_HEAD_KEYS
    }

    torch.manual_seed(99)
    loaded = SeqPlumpModel(small_config())
    initial = {
        key: loaded.state_dict()[key].clone()
        for key in RANK_BOUNDARY_HEAD_KEYS
    }
    assert load_seq_model_state_dict(loaded, old_state) == RANK_BOUNDARY_HEAD_KEYS
    for key in RANK_BOUNDARY_HEAD_KEYS:
        torch.testing.assert_close(loaded.state_dict()[key], initial[key])
    assert torch.count_nonzero(loaded.rank_boundary_head.weight) > 0


def test_zero_initialized_layer_extensions_preserve_the_former_forward_exactly():
    torch.manual_seed(4)
    legacy = SeqPlumpModel(small_config(turn_token="bid"))
    state = legacy.state_dict()
    extended = SeqPlumpModel(
        small_config(
            turn_token="bid",
            per_layer_embedding_dim=16,
            value_embedding_mode="alternating",
        )
    )
    load_seq_model_state_dict(extended, state)
    tokens = token_batch(4, 3, seeds=[0, 1])

    legacy.eval()
    extended.eval()
    with torch.no_grad():
        before = legacy.forward_full(tokens, aux_heads=False)
        after = extended.forward_full(tokens, aux_heads=False)
    assert torch.equal(before.hidden, after.hidden)
    assert torch.equal(before.bid_logits, after.bid_logits)
    assert torch.equal(before.card_logits, after.card_logits)


def test_layer_extensions_receive_gradient_from_the_first_backward():
    torch.manual_seed(5)
    model = SeqPlumpModel(
        small_config(
            turn_token="bid",
            per_layer_embedding_dim=16,
            value_embedding_mode="alternating",
        )
    )
    tokens = token_batch(4, 3, seeds=[0, 1])
    model.forward_full(tokens, aux_heads=False).bid_logits.sum().backward()

    assert model.bid_pressure_embedding.weight.grad.abs().sum() > 0
    assert model.per_layer_slot_embedding.weight.grad.abs().sum() > 0
    assert model.per_layer_context_projection.weight.grad.abs().sum() > 0
    assert model.value_slot_embedding.weight.grad.abs().sum() > 0


def test_extended_cached_decode_matches_full_after_extensions_are_nonzero():
    torch.manual_seed(6)
    config = small_config(
        n_kv_heads=2,
        turn_token="bid",
        per_layer_embedding_dim=16,
        value_embedding_mode="all",
    )
    model = SeqPlumpModel(config).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.startswith(("bid_pressure", "per_layer", "value_")):
                parameter.normal_(std=0.01)
    tokens = token_batch(4, 3, seeds=[0, 1])
    prefix_len = config.prefix_len(3)
    with torch.no_grad():
        full = model.forward_full(tokens, aux_heads=False)
        cache = model.new_cache(capacity=2)
        slots = torch.tensor(cache.alloc(2), dtype=torch.long)
        model.forward_prefill(tokens[:, :prefix_len], cache, slots)
        for position in range(prefix_len, tokens.shape[1]):
            step = model.forward_step(tokens[:, position], position, cache, slots)
    torch.testing.assert_close(
        step.bid_logits, full.bid_logits[:, -1], atol=ATOL, rtol=0
    )
    torch.testing.assert_close(
        step.card_logits, full.card_logits[:, -1], atol=ATOL, rtol=0
    )


def test_oracle_has_independent_extended_tables_at_its_longer_length():
    config = small_config(
        per_layer_embedding_dim=16,
        value_embedding_mode="alternating",
    )
    actor = SeqPlumpModel(config)
    oracle = SeqPPOOracleCritic(config, initialize_from=actor)
    assert oracle.backbone.per_layer_pos_embedding.num_embeddings == (
        config.oracle_max_seq_len
    )
    assert oracle.backbone.value_pos_embedding.num_embeddings == (
        config.oracle_max_seq_len
    )
    assert (
        oracle.backbone.per_layer_slot_embedding.weight.data_ptr()
        != actor.per_layer_slot_embedding.weight.data_ptr()
    )


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


def test_rollout_readout_selects_rows_and_only_the_requested_action_head():
    torch.manual_seed(0)
    model = SeqPlumpModel(small_config()).eval()
    tokens = token_batch(4, 3, seeds=[0, 1, 2])
    prefix_len = 4
    selected = torch.tensor([2, 0])

    with torch.no_grad():
        full_cache = model.new_cache(capacity=3)
        full_slots = torch.tensor(full_cache.alloc(3), dtype=torch.long)
        full = model.forward_prefill(
            tokens[:, :prefix_len], full_cache, full_slots
        )

        selected_cache = model.new_cache(capacity=3)
        selected_slots = torch.tensor(selected_cache.alloc(3), dtype=torch.long)
        bid = model.forward_prefill(
            tokens[:, :prefix_len],
            selected_cache,
            selected_slots,
            readout_indices=selected,
            phase="bid",
        )
        torch.testing.assert_close(bid.hidden, full.hidden[selected])
        torch.testing.assert_close(bid.bid_logits, full.bid_logits[selected])
        torch.testing.assert_close(bid.value, full.value[selected])
        assert bid.card_logits.shape == (2, 0)

        full_play = model.forward_step(
            tokens[:, prefix_len], prefix_len, full_cache, full_slots
        )
        play = model.forward_step(
            tokens[:, prefix_len],
            prefix_len,
            selected_cache,
            selected_slots,
            readout_indices=selected,
            phase="play",
        )
        torch.testing.assert_close(play.hidden, full_play.hidden[selected])
        torch.testing.assert_close(play.card_logits, full_play.card_logits[selected])
        torch.testing.assert_close(play.value, full_play.value[selected])
        assert play.bid_logits.shape == (2, 0)


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


@pytest.mark.parametrize("stacked", [False, True])
def test_branch_copy_matches_independent_full_forwards(stacked):
    """Both cache layouts must branch-copy identically.

    A stacked pool copies every layer in one indexed op and a per-layer pool
    does one op per layer, so this is the path where the two could diverge.
    """

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

        cache = model.new_cache(capacity=4, stacked=stacked)
        assert cache.stacked is stacked
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
    assert output.rank_boundary_logits.shape == (
        batch,
        length,
        config.belief_opponents + 1,
        4,
        2,
    )
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


def test_packed_linear_auxiliary_projection_matches_individual_heads():
    torch.manual_seed(9)
    model = SeqPlumpModel(small_config())
    hidden = torch.randn(17, model.config.d_model)
    packed = model.auxiliary_logits(
        hidden, {"trick", "rank_boundary", "suit"}
    )
    assert torch.allclose(packed["suit"], model.suit_presence_head(hidden))
    assert torch.allclose(
        packed["rank_boundary"], model.rank_boundary_head(hidden)
    )
    assert torch.allclose(packed["trick"], model.trick_count_head(hidden))
    with pytest.raises(ValueError):
        model.auxiliary_logits(hidden, {"bid_hit"})


def test_cache_layout_follows_the_element_limit():
    """Stack when it fits, split when it does not, decided on max_capacity."""

    from plump.seq.kv import MAX_TENSOR_ELEMENTS, KVCache

    config = small_config()
    row = config.kv_heads * config.max_seq_len * config.head_dim
    fits = MAX_TENSOR_ELEMENTS // (config.n_layers * row)

    assert KVCache.fits_stacked(config, fits, config.max_seq_len)
    assert not KVCache.fits_stacked(config, fits + 1, config.max_seq_len)

    small = KVCache(config, 8, "cpu")
    assert small.stacked

    # The pool that would overflow never allocates one: the choice is made from
    # max_capacity, not from the rows it starts with.
    huge = KVCache(config, 8, "cpu", max_capacity=fits + 1)
    assert not huge.stacked

    # And a pool whose ceiling is raised after construction drops to per-layer
    # rather than growing into a tensor that overflows.
    grown = KVCache(config, 8, "cpu", max_capacity=16)
    assert grown.stacked
    grown.max_capacity = fits + 1
    grown.ensure_capacity(fits + 1)
    assert not grown.stacked
    assert grown.capacity >= fits + 1


def test_stacked_views_write_through_to_the_shared_base():
    """self.k[layer] is a view of the stacked tensor, not a copy of it."""

    from plump.seq.kv import KVCache

    config = small_config()
    cache = KVCache(config, 4, "cpu", stacked=True)
    cache.k[1][2, :, :3] = 5.0
    assert (cache._base_k[1, 2, :, :3] == 5.0).all()
    assert (cache._base_k[0, 2, :, :3] == 0.0).all()


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
