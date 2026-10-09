"""Cards as rank + suit only, on both sides of the model.

``card_embedding_mode="factored"`` removes every per-card parameter: the
exact-card input rows and the 52 x d output matrix both go away, leaving rank
and suit directions at full model width. The token schema is deliberately
untouched, so these tests also pin the property the distillation pipeline
depends on -- a factored student and a dense teacher read identical tokens.
"""

import numpy as np
import pytest
import torch

from plump.seq.config import (
    NUM_CARDS,
    NUM_RANKS,
    SLOT_CARD,
    SeqModelConfig,
)
from plump.seq.model import SeqPlumpModel

from tests.test_seq_model import small_config, token_batch


def factored_config(**overrides) -> SeqModelConfig:
    return small_config(card_embedding_mode="factored", **overrides)


def test_factored_model_has_no_per_card_parameter():
    model = SeqPlumpModel(factored_config())
    names = dict(model.named_parameters())
    assert "card_head.weight" not in names
    assert model.card_head.weight is None
    # Only the factored bias survives on the head itself.
    assert names["card_head.rank_bias"].shape == (NUM_RANKS,)
    assert names["card_head.suit_bias"].shape == (4,)
    for name, parameter in names.items():
        assert NUM_CARDS not in tuple(parameter.shape), (
            f"{name} still carries a per-card axis: {tuple(parameter.shape)}"
        )


def test_dense_mode_is_untouched():
    """The teacher checkpoint must keep loading bit-for-bit."""

    dense = SeqPlumpModel(small_config())
    assert dense.card_head.weight.shape == (NUM_CARDS, dense.config.d_model)
    names = dict(dense.named_parameters())
    assert "card_head.weight" in names and "card_head.bias" in names


def test_factored_tables_drop_the_card_block():
    dense = SeqPlumpModel(small_config(per_layer_embedding_dim=8))
    factored = SeqPlumpModel(factored_config(per_layer_embedding_dim=8))
    # 53 card rows removed, one inert row kept so offset arithmetic is shared.
    assert (
        factored.slot_embedding.weight.shape[0]
        == dense.slot_embedding.weight.shape[0] - NUM_CARDS
    )
    assert (
        factored.per_layer_slot_embedding.weight.shape[0]
        == dense.per_layer_slot_embedding.weight.shape[0] - NUM_CARDS
    )


def test_token_schema_is_unchanged_so_teacher_and_student_share_rollouts():
    dense = SeqPlumpModel(small_config())
    factored = SeqPlumpModel(factored_config())
    assert dense.config.slot_vocab_sizes == factored.config.slot_vocab_sizes
    tokens = token_batch(4, 3, seeds=[0, 1])
    # The same tensor feeds both without reserialization.
    for model in (dense, factored):
        model.eval()
        with torch.no_grad():
            out = model.forward_full(tokens, aux_heads=False)
        assert out.card_logits.shape[-1] == NUM_CARDS


def test_card_column_is_inert_under_factoring():
    """Perturbing SLOT_CARD must not move a factored model's output.

    Rank and suit are carried in their own columns, so the exact-card id is
    redundant; this is the claim that justifies removing it.
    """

    torch.manual_seed(0)
    model = SeqPlumpModel(factored_config()).eval()
    tokens = token_batch(4, 3, seeds=[0, 1])
    with torch.no_grad():
        before = model.forward_full(tokens, aux_heads=False)
    scrambled = tokens.clone()
    card_column = scrambled[..., SLOT_CARD]
    # Rewrite every real card id, leaving the NA sentinel alone.
    real = card_column < NUM_CARDS
    card_column[real] = (card_column[real] + 7) % NUM_CARDS
    with torch.no_grad():
        after = model.forward_full(scrambled, aux_heads=False)
    torch.testing.assert_close(before.card_logits, after.card_logits)
    torch.testing.assert_close(before.bid_logits, after.bid_logits)
    torch.testing.assert_close(before.value, after.value)


def test_dense_model_does_depend_on_the_card_column():
    """Control for the test above: the dense path genuinely reads SLOT_CARD."""

    torch.manual_seed(0)
    model = SeqPlumpModel(small_config()).eval()
    tokens = token_batch(4, 3, seeds=[0, 1])
    with torch.no_grad():
        before = model.forward_full(tokens, aux_heads=False)
    scrambled = tokens.clone()
    column = scrambled[..., SLOT_CARD]
    real = column < NUM_CARDS
    column[real] = (column[real] + 7) % NUM_CARDS
    with torch.no_grad():
        after = model.forward_full(scrambled, aux_heads=False)
    assert not torch.allclose(before.card_logits, after.card_logits)


def test_factored_card_logits_are_rank_plus_suit_additive():
    """The intended structure: L[r][s] = a_r + b_s, exactly."""

    torch.manual_seed(0)
    model = SeqPlumpModel(factored_config()).eval()
    tokens = token_batch(4, 3, seeds=[0])
    with torch.no_grad():
        logits = model.forward_full(tokens, aux_heads=False).card_logits[0, -1]
    grid = logits.view(4, NUM_RANKS)  # card id = suit * NUM_RANKS + rank
    # Every pair of suits differs by a single constant across all ranks.
    for suit in range(1, 4):
        delta = grid[suit] - grid[0]
        torch.testing.assert_close(
            delta, delta.mean().expand_as(delta), atol=1e-5, rtol=1e-4
        )


def test_factored_model_trains_a_step():
    torch.manual_seed(0)
    model = SeqPlumpModel(factored_config())
    tokens = token_batch(4, 3, seeds=[0, 1])
    out = model.forward_full(tokens, aux_heads=False)
    loss = out.card_logits.square().mean() + out.value.square().mean()
    loss.backward()
    for name in ("card_rank_output_embedding.weight", "card_suit_output_embedding.weight"):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all()
        assert grad.abs().sum() > 0, f"{name} received no gradient"
