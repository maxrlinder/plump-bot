"""Round and position-in-round replace absolute sequence position.

An absolute index means different things at different table sizes -- index 14
is mid-bidding with five players and already the second trick with three -- so
the model reads the game's own coordinates instead: SLOT_TRICK for the round,
SLOT_POS_IN_TRICK for the seat order within it, SLOT_REL_PLAYER for who acted.
"""

from __future__ import annotations

import torch

from plump.seq.config import (
    SLOT_POS_IN_TRICK,
    SLOT_REL_PLAYER,
    SLOT_TRICK,
    SLOT_TYPE,
    TOKEN_HAND,
    TOKEN_PLAY,
    SeqModelConfig,
)
from plump.seq.model import SeqPlumpModel

from tests.test_seq_model import small_config, token_batch


def no_position_config(**overrides) -> SeqModelConfig:
    return small_config(absolute_position_embedding=False, **overrides)


def test_no_positional_parameters_remain():
    model = SeqPlumpModel(no_position_config(per_layer_embedding_dim=8))
    assert model.pos_embedding is None
    assert model.per_layer_pos_embedding is None
    assert model.value_pos_embedding is None
    assert not [name for name, _ in model.named_parameters() if "pos_embedding" in name]


def test_absolute_index_is_not_read():
    """Embedding the same tokens at a different sequence offset is a no-op.

    This is the whole claim: nothing in the model depends on *where* in the
    sequence a token sits, only on what the token says about itself.
    """

    torch.manual_seed(0)
    model = SeqPlumpModel(no_position_config(per_layer_embedding_dim=8)).eval()
    tokens = token_batch(4, 3, seeds=[0, 1])
    with torch.no_grad():
        at_zero = model.embed(tokens, 0)
        at_offset = model.embed(tokens, 7)
    torch.testing.assert_close(at_zero, at_offset)


def test_absolute_index_is_read_when_enabled():
    """Control: the same probe separates the two configurations."""

    torch.manual_seed(0)
    model = SeqPlumpModel(small_config(per_layer_embedding_dim=8)).eval()
    tokens = token_batch(4, 3, seeds=[0, 1])
    with torch.no_grad():
        at_zero = model.embed(tokens, 0)
        at_offset = model.embed(tokens, 7)
    assert not torch.allclose(at_zero, at_offset)


def test_hand_tokens_carry_a_player_but_no_round_coordinates():
    """A hand card is 'who holds it, and which card' and nothing more."""

    tokens = token_batch(4, 3, seeds=[0])
    hand = tokens[0][tokens[0][:, SLOT_TYPE] == TOKEN_HAND]
    assert hand.shape[0] == 3
    config = SeqModelConfig()
    # The observer's own hand: relative seat 0.
    assert torch.all(hand[:, SLOT_REL_PLAYER] == 0)
    # No round or position-in-round is asserted for a card merely held.
    assert torch.all(hand[:, SLOT_TRICK] == config.trick_na_id)
    assert torch.all(hand[:, SLOT_POS_IN_TRICK] == config.pos_na_id)


def test_round_and_position_in_round_still_drive_the_model():
    """The coordinates that replaced absolute position must actually be read."""

    torch.manual_seed(0)
    model = SeqPlumpModel(no_position_config(per_layer_embedding_dim=8)).eval()
    tokens = token_batch(4, 3, seeds=[0, 1])
    with torch.no_grad():
        base = model.forward_full(tokens, aux_heads=False)

    for slot in (SLOT_TRICK, SLOT_POS_IN_TRICK):
        moved = tokens.clone()
        plays = moved[..., SLOT_TYPE] == TOKEN_PLAY
        column = moved[..., slot]
        column[plays] = column[plays] + 1
        with torch.no_grad():
            after = model.forward_full(moved, aux_heads=False)
        assert not torch.allclose(base.card_logits, after.card_logits), (
            f"slot {slot} had no effect on the model"
        )


def test_player_identity_on_a_hand_card_is_read():
    torch.manual_seed(0)
    model = SeqPlumpModel(no_position_config(per_layer_embedding_dim=8)).eval()
    tokens = token_batch(4, 3, seeds=[0])
    with torch.no_grad():
        base = model.forward_full(tokens, aux_heads=False)
    moved = tokens.clone()
    hand = moved[..., SLOT_TYPE] == TOKEN_HAND
    moved[..., SLOT_REL_PLAYER][hand] = 1
    with torch.no_grad():
        after = model.forward_full(moved, aux_heads=False)
    assert not torch.allclose(base.card_logits, after.card_logits)


def test_runs_a_training_step_without_positions():
    torch.manual_seed(0)
    model = SeqPlumpModel(no_position_config(per_layer_embedding_dim=8))
    tokens = token_batch(4, 3, seeds=[0, 1])
    out = model.forward_full(tokens, aux_heads=False)
    (out.card_logits.square().mean() + out.value.square().mean()).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_default_still_uses_absolute_positions():
    """Existing checkpoints keep their architecture untouched."""

    model = SeqPlumpModel(small_config())
    assert model.config.absolute_position_embedding is True
    assert model.pos_embedding is not None
