"""Ragged decode: one call carrying rows at different cache lengths.

The merged rollout wave rests on these. A row must get exactly the hidden
state it would have got in a homogeneous wave of its own -- otherwise the
cache silently desynchronises from the token stream and the training data is
wrong in a way no loss curve shows.
"""

from __future__ import annotations

import torch

from plump.seq.config import (
    BASE_TOKEN_WIDTH,
    CORE_TOKEN_WIDTH,
    NA_PATTERN_MASKS,
    NUM_CARDS,
    SLOT_NA_PATTERN,
    SLOT_REMAINING_HAND_START,
    SeqModelConfig,
    TOKEN_WIDTH,
)
from plump.seq.kv import KVCache
from plump.seq.model import SeqPlumpModel


MODEL = SeqModelConfig(d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=64)


def _model(seed: int = 0) -> SeqPlumpModel:
    torch.manual_seed(seed)
    return SeqPlumpModel(MODEL).eval()


def _tokens(rows: int, seed: int) -> torch.Tensor:
    """Random tokens that are in range for every slot's own vocabulary."""

    generator = torch.Generator().manual_seed(seed)

    def draw(size: int, count: int) -> torch.Tensor:
        return torch.randint(
            0, size, (count,), generator=generator, dtype=torch.long
        )

    tokens = torch.zeros((rows, 1, TOKEN_WIDTH), dtype=torch.long)
    for slot, size in enumerate(MODEL.core_slot_vocab_sizes):
        tokens[:, 0, slot] = draw(size, rows)
    for offset, size in enumerate(MODEL.bid_pressure_slot_vocab_sizes):
        tokens[:, 0, CORE_TOKEN_WIDTH + offset] = draw(size, rows)
    for slot in range(BASE_TOKEN_WIDTH, SLOT_NA_PATTERN):
        tokens[:, 0, slot] = draw(NUM_CARDS + 1, rows)
    tokens[:, 0, SLOT_NA_PATTERN] = draw(len(NA_PATTERN_MASKS), rows)
    return tokens


def _cache(rows: int) -> KVCache:
    return KVCache(MODEL, rows, "cpu", dtype=torch.float32, max_len=MODEL.max_seq_len)


def _prefill(model, cache, tokens, length, slots=None, seed=100):
    """Drive every row to ``length`` through the ordinary lockstep path.

    Each position gets its own valid tokens, derived only from ``seed`` and the
    position, so a row driven alone and the same row driven inside a batch see
    an identical token stream.
    """

    rows = tokens.shape[0]
    for position in range(length):
        step = _tokens(rows, seed=seed + position)
        with torch.no_grad():
            model.forward_step(
                step, position, cache, slots, with_value=False
            )


def test_ragged_matches_lockstep_when_every_row_is_at_one_length():
    """With equal positions the two paths must agree to floating-point noise."""

    rows, length = 8, 5
    tokens = _tokens(rows, seed=1)

    lockstep_model, ragged_model = _model(), _model()
    lockstep_cache, ragged_cache = _cache(rows), _cache(rows)
    _prefill(lockstep_model, lockstep_cache, tokens, length)
    _prefill(ragged_model, ragged_cache, tokens, length)

    with torch.no_grad():
        expected = lockstep_model.forward_step(
            tokens, length, lockstep_cache, None, with_value=False
        ).hidden
        positions = torch.full((rows,), length, dtype=torch.long)
        actual = ragged_model.forward_step_ragged(
            tokens, positions, ragged_cache, None
        )

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)
    # The cache must also match: a hidden state that agrees while the cache
    # diverges corrupts every later step instead of this one.
    for layer in range(MODEL.n_layers):
        assert torch.allclose(
            ragged_cache.k[layer][:rows, :, : length + 1],
            lockstep_cache.k[layer][:rows, :, : length + 1],
            atol=1e-5,
        )


def _stream(rows: int, length: int, seed: int) -> torch.Tensor:
    """[rows, length, WIDTH] of valid tokens, indexable per row."""

    return torch.cat(
        [_tokens(rows, seed=seed + step) for step in range(length)], dim=1
    )


def test_each_row_matches_a_homogeneous_wave_at_its_own_length():
    """The real property: mixing lengths changes nobody's answer."""

    lengths = [1, 3, 4, 7, 2, 9, 5, 6]
    rows = len(lengths)
    stream = _stream(rows, max(lengths), seed=200)
    final = _tokens(rows, seed=300)
    model = _model()

    # Reference: each row driven alone, at its own length, in a wave of one.
    expected = []
    for index, length in enumerate(lengths):
        cache = _cache(1)
        with torch.no_grad():
            for position in range(length):
                model.forward_step(
                    stream[index : index + 1, position : position + 1],
                    position,
                    cache,
                    None,
                    with_value=False,
                )
            expected.append(
                model.forward_step(
                    final[index : index + 1], length, cache, None,
                    with_value=False,
                ).hidden
            )
    expected = torch.cat(expected)

    # Merged: all eight rows, every one at a different length, in one call.
    cache = _cache(rows)
    with torch.no_grad():
        for index, length in enumerate(lengths):
            for position in range(length):
                model.forward_step(
                    stream[index : index + 1, position : position + 1],
                    position,
                    cache,
                    torch.tensor([index]),
                    with_value=False,
                )
        actual = model.forward_step_ragged(
            final, torch.tensor(lengths), cache, None
        )

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)


def test_a_short_row_never_attends_past_its_own_prefix():
    """Padding beyond a row's own length must not reach its output."""

    lengths = [2, 9]
    stream = _stream(2, max(lengths), seed=400)
    final = _tokens(2, seed=500)
    model = _model()

    def run(poison: float):
        cache = _cache(2)
        with torch.no_grad():
            for index, length in enumerate(lengths):
                for position in range(length):
                    model.forward_step(
                        stream[index : index + 1, position : position + 1],
                        position,
                        cache,
                        torch.tensor([index]),
                        with_value=False,
                    )
            # Scribble on the short row's unused tail, which the merged read
            # covers because the long row needs it. A correct mask ignores it.
            for layer in range(MODEL.n_layers):
                cache.k[layer][0, :, lengths[0] + 1 :] = poison
                cache.v[layer][0, :, lengths[0] + 1 :] = poison
            return model.forward_step_ragged(
                final, torch.tensor(lengths), cache, None
            )

    assert torch.allclose(run(0.0), run(37.5), atol=1e-6)


def test_a_ragged_run_matches_the_same_run_at_a_shared_position():
    """Multi-token appends: the ramp of allowed keys must follow each row."""

    rows, length, run = 6, 4, 3
    stream = _stream(rows, length, seed=600)
    tail = _stream(rows, run, seed=700)
    model = _model()

    lockstep_cache, ragged_cache = _cache(rows), _cache(rows)
    with torch.no_grad():
        for cache in (lockstep_cache, ragged_cache):
            for position in range(length):
                model.forward_step(
                    stream[:, position : position + 1],
                    position,
                    cache,
                    None,
                    with_value=False,
                )
        expected = model.forward_step(
            tail, length, lockstep_cache, None, with_value=False
        ).hidden
        actual = model.forward_step_ragged(
            tail, torch.full((rows,), length, dtype=torch.long), ragged_cache
        )

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)
    for layer in range(MODEL.n_layers):
        assert torch.allclose(
            ragged_cache.k[layer][:rows, :, : length + run],
            lockstep_cache.k[layer][:rows, :, : length + run],
            atol=1e-5,
        )


def test_a_padded_short_run_reads_out_at_its_own_last_real_token():
    """A row appending fewer tokens than the batch must ignore its padding.

    This is the property the merged wave leans on: one game closes a trick and
    appends two tokens while another appends one, and the short row's answer
    has to be the answer it would have got on its own.
    """

    lengths = [5, 3]
    counts = [1, 2]
    stream = _stream(2, max(lengths), seed=800)
    tail = _stream(2, max(counts), seed=900)
    model = _model()

    expected = []
    with torch.no_grad():
        for index, (length, count) in enumerate(zip(lengths, counts)):
            cache = _cache(1)
            for position in range(length):
                model.forward_step(
                    stream[index : index + 1, position : position + 1],
                    position,
                    cache,
                    None,
                    with_value=False,
                )
            expected.append(
                model.forward_step(
                    tail[index : index + 1, :count],
                    length,
                    cache,
                    None,
                    with_value=False,
                ).hidden
            )
    expected = torch.cat(expected)

    def run(pad: torch.Tensor) -> torch.Tensor:
        cache = _cache(2)
        padded = tail.clone()
        # Row 0 appends one real token; whatever fills its second slot must
        # not reach its readout.
        padded[0, 1] = pad
        with torch.no_grad():
            for index, length in enumerate(lengths):
                for position in range(length):
                    model.forward_step(
                        stream[index : index + 1, position : position + 1],
                        position,
                        cache,
                        torch.tensor([index]),
                        with_value=False,
                    )
            return model.forward_step_ragged(
                padded,
                torch.tensor(lengths),
                cache,
                None,
                readout_offsets=torch.tensor([count - 1 for count in counts]),
            )

    actual = run(tail[0, 0])
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)
    # And the padding really is inert, not merely equal by luck.
    assert torch.allclose(actual, run(tail[1, 1]), atol=1e-6)
