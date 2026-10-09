"""One optimizer step per update, whatever the microbatch split.

``microbatch_positions`` exists to cap peak update memory, so it must be a
pure memory knob: splitting a batch into more chunks may not change the
gradient the optimizer receives, and may not turn one update into several.

The failure this guards against is real and was shipped once. Under
distillation a collection wave was split into ``updates_per_collect``
*separate* optimizer steps; the per-chunk loss was then a mean, so chunked
backwards summed rather than averaged and the policy gradient scaled with the
chunk count -- a 200-norm gradient that moved with the shape schedule.
"""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from plump.seq.model import SeqPlumpModel
from plump.seq.ppo import build_ppo_training_batch, normalize_ppo_advantages
from plump.seq.trainer import SeqTrainer, SeqUpdateStats

from tests.test_seq_ppo import MODEL, ppo_config


def _chunked_trainer(microbatch_positions: int) -> SeqTrainer:
    torch.manual_seed(0)
    config = replace(
        ppo_config(
            suit_coef=0.1,
            rank_boundary_coef=0.2,
            trick_coef=0.2,
            next_winner_coef=0.2,
            actor_value_coef=0.2,
            auxiliary_positions="all",
        ),
        microbatch_positions=microbatch_positions,
    )
    return SeqTrainer(SeqPlumpModel(MODEL), config, device="cpu")


def _capture_pre_clip_gradient(trainer: SeqTrainer, trees) -> tuple[dict, int]:
    """Snapshot ``.grad`` after the accumulation loop, before any rescaling.

    Both ``_apply_auxiliary_grad_budget`` and ``_clip_actor`` renormalize, so
    a scale error introduced by chunking would be clipped away and the test
    would pass on a broken build. This hooks the first thing that runs after
    ``_ppo_actor_backward`` returns. The chunk count comes from the real
    update rather than a replayed group build, so it cannot drift from it.
    """

    captured: dict = {}
    chunks = 0
    original_budget = trainer._apply_auxiliary_grad_budget
    original_microbatches = trainer._microbatches

    def counted(group):
        nonlocal chunks
        produced = list(original_microbatches(group))
        chunks += len(produced)
        return iter(produced)

    def hook():
        if not captured:
            captured.update(
                {
                    name: parameter.grad.detach().clone()
                    for name, parameter in trainer.model.named_parameters()
                    if parameter.grad is not None
                }
            )
        return original_budget()

    trainer._microbatches = counted
    trainer._apply_auxiliary_grad_budget = hook
    trainer.update(copy.deepcopy(trees))
    return captured, chunks


def test_microbatching_does_not_change_the_accumulated_gradient():
    source = _chunked_trainer(1 << 20)
    trees, _ = source.collect()

    whole_grad, whole_chunks = _capture_pre_clip_gradient(
        _chunked_trainer(1 << 20), trees
    )
    chunked_grad, chunked_chunks = _capture_pre_clip_gradient(
        _chunked_trainer(48), trees
    )

    assert chunked_chunks > whole_chunks, "chunked config must actually split"
    assert set(whole_grad) == set(chunked_grad)
    assert whole_grad, "no gradients were captured"
    for name, expected in whole_grad.items():
        torch.testing.assert_close(
            chunked_grad[name],
            expected,
            rtol=1e-4,
            atol=1e-6,
            msg=lambda formatted, name=name: f"{name}: {formatted}",
        )


def test_budgeted_microbatching_preserves_both_unscaled_gradient_halves():
    source = _chunked_trainer(1 << 20)
    trees, _ = source.collect()
    captured = []
    for width in (1 << 20, 48):
        trainer = _chunked_trainer(width)
        trainer.train = replace(trainer.train, ppo_auxiliary_grad_fraction=0.25)
        batch = build_ppo_training_batch(trees, MODEL, trainer.train)
        trainer._prepare_ppo_advantages(
            batch.policy_groups, batch.critic_groups, SeqUpdateStats()
        )
        normalize_ppo_advantages(batch.policy_groups)
        trainer.model.train()
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer._ppo_actor_backward(batch.policy_groups)
        # Capture before the relative budget or global clip can hide a scale
        # error. A correct total alone would not catch mixing the two halves.
        captured.append((trainer._policy_grads, trainer._auxiliary_grads))
    for whole, chunked in zip(*captured):
        assert len(whole) == len(chunked)
        for expected, actual in zip(whole, chunked):
            if expected is None:
                assert actual is None
            else:
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("microbatch_positions", (48, 1 << 20))
def test_one_update_takes_exactly_one_optimizer_step(microbatch_positions):
    trainer = _chunked_trainer(microbatch_positions)
    trees, _ = trainer.collect()

    steps = 0
    original = trainer.optimizer.step

    def counted(*args, **kwargs):
        nonlocal steps
        steps += 1
        return original(*args, **kwargs)

    trainer.optimizer.step = counted
    before = trainer.optimizer_steps
    trainer.update(trees)

    assert steps == 1
    assert trainer.optimizer_steps == before + 1
