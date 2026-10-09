"""On-policy distillation bootstrap.

The pipeline's premise is that a frozen dense teacher can supervise a
structurally different student -- different depth, different width, and cards
factored into rank + suit -- because both read the identical token stream.
Collection is on-policy: the student visits its own states and the teacher
labels them, so the recorded rollout distribution is the student's and only
the teacher's forward pass may be used as a target. These tests pin both.
"""

import math

import numpy as np
import pytest
import torch

from plump.seq.config import (
    BranchBudgetConfig,
    BranchRuleConfig,
    GameScheduleCell,
    SeqModelConfig,
    SeqTrainingConfig,
)
from plump.seq.model import SeqPlumpModel
from plump.seq.trainer import SeqTrainer, load_distillation_teacher

TEACHER_CONFIG = SeqModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128)
# Deliberately unlike the teacher on every axis the pipeline claims to bridge.
STUDENT_CONFIG = SeqModelConfig(
    d_model=48,
    n_layers=4,
    n_heads=6,
    n_kv_heads=2,
    d_ff=96,
    card_embedding_mode="factored",
)


def write_teacher(tmp_path, config=TEACHER_CONFIG):
    torch.manual_seed(1)
    teacher = SeqPlumpModel(config)
    path = tmp_path / "teacher.pt"
    torch.save(
        {
            "schema_version": __import__(
                "plump.seq.config", fromlist=["SEQ_SCHEMA_VERSION"]
            ).SEQ_SCHEMA_VERSION,
            "model_format_version": 1,
            "model_config": {
                key: value
                for key, value in config.__dict__.items()
                if not key.startswith("_")
            },
            "model_state_dict": teacher.state_dict(),
        },
        path,
    )
    return path


def make_distill_trainer(tmp_path, **overrides):
    torch.manual_seed(0)
    defaults = dict(
        schedule_cells=(
            GameScheduleCell(hand_size=3, num_players=3),
            GameScheduleCell(hand_size=4, num_players=3),
        ),
        branch_rule=BranchRuleConfig(bid_top_k=3),
        # Distillation wants breadth of states, not depth of tree.
        branch_budget=BranchBudgetConfig(branch_rate=0.0),
        microbatch_positions=512,
        snapshot_every=1,
        policy_objective="distill",
        distill_teacher_checkpoint=str(write_teacher(tmp_path)),
        ppo_entropy_mode="fixed",
        policy_kl_cap=0.0,
        policy_kl_p99_cap=0.0,
    )
    defaults.update(overrides)
    train = SeqTrainingConfig(**defaults)
    student = SeqPlumpModel(STUDENT_CONFIG)
    return SeqTrainer(student, train, device="cpu")


def test_teacher_loads_with_its_own_architecture(tmp_path):
    path = write_teacher(tmp_path)
    student = SeqPlumpModel(STUDENT_CONFIG)
    teacher = load_distillation_teacher(
        path, device=torch.device("cpu"), student_config=student.config
    )
    assert teacher.config.d_model == 64 and teacher.config.n_layers == 2
    assert teacher.config.card_embedding_mode == "dense"
    # Frozen: no gradient may ever reach the teacher.
    assert all(not p.requires_grad for p in teacher.parameters())
    assert not teacher.training


def test_teacher_with_a_different_token_schema_is_rejected(tmp_path):
    path = write_teacher(
        tmp_path, config=SeqModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128,
                                        trick_win_token=False)
    )
    student = SeqPlumpModel(STUDENT_CONFIG)
    with pytest.raises(ValueError, match="serialize tokens identically"):
        load_distillation_teacher(
            path, device=torch.device("cpu"), student_config=student.config
        )


def test_rollouts_are_on_policy_by_default(tmp_path):
    """The student visits its own states; the teacher only labels them."""

    trainer = make_distill_trainer(tmp_path)
    assert trainer.teacher is not None
    assert trainer.collector.model is trainer.model
    assert trainer.collector.model is not trainer.teacher


def test_teacher_can_drive_rollouts_when_configured(tmp_path):
    trainer = make_distill_trainer(tmp_path, distill_rollout_policy="teacher")
    assert trainer.collector.model is trainer.teacher


def test_target_is_the_teacher_not_the_behaviour_policy(tmp_path):
    """A student identical to the teacher must already have zero KL.

    Under on-policy collection the recorded distribution is the *student's*,
    so a pipeline that mistakenly distilled ``old_probs_full`` would also
    report ~0 here. The companion test below rules that out.
    """

    path = write_teacher(tmp_path, config=STUDENT_CONFIG)
    train = SeqTrainingConfig(
        schedule_cells=(GameScheduleCell(hand_size=3, num_players=3),),
        branch_rule=BranchRuleConfig(bid_top_k=3),
        branch_budget=BranchBudgetConfig(branch_rate=0.0),
        microbatch_positions=512,
        snapshot_every=1,
        policy_objective="distill",
        distill_teacher_checkpoint=str(path),
        ppo_entropy_mode="fixed",
        policy_kl_cap=0.0,
        policy_kl_p99_cap=0.0,
    )
    student = SeqPlumpModel(STUDENT_CONFIG)
    trainer = SeqTrainer(student, train, device="cpu")
    trainer.model.load_state_dict(trainer.teacher.state_dict())
    trainer.model.clear_card_output_cache()
    trees, _ = trainer.collect()
    stats = trainer.update(trees)
    assert stats.distill_kl == pytest.approx(0.0, abs=1e-6)
    assert stats.distill_agree == pytest.approx(1.0)


def test_a_mismatched_student_has_nonzero_kl(tmp_path):
    """Control: the assertion above is not vacuously true."""

    path = write_teacher(tmp_path, config=STUDENT_CONFIG)
    train = SeqTrainingConfig(
        schedule_cells=(GameScheduleCell(hand_size=3, num_players=3),),
        branch_rule=BranchRuleConfig(bid_top_k=3),
        branch_budget=BranchBudgetConfig(branch_rate=0.0),
        microbatch_positions=512,
        snapshot_every=1,
        policy_objective="distill",
        distill_teacher_checkpoint=str(path),
        ppo_entropy_mode="fixed",
        policy_kl_cap=0.0,
        policy_kl_p99_cap=0.0,
    )
    torch.manual_seed(99)
    trainer = SeqTrainer(SeqPlumpModel(STUDENT_CONFIG), train, device="cpu")
    trees, _ = trainer.collect()
    stats = trainer.update(trees)
    assert stats.distill_kl > 1e-4


def test_distill_update_runs_and_reports_kl(tmp_path):
    trainer = make_distill_trainer(tmp_path)
    trees, _ = trainer.collect()
    stats = trainer.update(trees)
    assert stats.policy_rows > 0
    assert math.isfinite(stats.distill_kl) and stats.distill_kl > 0
    assert math.isfinite(stats.distill_loss_value)
    assert math.isfinite(stats.distill_loss_aux)
    # Agreement is a rate.
    assert 0.0 <= stats.distill_agree <= 1.0
    assert 0.0 <= stats.distill_top1_mass_play <= 1.0
    # A random student cannot already match a random teacher's entropy.
    assert stats.distill_teacher_entropy_play >= 0.0
    assert stats.distill_student_entropy_play >= 0.0


def test_the_teacher_never_receives_gradient(tmp_path):
    trainer = make_distill_trainer(tmp_path)
    trees, _ = trainer.collect()
    trainer.update(trees)
    assert all(p.grad is None for p in trainer.teacher.parameters())


def test_distillation_reduces_kl_over_successive_updates(tmp_path):
    """The point of the whole pipeline: KL to the teacher must come down."""

    trainer = make_distill_trainer(
        tmp_path, learning_rate=1e-3, core_learning_rate=1e-3,
        auxiliary_learning_rate=1e-3, lr_warmup_updates=0,
    )
    first = None
    last = None
    for _ in range(6):
        trees, _ = trainer.collect()
        stats = trainer.update(trees)
        if first is None:
            first = stats.distill_kl
        last = stats.distill_kl
    assert last < first, f"KL did not fall: {first:.4f} -> {last:.4f}"


def test_student_parameters_actually_move(tmp_path):
    trainer = make_distill_trainer(tmp_path)
    before = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
    }
    trees, _ = trainer.collect()
    trainer.update(trees)
    moved = [
        name
        for name, parameter in trainer.model.named_parameters()
        if not torch.equal(before[name], parameter.detach())
    ]
    # The factored card head is the student's only card-scoring path, so it
    # must be among the parameters the update moves.
    assert "card_rank_output_embedding.weight" in moved
    assert "card_suit_output_embedding.weight" in moved


def test_aux_head_kl_is_reported_on_the_update_not_per_chunk(tmp_path):
    """Per-head KL must not be divided by the microbatch count.

    position_weight sums to 1 over the whole update, so each chunk returns a
    weighted sum and the chunks add to the mean exactly once. A stray division
    by chunk count made every belief head read near-zero even for a random
    student, which is how this was found.
    """

    # One microbatch, then many: the reported value must not change much.
    single = make_distill_trainer(tmp_path, microbatch_positions=10**7)
    trees, _ = single.collect()
    wide = single.update(trees).distill_aux_kl_suit

    narrow_trainer = make_distill_trainer(tmp_path, microbatch_positions=512)
    narrow_trainer.model.load_state_dict(single.model.state_dict())
    narrow = narrow_trainer.update(trees).distill_aux_kl_suit

    assert wide > 0 and narrow > 0
    assert narrow == pytest.approx(wide, rel=0.05), (
        f"chunking changed the reported KL: {wide:.5f} vs {narrow:.5f}"
    )


def test_a_nonfinite_gradient_skips_the_step_instead_of_poisoning_the_model(tmp_path):
    """One NaN step is otherwise unrecoverable.

    NaN gradients write NaN into the weights, after which every forward,
    gradient and metric is NaN forever and the run looks alive while learning
    nothing. That is exactly what happened at iteration 525 of the first long
    run. The step must be dropped and the weights left intact.
    """

    trainer = make_distill_trainer(tmp_path)
    trees, _ = trainer.collect()
    before = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
    }

    real_chunk_loss = trainer._distill_chunk_loss

    def poisoned(*args, **kwargs):
        loss, terms = real_chunk_loss(*args, **kwargs)
        # A finite forward with a NaN-producing graph, matching the observed
        # failure: the loss printed fine and only the gradient was NaN.
        poison = trainer.model.card_rank_output_embedding.weight.sum()
        return loss + (poison * float("inf")) * 0.0, terms

    trainer._distill_chunk_loss = poisoned
    stats = trainer.update(trees)

    assert stats.distill_skipped_steps >= 1.0
    for name, parameter in trainer.model.named_parameters():
        assert torch.isfinite(parameter).all(), f"{name} was poisoned"
        assert torch.equal(before[name], parameter.detach()), (
            f"{name} moved on a skipped step"
        )


def test_the_guard_does_not_fire_on_a_healthy_update(tmp_path):
    trainer = make_distill_trainer(tmp_path)
    trees, _ = trainer.collect()
    stats = trainer.update(trees)
    assert stats.distill_skipped_steps == 0.0


def test_policy_loss_does_not_scale_with_microbatch_count(tmp_path):
    """The policy term must be a weighted mean over the update, not per chunk.

    Every microbatch calls backward separately, so a per-chunk ``.mean()``
    adds instead of averaging and scales the policy gradient by the number of
    chunks. That drifts with the shape schedule, silently outweighs the value
    and auxiliary terms, and was what produced ~200 gradient norms against a
    clip of 1.0.
    """

    wide = make_distill_trainer(tmp_path, microbatch_positions=10**7)
    trees, _ = wide.collect()
    wide_stats = wide.update(trees)

    narrow = make_distill_trainer(tmp_path, microbatch_positions=512)
    narrow.model.load_state_dict(wide.model.state_dict())
    narrow.model.clear_card_output_cache()
    narrow_stats = narrow.update(trees)

    assert wide_stats.distill_loss_policy > 0
    assert narrow_stats.distill_loss_policy == pytest.approx(
        wide_stats.distill_loss_policy, rel=0.05
    ), (
        f"policy loss scaled with chunking: {wide_stats.distill_loss_policy:.4f} "
        f"vs {narrow_stats.distill_loss_policy:.4f}"
    )


def test_split_for_updates_keeps_every_shape_in_every_slice(tmp_path):
    """Slices must each see the schedule's shape mix, not one cell each.

    Trees arrive grouped by schedule cell, so a contiguous split would hand
    one update all the 3-card deals and another all the 10-card ones.
    build_training_groups renormalizes within whatever it is given, so a
    shape-skewed slice trains on a distribution the schedule never described.
    """

    trainer = make_distill_trainer(
        tmp_path,
        updates_per_collect=4,
        schedule_cells=(
            GameScheduleCell(hand_size=3, num_players=3, games=2),
            GameScheduleCell(hand_size=4, num_players=3, games=2),
        ),
    )
    trees, _ = trainer.collect()
    slices = trainer.split_for_updates(trees)

    assert len(slices) == 4
    assert sum(len(s) for s in slices) == len(trees)
    # Disjoint: no tree consumed twice.
    seen = [id(tree) for s in slices for tree in s]
    assert len(seen) == len(set(seen))
    # Every slice carries both configured hand sizes.
    for slice_ in slices:
        shapes = {(t.num_players, t.hand_size) for t in slice_}
        assert len(shapes) > 1, f"slice collapsed to one shape: {shapes}"


def test_updates_per_collect_one_is_the_original_cadence(tmp_path):
    trainer = make_distill_trainer(tmp_path)
    trees, _ = trainer.collect()
    slices = trainer.split_for_updates(trees)
    assert len(slices) == 1 and slices[0] is trees


def test_each_slice_trains_normally(tmp_path):
    trainer = make_distill_trainer(tmp_path, updates_per_collect=3)
    trees, _ = trainer.collect()
    for slice_ in trainer.split_for_updates(trees):
        stats = trainer.update(slice_)
        assert stats.policy_rows > 0
        assert math.isfinite(stats.distill_kl) and stats.distill_kl > 0
        assert stats.distill_skipped_steps == 0.0


def test_a_wave_collects_the_multiple_and_each_slice_is_the_per_update_batch(tmp_path):
    """The two halves of the knob must agree: collect Nx, split into N."""

    cells = (
        GameScheduleCell(hand_size=3, num_players=3, games=3),
        GameScheduleCell(hand_size=4, num_players=3, games=3),
    )
    single = make_distill_trainer(tmp_path, schedule_cells=cells)
    batched = make_distill_trainer(
        tmp_path, schedule_cells=cells, updates_per_collect=4
    )
    one, _ = single.collect()
    many, _ = batched.collect()

    assert len(many) == 4 * len(one), "wave did not collect the multiple"
    for slice_ in batched.split_for_updates(many):
        assert len(slice_) == len(one), "a slice is not the per-update batch"
