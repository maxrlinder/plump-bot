"""End-to-end smoke: collect -> update -> snapshot -> league -> eval."""

from __future__ import annotations

import torch

from plump.evaluation import DealBank, evaluate_policy
from plump.policies import RandomPolicy
from plump.seq.config import (
    BranchBudgetConfig,
    BranchRuleConfig,
    GameScheduleCell,
    SeqModelConfig,
    SeqTrainingConfig,
)
from plump.seq.model import SeqPlumpModel
from plump.seq.policy import SeqModelPolicy
from plump.seq.trainer import SeqTrainer


def test_two_iterations_and_eval(tmp_path):
    torch.manual_seed(0)
    model_config = SeqModelConfig(d_model=32, n_layers=1, n_heads=2, d_ff=64)
    train_config = SeqTrainingConfig(
        schedule_cells=(
            GameScheduleCell(hand_size=3, num_players=3),
            GameScheduleCell(hand_size=5),
        ),
        branch_rule=BranchRuleConfig(bid_top_k=3),
        branch_budget=BranchBudgetConfig(branch_rate=1.0),
        microbatch_positions=1024,
        snapshot_every=1,
        seed=7,
    )
    trainer = SeqTrainer(SeqPlumpModel(model_config), train_config, device="cpu")

    for iteration in (1, 2):
        trainer.iteration = iteration
        trees, summary = trainer.collect()
        assert summary.leaves >= len(trees)
        stats = trainer.update(trees)
        assert stats.positions > 0
        trainer.maybe_snapshot(tmp_path)

    assert trainer.league.has_snapshots()

    bank = DealBank.generate(
        player_counts=(3,),
        hand_sizes=(3,),
        deals_per_configuration=2,
        seed=5,
    )
    policy = SeqModelPolicy(trainer.model, device="cpu", greedy=True)
    report = evaluate_policy(policy, RandomPolicy(), bank, batch_size=16)
    assert report.rounds > 0
    assert -20.0 < report.macro_relative_reward < 20.0
