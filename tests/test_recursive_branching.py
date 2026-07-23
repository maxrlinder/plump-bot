from __future__ import annotations

import random
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from plump.env import PlumpEnv
from plump.modeling import ModelConfig
from plump.modeling.torch_model import PlumpTransformerModel
from plump.policies import RandomPolicy
from plump.rounds import RoundSpec, round_game_config
from plump.state import BidAction, Phase
from plump.training import PPOTrainer, TrainingConfig
from plump.training.env_workers import RoundResult
from plump.training.ppo import (
    _RecursiveBranchDecision,
    _branch_action_index,
)


def _tiny_branch_trainer() -> PPOTrainer:
    model_config = ModelConfig(
        max_seq_len=32,
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_ff=32,
        context_hidden_dim=32,
        dropout=0.0,
    )
    torch.manual_seed(1)
    model = PlumpTransformerModel(model_config)
    # Uniform raw policies make the branching shape independent of model
    # initialization and keep this collector integration test deterministic.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    config = TrainingConfig(
        player_counts=(3,),
        hand_sizes=(3,),
        rounds_per_configuration=2,
        num_envs=1,
        ppo_epochs=1,
        minibatch_size=64,
        learning_rate=1e-3,
        self_play_fraction=0.5,
        historical_fraction=0.5,
        heuristic_fraction=0.0,
        mixed_fraction=0.0,
        branch_rollouts=True,
        branch_decision_budget_per_arm=48,
        branch_update_decision_budget_per_arm=64,
        branch_max_active=64,
        branch_bid_max_actions=4,
        entropy_coef=0.0,
        seed=3,
        device="cpu",
        model_config=model_config,
    )
    trainer = PPOTrainer(model, config)
    trainer.add_historical_policy(RandomPolicy(11), snapshot_id="random")
    return trainer


@pytest.fixture(scope="module")
def collected_branch_batch():
    trainer = _tiny_branch_trainer()
    buffer = trainer.collect_rollouts()
    return trainer, buffer


def test_matched_arms_have_exactly_equal_samples_and_loss_mass(
    collected_branch_batch,
) -> None:
    _, buffer = collected_branch_batch

    sample_counts = Counter(sample.opponent_arm for sample in buffer.samples)
    outcome_counts = Counter(
        outcome.opponent_arm for outcome in buffer.round_outcomes
    )
    weight_by_arm = {
        arm: sum(
            sample.round_weight
            for sample in buffer.samples
            if sample.opponent_arm == arm
        )
        for arm in ("self", "historical")
    }

    assert sample_counts["self"] == sample_counts["historical"] > 0
    assert outcome_counts == {"self": 1, "historical": 1}
    assert weight_by_arm["self"] == pytest.approx(0.5)
    assert weight_by_arm["historical"] == pytest.approx(0.5)
    assert {sample.opponent_arm for sample in buffer.samples} == {
        "self",
        "historical",
    }


def test_reserved_decisions_and_terminal_leaf_stats_are_exact(
    collected_branch_batch,
) -> None:
    trainer, buffer = collected_branch_batch
    stats = trainer.last_collection_stats
    branch_samples = [
        sample
        for sample in buffer.samples
        if sample.branch_action_values is not None
    ]

    assert branch_samples
    assert stats.branch_self_decisions <= (
        trainer.config.branch_decision_budget_per_arm
    )
    assert stats.branch_historical_decisions <= (
        trainer.config.branch_decision_budget_per_arm
    )
    # This fixed matched deal leaves the same unused reservation in each arm.
    assert stats.branch_self_decisions == stats.branch_historical_decisions
    assert stats.branch_root_hands == 2
    assert stats.branch_roots_available >= len(branch_samples)
    assert stats.branch_roots_expanded == len(branch_samples)

    # Each B-way expansion replaces one pending terminal path with B paths.
    # Complete action values exist only after every descendant path resolves,
    # so this equality indirectly checks that no retained leaf was truncated.
    expected_terminal_paths = len(buffer.round_outcomes) + sum(
        len(sample.branch_action_values) - 1
        for sample in branch_samples
    )
    assert stats.branch_terminal_rollouts == expected_terminal_paths
    for sample in branch_samples:
        assert sample.branch_candidate_action_indices is not None
        assert sample.branch_prior_probabilities is not None
        assert len(sample.branch_action_values) == len(
            sample.branch_candidate_action_indices
        )
        assert len(sample.branch_action_values) == len(
            sample.branch_prior_probabilities
        )
        assert sample.return_target is not None


def test_bids_use_top_four_and_plays_cover_every_legal_action(
    collected_branch_batch,
) -> None:
    trainer, _ = collected_branch_batch
    env = PlumpEnv(
        round_game_config(RoundSpec(3, 5), bidding_start_player=0),
        seed=17,
    )
    env.reset()
    path = SimpleNamespace(env=env, rng=random.Random(23))

    # Bids use only the four highest raw-policy actions.
    bid_probabilities = [0.02, 0.05, 0.08, 0.15, 0.25, 0.45]
    bid_sample = SimpleNamespace(
        old_policy_probabilities=bid_probabilities
    )
    raw_bid = BidAction(env.current_player(), 1)
    bid_candidates, bid_masses, bid_groups = trainer._recursive_branch_candidates(
        path,
        bid_sample,
        raw_bid,
    )

    assert [action.bid for action in bid_candidates] == [5, 4, 3, 2]
    selected_total = sum(bid_probabilities[2:])
    assert bid_masses == pytest.approx(
        tuple(
            bid_probabilities[index] / selected_total
            for index in (5, 4, 3, 2)
        )
    )
    assert bid_groups == ((5,), (4,), (3,), (2,))
    assert sum(bid_masses) == pytest.approx(1.0)

    while env.phase() == Phase.BIDDING:
        env.step(env.legal_actions()[0])
    legal_plays = env.legal_actions()
    play_probabilities = [0.0] * 52
    for offset, action in enumerate(legal_plays, start=1):
        play_probabilities[_branch_action_index(action)] = float(offset)
    play_sample = SimpleNamespace(
        old_policy_probabilities=play_probabilities
    )
    play_candidates, play_masses, play_groups = trainer._recursive_branch_candidates(
        path,
        play_sample,
        legal_plays[0],
    )

    assert play_candidates == tuple(legal_plays)
    expected_total = sum(range(1, len(legal_plays) + 1))
    assert play_masses == pytest.approx(
        tuple(
            value / expected_total
            for value in range(1, len(legal_plays) + 1)
        )
    )
    assert play_groups == tuple(
        (_branch_action_index(action),)
        for action in legal_plays
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("entropy_coef", 0.01),
        ("mmd_enabled", True),
        ("explore_temperature_fraction", 0.1),
        ("explore_uniform_round_probability", 0.1),
        ("branch_support_floor", 0.05),
    ),
)
def test_recursive_mode_rejects_non_branch_exploration(
    collected_branch_batch,
    field: str,
    value: object,
) -> None:
    trainer, _ = collected_branch_batch
    config = replace(trainer.config, **{field: value})
    with pytest.raises(ValueError, match="branch|Branch|exploration"):
        PPOTrainer(PlumpTransformerModel(config.model_config), config)


def test_recursive_backup_uses_policy_mass_not_leaf_average(
    collected_branch_batch,
) -> None:
    trainer, _ = collected_branch_batch
    env = PlumpEnv(round_game_config(RoundSpec(3, 3)), seed=29)
    env.reset()
    root = SimpleNamespace(
        episode=SimpleNamespace(env=env),
        samples=[],
        decisions=[],
    )
    sample = SimpleNamespace(
        ppo_policy_enabled=True,
        position_intercept=0.5,
        old_value=1.0,
    )
    actions = (BidAction(0, 0), BidAction(0, 1))
    decision = _RecursiveBranchDecision(
        sample=sample,
        candidate_actions=actions,
        action_groups=((0,), (1,)),
        edge_probabilities=(0.25, 0.75),
        upstream=None,
        prefix_samples=[],
        on_policy_spine=False,
        canonical_action_index=None,
        retain_target=True,
    )
    terminal = RoundResult(
        round_scores={0: 0, 1: 0, 2: 0},
        bids=[],
        tricks_won={0: 0, 1: 0, 2: 0},
        hand_size=3,
    )

    trainer._resolve_recursive_edge(decision, 0, 2.0, terminal, root)
    assert root.samples == []
    trainer._resolve_recursive_edge(decision, 1, 6.0, terminal, root)

    assert sample.return_target == pytest.approx(5.0)
    assert sample.value_target == pytest.approx(4.5)
    assert sample.advantage_target == pytest.approx(4.0)
    assert sample.branch_prior_probabilities == [0.25, 0.75]
    assert sample.branch_action_values == [2.0, 6.0]
    assert root.samples == [sample]
    assert root.decisions == [decision]


def test_direct_counterfactual_advantages_improve_high_value_actions(
    collected_branch_batch,
) -> None:
    trainer, _ = collected_branch_batch
    probabilities = torch.tensor([0.02, 0.05, 0.08, 0.15, 0.25, 0.45])
    logits = probabilities.log().unsqueeze(0).requires_grad_()
    output = SimpleNamespace(
        masked_bid_logits=logits,
        masked_card_logits=torch.full((1, 52), float("-inf")),
    )
    sample = SimpleNamespace(
        phase="bid",
        branch_candidate_action_indices=list(range(6)),
        branch_action_groups=[[index] for index in range(6)],
        branch_action_values=[-3.0, -2.0, -1.0, 1.0, 2.0, 5.0],
        branch_prior_probabilities=probabilities.tolist(),
    )

    loss, kl, target_entropy = trainer._branch_policy_terms(output, [sample])

    assert kl.item() == pytest.approx(0.0, abs=1e-7)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert target_entropy.item() == 0.0
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    # Gradient descent raises the highest-value action and lowers the worst.
    assert logits.grad[0, 5].item() < 0.0
    assert logits.grad[0, 0].item() > 0.0


def test_branch_actor_rows_replace_ppo_while_unbranched_spine_rows_keep_it(
    collected_branch_batch,
) -> None:
    _, buffer = collected_branch_batch
    branch_samples = [
        sample
        for sample in buffer.samples
        if sample.branch_action_values is not None
    ]
    spine_samples = [
        sample
        for sample in buffer.samples
        if sample.branch_action_values is None
    ]

    assert branch_samples
    assert spine_samples
    assert all(not sample.ppo_policy_enabled for sample in branch_samples)
    assert all(sample.ppo_policy_enabled for sample in spine_samples)
