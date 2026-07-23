from __future__ import annotations

import random
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import plump.training.ppo as ppo_module
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


def _tiny_branch_trainer(**config_overrides) -> PPOTrainer:
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
    config = replace(config, **config_overrides)
    trainer = PPOTrainer(model, config)
    trainer.add_historical_policy(RandomPolicy(11), snapshot_id="random")
    return trainer


@pytest.fixture(scope="module")
def collected_branch_batch():
    trainer = _tiny_branch_trainer()
    buffer = trainer.collect_rollouts()
    return trainer, buffer


def test_matched_arms_have_independent_rows_and_exactly_equal_loss_mass(
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

    assert sample_counts["self"] > 0
    assert sample_counts["historical"] > 0
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
    # Each arm independently admits only complete expansions, so either may
    # leave a small unusable tail below the common cap.
    assert stats.branch_self_decisions > 0
    assert stats.branch_historical_decisions > 0
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


def test_every_admitted_expansion_owns_a_direct_update_slot() -> None:
    trainer = _tiny_branch_trainer(
        branch_decision_budget_per_arm=80,
        branch_update_decision_budget_per_arm=4,
        branch_tree_decision_budget_per_arm=20,
        branch_tree_update_decisions_per_arm=1,
    )
    buffer = trainer.collect_rollouts()
    stats = trainer.last_collection_stats
    branch_by_arm = Counter(
        sample.opponent_arm
        for sample in buffer.samples
        if sample.branch_action_values is not None
    )

    assert 0 < branch_by_arm["self"] <= 4
    assert 0 < branch_by_arm["historical"] <= 4
    assert sum(branch_by_arm.values()) == stats.branch_roots_expanded
    # A cap of one direct state per arm per matched deal must force the
    # collector to use multiple independently dealt roots.
    assert stats.branch_root_hands >= 4
    assert sum(
        sample.round_weight
        for sample in buffer.samples
        if sample.opponent_arm == "self"
    ) == pytest.approx(0.5)
    assert sum(
        sample.round_weight
        for sample in buffer.samples
        if sample.opponent_arm == "historical"
    ) == pytest.approx(0.5)


def test_each_tree_obeys_its_decision_and_direct_row_caps() -> None:
    trainer = _tiny_branch_trainer()
    episodes = trainer._new_matched_branch_episodes(
        RoundSpec(3, 3),
        pair_id=0,
    )

    roots = trainer._collect_recursive_tree_pair(
        episodes,
        decision_budgets={"self": 25, "historical": 25},
        retain_slots={"self": 1, "historical": 1},
    )

    for root in roots.values():
        direct_rows = [
            sample
            for sample in root.samples
            if sample.branch_action_values is not None
        ]
        assert root.rollout_decisions <= 25
        assert len(direct_rows) == 1


def test_collection_stops_once_both_direct_caps_are_full() -> None:
    trainer = _tiny_branch_trainer(
        branch_decision_budget_per_arm=80,
        branch_update_decision_budget_per_arm=1,
        branch_tree_decision_budget_per_arm=25,
        branch_tree_update_decisions_per_arm=1,
    )

    buffer = trainer.collect_rollouts()

    assert len(buffer.round_outcomes) == 2
    assert trainer.last_collection_stats.branch_root_hands == 2
    assert trainer.last_collection_stats.branch_roots_expanded == 2
    assert sum(
        sample.branch_action_values is not None
        for sample in buffer.samples
    ) == 2


def test_remaining_budget_excludes_only_each_players_final_card(
    collected_branch_batch,
) -> None:
    trainer, _ = collected_branch_batch
    env = PlumpEnv(
        round_game_config(RoundSpec(3, 3), bidding_start_player=0),
        seed=41,
    )
    env.reset()

    # Three bids plus two genuine card choices per player. The third card in
    # each hand is deterministic and therefore outside the neural budget.
    assert trainer._remaining_round_decisions(env) == 9
    assert not trainer._is_forced_final_card_play(env)

    while env.phase() == Phase.BIDDING:
        env.step(env.legal_actions()[0])
    assert trainer._remaining_round_decisions(env) == 6

    while any(
        len(hand) > 1
        for hand in env.state.current_round.current_hands.values()
    ):
        env.step(env.legal_actions()[0])
    assert env.phase() == Phase.PLAYING
    assert trainer._is_forced_final_card_play(env)
    assert trainer._remaining_round_decisions(env) == 0


def test_recursive_collector_never_forwards_a_players_final_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _tiny_branch_trainer()
    original_request = ppo_module.build_decision_request
    original_random_act = RandomPolicy.act

    def guarded_request(env, **kwargs):
        assert not PPOTrainer._is_forced_final_card_play(env)
        return original_request(env, **kwargs)

    def guarded_random_act(self, env, *, rng=None):
        assert not PPOTrainer._is_forced_final_card_play(env)
        return original_random_act(self, env, rng=rng)

    monkeypatch.setattr(
        ppo_module,
        "build_decision_request",
        guarded_request,
    )
    monkeypatch.setattr(RandomPolicy, "act", guarded_random_act)
    buffer = trainer.collect_rollouts()

    play_samples = [
        sample
        for sample in buffer.samples
        if sample.phase == "play"
    ]
    assert play_samples
    assert all(
        sample.observation is not None
        and len(sample.observation.my_hand) >= 2
        for sample in play_samples
    )
    assert trainer.last_collection_stats.branch_terminal_rollouts > 0


def test_current_policy_rows_can_share_a_paired_random_tape() -> None:
    trainer = _tiny_branch_trainer()
    env = PlumpEnv(
        round_game_config(RoundSpec(3, 3), bidding_start_player=0),
        seed=71,
    )
    env.reset()
    requests = [
        ppo_module.build_decision_request(
            env,
            episode_id=episode_id,
            opponent_arm="self",
            policy_ref="current",
            model_config=trainer.config.model_config,
            include_game_context=trainer.config.include_game_context,
            collect=True,
        )
        for episode_id in (1, 2)
    ]

    rows = trainer._forward_decision_rows(
        requests,
        rngs=[random.Random(19), random.Random(19)],
    )

    assert rows[0][1] == rows[1][1]
    assert rows[0][0] is not None
    assert rows[1][0] is not None
    assert rows[0][0].action_index == rows[1][0].action_index


def test_paired_inverse_cdf_never_selects_masked_boundary_actions() -> None:
    trainer = _tiny_branch_trainer()
    env = PlumpEnv(
        round_game_config(RoundSpec(3, 3), bidding_start_player=0),
        seed=73,
    )
    env.reset()
    while env.phase() == Phase.BIDDING:
        env.step(env.legal_actions()[0])
    legal = set(env.legal_actions())
    requests = [
        ppo_module.build_decision_request(
            env,
            episode_id=episode_id,
            opponent_arm="self",
            policy_ref="current",
            model_config=trainer.config.model_config,
            include_game_context=trainer.config.include_game_context,
            collect=True,
        )
        for episode_id in (1, 2)
    ]
    boundary_rngs = [
        SimpleNamespace(random=lambda: 0.0),
        SimpleNamespace(random=lambda: 1.0 - 1e-15),
    ]

    rows = trainer._forward_decision_rows(
        requests,
        rngs=boundary_rngs,
    )

    assert rows[0][1] in legal
    assert rows[1][1] in legal


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

    # Bids use the top actions, but the unconditional sampled bid always owns
    # one of the four slots so a capped node has a valid full-policy backup.
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

    assert [action.bid for action in bid_candidates] == [5, 4, 3, 1]
    selected_total = sum(
        bid_probabilities[index]
        for index in (5, 4, 3, 1)
    )
    assert bid_masses == pytest.approx(
        tuple(
            bid_probabilities[index] / selected_total
            for index in (5, 4, 3, 1)
        )
    )
    assert bid_groups == ((5,), (4,), (3,), (1,))
    assert sum(bid_masses) == pytest.approx(1.0)

    top_sampled_candidates, _, _ = (
        trainer._recursive_branch_candidates(
            path,
            bid_sample,
            BidAction(env.current_player(), 5),
        )
    )
    assert [action.bid for action in top_sampled_candidates] == [5, 4, 3]

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
        sampled_action_index=0,
        candidate_probability_mass=1.0,
        deterministic_candidate_count=2,
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


def test_partial_top_k_bid_backup_reuses_top_set_as_control_variate(
    collected_branch_batch,
) -> None:
    trainer, _ = collected_branch_batch
    env = PlumpEnv(round_game_config(RoundSpec(3, 3)), seed=31)
    env.reset()
    root = SimpleNamespace(
        episode=SimpleNamespace(env=env),
        samples=[],
    )
    sample = SimpleNamespace(
        ppo_policy_enabled=True,
        position_intercept=0.5,
        old_value=1.0,
        old_policy_probabilities=[0.2, 0.3, 0.5],
    )
    decision = _RecursiveBranchDecision(
        sample=sample,
        candidate_actions=(BidAction(0, 0), BidAction(0, 1)),
        action_groups=((0,), (1,)),
        edge_probabilities=(0.4, 0.6),
        upstream=None,
        prefix_samples=[],
        on_policy_spine=False,
        canonical_action_index=None,
        sampled_action_index=1,
        candidate_probability_mass=0.5,
        deterministic_candidate_count=1,
    )
    terminal = RoundResult(
        round_scores={0: 0, 1: 0, 2: 0},
        bids=[],
        tricks_won={0: 0, 1: 0, 2: 0},
        hand_size=3,
    )

    trainer._resolve_recursive_edge(decision, 0, 2.0, terminal, root)
    trainer._resolve_recursive_edge(decision, 1, 6.0, terminal, root)

    # Exact top-set contribution 0.2*2 plus the unweighted tail sample 6.
    # Across raw-policy samples, the indicator on tail membership supplies
    # the omitted sum_a p(a)Q(a) without discarding known top-action values.
    assert sample.return_target == pytest.approx(6.4)
    assert sample.value_target == pytest.approx(5.9)
    assert sample.branch_action_values == [2.0, 6.0]
    assert root.samples == [sample]


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


def test_neurd_gives_a_rare_good_action_a_direct_nonvanishing_gradient() -> None:
    trainer = _tiny_branch_trainer(
        branch_policy_objective="neurd",
        branch_neurd_kl_coef=1.0,
    )
    probabilities = torch.tensor([1e-6, 1.0 - 1e-6])
    logits = probabilities.log().unsqueeze(0).requires_grad_()
    output = SimpleNamespace(
        masked_bid_logits=logits,
        masked_card_logits=torch.full((1, 52), float("-inf")),
    )
    sample = SimpleNamespace(
        phase="bid",
        branch_candidate_action_indices=[0, 1],
        branch_action_groups=[[0], [1]],
        branch_action_values=[1.0, 0.0],
        branch_prior_probabilities=probabilities.tolist(),
    )

    loss, kl, _ = trainer._branch_policy_terms(output, [sample])
    loss.backward()

    assert kl.item() == pytest.approx(0.0, abs=1e-6)
    assert logits.grad is not None
    # A normal softmax policy-gradient signal here is O(1e-6). NeuRD applies
    # the signed local counterfactual advantage directly to the rare action.
    assert logits.grad[0, 0].item() < -0.2
    assert logits.grad[0, 1].item() > 0.2
    assert logits.grad.sum().item() == pytest.approx(0.0, abs=1e-6)


def test_neurd_packed_terms_match_scalar_objective_and_gradient() -> None:
    trainer = _tiny_branch_trainer(
        branch_policy_objective="neurd",
        branch_neurd_kl_coef=0.7,
    )
    probabilities = torch.tensor([0.02, 0.05, 0.08, 0.15, 0.25, 0.45])
    logits = probabilities.log().unsqueeze(0).requires_grad_()
    output = SimpleNamespace(
        masked_bid_logits=logits,
        masked_card_logits=torch.full((1, 52), float("-inf")),
    )
    values = [-3.0, -2.0, -1.0, 1.0, 2.0, 5.0]
    sample = SimpleNamespace(
        phase="bid",
        branch_candidate_action_indices=list(range(6)),
        branch_action_groups=[[index] for index in range(6)],
        branch_action_values=values,
        branch_prior_probabilities=probabilities.tolist(),
    )

    scalar_loss, scalar_kl, _ = trainer._branch_policy_terms(
        output,
        [sample],
    )
    packed_loss, packed_kl, packed_entropy = (
        trainer._packed_branch_policy_terms(
            output,
            phase_is_bid=torch.tensor([True]),
            candidate_indices=torch.arange(6).unsqueeze(0),
            candidate_mask=torch.ones((1, 6), dtype=torch.bool),
            old_probabilities=probabilities.unsqueeze(0),
            action_values=torch.tensor(values).unsqueeze(0),
        )
    )
    scalar_gradient = torch.autograd.grad(
        scalar_loss.sum(),
        logits,
        retain_graph=True,
    )[0]
    packed_gradient = torch.autograd.grad(
        packed_loss.sum(),
        logits,
    )[0]

    torch.testing.assert_close(packed_loss, scalar_loss)
    torch.testing.assert_close(packed_kl, scalar_kl)
    torch.testing.assert_close(packed_gradient, scalar_gradient)
    assert packed_entropy.item() == 0.0


def test_capped_bid_kl_anchors_excluded_probability_mass() -> None:
    trainer = _tiny_branch_trainer(
        branch_policy_objective="neurd",
    )
    old = torch.tensor([0.10, 0.20, 0.10, 0.20, 0.20, 0.20])
    changed_logits = old.log()
    changed_logits[5] += 1.0
    logits = changed_logits.unsqueeze(0).requires_grad_()
    output = SimpleNamespace(
        masked_bid_logits=logits,
        masked_card_logits=torch.full((1, 52), float("-inf")),
    )
    candidate_prior = (old[:2] / old[:2].sum()).tolist()
    full_prior = [*old.tolist(), *([0.0] * (52 - len(old)))]
    sample = SimpleNamespace(
        phase="bid",
        branch_candidate_action_indices=[0, 1],
        branch_action_groups=[[0], [1]],
        branch_action_values=[-1.0, 2.0],
        branch_prior_probabilities=candidate_prior,
        old_policy_probabilities=full_prior,
    )

    scalar_loss, scalar_kl, _ = trainer._branch_policy_terms(
        output,
        [sample],
    )
    packed_loss, packed_kl, _ = trainer._packed_branch_policy_terms(
        output,
        phase_is_bid=torch.tensor([True]),
        candidate_indices=torch.tensor([[0, 1]]),
        candidate_mask=torch.tensor([[True, True]]),
        old_probabilities=torch.tensor([candidate_prior]),
        action_values=torch.tensor([[-1.0, 2.0]]),
        full_old_probabilities=torch.tensor([full_prior]),
    )

    # Candidate-conditional KL would be exactly zero because neither
    # candidate logit moved relative to the other. Full-policy KL detects the
    # excluded bid taking probability mass from the selected set.
    assert scalar_kl.item() > 0.0
    torch.testing.assert_close(packed_kl, scalar_kl)
    torch.testing.assert_close(packed_loss, scalar_loss)


def test_packed_branch_terms_match_scalar_objective_and_gradient(
    collected_branch_batch,
) -> None:
    trainer, _ = collected_branch_batch
    probabilities = torch.tensor([0.02, 0.05, 0.08, 0.15, 0.25, 0.45])
    logits = probabilities.log().unsqueeze(0).requires_grad_()
    output = SimpleNamespace(
        masked_bid_logits=logits,
        masked_card_logits=torch.full((1, 52), float("-inf")),
    )
    values = [-3.0, -2.0, -1.0, 1.0, 2.0, 5.0]
    sample = SimpleNamespace(
        phase="bid",
        branch_candidate_action_indices=list(range(6)),
        branch_action_groups=[[index] for index in range(6)],
        branch_action_values=values,
        branch_prior_probabilities=probabilities.tolist(),
    )

    scalar_loss, scalar_kl, _ = trainer._branch_policy_terms(
        output,
        [sample],
    )
    packed_loss, packed_kl, packed_entropy = (
        trainer._packed_branch_policy_terms(
            output,
            phase_is_bid=torch.tensor([True]),
            candidate_indices=torch.arange(6).unsqueeze(0),
            candidate_mask=torch.ones((1, 6), dtype=torch.bool),
            old_probabilities=probabilities.unsqueeze(0),
            action_values=torch.tensor(values).unsqueeze(0),
        )
    )
    scalar_gradient = torch.autograd.grad(
        scalar_loss.sum(),
        logits,
        retain_graph=True,
    )[0]
    packed_gradient = torch.autograd.grad(
        packed_loss.sum(),
        logits,
    )[0]

    torch.testing.assert_close(packed_loss, scalar_loss)
    torch.testing.assert_close(packed_kl, scalar_kl)
    torch.testing.assert_close(packed_gradient, scalar_gradient)
    assert packed_entropy.item() == 0.0


@pytest.mark.parametrize("objective", ("ppo", "neurd"))
def test_packed_branch_padding_and_nonbranch_rows_stay_finite(
    objective: str,
) -> None:
    trainer = _tiny_branch_trainer(
        branch_policy_objective=objective,
    )
    policies = torch.tensor(
        [
            [0.02, 0.05, 0.08, 0.15, 0.25, 0.45],
            [0.10, 0.05, 0.10, 0.10, 0.20, 0.45],
            [0.20, 0.10, 0.10, 0.20, 0.20, 0.20],
        ]
    )
    logits = policies.log().requires_grad_()
    output = SimpleNamespace(
        masked_bid_logits=logits,
        masked_card_logits=torch.full((3, 52), float("-inf")),
    )
    samples = [
        SimpleNamespace(
            phase="bid",
            branch_candidate_action_indices=list(range(6)),
            branch_action_groups=[[index] for index in range(6)],
            branch_action_values=[-3.0, -2.0, -1.0, 1.0, 2.0, 5.0],
            branch_prior_probabilities=policies[0].tolist(),
        ),
        SimpleNamespace(
            phase="bid",
            branch_candidate_action_indices=[0, 5],
            branch_action_groups=[[0], [5]],
            branch_action_values=[-1.0, 3.0],
            branch_prior_probabilities=[0.2, 0.8],
        ),
        SimpleNamespace(
            phase="bid",
            branch_candidate_action_indices=None,
            branch_action_groups=None,
            branch_action_values=None,
            branch_prior_probabilities=None,
        ),
    ]
    candidate_indices = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [0, 5, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    candidate_mask = torch.tensor(
        [
            [True, True, True, True, True, True],
            [True, True, False, False, False, False],
            [False, False, False, False, False, False],
        ]
    )
    old_probabilities = torch.tensor(
        [
            policies[0].tolist(),
            [0.2, 0.8, 0.0, 0.0, 0.0, 0.0],
            [0.0] * 6,
        ]
    )
    action_values = torch.tensor(
        [
            [-3.0, -2.0, -1.0, 1.0, 2.0, 5.0],
            [-1.0, 3.0, 0.0, 0.0, 0.0, 0.0],
            [0.0] * 6,
        ]
    )

    scalar_loss, scalar_kl, _ = trainer._branch_policy_terms(
        output,
        samples,
    )
    packed_loss, packed_kl, packed_entropy = (
        trainer._packed_branch_policy_terms(
            output,
            phase_is_bid=torch.ones(3, dtype=torch.bool),
            candidate_indices=candidate_indices,
            candidate_mask=candidate_mask,
            old_probabilities=old_probabilities,
            action_values=action_values,
        )
    )
    scalar_gradient = torch.autograd.grad(
        scalar_loss.sum(),
        logits,
        retain_graph=True,
    )[0]
    packed_gradient = torch.autograd.grad(
        packed_loss.sum(),
        logits,
    )[0]

    assert torch.isfinite(packed_loss).all()
    assert torch.isfinite(packed_kl).all()
    torch.testing.assert_close(packed_loss, scalar_loss)
    torch.testing.assert_close(packed_kl, scalar_kl)
    torch.testing.assert_close(packed_gradient, scalar_gradient)
    assert torch.equal(packed_entropy, torch.zeros(3))


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


def test_branch_kl_overshoot_rolls_back_the_entire_epoch() -> None:
    trainer = _tiny_branch_trainer(
        branch_policy_objective="neurd",
        branch_neurd_regret_coef=1.0,
        branch_kl_cap=1e-12,
        ppo_epochs=1,
    )
    buffer = trainer.collect_rollouts()
    before = {
        key: value.detach().clone()
        for key, value in trainer.model.state_dict().items()
    }

    stats = trainer.update(buffer)

    assert stats.epochs_run == 0
    assert stats.branch_kl <= trainer.config.branch_kl_cap
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0.0, atol=0.0)
