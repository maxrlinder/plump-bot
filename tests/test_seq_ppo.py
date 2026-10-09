"""Branch-free PPO estimator, weighting, actor-pool, and precision tests."""

from __future__ import annotations

import copy
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from plump.seq.config import (
    BranchBudgetConfig,
    BranchRuleConfig,
    GameScheduleCell,
    NEXT_BID,
    NEXT_PLAY,
    NUM_CARDS,
    RolloutOptions,
    SLOT_CARD,
    SLOT_NEXT_ACTOR,
    SLOT_NEXT_PHASE,
    SLOT_REL_PLAYER,
    SLOT_TYPE,
    TOKEN_BID,
    TOKEN_HAND,
    TOKEN_PLAY,
    TOKEN_TRICK_WIN,
    TOKEN_TURN,
    SeqModelConfig,
    SeqTrainingConfig,
)
from plump.seq.model import (
    ORACLE_CARD_ORDER_HEAD_KEYS,
    SeqPPOOracleCritic,
    SeqPlumpModel,
)
from plump.seq.ppo import (
    build_ppo_training_batch,
    normalize_ppo_advantages,
    build_ppo_training_groups,
    center_zero_sum_values,
    oracle_gae_advantages,
    plackett_luce_card_order_terms,
    ppo_clipped_terms,
    ppo_policy_kl,
)
from plump.seq.trainer import SeqTrainer, SeqUpdateStats, _grad_norm
from plump.seq.tokens import IGNORE_LABEL, TOKEN_WIDTH


MODEL = SeqModelConfig(d_model=32, n_layers=1, n_heads=2, d_ff=64)


def ppo_config(
    *,
    cells=None,
    actors=1,
    precision="fp32",
    kv_dtype="fp16",
    critic_epochs=1,
    bucket_width=0,
    critic_bucket_width=None,
    suit_coef=0.0,
    rank_boundary_coef=0.0,
    trick_coef=0.0,
    next_winner_coef=0.0,
    oracle_card_order_coef=0.0,
    detach_actor_auxiliary=False,
    auxiliary_positions="policy",
    value_positions="policy",
    dense_supervision_start="sequence",
    advantage_estimator="monte_carlo",
    gae_lambda=0.95,
    critic_mode="oracle",
    policy_kl_cap=1.0,
    policy_kl_p99_cap=1.0,
    auxiliary_grad_fraction=0.0,
    actor_value_coef=0.0,
):
    return SeqTrainingConfig(
        schedule_cells=cells
        or (GameScheduleCell(hand_size=3, num_players=3, games=2),),
        branch_rule=BranchRuleConfig(bid_top_k=2, play_mode="none"),
        branch_budget=BranchBudgetConfig(branch_rate=0.5),
        rollout=RolloutOptions(
            opponent_mode="off",
            opponent_fraction=0.0,
            deals_per_batch=2,
            max_cache_rows=256,
        ),
        policy_objective="ppo",
        ppo_trainable_policies=actors,
        ppo_sequence_bucket_width=bucket_width,
        ppo_critic_sequence_bucket_width=critic_bucket_width,
        ppo_critic_epochs=critic_epochs,
        ppo_critic_mode=critic_mode,
        ppo_advantage_estimator=advantage_estimator,
        ppo_gae_lambda=gae_lambda,
        ppo_actor_auxiliary_detach_hidden=detach_actor_auxiliary,
        ppo_auxiliary_positions=auxiliary_positions,
        dense_supervision_start=dense_supervision_start,
        value_positions=value_positions,
        microbatch_positions=512,
        policy_kl_cap=policy_kl_cap,
        policy_kl_p99_cap=policy_kl_p99_cap,
        precision=precision,
        kv_dtype=kv_dtype,
        suit_coef=suit_coef,
        rank_boundary_coef=rank_boundary_coef,
        trick_coef=trick_coef,
        next_winner_coef=next_winner_coef,
        oracle_card_order_coef=oracle_card_order_coef,
        ppo_auxiliary_grad_fraction=auxiliary_grad_fraction,
        ppo_actor_value_coef=actor_value_coef,
    )


def test_oracle_gae_uses_next_decision_state_and_has_exact_endpoints():
    tokens = np.zeros((1, 7, TOKEN_WIDTH), dtype=np.int64)
    decision_positions = np.array([0, 3, 5])
    tokens[0, decision_positions, SLOT_NEXT_PHASE] = [
        NEXT_BID,
        NEXT_PLAY,
        NEXT_PLAY,
    ]
    values = np.zeros((1, 7, 4), dtype=np.float32)
    values[0, 0, :3] = [1.0, -2.0, 1.0]
    # These are intervening event/bookkeeping tokens, not successor states.
    values[0, 1, :3] = [1000.0, 2000.0, 3000.0]
    values[0, 3, :3] = [4.0, -1.0, -3.0]
    values[0, 4, :3] = [-4000.0, -5000.0, -6000.0]
    values[0, 5, :3] = [2.0, 3.0, -5.0]
    returns = np.array([[7.0, -4.0, -3.0, 0.0]], dtype=np.float32)
    player_counts = np.array([3], dtype=np.int8)

    monte_carlo = oracle_gae_advantages(
        tokens, values, returns, player_counts, 1.0
    )
    np.testing.assert_allclose(
        monte_carlo[0, decision_positions, :3],
        returns[0, :3] - values[0, decision_positions, :3],
    )

    one_step = oracle_gae_advantages(
        tokens, values, returns, player_counts, 0.0
    )
    np.testing.assert_allclose(
        one_step[0, 0, :3], values[0, 3, :3] - values[0, 0, :3]
    )
    np.testing.assert_allclose(
        one_step[0, 3, :3], values[0, 5, :3] - values[0, 3, :3]
    )
    np.testing.assert_allclose(
        one_step[0, 5, :3], returns[0, :3] - values[0, 5, :3]
    )
    assert np.count_nonzero(one_step[0, [1, 2, 4, 6]]) == 0
    assert np.count_nonzero(one_step[..., 3]) == 0

    mixed = oracle_gae_advantages(
        tokens, values, returns, player_counts, 0.5
    )
    expected_middle = one_step[0, 3, :3] + 0.5 * one_step[0, 5, :3]
    expected_first = one_step[0, 0, :3] + 0.5 * expected_middle
    np.testing.assert_allclose(mixed[0, 3, :3], expected_middle)
    np.testing.assert_allclose(mixed[0, 0, :3], expected_first)


def test_oracle_gae_configuration_is_validated():
    ppo_config(advantage_estimator="gae", gae_lambda=0.95).validate()
    with pytest.raises(ValueError, match="ppo_gae_lambda"):
        ppo_config(advantage_estimator="gae", gae_lambda=1.01).validate()
    with pytest.raises(ValueError, match="requires the oracle critic"):
        ppo_config(
            advantage_estimator="gae", critic_mode="independent"
        ).validate()


def test_oracle_gae_runs_through_a_full_ppo_update():
    config = ppo_config(
        cells=(GameScheduleCell(hand_size=3, num_players=3, games=1),),
        advantage_estimator="gae",
        gae_lambda=0.95,
    )
    trainer = SeqTrainer(SeqPlumpModel(MODEL), config, device="cpu")
    trees, _ = trainer.collect()
    stats = trainer.update(trees)

    assert stats.policy_rows > 0
    assert np.isfinite(stats.advantage_mean)
    assert np.isfinite(stats.advantage_std)
    assert stats.advantage_std > 0


@pytest.mark.parametrize("turn_token", ("off", "bid"))
def test_dense_supervision_can_start_strictly_after_each_card_prefix(turn_token):
    model_config = replace(MODEL, turn_token=turn_token)
    config = ppo_config(
        cells=(GameScheduleCell(hand_size=3, num_players=3, games=1),),
        auxiliary_positions="all",
        value_positions="all",
        dense_supervision_start="bidding",
        bucket_width=16,
        critic_bucket_width=16,
    )
    trainer = SeqTrainer(SeqPlumpModel(model_config), config, device="cpu")
    trees, _ = trainer.collect()
    batch = build_ppo_training_batch(trees, model_config, config)

    for group in batch.policy_groups:
        start = model_config.actor_post_deal_position(group.hand_size)
        genuine_length = model_config.seq_len(group.num_players, group.hand_size)
        assert np.all(group.position_weight[:, :start] == 0)
        assert np.all(group.position_weight[:, start:genuine_length] > 0)
        assert np.all(group.position_weight[:, genuine_length:] == 0)
        expected_type = TOKEN_TURN if turn_token == "bid" else TOKEN_BID
        assert np.all(group.tokens[:, start, SLOT_TYPE] == expected_type)

    for group in batch.critic_groups:
        start = model_config.oracle_post_deal_position(
            group.num_players, group.hand_size
        )
        genuine_length = model_config.oracle_seq_len(
            group.num_players, group.hand_size
        )
        assert np.all(group.position_weight[:, :start] == 0)
        assert np.all(group.position_weight[:, start:genuine_length] > 0)
        assert np.all(group.position_weight[:, genuine_length:] == 0)
        expected_type = TOKEN_TURN if turn_token == "bid" else TOKEN_BID
        assert np.all(group.tokens[:, start, SLOT_TYPE] == expected_type)

    assert sum(group.position_weight.sum() for group in batch.policy_groups) == (
        pytest.approx(1.0)
    )
    assert sum(group.position_weight.sum() for group in batch.critic_groups) == (
        pytest.approx(1.0)
    )


def test_ppo_collection_does_not_evaluate_unused_scalar_value_head(monkeypatch):
    trainer = SeqTrainer(SeqPlumpModel(MODEL), ppo_config(), device="cpu")

    def forbidden_value_forward(*_args, **_kwargs):
        raise AssertionError("PPO rollout evaluated the legacy scalar value head")

    monkeypatch.setattr(trainer.model.value_head, "forward", forbidden_value_forward)
    trees, _ = trainer.collect()

    records = [
        record
        for tree in trees
        for leaf in tree.leaves
        for record in leaf.decisions
    ]
    assert records
    assert all(record.old_value == 0.0 for record in records)


@pytest.mark.parametrize("auxiliary_positions", ("all", "policy"))
def test_oracle_next_winner_labels_land_on_the_matching_oracle_token(
    auxiliary_positions,
):
    """The oracle stream is (P-1)*N tokens longer than the actor's.

    Every dense per-position label therefore has to be re-addressed when it
    crosses over, and a label that is merely *shifted* rather than *aligned*
    still looks plausible -- same values, same counts, wrong states. Check the
    landed labels against the oracle's own token stream instead of against the
    actor array they came from.
    """

    config = ppo_config(
        cells=(GameScheduleCell(hand_size=4, num_players=3, games=2),),
        next_winner_coef=0.2,
        auxiliary_positions=auxiliary_positions,
        value_positions="all",
    )
    trainer = SeqTrainer(SeqPlumpModel(MODEL), config, device="cpu")
    trees, _ = trainer.collect()
    batch = build_ppo_training_batch(trees, MODEL, config)

    checked = 0
    for group in batch.critic_groups:
        types = group.tokens[:, :, SLOT_TYPE]
        bids_seen = np.cumsum(types == TOKEN_BID, axis=1)
        tricks_seen = np.cumsum(types == TOKEN_TRICK_WIN, axis=1)
        for row in range(group.tokens.shape[0]):
            players = int(group.player_counts[row])
            order = [
                int(group.tokens[row, position, SLOT_REL_PLAYER])
                for position in np.flatnonzero(types[row] == TOKEN_TRICK_WIN)
            ]
            assert len(order) == group.hand_size
            labelled = np.flatnonzero(
                group.next_winner_targets[row] != IGNORE_LABEL
            )
            for position in labelled:
                # Inside the window, and naming the trick that is next *at
                # this token*, on the oracle's absolute-seat axis.
                assert bids_seen[row, position] == players
                assert tricks_seen[row, position] < group.hand_size
                assert (
                    group.next_winner_targets[row, position]
                    == order[tricks_seen[row, position]]
                )
                checked += 1
            if auxiliary_positions == "all":
                # Dense mode must label every position in the window and no
                # others, so the two descriptions agree exactly.
                window = (bids_seen[row] == players) & (
                    tricks_seen[row] < group.hand_size
                )
                assert np.array_equal(np.flatnonzero(window), labelled)
    assert checked > 0


def test_oracle_values_are_projected_over_active_players_only():
    raw = torch.tensor(
        [[4.0, -1.0, 3.0, 99.0, 99.0], [2.0, 5.0, -4.0, 1.0, 99.0]],
        requires_grad=True,
    )
    players = torch.tensor([3, 4])
    centered = center_zero_sum_values(raw, players)

    torch.testing.assert_close(centered[0, :3].sum(), torch.tensor(0.0))
    torch.testing.assert_close(centered[1, :4].sum(), torch.tensor(0.0))
    assert torch.count_nonzero(centered[0, 3:]) == 0
    assert torch.count_nonzero(centered[1, 4:]) == 0

    target = torch.tensor(
        [[2.0, -3.0, 1.0, 0.0, 0.0], [3.0, 1.0, -2.0, -2.0, 0.0]]
    )
    active = torch.arange(5)[None, :] < players[:, None]
    centered_error = ((centered - target).square() * active).sum()
    raw_error = ((raw - target).square() * active).sum()
    assert centered_error <= raw_error
    centered_error.backward()
    assert raw.grad is not None


def test_plackett_luce_card_order_is_listwise_and_prefix_conditioned():
    logits = torch.zeros(1, 1, NUM_CARDS, requires_grad=True)
    with torch.no_grad():
        logits[0, 0, 2] = 3.0
        logits[0, 0, 1] = 2.0
        logits[0, 0, 0] = 1.0
    target = torch.full((1, 1, 4), NUM_CARDS)
    target[0, 0, :3] = torch.tensor([2, 1, 0])
    active = torch.ones(1, 1, dtype=torch.bool)

    full, correct, labeled = plackett_luce_card_order_terms(
        logits, target, torch.zeros(1, 1, dtype=torch.long), active
    )
    expected = (
        torch.logsumexp(torch.tensor([3.0, 2.0, 1.0]), dim=0) - 3.0
        + torch.logsumexp(torch.tensor([2.0, 1.0]), dim=0) - 2.0
    ) / 2.0
    torch.testing.assert_close(full, expected.view(1))
    assert correct.item() and labeled.item()

    suffix, suffix_correct, suffix_labeled = (
        plackett_luce_card_order_terms(
            logits, target, torch.ones(1, 1, dtype=torch.long), active
        )
    )
    torch.testing.assert_close(
        suffix,
        (torch.logsumexp(torch.tensor([2.0, 1.0]), dim=0) - 2.0).view(1),
    )
    assert suffix_correct.item() and suffix_labeled.item()
    full.sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) == 3


def test_ppo_zero_p99_cap_disables_only_tail_acceptance_guard(monkeypatch):
    trainer = SeqTrainer(
        SeqPlumpModel(MODEL),
        ppo_config(policy_kl_cap=0.6, policy_kl_p99_cap=0.0),
        device="cpu",
    )
    trees, _ = trainer.collect()
    monkeypatch.setattr(
        trainer,
        "_evaluate_ppo_kl",
        lambda _groups: (0.5, 2.0, 10.0, 20.0),
    )

    stats = trainer.update(trees)

    assert not stats.rolled_back
    assert stats.backtracks == 0
    assert stats.policy_kl == pytest.approx(0.5)
    assert stats.policy_kl_p99 == pytest.approx(10.0)
    assert stats.policy_kl_max == pytest.approx(20.0)
    assert not stats.proposed_mean_exceeded
    assert not stats.proposed_p99_exceeded


def test_negative_p99_cap_is_rejected():
    with pytest.raises(ValueError, match="policy_kl_p99_cap"):
        ppo_config(policy_kl_p99_cap=-1.0).validate()


@pytest.mark.parametrize("actors", (1, 2))
def test_disabled_trust_region_matches_accepted_steps_without_snapshots(
    monkeypatch, actors
):
    torch.manual_seed(37)
    config = replace(
        ppo_config(actors=actors, auxiliary_grad_fraction=0.25), epochs=2
    )
    guarded = SeqTrainer(SeqPlumpModel(MODEL), config, device="cpu")
    direct = SeqTrainer(
        SeqPlumpModel(MODEL),
        replace(config, ppo_trust_region=False),
        device="cpu",
    )
    for original, restored in zip(guarded.models, direct.models):
        restored.load_state_dict(original.state_dict())
    direct.critic.load_state_dict(guarded.critic.state_dict())
    trees, _ = guarded.collect()
    checked = guarded.update(trees)
    assert checked.policy_kl_checked
    assert not checked.rolled_back
    assert checked.backtracks == 0

    def redundant_work(*args, **kwargs):
        pytest.fail("Disabled trust region must not replay or snapshot the actor")

    monkeypatch.setattr(direct, "_evaluate_ppo_kl", redundant_work)
    monkeypatch.setattr(direct.optimizer, "state_dict", redundant_work)
    for model in direct.models:
        monkeypatch.setattr(model, "state_dict", redundant_work)
    unchecked = direct.update(trees)
    assert not unchecked.policy_kl_checked
    assert not unchecked.rolled_back
    assert unchecked.backtracks == 0
    assert unchecked.step_scale == 1.0
    assert direct.optimizer_steps == guarded.optimizer_steps == config.epochs
    for original, restored in zip(guarded.models, direct.models):
        for expected, actual in zip(original.parameters(), restored.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for expected, actual in zip(guarded.critic.parameters(), direct.critic.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for phase in ("bid", "play"):
        torch.testing.assert_close(
            direct.log_entropy_alpha[phase], guarded.log_entropy_alpha[phase],
            rtol=0, atol=0,
        )
    for expected, actual in zip(
        guarded.optimizer.state.values(), direct.optimizer.state.values()
    ):
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "head_name,target_name,coefficient",
    (
        ("suit_presence_head", "suit_targets", "suit_coef"),
        ("rank_boundary_head", "rank_boundary_targets", "rank_boundary_coef"),
    ),
)
def test_bf16_belief_loss_preserves_confident_positive_gradients(
    head_name, target_name, coefficient
):
    """BCE at logit 8 must still train: a BF16 sigmoid rounds to one."""

    config = replace(
        ppo_config(precision="bf16", auxiliary_positions="all"),
        policy_coef=0.0,
        ppo_entropy_mode="off",
        bid_hit_coef=0.0,
        **{coefficient: 1.0},
    )
    trainer = SeqTrainer(SeqPlumpModel(MODEL), config, device="cpu")
    trees, _ = trainer.collect()
    groups = build_ppo_training_groups(trees, MODEL, config)
    for group in groups:
        targets = getattr(group, target_name)
        targets[targets != IGNORE_LABEL] = 1
    head = getattr(trainer.model, head_name)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.fill_(8.0)
    trainer.model.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer._ppo_actor_backward(groups)
    assert head.bias.grad is not None
    assert head.bias.grad.min().item() < -1e-6
    assert head.bias.grad.max().item() <= 0.0


def test_ppo_ratio_is_one_before_update_and_gradient_follows_advantage():
    logits = torch.tensor(
        [[0.2, -0.4, 9.0], [0.2, -0.4, 9.0]], requires_grad=True
    )
    legal_logits = logits[:, :2].detach()
    old_legal = torch.softmax(legal_logits, dim=-1)
    old = torch.cat((old_legal, torch.zeros(2, 1)), dim=-1)
    actions = torch.tensor([0, 1])
    advantages = torch.tensor([1.0, -1.0])
    terms = ppo_clipped_terms(
        logits, old, actions, advantages, clip_ratio=0.1
    )
    torch.testing.assert_close(terms.ratios, torch.ones(2))
    torch.testing.assert_close(terms.divergences, torch.zeros(2), atol=1e-7, rtol=0)
    terms.losses.sum().backward()
    # Gradient descent raises the positively-advantaged selected logit and
    # lowers the negatively-advantaged selected logit.
    assert logits.grad[0, 0] < 0
    assert logits.grad[1, 1] > 0
    torch.testing.assert_close(logits.grad[:, 2], torch.zeros(2))


def test_kl_only_readout_matches_full_terms_with_masked_and_forced_actions():
    logits = torch.tensor([[0.2, -0.4, 9.0], [-1.0, 9.0, 2.0]])
    old = torch.tensor([[0.75, 0.25, 0.0], [0.0, 1.0, 0.0]])
    terms = ppo_clipped_terms(
        logits, old, torch.tensor([0, 1]), torch.ones(2), clip_ratio=0.1
    )
    actual = ppo_policy_kl(logits, old)
    torch.testing.assert_close(actual, terms.divergences, rtol=0, atol=0)
    assert actual[1].item() == 0.0


def test_ppo_collection_is_unbranched_and_learns_every_self_play_seat():
    trainer = SeqTrainer(
        SeqPlumpModel(MODEL), ppo_config(actors=2), device="cpu"
    )
    trees, summary = trainer.collect()
    assert all(tree.leaf_total == 1 and len(tree.leaves) == 1 for tree in trees)
    assert trainer.collector.stats.branch_decisions == 0
    assert all(
        {record.seat for record in tree.leaves[0].decisions}
        == set(range(tree.num_players))
        for tree in trees
    )
    assert {record.policy_id for tree in trees for record in tree.leaves[0].decisions} == {
        "current",
        "current:1",
    }


def test_ppo_divides_by_learned_seats_but_not_decision_count():
    cells = (
        GameScheduleCell(hand_size=3, num_players=3),
        GameScheduleCell(hand_size=5, num_players=3),
    )
    trainer = SeqTrainer(
        SeqPlumpModel(MODEL), ppo_config(cells=cells), device="cpu"
    )
    trees, _ = trainer.collect()
    groups = build_ppo_training_groups(trees, MODEL, trainer.train)
    weight_by_hand = {}
    row_weight = {}
    for group in groups:
        weights = [
            weight for rows in group.policy.values() for weight in rows.weight
        ]
        weight_by_hand[group.hand_size] = sum(weights)
        row_weight[group.hand_size] = set(weights)
    # Two equally weighted games and three learned seats: every decision gets
    # 1 / (2 * 3). The forced final card is omitted, leaving N decisions/seat,
    # so total policy weight grows linearly with hand length.
    assert tuple(row_weight[3]) == pytest.approx((1.0 / 6.0,))
    assert tuple(row_weight[5]) == pytest.approx((1.0 / 6.0,))
    assert weight_by_hand[3] == pytest.approx(1.5)
    assert weight_by_hand[5] == pytest.approx(2.5)


def test_ppo_checkpoint_round_trips_actor_pool_critic_and_entropy(tmp_path):
    # An untrained policy at three cards usually leaves every seat short of its
    # bid, and equal scores make the relative rewards identically zero. The
    # oracle is initialized by broadcasting the actor's scalar value head to
    # every seat, so its centered prediction starts at exactly zero too: with
    # zero targets the value gradient is legitimately zero and the critic
    # cannot move. Pin a seed whose sampled games are not tied, and assert that
    # precondition directly so a future initialization change that shifts the
    # RNG stream fails here instead of looking like a broken critic.
    torch.manual_seed(4)
    config = ppo_config(actors=2)
    trainer = SeqTrainer(SeqPlumpModel(MODEL), config, device="cpu")
    assert isinstance(trainer.critic, SeqPPOOracleCritic)
    critic_before = trainer.critic.player_value_head[2].weight.detach().clone()
    trees, _ = trainer.collect()
    stats = trainer.update(trees)
    assert trainer.critic is not None
    assert stats.loss_value > 0, "sampled games are tied; the critic has no signal"
    assert not torch.equal(
        critic_before, trainer.critic.player_value_head[2].weight
    )
    assert stats.critic_all_player_rmse > 0
    assert np.isfinite(stats.critic_all_player_correlation)
    assert stats.critic_loss_first_epoch > 0
    assert stats.critic_loss_last_epoch == pytest.approx(
        stats.critic_loss_first_epoch
    )
    assert stats.critic_loss_reduction == pytest.approx(0.0)
    trainer.iteration = 7
    path = tmp_path / "ppo.pt"
    trainer.save_checkpoint(path)

    restored = SeqTrainer(SeqPlumpModel(MODEL), config, device="cpu")
    restored.load_checkpoint(path)
    assert restored.iteration == 7
    assert len(restored.models) == 2
    assert restored.critic is not None
    assert restored._entropy_alpha("bid") == pytest.approx(
        trainer._entropy_alpha("bid")
    )
    for expected, actual in zip(trainer.models, restored.models):
        for left, right in zip(expected.parameters(), actual.parameters()):
            assert torch.equal(left, right)


def test_pre_card_order_oracle_checkpoint_preserves_existing_adam(tmp_path):
    config = ppo_config(oracle_card_order_coef=0.05)
    source = SeqTrainer(SeqPlumpModel(MODEL), config, device="cpu")
    trees, _ = source.collect()
    source.update(trees)
    current = tmp_path / "current.pt"
    source.save_checkpoint(current)
    payload = torch.load(current, map_location="cpu", weights_only=False)

    for key in ORACLE_CARD_ORDER_HEAD_KEYS:
        del payload["critic_state_dict"][key]
    payload["model_format_version"] = 3
    optimizer = payload["critic_optimizer_state_dict"]
    new_parameter_ids = optimizer["param_groups"][0]["params"][-2:]
    optimizer["param_groups"][0]["params"] = optimizer["param_groups"][0][
        "params"
    ][:-2]
    for parameter_id in new_parameter_ids:
        optimizer["state"].pop(parameter_id, None)
    old = tmp_path / "format3.pt"
    torch.save(payload, old)

    restored = SeqTrainer(SeqPlumpModel(MODEL), config, device="cpu")
    assert isinstance(restored.critic, SeqPPOOracleCritic)
    random_order_head = {
        key: restored.critic.state_dict()[key].clone()
        for key in ORACLE_CARD_ORDER_HEAD_KEYS
    }
    restored.load_checkpoint(old)

    for key, expected in payload["critic_state_dict"].items():
        torch.testing.assert_close(restored.critic.state_dict()[key], expected)
    for key, expected in random_order_head.items():
        torch.testing.assert_close(restored.critic.state_dict()[key], expected)
    assert len(restored.critic_optimizer.state) == len(optimizer["state"])
    assert restored.critic.card_order_head.weight not in restored.critic_optimizer.state
    assert restored.critic.card_order_head.bias not in restored.critic_optimizer.state


def test_coupled_equal_rate_ppo_uses_one_actor_optimizer_group():
    coupled = SeqTrainer(SeqPlumpModel(MODEL), ppo_config(), device="cpu")
    detached = SeqTrainer(
        SeqPlumpModel(MODEL),
        ppo_config(detach_actor_auxiliary=True),
        device="cpu",
    )

    assert len(coupled.optimizer.param_groups) == 1
    assert coupled.optimizer.param_groups[0]["kl_sensitive"] is True
    assert len(detached.optimizer.param_groups) == 2


def test_split_actor_adam_state_migrates_to_coupled_group(tmp_path):
    split_config = ppo_config(
        suit_coef=0.1,
        rank_boundary_coef=0.05,
        trick_coef=0.1,
        detach_actor_auxiliary=True,
    )
    split = SeqTrainer(SeqPlumpModel(MODEL), split_config, device="cpu")
    trees, _ = split.collect()
    split.update(trees)
    path = tmp_path / "split.pt"
    split.save_checkpoint(path)
    state_count = len(split.optimizer.state)

    coupled_config = replace(
        split_config,
        ppo_actor_auxiliary_detach_hidden=False,
    )
    coupled = SeqTrainer(
        SeqPlumpModel(MODEL), coupled_config, device="cpu"
    )
    coupled.load_checkpoint(path, allow_training_config_mismatch=True)

    assert len(coupled.optimizer.param_groups) == 1
    assert len(coupled.optimizer.state) == state_count


def test_entropy_target_reconfiguration_warm_starts_the_controller(tmp_path):
    """Retargeting keeps the multiplier; only the level it drives to changes."""

    source_config = ppo_config()
    source = SeqTrainer(SeqPlumpModel(MODEL), source_config, device="cpu")
    source.log_entropy_alpha["bid"].data.fill_(math.log(0.3))
    source.log_entropy_alpha["play"].data.fill_(math.log(0.2))
    path = tmp_path / "source.pt"
    source.save_checkpoint(path)

    changed_config = replace(
        source_config,
        ppo_bid_entropy_target=0.5,
        ppo_play_entropy_target=0.4,
    )
    restored = SeqTrainer(
        SeqPlumpModel(MODEL), changed_config, device="cpu"
    )
    restored.load_checkpoint(path, allow_training_config_mismatch=True)

    assert restored._entropy_alpha("bid") == pytest.approx(0.3)
    assert restored._entropy_alpha("play") == pytest.approx(0.2)
    assert restored.train.ppo_entropy_targets(0) == pytest.approx((0.5, 0.4))


def test_entropy_controller_resets_when_its_mechanics_change(tmp_path):
    """A different controller learning rate is a different controller."""

    source_config = ppo_config()
    source = SeqTrainer(SeqPlumpModel(MODEL), source_config, device="cpu")
    source.log_entropy_alpha["bid"].data.fill_(math.log(0.3))
    source.log_entropy_alpha["play"].data.fill_(math.log(0.2))
    path = tmp_path / "source.pt"
    source.save_checkpoint(path)

    changed_config = replace(
        source_config,
        ppo_entropy_learning_rate=source_config.ppo_entropy_learning_rate * 2,
    )
    restored = SeqTrainer(
        SeqPlumpModel(MODEL), changed_config, device="cpu"
    )
    restored.load_checkpoint(path, allow_training_config_mismatch=True)

    assert restored._entropy_alpha("bid") == pytest.approx(
        changed_config.ppo_entropy_coef
    )
    assert restored._entropy_alpha("play") == pytest.approx(
        changed_config.ppo_entropy_coef
    )
    assert restored.entropy_optimizer is not None
    assert restored.entropy_optimizer.state == {}


def test_cosine_entropy_schedule_is_continuous_and_preserves_controller(tmp_path):
    source_config = ppo_config()
    source = SeqTrainer(SeqPlumpModel(MODEL), source_config, device="cpu")
    source.iteration = 10
    source.log_entropy_alpha["bid"].data.fill_(math.log(0.3))
    source.log_entropy_alpha["play"].data.fill_(math.log(0.2))
    path = tmp_path / "source.pt"
    source.save_checkpoint(path)

    scheduled_config = replace(
        source_config,
        ppo_entropy_target_schedule="cosine",
        ppo_entropy_schedule_start_iteration=10,
        ppo_entropy_schedule_end_iteration=20,
        ppo_bid_entropy_target_end=0.3,
        ppo_play_entropy_target_end=0.2,
    )
    assert scheduled_config.ppo_entropy_targets(10) == pytest.approx(
        (source_config.ppo_bid_entropy_target, source_config.ppo_play_entropy_target)
    )
    midpoint = scheduled_config.ppo_entropy_targets(15)
    assert midpoint == pytest.approx(
        (
            (source_config.ppo_bid_entropy_target + 0.3) / 2,
            (source_config.ppo_play_entropy_target + 0.2) / 2,
        )
    )
    assert scheduled_config.ppo_entropy_targets(20) == pytest.approx((0.3, 0.2))

    restored = SeqTrainer(
        SeqPlumpModel(MODEL), scheduled_config, device="cpu"
    )
    restored.load_checkpoint(path, allow_training_config_mismatch=True)
    assert restored._entropy_alpha("bid") == pytest.approx(0.3)
    assert restored._entropy_alpha("play") == pytest.approx(0.2)


def test_linear_entropy_schedule_moves_at_a_constant_rate():
    config = replace(
        ppo_config(),
        ppo_bid_entropy_target=0.45,
        ppo_play_entropy_target=0.35,
        ppo_entropy_target_schedule="linear",
        ppo_entropy_schedule_start_iteration=46_900,
        ppo_entropy_schedule_end_iteration=58_000,
        ppo_bid_entropy_target_end=0.25,
        ppo_play_entropy_target_end=0.20,
    )

    assert config.ppo_entropy_targets(46_899) == pytest.approx((0.45, 0.35))
    assert config.ppo_entropy_targets(46_900) == pytest.approx((0.45, 0.35))
    assert config.ppo_entropy_targets(52_450) == pytest.approx((0.35, 0.275))
    assert config.ppo_entropy_targets(58_000) == pytest.approx((0.25, 0.20))
    assert config.ppo_entropy_targets(60_000) == pytest.approx((0.25, 0.20))


def test_oracle_critic_has_one_sequence_per_game_and_exact_seat_ties():
    trainer = SeqTrainer(
        SeqPlumpModel(MODEL),
        ppo_config(actors=2, oracle_card_order_coef=0.05),
        device="cpu",
    )
    trees, _ = trainer.collect()
    batch = build_ppo_training_batch(trees, MODEL, trainer.train)

    assert sum(group.tokens.shape[0] for group in batch.critic_groups) == len(trees)
    for group in batch.critic_groups:
        prefix = group.tokens[
            :, 1 : 1 + group.num_players * group.hand_size
        ]
        assert np.all(prefix[..., SLOT_TYPE] == TOKEN_HAND)
        for game_cards in prefix:
            owners = game_cards[:, SLOT_REL_PLAYER]
            cards = game_cards[:, SLOT_CARD]
            assert np.bincount(owners, minlength=group.num_players).tolist() == [
                group.hand_size
            ] * group.num_players
            assert len(set(cards.tolist())) == group.num_players * group.hand_size
        for seq_index in range(group.tokens.shape[0]):
            for player in range(group.num_players):
                order = group.card_play_order[seq_index, player]
                real = order[order < NUM_CARDS]
                assert len(real) == group.hand_size
                assert len(set(real.tolist())) == group.hand_size
            for position in np.flatnonzero(
                group.tokens[seq_index, :, SLOT_NEXT_PHASE] > 0
            ):
                prefix = group.tokens[seq_index, : position + 1]
                expected = np.zeros(MODEL.max_players, dtype=np.int64)
                plays = prefix[prefix[:, SLOT_TYPE] == TOKEN_PLAY]
                for player in plays[:, SLOT_REL_PLAYER]:
                    expected[player] += 1
                assert np.array_equal(
                    group.cards_played[seq_index, position], expected
                )

    for policy_group in batch.policy_groups:
        for rows in policy_group.policy.values():
            for index in range(len(rows.weight)):
                critic_group = batch.critic_groups[rows.critic_group[index]]
                token = critic_group.tokens[
                    rows.critic_seq_index[index], rows.critic_position[index]
                ]
                assert token[SLOT_NEXT_ACTOR] == rows.critic_seat[index]
                expected_phase = (
                    1 if rows is policy_group.policy["bid"] else 2
                )
                assert token[SLOT_NEXT_PHASE] == expected_phase


def test_oracle_dense_value_projection_reads_every_genuine_state():
    config = ppo_config(value_positions="all", auxiliary_positions="all")
    trainer = SeqTrainer(SeqPlumpModel(MODEL), config, device="cpu")
    trees, _ = trainer.collect()
    group = build_ppo_training_batch(trees, MODEL, config).critic_groups[0]
    tokens = torch.from_numpy(group.tokens)
    value_seq_np, value_pos_np = np.nonzero(group.position_weight > 0)
    value_seq = torch.from_numpy(value_seq_np)
    value_pos = torch.from_numpy(value_pos_np)
    decision_seq = torch.tensor(group.rows.seq_index)
    decision_pos = torch.tensor(group.rows.position)

    assert isinstance(trainer.critic, SeqPPOOracleCritic)
    trainer.critic.eval()
    with torch.inference_mode():
        dense_selected, _, _, _ = trainer.critic.forward_selected(
            tokens,
            decision_seq,
            decision_pos,
            value_seq=value_seq,
            value_pos=value_pos,
            dense_value_projection=True,
        )
        expected = trainer.critic.forward_full(tokens)[value_seq, value_pos]

    assert dense_selected.shape == (
        int(np.count_nonzero(group.position_weight)),
        MODEL.max_players,
    )
    torch.testing.assert_close(dense_selected, expected)


def test_oracle_critic_reports_loss_dynamics_across_epochs():
    torch.manual_seed(3)
    trainer = SeqTrainer(
        SeqPlumpModel(MODEL),
        ppo_config(critic_epochs=2),
        device="cpu",
    )
    trees, _ = trainer.collect()
    stats = trainer.update(trees)

    assert stats.critic_loss_first_epoch > 0
    assert stats.critic_loss_last_epoch > 0
    assert stats.critic_loss_reduction == pytest.approx(
        (stats.critic_loss_first_epoch - stats.critic_loss_last_epoch)
        / stats.critic_loss_first_epoch
    )


def test_ppo_actor_beliefs_and_oracle_trick_head_train_with_ten_card_stages():
    config = ppo_config(
        cells=(GameScheduleCell(hand_size=10, num_players=3, games=1),),
        suit_coef=0.1,
        rank_boundary_coef=0.05,
        trick_coef=0.1,
        next_winner_coef=0.2,
        oracle_card_order_coef=0.05,
        detach_actor_auxiliary=True,
        auxiliary_positions="all",
        value_positions="all",
        bucket_width=16,
    )
    trainer = SeqTrainer(SeqPlumpModel(MODEL), config, device="cpu")
    trees, _ = trainer.collect()
    batch = build_ppo_training_batch(trees, MODEL, config)

    assert sum(group.position_weight.sum() for group in batch.policy_groups) == (
        pytest.approx(1.0)
    )
    assert sum(group.position_weight.sum() for group in batch.critic_groups) == (
        pytest.approx(1.0)
    )
    policy_decisions = sum(
        len(rows.weight)
        for group in batch.policy_groups
        for rows in group.policy.values()
    )
    dense_actor_positions = sum(
        np.count_nonzero(group.position_weight) for group in batch.policy_groups
    )
    assert dense_actor_positions > policy_decisions
    for group in batch.policy_groups:
        real = group.position_weight > 0
        assert np.all(
            group.suit_targets[real, : group.num_players - 1] != IGNORE_LABEL
        )
        assert np.all(
            group.trick_masks[real, : group.num_players].any(axis=-1)
        )
    dense_oracle_positions = 0
    for group in batch.critic_groups:
        real = group.position_weight > 0
        real_seq, _ = np.nonzero(real)
        dense_oracle_positions += int(real.sum())
        assert np.array_equal(
            group.player_counts,
            np.full(group.tokens.shape[0], 3, dtype=np.int8),
        )
        assert np.allclose(group.returns[:, :3].sum(axis=1), 0.0)
        assert np.all(group.returns[:, 3:] == 0.0)
        active = group.trick_targets != IGNORE_LABEL
        assert np.all(
            group.trick_masks[real].any(axis=-1) == active[real_seq]
        )
    assert all(
        np.all(group.belief_stage_positions >= 0)
        for group in batch.policy_groups
    )
    assert isinstance(trainer.critic, SeqPPOOracleCritic)
    assert all(
        parameter.requires_grad
        for parameter in trainer.critic.backbone.trick_count_head.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in trainer.critic.backbone.next_winner_head.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in trainer.critic.backbone.suit_presence_head.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in trainer.critic.backbone.rank_boundary_head.parameters()
    )

    stats = trainer.update(trees)

    assert stats.loss_suit > 0
    assert stats.loss_rank_boundary > 0
    assert stats.loss_trick > 0
    assert stats.loss_oracle_trick > 0
    assert stats.loss_next_winner > 0
    assert stats.loss_oracle_next_winner > 0
    assert stats.loss_oracle_card_order > 0
    assert stats.value_rows == dense_oracle_positions
    assert stats.value_rows > stats.policy_rows
    for stage in (0, 4, 8):
        assert 0.0 <= getattr(stats, f"suit_accuracy_10c_{stage}") <= 1.0
        assert getattr(stats, f"suit_loss_10c_{stage}") > 0.0
        assert 0.0 <= getattr(
            stats, f"rank_boundary_balanced_accuracy_10c_{stage}"
        ) <= 1.0
        assert getattr(stats, f"rank_boundary_loss_10c_{stage}") > 0.0
        assert 0.0 <= getattr(stats, f"trick_accuracy_10c_{stage}") <= 1.0
    assert 0.0 <= stats.oracle_trick_accuracy <= 1.0
    assert 0.0 <= stats.next_winner_accuracy <= 1.0
    assert 0.0 <= stats.oracle_next_winner_accuracy <= 1.0
    assert 0.0 <= stats.oracle_card_order_top1_accuracy <= 1.0
    assert trainer.model.suit_presence_head.weight.grad is not None
    assert trainer.model.rank_boundary_head.weight.grad is not None
    assert trainer.model.trick_count_head.weight.grad is not None
    assert trainer.model.next_winner_head.weight.grad is not None
    assert trainer.critic.backbone.trick_count_head.weight.grad is not None
    assert trainer.critic.backbone.next_winner_head.weight.grad is not None
    # Padded seat columns are masked out of every softmax, so they can never
    # collect gradient however wide the head is.
    assert torch.equal(
        trainer.model.next_winner_head.weight.grad[3:],
        torch.zeros_like(trainer.model.next_winner_head.weight.grad[3:]),
    )
    assert trainer.critic.card_order_head.weight.grad is not None
    assert trainer.critic.backbone.final_norm.weight.grad is not None


@pytest.mark.parametrize(
    ("detach_actor_auxiliary", "expect_trunk_gradient"),
    ((False, True), (True, False)),
)
def test_ppo_actor_auxiliary_hidden_detachment_isolates_policy_trunk(
    detach_actor_auxiliary,
    expect_trunk_gradient,
):
    config = replace(
        ppo_config(
            cells=(GameScheduleCell(hand_size=10, num_players=3, games=1),),
            suit_coef=0.1,
            trick_coef=0.1,
            detach_actor_auxiliary=detach_actor_auxiliary,
        ),
        policy_coef=0.0,
        ppo_entropy_mode="off",
    )
    trainer = SeqTrainer(SeqPlumpModel(MODEL), config, device="cpu")
    trees, _ = trainer.collect()
    batch = build_ppo_training_batch(trees, MODEL, config)
    trainer.optimizer.zero_grad(set_to_none=True)

    trainer._ppo_actor_backward(batch.policy_groups)

    for head in (
        trainer.model.suit_presence_head,
        trainer.model.trick_count_head,
    ):
        assert head.weight.grad is not None
        assert torch.count_nonzero(head.weight.grad) > 0
    trunk_grad = trainer.model.final_norm.weight.grad
    has_trunk_gradient = (
        trunk_grad is not None and bool(torch.count_nonzero(trunk_grad))
    )
    assert has_trunk_gradient is expect_trunk_gradient


def test_ppo_length_bucketing_merges_shapes_and_preserves_causal_readouts():
    cells = (
        GameScheduleCell(hand_size=4, num_players=3, games=2),
        GameScheduleCell(hand_size=3, num_players=4, games=2),
    )
    exact_config = ppo_config(cells=cells, bucket_width=0)
    bucket_config = ppo_config(cells=cells, bucket_width=16)
    trainer = SeqTrainer(SeqPlumpModel(MODEL), exact_config, device="cpu")
    trees, _ = trainer.collect()
    exact = build_ppo_training_batch(trees, MODEL, exact_config)
    bucketed = build_ppo_training_batch(trees, MODEL, bucket_config)

    assert len(exact.policy_groups) == 2
    assert len(bucketed.policy_groups) == 1
    assert len(exact.critic_groups) == 2
    assert len(bucketed.critic_groups) == 1

    model = trainer.model.eval()
    offset = 0
    with torch.inference_mode():
        bucket_output = model.forward_full(
            torch.from_numpy(bucketed.policy_groups[0].tokens), aux_heads=False
        )
        for group in exact.policy_groups:
            count, length = group.tokens.shape[:2]
            exact_output = model.forward_full(
                torch.from_numpy(group.tokens), aux_heads=False
            )
            torch.testing.assert_close(
                bucket_output.bid_logits[offset : offset + count, :length],
                exact_output.bid_logits,
                atol=2e-6,
                rtol=2e-6,
            )
            torch.testing.assert_close(
                bucket_output.card_logits[offset : offset + count, :length],
                exact_output.card_logits,
                atol=2e-6,
                rtol=2e-6,
            )
            offset += count


def test_ppo_actor_and_critic_bucket_widths_are_independent():
    cells = (
        GameScheduleCell(hand_size=4, num_players=3, games=2),
        GameScheduleCell(hand_size=3, num_players=4, games=2),
    )
    exact_config = ppo_config(cells=cells, bucket_width=0)
    trainer = SeqTrainer(SeqPlumpModel(MODEL), exact_config, device="cpu")
    trees, _ = trainer.collect()

    critic_only = build_ppo_training_batch(
        trees,
        MODEL,
        ppo_config(
            cells=cells,
            bucket_width=0,
            critic_bucket_width=16,
        ),
    )
    actor_only = build_ppo_training_batch(
        trees,
        MODEL,
        ppo_config(
            cells=cells,
            bucket_width=16,
            critic_bucket_width=0,
        ),
    )

    assert len(critic_only.policy_groups) == 2
    assert len(critic_only.critic_groups) == 1
    assert len(actor_only.policy_groups) == 1
    assert len(actor_only.critic_groups) == 2


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires Apple MPS"
)
def test_ppo_adam_kernel_follows_resume_device_without_resetting_state(tmp_path):
    config = ppo_config()
    source = SeqTrainer(SeqPlumpModel(MODEL), config, device="cpu")
    trees, _ = source.collect()
    source.update(trees)
    cpu_checkpoint = tmp_path / "cpu.pt"
    source.save_checkpoint(cpu_checkpoint)

    accelerated = SeqTrainer(SeqPlumpModel(MODEL), config, device="mps")
    accelerated.load_checkpoint(cpu_checkpoint)
    for original, restored in (
        (source.optimizer, accelerated.optimizer),
        (source.critic_optimizer, accelerated.critic_optimizer),
    ):
        assert all(group["fused"] for group in restored.param_groups)
        for expected, actual in zip(original.state.values(), restored.state.values()):
            assert actual["step"].device.type == "mps"
            for name, value in expected.items():
                torch.testing.assert_close(value, actual[name].cpu(), rtol=0, atol=0)
    trees, _ = accelerated.collect()
    accelerated.update(trees)
    mps_checkpoint = tmp_path / "mps.pt"
    accelerated.save_checkpoint(mps_checkpoint)

    portable = SeqTrainer(SeqPlumpModel(MODEL), config, device="cpu")
    portable.load_checkpoint(mps_checkpoint)
    for original, restored in (
        (accelerated.optimizer, portable.optimizer),
        (accelerated.critic_optimizer, portable.critic_optimizer),
    ):
        assert all(not group["fused"] for group in restored.param_groups)
        for expected, actual in zip(original.state.values(), restored.state.values()):
            assert actual["step"].device.type == "cpu"
            for name, value in expected.items():
                torch.testing.assert_close(value.cpu(), actual[name], rtol=0, atol=0)
    trees, _ = portable.collect()
    portable.update(trees)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires Apple MPS"
)
def test_mps_bf16_ppo_smoke():
    # Known flaky, and not from anything in this file. The oracle critic's
    # `hidden[seq, pos]` backward intermittently sees an empty index on MPS
    # ("value tensor of shape [N, d] cannot be broadcast to indexing result of
    # shape [0, d]") and the trigger is the accelerator allocation pattern, not
    # the objective: adding a single unused `torch.zeros(())` per actor
    # microbatch and changing nothing else takes this test from 0/20 to 5/20
    # failures. It reproduces only at this tiny model size -- 30 consecutive
    # iterations of the real 768-game schedule are clean, as were v4's 47,600.
    # Fixing it belongs with the oracle card-order readout, not here.
    torch.manual_seed(11)
    np.random.seed(11)
    config = ppo_config(
        precision="bf16",
        kv_dtype="bf16",
        suit_coef=0.1,
        trick_coef=0.1,
        oracle_card_order_coef=0.05,
        auxiliary_positions="all",
        value_positions="all",
    )
    trainer = SeqTrainer(SeqPlumpModel(MODEL), config, device="mps")
    trees, _ = trainer.collect()
    stats = trainer.update(trees)
    torch.mps.synchronize()
    assert np.isfinite(stats.loss_policy)
    assert np.isfinite(stats.loss_value)
    assert stats.loss_suit > 0
    # A tiny sampled round can make a feasibility-masked auxiliary trivial;
    # the CPU objective test above checks positive gradients on a ten-card
    # case. This smoke test checks that dense bf16 execution remains finite.
    assert stats.loss_trick >= 0
    assert stats.loss_oracle_trick >= 0
    assert stats.loss_oracle_card_order >= 0
    assert stats.ppo_behavior_replay_kl < 1e-3
    # Autocast lowers compute, not the master weights or Adam state.
    assert next(trainer.model.parameters()).dtype == torch.float32
    assert trainer.collector._kv_dtype == torch.bfloat16


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires Apple MPS"
)
def test_mps_fp16_ppo_smoke():
    config = ppo_config(precision="fp16", kv_dtype="fp16")
    trainer = SeqTrainer(SeqPlumpModel(MODEL), config, device="mps")
    trees, _ = trainer.collect()
    stats = trainer.update(trees)
    torch.mps.synchronize()
    assert np.isfinite(stats.loss_policy)
    assert np.isfinite(stats.loss_value)
    assert stats.ppo_behavior_replay_kl < 1e-3
    assert next(trainer.model.parameters()).dtype == torch.float32
    assert trainer.collector._kv_dtype == torch.float16


def _actor_value_config(**overrides):
    return replace(
        ppo_config(value_positions="all", auxiliary_positions="all"),
        ppo_actor_value_coef=0.2,
        **overrides,
    )


def test_actor_value_probe_measures_exactly_the_oracle_objective():
    """The two value losses are the same measurement on the same games.

    Both heads predict the per-seat terminal rewards of every game, densely
    over genuine causal tokens, projected onto the zero-sum subspace and
    normalized to unit mass per game. The actor works on an observer-relative
    seat axis and one row per learned seat; the oracle on an absolute-seat axis
    and one row per game. Neither difference may change the number, so if both
    heads are pinned to the zero prediction the two losses must agree exactly.
    A fresh oracle already predicts zero -- it is initialized by broadcasting
    the actor's scalar value head to every seat, which centers to zero -- so
    only the actor's randomly initialized head needs pinning.
    """

    torch.manual_seed(4)
    config = _actor_value_config()
    trainer = SeqTrainer(SeqPlumpModel(MODEL), config, device="cpu")
    with torch.no_grad():
        trainer.model.player_value_head[2].weight.zero_()
        trainer.model.player_value_head[2].bias.zero_()
    trees, _ = trainer.collect()
    stats = trainer.update(trees)

    assert stats.actor_value_loss > 0, "tied games leave nothing to measure"
    assert stats.actor_value_loss == pytest.approx(
        stats.oracle_value_loss, rel=1e-5
    )
    # Both heads predict zero, so the reported RMSE is the target spread.
    assert stats.actor_all_player_rmse == pytest.approx(
        stats.actor_value_zero_rmse, rel=1e-5
    )
    assert stats.actor_all_player_rmse == pytest.approx(
        stats.critic_all_player_rmse, rel=1e-5
    )


def test_actor_value_probe_trains_its_head_and_leaves_the_critic_authoritative():
    torch.manual_seed(4)
    trainer = SeqTrainer(
        SeqPlumpModel(MODEL), _actor_value_config(), device="cpu"
    )
    before = trainer.model.player_value_head[2].weight.detach().clone()
    trees, _ = trainer.collect()
    stats = trainer.update(trees)

    assert not torch.equal(before, trainer.model.player_value_head[2].weight)
    assert stats.actor_value_loss > 0
    # The probe is a measurement, not a control signal: advantages still come
    # from the oracle critic, whose own head is trained independently.
    assert trainer.critic is not None
    assert stats.critic_all_player_rmse > 0


def test_actor_value_probe_is_off_by_default():
    torch.manual_seed(4)
    trainer = SeqTrainer(
        SeqPlumpModel(MODEL),
        ppo_config(value_positions="all", auxiliary_positions="all"),
        device="cpu",
    )
    assert trainer.train.ppo_actor_value_coef == 0.0
    before = trainer.model.player_value_head[2].weight.detach().clone()
    trees, _ = trainer.collect()
    stats = trainer.update(trees)

    assert torch.equal(before, trainer.model.player_value_head[2].weight)
    assert stats.actor_value_loss == 0.0


def test_actor_value_targets_are_observer_relative_and_zero_sum():
    torch.manual_seed(4)
    config = _actor_value_config()
    trainer = SeqTrainer(SeqPlumpModel(MODEL), config, device="cpu")
    trees, _ = trainer.collect()
    batch = build_ppo_training_batch(trees, trainer.model.config, config)

    for group in batch.policy_groups:
        counts = group.player_counts.astype(np.int64)
        active = (
            np.arange(trainer.model.config.max_players)[None, :]
            < counts[:, None]
        )
        # Relative seat 0 is the observer, so it must equal that row's own
        # return, and every table's active seats must still sum to zero.
        own = np.asarray(
            group.policy["bid"].returns + group.policy["play"].returns
        )
        rows = np.asarray(
            group.policy["bid"].seq_index + group.policy["play"].seq_index
        )
        np.testing.assert_allclose(
            group.value_targets[rows, 0], own, atol=1e-6
        )
        np.testing.assert_allclose(
            (group.value_targets * active).sum(axis=1), 0.0, atol=1e-5
        )
        assert not group.value_targets[~active].any()


def _budget_trainer(fraction, *, seed=0):
    config = replace(
        ppo_config(
            suit_coef=0.25,
            trick_coef=0.25,
            auxiliary_grad_fraction=fraction,
        ),
        seed=seed,
        learning_rate=1e-3,
        lr_warmup_updates=0,
    )
    torch.manual_seed(seed)
    return SeqTrainer(SeqPlumpModel(MODEL), config, device="cpu")


def test_auxiliary_grad_budget_sets_the_ratio_between_the_two_objectives():
    """The auxiliary half is rescaled to its configured share of the PPO half."""

    fraction = 0.25
    trainer = _budget_trainer(fraction)
    trees, _ = trainer.collect()
    stats = trainer.update(trees)

    assert stats.policy_loss_grad_norm > 0.0
    assert stats.auxiliary_loss_grad_norm > 0.0
    # The two are measured over the same parameters and differ only in which
    # half of the loss produced them, so whichever is larger the budget lands
    # them on the configured ratio.
    assert stats.auxiliary_grad_scale == pytest.approx(
        fraction * stats.policy_loss_grad_norm / stats.auxiliary_loss_grad_norm
    )
    # Post-scale the auxiliary contribution is exactly its share.
    scaled = stats.auxiliary_grad_scale * stats.auxiliary_loss_grad_norm
    assert scaled == pytest.approx(fraction * stats.policy_loss_grad_norm)


def test_auxiliary_gradient_budget_respects_frozen_actor_parameters():
    model = SeqPlumpModel(MODEL)
    model.slot_embedding.weight.requires_grad_(False)
    before = model.slot_embedding.weight.detach().clone()
    trainer = SeqTrainer(
        model, ppo_config(auxiliary_grad_fraction=0.25, suit_coef=0.1), device="cpu"
    )
    trees, _ = trainer.collect()
    stats = trainer.update(trees)
    assert stats.policy_loss_grad_norm > 0
    torch.testing.assert_close(model.slot_embedding.weight, before, rtol=0, atol=0)
    assert model.slot_embedding.weight.grad is None


def test_split_backward_partitions_the_loss_without_losing_a_term():
    """Both halves together must equal the one summed backward, exactly.

    This is the property that makes the budget safe: the split changes only
    how the two objectives are weighted against each other, never which
    terms reach the parameters. A term accidentally left out of both halves,
    or counted in the wrong one, shows up here.
    """

    joint = _budget_trainer(0.0)
    trees, _ = joint.collect()
    joint_gradients = _actor_gradients_after_backward(joint, trees)
    assert joint._auxiliary_grads is None

    split = _budget_trainer(0.0)
    split_trees, _ = split.collect()
    # Choose the fraction that makes the rebalance scale exactly 1.0, so the
    # halves are recombined unweighted and any difference is a partition bug
    # rather than the intended reweighting.
    measured = _budget_trainer(0.0)
    measured_trees, _ = measured.collect()
    measured.train = replace(
        measured.train, ppo_auxiliary_grad_fraction=1.0
    )
    probe = SeqUpdateStats()
    batch = build_ppo_training_batch(
        measured_trees, measured.model.config, measured.train
    )
    measured._prepare_ppo_advantages(batch.policy_groups, batch.critic_groups, probe)
    measured.model.train()
    measured.optimizer.zero_grad(set_to_none=True)
    measured._ppo_actor_backward(batch.policy_groups)
    policy_norm, auxiliary_norm, _ = measured._apply_auxiliary_grad_budget()
    assert policy_norm > 0.0 and auxiliary_norm > 0.0

    split.train = replace(
        split.train, ppo_auxiliary_grad_fraction=auxiliary_norm / policy_norm
    )
    split_gradients = _actor_gradients_after_backward(split, split_trees)

    assert joint_gradients.keys() == split_gradients.keys()
    for name in joint_gradients:
        assert torch.allclose(
            joint_gradients[name], split_gradients[name], atol=1e-6, rtol=1e-4
        ), name



def test_reported_halves_match_isolated_backwards_across_microbatches():
    """Each half's norm must be what that half alone produces.

    Regression: ``.grad`` accumulates, so draining only the auxiliary half
    after each microbatch left the policy half in place for the next
    microbatch's auxiliary backward to pick up. Nothing was lost -- the two
    still summed correctly -- but almost all of the policy gradient was
    reported and rescaled as auxiliary. One microbatch cannot show this;
    the misattribution starts with the second.
    """

    fraction = 0.25
    trainer = _budget_trainer(fraction)
    # Small enough that a single group needs several microbatches.
    trainer.train = replace(trainer.train, microbatch_positions=64)
    trees, _ = trainer.collect()
    batch = build_ppo_training_batch(trees, trainer.model.config, trainer.train)
    assert len(list(trainer._microbatches(batch.policy_groups[0]))) > 1

    # The update steps the optimizer, so the isolated backwards below have to
    # replay from the weights the reported norms were measured at.
    weights = copy.deepcopy(trainer.model.state_dict())
    stats = trainer.update(trees)
    assert stats.policy_loss_grad_norm > 0.0
    assert stats.auxiliary_loss_grad_norm > 0.0

    def isolated_norm(**overrides):
        solo = _budget_trainer(0.0)
        solo.train = replace(
            solo.train, microbatch_positions=64, **overrides
        )
        solo.model.load_state_dict(weights)
        solo_batch = build_ppo_training_batch(
            trees, solo.model.config, solo.train
        )
        groups = solo_batch.policy_groups
        solo._prepare_ppo_advantages(
            groups, solo_batch.critic_groups, SeqUpdateStats()
        )
        normalize_ppo_advantages(groups)
        solo.model.train()
        solo.optimizer.zero_grad(set_to_none=True)
        solo._ppo_actor_backward(groups)
        return _grad_norm(list(solo.model.parameters()))

    # Policy alone: every auxiliary coefficient off. bid_hit_coef is in the
    # list because it defaults to 0.25 and the PPO actor trains that head --
    # leaving it out puts a whole auxiliary into the "policy" baseline and
    # the comparison silently drifts.
    policy_only = isolated_norm(
        suit_coef=0.0,
        trick_coef=0.0,
        bid_hit_coef=0.0,
        ppo_actor_value_coef=0.0,
    )
    # Auxiliaries alone: no policy term and no entropy bonus.
    auxiliary_only = isolated_norm(policy_coef=0.0, ppo_entropy_mode="off")

    assert stats.policy_loss_grad_norm == pytest.approx(policy_only, rel=1e-4)
    assert stats.auxiliary_loss_grad_norm == pytest.approx(
        auxiliary_only, rel=1e-4
    )


def _actor_gradients_after_backward(trainer, trees):
    """Run one actor backward plus rebalance and return the resulting grads."""

    batch = build_ppo_training_batch(trees, trainer.model.config, trainer.train)
    groups = batch.policy_groups
    trainer._prepare_ppo_advantages(groups, batch.critic_groups, SeqUpdateStats())
    trainer.model.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer._ppo_actor_backward(groups)
    trainer._apply_auxiliary_grad_budget()
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in trainer.model.named_parameters()
        if parameter.grad is not None
    }


def test_auxiliary_grad_budget_covers_head_only_parameters():
    """Belief-head parameters get no PPO gradient, so the stash must carry them."""

    trainer = _budget_trainer(0.25)
    trees, _ = trainer.collect()
    batch = build_ppo_training_batch(trees, trainer.model.config, trainer.train)
    groups = batch.policy_groups
    trainer._prepare_ppo_advantages(groups, batch.critic_groups, SeqUpdateStats())
    trainer.model.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer._ppo_actor_backward(groups)

    # Before the rebalance, .grad holds the PPO half only, which cannot reach
    # a head the policy never reads.
    head = trainer.model.suit_presence_head
    assert all(parameter.grad is None for parameter in head.parameters())
    trainer._apply_auxiliary_grad_budget()
    assert all(parameter.grad is not None for parameter in head.parameters())
    assert trainer._auxiliary_grads is None


def test_auxiliary_grad_budget_survives_a_checkpoint_predating_the_field(tmp_path):
    """v4-era checkpoints have no such key and must still load."""

    trainer = _budget_trainer(0.0)
    path = tmp_path / "pre_budget.pt"
    trainer.save_checkpoint(path)
    state = torch.load(path, weights_only=False)
    del state["training_config"]["ppo_auxiliary_grad_fraction"]
    torch.save(state, path)

    resumed = _budget_trainer(0.25)
    resumed.load_checkpoint(path)
    assert resumed.train.ppo_auxiliary_grad_fraction == 0.25


def _bid_hit_gradient_norm(bid_hit_coef: float):
    """Run one PPO update and report the bid-hit head's gradient and loss."""

    torch.manual_seed(0)
    config = replace(
        ppo_config(auxiliary_positions="all"), bid_hit_coef=bid_hit_coef
    )
    trainer = SeqTrainer(SeqPlumpModel(MODEL), config, device="cpu")
    trees, _ = trainer.collect()

    captured: dict = {}
    original = trainer._apply_auxiliary_grad_budget

    def hook():
        if not captured:
            captured["norm"] = sum(
                float(parameter.grad.pow(2).sum())
                for parameter in trainer.model.bid_hit_head.parameters()
                if parameter.grad is not None
            )
        return original()

    trainer._apply_auxiliary_grad_budget = hook
    stats = trainer.update(trees)
    return captured.get("norm", 0.0), stats.loss_bid_hit


def test_ppo_trains_the_bid_hit_head_when_it_is_weighted():
    """The head is an MLP, so it cannot ride the packed linear projection.

    Before this was wired it was simply absent from the PPO actor's auxiliary
    set: ``bid_hit_coef`` parsed, validated and did nothing.
    """

    norm, loss = _bid_hit_gradient_norm(0.2)

    assert norm > 0.0
    assert loss > 0.0


def test_ppo_leaves_the_bid_hit_head_alone_at_zero_coefficient():
    norm, loss = _bid_hit_gradient_norm(0.0)

    assert norm == 0.0
    assert loss == 0.0
