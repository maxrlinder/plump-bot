"""Branch-free PPO estimator, weighting, actor-pool, and precision tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from plump.seq.config import (
    BranchBudgetConfig,
    BranchRuleConfig,
    GameScheduleCell,
    RolloutOptions,
    SLOT_CARD,
    SLOT_NEXT_ACTOR,
    SLOT_NEXT_PHASE,
    SLOT_REL_PLAYER,
    SLOT_TYPE,
    TOKEN_HAND,
    SeqModelConfig,
    SeqTrainingConfig,
)
from plump.seq.model import SeqPPOOracleCritic, SeqPlumpModel
from plump.seq.ppo import (
    build_ppo_training_batch,
    build_ppo_training_groups,
    ppo_clipped_terms,
)
from plump.seq.trainer import SeqTrainer


MODEL = SeqModelConfig(d_model=32, n_layers=1, n_heads=2, d_ff=64)


def ppo_config(
    *,
    cells=None,
    actors=1,
    precision="fp32",
    kv_dtype="fp16",
    critic_epochs=1,
    bucket_width=0,
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
        ppo_critic_epochs=critic_epochs,
        microbatch_positions=512,
        policy_kl_cap=1.0,
        policy_kl_p99_cap=1.0,
        precision=precision,
        kv_dtype=kv_dtype,
    )


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
    config = ppo_config(actors=2)
    trainer = SeqTrainer(SeqPlumpModel(MODEL), config, device="cpu")
    assert isinstance(trainer.critic, SeqPPOOracleCritic)
    critic_before = trainer.critic.player_value_head[2].weight.detach().clone()
    trees, _ = trainer.collect()
    stats = trainer.update(trees)
    assert trainer.critic is not None
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


def test_oracle_critic_has_one_sequence_per_game_and_exact_seat_ties():
    trainer = SeqTrainer(SeqPlumpModel(MODEL), ppo_config(actors=2), device="cpu")
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


def test_oracle_critic_reports_loss_dynamics_across_epochs():
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


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires Apple MPS"
)
def test_mps_bf16_ppo_smoke():
    config = ppo_config(precision="bf16", kv_dtype="bf16")
    trainer = SeqTrainer(SeqPlumpModel(MODEL), config, device="mps")
    trees, _ = trainer.collect()
    stats = trainer.update(trees)
    torch.mps.synchronize()
    assert np.isfinite(stats.loss_policy)
    assert np.isfinite(stats.loss_value)
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
