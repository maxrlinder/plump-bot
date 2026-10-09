"""Frozen CPU regression for the optimized schema-v6 training path."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json

import numpy as np
import pytest
import torch

from plump.seq.config import (
    BranchBudgetConfig,
    BranchRuleConfig,
    GameScheduleCell,
    RolloutOptions,
    SeqModelConfig,
    SeqTrainingConfig,
)
from plump.seq.model import SeqPlumpModel
from plump.seq.trainer import SeqTrainer, build_training_groups

# Frozen after adding observer-remaining-card composition at TRICK_WIN and
# shared rank/suit card-output directions, then refreshed for the appended
# rank-boundary readout and its zero-by-default telemetry fields. Refreshed
# again for the four causal bid-pressure columns and their zero-start additive
# embedding. Refreshed once more when fixed-width embedding sums moved to the
# mathematically equivalent embedding-bag kernel; rollout data and scalar
# update telemetry remained unchanged, while reduction ordering changed the
# post-update parameter bytes. Refreshed for compact missing-field patterns:
# the always-padding tenth remaining-hand column now carries the pattern id,
# and individual NA sentinel rows are omitted from embedding reductions. This
# covers both causal token features and parameter layout.
# Collection runs before any clipping, so it is unaffected by clip topology.
EXPECTED_COLLECTION = {
    "groups": "acc1c6b375b3812a4417729e0c1b3a67990db9ac94239d74cda0654e29518356",
    "summary": "74d6ed47c429ef53256124b7bfa7d0738c1c3fab999fd2019618ce4cbe89643a",
    "trees": "d41295fac5aa7eef45d33349d4a2b6c2fdaae6b6d41615ce0c218532c0b685c8",
}
# Post-update state. Both hashes moved when the actor became a single clip
# group: the trunk, the input embeddings and the auxiliary heads now share one
# global-norm budget and one rescale factor, where each previously had its own.
# They moved again when the actor gained its per-seat player_value_head: the
# update hash because SeqUpdateStats grew four value-probe fields, the weights
# hash because the state dict grew four keys. That head is untouched by this
# counterfactual objective, and dropping its keys from the state dict
# reproduces 9e62decff303... exactly, which is how the change was verified to
# leave every pre-existing parameter bit-identical.
#
# They moved a third time when the actor gained next_winner_head: the update
# hash because SeqUpdateStats grew loss_next_winner (the other three new
# fields are excluded below), the weights hash because the state dict grew two
# keys. Verified the same way -- against a control tree carrying every other
# change with only that head removed, the RNG position after construction is
# identical and every shared parameter is bit-identical both at init and after
# a full collect+update. Dropping the two keys reproduces d134b182c8b2...
EXPECTED_UPDATE = (
    "5b33da42f2dc74da8ac4c827eae159a00a04c628d6882a6dabd0250d46bb640d"
)
EXPECTED_WEIGHTS = (
    "6c1877aa20e315771c4922518e04070273a2b1c12f9a651044cb098c60c28575"
)


def _normalize(value):
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "hex": value.tobytes().hex(),
        }
    if isinstance(value, torch.Tensor):
        return _normalize(value.detach().cpu().contiguous().numpy())
    if dataclasses.is_dataclass(value):
        return {
            field.name: _normalize(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, dict):
        return {
            str(key): _normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        if isinstance(value, set):
            items = sorted(items, key=str)
        return [_normalize(item) for item in items]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha256(value) -> str:
    payload = json.dumps(
        _normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _branch_payload(branch):
    if branch is None:
        return None
    return {
        "candidate_indices": branch.candidate_indices,
        "prior_probs": branch.prior_probs,
        "raw_probs": branch.raw_probs,
        "inclusion_probs": branch.inclusion_probs,
        "deterministic_count": branch.deterministic_count,
        "sampled_index": branch.sampled_index,
        "candidate_mass": branch.candidate_mass,
        "child_values": branch.child_values,
        "backed_value": branch.backed_value,
    }


def _tree_payload(tree):
    return {
        "arm": tree.arm,
        "focal": tree.focal,
        "num_players": tree.num_players,
        "hand_size": tree.hand_size,
        "bidding_start_player": tree.bidding_start_player,
        "initial_hands": tree.initial_hands,
        "leaf_total": tree.leaf_total,
        "decision_total": tree.decision_total,
        "branch_decisions": tree.branch_decisions,
        "deepest_branch_trick": tree.deepest_branch_trick,
        "branched_tricks": tree.branched_tricks,
        "branch_layers": tree.branch_layers,
        "branch_decisions_by_stage": tree.branch_decisions_by_stage,
        "leaves_added_by_stage": tree.leaves_added_by_stage,
        "leaves": [
            {
                "events": leaf.env.state.event_log,
                "owned_from": leaf.owned_from,
                "terminal_value": leaf.terminal_value,
                "reach_weight": leaf.reach_weight,
                "segments": [
                    (
                        positions,
                        None if resolver is None else resolver.backed_value,
                        reach,
                    )
                    for positions, resolver, reach in leaf.segments
                ],
                "value_targets": leaf.value_targets(),
                "decisions": [
                    {
                        "position": record.position,
                        "phase": record.phase,
                        "action_index": record.action_index,
                        "old_probs": record.old_probs,
                        "old_value": record.old_value,
                        "reach_weight": record.reach_weight,
                        "depth": record.depth,
                        "branch": _branch_payload(record.branch),
                    }
                    for record in leaf.decisions
                ],
            }
            for leaf in tree.leaves
        ],
    }


def _group_payload(group):
    return {
        "num_players": group.num_players,
        "hand_size": group.hand_size,
        "tokens": group.tokens,
        "owned": group.owned,
        "value_targets": group.value_targets,
        "value_weight": group.value_weight,
        "position_weight": group.position_weight,
        "trick_targets": group.trick_targets,
        "trick_masks": group.trick_masks,
        "suit_targets": group.suit_targets,
        "bid_hit_targets": group.bid_hit_targets,
        "policy": group.policy,
    }


def test_schema_v6_collection_and_update_match_corrected_golden_hashes():
    torch.manual_seed(123)
    model_config = SeqModelConfig(
        d_model=32,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        d_ff=64,
        turn_token="bid",
    )
    train_config = SeqTrainingConfig(
        schedule_cells=(GameScheduleCell(hand_size=3, num_players=3),),
        branch_rule=BranchRuleConfig(
            bid_top_k=3,
            play_mode="stratified",
            play_top_k=3,
        ),
        branch_budget=BranchBudgetConfig(branch_rate=0.7),
        rollout=RolloutOptions(
            deals_per_batch=1,
            auto_deals_per_batch=False,
            opponent_mode="off",
            opponent_fraction=0.0,
            max_cache_rows=2048,
        ),
        microbatch_positions=2048,
        lr_warmup_updates=10,
        seed=17,
    )
    trainer = SeqTrainer(
        SeqPlumpModel(model_config),
        train_config,
        device="cpu",
    )
    trees, summary = trainer.collect()
    groups = build_training_groups(trees, model_config, train_config)

    observed = {
        "trees": _sha256([_tree_payload(tree) for tree in trees]),
        "groups": _sha256([_group_payload(group) for group in groups]),
        "summary": _sha256(
            {
                key: value
                for key, value in dataclasses.asdict(summary).items()
                if key
                not in {
                    "collect_sec",
                    # Reporting-only decomposition of outcomes already frozen
                    # through the underlying trees.
                    "bid_hit_focal",
                    "bid_hit_non_focal",
                        "reward_focal",
                        "reward_non_focal",
                        "reward_heuristic",
                        "trees_self",
                        "trees_heuristic",
                        "trees_historical",
                    }
                }
            ),
    }
    stats = trainer.update(trees)
    observed.update(
        {
            "update": _sha256(
                {
                    key: value
                    for key, value in dataclasses.asdict(stats).items()
                    if key
                    not in {
                        "update_sec",
                        "build_sec",
                        # Reporting-only trust-region diagnostics. The hashes
                        # below still freeze the accepted model update itself.
                        "backtracks",
                        "step_scale",
                        "loss_value_zero",
                        "proposed_policy_kl",
                        "proposed_policy_kl_p95",
                        "proposed_policy_kl_p99",
                        "proposed_policy_kl_max",
                        "proposed_mean_exceeded",
                        "proposed_p99_exceeded",
                        # Split-by-objective gradient telemetry. Zero unless a
                        # separate auxiliary budget is configured, which this
                        # NeuRD golden path never is.
                        "policy_loss_grad_norm",
                        "auxiliary_loss_grad_norm",
                        "auxiliary_grad_scale",
                            # Diagnostic readout of the logit direction the policy
                            # is invariant to. Cannot affect the update it reports.
                            "policy_logit_shift",
                            # Distillation-only reporting. Identically zero
                            # under this frozen NeuRD golden update, which
                            # never constructs a teacher.
                            "distill_kl",
                            "distill_kl_bid",
                            "distill_kl_play",
                            "distill_agree",
                            "distill_agree_bid",
                            "distill_agree_play",
                            "distill_top1_mass_bid",
                            "distill_top1_mass_play",
                            "distill_teacher_entropy_bid",
                            "distill_teacher_entropy_play",
                            "distill_student_entropy_bid",
                            "distill_student_entropy_play",
                            "distill_loss_policy",
                            "distill_loss_value",
                            "distill_loss_aux",
                            "distill_value_rmse",
                            "distill_value_correlation",
                            "distill_aux_kl_suit",
                            "distill_aux_kl_rank_boundary",
                            "distill_aux_kl_trick",
                            "distill_aux_kl_bid_hit",
                            "distill_aux_kl_next_winner",
                            "distill_skipped_steps",
                            "distill_aux_entropy_floor",
                            # PPO-only reporting fields are identically zero
                            # under this frozen NeuRD golden update.
                            "entropy_bid_normalized",
                            "entropy_play_normalized",
                            "entropy_target_bid",
                            "entropy_target_play",
                            "entropy_alpha_bid",
                            "entropy_alpha_play",
                            "ppo_ratio_clip_fraction",
                            "ppo_behavior_replay_kl",
                            "advantage_mean",
                            "advantage_std",
                            "critic_grad_norm",
                            "critic_all_player_rmse",
                            "critic_all_player_correlation",
                            "critic_loss_first_epoch",
                            "critic_loss_last_epoch",
                                "critic_loss_reduction",
                                "loss_oracle_trick",
                                "loss_oracle_card_order",
                                # Oracle-side and PPO-side next-trick-winner
                                # reporting. The NeuRD golden path fills
                                # loss_next_winner (kept in the hash below)
                                # and leaves these three at zero.
                                "loss_oracle_next_winner",
                                "next_winner_accuracy",
                                "oracle_next_winner_accuracy",
                                    "suit_accuracy_10c_0",
                                    "suit_accuracy_10c_4",
                                    "suit_accuracy_10c_8",
                                    "suit_loss_10c_0",
                                    "suit_loss_10c_4",
                                    "suit_loss_10c_8",
                                    "rank_boundary_loss_10c_0",
                                    "rank_boundary_loss_10c_4",
                                    "rank_boundary_loss_10c_8",
                                "trick_accuracy_10c_0",
                                "trick_accuracy_10c_4",
                                "trick_accuracy_10c_8",
                                "oracle_trick_accuracy",
                                "oracle_card_order_top1_accuracy",
                                "peak_update_device_bytes",
                        }
                }
            ),
            "weights": _sha256(trainer.model.state_dict()),
        }
    )

    for key, digest in EXPECTED_COLLECTION.items():
        assert observed[key] == digest, f"collection hash {key} moved"
    assert observed["update"] == EXPECTED_UPDATE
    assert observed["weights"] == EXPECTED_WEIGHTS
