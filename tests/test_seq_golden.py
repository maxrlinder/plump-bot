"""Frozen CPU regression for the optimized schema-v6 training path."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json

import numpy as np
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

# Regenerated for decision-aligned expected-value training and separately
# clipped optimizer groups. This golden still exercises stochastic
# stratification at the bid and exhaustive-within-budget play expansion.
EXPECTED = {
    "groups": "a54f006bb08fb1a031096665267ab68ba2f5bd7b2c9477891596f9a1bc615900",
    "summary": "5dcdb89befa1747a1cacb8387b0d17c8fb640dac48498318f45507d8b32e15ed",
    "trees": "fb6fdaa6b002e015c4da805826fc80e08c30f0435ce65169b2b0d8554e96d168",
    "update": "871bbcf76d1d20b604533e8c8e94c510df2f65166cfc451c302c8d89c8f7eb2e",
    "weights": "1715e292645c0afff516457aa3631d787d7f236417e37453d089f44990db359e",
}


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
            historical_arm="off",
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
                    }
                }
            ),
            "weights": _sha256(trainer.model.state_dict()),
        }
    )

    assert observed == EXPECTED
