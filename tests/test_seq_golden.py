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


EXPECTED = {
    "groups": "4bcd6ae49213d9a3743d906e74a10024a51614040a010d583ac3ac8f8f43e796",
    "summary": "d37875f56712b446caa2a77da516e3b8edb95a5216d457b6824f8118bfa260f5",
    "trees": "a03e81035014d4387d22197bcefd4c065094a03bc56e64c6af4cb3a0b88fdddd",
    "update": "5fcc9c7d1c6647a37a11db8e35d0dd34529b10588a45ffe920a20ec87f7a1e44",
    "weights": "45be7cbe8235876cfec99050861559b9fb43c2e77c4344852f62ecca7b50dcc3",
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
                "segments": [
                    (
                        positions,
                        None if resolver is None else resolver.backed_value,
                    )
                    for positions, resolver in leaf.segments
                ],
                "value_targets": leaf.value_targets(),
                "decisions": [
                    {
                        "position": record.position,
                        "phase": record.phase,
                        "action_index": record.action_index,
                        "old_probs": record.old_probs,
                        "old_value": record.old_value,
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
        "seq_weight": group.seq_weight,
        "trick_targets": group.trick_targets,
        "trick_masks": group.trick_masks,
        "suit_targets": group.suit_targets,
        "bid_hit_targets": group.bid_hit_targets,
        "policy": group.policy,
    }


def test_schema_v6_collection_and_update_match_pre_refactor_golden_hashes():
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
            play_mode="gumbel_top_k",
            play_top_k=2,
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
                if key != "collect_sec"
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
                    if key not in {"update_sec", "build_sec"}
                }
            ),
            "weights": _sha256(trainer.model.state_dict()),
        }
    )

    assert observed == EXPECTED
