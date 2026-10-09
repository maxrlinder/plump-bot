"""Trainer for the schema-v6 sequence pipeline.

One update consumes collected counterfactual trees as full causal forwards
grouped by (players, hand size). The default policy objective is sampled NeuRD.
For a frozen value control variate ``b`` and exact candidate inclusion ``q``:

    Q_hat(a) = b + I(a) / q(a) * (Q(a) - b)
    A_hat(a) = Q_hat(a) - sum_x pi_old(x) Q_hat(x)
    L = -sum_a sg[A_hat(a)] * y(a)

With exponent one and clipping/capping disabled, this gives the full legal
action regret vector in expectation. The selectable ``sampled_mirror`` option
exponentiates the same realized vector; it is a deliberate stochastic
bias/variance alternative, not claimed to be unbiased full-information mirror
descent.

Expanded paths carry empirical old-policy reach. Uniform-only explorer
descendants therefore supply their parent's Q value but receive zero
descendant policy/value/belief weight. Both mean and p99 post-update KL caps
must pass before an Adam step is accepted.
"""

from __future__ import annotations

import copy
import math
import os
import random
import time
import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from plump.rewards import compute_relative_rewards
from plump.rounds import rules_fingerprint

from .config import (
    NEXT_BID,
    SEQ_SCHEMA_VERSION,
    SeqModelConfig,
    SeqTrainingConfig,
)
from .model import (
    ADDITIVE_CONDITIONING_KEYS,
    NA_PATTERN_EMBEDDING_KEYS,
    NEXT_WINNER_HEAD_KEYS,
    ORACLE_CARD_ORDER_HEAD_KEYS,
    PLAYER_VALUE_HEAD_KEYS,
    RANK_BOUNDARY_HEAD_KEYS,
    SEQ_MODEL_FORMAT_VERSION,
    STRUCTURED_CARD_OUTPUT_KEYS,
    SeqPlumpModel,
    SeqPPOCritic,
    SeqPPOOracleCritic,
    EMBEDDING_NORM_KEYS,
    load_seq_model_state_dict,
)
from .policy import SeqLeague, best_seq_device
from .metal_trunk import (
    DualTrunkPass,
    metal_trunk_requested,
    unsupported_reason as metal_trunk_unsupported_reason,
)
from .precision import autocast_context
from .ppo import (
    PPO_BELIEF_STAGES,
    PPOCriticGroup,
    PPOTrainingGroup,
    build_ppo_training_batch,
    center_zero_sum_values,
    normalize_ppo_advantages,
    oracle_gae_advantages,
    plackett_luce_card_order_terms,
    ppo_clipped_terms,
    ppo_policy_kl,
    ppo_critic_rows_by_chunk,
    ppo_rows_by_chunk,
)
from .rollout import SeqRolloutCollector, SeqTree
from .tokens import (
    IGNORE_LABEL,
    TOKEN_WIDTH,
    build_replay_arrays,
    build_seat_tokens,
)


_AUXILIARY_HEAD_PREFIXES = (
    "value_head.",
    "player_value_head.",
    "trick_count_head.",
    "suit_presence_head.",
    "bid_hit_head.",
    "rank_boundary_head.",
    "next_winner_head.",
)

# Input-side embedding tables. Their gradients are scatter-adds accumulated
# over every token occurrence, so a handful of hot rows dominate the actor's
# global norm. The actor is clipped as one group, so this list only selects
# which parameters get their pre-clip norm reported separately. Output
# embeddings stay out of it: they are read like a head, once per decision.
# Training-config fields introduced after checkpoints were already being
# written. A checkpoint that predates one simply lacks the key, which must not
# read as a configuration mismatch and strand every earlier checkpoint.
_ADDITIVE_TRAINING_FIELDS: tuple[str, ...] = (
    "ppo_actor_value_coef",
    "ppo_critic_sequence_bucket_width",
    "ppo_advantage_estimator",
    "ppo_gae_lambda",
    "next_winner_coef",
    "dense_supervision_start",
    "ppo_auxiliary_grad_fraction",
    "ppo_trust_region",
)

_INPUT_EMBEDDING_PREFIXES = (
    "slot_embedding.",
    "pos_embedding.",
    "bid_pressure_embedding.",
    "na_pattern_embedding.",
    "per_layer_slot_embedding.",
    "per_layer_pos_embedding.",
    "per_layer_na_pattern_embedding.",
    "value_slot_embedding.",
    "value_pos_embedding.",
    "value_na_pattern_embedding.",
)


def _grad_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    """Global norm of the gradients currently held by ``parameters``."""

    grads = [
        parameter.grad.detach()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not grads:
        return 0.0
    return float(
        torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(grad) for grad in grads])
        )
    )


# --------------------------------------------------------------------- #
# Batch containers                                                       #
# --------------------------------------------------------------------- #


@dataclass
class PolicyRows:
    """Counterfactual policy rows for one phase within one group.

    One row per focal decision, branched or not. A branched decision carries
    its candidate set; an unbranched one carries the single realized action.
    """

    seq_index: list[int] = field(default_factory=list)
    position: list[int] = field(default_factory=list)
    old_probs_full: list[np.ndarray] = field(default_factory=list)
    candidates: list[np.ndarray] = field(default_factory=list)
    q_values: list[np.ndarray] = field(default_factory=list)
    # Frozen value-head prediction made before candidate sampling. It is an
    # independent control variate in Q_hat(a) = b + I(a)/q(a) * (Q(a) - b).
    baseline: list[float] = field(default_factory=list)
    # P(this action was expanded here), per candidate. See the module docstring.
    inclusion: list[np.ndarray] = field(default_factory=list)
    weight: list[float] = field(default_factory=list)
    # Diagnostics only: how the row was produced.
    branched: list[bool] = field(default_factory=list)


@dataclass
class SeqTrainingGroup:
    num_players: int
    hand_size: int
    tokens: np.ndarray  # [B, L, WIDTH]
    owned: np.ndarray  # [B, L] bool
    value_targets: np.ndarray  # [B, L] float32
    value_weight: np.ndarray  # [B, L] focal-decision value weight
    position_weight: np.ndarray  # [B, L] on-policy dense-auxiliary weight
    trick_targets: np.ndarray  # [B, max_players]
    trick_masks: np.ndarray
    next_winner_targets: np.ndarray  # [B, L], IGNORE_LABEL outside the play phase
    suit_targets: np.ndarray
    rank_boundary_targets: np.ndarray
    bid_hit_targets: np.ndarray  # [B, max_players]
    policy: dict[str, PolicyRows]


@dataclass
class SeqUpdateStats:
    loss_policy: float = 0.0
    loss_value: float = 0.0
    loss_value_zero: float = 0.0
    value_rmse: float = 0.0
    value_zero_rmse: float = 0.0
    value_correlation: float = 0.0
    value_prediction_std: float = 0.0
    loss_suit: float = 0.0
    loss_rank_boundary: float = 0.0
    loss_trick: float = 0.0
    loss_oracle_trick: float = 0.0
    loss_next_winner: float = 0.0
    loss_oracle_next_winner: float = 0.0
    loss_oracle_card_order: float = 0.0
    loss_bid_hit: float = 0.0
    suit_accuracy_10c_0: float = 0.0
    suit_accuracy_10c_4: float = 0.0
    suit_accuracy_10c_8: float = 0.0
    suit_loss_10c_0: float = 0.0
    suit_loss_10c_4: float = 0.0
    suit_loss_10c_8: float = 0.0
    rank_boundary_balanced_accuracy_10c_0: float = 0.0
    rank_boundary_balanced_accuracy_10c_4: float = 0.0
    rank_boundary_balanced_accuracy_10c_8: float = 0.0
    rank_boundary_loss_10c_0: float = 0.0
    rank_boundary_loss_10c_4: float = 0.0
    rank_boundary_loss_10c_8: float = 0.0
    trick_accuracy_10c_0: float = 0.0
    trick_accuracy_10c_4: float = 0.0
    trick_accuracy_10c_8: float = 0.0
    oracle_trick_accuracy: float = 0.0
    next_winner_accuracy: float = 0.0
    oracle_next_winner_accuracy: float = 0.0
    oracle_card_order_top1_accuracy: float = 0.0
    # --- distillation ---
    # Forward KL(teacher || student) in nats, per decision, split by phase.
    distill_kl_bid: float = 0.0
    distill_kl_play: float = 0.0
    distill_kl: float = 0.0
    # Share of decisions where the student's argmax matches the teacher's.
    # This is the headline "am I mimicking yet" number.
    distill_agree_bid: float = 0.0
    distill_agree_play: float = 0.0
    distill_agree: float = 0.0
    # Probability the student assigns to the teacher's argmax action.
    distill_top1_mass_bid: float = 0.0
    distill_top1_mass_play: float = 0.0
    # Teacher and student entropies, to show the student is matching the
    # teacher's spread rather than collapsing onto its mode.
    distill_teacher_entropy_bid: float = 0.0
    distill_teacher_entropy_play: float = 0.0
    distill_student_entropy_bid: float = 0.0
    distill_student_entropy_play: float = 0.0
    # Per-head KL(teacher || student), zero at a perfect match. Reported as
    # KL rather than the optimized cross-entropy so the heads are mutually
    # comparable and share a meaningful floor.
    distill_aux_kl_suit: float = 0.0
    distill_aux_kl_rank_boundary: float = 0.0
    distill_aux_kl_trick: float = 0.0
    distill_aux_kl_bid_hit: float = 0.0
    distill_aux_kl_next_winner: float = 0.0
    # Updates whose gradient was non-finite and therefore skipped. Any value
    # above zero means something upstream produced a NaN; the run survives it
    # but the cause still needs finding.
    distill_skipped_steps: float = 0.0
    # The part of ``distill_loss_aux`` that is the teacher's own entropy and
    # therefore cannot be trained away. Reported so the auxiliary curve can be
    # read against its floor: the cross-entropy is dominated by it, and looks
    # flat long before the learnable part stops moving.
    distill_aux_entropy_floor: float = 0.0
    distill_loss_policy: float = 0.0
    distill_loss_value: float = 0.0
    distill_loss_aux: float = 0.0
    # Student-vs-teacher value agreement on the same states.
    distill_value_rmse: float = 0.0
    distill_value_correlation: float = 0.0
    entropy: float = 0.0
    entropy_bid_normalized: float = 0.0
    entropy_play_normalized: float = 0.0
    entropy_target_bid: float = 0.0
    entropy_target_play: float = 0.0
    entropy_alpha_bid: float = 0.0
    entropy_alpha_play: float = 0.0
    ppo_ratio_clip_fraction: float = 0.0
    ppo_behavior_replay_kl: float = 0.0
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    # Weighted mean legal logit. The policy is invariant to shifting every
    # legal logit by a constant, so the KL guard cannot bound this direction;
    # NeuRD's gradient along it is nonzero. Watch for a monotone trend.
    policy_logit_shift: float = 0.0
    policy_kl_checked: bool = True
    policy_kl: float = 0.0
    policy_kl_p95: float = 0.0
    policy_kl_p99: float = 0.0
    policy_kl_max: float = 0.0
    # Exact KL of the nominal Adam proposal before any learning-rate
    # backtracking. These remain distinct from the accepted policy_kl fields
    # so a small accepted KL cannot hide an initially oversized proposal.
    proposed_policy_kl: float = 0.0
    proposed_policy_kl_p95: float = 0.0
    proposed_policy_kl_p99: float = 0.0
    proposed_policy_kl_max: float = 0.0
    proposed_mean_exceeded: bool = False
    proposed_p99_exceeded: bool = False
    rolled_back: bool = False
    backtracks: int = 0
    step_scale: float = 1.0
    policy_rows: int = 0
    # Split of policy_rows by how the row was produced. Diagnostics: both
    # feed the same loss.
    branched_rows: int = 0
    unbranched_rows: int = 0
    value_rows: int = 0
    positions: int = 0
    # Pre-clip global norm over every actor parameter -- the quantity the
    # single actor clip is applied to. The two below are sub-norms of it,
    # reported for diagnostics only; neither has its own clip budget.
    core_grad_norm: float = 0.0
    embedding_grad_norm: float = 0.0
    auxiliary_grad_norm: float = 0.0
    critic_grad_norm: float = 0.0
    # Separate-budget telemetry, non-zero only when
    # ppo_auxiliary_grad_fraction > 0. Unlike the three norms above, which
    # slice the summed gradient by parameter, these split it by *objective*:
    # each is the norm over every actor parameter of what one half of the
    # loss alone produced, measured before the rebalance and before the clip.
    # Their ratio is what the budget controls; auxiliary_grad_scale is the
    # factor applied to reach it.
    policy_loss_grad_norm: float = 0.0
    auxiliary_loss_grad_norm: float = 0.0
    auxiliary_grad_scale: float = 0.0
    # Oracle-critic telemetry. Acting-seat metrics above are the baseline PPO
    # consumes; these cover every active output head and optimization progress
    # across the repeated critic epochs.
    critic_all_player_rmse: float = 0.0
    critic_all_player_correlation: float = 0.0
    # The oracle's per-pass value loss, i.e. loss_value divided by the number
    # of critic epochs, so it is on the same footing as the actor's one pass.
    oracle_value_loss: float = 0.0
    # Actor per-seat value probe, measured on exactly the oracle's dense
    # objective: same targets, same zero-sum centering, same per-game mass.
    actor_value_loss: float = 0.0
    actor_all_player_rmse: float = 0.0
    actor_value_zero_rmse: float = 0.0
    critic_loss_first_epoch: float = 0.0
    critic_loss_last_epoch: float = 0.0
    critic_loss_reduction: float = 0.0
    peak_update_device_bytes: int = 0
    update_sec: float = 0.0
    build_sec: float = 0.0


@dataclass
class SeqRolloutSummary:
    trees: int = 0
    trees_self: int = 0
    trees_heuristic: int = 0
    trees_historical: int = 0
    leaves: int = 0
    decisions: int = 0
    # Legacy pooled accuracy across every seat, retained for old metrics and
    # consumers. The explicit focal/non-focal fields below are the useful
    # diagnostics: opponents take ordinary policy actions without focal
    # counterfactual branching.
    bid_hit_rate: float = 0.0
    bid_hit_focal: float = 0.0
    bid_hit_non_focal: float = 0.0
    reward_focal: float = 0.0
    reward_non_focal: float = 0.0
    reward_self: float = 0.0
    reward_heuristic: float = 0.0
    reward_historical: float = 0.0
    spine_entropy: float = 0.0
    collect_sec: float = 0.0
    forward_rows: int = 0


def summarize_trees(trees: list[SeqTree], collector_stats) -> SeqRolloutSummary:
    summary = SeqRolloutSummary(
        trees=len(trees),
        trees_self=sum(tree.arm == "self" for tree in trees),
        trees_heuristic=sum(tree.arm == "heuristic" for tree in trees),
        trees_historical=sum(tree.arm == "historical" for tree in trees),
        leaves=sum(tree.leaf_total for tree in trees),
        decisions=sum(tree.decision_total for tree in trees),
        collect_sec=collector_stats.collect_sec,
        forward_rows=collector_stats.forward_rows,
    )
    focal_hits = 0
    focal_bids = 0
    non_focal_hits = 0
    non_focal_bids = 0
    focal_rewards: list[float] = []
    non_focal_rewards: list[float] = []
    rewards: dict[str, list[float]] = defaultdict(list)
    entropies: list[float] = []
    for tree in trees:
        spine = next(leaf for leaf in tree.leaves if leaf.on_policy_spine)
        round_state = spine.env.state.current_round
        bids = {bid.player: bid.value for bid in round_state.bids}
        for player, bid in bids.items():
            hit = int(round_state.tricks_won[player] == bid)
            if player == tree.focal:
                focal_bids += 1
                focal_hits += hit
            else:
                non_focal_bids += 1
                non_focal_hits += hit
        relative_rewards = compute_relative_rewards(round_state.round_scores)
        focal_rewards.append(relative_rewards[tree.focal])
        non_focal_rewards.extend(
            reward
            for player, reward in relative_rewards.items()
            if player != tree.focal
        )
        rewards[tree.arm].append(spine.terminal_value)
        for record in spine.decisions:
            probs = record.old_probs[record.old_probs > 0]
            entropies.append(float(-(probs * np.log(probs)).sum()))
    total_hits = focal_hits + non_focal_hits
    total_bids = focal_bids + non_focal_bids
    summary.bid_hit_rate = total_hits / total_bids if total_bids else 0.0
    summary.bid_hit_focal = focal_hits / focal_bids if focal_bids else 0.0
    summary.bid_hit_non_focal = (
        non_focal_hits / non_focal_bids if non_focal_bids else 0.0
    )
    summary.reward_focal = (
        float(np.mean(focal_rewards)) if focal_rewards else 0.0
    )
    summary.reward_non_focal = (
        float(np.mean(non_focal_rewards)) if non_focal_rewards else 0.0
    )
    summary.reward_self = float(np.mean(rewards["self"])) if rewards["self"] else 0.0
    summary.reward_heuristic = (
        float(np.mean(rewards["heuristic"])) if rewards["heuristic"] else 0.0
    )
    summary.reward_historical = (
        float(np.mean(rewards["historical"])) if rewards["historical"] else 0.0
    )
    summary.spine_entropy = float(np.mean(entropies)) if entropies else 0.0
    return summary


def sampled_mirror_target(
    old_probs: torch.Tensor,
    advantages: torch.Tensor,
    legal: torch.Tensor,
    *,
    step_size: float,
    uniform_mix: float,
    target_kl: float,
    bisection_steps: int = 16,
) -> torch.Tensor:
    """Return an entropic mirror target for one sampled advantage vector.

    ``advantages`` is already a full-support stochastic estimate. The exact
    update for that realized estimate is

        target ∝ anchor * exp(step_size * advantages)

    where ``anchor`` is ``old_probs`` mixed with legal-uniform exploration.
    Written relative to ``old_probs``, this is one exponentiated direction.
    Scaling that direction is itself an entropic mirror-descent step, so a
    vectorized bisection can enforce ``KL(old || target) <= target_kl`` without
    clipping probabilities after normalization.
    """

    old_probs = old_probs.detach()
    advantages = advantages.detach()
    legal_float = legal.float()
    legal_count = legal_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
    uniform = legal_float / legal_count
    anchor = (1.0 - uniform_mix) * old_probs + uniform_mix * uniform

    log_old = torch.log(old_probs.clamp_min(1e-12))
    direction = torch.log(anchor.clamp_min(1e-12)) - log_old + step_size * advantages
    direction = torch.where(legal, direction, torch.zeros_like(direction))

    def at(scale: torch.Tensor) -> torch.Tensor:
        scores = log_old + scale * direction
        return torch.softmax(scores.masked_fill(~legal, float("-inf")), dim=-1)

    full = at(torch.ones((old_probs.shape[0], 1), device=old_probs.device))
    if target_kl <= 0:
        return full

    def old_to(candidate: torch.Tensor) -> torch.Tensor:
        log_candidate = torch.log(candidate.clamp_min(1e-12))
        return (old_probs * (log_old - log_candidate) * legal_float).sum(
            dim=-1, keepdim=True
        )

    full_kl = old_to(full)
    low = torch.zeros_like(full_kl)
    high = torch.ones_like(full_kl)
    for _ in range(bisection_steps):
        middle = (low + high) * 0.5
        acceptable = old_to(at(middle)) <= target_kl
        low = torch.where(acceptable, middle, low)
        high = torch.where(acceptable, high, middle)
    scale = torch.where(full_kl <= target_kl, torch.ones_like(low), low)
    return at(scale)


def control_variate_action_advantages(
    old_probs: torch.Tensor,
    candidates: torch.Tensor,
    q_values: torch.Tensor,
    inclusion: torch.Tensor,
    candidate_mask: torch.Tensor,
    baseline: torch.Tensor,
    *,
    inclusion_exponent: float = 1.0,
    inclusion_cap: float = 0.0,
    advantage_clip: float = 0.0,
) -> torch.Tensor:
    """Estimate the full legal-action advantage vector.

    For legal action ``a`` and a frozen, pre-sampling baseline ``b``:

    ``Q_hat(a) = b + I(a) / q(a) * (Q(a) - b)``.

    Centering that complete vector under ``old_probs`` gives a sampled NeuRD
    regret vector whose expectation is ``Q_pi(a) - V_pi`` when exponent is one,
    the cap and clip are disabled, and the supplied inclusion probabilities are
    exact. Unobserved actions receive the control variate, not a fabricated
    zero advantage.
    """

    residual = (q_values - baseline) * candidate_mask.float()
    if advantage_clip > 0:
        residual = residual.clamp(-advantage_clip, advantage_clip)
    correction = inclusion.clamp_min(1e-12).pow(-inclusion_exponent)
    if inclusion_cap > 0:
        correction = correction.clamp_max(inclusion_cap)
    corrected = residual * correction * candidate_mask.float()
    observed_residual = torch.zeros_like(old_probs).scatter_add(
        1, candidates, corrected
    )
    legal = old_probs > 0
    q_hat = torch.where(
        legal,
        baseline.expand_as(old_probs) + observed_residual,
        torch.zeros_like(old_probs),
    )
    value_hat = (old_probs * q_hat).sum(dim=-1, keepdim=True)
    return (q_hat - value_hat) * legal.float()


def _weighted_kl_summary(
    values: np.ndarray, weights: np.ndarray
) -> tuple[float, float, float, float]:
    positive = weights > 0
    values = values[positive]
    weights = weights[positive]
    if values.size == 0 or float(weights.sum()) <= 0:
        return 0.0, 0.0, 0.0, 0.0
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    total = float(cumulative[-1])

    def percentile(q: float) -> float:
        index = int(np.searchsorted(cumulative, q * total, side="left"))
        return float(ordered_values[min(index, len(ordered_values) - 1)])

    mean = float(np.dot(values, weights) / total)
    return mean, percentile(0.95), percentile(0.99), float(values.max())


# --------------------------------------------------------------------- #
# Batch building                                                         #
# --------------------------------------------------------------------- #


def _branch_depth_scale(trees, exponent: float) -> dict[int, float]:
    """Per-decision multiplier that re-weights rows by depth.

    Normalized per tree so the tree's total policy weight is untouched: this
    knob decides where inside a tree the gradient lands, and must not double as
    a way of making some trees count more.
    """

    if exponent == 0.0:
        return {}
    scale: dict[int, float] = {}
    for tree in trees:
        records = [record for leaf in tree.leaves for record in leaf.decisions]
        if not records:
            continue
        raw = {id(record): (1.0 + record.depth) ** exponent for record in records}
        reach_total = sum(record.reach_weight for record in records)
        scaled_total = sum(record.reach_weight * raw[id(record)] for record in records)
        if reach_total <= 0 or scaled_total <= 0:
            continue
        normalizer = reach_total / scaled_total
        for key, value in raw.items():
            scale[key] = value * normalizer
    return scale


def build_training_groups(
    trees: list[SeqTree],
    model_config: SeqModelConfig,
    train_config: SeqTrainingConfig,
) -> list[SeqTrainingGroup]:
    by_shape: dict[tuple[int, int], list[tuple[SeqTree, object]]] = defaultdict(list)
    for tree in trees:
        for leaf in tree.leaves:
            by_shape[(tree.num_players, tree.hand_size)].append((tree, leaf))

    per_tree = train_config.tree_weighting == "per_tree"
    dense_reach_by_leaf: dict[int, dict[int, float]] = {}
    for tree in trees:
        dense_start = train_config.dense_start_position(
            model_config,
            num_players=tree.num_players,
            hand_size=tree.hand_size,
        )
        for leaf in tree.leaves:
            dense_reach_by_leaf[id(leaf)] = {
                position: reach
                for position, reach in leaf.position_reach_weights().items()
                if position >= dense_start
            }
    owned_per_tree = {
        id(tree): sum(
            sum(dense_reach_by_leaf[id(leaf)].values()) for leaf in tree.leaves
        )
        for tree in trees
    }
    rows_per_tree_policy = {
        id(tree): sum(
            record.reach_weight for leaf in tree.leaves for record in leaf.decisions
        )
        for tree in trees
    }
    exponent = float(train_config.tree_weight_exponent) if per_tree else 1.0

    importance = {
        id(tree): (
            float(tree.hand_size) ** train_config.shape_importance_exponent
            * float(tree.num_players) ** train_config.player_importance_exponent
        )
        for tree in trees
    }

    def row_weights(rows_per_tree: dict[int, float]):
        """Per-row weight for each tree, from its share of the total.

        A tree's share is proportional to ``importance * rows ** exponent``: at
        exponent 0 every tree counts the same however large it grew, at 1 every
        row counts the same. Trees with no rows of this kind are dropped rather
        than given a share, so the remaining trees still sum to one.
        """

        shares = {
            key: importance[key] * float(rows) ** exponent if rows > 0 else 0.0
            for key, rows in rows_per_tree.items()
        }
        total = sum(shares.values())
        if total <= 0:
            return {key: 0.0 for key in rows_per_tree}
        return {
            key: shares[key] / total / max(rows_per_tree[key], 1e-12)
            for key in rows_per_tree
        }

    seq_weights = row_weights(owned_per_tree)
    policy_weights = row_weights(rows_per_tree_policy)

    depth_scale = _branch_depth_scale(trees, train_config.branch_depth_exponent)

    def seq_weight_for(tree: SeqTree) -> float:
        return seq_weights[id(tree)]

    def policy_weight_for(tree: SeqTree) -> float:
        return policy_weights[id(tree)]

    groups: list[SeqTrainingGroup] = []
    for (num_players, hand_size), entries in sorted(by_shape.items()):
        length = model_config.seq_len(num_players, hand_size)
        batch = len(entries)
        tokens = np.zeros((batch, length, TOKEN_WIDTH), dtype=np.int64)
        owned = np.zeros((batch, length), dtype=bool)
        value_targets = np.zeros((batch, length), dtype=np.float32)
        value_weight = np.zeros((batch, length), dtype=np.float32)
        position_weight = np.zeros((batch, length), dtype=np.float32)
        trick_targets = np.full(
            (batch, model_config.max_players), IGNORE_LABEL, np.int64
        )
        trick_masks = np.zeros(
            (batch, length, model_config.max_players, model_config.bid_count),
            dtype=bool,
        )
        next_winner_targets = np.full((batch, length), IGNORE_LABEL, np.int64)
        suit_targets = np.full(
            (batch, length, model_config.belief_opponents, 4), IGNORE_LABEL, np.int64
        )
        rank_boundary_targets = np.full(
            (
                batch,
                length,
                model_config.belief_opponents + 1,
                4,
                2,
            ),
            IGNORE_LABEL,
            np.int64,
        )
        bid_hit_targets = np.full(
            (batch, model_config.max_players), IGNORE_LABEL, np.int64
        )
        policy = {"bid": PolicyRows(), "play": PolicyRows()}

        # Tokens first, in a pass of their own, so a branch child can copy the
        # prefix its parent already wrote instead of replaying the same events.
        # A parent always branched strictly earlier than its child, so ordering
        # by owned_from is a topological order: the parent's row is filled by
        # the time the child reads it. The batch array is the only storage --
        # nothing extra is retained.
        #
        # This pass alone, and not the label loop below: the labels need the
        # replay state walked event by event to reach owned_from, so they are
        # not shareable the way the tokens are.
        row_of_leaf = {id(leaf): row for row, (_, leaf) in enumerate(entries)}
        for row in sorted(range(batch), key=lambda index: entries[index][1].owned_from):
            tree, leaf = entries[row]
            parent_row = (
                None if leaf.parent is None else row_of_leaf.get(id(leaf.parent))
            )
            tokens[row] = build_seat_tokens(
                model_config,
                leaf.env.state.event_log,
                tree.focal,
                num_players,
                hand_size,
                tree.initial_hands[tree.focal],
                tree.bidding_start_player,
                token_prefix=(
                    None
                    if parent_row is None
                    else tokens[parent_row, : leaf.owned_from]
                ),
            )

        # Row order here stays the entries order: it sets the order of the
        # policy rows, and reordering those would change float reduction order.
        for row, (tree, leaf) in enumerate(entries):
            arrays = build_replay_arrays(
                model_config,
                tree.initial_hands,
                leaf.env.state.event_log,
                tree.focal,
                num_players,
                hand_size,
                tree.bidding_start_player,
                label_from=leaf.owned_from,
                tokens=tokens[row],
                suit_labels=train_config.suit_coef > 0,
                trick_labels=train_config.trick_coef > 0,
                rank_boundary_labels=train_config.rank_boundary_coef > 0,
                next_winner_labels=train_config.next_winner_coef > 0,
            )
            owned[row, leaf.owned_from :] = True
            for position, value in leaf.value_targets().items():
                value_targets[row, position] = value
            for position, reach in dense_reach_by_leaf[id(leaf)].items():
                position_weight[row, position] = seq_weight_for(tree) * reach
            if train_config.value_positions == "all":
                value_weight[row] = position_weight[row]
            trick_targets[row] = arrays.trick_targets
            trick_masks[row] = arrays.trick_masks
            next_winner_targets[row] = arrays.next_winner_targets
            suit_targets[row] = arrays.suit_targets
            rank_boundary_targets[row] = arrays.rank_boundary_targets
            bid_hit_targets[row] = arrays.bid_hit_targets

            for record in leaf.decisions:
                phase_key = "bid" if record.phase == NEXT_BID else "play"
                rows = policy[phase_key]
                decision_weight = (
                    policy_weight_for(tree)
                    * record.reach_weight
                    * depth_scale.get(id(record), 1.0)
                )
                if train_config.value_positions == "policy":
                    value_weight[row, record.position] += decision_weight
                rows.seq_index.append(row)
                rows.position.append(record.position)
                rows.old_probs_full.append(record.old_probs)
                b = record.branch
                if b is None:
                    # k=1: the only Q we have is the realized return, so the
                    # baseline has to come from the value head. Inclusion is
                    # exactly the policy mass on the action, since sampling is
                    # the only way this action got a Q at all.
                    action = record.action_index
                    rows.candidates.append(np.asarray([action], dtype=np.int64))
                    rows.q_values.append(
                        np.asarray(
                            [leaf.value_target_at(record.position)],
                            dtype=np.float32,
                        )
                    )
                    rows.baseline.append(float(record.old_value))
                    rows.inclusion.append(
                        np.asarray([record.old_probs[action]], dtype=np.float32)
                    )
                    rows.branched.append(False)
                else:
                    rows.candidates.append(
                        np.asarray(b.candidate_indices, dtype=np.int64)
                    )
                    rows.q_values.append(
                        np.asarray(
                            [b.child_values[i] for i in b.candidate_indices],
                            dtype=np.float32,
                        )
                    )
                    rows.baseline.append(float(record.old_value))
                    rows.inclusion.append(
                        np.asarray(b.inclusion_probs, dtype=np.float32)
                    )
                    rows.branched.append(True)
                rows.weight.append(decision_weight)

        groups.append(
            SeqTrainingGroup(
                num_players=num_players,
                hand_size=hand_size,
                tokens=tokens,
                owned=owned,
                value_targets=value_targets,
                value_weight=value_weight,
                position_weight=position_weight,
                trick_targets=trick_targets,
                trick_masks=trick_masks,
                next_winner_targets=next_winner_targets,
                suit_targets=suit_targets,
                rank_boundary_targets=rank_boundary_targets,
                bid_hit_targets=bid_hit_targets,
                policy=policy,
            )
        )
    return groups


# --------------------------------------------------------------------- #
# Trainer                                                                #
# --------------------------------------------------------------------- #


def load_distillation_teacher(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    student_config: SeqModelConfig,
) -> SeqPlumpModel:
    """Load a frozen teacher, using the checkpoint's own architecture.

    The teacher keeps whatever width, depth and card-embedding mode it was
    trained with; only the *token schema* has to match the student, because
    both models read the same rollout tensors. That check is explicit here
    rather than left to a shape error deep inside the first forward.
    """

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    schema = int(payload.get("schema_version", 0))
    if schema != SEQ_SCHEMA_VERSION:
        raise ValueError(
            f"Teacher checkpoint has sequence schema {schema}, "
            f"expected {SEQ_SCHEMA_VERSION}."
        )
    teacher_config = SeqModelConfig(**payload["model_config"])
    mismatched = [
        name
        for name in ("trick_win_token", "turn_token", "max_players", "max_hand_size")
        if getattr(teacher_config, name) != getattr(student_config, name)
    ]
    if mismatched or teacher_config.slot_vocab_sizes != student_config.slot_vocab_sizes:
        raise ValueError(
            "Teacher and student must serialize tokens identically; they differ "
            f"in {mismatched or ['slot_vocab_sizes']}."
        )
    teacher = SeqPlumpModel(teacher_config)
    load_seq_model_state_dict(teacher, payload["model_state_dict"])
    teacher = teacher.to(device)
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher


class SeqTrainer:
    def __init__(
        self,
        model: SeqPlumpModel,
        train_config: SeqTrainingConfig,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        train_config.validate()
        self.device = torch.device(device) if device is not None else best_seq_device()
        self.train = train_config
        self.models = [model]
        if train_config.policy_objective == "ppo":
            self.models.extend(
                copy.deepcopy(model)
                for _ in range(train_config.ppo_trainable_policies - 1)
            )
        self.models = [candidate.to(self.device) for candidate in self.models]
        self.model = self.models[0]
        core_parameters = []
        auxiliary_parameters = []
        for actor in self.models:
            for name, parameter in actor.named_parameters():
                target = (
                    auxiliary_parameters
                    if name.startswith(_AUXILIARY_HEAD_PREFIXES)
                    else core_parameters
                )
                target.append(parameter)
        self._core_parameters = core_parameters
        self._auxiliary_parameters = auxiliary_parameters
        # The actor is optimized and clipped as one whole. This sub-list is a
        # reporting slice only: it lets the dashboard show how much of the
        # actor's pre-clip norm the input embedding tables contribute, without
        # giving them a separate budget or a separate step size.
        embedding_parameters = [
            parameter
            for actor in self.models
            for name, parameter in actor.named_parameters()
            if not name.startswith(_AUXILIARY_HEAD_PREFIXES)
            and name.startswith(_INPUT_EMBEDDING_PREFIXES)
        ]
        self.migrated_model_keys: frozenset[str] = frozenset()
        self._embedding_parameters = embedding_parameters
        # Per-parameter gradients for each half of the actor loss, held while
        # the microbatch loop runs. autograd.grad keeps both halves separate
        # without materializing and then copying/clearing parameter.grad.
        self._auxiliary_grads: list[torch.Tensor | None] | None = None
        self._policy_grads: list[torch.Tensor | None] | None = None
        self._actor_parameters = [
            parameter
            for parameter in (*core_parameters, *auxiliary_parameters)
            if parameter.requires_grad
        ]
        # Coupled PPO auxiliaries share both the trunk and the configured
        # learning rate with the policy.  Keep them in one Adam group so the
        # optimizer can process the actor as one parameter collection.  The
        # split topology remains necessary for detached heads (which are not
        # KL-sensitive) or when callers explicitly request different rates.
        self._unified_actor_optimizer = (
            train_config.policy_objective == "ppo"
            and not train_config.ppo_actor_auxiliary_detach_hidden
            and train_config.core_lr == train_config.auxiliary_lr
        )
        if self._unified_actor_optimizer:
            actor_parameter_groups = (
                {
                    "params": [*core_parameters, *auxiliary_parameters],
                    "kl_sensitive": True,
                    "lr": train_config.core_lr,
                },
            )
        else:
            actor_parameter_groups = (
                {
                    "params": core_parameters,
                    # Shared trunk plus action heads: these parameters can
                    # move the policy and must follow KL backtracking.
                    "kl_sensitive": True,
                    "lr": train_config.core_lr,
                },
                {
                    "params": auxiliary_parameters,
                    # Detached/readout-only parameters cannot change action
                    # logits and can retain their independent step size.
                    "kl_sensitive": False,
                    "lr": train_config.auxiliary_lr,
                },
            )
        self.optimizer = torch.optim.Adam(
            actor_parameter_groups,
            lr=train_config.learning_rate,
            betas=train_config.adam_betas,
            fused=self.device.type == "mps" and train_config.policy_objective == "ppo",
        )
        # Distillation freezes a teacher that supplies every target. Who
        # *acts* is a separate choice: on-policy (the default) has the student
        # visit its own states so the teacher can correct the drift the
        # student actually exhibits.
        self.teacher: SeqPlumpModel | None = None
        if train_config.policy_objective == "distill":
            self.teacher = load_distillation_teacher(
                train_config.distill_teacher_checkpoint,
                device=self.device,
                student_config=self.model.config,
            )
        teacher_acts = (
            self.teacher is not None
            and train_config.distill_rollout_policy == "teacher"
        )
        acting_model = self.teacher if teacher_acts else self.model
        acting_pool = [self.teacher] if teacher_acts else self.models
        self.collector = SeqRolloutCollector(
            acting_model,
            train_config,
            device=self.device,
            trainable_models=acting_pool,
        )
        self.critic: SeqPPOCritic | SeqPPOOracleCritic | None = None
        self.critic_optimizer: torch.optim.Optimizer | None = None
        self.log_entropy_alpha: dict[str, torch.nn.Parameter] = {}
        self.entropy_optimizer: torch.optim.Optimizer | None = None
        if train_config.policy_objective == "ppo":
            if train_config.ppo_critic_mode == "oracle":
                self.critic = SeqPPOOracleCritic(
                    self.model.config,
                    initialize_from=self.model,
                ).to(self.device)
            else:
                self.critic = SeqPPOCritic(
                    self.model.config,
                    privileged=train_config.ppo_critic_mode == "privileged",
                    initialize_from=self.model,
                ).to(self.device)
            critic_parameters = [
                parameter
                for parameter in self.critic.parameters()
                if parameter.requires_grad
            ]
            self.critic_optimizer = torch.optim.Adam(
                critic_parameters,
                lr=train_config.ppo_critic_learning_rate,
                betas=train_config.adam_betas,
                fused=self.device.type == "mps",
            )
            initial_alpha = max(train_config.ppo_entropy_coef, 1e-8)
            self.log_entropy_alpha = {
                phase: torch.nn.Parameter(
                    torch.tensor(math.log(initial_alpha), device=self.device)
                )
                for phase in ("bid", "play")
            }
            if train_config.ppo_entropy_mode == "adaptive":
                self.entropy_optimizer = torch.optim.Adam(
                    list(self.log_entropy_alpha.values()),
                    lr=train_config.ppo_entropy_learning_rate,
                )
        # Fused Metal trunk: a runtime switch (PLUMP_METAL_TRUNK=1), not a
        # training-config key, because it changes how the trunk is computed
        # rather than what is trained -- results match the PyTorch path within
        # bf16 rounding -- and a config key would make every existing
        # checkpoint refuse to resume.
        self.metal_trunk_enabled = (
            metal_trunk_requested()
            and self.device.type == "mps"
            and train_config.precision == "bf16"
            and metal_trunk_unsupported_reason(self.model.config) is None
        )
        if self.metal_trunk_enabled:
            for trunk_model in (*self.models, getattr(self.critic, "backbone", None)):
                if trunk_model is not None:
                    trunk_model.enable_metal_trunk()
        self.league = SeqLeague(
            train_config.league_max_snapshots,
            train_config.league_min_iteration,
            train_config.league_recent_fraction,
        )
        self.iteration = 0
        self.opponent_phase = train_config.rollout.initial_opponent
        self.heuristic_eval_win_streak = 0
        # -1 lets a fresh run consume iteration zero as a best-model baseline
        # without counting it toward the heuristic curriculum gate.
        self.last_heuristic_eval_iteration = -1
        # Optimizer steps actually kept (rolled-back steps do not count), for
        # the LR warmup ramp. Persisted in checkpoints.
        self.optimizer_steps = 0
        self.rng = random.Random(train_config.seed)
        # Set by the run-system wrapper. Kept outside SeqTrainingConfig so
        # reporting/evaluation cadence cannot affect the training hot path.
        self.resolved_config: dict | None = None

    # -------------------------------------------------------------- #
    # Collection                                                      #
    # -------------------------------------------------------------- #

    def _clip_actor(self) -> tuple[float, float, float]:
        """Clip every actor parameter against one global-norm budget.

        Returns ``(actor_norm, embedding_norm, auxiliary_norm)``. Only the
        first is a clip: the trunk, the action heads, the input embeddings and
        the auxiliary heads share a single budget and are rescaled by the same
        factor, so no part of the actor can be throttled independently of the
        rest. The other two are pre-clip diagnostics, measured before the
        rescale because ``clip_grad_norm_`` rewrites ``.grad`` in place.
        """

        embedding = _grad_norm(self._embedding_parameters)
        auxiliary = _grad_norm(self._auxiliary_parameters)
        actor = float(
            torch.nn.utils.clip_grad_norm_(
                self._actor_parameters, self.train.max_grad_norm
            )
        )
        return actor, embedding, auxiliary

    def split_for_updates(self, trees: list[SeqTree]) -> list[list[SeqTree]]:
        """Split one wave's trees into the slices each update consumes.

        Round-robin rather than contiguous: trees arrive grouped by schedule
        cell, so a contiguous split would hand one update all the 3-card deals
        and another all the 10-card ones. Every slice needs the same shape mix
        the schedule describes, because ``build_training_groups`` renormalizes
        weights within whatever it is given.
        """

        count = max(int(self.train.updates_per_collect), 1)
        if count == 1:
            return [trees]
        slices: list[list[SeqTree]] = [[] for _ in range(count)]
        for index, tree in enumerate(trees):
            slices[index % count].append(tree)
        return [slice_ for slice_ in slices if slice_]

    def collect(self) -> tuple[list[SeqTree], SeqRolloutSummary]:
        for model in (
            [self.teacher]
            if self.teacher is not None
            and self.train.distill_rollout_policy == "teacher"
            else self.models
        ):
            model.eval()
        trees = self.collector.collect(
            self.league,
            self.rng,
            iteration=self.iteration,
            opponent_phase=self.opponent_phase,
        )
        return trees, summarize_trees(trees, self.collector.stats)

    def record_heuristic_evaluation(
        self,
        reward: float,
        *,
        threshold: float,
        consecutive: int,
    ) -> bool:
        """Advance the persisted heuristic-to-history curriculum gate.

        Returns True exactly once, when this evaluation switches subsequent
        anchor rollouts to historical league opponents.
        """

        if (
            self.train.rollout.opponent_mode != "heuristic_then_historical"
            or self.opponent_phase != "heuristic"
        ):
            return False
        self.heuristic_eval_win_streak = (
            self.heuristic_eval_win_streak + 1 if reward > threshold else 0
        )
        if self.heuristic_eval_win_streak < consecutive:
            return False
        self.opponent_phase = "historical"
        return True

    # -------------------------------------------------------------- #
    # Update                                                          #
    # -------------------------------------------------------------- #

    def update(self, trees: list[SeqTree]) -> SeqUpdateStats:
        if self.train.policy_objective == "distill":
            return self._update_distill(trees)
        if self.train.policy_objective == "ppo":
            return self._update_ppo(trees)
        started = time.perf_counter()
        # The trees are plain Python objects from here on; the rollout's KV
        # pools are the biggest allocation in the process and are pure dead
        # weight during the backward pass.
        self.collector.release_caches()
        self._release_device_memory()
        groups = build_training_groups(trees, self.model.config, self.train)
        stats = SeqUpdateStats()
        stats.build_sec = time.perf_counter() - started
        stats.policy_rows = sum(
            len(rows.weight) for g in groups for rows in g.policy.values()
        )
        stats.branched_rows = sum(
            sum(rows.branched) for g in groups for rows in g.policy.values()
        )
        stats.unbranched_rows = stats.policy_rows - stats.branched_rows
        stats.value_rows = int(
            sum(np.count_nonzero(group.value_weight) for group in groups)
        )
        stats.positions = int(sum(g.owned.sum() for g in groups))

        for _ in range(self.train.epochs):
            snapshot = (
                copy.deepcopy(self.model.state_dict()),
                copy.deepcopy(self.optimizer.state_dict()),
            )
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            self._epoch_backward(groups, stats)
            (
                stats.core_grad_norm,
                stats.embedding_grad_norm,
                stats.auxiliary_grad_norm,
            ) = self._clip_actor()
            self._apply_warmup_lr()
            nominal_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
            accepted = False
            attempts = self.train.kl_backtrack_attempts + 1
            for attempt in range(attempts):
                if attempt:
                    self.model.load_state_dict(snapshot[0])
                    self.model.clear_card_output_cache()
                    self.optimizer.load_state_dict(snapshot[1])
                scale = self.train.kl_backtrack_factor**attempt
                for group, nominal_lr in zip(self.optimizer.param_groups, nominal_lrs):
                    group["lr"] = (
                        nominal_lr * scale
                        if group.get("kl_sensitive", True)
                        else nominal_lr
                    )
                self.optimizer.step()
                policy_kl, kl_p95, kl_p99, kl_max = self._evaluate_policy_kl(
                    groups,
                    # The nominal proposal is always measured in full for
                    # durable diagnostics. A rejected intermediate retry is
                    # immediately restored, so those passes may stop once
                    # enough above-cap weight proves weighted p99 must fail.
                    # The last attempt also runs in full so a rollback reports
                    # exact final diagnostics.
                    reject_p99_above=(
                        self.train.policy_kl_p99_cap
                        if (
                            self.train.policy_kl_p99_cap > 0
                            and attempt > 0
                            and attempt + 1 < attempts
                        )
                        else None
                    ),
                )
                if attempt == 0:
                    stats.proposed_policy_kl = policy_kl
                    stats.proposed_policy_kl_p95 = kl_p95
                    stats.proposed_policy_kl_p99 = kl_p99
                    stats.proposed_policy_kl_max = kl_max
                    stats.proposed_mean_exceeded = (
                        policy_kl > self.train.policy_kl_cap
                    )
                    stats.proposed_p99_exceeded = (
                        self.train.policy_kl_p99_cap > 0
                        and kl_p99 > self.train.policy_kl_p99_cap
                    )
                stats.policy_kl = policy_kl
                stats.policy_kl_p95 = kl_p95
                stats.policy_kl_p99 = kl_p99
                stats.policy_kl_max = kl_max
                if stats.policy_rows == 0 or (
                    policy_kl <= self.train.policy_kl_cap
                    and (
                        self.train.policy_kl_p99_cap == 0
                        or kl_p99 <= self.train.policy_kl_p99_cap
                    )
                ):
                    stats.backtracks += attempt
                    stats.step_scale = min(stats.step_scale, scale)
                    accepted = True
                    self.optimizer_steps += 1
                    break
            for group, nominal_lr in zip(self.optimizer.param_groups, nominal_lrs):
                group["lr"] = nominal_lr
            if not accepted:
                self.model.load_state_dict(snapshot[0])
                self.model.clear_card_output_cache()
                self.optimizer.load_state_dict(snapshot[1])
                stats.rolled_back = True
                stats.step_scale = 0.0
                break
        self._release_device_memory()
        stats.update_sec = time.perf_counter() - started
        return stats

    # -------------------------------------------------------------- #
    # Branch-free PPO                                                 #
    # -------------------------------------------------------------- #

    def _ppo_model(self, policy_id: str) -> SeqPlumpModel:
        try:
            return self.collector._trainable_model_map[policy_id]
        except KeyError as error:
            raise ValueError(f"Unknown trainable PPO policy {policy_id!r}.") from error

    def _device_allocated_bytes(self) -> int:
        if self.device.type == "mps":
            return int(torch.mps.driver_allocated_memory())
        if self.device.type == "cuda":
            return int(torch.cuda.memory_allocated(self.device))
        return 0

    def _update_ppo(self, trees: list[SeqTree]) -> SeqUpdateStats:
        started = time.perf_counter()
        if self.critic is None or self.critic_optimizer is None:
            raise RuntimeError("PPO requires an initialized independent critic.")
        self.collector.release_caches()
        batch = build_ppo_training_batch(trees, self.model.config, self.train)
        groups = batch.policy_groups
        stats = SeqUpdateStats(policy_kl_checked=self.train.ppo_trust_region)
        stats.build_sec = time.perf_counter() - started
        stats.policy_rows = sum(
            len(rows.weight) for group in groups for rows in group.policy.values()
        )
        stats.unbranched_rows = stats.policy_rows
        if (
            self.train.ppo_critic_mode == "oracle"
            and self.train.value_positions == "all"
        ):
            stats.value_rows = int(
                sum(
                    np.count_nonzero(group.position_weight)
                    for group in batch.critic_groups
                )
            )
        else:
            stats.value_rows = stats.policy_rows
        stats.positions = int(
            sum(group.tokens.shape[0] * group.tokens.shape[1] for group in groups)
        )

        advantage_mean, advantage_std = self._prepare_ppo_advantages(
            groups, batch.critic_groups, stats
        )
        stats.advantage_mean = advantage_mean
        stats.advantage_std = advantage_std
        (
            stats.entropy_target_bid,
            stats.entropy_target_play,
        ) = self.train.ppo_entropy_targets(self.iteration)
        if self.train.ppo_advantage_normalize:
            normalize_ppo_advantages(groups)
        stats.peak_update_device_bytes = max(
            stats.peak_update_device_bytes, self._device_allocated_bytes()
        )

        for actor_epoch in range(self.train.epochs):
            actor_snapshot = (
                (
                    [copy.deepcopy(model.state_dict()) for model in self.models],
                    copy.deepcopy(self.optimizer.state_dict()),
                )
                if self.train.ppo_trust_region
                else None
            )
            for model in self.models:
                model.train()
            self.optimizer.zero_grad(set_to_none=True)
            epoch = self._ppo_actor_backward(groups)
            stats.loss_policy += epoch["policy_loss"]
            stats.loss_suit += epoch["suit_loss"]
            stats.loss_rank_boundary += epoch["rank_boundary_loss"]
            stats.loss_trick += epoch["trick_loss"]
            stats.loss_bid_hit += epoch["bid_hit_loss"]
            stats.loss_next_winner += epoch["next_winner_loss"]
            stats.next_winner_accuracy = epoch["next_winner_accuracy"]
            stats.actor_value_loss += epoch["actor_value_loss"]
            stats.actor_all_player_rmse = epoch["actor_value_rmse"]
            stats.actor_value_zero_rmse = epoch["actor_value_zero_rmse"]
            stats.entropy = epoch["entropy"]
            stats.entropy_bid_normalized = epoch["bid_entropy"]
            stats.entropy_play_normalized = epoch["play_entropy"]
            stats.ppo_ratio_clip_fraction = epoch["clip_fraction"]
            if actor_epoch == 0:
                # Cached rollout may use reduced-precision K/V while the update
                # replays full sequences. This should remain numerical noise,
                # not silently turn the first PPO epoch off-policy.
                stats.ppo_behavior_replay_kl = epoch["behavior_kl"]
                for stage in PPO_BELIEF_STAGES:
                    setattr(
                        stats,
                        f"suit_accuracy_10c_{stage}",
                        epoch[f"suit_accuracy_10c_{stage}"],
                    )
                    setattr(
                        stats,
                        f"suit_loss_10c_{stage}",
                        epoch[f"suit_loss_10c_{stage}"],
                    )
                    setattr(
                        stats,
                        f"trick_accuracy_10c_{stage}",
                        epoch[f"trick_accuracy_10c_{stage}"],
                    )
                    setattr(
                        stats,
                        f"rank_boundary_balanced_accuracy_10c_{stage}",
                        epoch[
                            f"rank_boundary_balanced_accuracy_10c_{stage}"
                        ],
                    )
                    setattr(
                        stats,
                        f"rank_boundary_loss_10c_{stage}",
                        epoch[f"rank_boundary_loss_10c_{stage}"],
                    )
            stats.policy_logit_shift = epoch["logit_shift"]
            (
                stats.policy_loss_grad_norm,
                stats.auxiliary_loss_grad_norm,
                stats.auxiliary_grad_scale,
            ) = self._apply_auxiliary_grad_budget()
            (
                stats.core_grad_norm,
                stats.embedding_grad_norm,
                stats.auxiliary_grad_norm,
            ) = self._clip_actor()
            self._apply_warmup_lr()
            if not self.train.ppo_trust_region:
                # No post-step replay or rollback state is needed. The loss,
                # auxiliary budget, clipping and LR schedule above are shared
                # with the guarded path, as is the entropy controller below.
                self.optimizer.step()
                self.optimizer_steps += 1
                self._update_entropy_temperature(epoch)
                continue
            assert actor_snapshot is not None
            nominal_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
            accepted = False
            attempts = self.train.kl_backtrack_attempts + 1
            for attempt in range(attempts):
                if attempt:
                    for model, state in zip(self.models, actor_snapshot[0]):
                        model.load_state_dict(state)
                        model.clear_card_output_cache()
                    self.optimizer.load_state_dict(actor_snapshot[1])
                scale = self.train.kl_backtrack_factor**attempt
                for group, nominal_lr in zip(self.optimizer.param_groups, nominal_lrs):
                    group["lr"] = (
                        nominal_lr * scale
                        if group.get("kl_sensitive", True)
                        else nominal_lr
                    )
                self.optimizer.step()
                policy_kl, kl_p95, kl_p99, kl_max = self._evaluate_ppo_kl(groups)
                if attempt == 0:
                    stats.proposed_policy_kl = policy_kl
                    stats.proposed_policy_kl_p95 = kl_p95
                    stats.proposed_policy_kl_p99 = kl_p99
                    stats.proposed_policy_kl_max = kl_max
                    stats.proposed_mean_exceeded = policy_kl > self.train.policy_kl_cap
                    stats.proposed_p99_exceeded = (
                        self.train.policy_kl_p99_cap > 0
                        and kl_p99 > self.train.policy_kl_p99_cap
                    )
                stats.policy_kl = policy_kl
                stats.policy_kl_p95 = kl_p95
                stats.policy_kl_p99 = kl_p99
                stats.policy_kl_max = kl_max
                if policy_kl <= self.train.policy_kl_cap and (
                    self.train.policy_kl_p99_cap == 0
                    or kl_p99 <= self.train.policy_kl_p99_cap
                ):
                    stats.backtracks += attempt
                    stats.step_scale = min(stats.step_scale, scale)
                    self.optimizer_steps += 1
                    accepted = True
                    break
            for group, nominal_lr in zip(self.optimizer.param_groups, nominal_lrs):
                group["lr"] = nominal_lr
            if not accepted:
                for model, state in zip(self.models, actor_snapshot[0]):
                    model.load_state_dict(state)
                    model.clear_card_output_cache()
                self.optimizer.load_state_dict(actor_snapshot[1])
                stats.rolled_back = True
                stats.step_scale = 0.0
                break
            self._update_entropy_temperature(epoch)

        self._train_ppo_critic(groups, batch.critic_groups, stats)
        stats.entropy_alpha_bid = self._entropy_alpha("bid")
        stats.entropy_alpha_play = self._entropy_alpha("play")
        stats.peak_update_device_bytes = max(
            stats.peak_update_device_bytes, self._device_allocated_bytes()
        )
        self._release_device_memory()
        stats.update_sec = time.perf_counter() - started
        return stats

    def _prepare_ppo_advantages(
        self,
        groups: list[PPOTrainingGroup],
        critic_groups: list[PPOCriticGroup],
        stats: SeqUpdateStats,
    ) -> tuple[float, float]:
        assert self.critic is not None
        if self.train.ppo_critic_mode == "oracle":
            return self._prepare_oracle_ppo_advantages(
                groups, critic_groups, stats
            )
        self.critic.eval()
        predictions: list[float] = []
        targets: list[float] = []
        weights: list[float] = []
        with torch.inference_mode():
            for group in groups:
                chunks = list(self._microbatches(group))
                for (start, stop), rows_by_phase in zip(
                    chunks, ppo_rows_by_chunk(group, chunks)
                ):
                    tokens = torch.from_numpy(group.tokens[start:stop]).to(self.device)
                    hands = torch.from_numpy(group.initial_hands[start:stop]).to(
                        self.device
                    )
                    with autocast_context(self.device, self.train.precision):
                        values = self.critic.forward_full(tokens, hands)
                    for phase, rows in rows_by_phase.items():
                        if not rows["weight"]:
                            continue
                        seq = torch.tensor(
                            rows["seq_index"], device=self.device, dtype=torch.long
                        )
                        pos = torch.tensor(
                            rows["position"], device=self.device, dtype=torch.long
                        )
                        selected = values[seq, pos].float().cpu().numpy()
                        phase_rows = group.policy[phase]
                        for local, original in enumerate(rows["row_index"]):
                            target = float(rows["returns"][local])
                            prediction = float(selected[local])
                            phase_rows.advantages[original] = target - prediction
                            predictions.append(prediction)
                            targets.append(target)
                            weights.append(float(rows["weight"][local]))

        return self._ppo_value_statistics(predictions, targets, weights, stats)

    def _prepare_oracle_ppo_advantages(
        self,
        groups: list[PPOTrainingGroup],
        critic_groups: list[PPOCriticGroup],
        stats: SeqUpdateStats,
    ) -> tuple[float, float]:
        """Evaluate dense oracle values and address PPO baselines by seat."""

        assert isinstance(self.critic, SeqPPOOracleCritic)
        self.critic.eval()
        prediction_by_address: dict[tuple[int, int, int, int], float] = {}
        advantage_by_address: dict[tuple[int, int, int, int], float] = {}
        all_player_predictions: list[np.ndarray] = []
        all_player_targets: list[np.ndarray] = []
        all_player_weights: list[np.ndarray] = []
        with torch.inference_mode():
            for group_index, group in enumerate(critic_groups):
                chunks = list(self._microbatches(group))
                for (start, stop), rows in zip(
                    chunks, ppo_critic_rows_by_chunk(group, chunks)
                ):
                    if not rows["weight"]:
                        continue
                    tokens = torch.from_numpy(group.tokens[start:stop]).to(
                        self.device
                    )
                    with autocast_context(self.device, self.train.precision):
                        dense_values = self.critic.forward_full(tokens)
                    sequence_player_counts = torch.from_numpy(
                        group.player_counts[start:stop].astype(
                            np.int64, copy=False
                        )
                    ).to(self.device)
                    flat_player_counts = sequence_player_counts[:, None].expand(
                        -1, dense_values.shape[1]
                    ).reshape(-1)
                    dense_values = center_zero_sum_values(
                        dense_values.float().reshape(
                            -1, self.model.config.max_players
                        ),
                        flat_player_counts,
                    ).view_as(dense_values)
                    dense_values_cpu = dense_values.cpu().numpy()
                    dense_advantages = None
                    if self.train.ppo_advantage_estimator == "gae":
                        dense_advantages = oracle_gae_advantages(
                            group.tokens[start:stop],
                            dense_values_cpu,
                            group.returns[start:stop],
                            group.player_counts[start:stop],
                            self.train.ppo_gae_lambda,
                        )

                    # PPO consumes only these acting-seat decision addresses,
                    # but the oracle quality panel reflects the dense objective
                    # over every genuine state and every active player.
                    seq_cpu = np.asarray(rows["seq_index"], dtype=np.int64)
                    pos_cpu = np.asarray(rows["position"], dtype=np.int64)
                    selected_all = dense_values_cpu[seq_cpu, pos_cpu]
                    acting_seat = np.asarray(rows["acting_seat"], dtype=np.int64)
                    selected = selected_all[
                        np.arange(selected_all.shape[0]), acting_seat
                    ]
                    selected_advantages = (
                        None
                        if dense_advantages is None
                        else dense_advantages[seq_cpu, pos_cpu, acting_seat]
                    )

                    dense_seq, dense_pos = np.nonzero(
                        group.position_weight[start:stop] > 0
                    )
                    dense_prediction = dense_values_cpu[dense_seq, dense_pos]
                    dense_target = group.returns[start:stop][dense_seq]
                    dense_players = group.player_counts[start:stop][
                        dense_seq
                    ].astype(np.int64, copy=False)
                    dense_active = (
                        np.arange(self.model.config.max_players)[None, :]
                        < dense_players[:, None]
                    )
                    dense_weight = group.position_weight[start:stop][
                        dense_seq, dense_pos
                    ].astype(np.float64, copy=False)
                    all_player_predictions.append(
                        dense_prediction[dense_active]
                    )
                    all_player_targets.append(dense_target[dense_active])
                    all_player_weights.append(
                        np.repeat(dense_weight / dense_players, dense_players)
                    )
                    for local, prediction in enumerate(selected):
                        address = (
                            group_index,
                            rows["seq_index"][local] + start,
                            rows["position"][local],
                            rows["acting_seat"][local],
                        )
                        if address in prediction_by_address:
                            raise AssertionError(
                                "Oracle critic address appeared more than once."
                            )
                        prediction_by_address[address] = float(prediction)
                        if selected_advantages is not None:
                            advantage_by_address[address] = float(
                                selected_advantages[local]
                            )

        if all_player_predictions:
            prediction = np.concatenate(all_player_predictions).astype(
                np.float64, copy=False
            )
            target = np.concatenate(all_player_targets).astype(
                np.float64, copy=False
            )
            weight = np.concatenate(all_player_weights)
            total = float(weight.sum())
            error = prediction - target
            stats.critic_all_player_rmse = float(
                np.sqrt(np.dot(weight, error * error) / max(total, 1e-12))
            )
            prediction_mean = float(np.dot(weight, prediction) / total)
            target_mean = float(np.dot(weight, target) / total)
            prediction_centered = prediction - prediction_mean
            target_centered = target - target_mean
            prediction_var = float(
                np.dot(weight, prediction_centered**2) / total
            )
            target_var = float(np.dot(weight, target_centered**2) / total)
            covariance = float(
                np.dot(weight, prediction_centered * target_centered) / total
            )
            denominator = math.sqrt(max(prediction_var * target_var, 0.0))
            stats.critic_all_player_correlation = (
                covariance / denominator if denominator else 0.0
            )

        predictions: list[float] = []
        targets: list[float] = []
        advantages: list[float] = []
        weights: list[float] = []
        for group in groups:
            for rows in group.policy.values():
                for index, target in enumerate(rows.returns):
                    address = (
                        rows.critic_group[index],
                        rows.critic_seq_index[index],
                        rows.critic_position[index],
                        rows.critic_seat[index],
                    )
                    try:
                        prediction = prediction_by_address[address]
                    except KeyError as error:
                        raise AssertionError(
                            "Actor row has no oracle value prediction."
                        ) from error
                    advantage = (
                        advantage_by_address[address]
                        if self.train.ppo_advantage_estimator == "gae"
                        else float(target) - prediction
                    )
                    rows.advantages[index] = advantage
                    predictions.append(prediction)
                    targets.append(float(target))
                    advantages.append(advantage)
                    weights.append(float(rows.weight[index]))

        monte_carlo_statistics = self._ppo_value_statistics(
            predictions, targets, weights, stats
        )
        if self.train.ppo_advantage_estimator == "monte_carlo":
            return monte_carlo_statistics
        return self._weighted_mean_std(advantages, weights)

    @staticmethod
    def _weighted_mean_std(
        values: list[float], weights: list[float]
    ) -> tuple[float, float]:
        value = np.asarray(values, dtype=np.float64)
        weight = np.asarray(weights, dtype=np.float64)
        total = float(weight.sum())
        if total <= 0:
            return 0.0, 1.0
        mean = float(np.dot(weight, value) / total)
        variance = float(np.dot(weight, (value - mean) ** 2) / total)
        return mean, math.sqrt(max(variance, 1e-12))

    @staticmethod
    def _ppo_value_statistics(
        predictions: list[float],
        targets: list[float],
        weights: list[float],
        stats: SeqUpdateStats,
    ) -> tuple[float, float]:
        """Report baseline quality on the acting-player rows PPO consumes."""

        prediction = np.asarray(predictions, dtype=np.float64)
        target = np.asarray(targets, dtype=np.float64)
        weight = np.asarray(weights, dtype=np.float64)
        total = float(weight.sum())
        if total <= 0:
            return 0.0, 1.0
        error = prediction - target
        stats.value_rmse = float(np.sqrt(np.dot(weight, error * error) / total))
        stats.value_zero_rmse = float(np.sqrt(np.dot(weight, target * target) / total))
        prediction_mean = float(np.dot(weight, prediction) / total)
        target_mean = float(np.dot(weight, target) / total)
        prediction_centered = prediction - prediction_mean
        target_centered = target - target_mean
        prediction_var = float(np.dot(weight, prediction_centered**2) / total)
        target_var = float(np.dot(weight, target_centered**2) / total)
        covariance = float(
            np.dot(weight, prediction_centered * target_centered) / total
        )
        stats.value_prediction_std = math.sqrt(max(prediction_var, 0.0))
        denominator = math.sqrt(max(prediction_var * target_var, 0.0))
        stats.value_correlation = covariance / denominator if denominator else 0.0
        return SeqTrainer._weighted_mean_std(
            (target - prediction).tolist(), weight.tolist()
        )

    def _entropy_alpha(self, phase: str) -> float:
        if self.train.ppo_entropy_mode == "off":
            return 0.0
        if self.train.ppo_entropy_mode == "fixed":
            return self.train.ppo_entropy_coef
        return float(self.log_entropy_alpha[phase].detach().exp().clamp(max=10.0))

    def _auxiliary_budget_active(self) -> bool:
        return self.train.ppo_auxiliary_grad_fraction > 0

    def _dual_trunk_backward(self, model: SeqPlumpModel, tokens: torch.Tensor) -> bool:
        """Whether this microbatch's two objectives share one fused backward.

        Only worth it when the budget keeps the halves separate (otherwise
        they are summed and differentiated once anyway) and both reach the
        trunk (a detached auxiliary half never does).
        """

        return (
            self._auxiliary_budget_active()
            and not self.train.ppo_actor_auxiliary_detach_hidden
            and model.uses_metal_trunk(tokens)
        )

    def _backward_actor_losses(
        self,
        auxiliary_loss: torch.Tensor,
        policy_loss: torch.Tensor,
        dual: DualTrunkPass | None = None,
    ) -> None:
        """Backward one microbatch, keeping the two objectives separable.

        Without a budget the halves are summed and differentiated once. With
        one, request each gradient directly and accumulate it in its own
        buffers. Through autograd the shared trunk receives two backwards;
        a ``DualTrunkPass`` runs one fused trunk backward for both halves and
        hands back the same two gradient lists.
        """

        if dual is not None:
            auxiliary, policy = dual.gradients(
                [auxiliary_loss, policy_loss], self._actor_parameters
            )
            if auxiliary is not None:
                self._auxiliary_grads = self._accumulate_actor_grads(
                    self._auxiliary_grads, auxiliary
                )
            if policy is not None:
                self._policy_grads = self._accumulate_actor_grads(
                    self._policy_grads, policy
                )
            return
        if not self._auxiliary_budget_active():
            total = auxiliary_loss + policy_loss
            if total.requires_grad:
                total.backward()
            return
        if auxiliary_loss.requires_grad:
            gradients = torch.autograd.grad(
                auxiliary_loss,
                self._actor_parameters,
                retain_graph=policy_loss.requires_grad,
                allow_unused=True,
            )
            self._auxiliary_grads = self._accumulate_actor_grads(
                self._auxiliary_grads, gradients
            )
        if policy_loss.requires_grad:
            gradients = torch.autograd.grad(
                policy_loss, self._actor_parameters, allow_unused=True
            )
            self._policy_grads = self._accumulate_actor_grads(
                self._policy_grads, gradients
            )

    @staticmethod
    def _accumulate_actor_grads(
        stash: list[torch.Tensor | None] | None,
        gradients: Sequence[torch.Tensor | None],
    ) -> list[torch.Tensor | None]:
        """Own newly returned gradients and batch subsequent additions."""

        if stash is None:
            return [None if g is None else g.detach() for g in gradients]
        destinations = []
        additions = []
        for index, gradient in enumerate(gradients):
            if gradient is None:
                continue
            if stash[index] is None:
                stash[index] = gradient.detach()
            else:
                destinations.append(stash[index])
                additions.append(gradient)
        if destinations:
            torch._foreach_add_(destinations, additions)
        return stash

    @staticmethod
    def _stash_norm(stash: list[torch.Tensor | None] | None) -> float:
        present = (
            [] if stash is None else [g for g in stash if g is not None]
        )
        if not present:
            return 0.0
        return float(
            torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(g) for g in present])
            )
        )

    def _apply_auxiliary_grad_budget(self) -> tuple[float, float, float]:
        """Rescale the auxiliary gradient to its share, then fold it back in.

        Returns ``(policy_norm, auxiliary_norm, scale)``, both norms measured
        over every actor parameter before the rebalance and before the clip.

        The budget is a ratio because Adam is invariant to a uniform rescale
        of the gradient: only the relative size of the two contributions
        changes the step. The absolute size stays the business of the single
        ``max_grad_norm`` clip applied to the sum afterwards.
        """

        auxiliary_stash = self._auxiliary_grads
        policy_stash = self._policy_grads
        self._auxiliary_grads = None
        self._policy_grads = None
        if not self._auxiliary_budget_active():
            return 0.0, 0.0, 0.0
        policy_norm = self._stash_norm(policy_stash)
        auxiliary_norm = self._stash_norm(auxiliary_stash)
        # A zero policy norm would otherwise scale the auxiliary half to
        # nothing. Nothing is competing with it in that case, so pass it
        # through and let the global clip bound the step.
        if auxiliary_norm <= 0:
            scale = 0.0
        elif policy_norm <= 0:
            scale = 1.0
        else:
            scale = (
                self.train.ppo_auxiliary_grad_fraction
                * policy_norm
                / auxiliary_norm
            )
        for index, parameter in enumerate(self._actor_parameters):
            policy_gradient = (
                None if policy_stash is None else policy_stash[index]
            )
            auxiliary_gradient = (
                None if auxiliary_stash is None else auxiliary_stash[index]
            )
            if auxiliary_gradient is not None and scale != 1.0:
                auxiliary_gradient.mul_(scale)
            if policy_gradient is None:
                combined = auxiliary_gradient
            elif auxiliary_gradient is None:
                combined = policy_gradient
            else:
                combined = policy_gradient.add_(auxiliary_gradient)
            parameter.grad = combined
        return policy_norm, auxiliary_norm, scale

    def _ppo_actor_backward(self, groups: list[PPOTrainingGroup]) -> dict[str, float]:
        totals: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._auxiliary_grads = None
        self._policy_grads = None
        # Temperatures only change after an accepted actor step. Read them
        # once, avoiding a device-to-host synchronization in every microbatch.
        entropy_alpha = {
            phase: self._entropy_alpha(phase) for phase in ("bid", "play")
        }
        for group in groups:
            model = self._ppo_model(group.policy_id)
            chunks = list(self._microbatches(group))
            for (start, stop), rows_by_phase in zip(
                chunks, ppo_rows_by_chunk(group, chunks)
            ):
                tokens = torch.from_numpy(group.tokens[start:stop]).to(self.device)
                dual = None
                with autocast_context(self.device, self.train.precision):
                    if self._dual_trunk_backward(model, tokens):
                        dual = DualTrunkPass(model, model.metal_trunk, tokens)
                        hidden = dual.hidden
                    else:
                        hidden = model.forward_hidden(tokens)
                auxiliary_hidden = (
                    hidden.detach()
                    if self.train.ppo_actor_auxiliary_detach_hidden
                    else hidden
                )
                # The two halves are accumulated apart so each can be given
                # its own gradient-norm budget. With the budget off they are
                # summed again before the single backward, which is bit-for-bit
                # the previous behaviour.
                loss = torch.zeros((), device=self.device, dtype=torch.float32)
                policy_objective = torch.zeros(
                    (), device=self.device, dtype=torch.float32
                )

                active_auxiliary_heads = frozenset(
                    name
                    for name, coefficient in (
                        ("suit", self.train.suit_coef),
                        ("rank_boundary", self.train.rank_boundary_coef),
                        ("trick", self.train.trick_coef),
                        ("next_winner", self.train.next_winner_coef),
                    )
                    if coefficient > 0
                )
                wants_bid_hit = self.train.bid_hit_coef > 0
                auxiliary_logits: dict[str, torch.Tensor] = {}
                bid_hit_logits: torch.Tensor | None = None
                auxiliary_seq_np = np.empty(0, dtype=np.int64)
                auxiliary_pos_np = np.empty(0, dtype=np.int64)
                stage_indices = np.full(
                    (stop - start, len(PPO_BELIEF_STAGES)), -1, dtype=np.int64
                )
                if active_auxiliary_heads or wants_bid_hit:
                    if self.train.ppo_auxiliary_positions == "all":
                        # Padding introduced by length bucketing has zero
                        # position weight. Gather every genuine causal token
                        # and retain the precomputed per-trajectory average.
                        auxiliary_seq_np, auxiliary_pos_np = np.nonzero(
                            group.position_weight[start:stop] > 0
                        )
                        auxiliary_weight_np = group.position_weight[start:stop][
                            auxiliary_seq_np, auxiliary_pos_np
                        ]
                    else:
                        # Decision-only mode remains available as an ablation.
                        auxiliary_seq_np = np.concatenate(
                            [
                                np.asarray(
                                    rows_by_phase[phase]["seq_index"],
                                    dtype=np.int64,
                                )
                                for phase in ("bid", "play")
                            ]
                        )
                        auxiliary_pos_np = np.concatenate(
                            [
                                np.asarray(
                                    rows_by_phase[phase]["position"],
                                    dtype=np.int64,
                                )
                                for phase in ("bid", "play")
                            ]
                        )
                        decision_count = np.bincount(
                            auxiliary_seq_np, minlength=stop - start
                        )
                        if np.any(decision_count == 0):
                            raise AssertionError(
                                "Every PPO actor sequence must contain a decision."
                            )
                        sequence_weight = group.position_weight[start:stop].sum(
                            axis=1
                        )
                        auxiliary_weight_np = (
                            sequence_weight[auxiliary_seq_np]
                            / decision_count[auxiliary_seq_np]
                        )
                    auxiliary_weight = torch.from_numpy(
                        auxiliary_weight_np.astype(np.float32, copy=False)
                    ).to(self.device)
                    auxiliary_seq = torch.from_numpy(auxiliary_seq_np).to(self.device)
                    auxiliary_pos = torch.from_numpy(auxiliary_pos_np).to(self.device)
                    with autocast_context(self.device, self.train.precision):
                        if self.train.ppo_auxiliary_positions == "all":
                            # Project one contiguous dense matrix, then remove
                            # the small amount of bucket padding from logits.
                            # This is faster on MPS than gathering wide hidden
                            # rows before the matrix multiplication.
                            dense_hidden = auxiliary_hidden.reshape(
                                -1, auxiliary_hidden.shape[-1]
                            )
                            dense_logits = model.auxiliary_logits(
                                dense_hidden,
                                active_auxiliary_heads,
                            )
                            flat = auxiliary_seq * hidden.shape[1] + auxiliary_pos
                            auxiliary_logits = {
                                name: logits[flat]
                                for name, logits in dense_logits.items()
                            }
                            if wants_bid_hit:
                                # Gathered, not dense: this head's first layer
                                # is d_model x d_model, so projecting the
                                # padded matrix would cost far more than the
                                # narrow packed readouts above.
                                bid_hit_logits = model.bid_hit_head(
                                    dense_hidden[flat]
                                )
                        else:
                            gathered_hidden = auxiliary_hidden[
                                auxiliary_seq, auxiliary_pos
                            ]
                            auxiliary_logits = model.auxiliary_logits(
                                gathered_hidden,
                                active_auxiliary_heads,
                            )
                            if wants_bid_hit:
                                bid_hit_logits = model.bid_hit_head(
                                    gathered_hidden
                                )

                    lookup = np.full(
                        (stop - start, hidden.shape[1]), -1, dtype=np.int64
                    )
                    lookup[auxiliary_seq_np, auxiliary_pos_np] = np.arange(
                        auxiliary_seq_np.size, dtype=np.int64
                    )
                    raw_stage_positions = group.belief_stage_positions[start:stop]
                    valid_stage = raw_stage_positions >= 0
                    stage_rows, stage_columns = np.nonzero(valid_stage)
                    stage_indices[stage_rows, stage_columns] = lookup[
                        stage_rows, raw_stage_positions[valid_stage]
                    ]
                    if np.any(stage_indices[valid_stage] < 0):
                        raise AssertionError(
                            "A PPO belief stage has no auxiliary readout."
                        )

                if self.train.suit_coef > 0:
                    suit_logits = auxiliary_logits["suit"].view(
                        auxiliary_seq_np.size,
                        model.config.belief_opponents,
                        4,
                    ).float()
                    suit_targets = torch.from_numpy(
                        group.suit_targets[start:stop][
                            auxiliary_seq_np, auxiliary_pos_np
                        ]
                    ).to(self.device)
                    suit_labeled = suit_targets != IGNORE_LABEL
                    bce = F.binary_cross_entropy_with_logits(
                        suit_logits,
                        suit_targets.clamp_min(0).to(suit_logits.dtype),
                        reduction="none",
                    )
                    per_position = (
                        (bce * suit_labeled.to(bce.dtype)).sum(dim=(1, 2))
                        / suit_labeled.sum(dim=(1, 2)).clamp_min(1).to(bce.dtype)
                    ).float()
                    suit_loss = (per_position * auxiliary_weight).sum()
                    loss = loss + self.train.suit_coef * suit_loss
                    # Keep reporting storage independent of the packed-head
                    # backward graph. On MPS, the backward may reuse the
                    # first split's scalar buffer after a model-state snapshot,
                    # which previously made the reported suit loss become 0.
                    totals["suit_loss"].append(suit_loss.detach().clone())
                    self._record_ppo_belief_stage_accuracy(
                        totals,
                        "suit",
                        suit_logits,
                        suit_targets,
                        None,
                        stage_indices,
                    )

                if self.train.rank_boundary_coef > 0:
                    rank_boundary_logits = auxiliary_logits["rank_boundary"].view(
                        auxiliary_seq_np.size,
                        model.config.belief_opponents + 1,
                        4,
                        2,
                    ).float()
                    rank_boundary_targets = torch.from_numpy(
                        group.rank_boundary_targets[start:stop][
                            auxiliary_seq_np, auxiliary_pos_np
                        ]
                    ).to(self.device)
                    labeled = rank_boundary_targets != IGNORE_LABEL
                    bce = F.binary_cross_entropy_with_logits(
                        rank_boundary_logits,
                        rank_boundary_targets.clamp_min(0).to(
                            rank_boundary_logits.dtype
                        ),
                        reduction="none",
                    )
                    per_position = (
                        (bce * labeled.to(bce.dtype)).sum(dim=(1, 2, 3))
                        / labeled.sum(dim=(1, 2, 3)).clamp_min(1).to(bce.dtype)
                    ).float()
                    rank_boundary_loss = (per_position * auxiliary_weight).sum()
                    loss = (
                        loss
                        + self.train.rank_boundary_coef * rank_boundary_loss
                    )
                    totals["rank_boundary_loss"].append(
                        rank_boundary_loss.detach().clone()
                    )
                    self._record_ppo_rank_boundary_stage_accuracy(
                        totals,
                        rank_boundary_logits,
                        rank_boundary_targets,
                        stage_indices,
                    )

                if self.train.trick_coef > 0:
                    trick_logits = auxiliary_logits["trick"].view(
                        auxiliary_seq_np.size,
                        model.config.max_players,
                        model.config.bid_count,
                    ).float()
                    trick_targets = torch.from_numpy(
                        group.trick_targets[start:stop][auxiliary_seq_np]
                    ).to(self.device)
                    trick_masks = torch.from_numpy(
                        group.trick_masks[start:stop][
                            auxiliary_seq_np, auxiliary_pos_np
                        ]
                    ).to(self.device)
                    masked_logits = trick_logits.masked_fill(~trick_masks, -1e9)
                    log_probs = torch.log_softmax(masked_logits, dim=-1)
                    safe_targets = trick_targets.clamp_min(0)[:, :, None]
                    gathered = log_probs.gather(2, safe_targets).squeeze(-1)
                    labeled = (
                        (trick_targets != IGNORE_LABEL)
                        & trick_masks.any(dim=-1)
                    )
                    per_position = (-gathered * labeled.float()).sum(dim=1) / (
                        labeled.float().sum(dim=1).clamp_min(1.0)
                    )
                    trick_loss = (per_position * auxiliary_weight).sum()
                    loss = loss + self.train.trick_coef * trick_loss
                    totals["trick_loss"].append(trick_loss.detach().clone())
                    self._record_ppo_belief_stage_accuracy(
                        totals,
                        "trick",
                        masked_logits,
                        trick_targets,
                        trick_masks,
                        stage_indices,
                    )

                if bid_hit_logits is not None:
                    # Did each seat end the round on exactly its bid. One
                    # independent sigmoid per seat, averaged over the seats
                    # this row actually has, then weighted like every other
                    # dense auxiliary.
                    bid_hit_targets = torch.from_numpy(
                        group.bid_hit_targets[start:stop][auxiliary_seq_np]
                    ).to(self.device)
                    bid_hit_labeled = (bid_hit_targets != IGNORE_LABEL).float()
                    per_seat = F.binary_cross_entropy_with_logits(
                        bid_hit_logits.float(),
                        bid_hit_targets.clamp_min(0).float(),
                        reduction="none",
                    )
                    per_position = (per_seat * bid_hit_labeled).sum(dim=1) / (
                        bid_hit_labeled.sum(dim=1).clamp_min(1.0)
                    )
                    bid_hit_loss = (per_position * auxiliary_weight).sum()
                    loss = loss + self.train.bid_hit_coef * bid_hit_loss
                    totals["bid_hit_loss"].append(bid_hit_loss.detach().clone())

                if self.train.next_winner_coef > 0:
                    # One softmax per position over this row's active seats.
                    # Length bucketing can merge table sizes into one group, so
                    # the seat mask has to come from the row's own count rather
                    # than the group's nominal num_players.
                    next_winner_logits = auxiliary_logits["next_winner"].float()
                    next_winner_targets = torch.from_numpy(
                        group.next_winner_targets[start:stop][
                            auxiliary_seq_np, auxiliary_pos_np
                        ]
                    ).to(self.device)
                    seat_counts = torch.from_numpy(
                        group.player_counts[start:stop][
                            auxiliary_seq_np
                        ].astype(np.int64, copy=False)
                    ).to(self.device)
                    seat_active = (
                        torch.arange(
                            model.config.max_players, device=self.device
                        )[None, :]
                        < seat_counts[:, None]
                    )
                    masked_next_winner = next_winner_logits.masked_fill(
                        ~seat_active, -1e9
                    )
                    log_probs = torch.log_softmax(masked_next_winner, dim=-1)
                    labeled = next_winner_targets != IGNORE_LABEL
                    gathered = log_probs.gather(
                        1, next_winner_targets.clamp_min(0)[:, None]
                    ).squeeze(-1)
                    # Positions outside the play phase carry IGNORE_LABEL and
                    # drop out here, so the head is trained on exactly the
                    # window where "the next trick" refers to something.
                    per_position = -gathered * labeled.float()
                    next_winner_loss = (per_position * auxiliary_weight).sum()
                    loss = (
                        loss + self.train.next_winner_coef * next_winner_loss
                    )
                    totals["next_winner_loss"].append(
                        next_winner_loss.detach().clone()
                    )
                    totals["next_winner_correct"].append(
                        (
                            (masked_next_winner.argmax(dim=-1)
                             == next_winner_targets)
                            & labeled
                        )
                        .sum()
                        .detach()
                    )
                    totals["next_winner_total"].append(labeled.sum().detach())

                if self.train.ppo_actor_value_coef > 0:
                    # Always dense over every genuine causal token, regardless
                    # of ppo_auxiliary_positions: the oracle's value objective
                    # is dense, and matching it exactly is what makes the two
                    # losses comparable. When the belief heads are already
                    # dense these are the same addresses, so reuse them.
                    if (
                        active_auxiliary_heads
                        and self.train.ppo_auxiliary_positions == "all"
                    ):
                        value_seq_np = auxiliary_seq_np
                        value_pos_np = auxiliary_pos_np
                        assert auxiliary_weight is not None
                        value_weight = auxiliary_weight
                    else:
                        value_seq_np, value_pos_np = np.nonzero(
                            group.position_weight[start:stop] > 0
                        )
                        value_weight = torch.from_numpy(
                            group.position_weight[start:stop][
                                value_seq_np, value_pos_np
                            ]
                        ).to(self.device)
                    value_seq = torch.from_numpy(value_seq_np).to(self.device)
                    value_pos = torch.from_numpy(value_pos_np).to(self.device)
                    with autocast_context(self.device, self.train.precision):
                        # One contiguous projection then gather, exactly like
                        # the packed belief heads: faster on MPS than gathering
                        # wide hidden rows before the matrix multiplication.
                        dense_values = model.player_value_head(
                            auxiliary_hidden.reshape(
                                -1, auxiliary_hidden.shape[-1]
                            )
                        )
                    predictions = dense_values[
                        value_seq * hidden.shape[1] + value_pos
                    ].float()
                    value_players = torch.from_numpy(
                        group.player_counts[start:stop][value_seq_np].astype(
                            np.int64, copy=False
                        )
                    ).to(self.device)
                    value_targets = torch.from_numpy(
                        group.value_targets[start:stop][value_seq_np]
                    ).to(self.device)
                    value_active = (
                        torch.arange(
                            model.config.max_players, device=self.device
                        )[None, :]
                        < value_players[:, None]
                    ).float()
                    predictions = center_zero_sum_values(
                        predictions, value_players
                    )
                    raw_error = predictions - value_targets
                    error = raw_error / self.train.value_reward_scale
                    # Identical to the oracle's normalization: each game keeps
                    # unit objective mass, spread over its causal positions and
                    # averaged across the seats its table actually has.
                    actor_value_loss = (
                        0.5
                        * value_weight
                        * (error.square() * value_active).sum(dim=-1)
                        / value_players.float()
                    ).sum()
                    loss = loss + (
                        self.train.ppo_actor_value_coef * actor_value_loss
                    )
                    totals["actor_value_loss"].append(
                        actor_value_loss.detach().clone()
                    )
                    # Reward-unit RMSE accumulators, weighted per active seat
                    # exactly as critic_all_player_rmse is, so the two curves
                    # are the same measurement on the same rows.
                    per_seat_weight = value_weight / value_players.float()
                    totals["actor_value_squared_error"].append(
                        (
                            per_seat_weight
                            * (raw_error.square() * value_active).sum(dim=-1)
                        ).sum().detach()
                    )
                    totals["actor_value_target_squared"].append(
                        (
                            per_seat_weight
                            * (value_targets.square() * value_active).sum(dim=-1)
                        ).sum().detach()
                    )
                    totals["actor_value_weight"].append(
                        value_weight.sum().detach()
                    )

                for phase, rows in rows_by_phase.items():
                    if not rows["weight"]:
                        continue
                    seq = torch.tensor(
                        rows["seq_index"], device=self.device, dtype=torch.long
                    )
                    pos = torch.tensor(
                        rows["position"], device=self.device, dtype=torch.long
                    )
                    with autocast_context(self.device, self.train.precision):
                        logits = model.policy_logits(hidden[seq, pos], phase)
                    logits = logits.float()
                    old_probs = torch.from_numpy(
                        np.stack(rows["old_probs_full"])
                    ).to(self.device)
                    actions = torch.tensor(
                        rows["action"], device=self.device, dtype=torch.long
                    )
                    advantages = torch.tensor(
                        rows["advantages"], device=self.device, dtype=torch.float32
                    )
                    weight = torch.tensor(
                        rows["weight"], device=self.device, dtype=torch.float32
                    )
                    terms = ppo_clipped_terms(
                        logits,
                        old_probs,
                        actions,
                        advantages,
                        clip_ratio=self.train.ppo_clip_ratio,
                    )
                    policy_loss = (weight * terms.losses).sum()
                    policy_objective = (
                        policy_objective + self.train.policy_coef * policy_loss
                    )
                    eligible_weight = weight * terms.entropy_eligible.float()
                    entropy_term = (
                        eligible_weight * terms.normalized_entropy
                    ).sum()
                    alpha = entropy_alpha[phase]
                    if alpha:
                        policy_objective = policy_objective - alpha * entropy_term
                    totals["policy_loss"].append(policy_loss.detach())
                    totals["entropy_raw"].append((weight * terms.entropy).sum().detach())
                    totals["row_weight"].append(weight.sum().detach())
                    totals[f"{phase}_entropy"].append(entropy_term.detach())
                    totals[f"{phase}_entropy_weight"].append(
                        eligible_weight.sum().detach()
                    )
                    totals["clip"].append(
                        (weight * terms.clipped.float()).sum().detach()
                    )
                    totals["behavior_kl"].append(
                        (weight * terms.divergences).sum().detach()
                    )
                    legal = old_probs > 0
                    logit_shift = (
                        (logits * legal.float()).sum(dim=-1)
                        / legal.sum(dim=-1).clamp_min(1)
                    )
                    totals["logit_shift"].append(
                        (weight * logit_shift).sum().detach()
                    )
                self._backward_actor_losses(loss, policy_objective, dual)

        reduced = {
            key: math.fsum(float(value) for value in values)
            for key, values in totals.items()
        }
        row_weight = max(reduced.get("row_weight", 0.0), 1e-12)
        value_weight_total = reduced.get("actor_value_weight", 0.0)
        result = {
            "policy_loss": reduced.get("policy_loss", 0.0),
            "suit_loss": reduced.get("suit_loss", 0.0),
            "rank_boundary_loss": reduced.get("rank_boundary_loss", 0.0),
            "trick_loss": reduced.get("trick_loss", 0.0),
            "bid_hit_loss": reduced.get("bid_hit_loss", 0.0),
            "next_winner_loss": reduced.get("next_winner_loss", 0.0),
            "next_winner_accuracy": (
                reduced.get("next_winner_correct", 0.0)
                / reduced["next_winner_total"]
                if reduced.get("next_winner_total", 0.0) > 0
                else 0.0
            ),
            "actor_value_loss": reduced.get("actor_value_loss", 0.0),
            "actor_value_rmse": (
                math.sqrt(
                    max(reduced.get("actor_value_squared_error", 0.0), 0.0)
                    / value_weight_total
                )
                if value_weight_total > 0
                else 0.0
            ),
            "actor_value_zero_rmse": (
                math.sqrt(
                    max(reduced.get("actor_value_target_squared", 0.0), 0.0)
                    / value_weight_total
                )
                if value_weight_total > 0
                else 0.0
            ),
            "entropy": reduced.get("entropy_raw", 0.0) / row_weight,
            "clip_fraction": reduced.get("clip", 0.0) / row_weight,
            "behavior_kl": reduced.get("behavior_kl", 0.0) / row_weight,
            "logit_shift": reduced.get("logit_shift", 0.0) / row_weight,
        }
        for phase in ("bid", "play"):
            entropy_weight = reduced.get(f"{phase}_entropy_weight", 0.0)
            result[f"{phase}_entropy"] = (
                reduced.get(f"{phase}_entropy", 0.0) / entropy_weight
                if entropy_weight > 0
                else 0.0
            )
        for head in ("suit", "trick"):
            for stage in PPO_BELIEF_STAGES:
                total = reduced.get(f"{head}_stage_{stage}_total", 0.0)
                result[f"{head}_accuracy_10c_{stage}"] = (
                    reduced.get(f"{head}_stage_{stage}_correct", 0.0) / total
                    if total > 0
                    else 0.0
                )
        for stage in PPO_BELIEF_STAGES:
            suit_total = reduced.get(f"suit_stage_{stage}_total", 0.0)
            result[f"suit_loss_10c_{stage}"] = (
                reduced.get(f"suit_stage_{stage}_loss", 0.0) / suit_total
                if suit_total > 0
                else 0.0
            )
        for stage in PPO_BELIEF_STAGES:
            positive = reduced.get(
                f"rank_boundary_stage_{stage}_positive", 0.0
            )
            negative = reduced.get(
                f"rank_boundary_stage_{stage}_negative", 0.0
            )
            recalls = []
            if positive > 0:
                recalls.append(
                    reduced.get(f"rank_boundary_stage_{stage}_tp", 0.0)
                    / positive
                )
            if negative > 0:
                recalls.append(
                    reduced.get(f"rank_boundary_stage_{stage}_tn", 0.0)
                    / negative
                )
            result[f"rank_boundary_balanced_accuracy_10c_{stage}"] = (
                math.fsum(recalls) / len(recalls) if recalls else 0.0
            )
            rank_total = positive + negative
            result[f"rank_boundary_loss_10c_{stage}"] = (
                reduced.get(f"rank_boundary_stage_{stage}_loss", 0.0)
                / rank_total
                if rank_total > 0
                else 0.0
            )
        return result

    @staticmethod
    def _record_ppo_rank_boundary_stage_accuracy(
        totals: dict[str, list[torch.Tensor]],
        logits: torch.Tensor,
        targets: torch.Tensor,
        stage_indices: np.ndarray,
    ) -> None:
        """Accumulate class-balanced bit accuracy at fixed ten-card stages."""

        device = logits.device
        for stage_index, cards_played in enumerate(PPO_BELIEF_STAGES):
            selected_np = stage_indices[:, stage_index]
            selected_np = selected_np[selected_np >= 0]
            if selected_np.size == 0:
                continue
            selected = torch.from_numpy(selected_np).to(
                device=device, dtype=torch.long
            )
            selected_targets = targets[selected]
            selected_logits = logits[selected]
            selected_predictions = selected_logits >= 0
            labeled = selected_targets != IGNORE_LABEL
            positive = labeled & (selected_targets == 1)
            negative = labeled & (selected_targets == 0)
            prefix = f"rank_boundary_stage_{cards_played}"
            totals[f"{prefix}_tp"].append(
                (selected_predictions & positive).sum().detach()
            )
            totals[f"{prefix}_positive"].append(positive.sum().detach())
            totals[f"{prefix}_tn"].append(
                ((~selected_predictions) & negative).sum().detach()
            )
            totals[f"{prefix}_negative"].append(negative.sum().detach())
            bce = F.binary_cross_entropy_with_logits(
                selected_logits,
                selected_targets.clamp_min(0).float(),
                reduction="none",
            )
            totals[f"{prefix}_loss"].append(
                (bce * labeled.float()).sum().detach()
            )

    @staticmethod
    def _record_ppo_belief_stage_accuracy(
        totals: dict[str, list[torch.Tensor]],
        head: str,
        logits: torch.Tensor,
        targets: torch.Tensor,
        feasibility: torch.Tensor | None,
        stage_indices: np.ndarray,
    ) -> None:
        """Accumulate micro-accuracy before 1st/5th/9th ten-card plays."""

        device = logits.device
        for stage_index, cards_played in enumerate(PPO_BELIEF_STAGES):
            selected_np = stage_indices[:, stage_index]
            selected_np = selected_np[selected_np >= 0]
            if selected_np.size == 0:
                continue
            selected = torch.from_numpy(selected_np).to(
                device=device, dtype=torch.long
            )
            selected_logits = logits[selected]
            if head == "suit":
                selected_targets = targets[selected]
                labeled = selected_targets != IGNORE_LABEL
                predictions = selected_logits >= 0
                bce = F.binary_cross_entropy_with_logits(
                    selected_logits,
                    selected_targets.clamp_min(0).float(),
                    reduction="none",
                )
                totals[f"{head}_stage_{cards_played}_loss"].append(
                    (bce * labeled.float()).sum().detach()
                )
            else:
                selected_targets = targets[selected]
                assert feasibility is not None
                labeled = (
                    (selected_targets != IGNORE_LABEL)
                    & feasibility[selected].any(dim=-1)
                )
                predictions = selected_logits.argmax(dim=-1)
            totals[f"{head}_stage_{cards_played}_correct"].append(
                ((predictions == selected_targets) & labeled).sum().detach()
            )
            totals[f"{head}_stage_{cards_played}_total"].append(
                labeled.sum().detach()
            )

    def _update_entropy_temperature(self, epoch: dict[str, float]) -> None:
        if self.entropy_optimizer is None:
            return
        self.entropy_optimizer.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=self.device)
        bid_target, play_target = self.train.ppo_entropy_targets(self.iteration)
        targets = {"bid": bid_target, "play": play_target}
        for phase, target in targets.items():
            observed = torch.tensor(epoch[f"{phase}_entropy"], device=self.device)
            loss = loss + self.log_entropy_alpha[phase] * (observed - target)
        loss.backward()
        self.entropy_optimizer.step()
        with torch.no_grad():
            for parameter in self.log_entropy_alpha.values():
                parameter.clamp_(math.log(1e-6), math.log(10.0))

    def _evaluate_ppo_kl(
        self, groups: list[PPOTrainingGroup]
    ) -> tuple[float, float, float, float]:
        divergences: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        for model in self.models:
            model.eval()
        with torch.inference_mode():
            for group in groups:
                model = self._ppo_model(group.policy_id)
                chunks = list(self._microbatches(group))
                for (start, stop), rows_by_phase in zip(
                    chunks, ppo_rows_by_chunk(group, chunks)
                ):
                    tokens = torch.from_numpy(group.tokens[start:stop]).to(self.device)
                    with autocast_context(self.device, self.train.precision):
                        hidden = model.forward_hidden(tokens)
                    for phase, rows in rows_by_phase.items():
                        if not rows["weight"]:
                            continue
                        seq = torch.tensor(
                            rows["seq_index"], device=self.device, dtype=torch.long
                        )
                        pos = torch.tensor(
                            rows["position"], device=self.device, dtype=torch.long
                        )
                        old_probs = torch.from_numpy(
                            np.stack(rows["old_probs_full"])
                        ).to(self.device)
                        with autocast_context(self.device, self.train.precision):
                            logits = model.policy_logits(
                                hidden[seq, pos], phase
                            )
                        divergences.append(
                            ppo_policy_kl(logits, old_probs).cpu().numpy()
                        )
                        weights.append(np.asarray(rows["weight"], dtype=np.float32))
        if not divergences:
            return 0.0, 0.0, 0.0, 0.0
        return _weighted_kl_summary(
            np.concatenate(divergences), np.concatenate(weights)
        )

    def _train_ppo_critic(
        self,
        groups: list[PPOTrainingGroup],
        critic_groups: list[PPOCriticGroup],
        stats: SeqUpdateStats,
    ) -> None:
        assert self.critic is not None and self.critic_optimizer is not None
        if self.train.ppo_critic_mode == "oracle":
            self._train_oracle_ppo_critic(critic_groups, stats)
            return
        total_loss = 0.0
        epoch_losses: list[float] = []
        for _ in range(self.train.ppo_critic_epochs):
            self.critic.train()
            self.critic_optimizer.zero_grad(set_to_none=True)
            pending: list[torch.Tensor] = []
            for group in groups:
                chunks = list(self._microbatches(group))
                for (start, stop), rows_by_phase in zip(
                    chunks, ppo_rows_by_chunk(group, chunks)
                ):
                    tokens = torch.from_numpy(group.tokens[start:stop]).to(self.device)
                    hands = torch.from_numpy(group.initial_hands[start:stop]).to(
                        self.device
                    )
                    with autocast_context(self.device, self.train.precision):
                        values = self.critic.forward_full(tokens, hands)
                    loss = torch.zeros((), device=self.device, dtype=torch.float32)
                    for rows in rows_by_phase.values():
                        if not rows["weight"]:
                            continue
                        seq = torch.tensor(
                            rows["seq_index"], device=self.device, dtype=torch.long
                        )
                        pos = torch.tensor(
                            rows["position"], device=self.device, dtype=torch.long
                        )
                        targets = torch.tensor(
                            rows["returns"], device=self.device, dtype=torch.float32
                        )
                        weight = torch.tensor(
                            rows["weight"], device=self.device, dtype=torch.float32
                        )
                        error = (values[seq, pos].float() - targets) / self.train.value_reward_scale
                        loss = loss + (0.5 * weight * error.square()).sum()
                    if loss.requires_grad:
                        loss.backward()
                        pending.append(loss.detach())
            stats.critic_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in self.critic.parameters()
                        if parameter.requires_grad
                    ],
                    self.train.max_grad_norm,
                )
            )
            self.critic_optimizer.step()
            epoch_loss = math.fsum(float(value) for value in pending)
            epoch_losses.append(epoch_loss)
            total_loss += epoch_loss
        stats.loss_value = total_loss
        self._record_critic_loss_dynamics(stats, epoch_losses)

    def _train_oracle_ppo_critic(
        self,
        groups: list[PPOCriticGroup],
        stats: SeqUpdateStats,
    ) -> None:
        """Fit every active player's return at configured oracle states."""

        assert isinstance(self.critic, SeqPPOOracleCritic)
        assert self.critic_optimizer is not None
        value_terms: list[torch.Tensor] = []
        trick_terms: list[torch.Tensor] = []
        card_order_terms: list[torch.Tensor] = []
        next_winner_terms: list[torch.Tensor] = []
        epoch_losses: list[float] = []
        oracle_trick_correct: list[torch.Tensor] = []
        oracle_trick_total: list[torch.Tensor] = []
        oracle_card_order_correct: list[torch.Tensor] = []
        oracle_card_order_total: list[torch.Tensor] = []
        oracle_next_winner_correct: list[torch.Tensor] = []
        oracle_next_winner_total: list[torch.Tensor] = []
        for critic_epoch in range(self.train.ppo_critic_epochs):
            self.critic.train()
            self.critic_optimizer.zero_grad(set_to_none=True)
            pending: list[torch.Tensor] = []
            for group in groups:
                chunks = list(self._microbatches(group))
                for (start, stop), rows in zip(
                    chunks, ppo_critic_rows_by_chunk(group, chunks)
                ):
                    if not rows["weight"]:
                        continue
                    tokens = torch.from_numpy(group.tokens[start:stop]).to(
                        self.device
                    )
                    seq = torch.tensor(
                        rows["seq_index"], device=self.device, dtype=torch.long
                    )
                    pos = torch.tensor(
                        rows["position"], device=self.device, dtype=torch.long
                    )
                    seq_np = np.asarray(rows["seq_index"], dtype=np.int64)
                    pos_np = np.asarray(rows["position"], dtype=np.int64)
                    auxiliary_active = (
                        self.train.trick_coef > 0
                        or self.train.oracle_card_order_coef > 0
                        or self.train.next_winner_coef > 0
                    )
                    auxiliary_seq_np = np.empty(0, dtype=np.int64)
                    auxiliary_pos_np = np.empty(0, dtype=np.int64)
                    auxiliary_weight = None
                    auxiliary_seq = None
                    auxiliary_pos = None
                    if auxiliary_active:
                        if (
                            auxiliary_active
                            and self.train.ppo_auxiliary_positions == "all"
                        ):
                            auxiliary_seq_np, auxiliary_pos_np = np.nonzero(
                                group.position_weight[start:stop] > 0
                            )
                            auxiliary_weight_np = group.position_weight[start:stop][
                                auxiliary_seq_np, auxiliary_pos_np
                            ]
                        else:
                            auxiliary_seq_np = seq_np
                            auxiliary_pos_np = pos_np
                            decision_count = np.bincount(
                                seq_np, minlength=stop - start
                            )
                            if np.any(decision_count == 0):
                                raise AssertionError(
                                    "Every oracle sequence must contain a decision."
                                )
                            sequence_weight = group.position_weight[
                                start:stop
                            ].sum(axis=1)
                            auxiliary_weight_np = (
                                sequence_weight[seq_np]
                                / decision_count[seq_np]
                            )
                        auxiliary_weight = torch.from_numpy(
                            auxiliary_weight_np.astype(np.float32, copy=False)
                        ).to(self.device)
                        auxiliary_seq = torch.from_numpy(auxiliary_seq_np).to(
                            self.device
                        )
                        auxiliary_pos = torch.from_numpy(auxiliary_pos_np).to(
                            self.device
                        )

                    dense_value = self.train.value_positions == "all"
                    value_seq = None
                    value_pos = None
                    if dense_value:
                        if (
                            auxiliary_active
                            and self.train.ppo_auxiliary_positions == "all"
                        ):
                            value_seq_np = auxiliary_seq_np
                            value_pos_np = auxiliary_pos_np
                        else:
                            value_seq_np, value_pos_np = np.nonzero(
                                group.position_weight[start:stop] > 0
                            )
                        value_seq = torch.from_numpy(value_seq_np).to(
                            self.device
                        )
                        value_pos = torch.from_numpy(value_pos_np).to(
                            self.device
                        )
                    with autocast_context(self.device, self.train.precision):
                        (
                            predictions,
                            trick_logits,
                            card_order_logits,
                            next_winner_logits,
                        ) = self.critic.forward_selected(
                            tokens,
                            seq,
                            pos,
                            value_seq=value_seq,
                            value_pos=value_pos,
                            dense_value_projection=dense_value,
                            auxiliary_seq=auxiliary_seq,
                            auxiliary_pos=auxiliary_pos,
                            dense_auxiliary_projection=(
                                self.train.ppo_auxiliary_positions == "all"
                            ),
                            with_trick=self.train.trick_coef > 0,
                            with_card_order=(
                                self.train.oracle_card_order_coef > 0
                            ),
                            with_next_winner=(
                                self.train.next_winner_coef > 0
                            ),
                        )
                    if dense_value:
                        targets = torch.from_numpy(
                            group.returns[start:stop][value_seq_np]
                        ).to(self.device)
                        weight = torch.from_numpy(
                            group.position_weight[start:stop][
                                value_seq_np, value_pos_np
                            ].astype(np.float32, copy=False)
                        ).to(self.device)
                        players = torch.from_numpy(
                            group.player_counts[start:stop][value_seq_np].astype(
                                np.int64, copy=False
                            )
                        ).to(self.device)
                    else:
                        targets = torch.from_numpy(
                            np.stack(rows["returns"])
                        ).to(self.device)
                        weight = torch.tensor(
                            rows["weight"],
                            device=self.device,
                            dtype=torch.float32,
                        )
                        players = torch.tensor(
                            rows["num_players"],
                            device=self.device,
                            dtype=torch.long,
                        )
                    active = (
                        torch.arange(
                            self.model.config.max_players,
                            device=self.device,
                        )[None, :]
                        < players[:, None]
                    )
                    predictions = center_zero_sum_values(
                        predictions.float(), players
                    )
                    error = (
                        predictions - targets.float()
                    ) / self.train.value_reward_scale
                    # Each game retains unit objective mass: dense mode divides
                    # it across causal positions, while the output-seat mean
                    # supplies every active player without favoring wide tables.
                    value_loss = (
                        0.5
                        * weight
                        * (error.square() * active.float()).sum(dim=-1)
                        / players.float()
                    ).sum()
                    loss = value_loss
                    value_terms.append(value_loss.detach())

                    auxiliary_active_players = None
                    if auxiliary_active:
                        auxiliary_player_counts = torch.from_numpy(
                            (
                                group.trick_targets[start:stop][auxiliary_seq_np]
                                != IGNORE_LABEL
                            ).sum(axis=1)
                        ).to(self.device)
                        auxiliary_active_players = (
                            torch.arange(
                                self.model.config.max_players,
                                device=self.device,
                            )[None, :]
                            < auxiliary_player_counts[:, None]
                        )

                    if self.train.trick_coef > 0:
                        assert trick_logits is not None
                        assert auxiliary_weight is not None
                        trick_targets = torch.from_numpy(
                            group.trick_targets[start:stop][auxiliary_seq_np]
                        ).to(self.device)
                        trick_masks = torch.from_numpy(
                            group.trick_masks[start:stop][
                                auxiliary_seq_np, auxiliary_pos_np
                            ]
                        ).to(self.device)
                        masked_logits = trick_logits.float().masked_fill(
                            ~trick_masks, -1e9
                        )
                        log_probs = torch.log_softmax(masked_logits, dim=-1)
                        safe_targets = trick_targets.clamp_min(0)[:, :, None]
                        gathered = log_probs.gather(2, safe_targets).squeeze(-1)
                        labeled = (
                            (trick_targets != IGNORE_LABEL)
                            & trick_masks.any(dim=-1)
                        )
                        per_position = (
                            (-gathered * labeled.float()).sum(dim=1)
                            / labeled.float().sum(dim=1).clamp_min(1.0)
                        )
                        trick_loss = (per_position * auxiliary_weight).sum()
                        loss = loss + self.train.trick_coef * trick_loss
                        trick_terms.append(trick_loss.detach())
                        if critic_epoch == 0:
                            trick_predictions = masked_logits.argmax(dim=-1)
                            oracle_trick_correct.append(
                                ((trick_predictions == trick_targets) & labeled)
                                .sum()
                                .detach()
                            )
                            oracle_trick_total.append(labeled.sum().detach())

                    if self.train.next_winner_coef > 0:
                        assert next_winner_logits is not None
                        assert auxiliary_weight is not None
                        next_winner_targets = torch.from_numpy(
                            group.next_winner_targets[start:stop][
                                auxiliary_seq_np, auxiliary_pos_np
                            ]
                        ).to(self.device)
                        seat_counts = torch.from_numpy(
                            group.player_counts[start:stop][
                                auxiliary_seq_np
                            ].astype(np.int64, copy=False)
                        ).to(self.device)
                        seat_active = (
                            torch.arange(
                                self.model.config.max_players,
                                device=self.device,
                            )[None, :]
                            < seat_counts[:, None]
                        )
                        masked_next_winner = next_winner_logits.float().masked_fill(
                            ~seat_active, -1e9
                        )
                        log_probs = torch.log_softmax(masked_next_winner, dim=-1)
                        labeled = next_winner_targets != IGNORE_LABEL
                        gathered = log_probs.gather(
                            1, next_winner_targets.clamp_min(0)[:, None]
                        ).squeeze(-1)
                        per_position = -gathered * labeled.float()
                        next_winner_loss = (
                            per_position * auxiliary_weight
                        ).sum()
                        loss = (
                            loss
                            + self.train.next_winner_coef * next_winner_loss
                        )
                        next_winner_terms.append(next_winner_loss.detach())
                        if critic_epoch == 0:
                            oracle_next_winner_correct.append(
                                (
                                    (
                                        masked_next_winner.argmax(dim=-1)
                                        == next_winner_targets
                                    )
                                    & labeled
                                )
                                .sum()
                                .detach()
                            )
                            oracle_next_winner_total.append(
                                labeled.sum().detach()
                            )

                    if self.train.oracle_card_order_coef > 0:
                        assert card_order_logits is not None
                        assert auxiliary_weight is not None
                        assert auxiliary_active_players is not None
                        card_order_targets = torch.from_numpy(
                            group.card_play_order[start:stop][auxiliary_seq_np]
                        ).to(self.device)
                        played_counts = torch.from_numpy(
                            group.cards_played[start:stop][
                                auxiliary_seq_np, auxiliary_pos_np
                            ]
                        ).to(self.device)
                        per_state, order_correct, order_labeled = (
                            plackett_luce_card_order_terms(
                                card_order_logits,
                                card_order_targets,
                                played_counts,
                                auxiliary_active_players,
                            )
                        )
                        card_order_loss = (
                            per_state * auxiliary_weight
                        ).sum()
                        loss = (
                            loss
                            + self.train.oracle_card_order_coef
                            * card_order_loss
                        )
                        card_order_terms.append(card_order_loss.detach())
                        if critic_epoch == 0:
                            oracle_card_order_correct.append(
                                order_correct.sum().detach()
                            )
                            oracle_card_order_total.append(
                                order_labeled.sum().detach()
                            )
                    loss.backward()
                    pending.append(loss.detach())
            stats.critic_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in self.critic.parameters()
                        if parameter.requires_grad
                    ],
                    self.train.max_grad_norm,
                )
            )
            self.critic_optimizer.step()
            epoch_loss = math.fsum(float(value) for value in pending)
            epoch_losses.append(epoch_loss)
        stats.loss_value = math.fsum(float(value) for value in value_terms)
        stats.oracle_value_loss = stats.loss_value / max(
            1, self.train.ppo_critic_epochs
        )
        stats.loss_oracle_trick = math.fsum(
            float(value) for value in trick_terms
        )
        stats.loss_oracle_card_order = math.fsum(
            float(value) for value in card_order_terms
        )
        stats.loss_oracle_next_winner = math.fsum(
            float(value) for value in next_winner_terms
        )
        correct = math.fsum(float(value) for value in oracle_trick_correct)
        labeled = math.fsum(float(value) for value in oracle_trick_total)
        stats.oracle_trick_accuracy = (
            correct / labeled
            if labeled > 0
            else 0.0
        )
        order_correct = math.fsum(
            float(value) for value in oracle_card_order_correct
        )
        order_labeled = math.fsum(
            float(value) for value in oracle_card_order_total
        )
        stats.oracle_card_order_top1_accuracy = (
            order_correct / order_labeled
            if order_labeled > 0
            else 0.0
        )
        winner_correct = math.fsum(
            float(value) for value in oracle_next_winner_correct
        )
        winner_labeled = math.fsum(
            float(value) for value in oracle_next_winner_total
        )
        stats.oracle_next_winner_accuracy = (
            winner_correct / winner_labeled
            if winner_labeled > 0
            else 0.0
        )
        self._record_critic_loss_dynamics(stats, epoch_losses)

    @staticmethod
    def _record_critic_loss_dynamics(
        stats: SeqUpdateStats, epoch_losses: list[float]
    ) -> None:
        if not epoch_losses:
            return
        first = epoch_losses[0]
        last = epoch_losses[-1]
        stats.critic_loss_first_epoch = first
        stats.critic_loss_last_epoch = last
        if first > 0:
            stats.critic_loss_reduction = (first - last) / first

    def _apply_warmup_lr(self) -> None:
        """Linear LR ramp over the first ``lr_warmup_updates`` kept steps.

        Adam's early steps are sign steps -- ``m_hat/sqrt(v_hat)`` is +/-1
        whatever the gradient magnitude, so gradient clipping cannot soften
        them and a cold-started model jumps ~lr per parameter in one step,
        far past any reasonable KL cap. The ramp counts *kept* steps, so a
        rolled-back epoch retries at the same scale on the next iteration's
        fresh data instead of ratcheting up regardless.
        """

        warmup = self.train.lr_warmup_updates
        scale = 1.0 if warmup <= 0 else min(1.0, (self.optimizer_steps + 1) / warmup)
        for group in self.optimizer.param_groups:
            base = (
                self.train.core_lr
                if group.get("kl_sensitive", True)
                else self.train.auxiliary_lr
            )
            group["lr"] = base * scale

    def _release_device_memory(self) -> None:
        # MPS's caching allocator keeps its high-water mark resident; with a
        # tight unified-memory budget that pushes later phases into swap.
        if self.device.type == "mps":
            torch.mps.empty_cache()

    def _microbatches(self, group: SeqTrainingGroup):
        batch, length = group.tokens.shape[:2]
        rows_per_chunk = max(self.train.microbatch_positions // max(length, 1), 1)
        for start in range(0, batch, rows_per_chunk):
            yield start, min(start + rows_per_chunk, batch)

    def _epoch_backward(self, groups, stats: SeqUpdateStats) -> None:
        totals = defaultdict(float)
        # Keep reporting scalars on-device until every forward/backward has
        # been queued. Calling float(tensor) inside _chunk_loss synchronizes
        # MPS once per microbatch, inserting a CPU barrier between each
        # forward and backward. Converting here preserves the original Python
        # summation order (and therefore the exact reported values) while the
        # first conversion pays the only synchronization.
        pending_terms: list[tuple[str, list[torch.Tensor]]] = []
        for group in groups:
            by_chunk = _rows_by_chunk(group, self._microbatches(group))
            for (start, stop), policy_rows in zip(self._microbatches(group), by_chunk):
                loss, terms = self._chunk_loss(
                    group, start, stop, policy_rows, defer_terms=True
                )
                if loss is not None:
                    loss.backward()
                pending_terms.extend(terms.items())
        for key, values in pending_terms:
            subtotal = 0.0
            for value in values:
                subtotal += float(value)
            totals[key] += subtotal
        stats.loss_policy = totals["policy"]
        stats.loss_value = totals["value"]
        stats.loss_value_zero = totals["value_zero"]
        value_weight = totals["value_weight"]
        if value_weight > 0:
            stats.value_rmse = math.sqrt(
                max(totals["value_squared_error"] / value_weight, 0.0)
            )
            stats.value_zero_rmse = math.sqrt(
                max(totals["value_target_squared"] / value_weight, 0.0)
            )
            prediction_mean = totals["value_prediction_sum"] / value_weight
            target_mean = totals["value_target_sum"] / value_weight
            prediction_variance = max(
                totals["value_prediction_squared"] / value_weight
                - prediction_mean**2,
                0.0,
            )
            target_variance = max(
                totals["value_target_squared"] / value_weight - target_mean**2,
                0.0,
            )
            covariance = (
                totals["value_prediction_target"] / value_weight
                - prediction_mean * target_mean
            )
            stats.value_prediction_std = math.sqrt(prediction_variance)
            denominator = math.sqrt(prediction_variance * target_variance)
            stats.value_correlation = (
                covariance / denominator if denominator > 0 else 0.0
            )
        stats.loss_suit = totals["suit"]
        stats.loss_rank_boundary = totals["rank_boundary"]
        stats.loss_trick = totals["trick"]
        stats.loss_next_winner = totals["next_winner"]
        stats.loss_bid_hit = totals["bid_hit"]
        stats.entropy = totals["entropy"]
        policy_row_weight = totals["policy_row_weight"]
        stats.policy_logit_shift = (
            totals["policy_logit_shift"] / policy_row_weight
            if policy_row_weight > 0
            else 0.0
        )

    def _chunk_loss(
        self, group, start, stop, policy_rows, *, defer_terms: bool = False
    ):
        device = self.device
        train = self.train
        # Only the heads whose loss is actually weighted -- each one skipped is
        # its whole forward and backward saved.
        aux = frozenset(
            name
            for name, coef in (
                ("suit", train.suit_coef),
                ("rank_boundary", train.rank_boundary_coef),
                ("trick", train.trick_coef),
                ("bid_hit", train.bid_hit_coef),
                ("next_winner", train.next_winner_coef),
            )
            if coef > 0
        )
        tokens = torch.from_numpy(group.tokens[start:stop]).to(device)
        with autocast_context(self.device, self.train.precision):
            output = self.model.forward_full(tokens, aux_heads=aux)
        owned = torch.from_numpy(group.owned[start:stop]).to(device)
        position_weight = (
            torch.from_numpy(group.position_weight[start:stop]).to(device)
            * owned.float()
        )

        term_parts: dict[str, list[torch.Tensor]] = defaultdict(list)
        loss = tokens.new_zeros((), dtype=torch.float32)

        # Value. The baseline is consumed only at focal policy decisions, so
        # the default value_weight is aligned exactly to those rows. Normalized
        # MSE learns E[R|s]; Smooth-L1 remains an explicit legacy option.
        value_targets = torch.from_numpy(group.value_targets[start:stop]).to(device)
        value_weight = (
            torch.from_numpy(group.value_weight[start:stop]).to(device)
            * owned.float()
        )
        if train.value_objective == "mse":
            scaled_error = (
                output.value - value_targets
            ) / train.value_reward_scale
            value_loss = (0.5 * scaled_error.square() * value_weight).sum()
            value_zero_loss = (
                0.5
                * (value_targets / train.value_reward_scale).square()
                * value_weight
            ).sum()
        else:
            value_loss = (
                F.smooth_l1_loss(output.value, value_targets, reduction="none")
                * value_weight
            ).sum()
            target_magnitude = value_targets.abs()
            value_zero_loss = (
                torch.where(
                    target_magnitude < 1.0,
                    0.5 * target_magnitude.square(),
                    target_magnitude - 0.5,
                )
                * value_weight
            ).sum()
        raw_error = output.value - value_targets
        term_parts["value_weight"].append(value_weight.sum().detach())
        term_parts["value_squared_error"].append(
            (raw_error.square() * value_weight).sum().detach()
        )
        term_parts["value_target_squared"].append(
            (value_targets.square() * value_weight).sum().detach()
        )
        term_parts["value_prediction_sum"].append(
            (output.value * value_weight).sum().detach()
        )
        term_parts["value_target_sum"].append(
            (value_targets * value_weight).sum().detach()
        )
        term_parts["value_prediction_squared"].append(
            (output.value.square() * value_weight).sum().detach()
        )
        term_parts["value_prediction_target"].append(
            (output.value * value_targets * value_weight).sum().detach()
        )
        loss = loss + train.value_coef * value_loss
        term_parts["value"].append(value_loss.detach())
        term_parts["value_zero"].append(value_zero_loss.detach())

        # Suit presence.
        suit_labeled = None
        if train.suit_coef > 0:
            suit_targets = torch.from_numpy(group.suit_targets[start:stop]).to(device)
            suit_labeled = (suit_targets != IGNORE_LABEL) & owned[:, :, None, None]
        if suit_labeled is not None and suit_labeled.any():
            bce = F.binary_cross_entropy_with_logits(
                output.suit_logits,
                suit_targets.clamp_min(0).float(),
                reduction="none",
            )
            per_position = (bce * suit_labeled.float()).sum(dim=(2, 3)) / (
                suit_labeled.float().sum(dim=(2, 3)).clamp_min(1.0)
            )
            suit_loss = (per_position * position_weight).sum()
            loss = loss + train.suit_coef * suit_loss
            term_parts["suit"].append(suit_loss.detach())

        # Rank-boundary beliefs: opponent rows plus one any-opponent row,
        # normalized over only the meaningful bits at each position. Absent
        # observer suits and inactive seats remain IGNORE_LABEL.
        rank_boundary_labeled = None
        if train.rank_boundary_coef > 0:
            rank_boundary_targets = torch.from_numpy(
                group.rank_boundary_targets[start:stop]
            ).to(device)
            rank_boundary_labeled = (
                (rank_boundary_targets != IGNORE_LABEL)
                & owned[:, :, None, None, None]
            )
        if rank_boundary_labeled is not None and rank_boundary_labeled.any():
            bce = F.binary_cross_entropy_with_logits(
                output.rank_boundary_logits,
                rank_boundary_targets.clamp_min(0).float(),
                reduction="none",
            )
            labeled_count = rank_boundary_labeled.float().sum(
                dim=(2, 3, 4)
            )
            per_position = (
                (bce * rank_boundary_labeled.float()).sum(dim=(2, 3, 4))
                / labeled_count.clamp_min(1.0)
            )
            rank_boundary_loss = (per_position * position_weight).sum()
            loss = loss + train.rank_boundary_coef * rank_boundary_loss
            term_parts["rank_boundary"].append(rank_boundary_loss.detach())

        # Trick counts (feasibility-masked CE, per relative player).
        player_labeled = None
        if train.trick_coef > 0:
            trick_targets = torch.from_numpy(group.trick_targets[start:stop]).to(device)
            trick_masks = torch.from_numpy(group.trick_masks[start:stop]).to(device)
            player_labeled = trick_targets != IGNORE_LABEL  # [B, P]
        if player_labeled is not None and player_labeled.any():
            # -1e9 (not -inf): unowned positions have all-False feasibility
            # rows whose softmax must stay finite even though they are masked
            # out of the loss below.
            logits = output.trick_logits.masked_fill(~trick_masks, -1e9)
            log_probs = torch.log_softmax(logits, dim=-1)
            safe_targets = trick_targets.clamp_min(0)[:, None, :, None].expand(
                -1, logits.shape[1], -1, 1
            )
            gathered = log_probs.gather(3, safe_targets).squeeze(-1)  # [B,L,P]
            mask = (
                player_labeled[:, None, :] & owned[:, :, None] & trick_masks.any(dim=-1)
            )
            per_position = (-gathered * mask.float()).sum(dim=2) / (
                mask.float().sum(dim=2).clamp_min(1.0)
            )
            trick_loss = (per_position * position_weight).sum()
            loss = loss + train.trick_coef * trick_loss
            term_parts["trick"].append(trick_loss.detach())

        # Bid hit (per relative seat, sigmoid). One constant label per seat for
        # the whole round -- it is the outcome -- broadcast over every owned
        # position, so the same head is asked the same question from a
        # progressively more informed prefix.
        bid_hit_labeled = None
        if train.bid_hit_coef > 0:
            bid_hit_targets = torch.from_numpy(group.bid_hit_targets[start:stop]).to(
                device
            )
            bid_hit_labeled = bid_hit_targets != IGNORE_LABEL  # [B, P]
        if bid_hit_labeled is not None and bid_hit_labeled.any():
            logits = output.bid_hit_logits  # [B, L, P]
            bce = F.binary_cross_entropy_with_logits(
                logits,
                bid_hit_targets.clamp_min(0).float()[:, None, :].expand_as(logits),
                reduction="none",
            )
            mask = bid_hit_labeled[:, None, :].expand_as(logits).float()
            per_position = (bce * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
            bid_hit_loss = (per_position * position_weight).sum()
            loss = loss + train.bid_hit_coef * bid_hit_loss
            term_parts["bid_hit"].append(bid_hit_loss.detach())

        # Next trick winner (softmax over active seats, per position). The
        # label is IGNORE_LABEL outside the play phase, so unlike the targets
        # above this one supervises a strict sub-window of the owned tokens.
        if train.next_winner_coef > 0:
            next_winner_targets = torch.from_numpy(
                group.next_winner_targets[start:stop]
            ).to(device)
            labeled = (next_winner_targets != IGNORE_LABEL) & owned
            if labeled.any():
                seat_active = (
                    torch.arange(
                        self.model.config.max_players, device=device
                    )[None, None, :]
                    < torch.full(
                        (stop - start, 1, 1),
                        group.num_players,
                        device=device,
                        dtype=torch.long,
                    )
                )
                logits = output.next_winner_logits.masked_fill(
                    ~seat_active, -1e9
                )
                log_probs = torch.log_softmax(logits, dim=-1)
                gathered = log_probs.gather(
                    2, next_winner_targets.clamp_min(0)[:, :, None]
                ).squeeze(-1)
                next_winner_loss = (
                    -gathered * labeled.float() * position_weight
                ).sum()
                loss = loss + train.next_winner_coef * next_winner_loss
                term_parts["next_winner"].append(next_winner_loss.detach())

        # Policy: NeuRD over every focal decision, branched or not.
        for phase_key, rows in policy_rows.items():
            if not rows["weight"]:
                continue
            losses, _, entropies, weight, logit_shift = self._policy_terms(
                output, rows, phase_key
            )
            policy_loss = (weight * losses).sum()
            loss = loss + train.policy_coef * policy_loss
            term_parts["policy"].append(policy_loss.detach())
            if train.entropy_coef:
                entropy_term = (weight * entropies).sum()
                loss = loss - train.entropy_coef * entropy_term
                term_parts["entropy"].append(entropy_term.detach())
            else:
                term_parts["entropy"].append(
                    (weight * entropies).sum().detach()
                )
            term_parts["policy_logit_shift"].append(
                (weight * logit_shift).sum().detach()
            )
            term_parts["policy_row_weight"].append(weight.sum().detach())

        if defer_terms:
            terms = dict(term_parts)
        else:
            # Preserve the historical public helper contract used by focused
            # loss tests. Training itself requests deferred tensors above.
            terms = {}
            for key, values in term_parts.items():
                subtotal = 0.0
                for value in values:
                    subtotal += float(value)
                terms[key] = subtotal
        if loss.requires_grad:
            return loss, terms
        return None, terms

    def _policy_terms(self, output, rows, phase_key):
        """Per-row selected-objective losses, divergences, and entropies."""

        device = self.device
        train = self.train
        logits_all = output.bid_logits if phase_key == "bid" else output.card_logits

        def to_device(array: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(array).to(device)

        # Everything below is padded on the host and shipped in one transfer per
        # field. Padding these row by row on the device instead costs four
        # kernel launches per policy row -- ~200k tiny MPS dispatches an update,
        # which dominated the whole backward pass.
        seq_idx = to_device(np.asarray(rows["seq_index"], dtype=np.int64))
        pos_idx = to_device(np.asarray(rows["position"], dtype=np.int64))
        logits = logits_all[seq_idx, pos_idx].float()

        candidates = rows["candidates"]
        count = len(rows["weight"])
        max_k = max(len(c) for c in candidates)
        cand_np = np.zeros((count, max_k), dtype=np.int64)
        mask_np = np.zeros((count, max_k), dtype=bool)
        # Padding stays at 1.0, not 0.0: the 1/q correction raises this to a
        # negative power before the mask is applied.
        incl_np = np.ones((count, max_k), dtype=np.float32)
        q_np = np.zeros((count, max_k), dtype=np.float32)
        for row in range(count):
            k = len(candidates[row])
            cand_np[row, :k] = candidates[row]
            mask_np[row, :k] = True
            incl_np[row, :k] = rows["inclusion"][row]
            q_np[row, :k] = rows["q_values"][row]
        cand = to_device(cand_np)
        cand_mask = to_device(mask_np)
        inclusion = to_device(incl_np)
        q_values = to_device(q_np)
        old_full = to_device(np.stack(rows["old_probs_full"]))
        weight = to_device(np.asarray(rows["weight"], dtype=np.float32))
        baseline = to_device(np.asarray(rows["baseline"], dtype=np.float32)).unsqueeze(
            -1
        )

        # Select the explicitly configured estimator. Exact NeuRD uses
        # exponent=1 with no cap or clip. The sampled-mirror knobs remain
        # independent because it is an explicitly biased variance tradeoff.
        if train.policy_objective == "neurd":
            advantage_clip = train.neurd_advantage_clip
            exponent = train.neurd_inclusion_exponent
            inclusion_cap = train.neurd_inclusion_cap
        else:
            advantage_clip = train.sampled_mirror_advantage_clip
            exponent = train.sampled_mirror_inclusion_exponent
            inclusion_cap = train.sampled_mirror_inclusion_cap

        full_adv = control_variate_action_advantages(
            old_full,
            cand,
            q_values,
            inclusion,
            cand_mask,
            baseline,
            inclusion_exponent=exponent,
            inclusion_cap=inclusion_cap,
            advantage_clip=advantage_clip,
        )

        # Full-support KL(old || new) anchor and guard metric.
        legal = old_full > 0
        full_logprobs = torch.log_softmax(
            logits.masked_fill(~legal, float("-inf")), dim=-1
        )
        safe = torch.where(legal, full_logprobs, torch.zeros_like(full_logprobs))
        log_old = torch.log(old_full.clamp_min(1e-12))
        divergences = (old_full * (log_old - safe) * legal.float()).sum(dim=-1)
        probs_new = safe.exp() * legal.float()
        entropies = -(probs_new * safe).sum(dim=-1)

        # Common-mode logit drift. The NeuRD loss -sum_a A_hat(a) z(a) has
        # gradient -sum_a A_hat(a) along the all-ones logit direction, and that
        # sum is not zero in general -- only the pi_old-weighted one is. Adding
        # a constant to every legal logit leaves the policy identical, so the
        # post-step KL guard cannot see this direction and nothing else bounds
        # it. Reported so a slow drift into saturation is visible rather than
        # inferred after the fact.
        logit_shift = (logits * legal.float()).sum(dim=-1) / legal.sum(
            dim=-1
        ).clamp_min(1)

        if train.policy_objective == "neurd":
            # Full-support sampled NeuRD. The control-variate estimator is
            # centered over the old policy before this direct logit update, so
            # it has the paper's sampled-regret semantics even when only a
            # subset of action values was evaluated.
            finite_logits = torch.where(legal, logits, torch.zeros_like(logits))
            regret = -(full_adv.detach() * finite_logits).sum(dim=-1)
            losses = (
                train.neurd_regret_coef * regret + train.neurd_kl_coef * divergences
            )
        else:
            target = sampled_mirror_target(
                old_full,
                full_adv,
                legal,
                step_size=train.sampled_mirror_step_size,
                uniform_mix=train.sampled_mirror_uniform_mix,
                target_kl=train.sampled_mirror_target_kl,
            )
            log_target = torch.log(target.clamp_min(1e-12))
            # KL(target || current). The target is stopped by construction;
            # the gradient on each logit is pi_current - target.
            losses = (target * (log_target - safe) * legal.float()).sum(dim=-1)
        return losses, divergences, entropies, weight, logit_shift

    def _policy_divergences(self, output, rows, phase_key):
        """KL-only policy readout for the post-step trust-region guard.

        The guard does not consume candidate Q values, inclusion
        probabilities, baselines, sampled advantages, entropies, or mirror
        targets. Reusing _policy_terms here rebuilt and transferred all of
        those tensors and recomputed the complete objective after every
        attempted Adam step even though only KL(old || new) was retained.
        """

        logits_all = output.bid_logits if phase_key == "bid" else output.card_logits
        seq_idx = torch.from_numpy(
            np.asarray(rows["seq_index"], dtype=np.int64)
        ).to(self.device)
        pos_idx = torch.from_numpy(
            np.asarray(rows["position"], dtype=np.int64)
        ).to(self.device)
        logits = logits_all[seq_idx, pos_idx].float()
        old_full = torch.from_numpy(np.stack(rows["old_probs_full"])).to(self.device)
        legal = old_full > 0
        full_logprobs = torch.log_softmax(
            logits.masked_fill(~legal, float("-inf")), dim=-1
        )
        safe = torch.where(
            legal, full_logprobs, torch.zeros_like(full_logprobs)
        )
        log_old = torch.log(old_full.clamp_min(1e-12))
        divergences = (
            old_full * (log_old - safe) * legal.float()
        ).sum(dim=-1)
        weight = torch.from_numpy(
            np.asarray(rows["weight"], dtype=np.float32)
        ).to(self.device)
        return divergences, weight

    def _evaluate_policy_kl(
        self,
        groups,
        *,
        reject_p99_above: float | None = None,
    ) -> tuple[float, float, float, float]:
        """Mean, p95, p99, and max KL(old||new) over every policy row.

        Under the old split this covered branch rows only, leaving the spine
        half of the decisions outside the rollback guard. With one row type
        there is nothing left uncovered.

        A non-None ``reject_p99_above`` permits a proof-only early return. If
        the weight already observed above the cap is large enough that even
        assigning every unvisited row below it cannot save weighted p99, the
        attempted step is known to fail. The threshold includes a conservative
        IEEE-float32 summation error bound, so this cannot reject an attempt
        the complete existing percentile calculation would accept.
        """

        self.model.eval()
        all_divergences: list[np.ndarray] = []
        all_weights: list[np.ndarray] = []
        bad_weights: list[float] = []
        rejection_weight = None
        if reject_p99_above is not None:
            guard_weights = np.asarray(
                [
                    weight
                    for group in groups
                    for rows in group.policy.values()
                    for weight in rows.weight
                ],
                dtype=np.float32,
            )
            positive_weights = [
                float(weight) for weight in guard_weights[guard_weights > 0]
            ]
            count = len(positive_weights)
            total = math.fsum(positive_weights)
            # numpy.cumsum retains float32 for these weights. Bound the
            # order-dependent rounding of both the below-cap prefix and total
            # using Higham's gamma_n, then require the worst-case prefix still
            # to fall below 99% of the best-case total.
            unit_roundoff = 2.0**-24
            product = max(count - 1, 0) * unit_roundoff
            if total > 0 and product < 1.0:
                gamma = product / (1.0 - product)
                rejection_weight = total * (
                    1.0 - 0.99 * (1.0 - gamma) / (1.0 + gamma)
                )
        with torch.inference_mode():
            for group in groups:
                for (start, stop), policy_rows in zip(
                    self._microbatches(group),
                    _rows_by_chunk(group, self._microbatches(group)),
                ):
                    if not any(rows["weight"] for rows in policy_rows.values()):
                        continue
                    tokens = torch.from_numpy(group.tokens[start:stop]).to(self.device)
                    with autocast_context(self.device, self.train.precision):
                        output = self.model.forward_full(tokens, aux_heads=False)
                    for phase_key, rows in policy_rows.items():
                        if not rows["weight"]:
                            continue
                        divergences, weight = self._policy_divergences(
                            output, rows, phase_key
                        )
                        divergence_np = divergences.detach().cpu().numpy()
                        weight_np = weight.detach().cpu().numpy()
                        all_divergences.append(divergence_np)
                        all_weights.append(weight_np)
                        if rejection_weight is not None:
                            mask = (
                                (divergence_np > reject_p99_above)
                                & (weight_np > 0)
                            )
                            bad_weights.extend(
                                float(value) for value in weight_np[mask]
                            )
                    if (
                        rejection_weight is not None
                        and math.fsum(bad_weights) > rejection_weight
                    ):
                        rejected = float(
                            np.nextafter(reject_p99_above, math.inf)
                        )
                        return 0.0, 0.0, rejected, rejected
        if not all_divergences:
            return 0.0, 0.0, 0.0, 0.0
        return _weighted_kl_summary(
            np.concatenate(all_divergences),
            np.concatenate(all_weights),
        )

    # -------------------------------------------------------------- #
    # Checkpoints & league                                            #
    # -------------------------------------------------------------- #

    # ---------------------------- distillation --------------------------- #

    def _update_distill(self, trees: list[SeqTree]) -> SeqUpdateStats:
        """One supervised pass against the frozen teacher.

        No advantages, no critic, no trust region and no duals. Those
        controllers all exist to keep a policy-gradient step from moving too
        far relative to the behaviour policy that produced the data; a KL to a
        fixed target has no such failure mode, so the update is a plain
        accumulate-and-step.
        """

        started = time.perf_counter()
        self.collector.release_caches()
        self._release_device_memory()
        groups = build_training_groups(trees, self.model.config, self.train)
        stats = SeqUpdateStats()
        stats.build_sec = time.perf_counter() - started
        stats.policy_rows = sum(
            len(rows.weight) for g in groups for rows in g.policy.values()
        )
        stats.unbranched_rows = stats.policy_rows
        stats.value_rows = int(
            sum(np.count_nonzero(group.value_weight) for group in groups)
        )
        stats.positions = int(sum(g.owned.sum() for g in groups))

        totals: dict[str, float] = defaultdict(float)
        for _ in range(max(1, self.train.epochs)):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            totals = defaultdict(float)
            for group in groups:
                chunks = list(self._microbatches(group))
                for (start, stop), rows in zip(chunks, _rows_by_chunk(group, chunks)):
                    loss, terms = self._distill_chunk_loss(group, start, stop, rows)
                    loss.backward()
                    for key, value in terms.items():
                        totals[key] += value
            # A single NaN step is unrecoverable: it writes NaN into the
            # weights, and every subsequent forward, gradient and metric is
            # NaN forever after. The run then looks alive while learning
            # nothing. Checking before the step costs one reduction per
            # parameter and makes that failure mode survivable -- the step is
            # dropped and the model stays at its last good weights.
            finite = all(
                torch.isfinite(parameter.grad).all()
                for parameter in self._actor_parameters
                if parameter.grad is not None
            )
            if not finite:
                stats.distill_skipped_steps += 1.0
                self.optimizer.zero_grad(set_to_none=True)
                continue
            (
                stats.core_grad_norm,
                stats.embedding_grad_norm,
                stats.auxiliary_grad_norm,
            ) = self._clip_actor()
            self._apply_warmup_lr()
            self.optimizer.step()
            # The effective card matrix is rebuilt from rank/suit rows; a step
            # invalidates the evaluation-time cache of those 52 directions.
            self.model.clear_card_output_cache()
            self.optimizer_steps += 1
        self._record_distill_stats(stats, totals)
        stats.update_sec = time.perf_counter() - started
        return stats

    def _distill_chunk_loss(self, group, start, stop, policy_rows):
        device = self.device
        train = self.train
        temperature = float(train.distill_temperature)
        aux_names: frozenset[str] = (
            frozenset(("suit", "rank_boundary", "trick", "bid_hit", "next_winner"))
            if train.distill_aux_coef > 0
            else frozenset()
        )
        tokens = torch.from_numpy(group.tokens[start:stop]).to(device)
        with autocast_context(self.device, train.precision):
            with torch.no_grad():
                teacher = self.teacher.forward_full(tokens, aux_heads=aux_names)
            student = self.model.forward_full(tokens, aux_heads=aux_names)

        owned = torch.from_numpy(group.owned[start:stop]).to(device)
        position_weight = (
            torch.from_numpy(group.position_weight[start:stop]).to(device)
            * owned.float()
        )
        terms: dict[str, float] = defaultdict(float)
        loss = tokens.new_zeros((), dtype=torch.float32)

        # ---- policy: forward KL against the teacher's forward pass ---------
        #
        # Under on-policy collection the student is what acted, so the
        # recorded ``old_probs_full`` is the *student's* distribution and is
        # useless as a target. It is still the authority on which actions were
        # legal -- legality is a property of the state, not of the policy that
        # visited it -- so it survives here purely as the mask that both sides
        # are restricted and renormalized onto.
        if train.distill_policy_coef > 0:
            for phase_key, rows in policy_rows.items():
                if not rows["weight"]:
                    continue
                student_logits_all = (
                    student.bid_logits if phase_key == "bid" else student.card_logits
                )
                teacher_logits_all = (
                    teacher.bid_logits if phase_key == "bid" else teacher.card_logits
                )
                seq_idx = torch.from_numpy(
                    np.asarray(rows["seq_index"], dtype=np.int64)
                ).to(device)
                pos_idx = torch.from_numpy(
                    np.asarray(rows["position"], dtype=np.int64)
                ).to(device)
                logits = student_logits_all[seq_idx, pos_idx].float()
                teacher_logits = teacher_logits_all[seq_idx, pos_idx].float()
                behaviour = torch.from_numpy(
                    np.stack(rows["old_probs_full"]).astype(np.float32)
                ).to(device)
                legal = behaviour > 0
                target = torch.softmax(
                    teacher_logits.masked_fill(~legal, -float("inf")) / temperature,
                    dim=-1,
                )
                student_log_probs = torch.log_softmax(
                    logits.masked_fill(~legal, -float("inf")) / temperature, dim=-1
                )
                safe_log_student = torch.where(
                    legal, student_log_probs, torch.zeros_like(student_log_probs)
                )
                log_target = torch.where(
                    legal, target.clamp_min(1e-12).log(), torch.zeros_like(target)
                )
                # KL(teacher || student), summed over legal actions per row.
                row_kl = (target * (log_target - safe_log_student)).sum(dim=-1)
                # Weight by the group builder's own per-row weights, which sum
                # to exactly 1 across every group and phase in the update.
                #
                # A per-chunk ``.mean()`` here is wrong and was the source of
                # the ~200 gradient norms: every microbatch calls backward
                # separately, so per-chunk means *add* rather than average, and
                # the policy term ended up scaled by the number of chunks --
                # roughly 40x, drifting with however many shapes and
                # microbatches a given update happened to produce. Value and
                # auxiliary terms were already normalized this way, so policy
                # also silently dominated them by that same factor.
                row_weight = torch.from_numpy(
                    np.asarray(rows["weight"], dtype=np.float32)
                ).to(device)
                phase_loss = (row_kl * row_weight).sum() * (temperature**2)
                loss = loss + train.distill_policy_coef * phase_loss

                with torch.no_grad():
                    rows_n = float(row_kl.numel())
                    teacher_best = target.argmax(dim=-1)
                    student_best = student_log_probs.argmax(dim=-1)
                    student_probs = safe_log_student.exp() * legal
                    teacher_entropy = -(
                        target * log_target
                    ).sum(dim=-1)
                    student_entropy = -(student_probs * safe_log_student).sum(dim=-1)
                    terms[f"kl_{phase_key}"] += float(row_kl.sum())
                    terms[f"n_{phase_key}"] += rows_n
                    terms[f"agree_{phase_key}"] += float(
                        (teacher_best == student_best).sum()
                    )
                    terms[f"top1_{phase_key}"] += float(
                        student_probs.gather(1, teacher_best[:, None]).sum()
                    )
                    terms[f"tent_{phase_key}"] += float(teacher_entropy.sum())
                    terms[f"sent_{phase_key}"] += float(student_entropy.sum())
                terms["loss_policy"] += float(phase_loss.detach())

        # ---- value: match the teacher's prediction, not the return ---------
        #
        # Regressing on the teacher's value rather than the Monte Carlo return
        # is the point of distillation: the teacher's estimate is the return's
        # conditional mean with the sampling noise already integrated out.
        if train.distill_value_coef > 0 and float(position_weight.sum()) > 0:
            weight = position_weight
            total_weight = weight.sum().clamp_min(1.0)
            error = student.value.float() - teacher.value.float()
            value_loss = (error.square() * weight).sum() / total_weight
            loss = loss + train.distill_value_coef * value_loss
            with torch.no_grad():
                terms["loss_value"] += float(value_loss.detach())
                terms["value_sq"] += float((error.square() * weight).sum())
                terms["value_w"] += float(weight.sum())
                sp, tp = student.value.float(), teacher.value.float()
                terms["v_s"] += float((sp * weight).sum())
                terms["v_t"] += float((tp * weight).sum())
                terms["v_ss"] += float((sp * sp * weight).sum())
                terms["v_tt"] += float((tp * tp * weight).sum())
                terms["v_st"] += float((sp * tp * weight).sum())

        # ---- auxiliary heads: soft targets, not ground truth ---------------
        #
        # Every head is distilled against the teacher's *distribution* so the
        # student inherits its calibration, including where it is unsure.
        # Binary heads use the teacher's sigmoid as a soft BCE label; the two
        # categorical heads use its softmax as a soft cross-entropy label.
        if aux_names and float(position_weight.sum()) > 0:
            aux_loss = tokens.new_zeros((), dtype=torch.float32)
            total_weight = position_weight.sum().clamp_min(1.0)

            def reduce(per_position: torch.Tensor) -> torch.Tensor:
                return (per_position * position_weight).sum() / total_weight

            def collapse(values: torch.Tensor, extra_mask=None) -> torch.Tensor:
                """Average trailing head axes down to [B, L]."""

                while values.dim() > position_weight.dim():
                    if extra_mask is not None:
                        values = (values * extra_mask).sum(dim=-1) / extra_mask.sum(
                            dim=-1
                        ).clamp_min(1.0)
                        extra_mask = None
                    else:
                        values = values.mean(dim=-1)
                return values

            def soft_bce(student_logits, teacher_logits, extra_mask=None):
                """Soft-label BCE, plus the KL it implies.

                The cross-entropy the optimizer needs carries the teacher's own
                entropy as an additive floor, so it cannot reach zero and two
                heads' values are not comparable. Subtracting that entropy
                leaves KL(teacher || student), which is zero exactly when the
                student matches -- the form the dashboard plots.
                """

                target = torch.sigmoid(teacher_logits.float())
                bce = F.binary_cross_entropy_with_logits(
                    student_logits.float(), target, reduction="none"
                )
                entropy = F.binary_cross_entropy_with_logits(
                    teacher_logits.float(), target, reduction="none"
                )
                loss = reduce(collapse(bce, extra_mask))
                kl = reduce(collapse(bce - entropy, extra_mask))
                return loss, kl

            def soft_ce(student_logits, teacher_logits, mask=None):
                s = student_logits.float()
                t = teacher_logits.float()
                if mask is not None:
                    s = s.masked_fill(~mask, -1e9)
                    t = t.masked_fill(~mask, -1e9)
                target = torch.softmax(t / temperature, dim=-1)
                log_target = torch.log_softmax(t / temperature, dim=-1)
                per = -(target * torch.log_softmax(s / temperature, dim=-1)).sum(dim=-1)
                entropy = -(target * log_target).sum(dim=-1)
                scale = temperature**2
                loss = reduce(collapse(per)) * scale
                kl = reduce(collapse(per - entropy)) * scale
                return loss, kl

            heads: list[tuple[str, torch.Tensor, torch.Tensor]] = []
            if student.suit_logits is not None:
                heads.append(("suit", *soft_bce(student.suit_logits, teacher.suit_logits)))
            if student.rank_boundary_logits is not None:
                heads.append(
                    (
                        "rank_boundary",
                        *soft_bce(
                            student.rank_boundary_logits,
                            teacher.rank_boundary_logits,
                        ),
                    )
                )
            if student.bid_hit_logits is not None:
                heads.append(
                    ("bid_hit", *soft_bce(student.bid_hit_logits, teacher.bid_hit_logits))
                )
            if student.trick_logits is not None:
                trick_masks = torch.from_numpy(group.trick_masks[start:stop]).to(device)
                while trick_masks.dim() < student.trick_logits.dim():
                    trick_masks = trick_masks.unsqueeze(1)
                heads.append(
                    (
                        "trick",
                        *soft_ce(
                            student.trick_logits,
                            teacher.trick_logits,
                            trick_masks.expand_as(student.trick_logits),
                        ),
                    )
                )
            if student.next_winner_logits is not None:
                seat_active = torch.arange(
                    self.model.config.max_players, device=device
                )[None, None, :] < torch.full(
                    (stop - start, 1, 1),
                    group.num_players,
                    device=device,
                    dtype=torch.long,
                )
                heads.append(
                    (
                        "next_winner",
                        *soft_ce(
                            student.next_winner_logits,
                            teacher.next_winner_logits,
                            seat_active.expand_as(student.next_winner_logits),
                        ),
                    )
                )
            for name, head_loss, head_kl in heads:
                aux_loss = aux_loss + head_loss
                terms[f"auxkl_{name}"] += float(head_kl.detach())
                terms["auxkl_n"] += 1.0
            loss = loss + train.distill_aux_coef * aux_loss
            terms["loss_aux"] += float(aux_loss.detach())

        return loss, terms

    def _record_distill_stats(self, stats: SeqUpdateStats, totals) -> None:
        """Turn weighted sums into the per-decision means the dashboard reads."""

        def mean(numerator: str, denominator: str) -> float:
            count = totals.get(denominator, 0.0)
            return float(totals.get(numerator, 0.0) / count) if count else 0.0

        n_bid = totals.get("n_bid", 0.0)
        n_play = totals.get("n_play", 0.0)
        stats.distill_kl_bid = mean("kl_bid", "n_bid")
        stats.distill_kl_play = mean("kl_play", "n_play")
        stats.distill_agree_bid = mean("agree_bid", "n_bid")
        stats.distill_agree_play = mean("agree_play", "n_play")
        stats.distill_top1_mass_bid = mean("top1_bid", "n_bid")
        stats.distill_top1_mass_play = mean("top1_play", "n_play")
        stats.distill_teacher_entropy_bid = mean("tent_bid", "n_bid")
        stats.distill_teacher_entropy_play = mean("tent_play", "n_play")
        stats.distill_student_entropy_bid = mean("sent_bid", "n_bid")
        stats.distill_student_entropy_play = mean("sent_play", "n_play")
        total_rows = n_bid + n_play
        if total_rows:
            stats.distill_kl = (
                totals.get("kl_bid", 0.0) + totals.get("kl_play", 0.0)
            ) / total_rows
            stats.distill_agree = (
                totals.get("agree_bid", 0.0) + totals.get("agree_play", 0.0)
            ) / total_rows
        # No division by chunk count here, deliberately. ``position_weight``
        # is normalized to sum to 1 across the whole update, so a single
        # chunk's weights sum to far less than 1 and the clamp_min(1.0) in
        # reduce() leaves each chunk contributing a weighted *sum*. Those sums
        # add up to the weighted mean over the update exactly once -- dividing
        # again by the number of chunks would under-report by that factor,
        # which is what made every head look near-zero at initialization.
        for head in (
            "suit",
            "rank_boundary",
            "trick",
            "bid_hit",
            "next_winner",
        ):
            setattr(
                stats,
                f"distill_aux_kl_{head}",
                float(totals.get(f"auxkl_{head}", 0.0)),
            )
        stats.distill_aux_entropy_floor = max(
            float(totals.get("loss_aux", 0.0))
            - sum(
                float(totals.get(f"auxkl_{head}", 0.0))
                for head in ("suit", "rank_boundary", "trick", "bid_hit", "next_winner")
            ),
            0.0,
        )
        stats.distill_loss_policy = float(totals.get("loss_policy", 0.0))
        stats.distill_loss_value = float(totals.get("loss_value", 0.0))
        stats.distill_loss_aux = float(totals.get("loss_aux", 0.0))
        # Reuse the shared loss columns so existing tooling still resolves.
        stats.loss_policy = stats.distill_loss_policy
        stats.loss_value = stats.distill_loss_value
        weight = totals.get("value_w", 0.0)
        if weight:
            stats.distill_value_rmse = float(
                math.sqrt(max(totals.get("value_sq", 0.0) / weight, 0.0))
            )
            stats.value_rmse = stats.distill_value_rmse
            mean_s = totals.get("v_s", 0.0) / weight
            mean_t = totals.get("v_t", 0.0) / weight
            var_s = max(totals.get("v_ss", 0.0) / weight - mean_s * mean_s, 0.0)
            var_t = max(totals.get("v_tt", 0.0) / weight - mean_t * mean_t, 0.0)
            covariance = totals.get("v_st", 0.0) / weight - mean_s * mean_t
            if var_s > 0 and var_t > 0:
                stats.distill_value_correlation = float(
                    covariance / math.sqrt(var_s * var_t)
                )
            stats.value_prediction_std = float(math.sqrt(var_s))
        stats.entropy_bid_normalized = stats.distill_student_entropy_bid
        stats.entropy_play_normalized = stats.distill_student_entropy_play

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        league = []
        for snap in self.league.snapshots:
            snapshot_path = Path(snap.path)
            if snapshot_path.is_absolute():
                stored_path = os.path.relpath(snapshot_path, path.parent)
            else:
                stored_path = os.path.relpath(
                    (Path.cwd() / snapshot_path).resolve(),
                    path.parent.resolve(),
                )
            league.append((snap.snapshot_id, stored_path, snap.iteration))

        collector_state = {
            "peak_rows": self.collector._peak_rows,
            "rows_per_deal": self.collector._rows_per_deal,
            "seat_cursor": self.collector._seat_cursor,
            "policy_cursor": self.collector._policy_cursor,
        }
        payload = {
            "checkpoint_format_version": 1,
            "schema_version": SEQ_SCHEMA_VERSION,
            "model_format_version": SEQ_MODEL_FORMAT_VERSION,
            "iteration": self.iteration,
            "optimizer_steps": self.optimizer_steps,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": asdict(self.model.config),
            "training_config": asdict(self.train),
            "resolved_config": copy.deepcopy(self.resolved_config),
            "rules_fingerprint": rules_fingerprint(),
            "league": league,
            "trainer_rng_state": self.rng.getstate(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "collector_state": collector_state,
            "opponent_curriculum_state": {
                "phase": self.opponent_phase,
                "heuristic_eval_win_streak": self.heuristic_eval_win_streak,
                "last_heuristic_eval_iteration": (
                    self.last_heuristic_eval_iteration
                ),
            },
        }
        if self.train.policy_objective == "ppo" or len(self.models) > 1:
            payload["actor_state_dicts"] = [
                model.state_dict() for model in self.models
            ]
        if self.critic is not None and self.critic_optimizer is not None:
            payload["critic_state_dict"] = self.critic.state_dict()
            payload["critic_optimizer_state_dict"] = (
                self.critic_optimizer.state_dict()
            )
            payload["entropy_log_alpha"] = {
                phase: parameter.detach().cpu()
                for phase, parameter in self.log_entropy_alpha.items()
            }
            if self.entropy_optimizer is not None:
                payload["entropy_optimizer_state_dict"] = (
                    self.entropy_optimizer.state_dict()
                )
        if torch.cuda.is_available():
            payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        if hasattr(torch, "mps") and hasattr(torch.mps, "get_rng_state"):
            try:
                payload["mps_rng_state"] = torch.mps.get_rng_state()
            except RuntimeError:
                pass

        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            torch.save(payload, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            check = torch.load(
                temporary,
                map_location="cpu",
                weights_only=False,
            )
            if (
                check.get("schema_version") != SEQ_SCHEMA_VERSION
                or check.get("iteration") != self.iteration
                or check.get("rules_fingerprint") != rules_fingerprint()
            ):
                raise RuntimeError("Checkpoint validation failed.")
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        allow_training_config_mismatch: bool = False,
    ) -> None:
        path = Path(path)
        requested_resolved_config = self.resolved_config
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("checkpoint_format_version") != 1:
            raise ValueError("Checkpoint format mismatch.")
        if payload.get("schema_version") != SEQ_SCHEMA_VERSION:
            raise ValueError("Checkpoint schema mismatch.")
        model_format = int(payload.get("model_format_version", 1))
        if model_format > SEQ_MODEL_FORMAT_VERSION:
            raise ValueError(
                "Checkpoint model format is newer than this code: "
                f"{model_format} > {SEQ_MODEL_FORMAT_VERSION}."
            )
        if payload.get("rules_fingerprint") != rules_fingerprint():
            raise ValueError("Checkpoint rules fingerprint mismatch.")
        checkpoint_model_config = payload.get("model_config")
        current_model_config = asdict(self.model.config)
        additive_config_fields = {
            "per_layer_embedding_dim",
            "value_embedding_mode",
            "embedding_rms_norm",
        }
        comparable_current = {
            key: value
            for key, value in current_model_config.items()
            if key not in additive_config_fields
        }
        comparable_checkpoint = {
            key: value
            for key, value in checkpoint_model_config.items()
            if key not in additive_config_fields
        } if isinstance(checkpoint_model_config, dict) else None
        additive_mismatch = isinstance(checkpoint_model_config, dict) and any(
            field in checkpoint_model_config
            and checkpoint_model_config[field] != current_model_config[field]
            for field in additive_config_fields
        )
        if comparable_checkpoint != comparable_current or additive_mismatch:
            raise ValueError("Checkpoint model configuration mismatch.")
        if not allow_training_config_mismatch:
            checkpoint_training = payload.get("training_config")
            current_training = asdict(self.train)
            if isinstance(checkpoint_training, dict):
                for field in _ADDITIVE_TRAINING_FIELDS:
                    if field not in checkpoint_training:
                        current_training.pop(field, None)
            if checkpoint_training != current_training:
                raise ValueError("Checkpoint training configuration mismatch.")
        checkpoint_config = payload.get("resolved_config")
        if (
            not allow_training_config_mismatch
            and self.resolved_config is not None
            and checkpoint_config != self.resolved_config
        ):
            raise ValueError("Checkpoint resolved configuration mismatch.")
        self.resolved_config = (
            requested_resolved_config
            if allow_training_config_mismatch
            else checkpoint_config
        )
        actor_states = payload.get("actor_state_dicts")
        exact_actor_pool = (
            isinstance(actor_states, list) and len(actor_states) == len(self.models)
        )
        if exact_actor_pool:
            missing_model_keys: set[str] = set()
            for model, state in zip(self.models, actor_states):
                missing_model_keys.update(load_seq_model_state_dict(model, state))
            optimizer_state = self._migrate_optimizer_state(
                payload["optimizer_state_dict"],
                missing_model_keys=frozenset(missing_model_keys),
            )
            self.optimizer.load_state_dict(optimizer_state)
        else:
            primary_state = payload["model_state_dict"]
            missing_model_keys = load_seq_model_state_dict(
                self.model, primary_state
            )
            for model in self.models[1:]:
                load_seq_model_state_dict(model, self.model.state_dict())
            # A one-actor legacy optimizer has the same parameter topology and
            # can retain its moments. Changing the number of actors changes the
            # parameter set, so the new pool intentionally starts fresh Adam
            # state rather than fabricating moments for copied policies.
            source_actor_count = len(actor_states) if isinstance(actor_states, list) else 1
            if len(self.models) == 1 and source_actor_count == 1:
                optimizer_state = self._migrate_optimizer_state(
                    payload["optimizer_state_dict"],
                    missing_model_keys=missing_model_keys,
                )
                self.optimizer.load_state_dict(optimizer_state)

        # Keys the load had to synthesize because the checkpoint predates them.
        # Callers use this to tell an additive no-op apart from a key like the
        # embedding norm gain, which is meaningless until it is calibrated.
        self.migrated_model_keys = frozenset(missing_model_keys)

        if self.critic is not None and self.critic_optimizer is not None:
            source_training = payload.get("training_config", {})
            source_critic_mode = (
                source_training.get("ppo_critic_mode")
                if isinstance(source_training, dict)
                else None
            )
            compatible_critic = source_critic_mode == self.train.ppo_critic_mode
            if "critic_state_dict" in payload and compatible_critic:
                critic_result = self.critic.load_state_dict(
                    payload["critic_state_dict"], strict=False
                )
                rank_boundary_critic_keys = {
                    f"backbone.{key}" for key in RANK_BOUNDARY_HEAD_KEYS
                }
                additive_critic_keys = {
                    f"backbone.{key}" for key in ADDITIVE_CONDITIONING_KEYS
                } & set(self.critic.state_dict())
                additive_critic_families = (
                    {
                        "backbone.bid_pressure_embedding.weight",
                    } & additive_critic_keys,
                    {
                        key for key in additive_critic_keys
                        if key.startswith("backbone.per_layer_")
                        and key.removeprefix("backbone.")
                        not in NA_PATTERN_EMBEDDING_KEYS
                    },
                    {
                        key for key in additive_critic_keys
                        if key.startswith("backbone.value_")
                        and key.removeprefix("backbone.")
                        not in NA_PATTERN_EMBEDDING_KEYS
                    },
                    {"backbone.na_pattern_embedding.weight"}
                    & additive_critic_keys,
                    {"backbone.per_layer_na_pattern_embedding.weight"}
                    & additive_critic_keys,
                    {"backbone.value_na_pattern_embedding.weight"}
                    & additive_critic_keys,
                )
                embedding_norm_critic_keys = {
                    f"backbone.{key}" for key in EMBEDDING_NORM_KEYS
                } & set(self.critic.state_dict())
                # The critic's backbone carries a frozen copy of the actor's
                # per-seat head. It is never trained here -- the critic reads
                # values from its own player_value_head -- so a checkpoint
                # written before the actor had one is simply missing it.
                player_value_critic_keys = {
                    f"backbone.{key}" for key in PLAYER_VALUE_HEAD_KEYS
                } & set(self.critic.state_dict())
                next_winner_critic_keys = {
                    f"backbone.{key}" for key in NEXT_WINNER_HEAD_KEYS
                } & set(self.critic.state_dict())
                allowed_critic_missing = (
                    rank_boundary_critic_keys
                    | set(ORACLE_CARD_ORDER_HEAD_KEYS)
                    | additive_critic_keys
                    | embedding_norm_critic_keys
                    | player_value_critic_keys
                    | next_winner_critic_keys
                )
                critic_missing = set(critic_result.missing_keys)
                complete_missing_families = all(
                    not (critic_missing & family)
                    or (critic_missing & family) == family
                    for family in (
                        rank_boundary_critic_keys,
                        set(ORACLE_CARD_ORDER_HEAD_KEYS),
                        player_value_critic_keys,
                        next_winner_critic_keys,
                        *additive_critic_families,
                    )
                )
                if (
                    not critic_missing <= allowed_critic_missing
                    or not complete_missing_families
                    or critic_result.unexpected_keys
                ):
                    raise RuntimeError(
                        "Critic checkpoint state mismatch: "
                        f"missing={sorted(critic_missing)}, "
                        f"unexpected={sorted(critic_result.unexpected_keys)}"
                    )
                if critic_missing & {
                    f"backbone.{key}" for key in NA_PATTERN_EMBEDDING_KEYS
                }:
                    self.critic.backbone.initialize_na_pattern_embeddings()
                if critic_missing & embedding_norm_critic_keys:
                    # Gain 1.0 is a placeholder; the critic backbone is
                    # calibrated alongside the actors before training starts.
                    with torch.no_grad():
                        self.critic.backbone.embedding_norm.weight.fill_(1.0)
                    missing_model_keys = set(missing_model_keys) | set(
                        EMBEDDING_NORM_KEYS
                    )
                    self.migrated_model_keys = frozenset(missing_model_keys)
                if "critic_optimizer_state_dict" in payload:
                    try:
                        critic_optimizer_state = payload[
                            "critic_optimizer_state_dict"
                        ]
                        if critic_missing:
                            critic_optimizer_state = self._migrate_named_optimizer_parameters(
                                critic_optimizer_state,
                                self.critic_optimizer,
                                self.critic,
                                missing_keys=critic_missing,
                            )
                        self.critic_optimizer.load_state_dict(
                            critic_optimizer_state
                        )
                    except ValueError:
                        # Checkpoints written before the oracle trick-count
                        # readout became trainable have no Adam slots for that
                        # head. Keep all critic weights, but start fresh critic
                        # moments rather than inventing incompatible state.
                        pass
            else:
                # An actor-only checkpoint, or a checkpoint from a different
                # critic topology, begins from the loaded actor without
                # fabricating incompatible critic Adam moments.
                if isinstance(self.critic, SeqPPOOracleCritic):
                    self.critic.initialize_from_actor(self.model)
                else:
                    self.critic.backbone.load_state_dict(self.model.state_dict())
            # The adaptive temperatures are dual/controller state, not learned
            # policy state, so they carry over whenever the controller
            # mechanics match. The target level is what the controller drives
            # toward rather than part of its state: a converged multiplier is
            # a strictly better warm start for a new level than the cold
            # ppo_entropy_coef, and the controller closes the remaining gap at
            # its own learning rate. Resetting on any target change turned the
            # entropy constraint off for thousands of iterations after a nudge
            # of a few hundredths, which is the opposite of what retargeting
            # is asking for.
            entropy_fields = (
                "ppo_entropy_mode",
                "ppo_entropy_coef",
                "ppo_entropy_learning_rate",
            )
            compatible_entropy = isinstance(source_training, dict) and all(
                source_training.get(field) == getattr(self.train, field)
                for field in entropy_fields
            )
            if compatible_entropy:
                for phase, value in payload.get("entropy_log_alpha", {}).items():
                    if phase in self.log_entropy_alpha:
                        self.log_entropy_alpha[phase].data.copy_(
                            torch.as_tensor(value, device=self.device)
                        )
                if (
                    self.entropy_optimizer is not None
                    and "entropy_optimizer_state_dict" in payload
                ):
                    self.entropy_optimizer.load_state_dict(
                        payload["entropy_optimizer_state_dict"]
                    )
        self.iteration = int(payload.get("iteration", 0))
        self.optimizer_steps = int(payload.get("optimizer_steps", 0))
        curriculum = payload.get("opponent_curriculum_state", {})
        if self.train.rollout.opponent_mode == "heuristic_then_historical":
            phase = curriculum.get("phase")
            if phase in ("heuristic", "historical"):
                self.opponent_phase = phase
            self.heuristic_eval_win_streak = int(
                curriculum.get("heuristic_eval_win_streak", 0)
            )
            self.last_heuristic_eval_iteration = int(
                curriculum.get("last_heuristic_eval_iteration", 0)
            )
        else:
            self.opponent_phase = self.train.rollout.initial_opponent
            self.heuristic_eval_win_streak = 0
        self.league.snapshots.clear()
        self.league._policies.clear()
        for snapshot_id, snap_path, iteration in payload.get("league", []):
            resolved = Path(snap_path)
            if not resolved.is_absolute():
                resolved = (path.parent / resolved).resolve()
            if resolved.exists():
                self.league.add(snapshot_id, str(resolved), iteration)

        if "trainer_rng_state" in payload:
            self.rng.setstate(payload["trainer_rng_state"])
        if "python_rng_state" in payload:
            random.setstate(payload["python_rng_state"])
        if "numpy_rng_state" in payload:
            np.random.set_state(payload["numpy_rng_state"])
        if "torch_rng_state" in payload:
            torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state" in payload:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
        if (
            "mps_rng_state" in payload
            and hasattr(torch, "mps")
            and hasattr(torch.mps, "set_rng_state")
        ):
            try:
                torch.mps.set_rng_state(payload["mps_rng_state"])
            except RuntimeError:
                pass

        collector_state = payload.get("collector_state", {})
        self.collector._peak_rows = dict(collector_state.get("peak_rows", {}))
        self.collector._rows_per_deal = dict(collector_state.get("rows_per_deal", {}))
        self.collector._seat_cursor = dict(collector_state.get("seat_cursor", {}))
        self.collector._policy_cursor = int(
            collector_state.get("policy_cursor", 0)
        )
        self._configure_ppo_optimizer_execution()

    def _configure_ppo_optimizer_execution(self) -> None:
        """Select the runtime Adam kernel without resetting saved moments.

        Old checkpoints store unfused groups and CPU step counters. MPS fused
        Adam needs device counters; CPU preparation/evaluation must still be
        able to load its checkpoints with ordinary Adam. Kernel selection is
        therefore a runtime detail, independent of the checkpoint's device.
        """

        if self.train.policy_objective != "ppo":
            return
        fused = self.device.type == "mps"
        counter_device = self.device if fused else torch.device("cpu")
        for optimizer in (self.optimizer, self.critic_optimizer):
            if optimizer is None:
                continue
            for group in optimizer.param_groups:
                group["fused"] = fused
                group["foreach"] = False if fused else None
            for state in optimizer.state.values():
                if "step" in state:
                    state["step"] = state["step"].to(counter_device)

    @staticmethod
    def _migrate_named_optimizer_parameters(
        state: dict,
        optimizer: torch.optim.Optimizer,
        module: nn.Module,
        *,
        missing_keys: set[str],
    ) -> dict:
        """Map one-group optimizer state across inserted named parameters."""

        migrated = copy.deepcopy(state)
        source_groups = migrated.get("param_groups", [])
        target_groups = optimizer.state_dict().get("param_groups", [])
        if len(source_groups) != 1 or len(target_groups) != 1:
            raise ValueError("Named optimizer migration requires one group.")
        current_names = [
            name for name, parameter in module.named_parameters()
            if parameter.requires_grad
        ]
        source_names = [name for name in current_names if name not in missing_keys]
        source_ids = list(source_groups[0]["params"])
        target_ids = list(target_groups[0]["params"])
        if len(source_ids) != len(source_names) or len(target_ids) != len(current_names):
            raise ValueError("Checkpoint optimizer names do not match its parameters.")
        target_by_name = dict(zip(current_names, target_ids))
        identifier_map = {
            source_id: target_by_name[name]
            for source_id, name in zip(source_ids, source_names)
        }
        if set(migrated.get("state", {})) - set(identifier_map):
            raise ValueError("Checkpoint optimizer contains unknown parameters.")
        migrated["state"] = {
            identifier_map[source]: value
            for source, value in migrated.get("state", {}).items()
        }
        group = copy.deepcopy(source_groups[0])
        group["params"] = target_ids
        migrated["param_groups"] = [group]
        return migrated

    @staticmethod
    def _migrate_appended_optimizer_parameters(
        state: dict,
        optimizer: torch.optim.Optimizer,
        *,
        added_parameters: int,
    ) -> dict:
        """Preserve Adam state when trainable parameters are appended.

        Oracle card-order weight/bias are deliberately registered after all
        existing trainable critic parameters, so the old topology is an exact
        prefix. Existing moments retain their parameter identity; the new
        suffix starts with ordinary empty Adam state.
        """

        migrated = copy.deepcopy(state)
        source_groups = migrated.get("param_groups", [])
        target_groups = optimizer.state_dict().get("param_groups", [])
        if len(source_groups) != 1 or len(target_groups) != 1:
            raise ValueError(
                "Appended-parameter optimizer migration requires one group."
            )
        source_ids = list(source_groups[0]["params"])
        target_ids = list(target_groups[0]["params"])
        if len(source_ids) + added_parameters != len(target_ids):
            raise ValueError(
                "Checkpoint optimizer is not a prefix of the current one."
            )
        identifier_map = dict(zip(source_ids, target_ids[: len(source_ids)]))
        unknown = set(migrated.get("state", {})) - set(identifier_map)
        if unknown:
            raise ValueError(
                "Checkpoint optimizer contains state for unknown parameters."
            )
        migrated["state"] = {
            identifier_map[source]: value
            for source, value in migrated.get("state", {}).items()
        }
        group = copy.deepcopy(source_groups[0])
        group["params"] = target_ids
        migrated["param_groups"] = [group]
        return migrated

    def _migrate_optimizer_state(
        self,
        state: dict,
        *,
        missing_model_keys: frozenset[str],
    ) -> dict:
        """Map old Adam parameter IDs onto the current optimizer topology.

        Format-1 models can lack the final two core parameters (rank/suit card
        output embeddings), and format-2 models lack the final two auxiliary
        parameters (rank-boundary weight/bias). Existing Adam moments retain
        their identities and only the new parameters start without state. The
        same mapping permits lossless transitions between split and unified
        PPO actor groups. No other parameter-count mismatch is accepted.
        """

        migrated = copy.deepcopy(state)
        source_groups = migrated.get("param_groups", [])
        if len(source_groups) not in (1, 2):
            raise ValueError(
                "Checkpoint optimizer must contain one or two parameter groups."
            )

        templates = self.optimizer.state_dict()["param_groups"]
        if len(templates) not in (1, 2):
            raise ValueError("Current optimizer must contain one or two groups.")
        core_keys: list[tuple[int, str]] = []
        auxiliary_keys: list[tuple[int, str]] = []
        for actor_index, actor in enumerate(self.models):
            for name, _ in actor.named_parameters():
                target = (
                    auxiliary_keys
                    if name.startswith(_AUXILIARY_HEAD_PREFIXES)
                    else core_keys
                )
                target.append((actor_index, name))
        source_core_keys = [
            key for key in core_keys if key[1] not in missing_model_keys
        ]
        source_auxiliary_keys = [
            key for key in auxiliary_keys if key[1] not in missing_model_keys
        ]
        source_core = len(source_core_keys)
        source_auxiliary = len(source_auxiliary_keys)

        if len(source_groups) == 1:
            flat = list(source_groups[0]["params"])
            if len(flat) != source_core + source_auxiliary:
                raise ValueError(
                    "Checkpoint optimizer parameter count does not match the model."
                )
            source_parameter_groups = (
                flat[:source_core],
                flat[source_core:],
            )
            metadata_groups = (source_groups[0], source_groups[0])
        else:
            source_parameter_groups = tuple(
                list(group["params"]) for group in source_groups
            )
            if [len(parameters) for parameters in source_parameter_groups] != [
                source_core,
                source_auxiliary,
            ]:
                raise ValueError(
                    "Checkpoint optimizer group sizes do not match the model."
                )
            metadata_groups = tuple(source_groups)

        if len(templates) == 1:
            target_flat = list(templates[0]["params"])
            current_target_groups = (
                target_flat[: len(core_keys)],
                target_flat[len(core_keys) :],
            )
        else:
            current_target_groups = (
                list(templates[0]["params"]),
                list(templates[1]["params"]),
            )
        target_ids_by_key = {
            key: parameter_id
            for keys, parameter_ids in zip(
                (core_keys, auxiliary_keys), current_target_groups
            )
            for key, parameter_id in zip(keys, parameter_ids)
        }
        target_parameter_groups = (
            [target_ids_by_key[key] for key in source_core_keys],
            [target_ids_by_key[key] for key in source_auxiliary_keys],
        )
        identifier_map: dict[int, int] = {}
        for source_parameters, target_parameters in zip(
            source_parameter_groups,
            target_parameter_groups,
        ):
            identifier_map.update(zip(source_parameters, target_parameters))

        unknown_state = set(migrated.get("state", {})) - set(identifier_map)
        if unknown_state:
            raise ValueError(
                "Checkpoint optimizer contains state for unknown parameters."
            )
        migrated["state"] = {
            identifier_map[source]: value
            for source, value in migrated.get("state", {}).items()
        }

        groups = []
        target_metadata = (
            (metadata_groups[0],)
            if len(templates) == 1
            else metadata_groups
        )
        for metadata, template in zip(target_metadata, templates):
            group = copy.deepcopy(metadata)
            # Include every current parameter in the serialized group. The
            # two new output embeddings simply have no entry in ``state`` and
            # Adam will initialize their moments on the first kept update.
            group["params"] = list(template["params"])
            group["kl_sensitive"] = template["kl_sensitive"]
            groups.append(group)
        migrated["param_groups"] = groups
        return migrated

    def maybe_snapshot(self, checkpoint_dir: str | Path) -> Optional[str]:
        if self.iteration % self.train.snapshot_every != 0:
            return None
        directory = Path(checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"seq_v6_iter_{self.iteration:06d}.pt"
        self.save_checkpoint(path)
        self.league.add(f"iter_{self.iteration}", str(path), self.iteration)
        return str(path)


def _rows_by_chunk(group: SeqTrainingGroup, chunks) -> list:
    """Split the group's policy rows per microbatch chunk, reindexing rows."""

    fields = (
        "position",
        "old_probs_full",
        "candidates",
        "q_values",
        "baseline",
        "inclusion",
        "weight",
    )
    out = []
    for start, stop in list(chunks):
        chunk: dict[str, dict[str, list]] = {}
        for phase_key, rows in group.policy.items():
            selected: dict[str, list] = {"seq_index": []}
            for name in fields:
                selected[name] = []
            for i, seq_index in enumerate(rows.seq_index):
                if start <= seq_index < stop:
                    selected["seq_index"].append(seq_index - start)
                    for name in fields:
                        selected[name].append(getattr(rows, name)[i])
            chunk[phase_key] = selected
        out.append(chunk)
    return out
