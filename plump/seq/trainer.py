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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from plump.rewards import compute_relative_rewards
from plump.rounds import rules_fingerprint

from .config import (
    NEXT_BID,
    SEQ_SCHEMA_VERSION,
    SeqModelConfig,
    SeqTrainingConfig,
)
from .model import (
    SEQ_MODEL_FORMAT_VERSION,
    STRUCTURED_CARD_OUTPUT_KEYS,
    SeqPlumpModel,
    SeqPPOCritic,
    SeqPPOOracleCritic,
    load_seq_model_state_dict,
)
from .policy import SeqLeague, best_seq_device
from .precision import autocast_context
from .ppo import (
    PPO_BELIEF_STAGES,
    PPOCriticGroup,
    PPOTrainingGroup,
    build_ppo_training_batch,
    normalize_ppo_advantages,
    ppo_clipped_terms,
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
    "trick_count_head.",
    "suit_presence_head.",
    "bid_hit_head.",
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
    suit_targets: np.ndarray
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
    loss_trick: float = 0.0
    loss_oracle_trick: float = 0.0
    loss_bid_hit: float = 0.0
    suit_accuracy_10c_0: float = 0.0
    suit_accuracy_10c_4: float = 0.0
    suit_accuracy_10c_8: float = 0.0
    trick_accuracy_10c_0: float = 0.0
    trick_accuracy_10c_4: float = 0.0
    trick_accuracy_10c_8: float = 0.0
    oracle_trick_accuracy: float = 0.0
    entropy: float = 0.0
    entropy_bid_normalized: float = 0.0
    entropy_play_normalized: float = 0.0
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
    core_grad_norm: float = 0.0
    auxiliary_grad_norm: float = 0.0
    critic_grad_norm: float = 0.0
    # Oracle-critic telemetry. Acting-seat metrics above are the baseline PPO
    # consumes; these cover every active output head and optimization progress
    # across the repeated critic epochs.
    critic_all_player_rmse: float = 0.0
    critic_all_player_correlation: float = 0.0
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
    owned_per_tree = {
        id(tree): sum(
            sum(leaf.position_reach_weights().values()) for leaf in tree.leaves
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
        suit_targets = np.full(
            (batch, length, model_config.belief_opponents, 4), IGNORE_LABEL, np.int64
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
            )
            owned[row, leaf.owned_from :] = True
            for position, value in leaf.value_targets().items():
                value_targets[row, position] = value
            for position, reach in leaf.position_reach_weights().items():
                position_weight[row, position] = seq_weight_for(tree) * reach
            if train_config.value_positions == "all":
                value_weight[row] = position_weight[row]
            trick_targets[row] = arrays.trick_targets
            trick_masks[row] = arrays.trick_masks
            suit_targets[row] = arrays.suit_targets
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
                suit_targets=suit_targets,
                bid_hit_targets=bid_hit_targets,
                policy=policy,
            )
        )
    return groups


# --------------------------------------------------------------------- #
# Trainer                                                                #
# --------------------------------------------------------------------- #


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
        self.optimizer = torch.optim.Adam(
            (
                {
                    "params": core_parameters,
                    # Shared trunk plus action heads: these parameters can
                    # move the policy and must follow KL backtracking.
                    "kl_sensitive": True,
                    "lr": train_config.core_lr,
                },
                {
                    "params": auxiliary_parameters,
                    # Readout-only heads cannot change bid/card logits.
                    "kl_sensitive": False,
                    "lr": train_config.auxiliary_lr,
                },
            ),
            lr=train_config.learning_rate,
            betas=train_config.adam_betas,
        )
        self.collector = SeqRolloutCollector(
            self.model,
            train_config,
            device=self.device,
            trainable_models=self.models,
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
        self.league = SeqLeague(
            train_config.league_max_snapshots, train_config.league_min_iteration
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

    def collect(self) -> tuple[list[SeqTree], SeqRolloutSummary]:
        for model in self.models:
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
            stats.core_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self._core_parameters, self.train.max_grad_norm
                )
            )
            stats.auxiliary_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self._auxiliary_parameters, self.train.max_grad_norm
                )
            )
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
                        if attempt > 0 and attempt + 1 < attempts
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
                        kl_p99 > self.train.policy_kl_p99_cap
                    )
                stats.policy_kl = policy_kl
                stats.policy_kl_p95 = kl_p95
                stats.policy_kl_p99 = kl_p99
                stats.policy_kl_max = kl_max
                if stats.policy_rows == 0 or (
                    policy_kl <= self.train.policy_kl_cap
                    and kl_p99 <= self.train.policy_kl_p99_cap
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
        self._release_device_memory()
        batch = build_ppo_training_batch(trees, self.model.config, self.train)
        groups = batch.policy_groups
        stats = SeqUpdateStats()
        stats.build_sec = time.perf_counter() - started
        stats.policy_rows = sum(
            len(rows.weight) for group in groups for rows in group.policy.values()
        )
        stats.unbranched_rows = stats.policy_rows
        stats.value_rows = stats.policy_rows
        stats.positions = int(
            sum(group.tokens.shape[0] * group.tokens.shape[1] for group in groups)
        )

        advantage_mean, advantage_std = self._prepare_ppo_advantages(
            groups, batch.critic_groups, stats
        )
        stats.advantage_mean = advantage_mean
        stats.advantage_std = advantage_std
        if self.train.ppo_advantage_normalize:
            normalize_ppo_advantages(groups)
        stats.peak_update_device_bytes = max(
            stats.peak_update_device_bytes, self._device_allocated_bytes()
        )

        for actor_epoch in range(self.train.epochs):
            actor_snapshot = (
                [copy.deepcopy(model.state_dict()) for model in self.models],
                copy.deepcopy(self.optimizer.state_dict()),
            )
            for model in self.models:
                model.train()
            self.optimizer.zero_grad(set_to_none=True)
            epoch = self._ppo_actor_backward(groups)
            stats.loss_policy += epoch["policy_loss"]
            stats.loss_suit += epoch["suit_loss"]
            stats.loss_trick += epoch["trick_loss"]
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
                        f"trick_accuracy_10c_{stage}",
                        epoch[f"trick_accuracy_10c_{stage}"],
                    )
            stats.policy_logit_shift = epoch["logit_shift"]
            stats.core_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self._core_parameters, self.train.max_grad_norm
                )
            )
            stats.auxiliary_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self._auxiliary_parameters, self.train.max_grad_norm
                )
            )
            self._apply_warmup_lr()
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
                        kl_p99 > self.train.policy_kl_p99_cap
                    )
                stats.policy_kl = policy_kl
                stats.policy_kl_p95 = kl_p95
                stats.policy_kl_p99 = kl_p99
                stats.policy_kl_max = kl_max
                if policy_kl <= self.train.policy_kl_cap and kl_p99 <= self.train.policy_kl_p99_cap:
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
        """Evaluate each game once and address values by absolute actor seat."""

        assert isinstance(self.critic, SeqPPOOracleCritic)
        self.critic.eval()
        prediction_by_address: dict[tuple[int, int, int, int], float] = {}
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
                        values = self.critic.forward_full(tokens)
                    seq = torch.tensor(
                        rows["seq_index"], device=self.device, dtype=torch.long
                    )
                    pos = torch.tensor(
                        rows["position"], device=self.device, dtype=torch.long
                    )
                    selected_all = (
                        values[seq, pos].float().cpu().numpy()
                    )
                    acting_seat = np.asarray(rows["acting_seat"], dtype=np.int64)
                    selected = selected_all[
                        np.arange(selected_all.shape[0]), acting_seat
                    ]
                    target_all = np.stack(rows["returns"])
                    players = np.asarray(rows["num_players"], dtype=np.int64)
                    active = (
                        np.arange(self.model.config.max_players)[None, :]
                        < players[:, None]
                    )
                    row_weight = np.asarray(rows["weight"], dtype=np.float64)
                    all_player_predictions.append(selected_all[active])
                    all_player_targets.append(target_all[active])
                    all_player_weights.append(
                        np.repeat(row_weight / players, players)
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
                    rows.advantages[index] = target - prediction
                    predictions.append(prediction)
                    targets.append(float(target))
                    weights.append(float(rows.weight[index]))

        return self._ppo_value_statistics(predictions, targets, weights, stats)

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
        advantages = target - prediction
        mean = float(np.dot(weight, advantages) / total)
        variance = float(np.dot(weight, (advantages - mean) ** 2) / total)
        return mean, math.sqrt(max(variance, 1e-12))

    def _entropy_alpha(self, phase: str) -> float:
        if self.train.ppo_entropy_mode == "off":
            return 0.0
        if self.train.ppo_entropy_mode == "fixed":
            return self.train.ppo_entropy_coef
        return float(self.log_entropy_alpha[phase].detach().exp().clamp(max=10.0))

    def _ppo_actor_backward(self, groups: list[PPOTrainingGroup]) -> dict[str, float]:
        totals: dict[str, list[torch.Tensor]] = defaultdict(list)
        for group in groups:
            model = self._ppo_model(group.policy_id)
            chunks = list(self._microbatches(group))
            for (start, stop), rows_by_phase in zip(
                chunks, ppo_rows_by_chunk(group, chunks)
            ):
                tokens = torch.from_numpy(group.tokens[start:stop]).to(self.device)
                with autocast_context(self.device, self.train.precision):
                    hidden = model.forward_hidden(tokens)
                loss = torch.zeros((), device=self.device, dtype=torch.float32)

                position_weight = torch.from_numpy(
                    group.position_weight[start:stop]
                ).to(self.device)
                stage_positions = group.belief_stage_positions[start:stop]

                if self.train.suit_coef > 0:
                    with autocast_context(self.device, self.train.precision):
                        suit_logits = model.suit_presence_head(hidden).view(
                            hidden.shape[0],
                            hidden.shape[1],
                            model.config.belief_opponents,
                            4,
                        )
                    suit_logits = suit_logits.float()
                    suit_targets = torch.from_numpy(
                        group.suit_targets[start:stop]
                    ).to(self.device)
                    suit_labeled = suit_targets != IGNORE_LABEL
                    bce = F.binary_cross_entropy_with_logits(
                        suit_logits,
                        suit_targets.clamp_min(0).float(),
                        reduction="none",
                    )
                    per_position = (
                        (bce * suit_labeled.float()).sum(dim=(2, 3))
                        / suit_labeled.float().sum(dim=(2, 3)).clamp_min(1.0)
                    )
                    suit_loss = (per_position * position_weight).sum()
                    loss = loss + self.train.suit_coef * suit_loss
                    totals["suit_loss"].append(suit_loss.detach())
                    self._record_ppo_belief_stage_accuracy(
                        totals,
                        "suit",
                        suit_logits,
                        suit_targets,
                        None,
                        stage_positions,
                    )

                if self.train.trick_coef > 0:
                    with autocast_context(self.device, self.train.precision):
                        trick_logits = model.trick_count_head(hidden).view(
                            hidden.shape[0],
                            hidden.shape[1],
                            model.config.max_players,
                            model.config.bid_count,
                        )
                    trick_logits = trick_logits.float()
                    trick_targets = torch.from_numpy(
                        group.trick_targets[start:stop]
                    ).to(self.device)
                    trick_masks = torch.from_numpy(
                        group.trick_masks[start:stop]
                    ).to(self.device)
                    masked_logits = trick_logits.masked_fill(~trick_masks, -1e9)
                    log_probs = torch.log_softmax(masked_logits, dim=-1)
                    safe_targets = trick_targets.clamp_min(0)[:, None, :, None].expand(
                        -1, masked_logits.shape[1], -1, 1
                    )
                    gathered = log_probs.gather(3, safe_targets).squeeze(-1)
                    labeled = (
                        (trick_targets != IGNORE_LABEL)[:, None, :]
                        & trick_masks.any(dim=-1)
                    )
                    per_position = (-gathered * labeled.float()).sum(dim=2) / (
                        labeled.float().sum(dim=2).clamp_min(1.0)
                    )
                    trick_loss = (per_position * position_weight).sum()
                    loss = loss + self.train.trick_coef * trick_loss
                    totals["trick_loss"].append(trick_loss.detach())
                    self._record_ppo_belief_stage_accuracy(
                        totals,
                        "trick",
                        masked_logits,
                        trick_targets,
                        trick_masks,
                        stage_positions,
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
                    loss = loss + self.train.policy_coef * policy_loss
                    eligible_weight = weight * terms.entropy_eligible.float()
                    entropy_term = (
                        eligible_weight * terms.normalized_entropy
                    ).sum()
                    alpha = self._entropy_alpha(phase)
                    if alpha:
                        loss = loss - alpha * entropy_term
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
                if loss.requires_grad:
                    loss.backward()

        reduced = {
            key: math.fsum(float(value) for value in values)
            for key, values in totals.items()
        }
        row_weight = max(reduced.get("row_weight", 0.0), 1e-12)
        result = {
            "policy_loss": reduced.get("policy_loss", 0.0),
            "suit_loss": reduced.get("suit_loss", 0.0),
            "trick_loss": reduced.get("trick_loss", 0.0),
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
        return result

    @staticmethod
    def _record_ppo_belief_stage_accuracy(
        totals: dict[str, list[torch.Tensor]],
        head: str,
        logits: torch.Tensor,
        targets: torch.Tensor,
        feasibility: torch.Tensor | None,
        stage_positions: np.ndarray,
    ) -> None:
        """Accumulate micro-accuracy before 1st/5th/9th ten-card plays."""

        device = logits.device
        for stage_index, cards_played in enumerate(PPO_BELIEF_STAGES):
            positions_np = stage_positions[:, stage_index]
            rows_np = np.flatnonzero(positions_np >= 0)
            if rows_np.size == 0:
                continue
            rows = torch.from_numpy(rows_np).to(device=device, dtype=torch.long)
            positions = torch.from_numpy(positions_np[rows_np]).to(
                device=device, dtype=torch.long
            )
            selected_logits = logits[rows, positions]
            if head == "suit":
                selected_targets = targets[rows, positions]
                labeled = selected_targets != IGNORE_LABEL
                predictions = selected_logits >= 0
            else:
                selected_targets = targets[rows]
                assert feasibility is not None
                labeled = (
                    (selected_targets != IGNORE_LABEL)
                    & feasibility[rows, positions].any(dim=-1)
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
        targets = {
            "bid": self.train.ppo_bid_entropy_target,
            "play": self.train.ppo_play_entropy_target,
        }
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
                        actions = torch.tensor(
                            rows["action"], device=self.device, dtype=torch.long
                        )
                        advantages = torch.tensor(
                            rows["advantages"], device=self.device
                        )
                        with autocast_context(self.device, self.train.precision):
                            logits = model.policy_logits(
                                hidden[seq, pos], phase
                            )
                        terms = ppo_clipped_terms(
                            logits,
                            old_probs,
                            actions,
                            advantages,
                            clip_ratio=self.train.ppo_clip_ratio,
                        )
                        divergences.append(terms.divergences.cpu().numpy())
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
        """Fit every active player's return from each learned decision state."""

        assert isinstance(self.critic, SeqPPOOracleCritic)
        assert self.critic_optimizer is not None
        value_terms: list[torch.Tensor] = []
        trick_terms: list[torch.Tensor] = []
        epoch_losses: list[float] = []
        oracle_trick_correct: list[torch.Tensor] = []
        oracle_trick_total: list[torch.Tensor] = []
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
                    with autocast_context(self.device, self.train.precision):
                        if self.train.trick_coef > 0:
                            values, trick_logits = (
                                self.critic.forward_value_and_trick(tokens)
                            )
                        else:
                            values = self.critic.forward_full(tokens)
                    seq = torch.tensor(
                        rows["seq_index"], device=self.device, dtype=torch.long
                    )
                    pos = torch.tensor(
                        rows["position"], device=self.device, dtype=torch.long
                    )
                    targets = torch.from_numpy(np.stack(rows["returns"])).to(
                        self.device
                    )
                    weight = torch.tensor(
                        rows["weight"], device=self.device, dtype=torch.float32
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
                    predictions = values[seq, pos].float()
                    error = (
                        predictions - targets.float()
                    ) / self.train.value_reward_scale
                    # Each state retains its actor-objective weight. Averaging
                    # the output-seat axis supplies all-player supervision
                    # without multiplying larger-player games' critic weight.
                    value_loss = (
                        0.5
                        * weight
                        * (error.square() * active.float()).sum(dim=-1)
                        / players.float()
                    ).sum()
                    loss = value_loss
                    value_terms.append(value_loss.detach())

                    if self.train.trick_coef > 0:
                        trick_targets = torch.from_numpy(
                            group.trick_targets[start:stop]
                        ).to(self.device)
                        trick_masks = torch.from_numpy(
                            group.trick_masks[start:stop]
                        ).to(self.device)
                        position_weight = torch.from_numpy(
                            group.position_weight[start:stop]
                        ).to(self.device)
                        masked_logits = trick_logits.float().masked_fill(
                            ~trick_masks, -1e9
                        )
                        log_probs = torch.log_softmax(masked_logits, dim=-1)
                        safe_targets = trick_targets.clamp_min(0)[
                            :, None, :, None
                        ].expand(-1, masked_logits.shape[1], -1, 1)
                        gathered = log_probs.gather(3, safe_targets).squeeze(-1)
                        labeled = (
                            (trick_targets != IGNORE_LABEL)[:, None, :]
                            & trick_masks.any(dim=-1)
                        )
                        per_position = (
                            (-gathered * labeled.float()).sum(dim=2)
                            / labeled.float().sum(dim=2).clamp_min(1.0)
                        )
                        trick_loss = (per_position * position_weight).sum()
                        loss = loss + self.train.trick_coef * trick_loss
                        trick_terms.append(trick_loss.detach())
                        if critic_epoch == 0:
                            predictions = masked_logits.argmax(dim=-1)
                            oracle_trick_correct.append(
                                ((predictions == trick_targets[:, None, :]) & labeled)
                                .sum()
                                .detach()
                            )
                            oracle_trick_total.append(labeled.sum().detach())
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
        stats.loss_oracle_trick = math.fsum(
            float(value) for value in trick_terms
        )
        correct = math.fsum(float(value) for value in oracle_trick_correct)
        labeled = math.fsum(float(value) for value in oracle_trick_total)
        stats.oracle_trick_accuracy = (
            correct / labeled
            if labeled > 0
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
        stats.loss_trick = totals["trick"]
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
                ("trick", train.trick_coef),
                ("bid_hit", train.bid_hit_coef),
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
        if payload.get("model_config") != asdict(self.model.config):
            raise ValueError("Checkpoint model configuration mismatch.")
        if not allow_training_config_mismatch and payload.get(
            "training_config"
        ) != asdict(self.train):
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
            for model, state in zip(self.models, actor_states):
                load_seq_model_state_dict(model, state)
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        else:
            primary_state = payload["model_state_dict"]
            structured_output_migrated = load_seq_model_state_dict(
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
                    structured_output_migrated=structured_output_migrated,
                )
                self.optimizer.load_state_dict(optimizer_state)

        if self.critic is not None and self.critic_optimizer is not None:
            source_training = payload.get("training_config", {})
            source_critic_mode = (
                source_training.get("ppo_critic_mode")
                if isinstance(source_training, dict)
                else None
            )
            compatible_critic = source_critic_mode == self.train.ppo_critic_mode
            if "critic_state_dict" in payload and compatible_critic:
                self.critic.load_state_dict(payload["critic_state_dict"])
                if "critic_optimizer_state_dict" in payload:
                    try:
                        self.critic_optimizer.load_state_dict(
                            payload["critic_optimizer_state_dict"]
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

    def _migrate_optimizer_state(
        self,
        state: dict,
        *,
        structured_output_migrated: bool,
    ) -> dict:
        """Map old Adam parameter IDs onto the current two-group optimizer.

        Format-1 models lack the final two core parameters: the rank and suit
        card-output embeddings. Their exact-card and auxiliary Adam moments
        must retain their identities while the new rows start with empty Adam
        state. This also retains support for the former single flat group.
        No other parameter-count mismatch is accepted.
        """

        migrated = copy.deepcopy(state)
        source_groups = migrated.get("param_groups", [])
        if len(source_groups) not in (1, 2):
            raise ValueError(
                "Checkpoint optimizer must contain one or two parameter groups."
            )

        templates = self.optimizer.state_dict()["param_groups"]
        current_core = len(templates[0]["params"])
        current_auxiliary = len(templates[1]["params"])
        missing_count = (
            len(STRUCTURED_CARD_OUTPUT_KEYS)
            if structured_output_migrated
            else 0
        )
        source_core = current_core - missing_count
        if source_core < 0:
            raise ValueError("Checkpoint optimizer parameter count is invalid.")

        if len(source_groups) == 1:
            flat = list(source_groups[0]["params"])
            if len(flat) != source_core + current_auxiliary:
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
                current_auxiliary,
            ]:
                raise ValueError(
                    "Checkpoint optimizer group sizes do not match the model."
                )
            metadata_groups = tuple(source_groups)

        target_parameter_groups = (
            list(templates[0]["params"][:source_core]),
            list(templates[1]["params"]),
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
        for metadata, template in zip(
            metadata_groups,
            templates,
        ):
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
