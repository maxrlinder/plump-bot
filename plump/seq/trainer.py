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

from plump.rounds import rules_fingerprint

from .config import (
    NEXT_BID,
    SEQ_SCHEMA_VERSION,
    SeqModelConfig,
    SeqTrainingConfig,
)
from .model import SeqPlumpModel
from .policy import SeqLeague, best_seq_device
from .rollout import SeqRolloutCollector, SeqTree
from .tokens import IGNORE_LABEL, build_replay_arrays, build_seat_tokens

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
    position_weight: np.ndarray  # [B, L] on-policy aux/value weight
    trick_targets: np.ndarray  # [B, max_players]
    trick_masks: np.ndarray
    suit_targets: np.ndarray
    bid_hit_targets: np.ndarray  # [B, max_players]
    policy: dict[str, PolicyRows]


@dataclass
class SeqUpdateStats:
    loss_policy: float = 0.0
    loss_value: float = 0.0
    loss_suit: float = 0.0
    loss_trick: float = 0.0
    loss_bid_hit: float = 0.0
    entropy: float = 0.0
    policy_kl: float = 0.0
    policy_kl_p95: float = 0.0
    policy_kl_p99: float = 0.0
    policy_kl_max: float = 0.0
    rolled_back: bool = False
    backtracks: int = 0
    step_scale: float = 1.0
    policy_rows: int = 0
    # Split of policy_rows by how the row was produced. Diagnostics: both
    # feed the same loss.
    branched_rows: int = 0
    unbranched_rows: int = 0
    positions: int = 0
    update_sec: float = 0.0
    build_sec: float = 0.0


@dataclass
class SeqRolloutSummary:
    trees: int = 0
    leaves: int = 0
    decisions: int = 0
    bid_hit_rate: float = 0.0
    reward_self: float = 0.0
    reward_historical: float = 0.0
    spine_entropy: float = 0.0
    collect_sec: float = 0.0
    forward_rows: int = 0


def summarize_trees(trees: list[SeqTree], collector_stats) -> SeqRolloutSummary:
    summary = SeqRolloutSummary(
        trees=len(trees),
        leaves=sum(tree.leaf_total for tree in trees),
        decisions=sum(tree.decision_total for tree in trees),
        collect_sec=collector_stats.collect_sec,
        forward_rows=collector_stats.forward_rows,
    )
    hits = 0
    players = 0
    rewards: dict[str, list[float]] = defaultdict(list)
    entropies: list[float] = []
    for tree in trees:
        spine = next(leaf for leaf in tree.leaves if leaf.on_policy_spine)
        round_state = spine.env.state.current_round
        bids = {bid.player: bid.value for bid in round_state.bids}
        for player, bid in bids.items():
            players += 1
            hits += int(round_state.tricks_won[player] == bid)
        rewards[tree.arm].append(spine.terminal_value)
        for record in spine.decisions:
            probs = record.old_probs[record.old_probs > 0]
            entropies.append(float(-(probs * np.log(probs)).sum()))
    summary.bid_hit_rate = hits / players if players else 0.0
    summary.reward_self = float(np.mean(rewards["self"])) if rewards["self"] else 0.0
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
        tokens = np.zeros((batch, length, 12), dtype=np.int64)
        owned = np.zeros((batch, length), dtype=bool)
        value_targets = np.zeros((batch, length), dtype=np.float32)
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
            trick_targets[row] = arrays.trick_targets
            trick_masks[row] = arrays.trick_masks
            suit_targets[row] = arrays.suit_targets
            bid_hit_targets[row] = arrays.bid_hit_targets

            for record in leaf.decisions:
                phase_key = "bid" if record.phase == NEXT_BID else "play"
                rows = policy[phase_key]
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
                rows.weight.append(
                    policy_weight_for(tree)
                    * record.reach_weight
                    * depth_scale.get(id(record), 1.0)
                )

        groups.append(
            SeqTrainingGroup(
                num_players=num_players,
                hand_size=hand_size,
                tokens=tokens,
                owned=owned,
                value_targets=value_targets,
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
        self.model = model.to(self.device)
        self.train = train_config
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=train_config.learning_rate,
            betas=train_config.adam_betas,
        )
        self.collector = SeqRolloutCollector(
            self.model, train_config, device=self.device
        )
        self.league = SeqLeague(
            train_config.league_max_snapshots, train_config.league_min_iteration
        )
        self.iteration = 0
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
        self.model.eval()
        trees = self.collector.collect(self.league, self.rng, iteration=self.iteration)
        return trees, summarize_trees(trees, self.collector.stats)

    # -------------------------------------------------------------- #
    # Update                                                          #
    # -------------------------------------------------------------- #

    def update(self, trees: list[SeqTree]) -> SeqUpdateStats:
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
        stats.positions = int(sum(g.owned.sum() for g in groups))

        for _ in range(self.train.epochs):
            snapshot = (
                copy.deepcopy(self.model.state_dict()),
                copy.deepcopy(self.optimizer.state_dict()),
            )
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            self._epoch_backward(groups, stats)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.train.max_grad_norm
            )
            self._apply_warmup_lr()
            nominal_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
            accepted = False
            attempts = self.train.kl_backtrack_attempts + 1
            for attempt in range(attempts):
                if attempt:
                    self.model.load_state_dict(snapshot[0])
                    self.optimizer.load_state_dict(snapshot[1])
                scale = self.train.kl_backtrack_factor**attempt
                for group, nominal_lr in zip(self.optimizer.param_groups, nominal_lrs):
                    group["lr"] = nominal_lr * scale
                self.optimizer.step()
                policy_kl, kl_p95, kl_p99, kl_max = self._evaluate_policy_kl(groups)
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
                self.optimizer.load_state_dict(snapshot[1])
                stats.rolled_back = True
                stats.step_scale = 0.0
                break
        self._release_device_memory()
        stats.update_sec = time.perf_counter() - started
        return stats

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
            group["lr"] = self.train.learning_rate * scale

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
        for group in groups:
            by_chunk = _rows_by_chunk(group, self._microbatches(group))
            for (start, stop), policy_rows in zip(self._microbatches(group), by_chunk):
                loss, terms = self._chunk_loss(group, start, stop, policy_rows)
                if loss is not None:
                    loss.backward()
                for key, value in terms.items():
                    totals[key] += value
        stats.loss_policy = totals["policy"]
        stats.loss_value = totals["value"]
        stats.loss_suit = totals["suit"]
        stats.loss_trick = totals["trick"]
        stats.loss_bid_hit = totals["bid_hit"]
        stats.entropy = totals["entropy"]

    def _chunk_loss(self, group, start, stop, policy_rows):
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
        output = self.model.forward_full(tokens, aux_heads=aux)
        owned = torch.from_numpy(group.owned[start:stop]).to(device)
        position_weight = (
            torch.from_numpy(group.position_weight[start:stop]).to(device)
            * owned.float()
        )

        terms: dict[str, float] = {}
        loss = tokens.new_zeros((), dtype=torch.float32)

        # Value.
        value_targets = torch.from_numpy(group.value_targets[start:stop]).to(device)
        value_loss = (
            F.smooth_l1_loss(output.value, value_targets, reduction="none")
            * position_weight
        ).sum()
        loss = loss + train.value_coef * value_loss
        terms["value"] = float(value_loss.detach())

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
            terms["suit"] = float(suit_loss.detach())

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
            terms["trick"] = float(trick_loss.detach())

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
            terms["bid_hit"] = float(bid_hit_loss.detach())

        # Policy: NeuRD over every focal decision, branched or not.
        policy_total = 0.0
        entropy_total = 0.0
        for phase_key, rows in policy_rows.items():
            if not rows["weight"]:
                continue
            losses, _, entropies, weight = self._policy_terms(output, rows, phase_key)
            policy_loss = (weight * losses).sum()
            loss = loss + train.policy_coef * policy_loss
            policy_total += float(policy_loss.detach())
            if train.entropy_coef:
                entropy_term = (weight * entropies).sum()
                loss = loss - train.entropy_coef * entropy_term
                entropy_total += float(entropy_term.detach())
            else:
                entropy_total += float((weight * entropies).sum().detach())
        if policy_total:
            terms["policy"] = policy_total
            terms["entropy"] = entropy_total

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
        return losses, divergences, entropies, weight

    def _evaluate_policy_kl(self, groups) -> tuple[float, float, float, float]:
        """Mean, p95, p99, and max KL(old||new) over every policy row.

        Under the old split this covered branch rows only, leaving the spine
        half of the decisions outside the rollback guard. With one row type
        there is nothing left uncovered.
        """

        self.model.eval()
        all_divergences: list[np.ndarray] = []
        all_weights: list[np.ndarray] = []
        with torch.inference_mode():
            for group in groups:
                for (start, stop), policy_rows in zip(
                    self._microbatches(group),
                    _rows_by_chunk(group, self._microbatches(group)),
                ):
                    if not any(rows["weight"] for rows in policy_rows.values()):
                        continue
                    tokens = torch.from_numpy(group.tokens[start:stop]).to(self.device)
                    output = self.model.forward_full(tokens, aux_heads=False)
                    for phase_key, rows in policy_rows.items():
                        if not rows["weight"]:
                            continue
                        _, divergences, _, weight = self._policy_terms(
                            output, rows, phase_key
                        )
                        all_divergences.append(divergences.detach().cpu().numpy())
                        all_weights.append(weight.detach().cpu().numpy())
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
        }
        payload = {
            "checkpoint_format_version": 1,
            "schema_version": SEQ_SCHEMA_VERSION,
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
        }
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
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.iteration = int(payload.get("iteration", 0))
        self.optimizer_steps = int(payload.get("optimizer_steps", 0))
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
