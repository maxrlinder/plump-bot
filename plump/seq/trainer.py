"""Trainer for the schema-v6 sequence pipeline.

One update consumes the collected trees as full causal forwards grouped by
(players, hand size). Policy losses follow the v4 recursive pipeline:
guarded NeuRD (or PPO) on branch rows with counterfactual Q backups, PPO on
spine rows, plus per-position value/owner/suit/trick auxiliary losses.

Gradient-scale note: NeuRD's logit gradient is ~A while PPO's at ratio 1 is
~pi(a)*A, so branch rows push low-probability actions much harder. Branch and
spine contributions are therefore normalized by their own global weight sums
each update and combined via explicit ``branch_policy_coef`` /
``spine_policy_coef`` — the branch/spine mix cannot drift the effective
learning rate. Tree weighting keeps factorially larger games from dominating.
"""

from __future__ import annotations

import copy
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from plump.modeling.torch_model import masked_capacity_sinkhorn
from plump.rounds import rules_fingerprint

from .config import (
    NEXT_BID,
    NUM_CARDS,
    SEQ_SCHEMA_VERSION,
    SeqModelConfig,
    SeqTrainingConfig,
    seq_len,
)
from .model import SeqPlumpModel
from .policy import SeqLeague, best_seq_device
from .rollout import SeqRolloutCollector, SeqTree
from .tokens import IGNORE_LABEL, build_replay_arrays


# --------------------------------------------------------------------- #
# Batch containers                                                       #
# --------------------------------------------------------------------- #


@dataclass
class PolicyRows:
    """Spine PPO rows for one phase within one group."""

    seq_index: list[int] = field(default_factory=list)
    position: list[int] = field(default_factory=list)
    action: list[int] = field(default_factory=list)
    old_probs: list[np.ndarray] = field(default_factory=list)
    advantage: list[float] = field(default_factory=list)
    weight: list[float] = field(default_factory=list)


@dataclass
class BranchRows:
    """Branch rows (counterfactual Q backups) for one phase within one group."""

    seq_index: list[int] = field(default_factory=list)
    position: list[int] = field(default_factory=list)
    old_probs_full: list[np.ndarray] = field(default_factory=list)
    candidates: list[np.ndarray] = field(default_factory=list)
    # Backup weights: what the parent's value averages the child Q values with.
    # Zero for a purely explored arm, so these are *not* a behaviour policy.
    priors: list[np.ndarray] = field(default_factory=list)
    # The behaviour policy restricted to the candidate set. This is the correct
    # importance-ratio denominator; ``priors`` would divide by zero on an
    # explored arm and can be a Monte-Carlo frequency rather than a
    # probability on the sampling modes.
    behaviour: list[np.ndarray] = field(default_factory=list)
    q_values: list[np.ndarray] = field(default_factory=list)
    weight: list[float] = field(default_factory=list)


@dataclass
class SeqTrainingGroup:
    num_players: int
    hand_size: int
    tokens: np.ndarray          # [B, L, WIDTH]
    owned: np.ndarray           # [B, L] bool
    value_targets: np.ndarray   # [B, L] float32
    seq_weight: np.ndarray      # [B] float32 (aux/value weight per sequence)
    owner_targets: np.ndarray
    owner_valid: np.ndarray
    owner_capacities: np.ndarray
    trick_targets: np.ndarray   # [B, max_players]
    trick_masks: np.ndarray
    suit_targets: np.ndarray
    spine: dict[str, PolicyRows]
    branch: dict[str, BranchRows]


@dataclass
class SeqUpdateStats:
    loss_spine: float = 0.0
    loss_branch: float = 0.0
    loss_value: float = 0.0
    loss_owner: float = 0.0
    loss_suit: float = 0.0
    loss_trick: float = 0.0
    entropy: float = 0.0
    branch_kl: float = 0.0
    rolled_back: bool = False
    spine_rows: int = 0
    branch_rows: int = 0
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


# --------------------------------------------------------------------- #
# Batch building                                                         #
# --------------------------------------------------------------------- #


def _branch_depth(record) -> int:
    """How many branch decisions sit above this one on its path."""

    depth = 0
    upstream = record.branch.upstream
    while upstream is not None:
        depth += 1
        upstream = upstream[0].upstream
    return depth


def _branch_depth_scale(trees, exponent: float) -> dict[int, float]:
    """Per-branch-record multiplier that re-weights rows by depth.

    Normalized per tree so the tree's total branch weight is untouched: this
    knob decides where inside a tree the gradient lands, and must not double as
    a way of making some trees count more.
    """

    if exponent == 0.0:
        return {}
    scale: dict[int, float] = {}
    for tree in trees:
        records = [
            record
            for leaf in tree.leaves
            for record in leaf.decisions
            if record.branch is not None
        ]
        if not records:
            continue
        raw = {
            id(record): (1.0 + _branch_depth(record)) ** exponent
            for record in records
        }
        total = sum(raw.values())
        if total <= 0:
            continue
        normalizer = len(records) / total
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
            seq_len(tree.num_players, tree.hand_size) - leaf.owned_from
            for leaf in tree.leaves
        )
        for tree in trees
    }
    rows_per_tree_spine = {
        id(tree): sum(
            sum(1 for r in leaf.decisions if r.branch is None)
            for leaf in tree.leaves
        )
        for tree in trees
    }
    rows_per_tree_branch = {
        id(tree): sum(
            sum(1 for r in leaf.decisions if r.branch is not None)
            for leaf in tree.leaves
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

    def row_weights(rows_per_tree: dict[int, int]):
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
            key: shares[key] / total / max(rows_per_tree[key], 1)
            for key in rows_per_tree
        }

    seq_weights = row_weights(owned_per_tree)
    spine_weights = row_weights(rows_per_tree_spine)
    branch_weights = row_weights(rows_per_tree_branch)

    depth_scale = _branch_depth_scale(trees, train_config.branch_depth_exponent)

    def seq_weight_for(tree: SeqTree) -> float:
        return seq_weights[id(tree)]

    def spine_weight_for(tree: SeqTree) -> float:
        return spine_weights[id(tree)]

    def branch_weight_for(tree: SeqTree) -> float:
        return branch_weights[id(tree)]

    groups: list[SeqTrainingGroup] = []
    for (num_players, hand_size), entries in sorted(by_shape.items()):
        length = seq_len(num_players, hand_size)
        batch = len(entries)
        tokens = np.zeros((batch, length, 12), dtype=np.int64)
        owned = np.zeros((batch, length), dtype=bool)
        value_targets = np.zeros((batch, length), dtype=np.float32)
        seq_weight = np.zeros(batch, dtype=np.float32)
        owner_targets = np.full((batch, length, NUM_CARDS), IGNORE_LABEL, np.int64)
        owner_valid = np.zeros(
            (batch, length, NUM_CARDS, model_config.owner_class_count), dtype=bool
        )
        owner_capacities = np.zeros(
            (batch, length, model_config.owner_class_count), dtype=np.int64
        )
        trick_targets = np.full(
            (batch, model_config.max_players), IGNORE_LABEL, np.int64
        )
        trick_masks = np.zeros(
            (batch, length, model_config.max_players, model_config.bid_count),
            dtype=bool,
        )
        suit_targets = np.full(
            (batch, length, model_config.max_players, 4), IGNORE_LABEL, np.int64
        )
        spine = {"bid": PolicyRows(), "play": PolicyRows()}
        branch = {"bid": BranchRows(), "play": BranchRows()}

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
            )
            tokens[row] = arrays.tokens
            owned[row, leaf.owned_from :] = True
            for position, value in leaf.value_targets().items():
                value_targets[row, position] = value
            seq_weight[row] = seq_weight_for(tree)
            owner_targets[row] = arrays.owner_targets
            owner_valid[row] = arrays.owner_valid
            owner_capacities[row] = arrays.owner_capacities
            trick_targets[row] = arrays.trick_targets
            trick_masks[row] = arrays.trick_masks
            suit_targets[row] = arrays.suit_targets

            for record in leaf.decisions:
                phase_key = "bid" if record.phase == NEXT_BID else "play"
                if record.branch is None:
                    rows = spine[phase_key]
                    rows.seq_index.append(row)
                    rows.position.append(record.position)
                    rows.action.append(record.action_index)
                    rows.old_probs.append(record.old_probs)
                    rows.advantage.append(
                        leaf.value_target_at(record.position) - record.old_value
                    )
                    rows.weight.append(spine_weight_for(tree))
                else:
                    b = record.branch
                    rows = branch[phase_key]
                    rows.seq_index.append(row)
                    rows.position.append(record.position)
                    rows.old_probs_full.append(record.old_probs)
                    rows.candidates.append(
                        np.asarray(b.candidate_indices, dtype=np.int64)
                    )
                    rows.priors.append(np.asarray(b.prior_probs, dtype=np.float32))
                    raw = np.asarray(
                        [b.raw_probs[i] for i in b.candidate_indices],
                        dtype=np.float32,
                    )
                    rows.behaviour.append(raw / max(float(raw.sum()), 1e-12))
                    rows.q_values.append(
                        np.asarray(
                            [b.child_values[i] for i in b.candidate_indices],
                            dtype=np.float32,
                        )
                    )
                    rows.weight.append(
                        branch_weight_for(tree)
                        * depth_scale.get(id(record), 1.0)
                    )

        groups.append(
            SeqTrainingGroup(
                num_players=num_players,
                hand_size=hand_size,
                tokens=tokens,
                owned=owned,
                value_targets=value_targets,
                seq_weight=seq_weight,
                owner_targets=owner_targets,
                owner_valid=owner_valid,
                owner_capacities=owner_capacities,
                trick_targets=trick_targets,
                trick_masks=trick_masks,
                suit_targets=suit_targets,
                spine=spine,
                branch=branch,
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
        self.rng = random.Random(train_config.seed)

    # -------------------------------------------------------------- #
    # Collection                                                      #
    # -------------------------------------------------------------- #

    def collect(self) -> tuple[list[SeqTree], SeqRolloutSummary]:
        self.model.eval()
        trees = self.collector.collect(self.league, self.rng)
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
        stats.spine_rows = sum(
            len(rows.action) for g in groups for rows in g.spine.values()
        )
        stats.branch_rows = sum(
            len(rows.weight) for g in groups for rows in g.branch.values()
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
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.train.max_grad_norm
            )
            self.optimizer.step()
            branch_kl = self._evaluate_branch_kl(groups)
            stats.branch_kl = branch_kl
            if stats.branch_rows > 0 and branch_kl > self.train.branch_kl_cap:
                self.model.load_state_dict(snapshot[0])
                self.optimizer.load_state_dict(snapshot[1])
                stats.rolled_back = True
                break
        self._release_device_memory()
        stats.update_sec = time.perf_counter() - started
        return stats

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
            spine_by_chunk, branch_by_chunk = _rows_by_chunk(group, self._microbatches(group))
            for (start, stop), spine_rows, branch_rows in zip(
                self._microbatches(group), spine_by_chunk, branch_by_chunk
            ):
                loss, terms = self._chunk_loss(
                    group, start, stop, spine_rows, branch_rows
                )
                if loss is not None:
                    loss.backward()
                for key, value in terms.items():
                    totals[key] += value
        stats.loss_spine = totals["spine"]
        stats.loss_branch = totals["branch"]
        stats.loss_value = totals["value"]
        stats.loss_owner = totals["owner"]
        stats.loss_suit = totals["suit"]
        stats.loss_trick = totals["trick"]
        stats.entropy = totals["entropy"]

    def _chunk_loss(self, group, start, stop, spine_rows, branch_rows):
        device = self.device
        train = self.train
        tokens = torch.from_numpy(group.tokens[start:stop]).to(device)
        output = self.model.forward_full(tokens)
        owned = torch.from_numpy(group.owned[start:stop]).to(device)
        seq_weight = torch.from_numpy(group.seq_weight[start:stop]).to(device)
        position_weight = seq_weight[:, None] * owned.float()

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

        # Owner (Sinkhorn-constrained NLL + capacity term).
        owner_targets = torch.from_numpy(group.owner_targets[start:stop]).to(device)
        labeled = (owner_targets != IGNORE_LABEL) & owned[:, :, None]
        if labeled.any():
            positions = labeled.any(dim=2)
            fraction = self.train.owner_position_fraction
            if fraction < 1.0:
                keep = (
                    torch.rand_like(positions, dtype=torch.float32) < fraction
                )
                positions = positions & keep
            sel = positions.nonzero(as_tuple=True)
            if sel[0].numel() == 0:
                sel = None
        else:
            sel = None
        if sel is not None:
            logits = output.owner_logits[sel[0], sel[1]]
            valid = torch.from_numpy(group.owner_valid[start:stop]).to(device)[
                sel[0], sel[1]
            ]
            capacities = torch.from_numpy(
                group.owner_capacities[start:stop]
            ).to(device)[sel[0], sel[1]]
            targets = owner_targets[sel[0], sel[1]]
            weights = (seq_weight[:, None] * positions.float())[sel[0], sel[1]]
            probs = masked_capacity_sinkhorn(
                logits, valid, capacities,
                iterations=train.owner_sinkhorn_iterations,
            )
            card_labeled = targets != IGNORE_LABEL
            gathered = probs.gather(
                2, targets.clamp_min(0).unsqueeze(-1)
            ).squeeze(-1)
            nll = -torch.log(gathered.clamp_min(1e-9)) * card_labeled.float()
            hidden_counts = card_labeled.float().sum(dim=1).clamp_min(1.0)
            owner_loss = (weights * nll.sum(dim=1) / hidden_counts).sum()
            # -1e9 instead of -inf: fully-masked card rows (observer-held or
            # played cards) must stay NaN-free through softmax backward.
            pre = torch.softmax(
                logits.float().masked_fill(~valid, -1e9), dim=-1
            )
            pre = torch.where(
                valid.any(dim=-1, keepdim=True), pre, torch.zeros_like(pre)
            )
            capacity_mse = (
                (pre.sum(dim=1) - capacities.float()) ** 2
            ).mean(dim=-1)
            owner_loss = owner_loss + train.owner_capacity_coef * (
                weights * capacity_mse
            ).sum()
            loss = loss + train.owner_coef * owner_loss
            terms["owner"] = float(owner_loss.detach())

        # Suit presence.
        suit_targets = torch.from_numpy(group.suit_targets[start:stop]).to(device)
        suit_labeled = (suit_targets != IGNORE_LABEL) & owned[:, :, None, None]
        if suit_labeled.any():
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
        trick_targets = torch.from_numpy(group.trick_targets[start:stop]).to(device)
        trick_masks = torch.from_numpy(group.trick_masks[start:stop]).to(device)
        player_labeled = trick_targets != IGNORE_LABEL  # [B, P]
        if player_labeled.any():
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
                player_labeled[:, None, :]
                & owned[:, :, None]
                & trick_masks.any(dim=-1)
            )
            per_position = (-gathered * mask.float()).sum(dim=2) / (
                mask.float().sum(dim=2).clamp_min(1.0)
            )
            trick_loss = (per_position * position_weight).sum()
            loss = loss + train.trick_coef * trick_loss
            terms["trick"] = float(trick_loss.detach())

        # Spine PPO rows.
        spine_total = 0.0
        entropy_total = 0.0
        for phase_key, rows in spine_rows.items():
            if not rows["action"]:
                continue
            logits_all = (
                output.bid_logits if phase_key == "bid" else output.card_logits
            )
            seq_idx = torch.tensor(rows["seq_index"], device=device)
            pos_idx = torch.tensor(rows["position"], device=device)
            logits = logits_all[seq_idx, pos_idx].float()
            old_probs = torch.from_numpy(np.stack(rows["old_probs"])).to(device)
            legal = old_probs > 0
            masked_logits = logits.masked_fill(~legal, float("-inf"))
            log_probs = torch.log_softmax(masked_logits, dim=-1)
            actions = torch.tensor(rows["action"], device=device)
            advantage = torch.tensor(
                rows["advantage"], device=device, dtype=torch.float32
            )
            weight = torch.tensor(
                rows["weight"], device=device, dtype=torch.float32
            )
            logp_new = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
            logp_old = torch.log(
                old_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
            )
            ratio = torch.exp(logp_new - logp_old)
            clipped = ratio.clamp(
                1.0 - train.ppo_clip_eps, 1.0 + train.ppo_clip_eps
            )
            ppo = -torch.minimum(ratio * advantage, clipped * advantage)
            spine_loss = (weight * ppo).sum()
            probs_new = log_probs.exp() * legal.float()
            entropy = -(probs_new * log_probs.masked_fill(~legal, 0.0)).sum(-1)
            entropy_term = (weight * entropy).sum()
            loss = (
                loss
                + train.spine_policy_coef * spine_loss
                - train.entropy_coef * entropy_term
            )
            spine_total += float(spine_loss.detach())
            entropy_total += float(entropy_term.detach())
        if spine_total:
            terms["spine"] = spine_total
            terms["entropy"] = entropy_total

        # Branch rows (NeuRD or PPO over counterfactual Q values).
        branch_total = 0.0
        for phase_key, rows in branch_rows.items():
            if not rows["weight"]:
                continue
            branch_loss = self._branch_loss(output, rows, phase_key)
            loss = loss + train.branch_policy_coef * branch_loss
            branch_total += float(branch_loss.detach())
        if branch_total:
            terms["branch"] = branch_total

        if loss.requires_grad:
            return loss, terms
        return None, terms

    def _branch_terms(self, output, rows, phase_key):
        """Per-row branch losses and KL(old||new) divergences."""

        device = self.device
        train = self.train
        logits_all = output.bid_logits if phase_key == "bid" else output.card_logits
        seq_idx = torch.tensor(rows["seq_index"], device=device)
        pos_idx = torch.tensor(rows["position"], device=device)
        logits = logits_all[seq_idx, pos_idx].float()

        max_k = max(len(c) for c in rows["candidates"])
        count = len(rows["weight"])
        cand = torch.zeros(count, max_k, dtype=torch.long, device=device)
        cand_mask = torch.zeros(count, max_k, dtype=torch.bool, device=device)
        priors = torch.zeros(count, max_k, device=device)
        behaviour = torch.zeros(count, max_k, device=device)
        q_values = torch.zeros(count, max_k, device=device)
        for row in range(count):
            k = len(rows["candidates"][row])
            cand[row, :k] = torch.from_numpy(rows["candidates"][row])
            cand_mask[row, :k] = True
            priors[row, :k] = torch.from_numpy(rows["priors"][row])
            behaviour[row, :k] = torch.from_numpy(rows["behaviour"][row])
            q_values[row, :k] = torch.from_numpy(rows["q_values"][row])
        old_full = torch.from_numpy(np.stack(rows["old_probs_full"])).to(device)
        weight = torch.tensor(rows["weight"], device=device, dtype=torch.float32)

        gathered = logits.gather(1, cand)
        baseline = (priors * q_values).sum(dim=-1, keepdim=True)
        advantages = (q_values - baseline) * cand_mask.float()
        advantages = advantages.clamp(
            -train.neurd_advantage_clip, train.neurd_advantage_clip
        )

        # Full-support KL(old || new) anchor and guard metric.
        legal = old_full > 0
        full_logprobs = torch.log_softmax(
            logits.masked_fill(~legal, float("-inf")), dim=-1
        )
        safe = torch.where(legal, full_logprobs, torch.zeros_like(full_logprobs))
        log_old = torch.log(old_full.clamp_min(1e-12))
        divergences = (old_full * (log_old - safe) * legal.float()).sum(dim=-1)

        if train.branch_policy_objective == "neurd":
            counts = cand_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
            centered_adv = (
                advantages - advantages.sum(dim=-1, keepdim=True) / counts
            ) * cand_mask.float()
            finite = torch.where(cand_mask, gathered, torch.zeros_like(gathered))
            centered_logits = (
                finite - finite.sum(dim=-1, keepdim=True) / counts
            ) * cand_mask.float()
            regret = -(centered_adv.detach() * centered_logits).sum(
                dim=-1
            ) / counts.squeeze(-1)
            losses = (
                train.neurd_regret_coef * regret
                + train.neurd_kl_coef * divergences
            )
        else:
            cand_logits = gathered.masked_fill(~cand_mask, float("-inf"))
            cand_logprobs = torch.log_softmax(cand_logits, dim=-1)
            safe_cand = torch.where(
                cand_mask, cand_logprobs, torch.zeros_like(cand_logprobs)
            )
            new = safe_cand.exp() * cand_mask.float()
            log_behaviour = torch.log(behaviour.clamp_min(1e-12))
            clipped = torch.exp(
                (safe_cand - log_behaviour).clamp(-20.0, 20.0)
            ).clamp(1.0 - train.ppo_clip_eps, 1.0 + train.ppo_clip_eps)
            losses = -torch.minimum(
                new * advantages, behaviour * clipped * advantages
            ).sum(dim=-1)
        return losses, divergences, weight

    def _branch_loss(self, output, rows, phase_key):
        losses, _, weight = self._branch_terms(output, rows, phase_key)
        return (weight * losses).sum()

    def _evaluate_branch_kl(self, groups) -> float:
        self.model.eval()
        total = 0.0
        weight_sum = 0.0
        with torch.inference_mode():
            for group in groups:
                for (start, stop), spine_rows, branch_rows in zip(
                    self._microbatches(group),
                    *_rows_by_chunk(group, self._microbatches(group)),
                ):
                    if not any(rows["weight"] for rows in branch_rows.values()):
                        continue
                    tokens = torch.from_numpy(group.tokens[start:stop]).to(
                        self.device
                    )
                    output = self.model.forward_full(tokens, aux_heads=False)
                    for phase_key, rows in branch_rows.items():
                        if not rows["weight"]:
                            continue
                        _, divergences, weight = self._branch_terms(
                            output, rows, phase_key
                        )
                        total += float((weight * divergences).sum())
                        weight_sum += float(weight.sum())
        return total / weight_sum if weight_sum > 0 else 0.0

    # -------------------------------------------------------------- #
    # Checkpoints & league                                            #
    # -------------------------------------------------------------- #

    def save_checkpoint(self, path: str | Path) -> None:
        payload = {
            "schema_version": SEQ_SCHEMA_VERSION,
            "iteration": self.iteration,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": asdict(self.model.config),
            "training_config": asdict(self.train),
            "rules_fingerprint": rules_fingerprint(),
            "league": [
                (snap.snapshot_id, snap.path, snap.iteration)
                for snap in self.league.snapshots
            ],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def load_checkpoint(self, path: str | Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != SEQ_SCHEMA_VERSION:
            raise ValueError("Checkpoint schema mismatch.")
        if payload.get("rules_fingerprint") != rules_fingerprint():
            raise ValueError("Checkpoint rules fingerprint mismatch.")
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.iteration = int(payload.get("iteration", 0))
        for snapshot_id, snap_path, iteration in payload.get("league", []):
            if Path(snap_path).exists():
                self.league.add(snapshot_id, snap_path, iteration)

    def maybe_snapshot(self, checkpoint_dir: str | Path) -> Optional[str]:
        if self.iteration % self.train.snapshot_every != 0:
            return None
        directory = Path(checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"seq_v6_iter_{self.iteration:06d}.pt"
        self.save_checkpoint(path)
        self.league.add(f"iter_{self.iteration}", str(path), self.iteration)
        return str(path)


def _rows_by_chunk(group: SeqTrainingGroup, chunks) -> tuple[list, list]:
    """Split the group's policy rows per microbatch chunk, reindexing rows."""

    chunk_list = list(chunks)
    spine_chunks = []
    branch_chunks = []
    for start, stop in chunk_list:
        spine_chunk: dict[str, dict[str, list]] = {}
        for phase_key, rows in group.spine.items():
            selected = {
                "seq_index": [],
                "position": [],
                "action": [],
                "old_probs": [],
                "advantage": [],
                "weight": [],
            }
            for i, seq_index in enumerate(rows.seq_index):
                if start <= seq_index < stop:
                    selected["seq_index"].append(seq_index - start)
                    selected["position"].append(rows.position[i])
                    selected["action"].append(rows.action[i])
                    selected["old_probs"].append(rows.old_probs[i])
                    selected["advantage"].append(rows.advantage[i])
                    selected["weight"].append(rows.weight[i])
            spine_chunk[phase_key] = selected
        branch_chunk: dict[str, dict[str, list]] = {}
        for phase_key, rows in group.branch.items():
            selected = {
                "seq_index": [],
                "position": [],
                "old_probs_full": [],
                "candidates": [],
                "priors": [],
                "behaviour": [],
                "q_values": [],
                "weight": [],
            }
            for i, seq_index in enumerate(rows.seq_index):
                if start <= seq_index < stop:
                    selected["seq_index"].append(seq_index - start)
                    selected["position"].append(rows.position[i])
                    selected["old_probs_full"].append(rows.old_probs_full[i])
                    selected["candidates"].append(rows.candidates[i])
                    selected["priors"].append(rows.priors[i])
                    selected["behaviour"].append(rows.behaviour[i])
                    selected["q_values"].append(rows.q_values[i])
                    selected["weight"].append(rows.weight[i])
            branch_chunk[phase_key] = selected
        spine_chunks.append(spine_chunk)
        branch_chunks.append(branch_chunk)
    return spine_chunks, branch_chunks
