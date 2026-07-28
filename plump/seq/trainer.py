"""Trainer for the schema-v6 sequence pipeline.

One update consumes the collected trees as full causal forwards grouped by
(players, hand size). There is a single policy objective — NeuRD — over every
focal decision, plus per-position value/owner/suit/trick auxiliary losses.

Why one objective. The rollout is an external-sampling tree: the focal's
actions are enumerated (or drawn by an explicit rule) while chance and the
opponents are sampled. On that data an importance ratio corrects nothing —
ratios exist to repair sampling from the policy, and the candidate set was not
drawn that way. NeuRD's loss has no expectation over actions in it at all:

    L = -sum_a w_a * sg[A(a)] * y_a      =>      dL/dy_a = -w_a * A(a)

which is a per-action statement, true for each ``a`` independently of which
other actions were expanded. So an arbitrary candidate set is valid, provided
(1) Q(a) is unbiased for that action's continuation value, (2) the baseline V
is the full-policy value rather than a mean over the candidates, and (3) set
membership does not depend on the noise in Q(a).

A branched decision supplies Q(a) per candidate and V from the backup (exact
for full candidate mass, control-variate for a capped set). An unbranched one
is the k=1 case: Q is the realized return and V is the value head's estimate,
which is the same statement with a bootstrapped baseline. Both are rows of the
same loss, so there is no branch/spine mix to balance.

``w_a`` is 1/q(a)**exponent, where q is the chance the branch rule would have
expanded that action. Without it, a rule that expands by sampling the policy
gives E[gradient] ~ pi(a)*A(a) — the policy-gradient prefactor NeuRD exists to
remove, reintroduced through the candidate sampler. Tree weighting keeps
factorially larger games from dominating.
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
    """NeuRD rows for one phase within one group.

    One row per focal decision, branched or not. A branched decision carries
    its candidate set; an unbranched one carries the single realized action.
    """

    seq_index: list[int] = field(default_factory=list)
    position: list[int] = field(default_factory=list)
    old_probs_full: list[np.ndarray] = field(default_factory=list)
    candidates: list[np.ndarray] = field(default_factory=list)
    q_values: list[np.ndarray] = field(default_factory=list)
    # Full-policy value at this state, *not* a mean over the candidates: the
    # backup for a branched decision, the value head for an unbranched one.
    # NeuRD needs A(a) = Q(a) - V with V over all legal actions, because the
    # loss makes a claim about each candidate logit on its own.
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
    policy: dict[str, PolicyRows]


@dataclass
class SeqUpdateStats:
    loss_policy: float = 0.0
    loss_value: float = 0.0
    loss_owner: float = 0.0
    loss_suit: float = 0.0
    loss_trick: float = 0.0
    entropy: float = 0.0
    policy_kl: float = 0.0
    rolled_back: bool = False
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
        raw = {
            id(record): (1.0 + record.depth) ** exponent for record in records
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
            model_config.seq_len(tree.num_players, tree.hand_size)
            - leaf.owned_from
            for leaf in tree.leaves
        )
        for tree in trees
    }
    rows_per_tree_policy = {
        id(tree): sum(len(leaf.decisions) for leaf in tree.leaves)
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
                owner_labels=train_config.owner_coef > 0,
                suit_labels=train_config.suit_coef > 0,
                trick_labels=train_config.trick_coef > 0,
            )
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
                        np.asarray(
                            [record.old_probs[action]], dtype=np.float32
                        )
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
                    # The backup: exact sum_a pi(a) Q(a) at full candidate
                    # mass, control-variate estimate of the same quantity for
                    # a capped set. Either way it is the full-policy value,
                    # which is what A(a) must be measured against.
                    rows.baseline.append(float(b.backed_value))
                    rows.inclusion.append(
                        np.asarray(b.inclusion_probs, dtype=np.float32)
                    )
                    rows.branched.append(True)
                rows.weight.append(
                    policy_weight_for(tree) * depth_scale.get(id(record), 1.0)
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
            self.optimizer.step()
            policy_kl = self._evaluate_policy_kl(groups)
            stats.policy_kl = policy_kl
            if stats.policy_rows > 0 and policy_kl > self.train.policy_kl_cap:
                self.model.load_state_dict(snapshot[0])
                self.optimizer.load_state_dict(snapshot[1])
                stats.rolled_back = True
                break
            self.optimizer_steps += 1
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
        scale = 1.0 if warmup <= 0 else min(
            1.0, (self.optimizer_steps + 1) / warmup
        )
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
            for (start, stop), policy_rows in zip(
                self._microbatches(group), by_chunk
            ):
                loss, terms = self._chunk_loss(group, start, stop, policy_rows)
                if loss is not None:
                    loss.backward()
                for key, value in terms.items():
                    totals[key] += value
        stats.loss_policy = totals["policy"]
        stats.loss_value = totals["value"]
        stats.loss_owner = totals["owner"]
        stats.loss_suit = totals["suit"]
        stats.loss_trick = totals["trick"]
        stats.entropy = totals["entropy"]

    def _chunk_loss(self, group, start, stop, policy_rows):
        device = self.device
        train = self.train
        aux = train.owner_coef > 0 or train.suit_coef > 0 or train.trick_coef > 0
        tokens = torch.from_numpy(group.tokens[start:stop]).to(device)
        output = self.model.forward_full(tokens, aux_heads=aux)
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
        sel = None
        if train.owner_coef > 0:
            owner_targets = torch.from_numpy(
                group.owner_targets[start:stop]
            ).to(device)
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
        suit_labeled = None
        if train.suit_coef > 0:
            suit_targets = torch.from_numpy(
                group.suit_targets[start:stop]
            ).to(device)
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
            trick_targets = torch.from_numpy(
                group.trick_targets[start:stop]
            ).to(device)
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

        # Policy: NeuRD over every focal decision, branched or not.
        policy_total = 0.0
        entropy_total = 0.0
        for phase_key, rows in policy_rows.items():
            if not rows["weight"]:
                continue
            losses, _, entropies, weight = self._policy_terms(
                output, rows, phase_key
            )
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
        """Per-row NeuRD losses, KL(old||new) divergences, and entropies."""

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
        baseline = to_device(
            np.asarray(rows["baseline"], dtype=np.float32)
        ).unsqueeze(-1)

        gathered = logits.gather(1, cand)
        advantages = (q_values - baseline) * cand_mask.float()
        advantages = advantages.clamp(
            -train.neurd_advantage_clip, train.neurd_advantage_clip
        )

        # 1/q correction for which actions the branch rule expanded. Capped
        # because the deterministic and single-sample rules can put q at the
        # policy mass of a near-zero-probability action; with a uniform or
        # Gumbel arm in play the cap is essentially never reached.
        exponent = train.neurd_inclusion_exponent
        if exponent == 0.0:
            importance = cand_mask.float()
        else:
            importance = (
                inclusion.clamp_min(1e-6).pow(-exponent).clamp_max(
                    train.neurd_inclusion_cap
                )
                * cand_mask.float()
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

        # L = -sum_a w_a A(a) y_a, so dL/dy_a = -w_a A(a): each candidate logit
        # moves by its own regret, with no pi(a) prefactor and no dependence on
        # which siblings were expanded.
        #
        # Deliberately *not* centered over the candidate set. Centering is only
        # a null direction when the set is every legal action; for a capped set
        # it would cancel the one real signal about the set as a whole -- if
        # every expanded action beat the baseline, mass should move to them
        # from the unexpanded ones. Nor is the loss divided by the candidate
        # count: a k=1 row would then have exactly zero gradient, and scaling
        # each action's regret by how many siblings it happened to get is the
        # arbitrary weighting this objective exists to remove. What keeps the
        # per-action gradients honest across different set sizes is the 1/q
        # correction above, not a per-row rescale.
        scaled_adv = advantages * importance * cand_mask.float()
        finite = torch.where(cand_mask, gathered, torch.zeros_like(gathered))
        regret = -(scaled_adv.detach() * finite).sum(dim=-1)
        losses = (
            train.neurd_regret_coef * regret + train.neurd_kl_coef * divergences
        )
        return losses, divergences, entropies, weight

    def _evaluate_policy_kl(self, groups) -> float:
        """Weighted KL(old||new) over *every* policy row.

        Under the old split this covered branch rows only, leaving the spine
        half of the decisions outside the rollback guard. With one row type
        there is nothing left uncovered.
        """

        self.model.eval()
        total = 0.0
        weight_sum = 0.0
        with torch.inference_mode():
            for group in groups:
                for (start, stop), policy_rows in zip(
                    self._microbatches(group),
                    _rows_by_chunk(group, self._microbatches(group)),
                ):
                    if not any(rows["weight"] for rows in policy_rows.values()):
                        continue
                    tokens = torch.from_numpy(group.tokens[start:stop]).to(
                        self.device
                    )
                    output = self.model.forward_full(tokens, aux_heads=False)
                    for phase_key, rows in policy_rows.items():
                        if not rows["weight"]:
                            continue
                        _, divergences, _, weight = self._policy_terms(
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
            "optimizer_steps": self.optimizer_steps,
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
        self.optimizer_steps = int(payload.get("optimizer_steps", 0))
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
