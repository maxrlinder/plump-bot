"""Branch-free PPO batches and numerically explicit masked policy terms."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

from .config import (
    NEXT_BID,
    NUM_CARDS,
    SLOT_NEXT_ACTOR,
    SLOT_NEXT_PHASE,
    SeqModelConfig,
    SeqTrainingConfig,
)
from .rollout import SeqTree
from .tokens import TOKEN_WIDTH, build_oracle_tokens, build_seat_tokens, card_id


@dataclass
class PPOPolicyRows:
    seq_index: list[int] = field(default_factory=list)
    position: list[int] = field(default_factory=list)
    action: list[int] = field(default_factory=list)
    old_probs_full: list[np.ndarray] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    advantages: list[float] = field(default_factory=list)
    weight: list[float] = field(default_factory=list)
    # Address of this same pre-action state in the canonical oracle sequence.
    critic_group: list[int] = field(default_factory=list)
    critic_seq_index: list[int] = field(default_factory=list)
    critic_position: list[int] = field(default_factory=list)
    critic_seat: list[int] = field(default_factory=list)


@dataclass
class PPOTrainingGroup:
    policy_id: str
    num_players: int
    hand_size: int
    tokens: np.ndarray  # [B, L, TOKEN_WIDTH], one observer trajectory per row
    # Legacy observer-critic side input: [B, max_players, max_hand_size],
    # indexed by observer-relative owner and padded with NUM_CARDS. The oracle
    # critic instead consumes PPOCriticGroup.tokens.
    initial_hands: np.ndarray
    policy: dict[str, PPOPolicyRows]


@dataclass
class PPOCriticRows:
    seq_index: list[int] = field(default_factory=list)
    position: list[int] = field(default_factory=list)
    acting_seat: list[int] = field(default_factory=list)
    returns: list[np.ndarray] = field(default_factory=list)
    weight: list[float] = field(default_factory=list)


@dataclass
class PPOCriticGroup:
    num_players: int
    hand_size: int
    # [games, oracle_length, TOKEN_WIDTH], exactly one row per environment game.
    tokens: np.ndarray
    rows: PPOCriticRows


@dataclass
class PPOTrainingBatch:
    policy_groups: list[PPOTrainingGroup]
    critic_groups: list[PPOCriticGroup]


@dataclass
class PPOTerms:
    losses: torch.Tensor
    divergences: torch.Tensor
    entropy: torch.Tensor
    normalized_entropy: torch.Tensor
    entropy_eligible: torch.Tensor
    ratios: torch.Tensor
    clipped: torch.Tensor


def _learning_seats(tree: SeqTree, train: SeqTrainingConfig) -> tuple[int, ...]:
    if tree.arm == "self" and train.ppo_self_play_seats == "all":
        return tuple(range(tree.num_players))
    return (tree.focal,)


def _relative_initial_hands(
    tree: SeqTree, observer: int, config: SeqModelConfig
) -> np.ndarray:
    hands = np.full(
        (config.max_players, config.max_hand_size),
        NUM_CARDS,
        dtype=np.int64,
    )
    for absolute_seat, cards in tree.initial_hands.items():
        relative_seat = (absolute_seat - observer) % tree.num_players
        ids = sorted(card_id(card) for card in cards)
        hands[relative_seat, : len(ids)] = ids
    return hands


def build_ppo_training_batch(
    trees: list[SeqTree],
    model_config: SeqModelConfig,
    train_config: SeqTrainingConfig,
) -> PPOTrainingBatch:
    """Build one actor-observation row per learned seat of each sampled game.

    A game's policy weight is divided by its number of learned seats C_g, but
    not by its number of decisions. Consequently the requested objective is

        (1 / games) sum_g (1 / C_g) sum_i sum_t loss[g, i, t].

    Longer hands therefore contribute more decisions naturally, with no
    additional square-root or sequence-length weighting.
    """

    if train_config.policy_objective != "ppo":
        raise ValueError("PPO groups require policy_objective='ppo'.")
    game_importance = {
        id(tree): (
            float(tree.hand_size) ** train_config.shape_importance_exponent
            * float(tree.num_players) ** train_config.player_importance_exponent
        )
        for tree in trees
    }
    importance_total = sum(game_importance.values())
    if importance_total <= 0:
        raise ValueError("PPO game importance must have positive total mass.")

    games: list[tuple[SeqTree, object, tuple[int, ...], float]] = []
    entries: dict[
        tuple[str, int, int], list[tuple[SeqTree, object, int, float]]
    ] = defaultdict(list)
    for tree in trees:
        spines = [leaf for leaf in tree.leaves if leaf.on_policy_spine]
        if len(spines) != 1 or tree.leaf_total != 1:
            raise ValueError("PPO collection must contain one unbranched leaf per game.")
        leaf = spines[0]
        seats = _learning_seats(tree, train_config)
        seat_weight = game_importance[id(tree)] / importance_total / len(seats)
        games.append((tree, leaf, seats, seat_weight))
        for observer in seats:
            policy_id = tree.seat_policy_ids.get(observer)
            if policy_id is None:
                raise ValueError("A learned PPO seat has no trainable policy id.")
            entries[(policy_id, tree.num_players, tree.hand_size)].append(
                (tree, leaf, observer, seat_weight)
            )

    # Build the oracle half first so every actor row can store a checked,
    # integer address into its game's single canonical critic sequence.
    games_by_shape: dict[
        tuple[int, int], list[tuple[SeqTree, object, tuple[int, ...], float]]
    ] = defaultdict(list)
    for game in games:
        tree = game[0]
        games_by_shape[(tree.num_players, tree.hand_size)].append(game)

    critic_groups: list[PPOCriticGroup] = []
    critic_location: dict[int, tuple[int, int, int, int]] = {}
    for (players, hand_size), shape_games in sorted(games_by_shape.items()):
        length = model_config.oracle_seq_len(players, hand_size)
        tokens = np.empty(
            (len(shape_games), length, TOKEN_WIDTH), dtype=np.int64
        )
        critic_rows = PPOCriticRows()
        group_index = len(critic_groups)
        position_shift = (players - 1) * hand_size
        for seq_index, (tree, leaf, seats, seat_weight) in enumerate(shape_games):
            tokens[seq_index] = build_oracle_tokens(
                model_config,
                leaf.env.state.event_log,
                players,
                hand_size,
                tree.initial_hands,
                tree.bidding_start_player,
            )
            if leaf.terminal_rewards is None:
                raise ValueError("PPO leaf is missing terminal rewards.")
            returns = np.zeros(model_config.max_players, dtype=np.float32)
            returns[:players] = [
                float(leaf.terminal_rewards[seat]) for seat in range(players)
            ]
            for record in leaf.decisions:
                if record.seat not in seats:
                    continue
                position = record.position + position_shift
                token = tokens[seq_index, position]
                if (
                    int(token[SLOT_NEXT_ACTOR]) != record.seat
                    or int(token[SLOT_NEXT_PHASE]) != record.phase
                ):
                    raise AssertionError(
                        "Oracle value address is not tied to the acting seat/phase."
                    )
                if id(record) in critic_location:
                    raise AssertionError("A PPO decision was added to the critic twice.")
                critic_location[id(record)] = (
                    group_index,
                    seq_index,
                    position,
                    record.seat,
                )
                critic_rows.seq_index.append(seq_index)
                critic_rows.position.append(position)
                critic_rows.acting_seat.append(record.seat)
                critic_rows.returns.append(returns.copy())
                critic_rows.weight.append(seat_weight)
        critic_groups.append(
            PPOCriticGroup(
                num_players=players,
                hand_size=hand_size,
                tokens=tokens,
                rows=critic_rows,
            )
        )

    groups: list[PPOTrainingGroup] = []
    for (policy_id, players, hand_size), rows in sorted(entries.items()):
        length = model_config.seq_len(players, hand_size)
        tokens = np.empty((len(rows), length, TOKEN_WIDTH), dtype=np.int64)
        initial_hands = np.empty(
            (len(rows), model_config.max_players, model_config.max_hand_size),
            dtype=np.int64,
        )
        policy = {"bid": PPOPolicyRows(), "play": PPOPolicyRows()}
        for seq_index, (tree, leaf, observer, seat_weight) in enumerate(rows):
            tokens[seq_index] = build_seat_tokens(
                model_config,
                leaf.env.state.event_log,
                observer,
                players,
                hand_size,
                tree.initial_hands[observer],
                tree.bidding_start_player,
            )
            initial_hands[seq_index] = _relative_initial_hands(
                tree, observer, model_config
            )
            if leaf.terminal_rewards is None:
                raise ValueError("PPO leaf is missing terminal rewards.")
            terminal_return = float(leaf.terminal_rewards[observer])
            decisions = [record for record in leaf.decisions if record.seat == observer]
            for record in decisions:
                if record.policy_id != policy_id:
                    raise ValueError("Decision policy id disagrees with seat assignment.")
                phase = "bid" if record.phase == NEXT_BID else "play"
                target = policy[phase]
                target.seq_index.append(seq_index)
                target.position.append(record.position)
                target.action.append(record.action_index)
                target.old_probs_full.append(record.old_probs)
                target.returns.append(terminal_return)
                target.advantages.append(0.0)
                target.weight.append(seat_weight)
                try:
                    critic_group, critic_seq, critic_position, critic_seat = (
                        critic_location[id(record)]
                    )
                except KeyError as error:
                    raise AssertionError(
                        "Actor decision is absent from the oracle critic batch."
                    ) from error
                target.critic_group.append(critic_group)
                target.critic_seq_index.append(critic_seq)
                target.critic_position.append(critic_position)
                target.critic_seat.append(critic_seat)
        groups.append(
            PPOTrainingGroup(
                policy_id=policy_id,
                num_players=players,
                hand_size=hand_size,
                tokens=tokens,
                initial_hands=initial_hands,
                policy=policy,
            )
        )
    return PPOTrainingBatch(policy_groups=groups, critic_groups=critic_groups)


def build_ppo_training_groups(
    trees: list[SeqTree],
    model_config: SeqModelConfig,
    train_config: SeqTrainingConfig,
) -> list[PPOTrainingGroup]:
    """Compatibility wrapper returning only the actor-observation groups."""

    return build_ppo_training_batch(trees, model_config, train_config).policy_groups


def ppo_clipped_terms(
    logits: torch.Tensor,
    old_probs: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_ratio: float,
) -> PPOTerms:
    """Masked PPO terms; every probability-sensitive operation is fp32."""

    logits = logits.float()
    old_probs = old_probs.float()
    advantages = advantages.float().detach()
    legal = old_probs > 0
    log_probs = torch.log_softmax(
        logits.masked_fill(~legal, float("-inf")), dim=-1
    )
    safe_log_probs = torch.where(legal, log_probs, torch.zeros_like(log_probs))
    old_log_probs = torch.log(old_probs.clamp_min(1e-12))
    action_log_probs = log_probs.gather(1, actions[:, None]).squeeze(1)
    old_action_log_probs = old_log_probs.gather(1, actions[:, None]).squeeze(1)
    ratios = torch.exp(action_log_probs - old_action_log_probs)
    unclipped = ratios * advantages
    clipped_ratios = ratios.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    losses = -torch.minimum(unclipped, clipped_ratios * advantages)

    divergences = (
        old_probs * (old_log_probs - safe_log_probs) * legal.float()
    ).sum(dim=-1)
    probabilities = safe_log_probs.exp() * legal.float()
    entropy = -(probabilities * safe_log_probs).sum(dim=-1)
    legal_count = legal.sum(dim=-1)
    entropy_eligible = legal_count > 1
    denominator = legal_count.clamp_min(2).float().log()
    normalized_entropy = torch.where(
        entropy_eligible, entropy / denominator, torch.zeros_like(entropy)
    )
    clipped = (ratios - 1.0).abs() > clip_ratio
    return PPOTerms(
        losses=losses,
        divergences=divergences,
        entropy=entropy,
        normalized_entropy=normalized_entropy,
        entropy_eligible=entropy_eligible,
        ratios=ratios,
        clipped=clipped,
    )


def normalize_ppo_advantages(groups: list[PPOTrainingGroup]) -> tuple[float, float]:
    weighted_sum = 0.0
    weight_sum = 0.0
    for group in groups:
        for rows in group.policy.values():
            weighted_sum += math.fsum(
                advantage * weight
                for advantage, weight in zip(rows.advantages, rows.weight)
            )
            weight_sum += math.fsum(rows.weight)
    mean = weighted_sum / max(weight_sum, 1e-12)
    squared = 0.0
    for group in groups:
        for rows in group.policy.values():
            squared += math.fsum(
                weight * (advantage - mean) ** 2
                for advantage, weight in zip(rows.advantages, rows.weight)
            )
    std = math.sqrt(max(squared / max(weight_sum, 1e-12), 1e-12))
    for group in groups:
        for rows in group.policy.values():
            rows.advantages[:] = [
                (advantage - mean) / std for advantage in rows.advantages
            ]
    return mean, std


def ppo_rows_by_chunk(group: PPOTrainingGroup, chunks) -> list[dict[str, dict]]:
    fields = (
        "position",
        "action",
        "old_probs_full",
        "returns",
        "advantages",
        "weight",
    )
    out = []
    for start, stop in list(chunks):
        chunk: dict[str, dict[str, list]] = {}
        for phase, rows in group.policy.items():
            selected = {"seq_index": [], "row_index": []}
            for name in fields:
                selected[name] = []
            for index, seq_index in enumerate(rows.seq_index):
                if start <= seq_index < stop:
                    selected["seq_index"].append(seq_index - start)
                    selected["row_index"].append(index)
                    for name in fields:
                        selected[name].append(getattr(rows, name)[index])
            chunk[phase] = selected
        out.append(chunk)
    return out


def ppo_critic_rows_by_chunk(
    group: PPOCriticGroup, chunks
) -> list[dict[str, list]]:
    """Select oracle value rows for each contiguous game microbatch."""

    out: list[dict[str, list]] = []
    for start, stop in list(chunks):
        selected: dict[str, list] = {
            "seq_index": [],
            "position": [],
            "acting_seat": [],
            "returns": [],
            "weight": [],
        }
        for index, seq_index in enumerate(group.rows.seq_index):
            if start <= seq_index < stop:
                selected["seq_index"].append(seq_index - start)
                selected["position"].append(group.rows.position[index])
                selected["acting_seat"].append(group.rows.acting_seat[index])
                selected["returns"].append(group.rows.returns[index])
                selected["weight"].append(group.rows.weight[index])
        out.append(selected)
    return out
