"""Branch-free PPO batches and numerically explicit masked policy terms."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

from plump.state import EventType

from .config import (
    NEXT_BID,
    NEXT_PLAY,
    NUM_CARDS,
    SLOT_NEXT_ACTOR,
    SLOT_NEXT_PHASE,
    SLOT_REL_PLAYER,
    SLOT_TYPE,
    TOKEN_PLAY,
    SeqModelConfig,
    SeqTrainingConfig,
)
from .rollout import SeqTree
from .tokens import (
    IGNORE_LABEL,
    TOKEN_WIDTH,
    ReplayArrays,
    build_oracle_tokens,
    build_replay_arrays,
    build_seat_tokens,
    card_id,
)

PPO_BELIEF_STAGES = (0, 4, 8)


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
    # Dense belief supervision is normalized independently of policy rows:
    # each game has equal total mass, split across its learned seats and then
    # across the genuine (pre-bucketing) positions in that observer stream.
    position_weight: np.ndarray  # [B, L]
    # Terminal relative rewards on the observer-relative seat axis: entry r is
    # the seat r steps clockwise from the observer, so entry 0 is the observer
    # itself and equals that row's own return. A game-level quantity, supervised
    # at every causal token exactly like the oracle's returns.
    value_targets: np.ndarray  # float32 [B, max_players]
    # Length bucketing merges shapes, so a bucket can mix table sizes. The
    # per-seat value loss needs the row's own count to mask padded seats and to
    # project onto the zero-sum subspace.
    player_counts: np.ndarray  # int8 [B]
    trick_targets: np.ndarray  # [B, max_players]
    bid_hit_targets: np.ndarray  # [B, max_players], 1 if that seat hit
    trick_masks: np.ndarray  # [B, L, max_players, bid_count]
    # Observer-relative seat taking the next trick, IGNORE_LABEL outside the
    # play phase. Genuinely per-position, unlike the round-level targets above.
    next_winner_targets: np.ndarray  # [B, L]
    suit_targets: np.ndarray  # int8 [B, L, belief_opponents, 4]
    rank_boundary_targets: np.ndarray  # int8 [B,L,belief_opponents+1,4,2]
    # Readout positions before the observer's 1st, 5th, and 9th play. A value
    # of -1 means this is not a ten-card trajectory or the stage is absent.
    belief_stage_positions: np.ndarray  # [B, len(PPO_BELIEF_STAGES)]
    policy: dict[str, PPOPolicyRows]


@dataclass
class PPOCriticRows:
    seq_index: list[int] = field(default_factory=list)
    position: list[int] = field(default_factory=list)
    acting_seat: list[int] = field(default_factory=list)
    num_players: list[int] = field(default_factory=list)
    returns: list[np.ndarray] = field(default_factory=list)
    weight: list[float] = field(default_factory=list)


@dataclass
class PPOCriticGroup:
    num_players: int
    hand_size: int
    # [games, oracle_length, TOKEN_WIDTH], exactly one row per environment game.
    tokens: np.ndarray
    position_weight: np.ndarray  # [games, oracle_length]
    # Dense oracle targets are game-level quantities repeated conceptually at
    # every causal token.  Keep one canonical copy per game; the trainer uses
    # the sequence index to broadcast them without materializing B*L copies.
    player_counts: np.ndarray  # int8 [games]
    returns: np.ndarray  # float32 [games, max_players], absolute-seat axis
    trick_targets: np.ndarray  # [games, max_players], absolute-seat axis
    trick_masks: np.ndarray  # [games, oracle_length, max_players, bid_count]
    # Absolute seat taking the next trick, on oracle positions.  The canonical
    # replay uses observer 0, so its relative axis is already the absolute one.
    next_winner_targets: np.ndarray  # [games, oracle_length]
    # Full within-player play permutation and the number of cards already
    # played at each causal position.  Card id NUM_CARDS pads inactive slots.
    card_play_order: np.ndarray  # int8 [games, max_players, max_hand_size]
    cards_played: np.ndarray  # int8 [games, oracle_length, max_players]
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


def center_zero_sum_values(
    values: torch.Tensor,
    num_players: torch.Tensor,
) -> torch.Tensor:
    """Project active-seat values onto the exact zero-sum subspace.

    Plump's relative terminal rewards sum to zero for every realized game, so
    their conditional expectations must do the same.  Subtracting the active
    mean is the orthogonal projection under the critic's equal-per-seat MSE;
    padded seat columns are returned as zero and never enter the loss.
    """

    if values.ndim != 2:
        raise ValueError("Oracle values must be [rows, max_players].")
    if num_players.ndim != 1 or num_players.shape[0] != values.shape[0]:
        raise ValueError("num_players must contain one count per value row.")
    if torch.any(num_players < 2) or torch.any(num_players > values.shape[1]):
        raise ValueError("num_players is outside the oracle value width.")
    active = (
        torch.arange(values.shape[1], device=values.device)[None, :]
        < num_players[:, None]
    )
    active_float = active.to(values.dtype)
    mean = (values * active_float).sum(dim=-1, keepdim=True) / num_players[
        :, None
    ].to(values.dtype)
    return (values - mean) * active_float


def oracle_gae_advantages(
    tokens: np.ndarray,
    values: np.ndarray,
    terminal_returns: np.ndarray,
    player_counts: np.ndarray,
    gae_lambda: float,
) -> np.ndarray:
    """Compute undiscounted oracle GAE over actual game transitions.

    A transition runs from one genuine decision readout to the next: the
    chosen action and any automatic bookkeeping (including a trick-winner
    event or TURN token) have happened, but the next action has not. The last
    action transitions directly to ``terminal_returns``. Rewards are zero on
    every other transition, and Plump's episodic discount is one.

    The recurrence is evaluated for every active seat, then the trainer reads
    the component belonging to the acting seat. At lambda=1 it telescopes to
    ``terminal_return - V(s_t)``; at lambda=0 it is one-step TD.
    """

    if tokens.ndim != 3 or tokens.shape[2] <= SLOT_NEXT_PHASE:
        raise ValueError("tokens must be [games, positions, token_width].")
    if values.ndim != 3 or values.shape[:2] != tokens.shape[:2]:
        raise ValueError("values must be [games, positions, players].")
    if terminal_returns.shape != (values.shape[0], values.shape[2]):
        raise ValueError("terminal_returns must match [games, players].")
    if player_counts.shape != (values.shape[0],):
        raise ValueError("player_counts must contain one count per game.")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1].")

    advantages = np.zeros_like(values)
    phases = tokens[:, :, SLOT_NEXT_PHASE]
    decision_mask = (phases == NEXT_BID) | (phases == NEXT_PLAY)
    for game in range(values.shape[0]):
        players = int(player_counts[game])
        if players < 1 or players > values.shape[2]:
            raise ValueError("player_counts is outside the value width.")
        positions = np.flatnonzero(decision_mask[game])
        if positions.size == 0:
            raise ValueError("Every oracle game must contain a decision state.")
        next_value = terminal_returns[game, :players]
        running = np.zeros(players, dtype=values.dtype)
        for position in positions[::-1]:
            current_value = values[game, position, :players]
            td_error = next_value - current_value
            running = td_error + gae_lambda * running
            advantages[game, position, :players] = running
            next_value = current_value
    return advantages


def plackett_luce_card_order_terms(
    logits: torch.Tensor,
    target_order: torch.Tensor,
    cards_played: torch.Tensor,
    active_players: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalized listwise loss and next-card accuracy for oracle order.

    ``logits[r, c]`` is one contextual utility score for exact card ``c`` at
    state ``r``. The deal's disjoint hands partition those scores by absolute
    player; a ``[rows, players, 52]`` tensor is also accepted for focused
    tests. ``target_order`` lists each player's complete realized within-hand
    play order. Conditioning on the
    already-played prefix gives the Plackett-Luce probability

        prod_j exp(s[c_j]) / sum_{k >= j} exp(s[c_k]).

    It is a genuine distribution over permutations: every remaining card is
    selected exactly once, with no duplicate or missing order slots.  The NLL
    is averaged over non-trivial selections and then over active players so
    hand size and table width do not silently change this auxiliary's scale.
    Players with zero or one card remaining have no ordering question and are
    omitted.  The returned accuracy asks whether the largest remaining-card
    score identifies the next card in the realized order.
    """

    if logits.ndim not in (2, 3) or logits.shape[-1] != NUM_CARDS:
        raise ValueError("card-order logits must end in the 52-card axis.")
    if target_order.ndim != 3 or target_order.shape[0] != logits.shape[0]:
        raise ValueError("card-order targets must be [rows, players, slots].")
    if logits.ndim == 2:
        logits = logits[:, None, :].expand(-1, target_order.shape[1], -1)
    elif logits.shape[1] != target_order.shape[1]:
        raise ValueError("card-order player axes do not match.")
    if cards_played.shape != target_order.shape[:2]:
        raise ValueError("cards_played must be [rows, players].")
    if active_players.shape != target_order.shape[:2]:
        raise ValueError("active_players must be [rows, players].")

    logits = logits.float()
    target_order = target_order.long()
    cards_played = cards_played.long()
    valid_slot = target_order < NUM_CARDS
    safe_target = target_order.clamp(min=0, max=NUM_CARDS - 1)
    ordered_scores = logits.gather(-1, safe_target).masked_fill(
        ~valid_slot, -1e9
    )
    suffix_logsumexp = torch.logcumsumexp(
        ordered_scores.flip(-1), dim=-1
    ).flip(-1)

    slot = torch.arange(
        target_order.shape[-1], device=logits.device
    ).view(1, 1, -1)
    card_count = valid_slot.sum(dim=-1)
    remaining_count = (card_count - cards_played).clamp_min(0)
    choice_count = (remaining_count - 1).clamp_min(0)
    choice = (
        active_players[..., None]
        & (slot >= cards_played[..., None])
        & (slot < (card_count - 1)[..., None])
    )
    nll = torch.where(
        choice,
        suffix_logsumexp - ordered_scores,
        torch.zeros_like(ordered_scores),
    )
    per_player = nll.sum(dim=-1) / choice_count.clamp_min(1).to(nll.dtype)
    labeled_player = active_players & (choice_count > 0)
    per_state = (
        (per_player * labeled_player.to(per_player.dtype)).sum(dim=-1)
        / labeled_player.sum(dim=-1).clamp_min(1).to(per_player.dtype)
    )

    remaining_slot = valid_slot & (slot >= cards_played[..., None])
    predicted_slot = ordered_scores.masked_fill(
        ~remaining_slot, -1e9
    ).argmax(dim=-1)
    correct = (predicted_slot == cards_played) & labeled_player
    return per_state, correct, labeled_player


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
    dense_observer_zero_replays: dict[int, ReplayArrays] = {}
    for (players, hand_size), shape_games in sorted(games_by_shape.items()):
        length = model_config.oracle_seq_len(players, hand_size)
        tokens = np.empty(
            (len(shape_games), length, TOKEN_WIDTH), dtype=np.int64
        )
        position_weight = np.zeros((len(shape_games), length), dtype=np.float32)
        player_counts = np.full(len(shape_games), players, dtype=np.int8)
        game_returns = np.zeros(
            (len(shape_games), model_config.max_players), dtype=np.float32
        )
        trick_targets = np.full(
            (len(shape_games), model_config.max_players),
            IGNORE_LABEL,
            dtype=np.int64,
        )
        trick_masks = np.zeros(
            (
                len(shape_games),
                length,
                model_config.max_players,
                model_config.bid_count,
            ),
            dtype=bool,
        )
        next_winner_targets = np.full(
            (len(shape_games), length), IGNORE_LABEL, dtype=np.int64
        )
        card_play_order = np.full(
            (
                len(shape_games),
                model_config.max_players,
                model_config.max_hand_size,
            ),
            NUM_CARDS,
            dtype=np.int8,
        )
        cards_played = np.zeros(
            (len(shape_games), length, model_config.max_players),
            dtype=np.int8,
        )
        critic_rows = PPOCriticRows()
        group_index = len(critic_groups)
        position_shift = (players - 1) * hand_size
        dense_start = train_config.dense_start_position(
            model_config,
            num_players=players,
            hand_size=hand_size,
            oracle=True,
        )
        dense_positions = length - dense_start
        if dense_positions <= 0:
            raise AssertionError("Oracle dense-supervision window is empty.")
        for seq_index, (tree, leaf, seats, seat_weight) in enumerate(shape_games):
            tokens[seq_index] = build_oracle_tokens(
                model_config,
                leaf.env.state.event_log,
                players,
                hand_size,
                tree.initial_hands,
                tree.bidding_start_player,
            )
            if train_config.oracle_card_order_coef > 0:
                next_slot = np.zeros(model_config.max_players, dtype=np.int64)
                for event in leaf.env.state.event_log:
                    if event.type != EventType.PLAY:
                        continue
                    slot = next_slot[event.player]
                    if slot >= hand_size:
                        raise AssertionError(
                            "Oracle card-order target exceeds the dealt hand."
                        )
                    card_play_order[seq_index, event.player, slot] = card_id(
                        event.card
                    )
                    next_slot[event.player] += 1
                if not np.all(next_slot[:players] == hand_size):
                    raise AssertionError(
                        "Oracle card-order target does not cover every play."
                    )

                # A hidden row includes its own token.  Increment before
                # storing so the readout on a PLAY token treats that public
                # card as already played; the next action then receives the
                # exact remaining suffix of each player's permutation.
                running = np.zeros(model_config.max_players, dtype=np.int8)
                for position, token in enumerate(tokens[seq_index]):
                    if int(token[SLOT_TYPE]) == TOKEN_PLAY:
                        player = int(token[SLOT_REL_PLAYER])
                        if player >= players:
                            raise AssertionError(
                                "Oracle PLAY token names an inactive seat."
                            )
                        running[player] += 1
                    cards_played[seq_index, position] = running
            actor_tokens = build_seat_tokens(
                model_config,
                leaf.env.state.event_log,
                0,
                players,
                hand_size,
                tree.initial_hands[0],
                tree.bidding_start_player,
            )
            dense_auxiliary = train_config.ppo_auxiliary_positions == "all"
            cache_observer_zero = dense_auxiliary and 0 in seats
            arrays = build_replay_arrays(
                model_config,
                tree.initial_hands,
                leaf.env.state.event_log,
                0,
                players,
                hand_size,
                tree.bidding_start_player,
                tokens=actor_tokens,
                suit_labels=(cache_observer_zero and train_config.suit_coef > 0),
                trick_labels=train_config.trick_coef > 0,
                rank_boundary_labels=(
                    cache_observer_zero and train_config.rank_boundary_coef > 0
                ),
                next_winner_labels=train_config.next_winner_coef > 0,
                label_decision_seats=None if dense_auxiliary else seats,
            )
            if cache_observer_zero:
                dense_observer_zero_replays[id(tree)] = arrays
            trick_targets[seq_index] = arrays.trick_targets
            # The canonical actor and oracle streams share the public suffix.
            # The oracle merely replaces the observer's N-card hand prefix by
            # P*N owner-tagged card tokens, all of which describe the same
            # post-deal state -- so a dense per-position label is broadcast
            # over the whole oracle prefix and copied verbatim after it. In
            # decision-only mode the labels are sparse, and every address
            # simply shifts by the oracle's extra (P - 1) * N HAND tokens.
            actor_prefix = model_config.prefix_len(hand_size)
            oracle_prefix = actor_prefix + position_shift
            actor_length = arrays.tokens.shape[0]

            def place(destination, source) -> None:
                if dense_auxiliary:
                    destination[seq_index, :oracle_prefix] = source[0]
                    destination[seq_index, oracle_prefix:] = source[actor_prefix:]
                else:
                    destination[
                        seq_index,
                        position_shift : position_shift + actor_length,
                    ] = source

            if train_config.trick_coef > 0:
                place(trick_masks, arrays.trick_masks)
            if train_config.next_winner_coef > 0:
                place(next_winner_targets, arrays.next_winner_targets)
            game_weight = seat_weight * len(seats)
            position_weight[seq_index, dense_start:] = (
                game_weight / float(dense_positions)
            )
            if leaf.terminal_rewards is None:
                raise ValueError("PPO leaf is missing terminal rewards.")
            terminal_returns = np.zeros(
                model_config.max_players, dtype=np.float32
            )
            terminal_returns[:players] = [
                float(leaf.terminal_rewards[seat]) for seat in range(players)
            ]
            game_returns[seq_index] = terminal_returns
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
                critic_rows.num_players.append(players)
                critic_rows.returns.append(terminal_returns.copy())
                critic_rows.weight.append(seat_weight)
        critic_groups.append(
            PPOCriticGroup(
                num_players=players,
                hand_size=hand_size,
                tokens=tokens,
                position_weight=position_weight,
                player_counts=player_counts,
                returns=game_returns,
                trick_targets=trick_targets,
                trick_masks=trick_masks,
                next_winner_targets=next_winner_targets,
                card_play_order=card_play_order,
                cards_played=cards_played,
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
        position_weight = np.zeros((len(rows), length), dtype=np.float32)
        value_targets = np.zeros(
            (len(rows), model_config.max_players), dtype=np.float32
        )
        player_counts = np.full(len(rows), players, dtype=np.int8)
        trick_targets = np.full(
            (len(rows), model_config.max_players), IGNORE_LABEL, dtype=np.int64
        )
        bid_hit_targets = np.full(
            (len(rows), model_config.max_players), IGNORE_LABEL, dtype=np.int64
        )
        trick_masks = np.zeros(
            (
                len(rows),
                length,
                model_config.max_players,
                model_config.bid_count,
            ),
            dtype=bool,
        )
        next_winner_targets = np.full(
            (len(rows), length), IGNORE_LABEL, dtype=np.int64
        )
        suit_targets = np.full(
            (len(rows), length, model_config.belief_opponents, 4),
            IGNORE_LABEL,
            dtype=np.int8,
        )
        rank_boundary_targets = np.full(
            (
                len(rows),
                length,
                model_config.belief_opponents + 1,
                4,
                2,
            ),
            IGNORE_LABEL,
            dtype=np.int8,
        )
        belief_stage_positions = np.full(
            (len(rows), len(PPO_BELIEF_STAGES)), -1, dtype=np.int64
        )
        policy = {"bid": PPOPolicyRows(), "play": PPOPolicyRows()}
        dense_start = train_config.dense_start_position(
            model_config,
            num_players=players,
            hand_size=hand_size,
        )
        dense_positions = length - dense_start
        if dense_positions <= 0:
            raise AssertionError("Actor dense-supervision window is empty.")
        for seq_index, (tree, leaf, observer, seat_weight) in enumerate(rows):
            arrays = (
                dense_observer_zero_replays.get(id(tree))
                if observer == 0
                else None
            )
            if arrays is None:
                tokens[seq_index] = build_seat_tokens(
                    model_config,
                    leaf.env.state.event_log,
                    observer,
                    players,
                    hand_size,
                    tree.initial_hands[observer],
                    tree.bidding_start_player,
                )
                arrays = build_replay_arrays(
                    model_config,
                    tree.initial_hands,
                    leaf.env.state.event_log,
                    observer,
                    players,
                    hand_size,
                    tree.bidding_start_player,
                    tokens=tokens[seq_index],
                    suit_labels=train_config.suit_coef > 0,
                    trick_labels=train_config.trick_coef > 0,
                    rank_boundary_labels=train_config.rank_boundary_coef > 0,
                    next_winner_labels=train_config.next_winner_coef > 0,
                    label_decision_seats=(
                        None
                        if train_config.ppo_auxiliary_positions == "all"
                        else (observer,)
                    ),
                )
            else:
                tokens[seq_index] = arrays.tokens
            initial_hands[seq_index] = _relative_initial_hands(
                tree, observer, model_config
            )
            # Auxiliary loss scale is invariant to the supervised window's
            # length while the requested PPO policy objective still grows with
            # decision count. Prefix and bucket-padding positions stay zero.
            position_weight[seq_index, dense_start:] = (
                seat_weight / float(dense_positions)
            )
            trick_targets[seq_index] = arrays.trick_targets
            bid_hit_targets[seq_index] = arrays.bid_hit_targets
            trick_masks[seq_index] = arrays.trick_masks
            next_winner_targets[seq_index] = arrays.next_winner_targets
            suit_targets[seq_index] = arrays.suit_targets
            rank_boundary_targets[seq_index] = arrays.rank_boundary_targets
            if leaf.terminal_rewards is None:
                raise ValueError("PPO leaf is missing terminal rewards.")
            terminal_return = float(leaf.terminal_rewards[observer])
            for absolute_seat in range(players):
                value_targets[
                    seq_index, (absolute_seat - observer) % players
                ] = float(leaf.terminal_rewards[absolute_seat])
            decisions = [record for record in leaf.decisions if record.seat == observer]
            if hand_size == 10:
                play_decisions = [
                    record for record in decisions if record.phase != NEXT_BID
                ]
                for stage_index, cards_played in enumerate(PPO_BELIEF_STAGES):
                    if cards_played < len(play_decisions):
                        belief_stage_positions[seq_index, stage_index] = (
                            play_decisions[cards_played].position
                        )
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
                position_weight=position_weight,
                value_targets=value_targets,
                player_counts=player_counts,
                trick_targets=trick_targets,
                bid_hit_targets=bid_hit_targets,
                trick_masks=trick_masks,
                next_winner_targets=next_winner_targets,
                suit_targets=suit_targets,
                rank_boundary_targets=rank_boundary_targets,
                belief_stage_positions=belief_stage_positions,
                policy=policy,
            )
        )
    batch = PPOTrainingBatch(policy_groups=groups, critic_groups=critic_groups)
    return bucket_ppo_training_batch(
        batch,
        width=train_config.ppo_sequence_bucket_width,
        critic_width=train_config.ppo_critic_sequence_bucket_width,
        model_config=model_config,
    )


def _bucket_length(length: int, width: int, capacity: int) -> int:
    if width <= 0:
        return length
    return min(((length + width - 1) // width) * width, capacity)


def bucket_ppo_training_batch(
    batch: PPOTrainingBatch,
    *,
    width: int,
    critic_width: int | None = None,
    model_config: SeqModelConfig,
) -> PPOTrainingBatch:
    """Merge shape groups after tail-padding to nearby sequence lengths.

    Every PPO readout precedes the appended padding, so causal attention makes
    the operation exactly loss/logit preserving. The larger rectangular
    batches trade a small amount of padded FLOPs for substantially fewer
    transformer launches and better accelerator utilization.
    """

    if critic_width is None:
        critic_width = width
    if width <= 0 and critic_width <= 0:
        return batch

    critic_buckets: dict[int, list[tuple[int, PPOCriticGroup]]] = defaultdict(list)
    if critic_width > 0:
        for old_index, group in enumerate(batch.critic_groups):
            length = _bucket_length(
                group.tokens.shape[1],
                critic_width,
                model_config.oracle_max_seq_len,
            )
            critic_buckets[length].append((old_index, group))
    else:
        for old_index, group in enumerate(batch.critic_groups):
            # Keep exact groups distinct when critic bucketing is disabled.
            critic_buckets[-old_index - 1].append((old_index, group))

    critic_groups: list[PPOCriticGroup] = []
    critic_address: dict[tuple[int, int], tuple[int, int]] = {}
    critic_fields = tuple(PPOCriticRows.__dataclass_fields__)
    for bucket_key, sources in sorted(critic_buckets.items()):
        length = (
            sources[0][1].tokens.shape[1]
            if bucket_key < 0
            else bucket_key
        )
        total = sum(group.tokens.shape[0] for _, group in sources)
        tokens = np.zeros((total, length, TOKEN_WIDTH), dtype=np.int64)
        position_weight = np.zeros((total, length), dtype=np.float32)
        player_counts = np.zeros(total, dtype=np.int8)
        returns = np.zeros(
            (total, model_config.max_players), dtype=np.float32
        )
        trick_targets = np.full(
            (total, model_config.max_players), IGNORE_LABEL, dtype=np.int64
        )
        trick_masks = np.zeros(
            (total, length, model_config.max_players, model_config.bid_count),
            dtype=bool,
        )
        next_winner_targets = np.full(
            (total, length), IGNORE_LABEL, dtype=np.int64
        )
        card_play_order = np.full(
            (total, model_config.max_players, model_config.max_hand_size),
            NUM_CARDS,
            dtype=np.int8,
        )
        cards_played = np.zeros(
            (total, length, model_config.max_players), dtype=np.int8
        )
        rows = PPOCriticRows()
        offset = 0
        new_group_index = len(critic_groups)
        for old_group_index, group in sources:
            count, source_length = group.tokens.shape[:2]
            tokens[offset : offset + count, :source_length] = group.tokens
            position_weight[offset : offset + count, :source_length] = (
                group.position_weight
            )
            player_counts[offset : offset + count] = group.player_counts
            returns[offset : offset + count] = group.returns
            trick_targets[offset : offset + count] = group.trick_targets
            trick_masks[offset : offset + count, :source_length] = group.trick_masks
            next_winner_targets[offset : offset + count, :source_length] = (
                group.next_winner_targets
            )
            card_play_order[offset : offset + count] = group.card_play_order
            cards_played[offset : offset + count, :source_length] = (
                group.cards_played
            )
            for seq_index in range(count):
                critic_address[(old_group_index, seq_index)] = (
                    new_group_index,
                    offset + seq_index,
                )
            for name in critic_fields:
                values = getattr(group.rows, name)
                if name == "seq_index":
                    getattr(rows, name).extend(value + offset for value in values)
                else:
                    getattr(rows, name).extend(values)
            offset += count
        critic_groups.append(
            PPOCriticGroup(
                num_players=max(group.num_players for _, group in sources),
                hand_size=max(group.hand_size for _, group in sources),
                tokens=tokens,
                position_weight=position_weight,
                player_counts=player_counts,
                returns=returns,
                trick_targets=trick_targets,
                trick_masks=trick_masks,
                next_winner_targets=next_winner_targets,
                card_play_order=card_play_order,
                cards_played=cards_played,
                rows=rows,
            )
        )

    # Redirect actor decision rows to their game's new critic bucket address.
    for group in batch.policy_groups:
        for rows in group.policy.values():
            for index in range(len(rows.weight)):
                address = (
                    rows.critic_group[index],
                    rows.critic_seq_index[index],
                )
                new_group, new_seq = critic_address[address]
                rows.critic_group[index] = new_group
                rows.critic_seq_index[index] = new_seq

    if width <= 0:
        return PPOTrainingBatch(
            policy_groups=batch.policy_groups,
            critic_groups=critic_groups,
        )

    policy_buckets: dict[
        tuple[str, int], list[PPOTrainingGroup]
    ] = defaultdict(list)
    for group in batch.policy_groups:
        length = _bucket_length(
            group.tokens.shape[1], width, model_config.max_seq_len
        )
        policy_buckets[(group.policy_id, length)].append(group)

    policy_groups: list[PPOTrainingGroup] = []
    policy_fields = tuple(PPOPolicyRows.__dataclass_fields__)
    for (policy_id, length), sources in sorted(policy_buckets.items()):
        total = sum(group.tokens.shape[0] for group in sources)
        tokens = np.zeros((total, length, TOKEN_WIDTH), dtype=np.int64)
        initial_hands = np.empty(
            (total, model_config.max_players, model_config.max_hand_size),
            dtype=np.int64,
        )
        position_weight = np.zeros((total, length), dtype=np.float32)
        value_targets = np.zeros(
            (total, model_config.max_players), dtype=np.float32
        )
        player_counts = np.zeros(total, dtype=np.int8)
        trick_targets = np.full(
            (total, model_config.max_players), IGNORE_LABEL, dtype=np.int64
        )
        bid_hit_targets = np.full(
            (total, model_config.max_players), IGNORE_LABEL, dtype=np.int64
        )
        trick_masks = np.zeros(
            (total, length, model_config.max_players, model_config.bid_count),
            dtype=bool,
        )
        next_winner_targets = np.full(
            (total, length), IGNORE_LABEL, dtype=np.int64
        )
        suit_targets = np.full(
            (total, length, model_config.belief_opponents, 4),
            IGNORE_LABEL,
            dtype=np.int8,
        )
        rank_boundary_targets = np.full(
            (
                total,
                length,
                model_config.belief_opponents + 1,
                4,
                2,
            ),
            IGNORE_LABEL,
            dtype=np.int8,
        )
        belief_stage_positions = np.full(
            (total, len(PPO_BELIEF_STAGES)), -1, dtype=np.int64
        )
        policy = {"bid": PPOPolicyRows(), "play": PPOPolicyRows()}
        offset = 0
        for group in sources:
            count, source_length = group.tokens.shape[:2]
            tokens[offset : offset + count, :source_length] = group.tokens
            initial_hands[offset : offset + count] = group.initial_hands
            position_weight[offset : offset + count, :source_length] = (
                group.position_weight
            )
            value_targets[offset : offset + count] = group.value_targets
            player_counts[offset : offset + count] = group.player_counts
            trick_targets[offset : offset + count] = group.trick_targets
            bid_hit_targets[offset : offset + count] = group.bid_hit_targets
            trick_masks[offset : offset + count, :source_length] = group.trick_masks
            next_winner_targets[offset : offset + count, :source_length] = (
                group.next_winner_targets
            )
            suit_targets[offset : offset + count, :source_length] = group.suit_targets
            rank_boundary_targets[
                offset : offset + count, :source_length
            ] = group.rank_boundary_targets
            belief_stage_positions[offset : offset + count] = (
                group.belief_stage_positions
            )
            for phase in policy:
                source_rows = group.policy[phase]
                target_rows = policy[phase]
                for name in policy_fields:
                    values = getattr(source_rows, name)
                    if name == "seq_index":
                        getattr(target_rows, name).extend(
                            value + offset for value in values
                        )
                    else:
                        getattr(target_rows, name).extend(values)
            offset += count
        policy_groups.append(
            PPOTrainingGroup(
                policy_id=policy_id,
                num_players=max(group.num_players for group in sources),
                hand_size=max(group.hand_size for group in sources),
                tokens=tokens,
                initial_hands=initial_hands,
                position_weight=position_weight,
                value_targets=value_targets,
                player_counts=player_counts,
                trick_targets=trick_targets,
                bid_hit_targets=bid_hit_targets,
                trick_masks=trick_masks,
                next_winner_targets=next_winner_targets,
                suit_targets=suit_targets,
                rank_boundary_targets=rank_boundary_targets,
                belief_stage_positions=belief_stage_positions,
                policy=policy,
            )
        )
    return PPOTrainingBatch(
        policy_groups=policy_groups,
        critic_groups=critic_groups,
    )


def build_ppo_training_groups(
    trees: list[SeqTree],
    model_config: SeqModelConfig,
    train_config: SeqTrainingConfig,
) -> list[PPOTrainingGroup]:
    """Compatibility wrapper returning only the actor-observation groups."""

    return build_ppo_training_batch(trees, model_config, train_config).policy_groups


def ppo_policy_kl(logits: torch.Tensor, old_probs: torch.Tensor) -> torch.Tensor:
    """KL(old || new) without unused loss, ratio, or entropy readouts."""

    old_probs = old_probs.float()
    legal = old_probs > 0
    log_probs = torch.log_softmax(
        logits.float().masked_fill(~legal, float("-inf")), dim=-1
    )
    safe_log_probs = torch.where(legal, log_probs, torch.zeros_like(log_probs))
    old_log_probs = torch.log(old_probs.clamp_min(1e-12))
    return (
        old_probs * (old_log_probs - safe_log_probs) * legal.float()
    ).sum(dim=-1)


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
            "num_players": [],
            "returns": [],
            "weight": [],
        }
        for index, seq_index in enumerate(group.rows.seq_index):
            if start <= seq_index < stop:
                selected["seq_index"].append(seq_index - start)
                selected["position"].append(group.rows.position[index])
                selected["acting_seat"].append(group.rows.acting_seat[index])
                selected["num_players"].append(group.rows.num_players[index])
                selected["returns"].append(group.rows.returns[index])
                selected["weight"].append(group.rows.weight[index])
        out.append(selected)
    return out
