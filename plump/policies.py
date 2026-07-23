"""Reusable action-policy interfaces for training, evaluation, search, and GUI play."""

from __future__ import annotations

import random
from collections import defaultdict
from math import comb, exp
from pathlib import Path
from typing import Literal, Protocol, Sequence

import numpy as np
import torch
from torch.distributions import Categorical

from plump.cards import Card, Rank, Suit
from plump.env import PlumpEnv
from plump.modeling import ModelConfig, SCHEMA_VERSION, card_from_id, encode_observation
from plump.modeling.torch_model import (
    PlumpTransformerModel,
    PlumpSearchModel,
    best_torch_device,
    combined_action_logits,
    encoded_observations_to_batch,
    load_v2_weights,
    load_v3_weights,
    model_autocast,
    slice_model_output,
)
from plump.rules import determine_trick_winner
from plump.state import BidAction, Observation, Phase, PlayCardAction, Trick, TrickPlay


class ActionPolicy(Protocol):
    name: str
    forward_passes: int

    def act(self, env: PlumpEnv, *, rng: random.Random | None = None) -> BidAction | PlayCardAction:
        ...

    def reset_counters(self) -> None:
        ...


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.forward_passes = 0

    def act(self, env: PlumpEnv, *, rng: random.Random | None = None) -> BidAction | PlayCardAction:
        chooser = rng or self.rng
        return chooser.choice(env.legal_actions())

    def reset_counters(self) -> None:
        self.forward_passes = 0


class HeuristicPolicy:
    """Small deterministic baseline that bids strength and plays toward its bid."""

    name = "heuristic"
    bid_signal_strength = 0.12
    max_bid_signal = 2.0

    def __init__(self) -> None:
        self.forward_passes = 0

    def act(self, env: PlumpEnv, *, rng: random.Random | None = None) -> BidAction | PlayCardAction:
        player = env.current_player()
        observation = env.get_observation(player)
        if env.phase() == Phase.BIDDING:
            card_distribution = _rank_only_trick_distribution(
                observation.my_hand,
                num_players=env.config.num_players,
            )
            card_only_bid = _select_expected_score_bid(
                card_distribution,
                observation.legal_bids,
            )
            adjusted_distribution = _adjust_distribution_for_prior_bids(
                card_distribution,
                prior_bids=[bid.value for bid in observation.bids],
                hand_size=observation.hand_size,
                num_players=env.config.num_players,
                strength=self.bid_signal_strength,
                max_signal=self.max_bid_signal,
            )
            nearby_legal_bids = [
                value
                for value in observation.legal_bids
                if abs(value - card_only_bid) <= 1
            ]
            bid = _select_expected_score_bid(adjusted_distribution, nearby_legal_bids)
            return BidAction(player, bid)
        if env.phase() == Phase.PLAYING:
            card = _select_heuristic_play(
                observation,
                num_players=env.config.num_players,
            )
            return PlayCardAction(player, card)
        raise RuntimeError(f"Cannot act in phase {env.phase().value}.")

    def reset_counters(self) -> None:
        self.forward_passes = 0


def _rank_only_trick_distribution(
    hand: list[Card],
    *,
    num_players: int,
) -> dict[int, float]:
    """Exact trick distribution when only same-suit rank strength matters."""

    hand_size = len(hand)
    unknown_count = 52 - hand_size
    opponent_card_count = (num_players - 1) * hand_size
    if opponent_card_count > unknown_count:
        raise ValueError("The hand does not fit the requested player count.")

    hand_ranks = {
        suit: {int(card.rank) for card in hand if card.suit == suit}
        for suit in Suit
    }
    # State is (opponent cards assigned, unbeaten cards in our hand) -> deal count.
    distribution_counts: dict[tuple[int, int], int] = {(0, 0): 1}
    for suit in Suit:
        own = hand_ranks[suit]
        unknown = [int(rank) for rank in Rank if int(rank) not in own]
        local_counts: dict[tuple[int, int], int] = defaultdict(int)
        for selected in range(min(len(unknown), opponent_card_count) + 1):
            if selected == 0:
                local_counts[(0, len(own))] = 1
                continue
            for max_index, opponent_highest in enumerate(unknown):
                if max_index < selected - 1:
                    continue
                winners = sum(rank > opponent_highest for rank in own)
                local_counts[(selected, winners)] += comb(max_index, selected - 1)

        next_counts: dict[tuple[int, int], int] = defaultdict(int)
        for (assigned, winners), ways in distribution_counts.items():
            for (local_assigned, local_winners), local_ways in local_counts.items():
                total_assigned = assigned + local_assigned
                if total_assigned <= opponent_card_count:
                    next_counts[(total_assigned, winners + local_winners)] += ways * local_ways
        distribution_counts = next_counts

    total_deals = comb(unknown_count, opponent_card_count)
    probabilities: dict[int, float] = defaultdict(float)
    for (assigned, winners), ways in distribution_counts.items():
        if assigned == opponent_card_count:
            probabilities[winners] += ways / total_deals
    return dict(probabilities)


def _adjust_distribution_for_prior_bids(
    distribution: dict[int, float],
    *,
    prior_bids: list[int],
    hand_size: int,
    num_players: int,
    strength: float,
    max_signal: float,
) -> dict[int, float]:
    """Weakly condition rank strength on whether earlier bids look high or low."""

    if not prior_bids:
        return distribution
    expected_prior_total = len(prior_bids) * hand_size / num_players
    signal = sum(prior_bids) - expected_prior_total
    signal = max(-max_signal, min(max_signal, signal))
    if abs(signal) <= 1e-12:
        return distribution

    weighted = {
        tricks: probability * exp(-strength * signal * tricks)
        for tricks, probability in distribution.items()
    }
    normalizer = sum(weighted.values())
    return {
        tricks: probability / normalizer
        for tricks, probability in weighted.items()
    }


def _select_expected_score_bid(
    distribution: dict[int, float],
    legal_bids: list[int],
) -> int:
    """Choose the legal bid with the highest expected exact-hit score."""

    expected_tricks = sum(tricks * probability for tricks, probability in distribution.items())
    return max(
        legal_bids,
        key=lambda value: (
            distribution.get(value, 0.0) * (5 if value == 0 else 10 + value),
            -abs(value - expected_tricks),
            -value,
        ),
    )


def _select_heuristic_play(
    observation: Observation,
    *,
    num_players: int,
) -> Card:
    """Play toward the bid, or disrupt opponents once our bid is lost."""

    player = observation.player_id
    bid_by_player = {bid.player: bid.value for bid in observation.bids}
    own_bid = bid_by_player[player]
    tricks_won = observation.tricks_won.get(player, 0)
    remaining_tricks = len(observation.my_hand)
    gone_over = tricks_won > own_bid
    cannot_reach = tricks_won + remaining_tricks < own_bid

    if gone_over or cannot_reach:
        total_bid = sum(bid_by_player.values())
        if gone_over and total_bid > observation.hand_size:
            wants_trick = True
        elif cannot_reach and total_bid < observation.hand_size:
            wants_trick = False
        else:
            current_winner = _current_trick_winner(
                observation.current_trick,
                observation.trump_suit,
            )
            wants_trick = _lost_player_should_take(
                player=player,
                bids=bid_by_player,
                tricks_won=observation.tricks_won,
                remaining_tricks=remaining_tricks,
                current_winner=current_winner,
                num_players=num_players,
                total_round_tricks=observation.hand_size,
            )
    else:
        wants_trick = tricks_won < own_bid

    return _select_card_for_intent(
        observation.legal_cards,
        player=player,
        current_trick=observation.current_trick,
        trump_suit=observation.trump_suit,
        wants_trick=wants_trick,
    )


def _select_card_for_intent(
    legal_cards: list[Card],
    *,
    player: int,
    current_trick: Trick | None,
    trump_suit: Suit | None,
    wants_trick: bool,
) -> Card:
    """Take with the highest winner, or shed the highest card that still loses."""

    if not legal_cards:
        raise ValueError("Heuristic play requires at least one legal card.")
    ordered = sorted(legal_cards, key=lambda card: (int(card.rank), card.suit.value))
    winners = [
        card
        for card in ordered
        if _card_is_current_winner(
            card,
            player=player,
            current_trick=current_trick,
            trump_suit=trump_suit,
        )
    ]
    if wants_trick:
        return winners[-1] if winners else ordered[0]

    winner_set = set(winners)
    losing_cards = [card for card in ordered if card not in winner_set]
    return losing_cards[-1] if losing_cards else ordered[0]


def _card_is_current_winner(
    card: Card,
    *,
    player: int,
    current_trick: Trick | None,
    trump_suit: Suit | None,
) -> bool:
    if current_trick is None:
        return True
    led_suit = current_trick.led_suit
    if not current_trick.plays:
        led_suit = card.suit
    trial = Trick(
        trick_index=current_trick.trick_index,
        leader=current_trick.leader,
        led_suit=led_suit,
        plays=list(current_trick.plays)
        + [TrickPlay(player=player, card=card, position=len(current_trick.plays))],
    )
    return determine_trick_winner(trial, trump_suit) == player


def _current_trick_winner(
    current_trick: Trick | None,
    trump_suit: Suit | None,
) -> int | None:
    if current_trick is None or not current_trick.plays:
        return None
    return determine_trick_winner(current_trick, trump_suit)


def _lost_player_should_take(
    *,
    player: int,
    bids: dict[int, int],
    tricks_won: dict[int, int],
    remaining_tricks: int,
    current_winner: int | None,
    num_players: int,
    total_round_tricks: int,
) -> bool:
    """Choose the current outcome that minimizes opponents' expected hit points."""

    take_score = _expected_opponent_hit_points(
        player=player,
        bids=bids,
        tricks_won=tricks_won,
        future_tricks=max(remaining_tricks - 1, 0),
        current_recipient=player,
        num_players=num_players,
    )
    if current_winner is not None and current_winner != player:
        offload_score = _expected_opponent_hit_points(
            player=player,
            bids=bids,
            tricks_won=tricks_won,
            future_tricks=max(remaining_tricks - 1, 0),
            current_recipient=current_winner,
            num_players=num_players,
        )
    else:
        opponent_scores = [
            _expected_opponent_hit_points(
                player=player,
                bids=bids,
                tricks_won=tricks_won,
                future_tricks=max(remaining_tricks - 1, 0),
                current_recipient=opponent,
                num_players=num_players,
            )
            for opponent in bids
            if opponent != player
        ]
        offload_score = sum(opponent_scores) / len(opponent_scores)

    if abs(take_score - offload_score) > 1e-12:
        return take_score < offload_score
    return sum(bids.values()) >= total_round_tricks


def _expected_opponent_hit_points(
    *,
    player: int,
    bids: dict[int, int],
    tricks_won: dict[int, int],
    future_tricks: int,
    current_recipient: int,
    num_players: int,
) -> float:
    probability_per_trick = 1.0 / num_players
    expected_points = 0.0
    for opponent, bid in bids.items():
        if opponent == player:
            continue
        current_delta = int(current_recipient == opponent)
        needed = bid - tricks_won.get(opponent, 0) - current_delta
        probability = _binomial_probability(
            trials=future_tricks,
            successes=needed,
            probability=probability_per_trick,
        )
        hit_points = 5 if bid == 0 else 10 + bid
        expected_points += probability * hit_points
    return expected_points


def _binomial_probability(
    *,
    trials: int,
    successes: int,
    probability: float,
) -> float:
    if successes < 0 or successes > trials:
        return 0.0
    return (
        comb(trials, successes)
        * probability**successes
        * (1.0 - probability) ** (trials - successes)
    )


class ModelPolicy:
    """Observation-only schema-v4 neural policy."""

    def __init__(
        self,
        model: PlumpTransformerModel,
        *,
        device: str | torch.device | None = None,
        greedy: bool = True,
        include_game_context: bool = False,
        precision: str = "fp32",
        event_length_buckets: Sequence[int] = (),
        batch_packing: Literal["torch", "numpy"] = "torch",
        lean_action_forward: bool = False,
        name: str = "model",
    ) -> None:
        self.device = torch.device(device) if device is not None else best_torch_device()
        self.model = model.to(self.device)
        self.model.eval()
        self.model_config = model.config
        self.greedy = greedy
        self.include_game_context = include_game_context
        self.precision = precision
        self.event_length_buckets = tuple(event_length_buckets)
        self.batch_packing = batch_packing
        self.lean_action_forward = lean_action_forward
        self.name = name
        self.forward_passes = 0

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device | None = None,
        greedy: bool = True,
        event_length_buckets: Sequence[int] = (),
        batch_packing: Literal["torch", "numpy"] = "torch",
        lean_action_forward: bool = False,
        name: str | None = None,
    ) -> "ModelPolicy | LegacyCheckpointPolicy":
        payload = _load_payload(checkpoint_path)
        checkpoint_schema = int(payload.get("schema_version", 1))
        if checkpoint_schema == 1:
            return LegacyCheckpointPolicy(checkpoint_path, device=device, greedy=greedy, name=name)
        config_data = payload.get("model_config")
        if not isinstance(config_data, dict):
            raise ValueError("Checkpoint is missing model_config.")
        config_data = dict(config_data)
        config_data["schema_version"] = SCHEMA_VERSION
        config = ModelConfig(**config_data)
        model = (
            PlumpSearchModel(config)
            if checkpoint_schema == 5
            else PlumpTransformerModel(config)
        )
        if checkpoint_schema == 2:
            load_v2_weights(model, payload["model_state_dict"])
        elif checkpoint_schema == 3:
            load_v3_weights(model, payload["model_state_dict"])
        elif checkpoint_schema in {SCHEMA_VERSION, 5}:
            model.load_state_dict(payload["model_state_dict"])
        else:
            raise ValueError(f"Unsupported checkpoint schema {checkpoint_schema}.")
        return cls(
            model,
            device=device,
            greedy=greedy,
            include_game_context=(
                bool(payload.get("include_game_context", False))
                if checkpoint_schema in {3, SCHEMA_VERSION, 5}
                else False
            ),
            precision=str(payload.get("precision", "fp32")),
            event_length_buckets=event_length_buckets,
            batch_packing=batch_packing,
            lean_action_forward=lean_action_forward,
            name=name or Path(checkpoint_path).stem,
        )

    def act(self, env: PlumpEnv, *, rng: random.Random | None = None) -> BidAction | PlayCardAction:
        return self.act_many([env], rngs=[rng or random.Random()])[0]

    def act_many(
        self,
        envs: list[PlumpEnv],
        *,
        rngs: list[random.Random] | None = None,
    ) -> list[BidAction | PlayCardAction]:
        if not envs:
            return []
        players = [env.current_player() for env in envs]
        phases = [env.phase() for env in envs]
        encoded = [
            encode_observation(
                env.get_observation(player),
                self.model_config,
                include_game_context=self.include_game_context,
            )
            for env, player in zip(envs, players)
        ]
        return self.act_encoded(encoded, phases=phases, players=players, rngs=rngs)

    def act_encoded(
        self,
        encoded: list,
        *,
        phases: list[Phase],
        players: list[int],
        rngs: list[random.Random] | None = None,
    ) -> list[BidAction | PlayCardAction]:
        """Act on pre-encoded observations (e.g. produced in env workers)."""

        if not encoded:
            return []
        if rngs is None:
            rngs = [random.Random() for _ in encoded]
        if len(rngs) != len(encoded):
            raise ValueError("rngs must match encoded observations.")
        output = (
            self.predict_policy_encoded(encoded)
            if self.lean_action_forward
            else self.predict_encoded(encoded, need_owner=False)
        )
        return self._actions_from_output(output, phases=phases, players=players, rngs=rngs)

    def act_batch(
        self,
        batch,
        *,
        phases: list[Phase],
        players: list[int],
        rngs: list[random.Random] | None = None,
    ) -> list[BidAction | PlayCardAction]:
        """Act on a prebuilt ModelBatch (e.g. one model's group of a pooled wave).

        Unlike ``act_encoded`` this samples on the CPU with vectorized numpy
        (one device sync per head instead of a mask upload, a GPU softmax, and
        a Python loop per row), so pooled evaluation waves with many small
        model groups do not serialize on the GPU queue. Sampling consumes one
        ``rng.random()`` draw per row, a different stream use than
        ``act_encoded``'s ``rng.choices`` — statistically equivalent, not
        bitwise identical.
        """

        if rngs is None:
            rngs = [random.Random() for _ in players]
        if len(rngs) != len(players):
            raise ValueError("rngs must match batch rows.")
        with torch.inference_mode(), model_autocast(self.device, self.precision):
            output = (
                self.model.forward_policy(batch)
                if self.lean_action_forward
                else self.model(batch, need_owner=False)
            )
        self.forward_passes += len(players)

        bid_logits = output.masked_bid_logits.float().cpu().numpy()
        card_logits = output.masked_card_logits.float().cpu().numpy()
        bidding = np.fromiter(
            (phase == Phase.BIDDING for phase in phases),
            dtype=np.bool_,
            count=len(phases),
        )

        def sample_indices(logits: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
            if self.greedy:
                return logits.argmax(axis=1)
            stable = logits - logits.max(axis=1, keepdims=True)
            probabilities = np.exp(stable)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            cumulative = np.cumsum(probabilities, axis=1)
            selected = (cumulative < uniforms[:, None]).sum(axis=1)
            # Rounding can leave the final cumulative a hair under the draw;
            # the overflow index would be out of range or a masked action, so
            # clip to each row's last positive-probability index.
            last_legal = (probabilities > 0).cumsum(axis=1).argmax(axis=1)
            return np.minimum(selected, last_legal)

        uniforms = np.fromiter(
            (rng.random() for rng in rngs), dtype=np.float64, count=len(rngs)
        )
        indices = np.empty(len(players), dtype=np.int64)
        if bidding.any():
            indices[bidding] = sample_indices(bid_logits[bidding], uniforms[bidding])
        if not bidding.all():
            play = ~bidding
            indices[play] = sample_indices(card_logits[play], uniforms[play])

        actions: list[BidAction | PlayCardAction] = []
        for player, is_bid, action_index in zip(players, bidding, indices):
            if is_bid:
                actions.append(BidAction(player, int(action_index)))
            else:
                actions.append(PlayCardAction(player, card_from_id(int(action_index))))
        return actions

    def _actions_from_output(
        self,
        output,
        *,
        phases: list[Phase],
        players: list[int],
        rngs: list[random.Random],
    ) -> list[BidAction | PlayCardAction]:
        bid_mask = torch.tensor(
            [phase == Phase.BIDDING for phase in phases],
            dtype=torch.bool,
            device=self.device,
        )
        logits = combined_action_logits(output, bid_mask)
        if self.greedy:
            selected_indices = logits.argmax(dim=-1).cpu().tolist()
        else:
            # One device transfer and vectorized CPU sampling is materially
            # cheaper than materializing nested Python probability lists and
            # calling random.choices once per historical-policy row.
            logit_rows = logits.float().cpu().numpy()
            stable = logit_rows - logit_rows.max(axis=1, keepdims=True)
            probabilities = np.exp(stable)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            cumulative = np.cumsum(probabilities, axis=1)
            uniforms = np.fromiter(
                (rng.random() for rng in rngs),
                dtype=np.float64,
                count=len(rngs),
            )
            selected = (cumulative < uniforms[:, None]).sum(axis=1)
            last_legal = (
                (probabilities > 0).cumsum(axis=1).argmax(axis=1)
            )
            selected_indices = np.minimum(
                selected,
                last_legal,
            ).tolist()
        actions: list[BidAction | PlayCardAction] = []
        for player, phase, action_index in zip(players, phases, selected_indices):
            if phase == Phase.BIDDING:
                actions.append(BidAction(player, action_index))
            elif phase == Phase.PLAYING:
                actions.append(PlayCardAction(player, card_from_id(action_index)))
            else:
                raise RuntimeError(f"Cannot act in phase {phase.value}.")
        return actions

    def predict(self, env: PlumpEnv):
        player = env.current_player()
        encoded, output = self.predict_observation(env.get_observation(player))
        return player, encoded, output

    def predict_observation(self, observation, *, need_owner: bool = True):
        encoded, output = self.predict_observations(
            [observation],
            need_owner=need_owner,
        )
        return encoded[0], output

    def predict_observations(self, observations, *, need_owner: bool = True):
        encoded = [
            encode_observation(
                observation,
                self.model_config,
                include_game_context=self.include_game_context,
            )
            for observation in observations
        ]
        return encoded, self.predict_encoded(encoded, need_owner=need_owner)

    def predict_encoded(self, encoded, *, need_owner: bool = True):
        inference_rows = encoded
        if self.device.type == "mps" and self.precision != "fp32":
            padded_size = _inference_bucket(len(encoded))
            if padded_size > len(encoded):
                inference_rows = [
                    *encoded,
                    *([encoded[-1]] * (padded_size - len(encoded))),
                ]
        batch = encoded_observations_to_batch(
            inference_rows,
            device=self.device,
            event_length_buckets=self.event_length_buckets,
            packing=self.batch_packing,
        )
        with torch.inference_mode(), model_autocast(self.device, self.precision):
            output = self.model(batch, need_owner=need_owner)
        if len(inference_rows) != len(encoded):
            output = slice_model_output(output, len(encoded))
        self.forward_passes += len(encoded)
        return output

    def predict_policy_encoded(self, encoded):
        """Run the action-only model path for inference policies."""

        inference_rows = encoded
        if self.device.type == "mps" and self.precision != "fp32":
            padded_size = _inference_bucket(len(encoded))
            if padded_size > len(encoded):
                inference_rows = [
                    *encoded,
                    *([encoded[-1]] * (padded_size - len(encoded))),
                ]
        batch = encoded_observations_to_batch(
            inference_rows,
            device=self.device,
            event_length_buckets=self.event_length_buckets,
            packing=self.batch_packing,
        )
        with torch.inference_mode(), model_autocast(self.device, self.precision):
            output = self.model.forward_policy(batch)
        if len(inference_rows) != len(encoded):
            output = slice_model_output(output, len(encoded))
        self.forward_passes += len(encoded)
        return output

    def reset_counters(self) -> None:
        self.forward_passes = 0


class LegacyCheckpointPolicy:
    """Read-only schema-v1 checkpoint adapter."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device | None = None,
        greedy: bool = True,
        name: str | None = None,
    ) -> None:
        from plump.modeling.legacy_encoding_v1 import ModelConfig as LegacyModelConfig
        from plump.modeling.legacy_torch_model_v1 import PlumpTransformerModel as LegacyModel

        self.device = torch.device(device) if device is not None else best_torch_device()
        payload = _load_payload(checkpoint_path)
        state = payload["model_state_dict"]
        stored = getattr(payload.get("training_config"), "model_config", None)
        layer_indices = {
            int(key.split(".")[2])
            for key in state
            if key.startswith("transformer.layers.") and key.split(".")[2].isdigit()
        }
        config = LegacyModelConfig(
            max_players=state["player_query_emb.weight"].shape[0],
            max_hand_size=state["bid_head.weight"].shape[0] - 1,
            max_rounds=state["round_emb.weight"].shape[0] - 1,
            max_seq_len=state["abs_pos_emb.weight"].shape[0],
            d_model=state["type_emb.weight"].shape[1],
            n_layers=max(layer_indices) + 1 if layer_indices else 1,
            n_heads=getattr(stored, "n_heads", 8),
            d_ff=state["transformer.layers.0.linear1.weight"].shape[0],
            context_hidden_dim=state["context_mlp.0.weight"].shape[0],
            dropout=getattr(stored, "dropout", 0.0),
        )
        self.model_config = config
        self.model = LegacyModel(config).to(self.device)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.greedy = greedy
        self.name = name or f"legacy:{Path(checkpoint_path).stem}"
        self.forward_passes = 0

    def act(self, env: PlumpEnv, *, rng: random.Random | None = None) -> BidAction | PlayCardAction:
        player = env.current_player()
        _, output = self.predict_observation(env.get_observation(player))
        logits = output.masked_bid_logits[0] if env.phase() == Phase.BIDDING else output.masked_card_logits[0]
        if self.greedy:
            action_index = int(logits.argmax(dim=-1).item())
        else:
            distribution = Categorical(logits=logits)
            action_index = int(distribution.sample().item())
        if env.phase() == Phase.BIDDING:
            return BidAction(player, action_index)
        return PlayCardAction(player, card_from_id(action_index))

    def predict_observation(self, observation):
        from plump.modeling.legacy_encoding_v1 import encode_observation
        from plump.modeling.legacy_torch_model_v1 import encoded_observations_to_batch

        encoded = encode_observation(observation, self.model_config)
        batch = encoded_observations_to_batch([encoded], device=self.device)
        with torch.no_grad():
            output = self.model(batch)
        self.forward_passes += 1
        return encoded, output

    def reset_counters(self) -> None:
        self.forward_passes = 0


def _load_payload(path: str | Path) -> dict:
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older PyTorch
        return torch.load(Path(path), map_location="cpu")


def _inference_bucket(size: int) -> int:
    for bucket in (4, 8, 16, 32, 64, 128, 256, 576):
        if size <= bucket:
            return bucket
    return size
