"""ActionPolicy adapter and league for the schema-v6 sequence model."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from plump.env import PlumpEnv
from plump.state import BidAction, Phase, PlayCardAction

from .config import (
    NEXT_BID,
    NEXT_NONE,
    NEXT_PLAY,
    NUM_CARDS,
    SEQ_SCHEMA_VERSION,
    SeqModelConfig,
)
from .model import SeqPlumpModel
from .tokens import TOKEN_WIDTH, build_seat_tokens, card_from_id, card_id


def best_seq_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def masked_probabilities(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Softmax over legal entries only; illegal entries get exactly zero."""

    constrained = np.where(mask, logits, -np.inf)
    stable = constrained - constrained.max(axis=-1, keepdims=True)
    weights = np.exp(stable)
    weights[~mask] = 0.0
    return weights / weights.sum(axis=-1, keepdims=True)


def sample_index(probabilities: np.ndarray, uniform: float) -> int:
    cumulative = np.cumsum(probabilities)
    selected = int((cumulative < uniform).sum())
    legal = np.flatnonzero(probabilities > 0)
    return int(min(selected, legal[-1]))


class SeqModelPolicy:
    """Observation-only policy over per-decision full causal forwards."""

    def __init__(
        self,
        model: SeqPlumpModel,
        *,
        device: str | torch.device | None = None,
        greedy: bool = True,
        name: str = "seq",
    ) -> None:
        self.device = torch.device(device) if device is not None else best_seq_device()
        self.model = model.to(self.device)
        self.model.eval()
        self.config = model.config
        self.greedy = greedy
        self.name = name
        self.forward_passes = 0

    def reset_counters(self) -> None:
        self.forward_passes = 0

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device | None = None,
        greedy: bool = True,
        name: str | None = None,
    ) -> "SeqModelPolicy":
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        schema = int(payload.get("schema_version", 0))
        if schema != SEQ_SCHEMA_VERSION:
            raise ValueError(
                f"Expected schema {SEQ_SCHEMA_VERSION} checkpoint, got {schema}."
            )
        config = SeqModelConfig(**payload["model_config"])
        model = SeqPlumpModel(config)
        model.load_state_dict(payload["model_state_dict"])
        return cls(
            model,
            device=device,
            greedy=greedy,
            name=name or Path(checkpoint_path).stem,
        )

    def _tokens_for_env(self, env: PlumpEnv, player: int) -> np.ndarray:
        observation = env.get_observation(player)
        initial_hand = list(observation.my_hand) + list(
            observation.played_cards_by_player[player]
        )
        pending_phase = (
            NEXT_BID if observation.phase == Phase.BIDDING else NEXT_PLAY
        )
        return build_seat_tokens(
            self.config,
            observation.event_log,
            player,
            env.config.num_players,
            observation.hand_size,
            initial_hand,
            observation.bidding_start_player,
            pending_actor=player,
            pending_phase=pending_phase,
        )

    def act(
        self, env: PlumpEnv, *, rng: random.Random | None = None
    ) -> BidAction | PlayCardAction:
        return self.act_many([env], rngs=[rng or random.Random()])[0]

    def act_many(
        self,
        envs: list[PlumpEnv],
        *,
        rngs: list[random.Random] | None = None,
    ) -> list[BidAction | PlayCardAction]:
        if not envs:
            return []
        if rngs is None:
            rngs = [random.Random() for _ in envs]
        players = [env.current_player() for env in envs]
        phases = [env.phase() for env in envs]
        token_rows = [
            self._tokens_for_env(env, player) for env, player in zip(envs, players)
        ]
        lengths = [rows.shape[0] for rows in token_rows]
        max_length = max(lengths)
        batch = np.zeros((len(envs), max_length, TOKEN_WIDTH), dtype=np.int64)
        for row, tokens in enumerate(token_rows):
            batch[row, : tokens.shape[0]] = tokens

        with torch.inference_mode():
            output = self.model.forward_full(
                torch.from_numpy(batch).to(self.device), aux_heads=False
            )
        self.forward_passes += len(envs)
        last = torch.tensor(lengths, device=self.device) - 1
        rows = torch.arange(len(envs), device=self.device)
        bid_logits = output.bid_logits[rows, last].float().cpu().numpy()
        card_logits = output.card_logits[rows, last].float().cpu().numpy()

        actions: list[BidAction | PlayCardAction] = []
        for index, (env, player, phase) in enumerate(zip(envs, players, phases)):
            if phase == Phase.BIDDING:
                mask = np.zeros(self.config.bid_count, dtype=bool)
                observation_bids = env.legal_actions()
                for action in observation_bids:
                    mask[action.bid] = True
                probabilities = masked_probabilities(bid_logits[index], mask)
                if self.greedy:
                    choice = int(probabilities.argmax())
                else:
                    choice = sample_index(probabilities, rngs[index].random())
                actions.append(BidAction(player, choice))
            else:
                mask = np.zeros(NUM_CARDS, dtype=bool)
                for action in env.legal_actions():
                    mask[card_id(action.card)] = True
                probabilities = masked_probabilities(card_logits[index], mask)
                if self.greedy:
                    choice = int(probabilities.argmax())
                else:
                    choice = sample_index(probabilities, rngs[index].random())
                actions.append(PlayCardAction(player, card_from_id(choice)))
        return actions


@dataclass
class SeqLeagueSnapshot:
    snapshot_id: str
    path: str
    iteration: int


class SeqLeague:
    """Uniform draw over saved schema-v6 snapshots with a minimum iteration."""

    def __init__(self, max_snapshots: int, min_iteration: int = 0):
        self.max_snapshots = max_snapshots
        self.min_iteration = min_iteration
        self.snapshots: list[SeqLeagueSnapshot] = []
        self._policies: dict[str, SeqModelPolicy] = {}

    def add(self, snapshot_id: str, path: str, iteration: int) -> None:
        if iteration < self.min_iteration:
            return
        self.snapshots.append(SeqLeagueSnapshot(snapshot_id, path, iteration))
        if len(self.snapshots) > self.max_snapshots:
            dropped = self.snapshots.pop(0)
            self._policies.pop(dropped.snapshot_id, None)

    def has_snapshots(self) -> bool:
        return bool(self.snapshots)

    def draw(
        self, rng: random.Random, device: str | torch.device | None = None
    ) -> tuple[str, SeqModelPolicy]:
        snapshot = rng.choice(self.snapshots)
        if snapshot.snapshot_id not in self._policies:
            self._policies[snapshot.snapshot_id] = SeqModelPolicy.from_checkpoint(
                snapshot.path,
                device=device,
                greedy=False,
                name=snapshot.snapshot_id,
            )
        return snapshot.snapshot_id, self._policies[snapshot.snapshot_id]
