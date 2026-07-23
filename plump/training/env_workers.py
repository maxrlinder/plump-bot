"""Multiprocess environment workers for rollout collection.

Workers own the Python-bound half of collection: engine stepping, observation
encoding, privileged targets, and heuristic opponents. Model inference stays
in the trainer process. Each wave, every worker sends one ``DecisionRequest``
per active episode blocked on a model action; the trainer batches forwards by
policy and replies with actions.

Only stdlib imports at module level — the heavy plump/torch imports happen
inside the spawned worker so the parent's GPU state is never touched here.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from multiprocessing import get_context


@dataclass
class EpisodeAssignment:
    episode_id: int
    num_players: int
    hand_size: int
    opponent_arm: str
    focal_player: int
    start_player: int
    env_seed: int
    # player -> "current" | "heuristic" | league snapshot id
    seat_policy_refs: dict[int, str]
    opponent_snapshot_id: str | None
    explore_tempered: bool = False
    # Focal decision index (0 = bid) sampled uniformly over legal actions
    # this round; None = no injected action.
    explore_uniform_index: int | None = None


@dataclass
class DecisionRequest:
    episode_id: int
    player: int
    phase: str  # "bid" | "play"
    policy_ref: str  # "current" | league snapshot id
    encoded: object  # EncodedObservation
    owner_targets: list[int]
    suit_presence_targets: list[list[int]]
    observation: object | None  # trimmed snapshot; absent for discarded rows
    round_index: int
    num_players: int
    hand_size: int
    opponent_arm: str
    explore_tempered: bool
    trick_position: int
    round_progress: float
    # False for frozen current-weight seats (explore_self opponents): the row
    # samples raw (no temperature/eps) and never becomes a training sample.
    collect: bool = True
    # True for the round's single uniform-injected focal decision.
    explore_uniform: bool = False


@dataclass
class RoundResult:
    """The slice of terminal round state that training targets need."""

    round_scores: dict[int, int]
    bids: list
    tricks_won: dict[int, int]
    hand_size: int

    @classmethod
    def from_round_state(cls, round_state) -> "RoundResult":
        return cls(
            round_scores=dict(round_state.round_scores),
            bids=list(round_state.bids),
            tricks_won=dict(round_state.tricks_won),
            hand_size=round_state.hand_size,
        )


@dataclass
class EpisodeResult:
    episode_id: int
    result: RoundResult


@dataclass
class WorkerWave:
    requests: list[DecisionRequest] = field(default_factory=list)
    results: list[EpisodeResult] = field(default_factory=list)
    active: int = 0
    queued: int = 0


def _worker_main(connection, model_config, include_game_context, max_active) -> None:
    # Imported after spawn: workers are CPU-only engine/encoding processes.
    from plump.env import PlumpEnv
    from plump.policies import HeuristicPolicy
    from plump.rounds import RoundSpec, round_game_config
    from plump.training.ppo import build_decision_request

    heuristic = HeuristicPolicy()

    class _Episode:
        __slots__ = ("assignment", "env", "rng", "trainable_decisions")

        def __init__(self, assignment):
            self.assignment = assignment
            self.env = PlumpEnv(
                round_game_config(
                    RoundSpec(assignment.num_players, assignment.hand_size),
                    bidding_start_player=assignment.start_player,
                ),
                seed=assignment.env_seed,
            )
            self.env.reset()
            self.rng = random.Random(assignment.env_seed ^ 0x9E3779B9)
            # Decision index within the round for "current" (collecting)
            # seats; drives the single uniform-injected explore action.
            self.trainable_decisions = 0

    def advance(episode) -> DecisionRequest | None:
        """Play heuristic turns until a model action is needed or round ends."""

        while not episode.env.is_done():
            player = episode.env.current_player()
            ref = episode.assignment.seat_policy_refs.get(player, "current")
            if ref == "heuristic":
                episode.env.step(heuristic.act(episode.env, rng=episode.rng))
                continue
            collect = ref == "current"
            explore_uniform = (
                collect
                and episode.assignment.explore_uniform_index is not None
                and episode.trainable_decisions
                == episode.assignment.explore_uniform_index
            )
            if collect:
                episode.trainable_decisions += 1
            return build_decision_request(
                episode.env,
                episode_id=episode.assignment.episode_id,
                opponent_arm=episode.assignment.opponent_arm,
                policy_ref=ref,
                model_config=model_config,
                include_game_context=include_game_context,
                explore_tempered=(
                    episode.assignment.explore_tempered and collect
                ),
                collect=collect,
                explore_uniform=explore_uniform,
            )
        return None

    while True:
        message = connection.recv()
        if message[0] == "shutdown":
            connection.close()
            return
        assert message[0] == "collect"
        queue: list[EpisodeAssignment] = list(message[1])
        active: dict[int, object] = {}
        # Episodes whose last request has not been answered yet.
        pending: set[int] = set()
        needs_advance: list[int] = []

        while True:
            while queue and len(active) < max_active:
                assignment = queue.pop(0)
                active[assignment.episode_id] = _Episode(assignment)
                needs_advance.append(assignment.episode_id)

            wave = WorkerWave()
            while needs_advance:
                episode_id = needs_advance.pop()
                episode = active[episode_id]
                request = advance(episode)
                if request is None:
                    del active[episode_id]
                    wave.results.append(
                        EpisodeResult(
                            episode_id=episode_id,
                            result=RoundResult.from_round_state(
                                episode.env.state.current_round
                            ),
                        )
                    )
                    while queue and len(active) < max_active:
                        assignment = queue.pop(0)
                        active[assignment.episode_id] = _Episode(assignment)
                        needs_advance.append(assignment.episode_id)
                else:
                    pending.add(episode_id)
                    wave.requests.append(request)
            wave.active = len(active)
            wave.queued = len(queue)
            connection.send(wave)
            if not active and not queue:
                break

            message = connection.recv()
            assert message[0] == "actions"
            for episode_id, action in message[1].items():
                active[episode_id].env.step(action)
                pending.discard(episode_id)
                needs_advance.append(episode_id)


class EnvWorkerPool:
    """Persistent spawn-based worker pool driven wave-by-wave by the trainer."""

    def __init__(
        self,
        *,
        num_workers: int,
        model_config,
        include_game_context: bool,
        num_envs: int,
    ) -> None:
        if num_workers < 1:
            raise ValueError("num_workers must be positive.")
        context = get_context("spawn")
        max_active = max(1, math.ceil(num_envs / num_workers))
        self._connections = []
        self._processes = []
        for _ in range(num_workers):
            parent_connection, child_connection = context.Pipe()
            process = context.Process(
                target=_worker_main,
                args=(
                    child_connection,
                    model_config,
                    include_game_context,
                    max_active,
                ),
                daemon=True,
            )
            process.start()
            child_connection.close()
            self._connections.append(parent_connection)
            self._processes.append(process)
        self._episode_owner: dict[int, int] = {}
        self._finished: set[int] = set()

    @property
    def num_workers(self) -> int:
        return len(self._processes)

    def begin_iteration(self, assignments: list[EpisodeAssignment]) -> None:
        self._episode_owner.clear()
        self._finished.clear()
        shards: list[list[EpisodeAssignment]] = [[] for _ in self._connections]
        for index, assignment in enumerate(assignments):
            worker_id = index % len(self._connections)
            shards[worker_id].append(assignment)
            self._episode_owner[assignment.episode_id] = worker_id
        for connection, shard in zip(self._connections, shards):
            connection.send(("collect", shard))

    def gather_wave(self) -> tuple[list[DecisionRequest], list[EpisodeResult], bool]:
        requests: list[DecisionRequest] = []
        results: list[EpisodeResult] = []
        for worker_id, connection in enumerate(self._connections):
            if worker_id in self._finished:
                continue
            wave: WorkerWave = connection.recv()
            requests.extend(wave.requests)
            results.extend(wave.results)
            if wave.active == 0 and wave.queued == 0:
                self._finished.add(worker_id)
        done = len(self._finished) == len(self._connections)
        return requests, results, done

    def send_actions(self, actions: dict[int, object]) -> None:
        by_worker: dict[int, dict[int, object]] = defaultdict(dict)
        for episode_id, action in actions.items():
            by_worker[self._episode_owner[episode_id]][episode_id] = action
        for worker_id, connection in enumerate(self._connections):
            if worker_id in self._finished:
                continue
            connection.send(("actions", by_worker.get(worker_id, {})))

    def close(self) -> None:
        for connection in self._connections:
            try:
                connection.send(("shutdown",))
                connection.close()
            except (BrokenPipeError, OSError):
                pass
        for process in self._processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()

    def __enter__(self) -> "EnvWorkerPool":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
