"""Wave-synchronized branching rollout engine over KV caches (schema v6).

Scheduled deals of the same shape may share a wave loop. Historical games, when
enabled, always use independent deals. Every leaf advances one public event per
wave, so all cache slots share a single position counter. Branching a leaf
clones its env and copies every seat's KV prefix.

Terminal rewards are backed up through the counterfactual tree with disjoint
policy-mass strata. One conditional-policy representative per stratum carries
that stratum's mass as its backup and downstream reach weight, giving fixed
distinct candidate width and an unbiased estimate of ``V_pi_old``.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from plump.env import PlumpEnv
from plump.policies import HeuristicPolicy
from plump.rewards import compute_relative_rewards
from plump.rounds import RoundSpec, round_game_config
from plump.state import BidAction, EventType, GameEvent, Phase, PlayCardAction

from .config import (
    NEXT_BID,
    NEXT_PLAY,
    NUM_CARDS,
    SLOT_NEXT_ACTOR,
    SLOT_NEXT_PHASE,
    SLOT_REL_PLAYER,
    GameScheduleCell,
    SeqModelConfig,
    SeqTrainingConfig,
)
from .kv import KVCache
from .model import SeqPlumpModel
from .policy import SeqLeague, masked_probabilities
from .tokens import (
    TOKEN_WIDTH,
    card_from_id,
    card_id,
    emits_token,
    event_token,
    prefix_tokens,
    set_remaining_hand,
    turn_token,
    turn_token_for_phase,
)

CURRENT = "current"
OPPONENT = "opponent"
HEURISTIC = "heuristic"


def _rng_from_state(state) -> random.Random:
    """A Random restored to ``state``, without seeding it first.

    ``random.Random()`` draws 32 bytes from the OS and runs the Mersenne key
    schedule, all of which ``setstate`` then discards -- ~15us against ~1.5us
    here, and this runs once per branch child. ``__new__`` skips the seeding;
    ``setstate`` restores the full state including ``gauss_next``, so the
    resulting stream is identical.
    """

    rng = random.Random.__new__(random.Random)
    rng.setstate(state)
    return rng


def _upstream_depth(upstream) -> int:
    """How many branch decisions sit above this point on its path."""

    depth = 0
    while upstream is not None:
        depth += 1
        upstream = upstream[0].upstream
    return depth


@dataclass
class SeqBranchRecord:
    """Counterfactual expansion at one focal decision."""

    candidate_indices: tuple[int, ...]
    prior_probs: tuple[float, ...]  # renormalized over candidates
    raw_probs: dict[int, float]  # unrenormalized old-policy masses
    # P(a is in the candidate set) under whatever rule selected it, per
    # candidate. NeuRD's logit gradient is per-action, so an action expanded
    # with probability q contributes q * A(a) in expectation over updates --
    # 1/q restores the true A(a). Exhaustive expansion is all ones.
    inclusion_probs: tuple[float, ...]
    deterministic_count: int
    sampled_index: int
    candidate_mass: float
    upstream: Optional[tuple["SeqBranchRecord", int]]
    child_values: dict[int, float] = field(default_factory=dict)
    backed_value: Optional[float] = None

    def resolve(self, action_index: int, value: float) -> None:
        self.child_values[action_index] = value
        if len(self.child_values) < len(self.candidate_indices):
            return
        if math.isclose(self.candidate_mass, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            backed = sum(
                probability * self.child_values[index]
                for probability, index in zip(self.prior_probs, self.candidate_indices)
            )
        else:
            # Control-variate estimator: exact mass over the deterministic top
            # set plus a one-sample estimate of its complement (unbiased for
            # the full-policy value).
            deterministic = set(self.candidate_indices[: self.deterministic_count])
            backed = sum(
                self.raw_probs[index] * self.child_values[index]
                for index in deterministic
            )
            if self.sampled_index not in deterministic:
                backed += self.child_values[self.sampled_index]
        self.backed_value = backed
        if self.upstream is not None:
            self.upstream[0].resolve(self.upstream[1], backed)


@dataclass
class SeqDecisionRecord:
    """One focal decision owned by one leaf."""

    position: int
    phase: int  # NEXT_BID or NEXT_PLAY
    action_index: int
    old_probs: np.ndarray  # masked policy over 11 bids or 52 cards
    old_value: float
    # Empirical old-policy reach represented by this row inside its expanded
    # tree. Uniform-only explorer descendants have zero reach.
    reach_weight: float = 1.0
    # Branch decisions above this one on its path. Recorded here rather than
    # walked at update time because an unbranched decision has no record of
    # its own to walk up from.
    depth: int = 0
    branch: Optional[SeqBranchRecord] = None


@dataclass
class _Pending:
    """Policy readout captured when the last token reached the acting seat."""

    seat: int
    bid_logits: np.ndarray
    card_logits: np.ndarray
    value: float


@dataclass
class SeqLeaf:
    env: PlumpEnv
    tree: "SeqTree"
    rng: random.Random
    slots: dict[int, tuple[str, int]]  # seat -> (policy_id, cache slot)
    owned_from: int
    on_policy_spine: bool
    upstream: Optional[tuple[SeqBranchRecord, int]]
    # Product of empirical old-policy branch weights along this path.
    reach_weight: float = 1.0
    # Index of the root-bid candidate this leaf descends from. Kept as a
    # census of how the tree divides across the focal's bid options, and as the
    # structural invariant a bid-split pass is checked against.
    bid_group: int = 0
    decisions: list[SeqDecisionRecord] = field(default_factory=list)
    open_positions: list[int] = field(default_factory=list)
    segments: list[tuple[list[int], Optional[SeqBranchRecord], float]] = field(
        default_factory=list
    )
    pending: Optional[_Pending] = None
    terminal_value: Optional[float] = None
    covered_until: int = 0
    # Cache-free mode only: [num_players, max_len, TOKEN_WIDTH] token history
    # re-encoded from scratch at every decision.
    history: Optional[np.ndarray] = None
    # The leaf this one branched off. Its token prefix up to ``owned_from`` is
    # identical to this leaf's -- the env was cloned there -- so the update can
    # copy it instead of replaying the shared events again. No extra retention:
    # ``tree.leaves`` already holds every leaf.
    parent: Optional["SeqLeaf"] = None

    def value_target_at(self, position: int) -> float:
        for positions, resolver, _ in self.segments:
            if position in positions:
                if resolver is None:
                    return self.terminal_value
                return resolver.backed_value
        raise KeyError(f"Position {position} is not owned by this leaf.")

    def value_targets(self) -> dict[int, float]:
        targets: dict[int, float] = {}
        for positions, resolver, _ in self.segments:
            value = self.terminal_value if resolver is None else resolver.backed_value
            for position in positions:
                targets[position] = value
        return targets

    def position_reach_weights(self) -> dict[int, float]:
        return {
            position: reach
            for positions, _, reach in self.segments
            for position in positions
        }


@dataclass
class SeqTree:
    arm: str  # "self" | "heuristic" | "historical"
    focal: int
    num_players: int
    hand_size: int
    bidding_start_player: int
    initial_hands: dict[int, list]
    opponent_id: Optional[str] = None
    leaves: list[SeqLeaf] = field(default_factory=list)
    leaf_total: int = 0
    # How the tree divides across the focal's root-bid candidates.
    # ``bid_groups`` is the total candidate count, known to every bid-split
    # pass even though a pass only ever builds its own subtrees.
    bid_groups: int = 1
    leaves_by_bid_group: dict[int, int] = field(default_factory=dict)
    decision_total: int = 0
    branch_decisions: int = 0
    # Deepest stage that still branched: -1 is bidding, t is trick index t.
    deepest_branch_trick: int = -2
    branched_tricks: set[int] = field(default_factory=set)
    branch_layers: int = 0
    # stage -> how many focal decisions branched there / leaves it added.
    branch_decisions_by_stage: dict[int, int] = field(default_factory=dict)
    leaves_added_by_stage: dict[int, int] = field(default_factory=dict)
    # Shared across bid-split passes: the root bid expansion is created once
    # and each pass resolves a disjoint subset of its children.
    split_bid_record: Optional[SeqBranchRecord] = None
    split_bid_candidates: tuple[int, ...] = ()
    # Candidate selection consumes the leaf RNG before children start. Later
    # split passes restore this post-selection state so every root child gets
    # the same continuation tape it would have received in an unsplit tree.
    split_post_sample_rng_state: Optional[tuple] = None


@dataclass
class ShapeCost:
    """What one (players, cards) cell cost in one collect()."""

    deals: int = 0
    batches: int = 0
    leaves: int = 0
    sec: float = 0.0

    @property
    def deals_per_batch(self) -> float:
        return self.deals / max(self.batches, 1)


@dataclass
class SeqCollectionStats:
    games: int = 0
    trees: int = 0
    leaves: int = 0
    decisions: int = 0
    branch_decisions: int = 0
    forward_rows: int = 0
    # Cache pressure: rows actually live at the high-water mark of any wave
    # loop, versus rows preallocated. Splitting a tree lowers the former; only
    # the memory budget lowers the latter.
    peak_cache_rows: int = 0
    cache_rows_allocated: int = 0
    bytes_per_row: int = 0
    # Branch points refused because the KV rows ran out. This should be zero:
    # rows run out late in a game, so a nonzero count means the rate is too
    # high for the row cap and the endgame is being truncated.
    blocked_by_cache: int = 0
    # Decisions that could have branched but were not drawn as branch points.
    # Not a shortfall: this is the mechanism.
    skipped_by_placement: int = 0
    # High-water device memory during collection, sampled once per wave.
    peak_device_bytes: int = 0
    # (players, hand size) -> wall time and batch shape of that cell. This is
    # what says whether the schedule is spending its time where the positions
    # are, and whether a cell is running at a batch worth batching.
    by_shape: dict[tuple[int, int], ShapeCost] = field(default_factory=dict)
    collect_sec: float = 0.0
    sample_sec: float = 0.0
    step_sec: float = 0.0
    compact_sec: float = 0.0
    token_build_sec: float = 0.0
    forward_sec: float = 0.0


class SeqRolloutCollector:
    def __init__(
        self,
        model: SeqPlumpModel,
        train_config: SeqTrainingConfig,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        self.model_config: SeqModelConfig = model.config
        self.train = train_config
        self.device = torch.device(device) if device is not None else model.device
        self.model = model.to(self.device)
        self._caches: dict[str, KVCache] = {}
        self._kv_dtype = (
            torch.float16 if train_config.kv_dtype == "fp16" else torch.float32
        )
        self.use_cache = train_config.use_kv_cache
        self.heuristic = HeuristicPolicy()
        self._total_leaves = 0
        # Widest tree seen per (policy, game shape): what the pool is sized
        # from. Survives across collect() calls so training only pays the
        # cold-start growth on its first iteration.
        self._peak_rows: dict[tuple, int] = {}
        # (players, hand size) -> widest observed cache rows per deal. Persists
        # across collect() calls, so the batch size converges after the first
        # iteration instead of re-probing every time.
        self._rows_per_deal: dict[tuple[int, int], float] = {}
        # Player count -> next bidding position to assign. Persists across
        # collect() calls so the walk continues rather than restarting.
        self._seat_cursor: dict[int, int] = {}
        self._loop_row_cap = self._row_cap()
        self._split_plan: Optional[tuple[int, int]] = None
        self._split_done: set[int] = set()
        self.profile_sync = False
        self.stats = SeqCollectionStats()

    def _sync(self) -> None:
        if not self.profile_sync:
            return
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def release_caches(self) -> None:
        """Drop the KV pools between collection and the training update.

        The pools are the largest live allocation in the process and are dead
        weight once the trees are collected -- holding them through the
        backward pass roughly doubles peak memory. The per-shape high-water
        marks survive, so the next collection reserves the right size again in
        one allocation rather than growing into it.
        """

        self._caches.clear()
        self._release_allocator()

    def _release_allocator(self) -> None:
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Schedule                                                            #
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def collect(
        self,
        league: Optional[SeqLeague],
        rng: random.Random,
        *,
        iteration: int = 0,
        opponent_phase: str | None = None,
    ) -> list[SeqTree]:
        started = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.stats = SeqCollectionStats()
        trees: list[SeqTree] = []
        self._total_leaves = 0
        phase = opponent_phase or self.train.rollout.initial_opponent
        opponent_counts = self._opponent_games_by_cell(phase)
        for cell_index, cell in enumerate(self.train.schedule_cells):
            opponent_games = opponent_counts[cell_index]
            arms = [
                (
                    phase
                    if ((index + 1) * opponent_games) // cell.games
                    > (index * opponent_games) // cell.games
                    else "self"
                )
                for index in range(cell.games)
            ]
            remaining = len(arms)
            offset = 0
            # Resolve the shape once per cell rather than once per batch: with
            # deals_per_batch > 1 a per-batch draw gives the player mix only a
            # handful of samples per update, so the realized mix wobbles for no
            # reason. An explicit cell (build_game_schedule) fixes it outright.
            num_players = self._sample_players(cell, rng)
            cost = self.stats.by_shape.setdefault(
                (num_players, cell.hand_size), ShapeCost()
            )
            while remaining > 0:
                deals = min(remaining, self._deals_for(cell, num_players, remaining))
                batch_started = time.perf_counter()
                batch = self._collect_deal_batch(
                    cell,
                    league,
                    rng,
                    arms[offset : offset + deals],
                    num_players,
                    iteration,
                )
                cost.sec += time.perf_counter() - batch_started
                cost.deals += deals
                cost.batches += 1
                cost.leaves += sum(tree.leaf_total for tree in batch)
                trees.extend(batch)
                remaining -= deals
                offset += deals
        self.stats.collect_sec = time.perf_counter() - started
        self.stats.bytes_per_row = self._bytes_per_row()
        self.stats.cache_rows_allocated = sum(
            cache.capacity for cache in self._caches.values()
        )
        self.stats.trees = len(trees)
        self.stats.leaves = sum(tree.leaf_total for tree in trees)
        self.stats.decisions = sum(tree.decision_total for tree in trees)
        return trees

    def _opponent_games_by_cell(self, phase: str) -> list[int]:
        """Largest-remainder split of the fixed schedule into self/anchor games."""

        cells = self.train.schedule_cells
        if phase == "off" or self.train.rollout.opponent_fraction <= 0.0:
            return [0] * len(cells)
        fraction = self.train.rollout.opponent_fraction
        quotas = [cell.games * fraction for cell in cells]
        counts = [math.floor(quota) for quota in quotas]
        target = round(sum(cell.games for cell in cells) * fraction)
        remaining = target - sum(counts)
        order = sorted(
            range(len(cells)),
            key=lambda index: (-(quotas[index] - counts[index]), index),
        )
        for index in order[:remaining]:
            counts[index] += 1
        return counts

    def _sample_players(self, cell: GameScheduleCell, rng: random.Random) -> int:
        if cell.num_players is not None:
            return cell.num_players
        return rng.choices(
            self.train.player_counts, weights=self.train.player_count_weights
        )[0]

    def _deals_for(
        self, cell: GameScheduleCell, num_players: int, remaining: int
    ) -> int:
        """How many deals of this shape to advance in one wave loop."""

        options = self.train.rollout
        if not options.auto_deals_per_batch:
            if (
                options.parallel_deals_max_hand_size is not None
                and cell.hand_size > options.parallel_deals_max_hand_size
            ):
                return 1
            return options.deals_per_batch
        target = options.auto_target_rows
        if target is None:
            target = self._row_cap()
        target = int(target * (1.0 - options.auto_deals_headroom))

        observed = self._rows_per_deal.get((num_players, cell.hand_size))
        if observed is None:
            # Probing blind is what breaks the memory ceiling: two deals of an
            # unmeasured wide shape can outgrow the pool on their own, and with
            # rows as the only bound there is nothing to predict a tree's size
            # from. Probe one deal and let the measurement size the next batch.
            return 1
        deals = int(target // max(observed, 1.0))
        return max(min(deals, options.max_deals_per_batch), 1)

    def _focal_seat(
        self,
        cell: GameScheduleCell,
        num_players: int,
        rng: random.Random,
        bidding_start: int,
    ) -> int:
        if cell.focal_seat is not None:
            return cell.focal_seat
        if self.train.rollout.bid_position_mode == "cycle":
            # Round-robin the bidding position, not the seat. The cursor is
            # per player count and deliberately not reset per cell: cells hold
            # only a few deals each, so a per-cell cursor would never reach the
            # late positions of a 5-player game at all.
            position = self._seat_cursor.get(num_players, 0)
            self._seat_cursor[num_players] = position + 1
            return (bidding_start + position) % num_players
        return rng.randrange(num_players)

    # ------------------------------------------------------------------ #
    # Deal batches, arms, and bid-split passes                            #
    # ------------------------------------------------------------------ #

    def _new_deal(
        self, num_players: int, hand_size: int, rng: random.Random
    ) -> tuple[PlumpEnv, int, int]:
        start_player = rng.randrange(num_players)
        seed = rng.getrandbits(48)
        env = PlumpEnv(
            round_game_config(
                RoundSpec(num_players, hand_size),
                bidding_start_player=start_player,
            ),
            seed=seed,
        )
        # The wave loop reads env.state.event_log and env.is_done() directly and
        # never touches StepResult, so the observation every step() built was
        # pure waste -- and it is rebuilt once per step per leaf. clone() carries
        # the flag, so every leaf in every tree inherits it.
        env.emit_observations = False
        env.reset(seed=seed)
        return env, start_player, seed

    def _collect_deal_batch(
        self,
        cell: GameScheduleCell,
        league: Optional[SeqLeague],
        rng: random.Random,
        arms: list[str],
        num_players: int,
        iteration: int,
    ) -> list[SeqTree]:
        """Build the requested fixed-budget arms and run their wave loops.

        Every deal in a batch shares (players, hand size) so that all leaves
        advance in lockstep and simply widen each forward.
        """

        options = self.train.rollout
        hand_size = cell.hand_size

        models: dict[str, SeqPlumpModel] = {CURRENT: self.model}
        opponent_id: Optional[str] = None
        if (
            "historical" in arms
            and league is not None
            and league.has_snapshots(iteration)
        ):
            opponent_id, opponent_policy = league.draw(
                rng, iteration=iteration, device=self.device
            )
            models[OPPONENT] = opponent_policy.model
        elif "historical" in arms:
            # A brand-new run may request historical play before its first
            # checkpoint is eligible. Preserve the fixed game budget and fall
            # back to self-play until the league is populated.
            arms = ["self" if arm == "historical" else arm for arm in arms]

        # A unit is one wave loop: a list of (tree, env, crn_state).
        concurrent_unit: list[tuple[SeqTree, PlumpEnv, tuple]] = []
        sequential_units: list[list[tuple[SeqTree, PlumpEnv, tuple]]] = []
        trees: list[SeqTree] = []

        def make_tree(arm: str, env: PlumpEnv, start: int, focal: int) -> SeqTree:
            tree = SeqTree(
                arm=arm,
                focal=focal,
                num_players=num_players,
                hand_size=hand_size,
                bidding_start_player=start,
                initial_hands={
                    player: list(hand)
                    for player, hand in env.state.current_round.initial_hands.items()
                },
                opponent_id=opponent_id if arm == "historical" else None,
            )
            trees.append(tree)
            return tree

        for arm in arms:
            env, start, seed = self._new_deal(num_players, hand_size, rng)
            focal = self._focal_seat(cell, num_players, rng, start)
            crn = random.Random(seed ^ 0x5EED).getstate()
            entry = (make_tree(arm, env, start, focal), env, crn)
            if arm != "self" and options.opponent_packing == "sequential":
                sequential_units.append([entry])
            else:
                concurrent_unit.append(entry)

        splits = options.splits_for(hand_size)
        for unit in (concurrent_unit, *sequential_units):
            if not unit:
                continue
            for group in range(splits):
                self._run_wave_loop(unit, models, group, splits)
        for tree in trees:
            for leaf in tree.leaves:
                for _, resolver, _ in leaf.segments:
                    if resolver is not None and resolver.backed_value is None:
                        raise AssertionError("Unresolved branch record after game.")
        self.stats.games += len(arms)
        return trees

    def _run_wave_loop(
        self,
        unit: list[tuple[SeqTree, PlumpEnv, tuple]],
        models: dict[str, SeqPlumpModel],
        split_group: int,
        split_total: int,
    ) -> None:
        """Advance every leaf of ``unit`` in lockstep until all terminate.

        With ``split_total > 1`` this runs one pass over a subset of the
        focal's bid candidates; the shared prefix is replayed each pass and
        only pass 0 owns its training positions.
        """

        trees = [tree for tree, _, _ in unit]
        num_players = trees[0].num_players
        hand_size = trees[0].hand_size
        prefix_len = self.model_config.prefix_len(hand_size)
        self._split_plan = (split_group, split_total) if split_total > 1 else None
        self._split_done = set()

        policy_count = 2 if OPPONENT in models else 1
        self._loop_row_cap = self._policy_row_cap(policy_count)
        arm_signature = tuple(
            (arm, sum(tree.arm == arm for tree in trees))
            for arm in ("self", "heuristic", "historical")
            if any(tree.arm == arm for tree in trees)
        )
        shape = (num_players, hand_size, len(unit), split_total, arm_signature)
        if self.use_cache:
            for policy_id in (CURRENT, OPPONENT):
                if policy_id != CURRENT and policy_id not in models:
                    continue
                self._ensure_cache(
                    policy_id,
                    self._cache_capacity(policy_id, shape, policy_count),
                    self.model_config.max_seq_len,
                    self._loop_row_cap,
                )

        leaves: list[SeqLeaf] = []
        row_counters: dict[str, int] = {CURRENT: 0, OPPONENT: 0}
        for tree, deal_env, crn_state in unit:
            # matched arms share the random tape
            leaf_rng = _rng_from_state(crn_state)
            slots = {}
            for seat in range(num_players):
                policy_id = self._policy_for_seat(tree.arm, tree.focal, seat)
                if policy_id is None:
                    continue
                slots[seat] = (policy_id, row_counters[policy_id])
                row_counters[policy_id] += 1
            # Later split passes replay the shared prefix but must not claim
            # its training positions a second time.
            owned_from = 0 if split_group == 0 else self._split_position(tree)
            leaf = SeqLeaf(
                env=deal_env.clone_for_rollout(),
                tree=tree,
                rng=leaf_rng,
                slots=slots,
                owned_from=owned_from,
                on_policy_spine=True,
                upstream=None,
                covered_until=max(prefix_len, owned_from),
            )
            if owned_from == 0:
                leaf.open_positions.extend(range(prefix_len))
                tree.leaf_total = 1
                tree.leaves_by_bid_group[0] = 1
                self._total_leaves += 1
            leaves.append(leaf)

        if self.use_cache:
            for policy_id, rows in row_counters.items():
                if rows and policy_id in self._caches:
                    self._caches[policy_id].ensure_capacity(rows)
        self._prefill(leaves, models, prefix_len)
        position = prefix_len
        alive = list(leaves)

        while alive:
            appends: list[tuple[SeqLeaf, list[GameEvent]]] = []
            alive_next: list[SeqLeaf] = []
            phase = alive[0].env.phase()
            t0 = time.perf_counter()
            neural_alive = [
                leaf
                for leaf in alive
                if not (
                    leaf.tree.arm == HEURISTIC
                    and leaf.env.current_player() != leaf.tree.focal
                )
            ]
            if neural_alive:
                probs_batch, sampled_batch, legal_lists = self._batch_sample(
                    neural_alive, phase
                )
                sampled_by_leaf = {
                    id(leaf): (
                        probs_batch[index],
                        int(sampled_batch[index]),
                        legal_lists[index],
                    )
                    for index, leaf in enumerate(neural_alive)
                }
            else:
                sampled_by_leaf = {}
            heuristic_alive = [
                leaf
                for leaf in alive
                if leaf.tree.arm == HEURISTIC
                and leaf.env.current_player() != leaf.tree.focal
            ]
            heuristic_actions = dict(
                zip(
                    map(id, heuristic_alive),
                    self.heuristic.act_many(
                        [leaf.env for leaf in heuristic_alive],
                        rngs=[leaf.rng for leaf in heuristic_alive],
                    ),
                )
            )
            t1 = time.perf_counter()
            branch_copies: dict[str, tuple[list[int], list[int]]] = {}
            layer_branched: set[int] = set()
            for leaf in alive:
                if (
                    leaf.tree.arm == HEURISTIC
                    and leaf.env.current_player() != leaf.tree.focal
                ):
                    self._advance_leaf(
                        leaf,
                        heuristic_actions[id(leaf)],
                        appends,
                        alive_next,
                    )
                    continue
                probs, sampled, legal = sampled_by_leaf[id(leaf)]
                self._decide_and_step(
                    leaf,
                    phase,
                    probs,
                    sampled,
                    legal,
                    position,
                    appends,
                    alive_next,
                    row_counters,
                    branch_copies,
                    layer_branched,
                )
            t2 = time.perf_counter()
            # Games are fixed length, so leaves only ever get appended until
            # they all terminate together. Rows therefore stay dense with one
            # batched parent->child prefix copy per wave and never need a
            # permutation pass.
            for policy_id, (parents, children) in branch_copies.items():
                self._caches[policy_id].ensure_capacity(row_counters[policy_id])
                self._caches[policy_id].branch_copy(
                    torch.from_numpy(np.asarray(parents, dtype=np.int64)).to(
                        self.device
                    ),
                    torch.from_numpy(np.asarray(children, dtype=np.int64)).to(
                        self.device
                    ),
                    length=position,
                )
            if branch_copies:
                self._sync()
            t3 = time.perf_counter()
            self.stats.peak_cache_rows = max(
                self.stats.peak_cache_rows, sum(row_counters.values())
            )
            if self.device.type == "mps":
                self.stats.peak_device_bytes = max(
                    self.stats.peak_device_bytes,
                    torch.mps.driver_allocated_memory(),
                )
            elif self.device.type == "cuda":
                self.stats.peak_device_bytes = max(
                    self.stats.peak_device_bytes,
                    torch.cuda.max_memory_allocated(self.device),
                )
            self.stats.sample_sec += t1 - t0
            self.stats.step_sec += t2 - t1
            self.stats.compact_sec += t3 - t2
            if appends:
                position += self._append_wave(appends, models, position, row_counters)
            alive = alive_next

        # Freeing between passes is what makes splitting a memory win.
        total_rows = 0
        for policy_id, rows in row_counters.items():
            if rows:
                key = (policy_id, shape)
                self._peak_rows[key] = max(self._peak_rows.get(key, 0), rows)
                total_rows += rows
        # Rows per deal, for sizing later batches of this shape. Take the max
        # rather than an average: tree size varies severalfold between deals,
        # and undersizing the batch only costs throughput while oversizing it
        # costs the memory ceiling.
        deals = max(len(unit), 1)
        key = (num_players, hand_size)
        self._rows_per_deal[key] = max(
            self._rows_per_deal.get(key, 0.0), total_rows / deals
        )
        for cache in self._caches.values():
            cache.reset()
        self._split_plan = None

    def _split_position(self, tree: SeqTree) -> int:
        """Token position of the focal's own bid: where a split pass starts."""

        bid_index = (tree.focal - tree.bidding_start_player) % tree.num_players
        return self.model_config.bid_token_position(tree.hand_size, bid_index)

    def _policy_for_seat(
        self, arm: str, focal: int, seat: int
    ) -> str | None:
        if seat == focal or arm == "self":
            return CURRENT
        if arm == "historical":
            return OPPONENT
        if arm == HEURISTIC:
            return None
        raise ValueError(f"Unknown rollout arm {arm!r}.")

    def _ensure_cache(
        self, policy_id: str, capacity: int, max_len: int, max_capacity: int
    ) -> None:
        """Size the pool between wave loops, where growing is free.

        Nothing is live at this point, so a short pool is replaced outright
        rather than copied. Capacity only ever grows and is never re-shrunk
        for a smaller hand: churning multi-GB buffers through the allocator
        every deal dominates the rollout at large budgets.
        """

        cache = self._caches.get(policy_id)
        if cache is None or cache.capacity < capacity or cache.max_len < max_len:
            self._caches.pop(policy_id, None)
            del cache
            # Hand the old pool back before reserving the new one: the MPS
            # allocator keeps freed blocks, so a few growth steps otherwise
            # leave several stale multi-GB pools resident.
            self._release_allocator()
            self._caches[policy_id] = KVCache(
                self.model_config,
                capacity,
                self.device,
                self._kv_dtype,
                max_len,
                max_capacity=max_capacity,
            )
        else:
            cache.max_capacity = max(max_capacity, cache.capacity)
            cache.reset()

    def _bytes_per_row(self) -> int:
        config = self.model_config
        element = 2 if self._kv_dtype == torch.float16 else 4
        return (
            config.n_layers
            * config.kv_heads
            * config.max_seq_len
            * config.head_dim
            * 2  # K and V
            * element
        )

    def _row_cap(self) -> int:
        """Total cached rows the memory budget allows for this model."""

        options = self.train.rollout
        if options.max_cache_rows is not None:
            return options.max_cache_rows
        return max(int(options.cache_budget_gb * 1e9 / self._bytes_per_row()), 64)

    def _policy_row_cap(self, arms: int) -> int:
        return self._row_cap() if arms < 2 else self._row_cap() // 2

    def _cache_capacity(self, policy_id: str, shape: tuple, arms: int) -> int:
        """Rows to reserve for the next wave loop.

        Nothing bounds a tree ahead of time -- branching runs at its rate until
        the rows are gone -- so the row cap is the only worst case there is,
        and reserving it up front would pin gigabytes that never get written.
        Instead reserve a little above the widest tree seen for this game shape
        and let the first iteration grow into it: growth between wave loops is
        a free reallocation, so from the second iteration on the pool is sized
        to the real tree.
        """

        options = self.train.rollout
        if options.cache_preallocate:
            return self._policy_row_cap(arms)
        hand_size = shape[1]
        worst_case = self._policy_row_cap(arms)
        observed = self._peak_rows.get((policy_id, shape))
        if observed is None:
            # Cold start: scale the seed guess with the shape, since the tree
            # grows super-linearly in hand size and a too-small pool has to be
            # grown mid-loop, which copies everything already cached.
            target = options.cache_initial_rows * max(hand_size - 5, 1)
        else:
            target = observed + observed // 4 + 64
        return max(min(int(target), int(worst_case), self._policy_row_cap(arms)), 64)

    def _would_exceed_cache(
        self, leaf: SeqLeaf, new_children: int, row_counters: dict[str, int], cap: int
    ) -> bool:
        """Whether expanding would push past the memory budget.

        The bound is the budget-derived row cap, not the pool's current size:
        the pool grows on demand, so clamping to it would let an early small
        allocation permanently cap the tree.
        """

        if not self.use_cache or new_children <= 0:
            return False
        needed: dict[str, int] = {}
        for policy_id, _ in leaf.slots.values():
            needed[policy_id] = needed.get(policy_id, 0) + new_children
        return any(
            row_counters[policy_id] + rows > cap for policy_id, rows in needed.items()
        )

    # ------------------------------------------------------------------ #
    # Decisions, branching, env stepping                                  #
    # ------------------------------------------------------------------ #

    def _batch_sample(
        self, alive: list[SeqLeaf], phase: Phase
    ) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
        """Masked softmax + inverse-CDF sampling for a whole wave at once."""

        width = self.model_config.bid_count if phase == Phase.BIDDING else NUM_CARDS
        logits = np.empty((len(alive), width), dtype=np.float32)
        masks = np.zeros((len(alive), width), dtype=bool)
        legal_lists: list[list[int]] = []
        for index, leaf in enumerate(alive):
            pending = leaf.pending
            actor = leaf.env.current_player()
            if pending is None or pending.seat != actor:
                raise AssertionError("Pending readout does not match the actor.")
            if leaf.env.phase() != phase:
                raise AssertionError("Wave leaves diverged in phase.")
            logits[index] = (
                pending.bid_logits if phase == Phase.BIDDING else pending.card_logits
            )
            if phase == Phase.BIDDING:
                legal = leaf.env.legal_bid_values()
            else:
                legal = [card_id(card) for card in leaf.env.legal_card_values()]
            masks[index, legal] = True
            legal_lists.append(legal)
        probs = masked_probabilities(logits, masks)
        uniforms = np.fromiter(
            (leaf.rng.random() for leaf in alive),
            dtype=np.float64,
            count=len(alive),
        )
        cumulative = np.cumsum(probs, axis=1)
        sampled = (cumulative < uniforms[:, None]).sum(axis=1)
        last_legal = width - 1 - np.argmax(masks[:, ::-1], axis=1)
        sampled = np.minimum(sampled, last_legal)
        bad = ~masks[np.arange(len(alive)), sampled]
        sampled[bad] = last_legal[bad]
        return probs, sampled, legal_lists

    def _decide_and_step(
        self,
        leaf: SeqLeaf,
        phase: Phase,
        probs: np.ndarray,
        sampled: int,
        legal: list[int],
        position: int,
        appends: list[tuple[SeqLeaf, list[GameEvent]]],
        alive_next: list[SeqLeaf],
        row_counters: dict[str, int],
        branch_copies: dict[str, tuple[list[int], list[int]]],
        layer_branched: set[int],
    ) -> None:
        env = leaf.env
        actor = env.current_player()

        if actor != leaf.tree.focal:
            self._advance_leaf(
                leaf, self._action_for(actor, phase, sampled), appends, alive_next
            )
            return

        if (
            self._split_plan is not None
            and phase == Phase.BIDDING
            and id(leaf.tree) not in self._split_done
        ):
            self._expand_bid_split(
                leaf,
                probs,
                sampled,
                legal,
                position,
                appends,
                alive_next,
                row_counters,
                branch_copies,
            )
            return

        readout_position = position - 1
        record = SeqDecisionRecord(
            position=readout_position,
            phase=NEXT_BID if phase == Phase.BIDDING else NEXT_PLAY,
            action_index=sampled,
            old_probs=probs,
            old_value=leaf.pending.value,
            reach_weight=leaf.reach_weight,
            depth=_upstream_depth(leaf.upstream),
        )
        leaf.decisions.append(record)
        leaf.tree.decision_total += 1
        self.stats.decisions += 1

        candidates, backup_weights, deterministic_count, inclusion = (
            self._branch_candidates(leaf, phase, legal, probs, sampled)
        )
        allow_branch = True
        # The rate decides *whether* this decision is a branch point. The
        # focal's own bid is always expanded: it is the root of the tree and
        # sets the whole round's target, so it is never left to a coin flip.
        if (
            len(candidates) > 1
            and phase != Phase.BIDDING
            and not self._is_branch_point(
                leaf,
                len(candidates),
                len(
                    [t for t in env.state.current_round.tricks if t.winner is not None]
                ),
            )
        ):
            allow_branch = False
            self.stats.skipped_by_placement += 1
        # The cache is the one hard bound: rather than growing past the memory
        # budget, stop branching and let these leaves run to terminal.
        if allow_branch and self._would_exceed_cache(
            leaf, len(candidates) - 1, row_counters, self._loop_row_cap
        ):
            allow_branch = False
            if len(candidates) > 1:
                self.stats.blocked_by_cache += 1
        if not (len(candidates) > 1 and allow_branch):
            self._advance_leaf(
                leaf, self._action_for(actor, phase, sampled), appends, alive_next
            )
            return

        raw = {index: float(probs[index]) for index in candidates}
        if backup_weights is not None:
            # Stochastic rules supply their exact realized estimator weights:
            # empirical frequencies for iid draws or policy masses for strata.
            priors = tuple(backup_weights)
            mass = 1.0
        else:
            mass = float(sum(raw.values()))
            priors = tuple(raw[index] / mass for index in candidates)
        branch = SeqBranchRecord(
            candidate_indices=tuple(candidates),
            prior_probs=priors,
            raw_probs=raw,
            inclusion_probs=tuple(inclusion),
            deterministic_count=deterministic_count,
            sampled_index=sampled,
            candidate_mass=mass,
            upstream=leaf.upstream,
        )
        record.branch = branch
        parent_reach = leaf.reach_weight
        reach_by_candidate = dict(zip(candidates, priors))
        leaf.segments.append((leaf.open_positions, branch, parent_reach))
        leaf.open_positions = []
        leaf.upstream = (branch, sampled)
        self.stats.branch_decisions += 1
        tree = leaf.tree
        tree.branch_decisions += 1
        stage = (
            -1
            if phase == Phase.BIDDING
            else len(
                [t for t in env.state.current_round.tricks if t.winner is not None]
            )
        )
        tree.branched_tricks.add(stage)
        tree.deepest_branch_trick = max(tree.deepest_branch_trick, stage)
        tree.branch_decisions_by_stage[stage] = (
            tree.branch_decisions_by_stage.get(stage, 0) + 1
        )
        tree.leaves_added_by_stage[stage] = (
            tree.leaves_added_by_stage.get(stage, 0) + len(candidates) - 1
        )
        if id(tree) not in layer_branched:
            layer_branched.add(id(tree))
            tree.branch_layers += 1

        # The focal's own bid is always the first focal decision, so a bidding
        # branch here is the root expansion that defines the budget groups.
        root_bid = phase == Phase.BIDDING
        if root_bid:
            tree.bid_groups = len(candidates)
            tree.leaves_by_bid_group = {}

        crn_state = leaf.rng.getstate()
        for slot_index, candidate in enumerate(candidates):
            if root_bid:
                group = slot_index
                tree.leaves_by_bid_group[group] = 1
            else:
                # The parent leaf is already counted in its own group.
                group = leaf.bid_group
                if candidate != sampled:
                    census = tree.leaves_by_bid_group
                    census[group] = census.get(group, 0) + 1
            if candidate == sampled:
                leaf.bid_group = group
                continue
            child_rng = _rng_from_state(crn_state)
            child_slots: dict[int, tuple[str, int]] = {}
            if self.use_cache:
                for seat, (policy_id, parent_row) in leaf.slots.items():
                    child_row = row_counters[policy_id]
                    row_counters[policy_id] += 1
                    child_slots[seat] = (policy_id, child_row)
                    parents, children = branch_copies.setdefault(policy_id, ([], []))
                    parents.append(parent_row)
                    children.append(child_row)
            else:
                child_slots = dict(leaf.slots)
            child = SeqLeaf(
                env=leaf.env.clone_for_rollout(),
                tree=leaf.tree,
                rng=child_rng,
                history=None if leaf.history is None else leaf.history.copy(),
                slots=child_slots,
                owned_from=position,
                on_policy_spine=False,
                upstream=(branch, candidate),
                reach_weight=parent_reach * reach_by_candidate[candidate],
                bid_group=group,
                covered_until=position,
                parent=leaf,
            )
            leaf.tree.leaf_total += 1
            self._total_leaves += 1
            self._advance_leaf(
                child, self._action_for(actor, phase, candidate), appends, alive_next
            )
        leaf.reach_weight = parent_reach * reach_by_candidate[sampled]
        self._advance_leaf(
            leaf, self._action_for(actor, phase, sampled), appends, alive_next
        )

    def _expand_bid_split(
        self,
        leaf: SeqLeaf,
        probs: np.ndarray,
        sampled: int,
        legal: list[int],
        position: int,
        appends: list[tuple[SeqLeaf, list[GameEvent]]],
        alive_next: list[SeqLeaf],
        row_counters: dict[str, int],
        branch_copies: dict[str, tuple[list[int], list[int]]],
    ) -> None:
        """Expand this pass's share of the focal's bid candidates.

        The bid expansion itself is shared across passes: pass 0 creates the
        branch record and owns the prefix positions, and every pass resolves a
        disjoint subset of the children. Once the last pass finishes, the
        record has all its children and backs up exactly as an unsplit tree
        would.
        """

        tree = leaf.tree
        group_index, group_total = self._split_plan
        self._split_done.add(id(tree))
        actor = leaf.env.current_player()
        first_pass = tree.split_bid_record is None

        if first_pass:
            candidates, backup_weights, deterministic_count, inclusion = (
                self._branch_candidates(leaf, Phase.BIDDING, legal, probs, sampled)
            )
            if len(candidates) <= 1:
                # Nothing to split; fall back to the ordinary path.
                self._split_done.discard(id(tree))
                self._split_plan = None
                try:
                    self._decide_and_step(
                        leaf,
                        Phase.BIDDING,
                        probs,
                        sampled,
                        legal,
                        position,
                        True,
                        appends,
                        alive_next,
                        row_counters,
                        branch_copies,
                        set(),
                    )
                finally:
                    self._split_plan = (group_index, group_total)
                return
            raw = {index: float(probs[index]) for index in candidates}
            if backup_weights is not None:
                priors = tuple(backup_weights)
                mass = 1.0
            else:
                mass = float(sum(raw.values()))
                priors = tuple(raw[index] / mass for index in candidates)
            record = SeqBranchRecord(
                candidate_indices=tuple(candidates),
                prior_probs=priors,
                raw_probs=raw,
                inclusion_probs=tuple(inclusion),
                deterministic_count=deterministic_count,
                sampled_index=sampled,
                candidate_mass=mass,
                upstream=None,
            )
            tree.split_bid_record = record
            tree.split_bid_candidates = tuple(candidates)
            tree.split_post_sample_rng_state = leaf.rng.getstate()
            # Same group layout an unsplit tree would build, so the per-group
            # budget each pass sees is identical to the unsplit one.
            tree.bid_groups = len(candidates)
            tree.leaves_by_bid_group = {}

            decision = SeqDecisionRecord(
                position=position - 1,
                phase=NEXT_BID,
                action_index=sampled,
                old_probs=probs,
                old_value=leaf.pending.value,
                reach_weight=leaf.reach_weight,
                branch=record,
            )
            leaf.decisions.append(decision)
            leaf.segments.append((leaf.open_positions, record, leaf.reach_weight))
            leaf.open_positions = []
            tree.decision_total += 1
            self.stats.decisions += 1
            self.stats.branch_decisions += 1
            tree.branch_decisions += 1
            tree.branched_tricks.add(-1)
            tree.deepest_branch_trick = max(tree.deepest_branch_trick, -1)
            tree.branch_decisions_by_stage[-1] = (
                tree.branch_decisions_by_stage.get(-1, 0) + 1
            )
            tree.leaves_added_by_stage[-1] = (
                tree.leaves_added_by_stage.get(-1, 0) + len(candidates) - 1
            )
            tree.branch_layers += 1
        else:
            record = tree.split_bid_record
            leaf.rng.setstate(tree.split_post_sample_rng_state)

        all_candidates = tree.split_bid_candidates
        group = list(all_candidates[group_index::group_total])
        if not group:
            return
        groups_of = {candidate: all_candidates.index(candidate) for candidate in group}
        reach_by_candidate = dict(zip(record.candidate_indices, record.prior_probs))
        for candidate in group:
            tree.leaves_by_bid_group[groups_of[candidate]] = 1

        added = len(group) if not first_pass else len(group) - 1
        tree.leaf_total += added
        self._total_leaves += added

        crn_state = leaf.rng.getstate()
        # The parent leaf carries the first candidate; the rest are clones.
        for candidate in group[1:]:
            child_rng = _rng_from_state(crn_state)
            child_slots: dict[int, tuple[str, int]] = {}
            if self.use_cache:
                for seat, (policy_id, parent_row) in leaf.slots.items():
                    child_row = row_counters[policy_id]
                    row_counters[policy_id] += 1
                    child_slots[seat] = (policy_id, child_row)
                    parents, children = branch_copies.setdefault(policy_id, ([], []))
                    parents.append(parent_row)
                    children.append(child_row)
            else:
                child_slots = dict(leaf.slots)
            child = SeqLeaf(
                env=leaf.env.clone_for_rollout(),
                tree=tree,
                rng=child_rng,
                history=None if leaf.history is None else leaf.history.copy(),
                slots=child_slots,
                owned_from=position,
                on_policy_spine=(candidate == record.sampled_index),
                upstream=(record, candidate),
                reach_weight=reach_by_candidate[candidate],
                bid_group=groups_of[candidate],
                covered_until=position,
                parent=leaf,
            )
            self._advance_leaf(
                child,
                self._action_for(actor, Phase.BIDDING, candidate),
                appends,
                alive_next,
            )
        leaf.owned_from = min(leaf.owned_from, position) if first_pass else position
        leaf.on_policy_spine = group[0] == record.sampled_index
        leaf.upstream = (record, group[0])
        leaf.reach_weight = reach_by_candidate[group[0]]
        leaf.bid_group = groups_of[group[0]]
        self._advance_leaf(
            leaf,
            self._action_for(actor, Phase.BIDDING, group[0]),
            appends,
            alive_next,
        )

    def _is_branch_point(
        self, leaf: SeqLeaf, branch_factor: int, trick_index: int
    ) -> bool:
        """Should this focal play decision be one of the path's branch points?

        The draw comes off the leaf's own CRN tape, so a bid-split pass makes
        the same choices as the unsplit run.
        """

        config = self.train.branch_budget
        probability = config.rate_for_shape(leaf.tree.num_players, leaf.tree.hand_size)
        if probability is None:
            raise AssertionError(
                "No branch_rate for "
                f"({leaf.tree.num_players}p, {leaf.tree.hand_size}c); "
                "SeqTrainingConfig.validate() should have caught this."
            )
        if config.branch_rate_decay:
            progress = trick_index / max(leaf.tree.hand_size, 1)
            probability *= max(1.0 - progress, 0.0) ** config.branch_rate_decay
        if probability >= 1.0:
            return True
        if probability <= 0.0:
            return False
        return leaf.rng.random() < probability

    @staticmethod
    def _draw(probs: np.ndarray, rng: random.Random) -> int:
        cumulative = np.cumsum(probs)
        index = int((cumulative < rng.random()).sum())
        legal = np.flatnonzero(probs > 0)
        return int(min(index, legal[-1]))

    def _branch_candidates(
        self,
        leaf: SeqLeaf,
        phase: Phase,
        legal: list[int],
        probs: np.ndarray,
        sampled: int,
    ) -> tuple[list[int], Optional[list[float]], int, list[float]]:
        """Return (candidates, backup weights or None, deterministic count, q).

        Stochastic rules return the weights for their realized unbiased
        old-policy value estimator: stratum masses for stratified sampling or
        empirical frequencies for collapsed iid draws. Deterministic rules
        return None and the caller renormalizes their covered policy mass.

        ``q`` is the inclusion probability of each returned candidate -- the
        chance this rule would have expanded that action at this node. It is
        what the NeuRD loss divides by, so a rule that expands an action only
        when the policy happens to sample it does not silently reintroduce the
        pi(a) prefactor NeuRD exists to remove.
        """

        rule = self.train.branch_rule
        if phase == Phase.BIDDING:
            mode, top_k = rule.bid_rule()
        else:
            trick_index = len(
                [t for t in leaf.env.state.current_round.tricks if t.winner is not None]
            )
            mode, top_k = rule.play_rule_for_trick(trick_index)
        return self._candidates_for_mode(leaf, legal, probs, sampled, mode, top_k)

    def _candidates_for_mode(
        self,
        leaf: SeqLeaf,
        legal: list[int],
        probs: np.ndarray,
        sampled: int,
        mode: str,
        top_k: int,
    ) -> tuple[list[int], Optional[list[float]], int, list[float]]:
        """Apply one selection rule, with no knowledge of which phase asked.

        Bids and plays share this outright rather than each holding a copy:
        ``bid_mode = "same_as_play"`` has to mean *the same rule*, and two
        parallel implementations agreeing today is not the same guarantee as
        one implementation serving both.
        """

        if mode == "none":
            return [sampled], None, 0, [float(probs[sampled])]

        if mode == "stratified":
            if top_k < 1:
                raise ValueError("top_k must be >= 1 for stratified.")
            if len(legal) <= top_k:
                ordered = sorted(legal, key=lambda index: (-probs[index], index))
                return ordered, None, len(ordered), [1.0] * len(ordered)

            groups = self._policy_mass_strata(legal, probs, top_k)
            candidates: list[int] = []
            weights: list[float] = []
            inclusion: list[float] = []
            for group in groups:
                mass = float(sum(float(probs[action]) for action in group))
                if sampled in group:
                    representative = sampled
                else:
                    threshold = leaf.rng.random() * mass
                    cumulative = 0.0
                    representative = group[-1]
                    for action in group:
                        cumulative += float(probs[action])
                        if threshold < cumulative:
                            representative = action
                            break
                candidates.append(representative)
                weights.append(mass)
                inclusion.append(float(probs[representative]) / mass)
            return candidates, weights, 0, inclusion

        if mode in ("sample_k", "sample_k_plus_uniform"):
            if top_k < 1:
                raise ValueError(f"top_k must be >= 1 for {mode}.")
            # The realized action is the first draw, so the spine child stays
            # the on-policy one and all k draws are i.i.d. from the policy.
            draws = [sampled]
            for _ in range(top_k - 1):
                draws.append(self._draw(probs, leaf.rng))
            counts: dict[int, int] = {}
            for action in draws:
                counts[action] = counts.get(action, 0) + 1
            uniform_rate = 0.0
            if mode == "sample_k_plus_uniform":
                # Explored uniformly, so it must not enter the backup: adding
                # it with weight 1/(k+1) would tilt the parent's value toward
                # actions the policy does not actually take. counts.setdefault
                # leaves the weight at zero unless the policy also drew it.
                explored = leaf.rng.choice(legal)
                counts.setdefault(explored, 0)
                uniform_rate = 1.0 / len(legal)
            candidates = sorted(counts)
            weights = [counts[action] / top_k for action in candidates]
            # k i.i.d. policy draws, plus an independent uniform arm. The
            # uniform arm is what puts a 1/|legal| floor under q, which is
            # what keeps 1/q bounded.
            inclusion = [
                1.0 - (1.0 - float(probs[action])) ** top_k * (1.0 - uniform_rate)
                for action in candidates
            ]
            return candidates, weights, len(candidates), inclusion

        if mode == "all_legal":
            ordered = sorted(legal, key=lambda index: (-probs[index], index))
            return ordered, None, len(ordered), [1.0] * len(ordered)
        raise ValueError(f"Unknown branch mode {mode!r}.")

    @staticmethod
    def _policy_mass_strata(
        legal: list[int], probs: np.ndarray, count: int
    ) -> list[list[int]]:
        """Partition legal actions into nonempty, disjoint mass-balanced strata.

        Longest-processing-time bin packing is deterministic given the frozen
        policy: place actions from highest to lowest probability into the
        currently lightest stratum. Sampling one conditional-policy action per
        stratum then gives exactly ``count`` distinct actions, backup/reach
        weights that sum to one, and closed-form q(a) = pi(a) / mass(stratum).
        """

        if not 1 <= count <= len(legal):
            raise ValueError("stratum count must be in [1, len(legal)].")
        groups: list[list[int]] = [[] for _ in range(count)]
        masses = [0.0] * count
        ranked = sorted(legal, key=lambda action: (-probs[action], action))
        for action in ranked:
            group_index = min(range(count), key=lambda index: (masses[index], index))
            groups[group_index].append(action)
            masses[group_index] += float(probs[action])
        return groups

    @staticmethod
    def _action_for(
        player: int, phase: Phase, index: int
    ) -> BidAction | PlayCardAction:
        if phase == Phase.BIDDING:
            return BidAction(player, index)
        return PlayCardAction(player, card_from_id(index))

    def _advance_leaf(
        self,
        leaf: SeqLeaf,
        action: BidAction | PlayCardAction,
        appends: list[tuple[SeqLeaf, list[GameEvent]]],
        alive_next: list[SeqLeaf],
    ) -> None:
        env = leaf.env
        log_start = len(env.state.event_log)
        env.step_unchecked(action)
        if not env.is_done() and self._is_forced_runout(env):
            while not env.is_done():
                player = env.current_player()
                hand = env.state.current_round.current_hands[player]
                if len(hand) != 1:
                    raise AssertionError("Forced run-out expects one legal action.")
                env.step_unchecked(PlayCardAction(player, hand[0]))
        if env.is_done():
            self._finalize_leaf(leaf)
            return
        new_events = [
            event
            for event in env.state.event_log[log_start:]
            if event.type in (EventType.BID, EventType.PLAY, EventType.TRICK_WIN)
        ]
        appends.append((leaf, new_events))
        alive_next.append(leaf)

    @staticmethod
    def _is_forced_runout(env: PlumpEnv) -> bool:
        if env.phase() != Phase.PLAYING:
            return False
        # At a valid decision the next actor is one of the players with the
        # most cards remaining. One lookup is therefore equivalent to scanning
        # every hand, and this predicate runs once per leaf per public step.
        player = env.current_player()
        return len(env.state.current_round.current_hands[player]) <= 1

    def _finalize_leaf(self, leaf: SeqLeaf) -> None:
        scores = leaf.env.state.current_round.round_scores
        leaf.terminal_value = compute_relative_rewards(scores)[leaf.tree.focal]
        # Positions past the last appended token (forced run-out tail, or a
        # child terminating right after its branch action) still belong to
        # this leaf's terminal segment.
        total_len = self.model_config.seq_len(
            leaf.tree.num_players, leaf.tree.hand_size
        )
        leaf.open_positions.extend(range(leaf.covered_until, total_len))
        leaf.covered_until = total_len
        leaf.segments.append((leaf.open_positions, None, leaf.reach_weight))
        leaf.open_positions = []
        if leaf.upstream is not None:
            leaf.upstream[0].resolve(leaf.upstream[1], leaf.terminal_value)
        leaf.pending = None
        leaf.tree.leaves.append(leaf)

    # ------------------------------------------------------------------ #
    # Batched cache forwards                                              #
    # ------------------------------------------------------------------ #

    def _prefill(
        self,
        leaves: list[SeqLeaf],
        models: dict[str, SeqPlumpModel],
        prefix_len: int,
    ) -> None:
        rows_by_policy: dict[str, list[tuple[SeqLeaf, int, np.ndarray]]] = {}
        for leaf in leaves:
            tree = leaf.tree
            if not self.use_cache:
                leaf.history = np.zeros(
                    (
                        tree.num_players,
                        self.model_config.seq_len(tree.num_players, tree.hand_size),
                        TOKEN_WIDTH,
                    ),
                    dtype=np.int64,
                )
            for seat in leaf.slots:
                rows = np.asarray(
                    prefix_tokens(
                        self.model_config,
                        seat,
                        tree.num_players,
                        tree.hand_size,
                        tree.initial_hands[seat],
                        tree.bidding_start_player,
                    ),
                    dtype=np.int64,
                )
                if leaf.history is not None:
                    leaf.history[seat, :prefix_len] = rows
                policy_id = leaf.slots[seat][0]
                rows_by_policy.setdefault(policy_id, []).append((leaf, seat, rows))
        for policy_id, entries in rows_by_policy.items():
            tokens = torch.from_numpy(
                np.stack([tokens for _, _, tokens in entries])
            ).to(self.device)
            # Root rows were assigned densely in this same iteration order.
            if self.use_cache:
                output = models[policy_id].forward_prefill(
                    tokens, self._caches[policy_id], None
                )
            else:
                output = models[policy_id].forward_prefix(tokens)
            self.stats.forward_rows += len(entries) * prefix_len
            bid_logits = output.bid_logits.float().cpu().numpy()
            card_logits = output.card_logits.float().cpu().numpy()
            values = output.value.float().cpu().numpy()
            for row, (leaf, seat, _) in enumerate(entries):
                if leaf.env.state.current_player == seat:
                    leaf.pending = _Pending(
                        seat=seat,
                        bid_logits=bid_logits[row],
                        card_logits=card_logits[row],
                        value=float(values[row]),
                    )

    @staticmethod
    def _next_play_slot(events: list[GameEvent]) -> tuple[int, int]:
        """(trick index, position in trick) of the play that follows a wave."""

        for event in reversed(events):
            if event.type == EventType.TRICK_WIN:
                return event.trick_index + 1, 0
            if event.type == EventType.PLAY:
                return event.trick_index, event.position_in_trick + 1
        return 0, 0  # the wave was a bid; play opens at the first trick

    def _wave_blocks(self, leaf: SeqLeaf, events: list[GameEvent]) -> list[np.ndarray]:
        """Per-seat token rows this wave appends: one [P, WIDTH] block each.

        A wave is one action plus whatever it implied (the TRICK_WIN closing a
        trick), then the next actor's TURN token when TURN tokens are on. Every
        seat gets a token at every one of these positions -- only the
        observer-relative fields differ -- which is what keeps every cache row
        the same length and the loop's single scalar ``position`` valid. A
        TURN token appended to the acting seat alone would be ~1/P the tokens
        but would desynchronise the rows by one, and the dense zero-copy cache
        read that makes the wave loop fast needs a single row length.
        """

        config = self.model_config
        tree = leaf.tree
        num_players = tree.num_players
        hand_size = tree.hand_size
        seats = np.arange(num_players)

        def tile(row: list[int], reference_player: int) -> np.ndarray:
            block = np.empty((num_players, TOKEN_WIDTH), dtype=np.int64)
            block[:] = row
            block[:, SLOT_REL_PLAYER] = (reference_player - seats) % num_players
            return block

        blocks = []
        for event in events:
            if not emits_token(config, event):
                continue
            block = tile(
                event_token(config, event, 0, num_players, hand_size),
                event.player,
            )
            if event.type == EventType.TRICK_WIN:
                current_hands = leaf.env.state.current_round.current_hands
                for seat in range(num_players):
                    set_remaining_hand(block[seat], current_hands[seat], hand_size)
            blocks.append(block)
        next_actor = leaf.env.state.current_player
        bidding = leaf.env.phase() == Phase.BIDDING
        next_phase = NEXT_BID if bidding else NEXT_PLAY
        if turn_token_for_phase(config, next_phase):
            trick, pos = (None, None) if bidding else self._next_play_slot(events)
            blocks.append(
                tile(
                    turn_token(
                        config, num_players, hand_size, 0, next_phase, trick, pos
                    ),
                    next_actor,
                )
            )
        blocks[-1][:, SLOT_NEXT_ACTOR] = (next_actor - seats) % num_players
        blocks[-1][:, SLOT_NEXT_PHASE] = next_phase
        return blocks

    def _append_wave(
        self,
        appends: list[tuple[SeqLeaf, list[GameEvent]]],
        models: dict[str, SeqPlumpModel],
        position: int,
        row_counters: dict[str, int],
    ) -> int:
        blocks_by_leaf = [
            (leaf, self._wave_blocks(leaf, events)) for leaf, events in appends
        ]
        event_count = len(blocks_by_leaf[0][1])
        if any(len(blocks) != event_count for _, blocks in blocks_by_leaf):
            raise AssertionError("Wave-synchronized leaves diverged in token count.")
        # Only the last token of a wave produces a readout -- a trick's
        # completing play is appended together with its TRICK_WIN token, and
        # nobody acts in between. So the cached path appends the whole run in
        # one forward instead of one per token; the earlier ones exist purely
        # to advance the cache. The cache-free path cannot merge: it re-encodes
        # the prefix from scratch, so every offset has a different length.
        runs = (
            [(0, event_count)]
            if self.use_cache
            else [(offset, 1) for offset in range(event_count)]
        )
        for run_start, run_len in runs:
            t_start = time.perf_counter()
            width = position + run_start + 1 if not self.use_cache else run_len
            # Cached mode addresses rows directly, so the batch is laid out by
            # cache row; rows belonging to already-terminated leaves stay
            # padding and their outputs are ignored.
            token_arrays: dict[str, np.ndarray] = (
                {
                    policy_id: np.zeros((count, width, TOKEN_WIDTH), dtype=np.int64)
                    for policy_id, count in row_counters.items()
                    if count > 0
                }
                if self.use_cache
                else {}
            )
            token_blocks: dict[str, list[np.ndarray]] = {}
            captures: dict[str, list[tuple[int, int, SeqLeaf]]] = {}
            for leaf, blocks in blocks_by_leaf:
                for step in range(run_len):
                    offset = run_start + step
                    last = offset == event_count - 1
                    block = blocks[offset]
                    next_actor = leaf.env.state.current_player if last else None
                    if leaf.history is not None:
                        leaf.history[:, position + offset] = block
                    for seat, (policy_id, row) in leaf.slots.items():
                        if self.use_cache:
                            token_arrays[policy_id][row, step] = block[seat]
                            capture_row = row
                        else:
                            rows = token_blocks.setdefault(policy_id, [])
                            rows.append(leaf.history[seat, : position + offset + 1])
                            capture_row = len(rows) - 1
                        if next_actor == seat:
                            captures.setdefault(policy_id, []).append(
                                (capture_row, seat, leaf)
                            )
            t_build = time.perf_counter()
            self.stats.token_build_sec += t_build - t_start
            batches = (
                token_arrays
                if self.use_cache
                else {key: np.stack(rows) for key, rows in token_blocks.items()}
            )
            for policy_id, block_array in batches.items():
                row_count = block_array.shape[0]
                tokens = torch.from_numpy(block_array).to(self.device)
                if self.use_cache:
                    output = models[policy_id].forward_step(
                        tokens,
                        position + run_start,
                        self._caches[policy_id],
                        None,
                    )
                else:
                    output = models[policy_id].forward_prefix(tokens)
                self._sync()
                self.stats.forward_rows += row_count * (
                    run_len if self.use_cache else position + run_start + 1
                )
                capture_list = captures.get(policy_id)
                if capture_list:
                    # via numpy: torch.tensor() on a Python list walks it
                    # element by element, which is ~1.5x the cost of a numpy
                    # buffer copy at wave-sized lists.
                    indices = torch.from_numpy(
                        np.fromiter(
                            (row for row, _, _ in capture_list),
                            dtype=np.int64,
                            count=len(capture_list),
                        )
                    ).to(self.device)
                    bid_logits = output.bid_logits[indices].float().cpu().numpy()
                    card_logits = output.card_logits[indices].float().cpu().numpy()
                    values = output.value[indices].float().cpu().numpy()
                    for i, (_, seat, leaf) in enumerate(capture_list):
                        leaf.pending = _Pending(
                            seat=seat,
                            bid_logits=bid_logits[i],
                            card_logits=card_logits[i],
                            value=float(values[i]),
                        )
            self.stats.forward_sec += time.perf_counter() - t_build
            if run_start + run_len == event_count:
                for leaf, _ in appends:
                    for extra in range(event_count):
                        # Later split passes replay the shared prefix; only
                        # pass 0 owns those training positions.
                        if position + extra >= leaf.owned_from:
                            leaf.open_positions.append(position + extra)
                    leaf.covered_until = position + event_count
        return event_count
