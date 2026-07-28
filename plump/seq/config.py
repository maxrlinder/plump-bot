"""Configuration for the schema-v6 autoregressive sequence pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

SEQ_SCHEMA_VERSION = 6

NUM_CARDS = 52
NUM_SUITS = 4
NUM_RANKS = 13

# Token type ids (slot 0).
TOKEN_PAD = 0
TOKEN_GAME = 1
TOKEN_HAND = 2
TOKEN_BID = 3
TOKEN_PLAY = 4
TOKEN_TRICK_WIN = 5
# "Someone is about to act" -- rel_player 0 means the observer. Carries no game
# content beyond whose turn it is: it exists to be a blank canvas.
TOKEN_TURN = 6
NUM_TOKEN_TYPES = 7

# Next-phase ids (slot 11).
NEXT_NONE = 0
NEXT_BID = 1
NEXT_PLAY = 2
NUM_NEXT_PHASES = 3

TOKEN_WIDTH = 12

# Slot indices into a token row.
SLOT_TYPE = 0
SLOT_REL_PLAYER = 1
SLOT_RANK = 2
SLOT_SUIT = 3
SLOT_CARD = 4
SLOT_BID = 5
SLOT_TRICK = 6
SLOT_POS_IN_TRICK = 7
SLOT_HAND_SIZE = 8
SLOT_NUM_PLAYERS = 9
SLOT_NEXT_ACTOR = 10
SLOT_NEXT_PHASE = 11


# Where a TURN token is inserted, if any.
#   off  -> never
#   bid  -> before each bid only (the single highest-leverage decision, and it
#           costs P tokens instead of P + N*P)
#   all  -> before every bid and every card play
TurnTokenMode = Literal["off", "bid", "all"]


def seq_len(
    num_players: int,
    hand_size: int,
    *,
    trick_win_token: bool = True,
    turn_token: TurnTokenMode = "off",
) -> int:
    """[GAME] [HAND x N] [BID x P] { [PLAY x P] [TRICK_WIN] } x N.

    ``trick_win_token=False`` drops the per-trick winner token; ``turn_token``
    inserts a contentless "X is about to act" token before actions.
    """

    length = 1 + hand_size + num_players + hand_size * num_players
    if trick_win_token:
        length += hand_size
    if turn_token != "off":
        length += num_players
    if turn_token == "all":
        length += hand_size * num_players
    return length


@dataclass(frozen=True)
class SeqModelConfig:
    """Architecture and vocabulary limits for the causal sequence model."""

    schema_version: int = SEQ_SCHEMA_VERSION
    max_players: int = 5
    max_hand_size: int = 10

    # --- sequence schema knobs (change the token stream, not the trunk) ---
    # The trick winner is already derivable: these rounds have no trump, so the
    # highest card of the led suit wins, and the winner leads next -- which the
    # trick's last PLAY token announces in SLOT_NEXT_ACTOR. Dropping the token
    # costs ~13-16% of every sequence. What it buys back is (a) counting
    # tricks-won stops being "count tokens of type TRICK_WIN" and becomes a
    # two-slot conjunction, and (b) the model loses a compute step sitting
    # exactly where trick state has to be revised.
    trick_win_token: bool = True
    # A pause/register token before an action. The hidden state that the policy
    # head reads currently doubles as the representation of whatever event
    # happened last; a TURN token gives the head a position whose only job is
    # to be read, and buys n_layers of extra serial compute before acting.
    # It is appended to *every* seat's sequence at the same position (its
    # rel_player differs per observer), which is what keeps the wave loop
    # rectangular -- see the note in rollout._append_wave.
    turn_token: TurnTokenMode = "off"


    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    # None -> full multi-head KV (no GQA). Reduce only if memory-bound.
    n_kv_heads: Optional[int] = None
    d_ff: int = 768
    dropout: float = 0.0

    @property
    def kv_heads(self) -> int:
        return self.n_kv_heads if self.n_kv_heads is not None else self.n_heads

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def seq_len(self, num_players: int, hand_size: int) -> int:
        return seq_len(
            num_players,
            hand_size,
            trick_win_token=self.trick_win_token,
            turn_token=self.turn_token,
        )

    def prefix_len(self, hand_size: int) -> int:
        """[GAME] [HAND x N] plus the TURN token for the opening bid."""

        return 1 + hand_size + (0 if self.turn_token == "off" else 1)

    def bid_token_position(self, hand_size: int, bid_index: int) -> int:
        """Token position of the ``bid_index``-th bid event."""

        stride = 1 if self.turn_token == "off" else 2
        return self.prefix_len(hand_size) + bid_index * stride

    @property
    def max_seq_len(self) -> int:
        return self.seq_len(self.max_players, self.max_hand_size)

    @property
    def bid_count(self) -> int:
        return self.max_hand_size + 1

    @property
    def owner_class_count(self) -> int:
        # Relative opponents 1..max_players-1, then the undealt pool.
        return self.max_players

    @property
    def undealt_owner_class(self) -> int:
        return self.max_players - 1

    # NA ids used to pad token slots.
    @property
    def player_na_id(self) -> int:
        return self.max_players

    @property
    def rank_na_id(self) -> int:
        return NUM_RANKS

    @property
    def suit_na_id(self) -> int:
        return NUM_SUITS

    @property
    def card_na_id(self) -> int:
        return NUM_CARDS

    @property
    def bid_na_id(self) -> int:
        return self.max_hand_size + 1

    @property
    def trick_na_id(self) -> int:
        return self.max_hand_size

    @property
    def pos_na_id(self) -> int:
        return self.max_players

    @property
    def slot_vocab_sizes(self) -> tuple[int, ...]:
        return (
            NUM_TOKEN_TYPES,
            self.max_players + 1,
            NUM_RANKS + 1,
            NUM_SUITS + 1,
            NUM_CARDS + 1,
            self.max_hand_size + 2,
            self.max_hand_size + 1,
            self.max_players + 1,
            self.max_hand_size + 1,
            self.max_players + 1,
            self.max_players + 1,
            NUM_NEXT_PHASES,
        )

    def validate(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.kv_heads > self.n_heads or self.n_heads % self.kv_heads != 0:
            raise ValueError("n_kv_heads must divide n_heads.")
        if len(self.slot_vocab_sizes) != TOKEN_WIDTH:
            raise AssertionError("Slot vocabulary must match TOKEN_WIDTH.")


PlayBranchMode = Literal[
    "all_legal",
    "top_k",
    "top_k_plus_random",
    # k i.i.d. draws from the masked policy (the realized action is the first
    # draw). Duplicates collapse into multiplicity weights, so the backup is
    # the unbiased Monte-Carlo value estimate rather than a top-k truncation.
    "sample_k",
    # ``sample_k`` plus one extra arm drawn uniformly over the legal actions.
    # The uniform arm carries zero backup weight, so the parent's value stays
    # the unbiased on-policy estimate; it exists purely so the policy loss sees
    # a Q-value at an action the policy currently under-weights.
    "sample_k_plus_uniform",
    # k *distinct* actions: the realized action, then a Gumbel-top-(k-1) draw
    # over the rest. argtop(log pi + Gumbel) is exactly sampling without
    # replacement (Kool et al., ICML 2019), so the extra arms still follow the
    # policy but can never repeat. Duplicates are pure waste in a branching
    # tree -- a second copy of an action costs a full subtree and adds no
    # counterfactual -- so this dominates ``sample_k`` at equal k.
    "gumbel_top_k",
    "none",
]


@dataclass(frozen=True)
class StageBranchRule:
    """Branch-rule override active from ``from_trick`` onward (bidding = -1)."""

    from_trick: int
    play_mode: PlayBranchMode = "all_legal"
    play_top_k: int = 4


@dataclass(frozen=True)
class ShapeBranchRate:
    """Per-shape override of ``branch_rate``. ``None`` matches any value.

    One global rate cannot be right. A branch point multiplies a path by the
    branching factor, so the same rate compounds over ``hand_size`` decisions:
    at rate 0.5 a 10-card game grows ~2^4.5 more than a 3-card one. Holding the
    rate flat therefore spends nearly the whole budget on the long games and
    leaves the short ones barely branched, even though a 3-card round is worth
    exactly as many points.
    """

    rate: float
    num_players: Optional[int] = None
    hand_size: Optional[int] = None

    @property
    def specificity(self) -> int:
        return (self.num_players is not None) + (self.hand_size is not None)

    def matches(self, num_players: int, hand_size: int) -> bool:
        return self.num_players in (None, num_players) and self.hand_size in (
            None,
            hand_size,
        )


@dataclass(frozen=True)
class BranchRuleConfig:
    """Pluggable branching restrictions; tunable without code changes."""

    # Bidding: branch the top (bid_top_k - 1) bids by policy probability plus
    # the unconditional raw-policy sample (control-variate estimator).
    # 0 disables the cap (branch every legal bid).
    bid_top_k: int = 4
    play_mode: PlayBranchMode = "all_legal"
    play_top_k: int = 4
    stage_rules: tuple[StageBranchRule, ...] = ()

    def play_rule_for_trick(self, trick_index: int) -> tuple[PlayBranchMode, int]:
        mode, top_k = self.play_mode, self.play_top_k
        for rule in self.stage_rules:
            if trick_index >= rule.from_trick:
                mode, top_k = rule.play_mode, rule.play_top_k
        return mode, top_k


@dataclass(frozen=True)
class BranchBudgetConfig:
    """Where branch points land, and nothing else.

    There used to be a leaf budget here, interpreted as a floor: branch every
    layer until the limit is crossed, then never again. It is gone. Two things
    were wrong with it. It spent everything on the bid and the opening tricks
    and left the endgame -- where counterfactuals are most decidable -- wholly
    unbranched; and a leaf count was only ever a proxy for the resource that is
    actually finite, which is KV cache rows.

    So branching is now decided per focal play decision by ``branch_rate``, and
    the only limit is ``RolloutOptions.max_cache_rows``. The rate is the
    control; the row cap is the backstop, and a run that hits it is a run whose
    rate is set too high for its budget (rows run out *late*, so the cap
    truncates exactly the endgame the rate exists to reach).
    """

    # Probability that an eligible play decision is a branch point. A path
    # through an N-card game has N - 1 of them, so the rate compounds over game
    # length -- which is why one flat value cannot be right across the grid and
    # branch_rate_by_shape exists.
    branch_rate: Optional[float] = None
    # Per-shape overrides of branch_rate, most specific match wins. Build one
    # with build_branch_rate_table() rather than by hand.
    branch_rate_by_shape: tuple[ShapeBranchRate, ...] = ()
    # Shape of the rate over the game: rate * (1 - trick/hand_size) ** decay.
    # 0 is flat. Positive values branch less late, which is the right direction
    # if you think the endgame is largely determined by then; negative values
    # push exploration towards the endgame.
    branch_rate_decay: float = 0.0

    def rate_for_shape(
        self, num_players: int, hand_size: int
    ) -> Optional[float]:
        """The branch rate this shape should use, or None if uncovered."""

        rate = self.branch_rate
        best = -1
        for rule in self.branch_rate_by_shape:
            if not rule.matches(num_players, hand_size):
                continue
            if rule.specificity > best:
                rate, best = rule.rate, rule.specificity
        return rate


HistoricalArmMode = Literal["off", "paired", "separate"]


@dataclass(frozen=True)
class RolloutOptions:
    """How deals, arms and large trees are packed into rollouts."""

    # Deals of the same (players, hand size) that advance in one wave loop.
    # Games are fixed length, so same-shape deals stay in lockstep and simply
    # widen the batch. >1 raises GPU utilisation on small hands.
    deals_per_batch: int = 1

    # Choose deals_per_batch per shape instead, from measured cache rows per
    # deal. One global value cannot be right: a 3-card 3-player deal occupies
    # ~30 cache rows and a 5-player 8-card deal ~3000, so the value that keeps
    # the big shape under the memory ceiling leaves the small one running at a
    # small fraction of achievable throughput.
    auto_deals_per_batch: bool = False
    # Cache rows one wave loop may reach. None derives it from cache_budget_gb.
    # Tree size varies severalfold between deals of the same shape, so this is
    # deliberately a soft target: auto_deals_headroom absorbs the overshoot.
    auto_target_rows: Optional[int] = None
    auto_deals_headroom: float = 0.5
    max_deals_per_batch: int = 64

    # How the focal's seat is picked. Absolute seat is invisible to the model:
    # every player reference in a sequence is observer-relative, so the only
    # positional signal is the focal's *bidding position*, (focal - start) mod
    # P, carried on the GAME token. "cycle" therefore walks bidding position
    # round-robin per player count -- balancing absolute seat would balance a
    # relabeling. "uniform" draws the seat uniformly, which leaves bidding
    # position unbiased but lumpy at these deal counts.
    bid_position_mode: Literal["uniform", "cycle"] = "cycle"

    # off      -> self-play only, no historical opponents.
    # paired   -> historical arm shares the deal and the same wave loop
    #             (matched pair, common random numbers).
    # separate -> historical arm runs as its own wave loop, which frees it to
    #             use a different deal.
    historical_arm: HistoricalArmMode = "paired"
    # Only meaningful for "separate": reuse the self arm's deal or draw a new
    # one. A fresh deal buys hand diversity; the same deal keeps the matched
    # comparison.
    historical_same_deal: bool = True

    # Memory the KV cache may occupy. Rows are derived from this and the
    # model's actual bytes-per-row, because that cost scales with depth and
    # head count -- a fixed row count silently doubles the footprint when the
    # model grows. Leave headroom: activations and token buffers sit on top.
    cache_budget_gb: float = 8.0
    # Explicit row override; None derives rows from cache_budget_gb.
    max_cache_rows: Optional[int] = None
    # Rows reserved before any tree has been measured. The pool then tracks
    # the widest tree actually seen, so this only sets how much the first few
    # deals may have to grow -- growth between wave loops is free, growth
    # inside one copies the live cache.
    cache_initial_rows: int = 1024
    # Reserve the whole row budget up front instead of tracking the widest tree
    # seen per shape. The budget has to be paid for eventually, so if grow
    # steps cost measurable time this trades nothing away -- but a pool sized
    # for the worst shape is resident while the narrow shapes run, so it only
    # pays off if allocation really is the bottleneck. Measure before enabling.
    cache_preallocate: bool = False

    # Split one deal's tree across N wave loops partitioned by the focal's bid
    # candidates (e.g. 2 gives two rollouts starting from 2 bids each). The
    # shared prefix is replayed per pass and the cache is freed between them,
    # so peak memory scales down roughly by this factor. 1 disables splitting.
    bid_split_groups: int = 1
    # Apply the split only from this hand size upward; smaller hands are cheap
    # and pay the replay cost for nothing.
    bid_split_min_hand_size: int = 0

    def validate(self) -> None:
        if self.deals_per_batch < 1:
            raise ValueError("deals_per_batch must be >= 1.")
        if self.bid_split_groups < 1:
            raise ValueError("bid_split_groups must be >= 1.")
        if self.max_deals_per_batch < 1:
            raise ValueError("max_deals_per_batch must be >= 1.")
        if not 0.0 <= self.auto_deals_headroom < 1.0:
            raise ValueError("auto_deals_headroom must be in [0, 1).")

    def splits_for(self, hand_size: int) -> int:
        if hand_size < self.bid_split_min_hand_size:
            return 1
        return self.bid_split_groups


@dataclass(frozen=True)
class GameScheduleCell:
    """One entry of the per-update game schedule."""

    hand_size: int
    # None -> sampled from SeqTrainingConfig.player_count_weights.
    num_players: Optional[int] = None
    # None -> uniform random focal seat.
    focal_seat: Optional[int] = None
    games: int = 1


def default_schedule_cells() -> tuple[GameScheduleCell, ...]:
    return tuple(GameScheduleCell(hand_size=n) for n in range(3, 11))


def _apportion(weights: dict, total: int) -> dict:
    """Largest-remainder apportionment of ``total`` across ``weights``.

    Sampling the mix would make the realized composition of an update a random
    variable with only as many draws as there are cells, so the (players,
    cards) mix would rattle from update to update on top of the deal noise we
    actually want. Apportioning makes the mix exact and leaves the randomness
    where it belongs: in the deals.
    """

    mass = sum(weights.values())
    if mass <= 0:
        raise ValueError("Schedule weights must have positive mass.")
    exact = {key: total * weight / mass for key, weight in weights.items()}
    counts = {key: int(value) for key, value in exact.items()}
    shortfall = total - sum(counts.values())
    order = sorted(
        exact, key=lambda key: (-(exact[key] - counts[key]), key)
    )
    for key in order[:shortfall]:
        counts[key] += 1
    return counts


def build_branch_rate_table(
    reference_rate: float,
    *,
    exhaustive_until: int = 7,
    reference_hand_size: int = 10,
    reference_players: int = 5,
    hand_sizes: tuple[int, ...] = tuple(range(3, 11)),
    player_counts: tuple[int, ...] = (3, 4, 5),
    player_exponent: float = 0.0,
) -> tuple[ShapeBranchRate, ...]:
    """Per-shape branch rates: exhaustive on short games, tapering on long ones.

    A path through an ``N``-card game has ``N - 1`` branchable decisions (the
    last card is forced -- in practice slightly fewer, since follow-suit often
    leaves a single legal card), and each branch point multiplies the path by
    the branching factor. A tree is therefore about ``b ** (rate * (N - 1))``
    leaves: the rate compounds over the *length* of the game. One flat rate
    across the grid is thus not one amount of branching -- at 0.5 a 10-card
    tree is ~2^4.5 the size of a 3-card one, so the long games take the whole
    budget and the short ones stay nearly unbranched, even though a 3-card
    round pays the same points.

    Two regimes, because the cost curve has two regimes:

    - ``N <= exhaustive_until``: rate 1.0, branch every eligible decision.
      Measured, these games' wall time is set by the number of waves (one
      forward per game event) rather than by tree size, so the branching is
      nearly free and there is no reason to sample.
    - above it: taper geometrically to ``reference_rate`` at
      ``reference_hand_size``, i.e. equal multiplicative steps per extra card.

    The taper is deliberately steeper than the "equal tree size" law
    (``rate * (N - 1)`` constant) would give. Equal tree size is the right
    target when nothing binds; past ~8 cards time and memory do bind, and they
    bind super-linearly, so the long games are thinned harder than parity.

    ``player_exponent`` scales the rate by ``(reference_players / P) ** e``.
    Rows scale with P (one cache row per seat per leaf), so a positive exponent
    buys back some of that; 0.0 leaves player count alone, which the measured
    grid supports -- the spread across P at fixed N is within deal noise.
    """

    if not 0.0 < reference_rate <= 1.0:
        raise ValueError("reference_rate must be in (0, 1].")
    if exhaustive_until >= reference_hand_size:
        raise ValueError(
            "exhaustive_until must be below reference_hand_size, or there is "
            "nothing left to taper over."
        )
    span = reference_hand_size - exhaustive_until

    def rate_for(hand_size: int, players: int) -> float:
        steps = max(hand_size - exhaustive_until, 0)
        rate = reference_rate ** (steps / span)
        return min(1.0, rate * (reference_players / players) ** player_exponent)

    return tuple(
        ShapeBranchRate(
            rate=rate_for(hand_size, players),
            num_players=players,
            hand_size=hand_size,
        )
        for players in player_counts
        for hand_size in hand_sizes
    )


def build_position_balanced_schedule(
    hand_sizes: tuple[int, ...] = tuple(range(3, 11)),
    player_counts: tuple[int, ...] = (3, 4, 5),
    repeats: int = 1,
) -> tuple[GameScheduleCell, ...]:
    """One deal per (player count, hand size, bidding position).

    A 3-player cell gets 3 deals and a 5-player cell 5, so every bidding
    position of every shape is covered exactly ``repeats`` times per update.
    Each of those deals is dealt independently -- the seat rotation must not be
    the same hand seen from a different chair, or the update would contain P
    correlated copies of one deal instead of P samples.

    Note this drops the tilt toward longer games in *deal counts*: every hand
    size gets the same number of deals. Long games still dominate compute, and
    ``tree_weight_exponent`` is the knob for how much of the gradient follows
    that compute.
    """

    if repeats < 1:
        raise ValueError("repeats must be >= 1.")
    return tuple(
        GameScheduleCell(
            hand_size=hand_size, num_players=players, games=players * repeats
        )
        for players, hand_size in sorted(
            ((p, n) for p in player_counts for n in hand_sizes),
            key=lambda shape: -shape[0] * shape[1],
        )
    )


def build_game_schedule(
    games_total: int,
    hand_sizes: tuple[int, ...] = tuple(range(3, 11)),
    player_counts: tuple[int, ...] = (3, 4, 5),
    hand_size_tilt: float = 1.0,
    player_weights: Optional[tuple[float, ...]] = None,
    hand_size_weights: Optional[tuple[float, ...]] = None,
) -> tuple[GameScheduleCell, ...]:
    """An explicit deal quota per (players, hand size) cell.

    Every tree carries the same weight in the loss regardless of how large it
    grew (see ``tree_weighting="per_tree"``), so the share of trees a cell gets
    *is* its share of the gradient. That makes this function the place where
    the training mix is decided.

    ``hand_size_tilt`` is the exponent on hand size: 0 spreads deals evenly,
    1 gives a 10-card game 10/3 the deals of a 3-card game. Longer games are
    where bidding and card-play actually interact, so the default leans on
    them, but only linearly -- they also cost an order of magnitude more each,
    and starving the short games costs coverage of the endgame-only states that
    short deals reach cheaply.
    """

    if games_total < 1:
        raise ValueError("games_total must be >= 1.")
    if hand_size_weights is not None:
        if len(hand_size_weights) != len(hand_sizes):
            raise ValueError("hand_size_weights must match hand_sizes.")
        by_hand = dict(zip(hand_sizes, hand_size_weights))
    else:
        by_hand = {n: float(n) ** hand_size_tilt for n in hand_sizes}
    if player_weights is not None:
        if len(player_weights) != len(player_counts):
            raise ValueError("player_weights must match player_counts.")
        by_players = dict(zip(player_counts, player_weights))
    else:
        by_players = {p: 1.0 for p in player_counts}

    # Apportion hand sizes first, then split each one across player counts.
    # Doing it in one pass over the 24 cells rounds each cell independently and
    # the per-axis marginals drift by ~10% at realistic totals, which would
    # quietly mistune the very mix this function exists to set.
    per_hand = _apportion(by_hand, games_total)
    counts = {}
    # Split each hand size across player counts against a *cumulative* target
    # rather than independently. Apportioning each hand size on its own sends
    # every leftover deal to the same player count, so the player marginal ends
    # up several percent off while the hand-size marginal is exact.
    assigned = {players: 0 for players in player_counts}
    done = 0
    for hand_size in hand_sizes:
        games = per_hand[hand_size]
        done += games
        target = _apportion(by_players, done)
        share = {p: max(target[p] - assigned[p], 0) for p in player_counts}
        # Non-monotonic rounding can leave the row short or long by a deal.
        drift = games - sum(share.values())
        for players in sorted(player_counts, key=lambda p: -by_players[p]):
            if drift == 0:
                break
            step = 1 if drift > 0 else -1
            if share[players] + step >= 0:
                share[players] += step
                drift -= step
        for players in player_counts:
            assigned[players] += share[players]
            counts[(players, hand_size)] = share[players]
    return tuple(
        GameScheduleCell(hand_size=hand_size, num_players=players, games=games)
        # Widest shape first: it sets the cache high-water, so growing into it
        # once beats growing on every shape transition.
        for (players, hand_size), games in sorted(
            counts.items(), key=lambda item: -item[0][0] * item[0][1]
        )
        if games > 0
    )


@dataclass(frozen=True)
class SeqTrainingConfig:
    """Training configuration for the sequence pipeline."""

    # Game schedule: each cell yields a matched self/historical pair per game.
    schedule_cells: tuple[GameScheduleCell, ...] = field(
        default_factory=default_schedule_cells
    )
    player_counts: tuple[int, ...] = (3, 4, 5)
    player_count_weights: tuple[float, ...] = (2.0, 3.0, 4.0)

    branch_rule: BranchRuleConfig = field(default_factory=BranchRuleConfig)
    branch_budget: BranchBudgetConfig = field(default_factory=BranchBudgetConfig)
    rollout: RolloutOptions = field(default_factory=RolloutOptions)

    # Optimization.
    learning_rate: float = 2e-4
    adam_betas: tuple[float, float] = (0.9, 0.999)
    # Optimizer steps over which the learning rate ramps linearly to
    # learning_rate. Adam's first steps are sign steps (m_hat/sqrt(v_hat) is
    # +/-1 whatever the gradient scale, so gradient clipping cannot soften
    # them): measured on a cold start, one full-LR step moved branch KL to
    # 0.16 against a 0.005 cap, which under rollback means no update ever
    # survives. 0 disables.
    lr_warmup_updates: int = 100
    epochs: int = 1
    # Microbatch size in token positions (sequences x length), sized to memory.
    microbatch_positions: int = 16384
    max_grad_norm: float = 1.0

    # Policy objective. One loss over every focal decision: NeuRD, whose
    # gradient on the logit of action a is -A(a), independent of pi(a).
    #
    # Why not a policy gradient: the softmax PG gradient is pi(a)*A(a), so an
    # action the policy has drifted away from moves quadratically slowly in
    # policy space however large its advantage. In self-play that is the
    # central failure -- the opponent distribution moves, an action correctly
    # suppressed early becomes correct later, and the policy has no escape
    # velocity. Dropping the pi(a) prefactor gives replicator dynamics, and
    # softmax-of-accumulated-advantage is Hedge, a no-regret algorithm.
    policy_coef: float = 1.0
    neurd_regret_coef: float = 1.0
    neurd_kl_coef: float = 1.0
    policy_kl_cap: float = 0.005
    # Clamp on A(a) = Q(a) - V. Relative rewards reach ~+/-20 at five players
    # and the value baseline is untrained early, so one outlier row could
    # otherwise dominate the normalized weight sum. 0 disables.
    neurd_advantage_clip: float = 10.0

    # Correction for *which* actions got expanded. An action expanded with
    # probability q receives q*A(a) in expectation over updates, so a rule
    # that expands by sampling the policy silently reintroduces the pi(a)
    # prefactor NeuRD exists to remove. Weighting by q**-exponent restores
    # A(a):
    #   1.0  true NeuRD -- every legal action's logit moves by its own regret
    #   0.0  no correction, i.e. back to policy-gradient-shaped weighting
    # This is MCCFR's importance correction on sampled regret. Normally 1/q
    # is a variance disaster, but the uniform arm floors q at 1/|legal| and
    # |legal| <= 11 here, so the cap below is rarely reached.
    neurd_inclusion_exponent: float = 1.0
    neurd_inclusion_cap: float = 12.0

    # Loss weighting across games/trees.
    tree_weighting: Literal["per_tree", "per_row"] = "per_tree"
    # How much of a tree's weight follows its size. A tree's share of the loss
    # is proportional to (its rows) ** exponent:
    #   0.0  every tree counts the same, however large it grew (per_tree)
    #   1.0  every row counts the same, so a branched 10-card tree outweighs a
    #        3-card one by its row count (per_row)
    # Between the two, a 10-card tree with 30x the rows of a 3-card tree gets
    # 30**exponent times the weight -- 5.5x at 0.5. Setting "per_row" above
    # forces 1.0 and ignores this.
    #
    # The three exponents below are separate questions that all end up as row
    # weights, and it helps to keep them apart:
    #   tree_weight_exponent    how much weight follows a tree's *size*
    #   shape_importance_exponent  how much a *shape* matters in the objective
    #   branch_depth_exponent   where inside a tree the weight sits
    tree_weight_exponent: float = 0.0

    # Loss weight per shape, as (hand_size ** a) * (num_players ** b), applied
    # on top of the size weighting.
    #
    # Both default to 0 -- every deal equally important -- because that is what
    # the scoring says: a 3-card round pays the same points as a 10-card round,
    # so getting it right is worth the same. Note this is *not* the same as
    # doing nothing, since without it a shape's weight would drift with however
    # many rows its trees happened to produce. Raise these only to say a shape
    # matters more *as an objective*, not because it generates more rows.
    shape_importance_exponent: float = 0.0
    player_importance_exponent: float = 0.0

    # Weight of a branch row by its depth in the tree, as (1 + depth) **
    # exponent, renormalized so a tree's total weight is unchanged.
    #
    # 0.0 gives every branch node the same weight, which sounds neutral and is
    # not: branch nodes multiply with depth, so a 9-card tree has ~1 node at
    # the bid against ~100 at trick 5, and the bid -- the decision that sets
    # the whole round's target -- receives well under 1% of the tree's policy
    # gradient. Negative values pull weight back toward the early decisions;
    # -1.0 roughly equalizes weight across depths for a branching factor of 2.
    branch_depth_exponent: float = 0.0

    # Auxiliary losses. A coefficient of exactly 0 skips computing that loss,
    # and with owner, suit and trick all 0 the auxiliary heads are not run at
    # all -- the owner einsum + Sinkhorn is the expensive one.
    value_coef: float = 0.5
    trick_coef: float = 0.25
    owner_coef: float = 0.25
    owner_capacity_coef: float = 0.05
    owner_sinkhorn_iterations: int = 8
    # Train the owner head on a random fraction of labeled positions per
    # update (unbiased; caps the Sinkhorn autograd graph size).
    owner_position_fraction: float = 0.5
    suit_coef: float = 0.1
    # Entropy bonus over the full legal support at every decision position.
    # Defaults off: it exists to fight the entropy collapse of a pi(a)-scaled
    # policy gradient, and NeuRD does not have that failure mode -- mixing is
    # emergent, because a suppressed action's logit still moves by its full
    # regret. Raise it only if measured entropy actually collapses.
    entropy_coef: float = 0.0

    # League.
    league_max_snapshots: int = 8
    league_min_iteration: int = 0
    snapshot_every: int = 200

    # Cadence and bookkeeping.
    eval_every: int = 100
    checkpoint_every: int = 200
    device: Optional[str] = None
    precision: Literal["fp32", "bf16"] = "fp32"
    kv_dtype: Literal["fp32", "fp16"] = "fp32"
    # False re-encodes each leaf's full prefix every decision (no cache).
    # Kept as a knob so the cache's FLOPs-vs-memory-traffic tradeoff stays
    # measurable at any batch size.
    use_kv_cache: bool = True
    seed: int = 0

    def validate(self) -> None:
        self.rollout.validate()
        if len(self.player_counts) != len(self.player_count_weights):
            raise ValueError("player_count_weights must match player_counts.")
        if any(weight < 0 for weight in self.player_count_weights):
            raise ValueError("player_count_weights must be non-negative.")
        if sum(self.player_count_weights) <= 0:
            raise ValueError("player_count_weights must have positive mass.")
        if self.branch_rule.bid_top_k < 0:
            raise ValueError("bid_top_k must be >= 0.")
        budget = self.branch_budget
        if budget.branch_rate is not None and not 0.0 <= budget.branch_rate <= 1.0:
            raise ValueError("branch_rate must be in [0, 1].")
        for rule in budget.branch_rate_by_shape:
            if not 0.0 <= rule.rate <= 1.0:
                raise ValueError("branch_rate_by_shape rates must be in [0, 1].")
        # The rate is the only thing that decides branching now, so every shape
        # the schedule can produce must resolve to one. Catch a partial table
        # here rather than deep inside the wave loop.
        uncovered = [
            (players, cell.hand_size)
            for cell in self.schedule_cells
            for players in (
                (cell.num_players,)
                if cell.num_players is not None
                else self.player_counts
            )
            if budget.rate_for_shape(players, cell.hand_size) is None
        ]
        if uncovered:
            raise ValueError(
                "Every scheduled shape needs a branch_rate (set branch_rate or "
                "build_branch_rate_table); uncovered shapes: "
                f"{sorted(set(uncovered))}"
            )
        for cell in self.schedule_cells:
            if not 3 <= cell.hand_size <= 10:
                raise ValueError("Schedule hand sizes must be in 3..10.")
            if cell.num_players is not None and cell.num_players not in self.player_counts:
                raise ValueError("Schedule cell player count not in player_counts.")
