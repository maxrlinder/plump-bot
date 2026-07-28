# Plump Bot — schema-v6 autoregressive sequence pipeline

A self-play reinforcement-learning agent for **Plump** (Swedish trick-taking
card game), built around a causal transformer that reads the game as a token
stream and a KV-cached rollout engine that expands thousands of counterfactual
branches per deal at one forward pass per game event.

This document describes the **current** pipeline (`plump/seq`, schema v6). The
older schema-v1..v5 pipelines still live in the repository as evaluation and
legacy artifacts; see [Legacy pipelines](#legacy-pipelines) at the end.

---

## 1. The game

The engine in [plump/](plump/) is the source of truth for the rules. A *round*
is self-contained and is the unit this pipeline trains on.

**Deal.** `N` cards each to `P` players from a standard 52-card deck. The
remaining `52 - N·P` cards are the *kitty*: undealt, unseen, never played. Both
`N` and `P` matter to strategy, and the kitty is a real source of hidden
information (at 3 players × 3 cards, 43 of 52 cards are unaccounted for).

**Bidding.** Starting from the bidding-start player and going clockwise, each
player announces how many tricks they intend to take, `0..N`. The last bidder
may **not** choose the value that would make the total bids equal `N` — so at
least one player must miss. This "hook" rule is what makes bidding
interactive rather than a private estimate.

**Play.** The highest bidder leads first (ties broken by earliest position in
the bidding order). Players must **follow suit** if able; otherwise anything.
There is **no trump** (`TrumpPolicy.NONE`), so the highest card of the led suit
wins the trick, and the winner leads the next one.

**Scoring** (per round, `ScoringConfig`):

| outcome | points |
|---|---|
| bid `b > 0` and took exactly `b` | `10 + b` |
| bid `0` and took `0` | `5` |
| missed the bid | `0` |

Note what this implies: a 3-card round pays roughly the same as a 10-card
round. Long games cost an order of magnitude more compute but are not worth
more points — which is why the training schedule and loss weighting treat every
deal as equally important by default rather than letting compute decide.

**Deployment space.** Trained and evaluated over:

```
players: 3, 4, 5
cards:   3, 4, 5, 6, 7, 8, 9, 10
```

24 shapes. The engine supports more for tests and interactive play, but that
grid is the contract. Every checkpoint stores `rules_fingerprint()` — a hash of
deck, follow-suit, first-leader, hook rule, scoring and trump policy — and a
mismatched checkpoint is rejected rather than silently evaluated under
different rules.

## 2. RL formulation

**Environment.** [plump/env.py](plump/env.py) — `PlumpEnv`, one round per
episode, `clone()` for branching, an append-only `event_log` that is the
canonical replayable record of what happened.

**Observation.** Per-seat and observer-relative. A player sees their own dealt
hand plus the public event stream (bids, plays, trick winners). Every player
reference in a sequence is `(p - observer) mod P`, so absolute seat is
invisible to the model; the only positional signal is the focal's *bidding
position*, carried explicitly on the opening token.

**Actions.** Bids `0..10` (11 slots) and cards `0..51`. **Legality reaches the
model only through logit masking** — the legal set is applied as a `-inf` mask
at the head, never fed in as an input feature. Hand state, voids, tricks won,
and led suit are all derivable from the sequence, so there is no side-channel
context vector.

**Reward.** Terminal, per round, relative:

```
r_focal = score_focal − mean(score of every other player)
```

(`compute_relative_rewards` in [plump/training/common.py](plump/training/common.py)).
Zero-sum-shaped, so beating the field is what is rewarded, not accumulating raw
points.

**Opponents.** Self-play by default. A league of frozen historical snapshots
(`SeqLeague`) supplies a second *arm* per deal: the same deal, same focal seat,
same random tape, but opponents drawn from an older checkpoint. The two arms
run in the same wave loop as a matched pair (common random numbers), so their
reward difference is a low-variance comparison rather than two independent
samples.

## 3. Layout

```
plump/seq/
  config.py     schema constants, SeqModelConfig, SeqTrainingConfig,
                branch rules/budgets, schedule builders
  tokens.py     token vocabulary, per-seat sequence builder, replay labeller
  model.py      SeqPlumpModel: causal blocks, all heads, full/prefill/step
  kv.py         KVCache: preallocated pool, branch copy, free list
  rollout.py    SeqRolloutCollector: wave loop, branching, backups, CRN
  trainer.py    SeqTrainer: batch assembly, losses, KL guard, checkpoints
  policy.py     SeqModelPolicy (ActionPolicy adapter) + SeqLeague

examples/train_seq.py            training entrypoint
scripts/calibrate_seq_schedule.py    rollout-only timing over the shape grid
scripts/benchmark_seq_throughput.py  full collect+update cycle
scripts/report_seq_*.py              branch shape / rate grid / shape cost
tests/test_seq_*.py                  147 tests
```

## 4. Schema v6: the token stream

One sequence **per observer seat** per leaf. Layout for `(P, N)`:

```
[GAME] [HAND × N] [TURN] [BID × P] { [PLAY × P] [TRICK_WIN] } × N
```

Length: `1 + N + P + N·P` plus `N` if trick-win tokens are on, plus `P` if the
pre-bid TURN token is on. At the maximum shape (5 players, 10 cards) that is
**81 tokens** in the shipped configuration.

Each token is **12 integer slots**, each with its own vocabulary, summed into
one embedding plus a learned absolute position:

| slot | content |
|---|---|
| `type` | PAD / GAME / HAND / BID / PLAY / TRICK_WIN / TURN |
| `rel_player` | actor of this event, observer-relative |
| `rank`, `suit`, `card` | the card (HAND, PLAY); NA otherwise |
| `bid` | bid value (BID); NA otherwise |
| `trick`, `pos_in_trick` | where in the round this sits |
| `hand_size`, `num_players` | the shape, repeated on every token |
| `next_actor`, `next_phase` | who acts next and in which phase |

The GAME token also carries the observer's **bidding position** in the
`pos_in_trick` slot.

**Decision positions.** The hidden state `h_t` at the token whose `next_actor`
is the observer predicts the observer's action — ordinary next-token
prediction. The value and belief heads are read at *every* position, not only at
the focal's own decisions.

### The two schema flags

`SeqModelConfig.trick_win_token` (bool) and `turn_token` (`off`/`bid`/`all`)
change the token stream without touching the trunk. Settled configuration:
**`trick_win_token=True, turn_token="bid"`**.

**TRICK_WIN is formally redundant** — no trump, so the highest card of the led
suit wins, and the winner leads next, which the trick's last PLAY already
announces in `next_actor`. Dropping it measured ~10% off the update. It stays
because it is a compute step sitting exactly where trick state is revised, and
because it turns "count tricks won" into counting tokens of one type rather
than evaluating a two-slot conjunction.

**TURN is a pause/register token** (cf. Goyal et al. pause tokens, Darcet et
al. ViT registers). Deliberately near-empty — rank/suit/card/bid all stay NA —
so the only thing it contributes to the residual stream is "an action is due,
from this seat, at this point". The policy head then reads a state summary
rather than a state summary tangled up with whatever card happened to be played
last, and it buys `n_layers` of extra serial compute before acting. It carries
the actor's player embedding, so the round opener is *named* rather than
inferred.

Why before bids only, and not before every play:

> The sequence is the **whole game from one observer's view**, not that
> observer's own moves. At 5 players × 10 cards there are 55 actions in the
> game; the focal makes 11 of them and observes 44. `turn_token="all"` adds 55
> tokens, not 11 — `seq_len` 76 → 131.

Measured on the 96-deal rollout (3 steady-state runs each, 3.86M model):

| schema | 96-deal rollout | peak |
|---|---:|---:|
| TRICK_WIN only | 10.4s | 4.39 GB |
| **TRICK_WIN + TURN before bids (shipped)** | **10.8s** | **4.39 GB** |
| TRICK_WIN + TURN before every action | 13.4s | 10.8 GB |

The shipped schema is +4% wall clock at identical memory — effectively free.

**Why every seat gets the TURN token, not just the actor.** The cheap version
(a TURN before *each seat's own* actions — 11 tokens, not 55) was not built
because it desynchronises the cache rows. Each seat acts `N+1` times so
sequence *lengths* stay identical, but *positions* drift by ±1 within a trick
and re-align only at trick boundaries. That ±1 is fatal to the three scalars
the wave loop rests on: `KVCache.read(slots=None, length, count)` returns a
zero-copy view at one length for all rows, `write_range` writes at one scalar
start, and `embed(tokens, start)` offsets by one scalar. Ragged rows would need
per-row positional embeddings, a scatter write, and a key-padding mask in the
decode attention — ~200 lines in the hot loop, to save tokens the measurement
says are already cheap.

`tokens.token_layout()` is the single place that decides which token sits
where; both the rollout wave loop and the replay labeller walk it, so the flags
cannot make the two disagree.

## 5. The model

[plump/seq/model.py](plump/seq/model.py) — `SeqPlumpModel`, a pre-LN causal
decoder written out by hand (`nn.TransformerEncoder` cannot do cached decode).

- **Block**: LN → fused QKV projection → attention → residual → LN → GELU MLP →
  residual. One fused `qkv_proj` rather than three, because the rollout issues
  a forward per game event and kernel-launch count dominates at small batch.
- **Embedding**: all twelve slots share one table with per-slot id offsets, so
  a token costs one gather instead of twelve lookups and eleven adds.
- **GQA**: `n_kv_heads` is configurable; `2` is the operating point. It cuts KV
  bytes per row ~4× and measured ~1.8× throughput against full MHA.
- **Decode attention is hand-written** rather than delegated to SDPA. On MPS the
  fused kernel is markedly slower at query length 1, and its GQA path slower
  still — which would make grouped KV a memory saving paid for in wall time.
  The implementation merges the query head-group and time axes into a 4-D
  batched matmul against un-expanded K/V; broadcasting a group axis instead
  makes MPS materialise the expanded K/V (measured 3× slower, 2× memory).
  Softmax runs in fp32 so an fp16 cache costs no accuracy.
- **Prefill** (`start == 0`) does use SDPA's square causal kernel, which is
  exactly right at that length.

**Heads**, all read per-position on `h_t`:

| head | shape | role |
|---|---|---|
| `bid_head` | `[.., 11]` | bid policy, read at decision positions |
| `card_head` | `[.., 52]` | card policy, read at decision positions |
| `value_head` | `[.., 1]` | MLP; focal relative reward / backed value |
| `suit_presence_head` | `[.., P, 4]` | **belief**: does seat `p` still hold suit `s`? |
| `bid_hit_head` | `[.., P]` | **belief**: will seat `p` finish on its bid? |
| `trick_count_head` | `[.., P, 11]` | final tricks-won per seat, feasibility-masked. Optional, off by default |

Both belief heads emit **one logit per (relative seat, class)** and let the loss
mask seats a shape does not have. The alternative — a few logits plus a seat
embedding evaluated once per seat — turns one `d × 20` matmul into `P` passes
over `[B, L, d]` to distinguish at most five seats the relative index already
separates. Both beliefs cover **every seat including the observer's own**: own
suit presence is derivable from the prefix rather than a belief, but it costs one
column and makes "do I still hold a spade" explicitly supervised.

**Suit presence is linear and bid hit is an MLP**, and that asymmetry was
measured rather than assumed. Suit presence is an *existence* question, monotone
in an accumulated feature, so a linear readout suffices. Bid hit is `tricks_won
== bid` — a **bump**, not a threshold. On completed rounds, held out by deal, a
linear head learns `tricks_won >= k` essentially perfectly and `tricks_won == k`
barely at all:

| task at the final position | linear head | MLP head |
|---|---:|---:|
| `won >= 2` (threshold) | **+0.685** (solved) | +0.686 |
| `won == 2` (exact count, no bid) | +0.031 | +0.046 |
| `won == bid` (bid hit) | +0.005 | **+0.027** |

(gain in nats of held-out BCE against a base rate fitted per task). The trunk
counts fine — thresholds are solved. Equality needs two decision boundaries, so
`bid_hit_head` is `Linear → GELU → Linear`. **Check this shape question before
adding any new belief head.**

`forward_full(aux_heads=...)` takes a **set of head names**, so a head whose loss
coefficient is zero costs neither a forward nor a backward. Three entry points:
`forward_full` (all positions — training and eval), `forward_prefill` (encode a
fresh prefix into empty cache rows), and `forward_step` (append one token or a
run of tokens, read the last — the rollout hot path, lean heads only).

**Sizes:**

| dims | params |
|---|---:|
| `d_model 256, 6 layers, 8 heads, kv 2, ff 768` | 3,579,024 |
| `d_model 320, 8 layers, 10 heads, kv 2, ff 960` | 7,223,184 |
| `d_model 384, 6 layers, 12 heads, kv 2, ff 1024` (**default**) | 7,235,472 |

The 7.24M config is the `examples/train_seq.py` default: head_dim stays 32 at 6
layers so KV bytes/row are unchanged from the 3.58M model, and the rollout is
latency-bound, so doubling the parameters measured only +6% rollout time.

## 6. KV cache

[plump/seq/kv.py](plump/seq/kv.py) — one preallocated pool per *set of
weights* (the current policy, and each frozen opponent snapshot). Rows are
`(leaf, seat)` pairs: branching a leaf copies every seat's prefix, because all
`P` sequences share the same public history.

- Storage is **one tensor per layer**, not one stacked tensor — at these batch
  sizes a stacked cache exceeds the `INT_MAX` element limit MPSGraph imposes on
  a single tensor.
- `branch_copy` is one batched advanced-index copy per layer, chunked so the
  gather temporary stays under 256 MB (a wide layer's temporary otherwise
  rivals the pool itself and pushes the allocator into thrashing exactly when
  memory is tightest).
- The dense read path (`slots=None`) returns **zero-copy views** of rows
  `0..count-1`. This is the fast path the whole wave design exists to keep: the
  alternative, an advanced-index gather, is a copy.
- The pool **grows into the widest tree actually seen** per game shape rather
  than reserving the worst case, since nothing bounds a tree ahead of time.
  High-water marks survive across iterations, so only the first iteration pays
  cold-start growth. Pools are released between collection and the update — they
  are the largest live allocation in the process and pure dead weight during the
  backward pass.
- `poison=True` fills unwritten rows with NaN so a read-before-write raises
  instead of silently attending to zeros. Tests use it; production does not.

## 7. Rollout

[plump/seq/rollout.py](plump/seq/rollout.py) — `SeqRolloutCollector`. This is
the centre of the design.

### The wave loop

All leaves of a batch share `(players, hand_size)`, so games are the same
length and every leaf advances **one public event per wave**. That single fact
is what makes the whole thing cheap:

1. every cache row has the same length, so `position` is one scalar;
2. reads are dense and zero-copy;
3. writes are one `write_range` at one scalar offset;
4. rows never need permuting, because leaves only ever get *added*, and they
   all terminate together.

One wave:

- **Batched sample.** Masked softmax + inverse-CDF sampling for the whole wave
  in one vectorised numpy pass, off each leaf's own RNG tape.
- **Decide and step.** Per leaf: branch or not, clone envs for children, step
  the env. A *forced run-out* (every hand down to one card) is stepped straight
  to terminal with no forward passes at all.
- **Branch copy.** One batched parent→child prefix copy per policy per wave.
- **Append.** The realized event token — plus the TRICK_WIN closing a trick,
  plus the next actor's TURN token — is appended to **all P seats of every
  leaf** in one `forward_step` per policy. Only the last token of the run
  produces a readout; the earlier ones exist purely to advance the cache, which
  is why merging the run into one forward is worth it.

Deals of the same shape are batched into one wave loop (`deals_per_batch`, or
`auto_deals_per_batch` which sizes the batch from measured cache rows per deal
— a 3-player/3-card deal occupies ~30 rows and a 5-player/8-card deal ~3000, so
one global batch size cannot be right).

### Branching

**Which decisions branch** is decided per focal decision by `branch_rate`.
There used to be a leaf budget interpreted as a floor — branch every layer
until the limit, then never again — and it is gone. It spent everything on the
bid and opening tricks and left the endgame, where counterfactuals are most
decidable, wholly unbranched. The KV row cap is now the only hard limit, and a
run that hits it (`blocked_by_cache > 0`) is a run whose rate is too high for
its budget.

**Rates are per shape, not global.** A path through an `N`-card game has `N-1`
branchable decisions and each branch point multiplies the path, so a tree is
about `b^(rate·(N-1))` leaves — the rate compounds over *game length*. One flat
rate is not one amount of branching: at 0.5 a 10-card tree is ~2^4.5 the size
of a 3-card one, so long games take the whole budget while short ones stay
nearly unbranched, even though a 3-card round pays the same points.
`build_branch_rate_table` therefore uses two regimes: **exhaustive (rate 1.0)
up to 7 cards**, where wall time is set by the number of waves rather than by
tree size and branching is nearly free, then a geometric taper to
`reference_rate` at 10 cards.

**Candidate sets** (`BranchRuleConfig`):

- *Bidding*: top `bid_top_k − 1` bids by policy probability, plus the
  unconditional sample. The focal's own bid is **always** expanded — it is the
  root of the tree and sets the round's target, never left to a coin flip.
- *Play*: `all_legal`, `top_k`, `top_k_plus_random`, `sample_k`,
  `sample_k_plus_uniform`, or **`gumbel_top_k`** (the operating point, `k=3`).
- `gumbel_top_k` returns `k` **distinct** actions: the realized one, then a
  Gumbel top-`k−1` over the rest. `argtop(log π + Gumbel)` is exactly sampling
  without replacement ([Kool et al., ICML 2019](https://arxiv.org/abs/1903.06059)),
  so the extra arms still follow the policy but can never repeat. Duplicates
  are pure waste in a branching tree — a second copy of an action costs a whole
  subtree and buys no counterfactual — so this dominates `sample_k` at equal
  `k`. Backup weights are Hájek (self-normalized Horvitz–Thompson), `π(a)/q(a)`
  renormalized: without replacement the empirical-frequency weighting `sample_k`
  uses is meaningless, since every candidate appears exactly once.
- Inclusion probabilities `q(a)` are recorded per candidate, because only the
  rule knows how it drew. `gumbel_top_k` gets them exactly from the
  `(k+1)`-th largest perturbed value κ: conditional on κ, selection events are
  independent Bernoullis with `q(a) = 1 − exp(−exp(log π(a) − κ))`. These feed
  the `1/q` weight in the update (§8) — without them, drawing candidates from
  the policy silently reintroduces the `π(a)` prefactor NeuRD exists to remove.

Stage-dependent overrides (`StageBranchRule`) let the rule differ by trick
index without code changes.

### Backups

Terminal reward propagates up through `SeqBranchRecord.resolve`, which fires
when the last child of a node reports in:

- **Full candidate mass** (the candidate set covers all legal actions, or the
  weights are Monte-Carlo frequencies summing to 1):
  `V = Σ π(a)·Q(a)` — exact.
- **Capped set** (top-k bids): the control-variate estimator — exact mass over
  the deterministic top set, plus a one-sample estimate of its complement.
  Unbiased for the full-policy value.

### Common random numbers

Matched arms share a random tape (`crn_state`), and a branching child inherits
its parent's RNG state at the split, so the counterfactual differs from the
realized line *only* in the action taken — not in the subsequent dice. Branch
placement itself is drawn off the same tape, so a bid-split pass makes the same
choices an unsplit run would.

### Ownership: every position trained exactly once

Each leaf owns the suffix after its last branch point (`owned_from`); the spine
child owns the shared prefix. `h_t` at a branch position is identical across
children, so any single owner is valid, and this keeps the "one update row per
decision" invariant. It is implemented as a per-position loss mask, and it is
also why `build_replay_arrays` takes `label_from` — a leaf that owns only a
short suffix skips the expensive label work for everything above it.

### Bid splitting

`bid_split_groups` partitions one deal's tree across N wave loops by the
focal's root bid candidates. The shared prefix is replayed per pass and the
cache is freed between them, so peak memory scales down roughly by that factor.
The bid branch record is created once and each pass resolves a disjoint subset
of its children, so the backup is identical to an unsplit tree's.

## 8. The update

[plump/seq/trainer.py](plump/seq/trainer.py). Collected trees are replayed into
full causal forwards grouped by `(players, hand_size)`, in
`microbatch_positions`-sized chunks with gradient accumulation.

### How the batching actually works

There are no waves here — waves are a rollout concept. The rounds are complete,
so each finished sequence is re-encoded in one causal forward.

The unit of work is a **leaf**, not a game: every leaf of every tree is one row
holding the full token sequence for that round from the focal seat's view. Rows
group by **shape**, because a rectangular `[B, L, 12]` tensor needs uniform `L`.
Each group is then chunked by a *position* budget (`microbatch_positions`,
16384), so rows per microbatch fall as sequences lengthen. One measured
iteration (96 deals, 35,578 leaves):

| | rows | `L` | rows/microbatch | microbatches |
|---|---:|---:|---:|---:|
| 3p × 3c | 46 | 22 | 744 | 1 |
| 4p × 9c | 3,803 | 63 | 260 | 15 |
| 5p × 10c | 3,879 | 81 | 202 | 20 |

**24 groups → 144 microbatches → 144 forward+backward pairs → gradients
accumulate across all of them → one optimizer step.** `epochs` (default 1) is
the only knob that makes it more than one step.

Note the cost of the ownership rule: of the 2,098,340 positions the forward
computes, only **565,101 (27%) carry loss**. Each leaf owns only the suffix after
its last branch point, but every sibling still has to *run* the shared prefix —
you cannot get `h_t` at position 60 without computing 0–59. The rollout avoids
exactly this redundancy via `branch_copy`; the update throws it away and
re-encodes each row independently. Fixing that means attention over the tree
rather than over independent sequences, so the branch rate is effectively a
redundant-prefix multiplier on update cost.

Nothing is stored as tensors during rollout: a leaf keeps its env's event log
plus per-decision records, and `build_replay_arrays` re-derives tokens and
*all* per-position labels at update time by replaying the log against the true
deal. Correctness over rollout-time bookkeeping — the labeller is the one place
that can get a position wrong, and it is exercised by
`test_rollout_probs_match_full_forward_replay`, parametrised over every schema
flag combination. That test is the correctness anchor for the entire cached
design: it asserts that what the cached rollout computed equals what a full
forward replay computes.

**The policy loss.** One objective over every focal decision: **NeuRD**.

```
L = -Σ_a w_a · sg[A(a)] · y_a          =>   dL/dy_a = -w_a · A(a)
A(a) = Q(a) - V
```

*Why not a policy gradient.* The softmax PG gradient on logit `a` is
`π(a)·A(a)`. Pushed through to policy space that is quadratic in `π`, so an
action the policy has drifted away from moves quadratically slowly however
large its advantage. In a stationary MDP that is benign. In self-play it is the
central failure mode: the opponent distribution moves, an action correctly
suppressed at iteration 300 becomes correct at iteration 900, and the policy
has no escape velocity. Dropping the `π(a)` prefactor gives replicator
dynamics, and softmax-of-accumulated-advantage is Hedge — a no-regret
algorithm, which is CFR's local ingredient with a network in place of the
regret table.

*Why one row type.* The rollout is an external-sampling tree: the focal's
actions are enumerated or drawn by an explicit rule, while chance and the
opponents are sampled. On that data an importance ratio corrects nothing —
ratios exist to repair sampling *from the policy*, and the candidate set was
not drawn that way. NeuRD has no expectation over actions in it at all; the
loss is a per-action statement, true for each `a` independently of which
siblings were expanded. So an arbitrary candidate set is valid, provided

1. `Q(a)` is unbiased for that action's continuation value,
2. `V` is the **full-policy** value, not a mean over the candidates, and
3. set membership does not depend on the noise in `Q(a)`.

A branched decision supplies `Q(a)` per candidate and `V` from the backup
(exact at full candidate mass, control-variate for a capped set). An unbranched
one is the `k=1` case: `Q` is the realised return and `V` is the value head's
estimate. Same loss, same rows — there is no branch/spine mix to balance and no
second objective to normalise against.

Two things the implementation must *not* do, both of which fail silently:
centering `A` over the candidate set, or dividing the row loss by the candidate
count. Either makes a `k=1` row exactly zero — roughly half the decisions in a
typical iteration, contributing nothing with no error raised. Centering is also
wrong on principle for a capped set: it is a null direction only when the set
is every legal action, and otherwise cancels the one real signal about the set
as a whole. `test_unbranched_rows_produce_a_nonzero_policy_gradient` pins this.

**The `1/q` correction.** `w_a = q(a)^-neurd_inclusion_exponent`, where `q(a)`
is the probability the branch rule would have expanded that action — recorded
at collection time in `SeqBranchRecord.inclusion_probs`, since only the rule
knows how it drew.

This matters more than it looks. If candidates are drawn by sampling the
policy, then `E[gradient on a] = q(a)·A(a) ≈ k·π(a)·A(a)` — the `π(a)`
prefactor NeuRD exists to remove, walking straight back in through the
candidate sampler. You would have the NeuRD loss, the guard, the KL anchor, and
none of the escape velocity. It is MCCFR's importance correction on sampled
regret, and normally `1/q` is a variance disaster; here the Gumbel/uniform arms
floor `q` and `|legal| ≤ 11`, so `neurd_inclusion_cap` (12.0) is rarely
reached. Exponent 1.0 is true NeuRD, 0.0 is back to PG-shaped weighting.

**Guard.** A full-support `KL(old‖new)` anchor per row, plus a per-epoch
rollback if weighted KL exceeds `policy_kl_cap`. The guard now covers *every*
policy row; under the old split it measured branch rows only, leaving the spine
half of the decisions outside it.

**Auxiliary losses.** Value (smooth-L1, every owned position — terminal reward
on the spine tail, backed value at resolved branch positions), plus **two
beliefs**, both sigmoid, both per seat, both including the observer's own, both
read at every owned position rather than only at focal decisions:

| loss | label | shape of the label |
|---|---|---|
| **suit presence** (`--suit-coef`, 0.25) | does seat `p` still hold suit `s`? | changes through the round, so rebuilt at every position of the replay |
| **bid hit** (`--bid-hit-coef`, 0.25) | will seat `p` finish on its bid? | an outcome, so one constant per seat broadcast over every position |

Bid hit is labelled from position 0, including positions before a seat has bid —
there the question is "will this seat hit whatever it ends up bidding", which is
a genuine forecast rather than leakage.

Trick count (`--trick-coef`) is retained as an option but **defaults to 0**: it
is the same question as bid hit in a wider, harder form. A coefficient of exactly
0 skips both the loss and the head's forward.

**The per-card ownership head is deleted**, not merely disabled — head, labels,
Sinkhorn loss, config, CLI flag and metrics column. It was by far the most
expensive thing in the auxiliary forward (17s of a 66s update, against 1.6s for
both beliefs combined) and nothing reads it. Removing it also dropped the replay
bookkeeping that existed only to feed it: the per-card void table, per-seat
played counts, suit-of-card index and kitty capacities, all of which ran on every
PLAY event regardless of whether owner labels were gated on. The v4 pipeline's
owner head in `plump/training` and `plump/modeling` is untouched.

**Caveat on bid hit.** It is conceptually valid and correctly wired, but it is
the weaker of the two. Suit presence generalises strongly on held-out deals
(BCE 0.668 → 0.369, strongest mid-to-late round); bid hit reached only +0.027
against its base rate on a fixed 4.7k-deal probe. Part of that is real
difficulty, part is that a constant-per-round label repeated across ~57 positions
is highly memorisable on a *fixed* set — a probe artifact, not the training
regime, which is `epochs=1` on a fresh deal stream. Judge `loss_bid_hit` against
its base-rate floor (≈0.41 at 4p × 8c), never against zero.

The entropy bonus defaults to **0**. It exists to fight the entropy collapse of
a `π(a)`-scaled policy gradient, and NeuRD does not have that failure mode —
mixing is emergent, because a suppressed action's logit still moves by its full
regret. Raise `--entropy-coef` only if measured entropy actually collapses.
`neurd_advantage_clip` (10.0) bounds `A(a)`; relative rewards reach ~±20 at five
players and the value baseline is untrained early, so one outlier row could
otherwise dominate the normalised weight sum.

**Loss weighting.** Four exponents that all end as row weights, deliberately
kept separate because they answer different questions:

| knob | question |
|---|---|
| `tree_weight_exponent` | how much weight follows a tree's *size* (0 = every tree equal, 1 = every row equal) |
| `shape_importance_exponent`, `player_importance_exponent` | how much a *shape* matters as an objective |
| `branch_depth_exponent` | where *inside* a tree the weight sits |

Defaults are 0 everywhere, i.e. every deal equally important — which is what
the scoring says, and is *not* the same as doing nothing, since without it a
shape's weight would drift with however many rows its trees happened to
produce. `branch_depth_exponent=0` sounds neutral and is not: branch nodes
multiply with depth, so a 9-card tree has ~1 node at the bid against ~100 at
trick 5, and the bid — the decision that sets the whole round's target —
receives well under 1% of the tree's policy gradient. Negative values pull
weight back toward early decisions. It applies to every decision, branched or
not, normalized per tree so the tree's total weight is untouched.

**KL guard.** Per epoch: snapshot weights *and* Adam moments, step, recompute
weighted KL over every policy row, and roll the whole epoch back if it exceeds
`policy_kl_cap`.

**LR warmup** (`lr_warmup_updates`, default 100 kept steps) exists because the
guard and Adam interact badly on a cold start: Adam's first steps are sign
steps — `m̂/√v̂` is ±1 whatever the gradient magnitude, so gradient clipping
cannot soften them — and one full-LR step on a fresh model measured KL
0.16–0.22 against the 0.005 cap. Under rollback that means *no update ever
survives*. The ramp counts kept steps only, so a rolled-back epoch retries at
the same scale on fresh data.

**Checkpoints** carry `schema_version=6`, model and optimizer state, both
configs, `rules_fingerprint()`, and league references. `SeqModelPolicy.from_checkpoint`
loads one for evaluation.

## 9. Measured performance

All on an M-series Mac (24 GB unified memory, MPS), the 96-deal
position-balanced schedule: one deal per (player count, hand size, bidding
position), `gumbel_top_k` k=3, exhaustive to 7 cards tapering to 0.5
at 10, `n_kv_heads=2`, 65536-row cap.

> **The schedule is 24 cells, not 96.** `build_position_balanced_schedule()`
> returns one cell per (players, cards) and each cell carries `games = players`,
> giving 96 deals. Replicating it out to 96 *cells* silently produces 384 deals,
> ~170k leaves and a ~305s iteration — a 4× oversized benchmark.

### Current full cycle

3 consecutive iterations, `gumbel_top_k` k=4, rate 0.5 at 10 cards, 3.58M model,
fp32 KV, objective = NeuRD + value + suit presence + bid hit:

| leaves | total | collect | update | of which label build | decisions/s |
|---:|---:|---:|---:|---:|---:|
| 35,578 | 66.4s | 23.4s | **43.0s** | 3.2s | 691 |
| 37,607 | 68.6s | 23.1s | **45.5s** | 3.6s | 692 |
| 44,851 | 76.7s | 24.8s | **51.9s** | 3.9s | 706 |

**~47s update, ~70s iteration, ~700 decisions/s** (a *decision* is a policy row:
one focal choice the update trains on, branched or not). Collect is 34% and the
update 66%; inside the update the label and token build is only 3–4s and the rest
is the backward. decisions/s is flat across a 26% swing in leaf count, so cost is
linear in leaves at this scale — no batching cliff.

The numbers in the rest of this section predate both the optimisation work in
§10 and the deletion of the owner head, and are kept for the *relative*
comparisons they make (schema flags, model scale, deals vs depth), not as current
absolute costs. Where they say "3.86M model" or "7.85M parameters" they mean the
same dims listed in §5, which are 3.58M and 7.24M now that the owner projection
and its card embedding are gone.

**Rollout only** (`calibrate_seq_schedule.py`), 3.86M model, fp32 KV:

```
warmup   ~15-18s   (KV pool growing into the widest tree)
steady    10.8s    96 trees, ~13k leaves, ~220k positions, peak 4.4 GB
```

Very repeatable: 11.0 / 11.0 / 10.5 across three steady-state runs. Zero cache
blocking.

**Where the time goes**, by hand size (share of rollout wall time vs share of
positions):

| cards | sec % | positions % |
|---:|---:|---:|
| 3 | 2.2% | 1.0% |
| 5 | 3.9% | 2.8% |
| 7 | 16.3% | 16.9% |
| 8 | 17.7% | 18.0% |
| 10 | 30.5% | 31.0% |

Time tracks positions almost exactly — the schedule is not wasting wall clock
anywhere in particular.

**Model scale is nearly free in rollout.** The same 96-deal rollout at 7.85M
parameters (`384/6/12/kv2/1024`) took **11.4s** against 10.8s at 3.86M: +6%
wall time (+9% per position) for **2.03× the parameters**, at identical peak
memory. The rollout is latency- and memory-traffic-bound, not compute-bound.

**Full cycle** (`benchmark_seq_throughput.py`), 7.65M model, fp16 KV:

```
collect 13.6s + update 35.3s = 48.9s/iteration, 3.5 GB peak, ~190k positions
```

The **update is ~72% of the cycle** at a flat ~4,800 positions/s regardless of
batch size — compute-bound and scaling linearly. Rollout micro-optimisation has
little headroom left; the auxiliary-head cost in the update is where the next
real gains are.

**Buy more data with more deals, not deeper trees.** Medians over 2 cycles
after warmup:

| config | deals | collect | update | total | positions | peak |
|---|---:|---:|---:|---:|---:|---:|
| ref 0.5, repeats 1 | 96 | 13.6s | 35.3s | **48.9s** | ~190k | 3.5 GB |
| ref 0.6, repeats 1 | 96 | 17.7s | 53.2s | **70.8s** | ~274k | 7.8 GB |
| ref 0.5, repeats 2 | 192 | 22.2s | 70.8s | **93.0s** | ~390k | 6.6 GB |

More deals costs 0.238 ms/position against 0.258 for deeper trees, on *less*
peak memory, with twice the deal diversity, and the update runs faster per
position on the bigger batch.

**Memory note.** Keep rollout peak under ~12 GB. Not a hard wall — it is the
point past which benchmark numbers stop meaning anything. At ~12 GB a
collection ran in 16.3s; pushing the same config to 18.3 GB made the KV
branch-copy stage jump from 2.0s to 21.3s for 2× the work, and chunking the
gather buffer did not help, so it is system-level memory pressure rather than
an allocation-size problem. Levers in the order that pays best: GQA, then fp16
KV, then `bid_split_groups`, then fewer deals per batch.

## 10. Optimizations

The architectural optimisations live where they belong — GQA and hand-written
decode attention in §5, the cache design in §6, the wave loop in §7. This
section is about the **host-side** work, which turned out to matter more than any
of them: the cycle went **144s → ~70s** on the same schedule with byte-identical
output, and almost none of that came from touching the model.

### The four Python traps

All four share one shape: **per-item work that looks free.**

- **`PlumpEnv.step()` built a full `Observation` on every call** — voids over
  every player and suit, played-card dicts, legal actions, a copy of the event
  log — which the wave loop never reads, since it goes straight to `event_log`
  and `is_done()`. ~20% of collect. Fixed with `emit_observations = False`, set
  in `_new_deal` and carried through `clone()`.
- **`_policy_terms` padded its candidate tensors with a Python loop of
  device-side slice assignments** — four MPS kernel launches per policy row,
  ~200k tiny dispatches per update. Padding on the host in numpy and shipping one
  transfer per field took it **54.6s → 11.0s**. This was the single largest cost
  in the entire pipeline and it was invisible as any one line.
- **`random.Random()` + `setstate`** draws 32 bytes of OS entropy and runs the
  key schedule (~15 µs) purely for `setstate` to discard it. `Random.__new__` +
  `setstate` is ~1.5 µs with a provably identical stream. Hit once per branch
  child and once per `env.clone()`.
- **`torch.tensor(python_list, device=...)`** walks the list elementwise; going
  via numpy is ~1.5× faster.

### Not recomputing what a sibling already built

A branch child's tokens are identical to its parent's up to the branch — the env
was cloned there. `SeqLeaf.parent` plus `build_seat_tokens(token_prefix=...)`
copies that prefix instead of replaying the events. Ordering the pass by
`owned_from` is a free topological order, since a parent always branched strictly
earlier than its child. Token build **1.90s → 0.86s** over 37k leaves.

Two things this must not disturb: the *policy row* order stays the entries order,
because reordering changes float reduction order; and the labels are **not**
shareable the way tokens are, since they need the replay state walked event by
event to reach `owned_from`.

### Skipping labels nothing will read

`build_replay_arrays` gates each label group on its loss coefficient. Arrays are
still returned at full shape, left at `IGNORE_LABEL`/`False`, so a loss enabled
against labels that were not built sees *no* labelled positions rather than wrong
ones. Together with the auxiliary-head selection in §5, an unused belief now
costs nothing on either the host or the device.

### How this was verified, and how it should be next time

**Bit-exact fingerprints.** A SHA-256 over every tree, leaf, event log, decision,
probability array, backed value and inclusion probability (collect), and over
every `SeqUpdateStats` field plus all final weights (update) — run on **CPU**,
where reductions are bitwise reproducible. Exclude `update_sec` / `build_sec`:
they are wall times and will always differ.

Three traps in the measurement itself, each of which produced a wrong conclusion
before being caught:

- **`git stash` is a no-op once the work is committed**, so the "baseline" run
  silently re-measured the optimised code. Use `git checkout <base> -- <files>`.
- **cProfile lies about this code path.** It reported the token build at 7.15s
  against a true **1.90s**: the path makes ~9M tiny calls and the profiler adds
  ~1 µs to each. An optimisation sized off that number was predicted at ~5s and
  delivered ~1s. cProfile is fine for finding *which* function, never for how
  much. Time with `perf_counter` around the real thing.
- **The machine throttles**, so sequential A-then-B flatters whichever ran first.
  Interleave A/B/A/B. (A closed lid mid-run is worse still — one iteration
  measured 97.7s against a true 68.6s.)

### On CPU cores

Python 3.12 with the GIL on, so threads cannot parallelise any of this. The
rollout's remaining CPU sits in a strict `forward → sample → step → forward`
chain that cannot overlap anyway. The one genuinely parallel load,
`build_replay_arrays`, is now ~4s — too small to justify a process pool that
would have to move gigabytes of label arrays back through shared memory. Worth
revisiting only if the label set grows or leaf counts rise substantially.

## 11. Running it

Use `.venv/bin/python -m <tool>` rather than `uv run` (stale shebangs in the
venv).

**Tests** — 147 seq tests, 351 in the full suite:

```bash
.venv/bin/python -m pytest tests/test_seq_*.py
.venv/bin/python -m pytest tests/
```

**Rollout timing over the whole shape grid** (collect only — prefer this when
only rollout cost is in question, since the update otherwise drowns out the
difference):

```bash
.venv/bin/python -u scripts/calibrate_seq_schedule.py \
  --schedule balanced --repeats 1 --reference-rate 0.5 --exhaustive-until 7 \
  --max-cache-rows 65536 --play-mode gumbel_top_k --play-top-k 3 \
  --kv-dtype fp32 --iterations 4 --turn-token bid
```

**Full collect + update cycle:**

```bash
.venv/bin/python -u scripts/benchmark_seq_throughput.py \
  --balanced --repeats 1 --reference-rate 0.5 --max-cache-rows 65536 \
  --auto-deals --play-mode gumbel_top_k --play-top-k 3 \
  --d-model 320 --n-layers 8 --n-heads 10 --n-kv-heads 2 --d-ff 960 \
  --kv-dtype fp16 --cache-budget-gb 10 --historical-arm off
```

> Gotcha: this script defaults to `--play-mode all_legal`, which is **not** the
> operating point and produces trees 4–5× larger (95s collect instead of 12s).
> Always pass `--play-mode gumbel_top_k --play-top-k 3`.
> `calibrate_seq_schedule.py` already defaults to it.

**Training:**

```bash
PYTHONPATH=. .venv/bin/python examples/train_seq.py \
  --iterations 2000 --reference-rate 0.5 \
  --checkpoint-dir checkpoints/seq_v6_run1 --log-dir logs/seq_v6_run1
```

Writes `metrics.csv` (per-iteration losses, bid-hit rate, entropy, branch KL,
rollback flag, leaf/position counts, wall-time split) and `config.json` to the
log directory, checkpoints and league snapshots to the checkpoint directory.

**Reporting:** `report_seq_branch_shape.py` (tree shape per game shape),
`report_seq_rate_grid.py` (cost vs branch rate), `report_seq_shape_cost.py`,
`sweep_seq_rollout.py`, `benchmark_seq_rollout_scaling.py`.

## 12. Status

Built and tested: token schema, model, KV cache, wave-loop rollout with
branching and exact backups, the trainer with NeuRD plus the value and belief
losses and the KL guard, checkpoints, league, policy adapter, and the full
benchmark/reporting toolchain. Rollout throughput and the full-cycle cost are
measured and documented above, and the cycle has been optimised to ~70s with
bit-identical output at every step.

**Not yet done** — this document will grow here:

- No real training run has been launched. All numbers above are throughput, not
  learning.
- Whether bid hit earns its place. It is correctly wired and cheap, but on a
  fixed-deal probe it barely separates from its base rate (§8). The streaming
  regime should be much kinder to it; that is an empirical question a real run
  answers.
- Quality validation of `n_kv_heads=2` against full MHA via `evaluate_policy`,
  in **both** matchup directions (directed matchups favour the lone focal seat
  by ~0.1 points/round, so a one-directional comparison is not evidence).
- Whether to adopt the 7.24M dims as the default, and whether
  `branch_depth_exponent=-0.5` is worth it.
- Sharing branch prefixes in the *update* the way the rollout already does
  (§8): 73% of forwarded positions carry no loss.
- Modal / L40S scale-up (the older v9 pipeline already trains there).

## Legacy pipelines

Schema v1–v5 remain in the repository as evaluation and legacy artifacts and
are untouched by this work: `plump/modeling/legacy_*_v1.py` (v1 checkpoints,
evaluation only), the schema-v4 PPO pipeline with counterfactual root search
(`examples/train_ppo.py`), and the schema-v5 expert-iteration pipeline
(`scripts/run_training_v5_50m_seed1.zsh`). V1–v4 checkpoints can act as frozen
opponents or initialise v4 weights but cannot resume a v5 optimizer, and
**none** of them can initialise or resume schema v6 — it is a fresh
architecture from random init.
