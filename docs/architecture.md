# Schema-v6 Architecture

## Environment

Training covers 3–5 players and 3–10 cards. A round has bidding, must-follow
suit play, no trump, and terminal relative reward:

```text
score_focal - mean(score_opponents)
```

The engine's append-only event log is the replay source for both observable
tokens and training-only labels. `rules_fingerprint()` is stored in every
checkpoint and prevents loading weights under different game rules.

## Token stream

Each observer receives a causal sequence:

```text
[GAME] [HAND × N] { [TURN] [BID] } × P
{ [PLAY × P] [TRICK_WIN] } × N
```

Every token contains twelve checkpoint-stable core categorical slots: type,
relative player, rank, suit, exact card, bid, trick, position in trick, hand
size, player count, next actor, and next phase. Their embeddings remain fused
in the original lookup table and are summed with an absolute position
embedding. Four additional causal bidding-state slots encode bids made, bid
sum, `hand_size - bid_sum`, and bids remaining. They are populated on GAME,
HAND, bid-TURN, and BID rows; BID rows contain the state *after* that public
bid, while the preceding TURN contains the state before it. Play-phase rows
use NA. A separate table embeds these four fields.

The fixed-width row also has nine remaining-hand card-id positions. They are NA
except on `TRICK_WIN`, where they contain the observer's still-held cards after
that trick. Nine is sufficient because the first trick has consumed one card
before any `TRICK_WIN` row exists. The model adds the sum of those cards'
existing exact-card + rank + suit input directions to the winner token. This
is observer-visible state, does not expose opponents' hands, adds no model
parameters, and gives the between-trick state update a direct representation
of the hand that remains.

The final fixed-width slot identifies the complete pattern of absent base
fields on the token. Individual NA sentinel rows are not gathered. Instead,
one learned row per valid absence pattern supplies their combined contribution
to the ordinary input, PLE, and attention-value streams. When loading an older
checkpoint, each pattern row is initialized to the sum of the old sentinel
rows it replaces. The migration is therefore algebraically equivalent at its
boundary while avoiding repeatedly accumulating gradients into shared NA
rows. Floating-point reduction order can produce tiny numerical differences;
it is not promised to be byte-identical.

Only observable information enters the sequence. Suit-presence, rank-boundary,
final bid-hit, trick-count, and value labels are reconstructed from the true
completed deal for training and never become model inputs.

The belief heads do not all use the same seat axis, and the split is
deliberate. Suit presence and rank boundaries cover **opponents only**: the
observer's own dealt
hand and every card it has played are already in its token stream verbatim, so
a column for its own cards would supervise an identity rather than a belief.
For every opponent and suit, the rank-boundary head has two sigmoid logits:
whether that opponent currently holds a card strictly below the observer's
lowest remaining rank and strictly above its highest. A final redundant row
ORs each bit over any active opponent. If the observer no longer holds that
suit, every corresponding target is masked because its bounds are undefined.
Outcome beliefs — trick count, and bid hit when enabled — keep **every seat
including the observer's own**, because how many tricks the observer ends up
taking depends on how the rest of the round plays out and is not readable off
its hand.

## Model and rollout

`SeqPlumpModel` is a causal transformer with grouped-query attention. The MPS
PPO preset uses a 384-wide, six-layer model with twelve query heads, two KV
heads, and a 1152-wide feed-forward block. With the conditioning extensions
below it has 8.51M actor parameters.

The actor has a packed 128-dimensional per-layer embedding (PLE) stream. One
lookup covers every present categorical slot, including the bidding-state
fields, and a compact pattern lookup replaces all absent-field rows; the
remaining-hand cards reuse the exact-card + rank + suit rows. Absolute position
has a second packed table. A packed projection of the ordinary input embedding
supplies the context-aware term. The result is reshaped into one slice per
transformer block, deterministically expanded to model width, and added after
that block's attention and feed-forward residuals. Preparing the packed tensor
once is important on cached MPS rollout: it avoids a separate small projection
at every block.

Alternating blocks (1, 3, and 5 with zero-based indexing) also receive a
separate packed token/position value-embedding slice. It is reshaped to the
two-KV-head value width and added directly to attention V before the KV cache
write. This is token-identity capacity in the value stream, not ResFormer's
reuse of a computed first-layer V.

The capacity-extension lookup tables and context projection start at zero.
The NA-pattern tables are the exception: on migration they start at the summed
learned contribution of the omitted sentinel rows. The direct residual and V
additions give newly zero-initialized rows nonzero gradients on the first
backward.
The bid-pressure, PLE, and value parameters are core actor parameters and are
covered by the same PPO KL acceptance rule. The oracle owns independent copies
trained by critic losses. Its player slot denotes absolute ownership rather
than the actor's observer-relative seat, and its PLE/value position tables are
extended to the longer all-hands sequence; none of these parameters are tied
between actor and oracle.

The rollout collector advances every live branch in event waves. KV prefixes
are copied when a focal decision branches, so siblings only decode their new
suffixes. Opponent turns and self/historical arms are batched by model.

A wave loop is lockstep: every leaf in it shares one token position, so it can
only hold deals that agree on table size and hand size. A 24-cell schedule
therefore decodes 24 narrow streams in sequence. `rollout.merged_waves` runs
the whole schedule as one continuously-batched loop instead. Each row carries
its own cache offset, attends only to its own prefix, and appends however many
tokens its own game produced this wave, with shorter runs padded to the batch's
longest; the padding lands on positions the row has not reached, which it
overwrites later and never attends to in the meantime. Games are dealt into the
pool longest-lived first, so retirement clears rows from the tail and the live
set stays a dense prefix that the attention read can slice zero-copy — the
collector still compacts if a shape ever retires out of turn, because the dense
read is a correctness invariant rather than an optimization. Merging applies
only to the unbranched single-policy PPO path; branching, bid splits,
heuristic/historical arms and a multi-policy pool each break the dense-row
invariant, and `merged_waves_available` falls back to the lockstep loop rather
than failing.

Under `policy_objective="ppo"`, branching and bid splitting are disabled and
each deal has one sampled leaf. Self-play can record every seat's decisions;
the same public token positions remain rectangular across their observer
caches. `ppo_trainable_policies` can assign those seats one shared actor or
several independent actor weight sets.

PPO uses a separate oracle critic transformer. After collection, each
environment game becomes one canonical critic sequence: GAME, every player's
HAND cards ordered by absolute seat, and the public event stream. Every card is
a separate token carrying owner, exact-card, rank, and suit fields; the deal is
not pooled. The critic sequence is longer than an actor sequence by
`(players - 1) * hand_size`, but it runs only in the update and never changes
actor rollout tokens or KV caches. Its value head emits one column per absolute
seat. Input owner/actor id `s` and value column `s` therefore name the same
player throughout a game. Hidden cards never enter the deployed actor.
The oracle also has a trainable feasibility-masked final-trick head on the
same absolute-seat axis and a 52-score exact-card ordering head. The latter is
normalized independently within every player's remaining hand as a
Plackett--Luce distribution over its future play permutation. These oracle
auxiliary readouts are applied at every causal token from the first bidding
state onward; the value MLP is likewise applied there and emits every active
player's centered EV. GAME and owner-tagged dealt-card tokens have zero loss
weight. Only the
acting-player value at learned pre-action states is gathered for PPO advantage
construction. Final-trick and order logits
are packed into one linear projection,
and all oracle losses share one backward pass. Its suit-presence head stays frozen: every owner's
remaining suits are directly recoverable from the perfect-information card
prefix and public plays, so that task would be trivial. The observer-limited
actor retains the nontrivial observer-limited belief heads. In the active PPO profile their
losses differentiate through the actor hidden-state graph, training the shared
embeddings, transformer, and normalization alongside the readouts. Their
shared-trunk effects on action probabilities remain subject to the PPO KL
guard. Equal core and auxiliary rates allow the actor to use one Adam parameter
group. The oracle critic is a separate network and keeps full gradients.

Update-time sequences are tail-padded into configurable length buckets. The
active width 16 coalesces neighboring player/hand shapes while avoiding the
larger padding cost measured at width 32. Every supervised readout precedes
padding, so causal
attention makes the genuine hidden states and losses identical to exact-length
batches. PPO actor bid/card heads remain sparse at decisions. Enabled actor
belief heads and all enabled oracle heads are projected densely over the
rectangular hidden matrix, with zero-weight padding removed before loss
reduction. The active linear auxiliaries are packed into one projection; all
their losses and the policy loss are summed before the microbatch backward.

For counterfactual objectives, branch placement and candidate selection are
controlled by the training configuration. The NeuRD preset branches every
eligible decision through seven cards and tapers geometrically to rate `0.5`
at ten cards under a fixed cache-row budget.

The NeuRD preset contains 24 self-play deals and 24 anchor deals. The active
local PPO profile instead contains 768 current-policy self-play games: 32 games
for every one of the 24 player/hand shapes. Every seat samples from and updates
the shared actor. The deterministic heuristic remains available to the
separate evaluator but is not used during collection.

Bids use five distinct policy-mass strata and plays use four. If no more than
that many actions are legal, every action is evaluated. Otherwise actions are
partitioned deterministically into disjoint, mass-balanced strata and one
representative is sampled from each under the old policy conditioned on its
stratum. For stratum `G` with mass `M_G`:

```text
P(a selected) = pi_old(a) / M_G
backup/reach weight = M_G
```

The candidate count is exactly the configured width, weights sum to one, the
already sampled on-policy spine is retained as its stratum's representative,
and both the recursive value backup and represented descendant distribution
are unbiased.

## Heads

- Bid logits over `0..hand_size`.
- Card logits over all 52 cards, masked to legal cards. Each card's effective
  output direction is an exact-card residual plus one rank/value embedding and
  one suit embedding. This shares rank behavior across suits and suit behavior
  across ranks without forcing different cards to be identical.
- Relative value readout, supervised at focal policy decisions by default.
- Oracle PPO value MLP with one output column per absolute seat, trained on all
  active players at each learned pre-action state, hard-centered across active
  seats to sum to zero, and used only during updates.
- Oracle PPO final-trick logits on the same absolute-seat axis, trained at
  every token from bidding onward; no oracle suit-presence objective.
- Oracle PPO exact-card order scores, listwise-normalized within each absolute
  player's remaining hand at every token from bidding onward.
- Per-opponent suit-presence logits (sigmoid, four suits).
- Per-opponent plus any-opponent rank-boundary logits (two sigmoids per suit:
  below the observer's current minimum and above its current maximum; absent
  observer suits masked).
- Per-seat final trick-count logits (softmax over `0..hand_size`,
  feasibility-masked), the observer included.
- Optional per-seat bid-hit logits (sigmoid), off in both local presets.

The shared output rows start at zero, so introducing them into an existing
checkpoint preserves every card logit exactly. They then learn with the core
policy parameters. Evaluation caches the 52 composed rows, retaining one
card-logit matrix multiplication per autoregressive step.

The policy reads only the final visible position. In PPO, beliefs are
supervised at learned decision positions; in the legacy branching objective
they remain supervised at every owned position of every branch leaf. Value is
supervised at the focal decision positions where it is used as a control
variate.
