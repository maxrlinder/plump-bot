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
[GAME] [HAND × N] [TURN] [BID × P]
{ [PLAY × P] [TRICK_WIN] } × N
```

Every token contains twelve base categorical slots: type, relative player,
rank, suit, exact card, bid, trick, position in trick, hand size, player count,
next actor, and next phase. Slot embeddings are fused into one lookup table
and summed with an absolute position embedding.

The fixed-width row also has ten remaining-hand card-id positions. They are NA
except on `TRICK_WIN`, where they contain the observer's still-held cards after
that trick. The model adds the sum of those cards' existing exact-card + rank +
suit input directions to the winner token. This is observer-visible state,
does not expose opponents' hands, adds no model parameters, and gives the
between-trick state update a direct representation of the hand that remains.

Only observable information enters the sequence. Suit-presence, final bid-hit,
trick-count, and value labels are reconstructed from the true completed deal
for training and never become model inputs.

The belief heads do not all use the same seat axis, and the split is
deliberate. Suit presence covers **opponents only**: the observer's own dealt
hand and every card it has played are already in its token stream verbatim, so
a column for its own suits would supervise an identity rather than a belief.
Outcome beliefs — trick count, and bid hit when enabled — keep **every seat
including the observer's own**, because how many tricks the observer ends up
taking depends on how the rest of the round plays out and is not readable off
its hand.

## Model and rollout

`SeqPlumpModel` is a causal transformer with grouped-query attention. The
current preset uses a 384-wide, six-layer model with twelve query heads, two KV
heads, and a 1152-wide feed-forward block — 7.83M parameters.

The rollout collector advances every live branch in event waves. KV prefixes
are copied when a focal decision branches, so siblings only decode their new
suffixes. Opponent turns and self/historical arms are batched by model.

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

Branch placement and candidate selection are controlled by the training
configuration. The active preset branches every eligible decision through
seven cards and tapers geometrically to rate `0.5` at ten cards under a fixed
cache-row budget.

Each update contains 24 self-play deals and 24 anchor deals. The anchor starts
as the deterministic heuristic and changes to recent historical checkpoints
after four consecutive positive sampled-policy evaluations. Heuristic games
share the wave scheduler with self-play but allocate model state only for the
focal learner, so opponent actions add no transformer forwards or KV rows.

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
- Per-opponent suit-presence logits (sigmoid, four suits).
- Per-seat final trick-count logits (softmax over `0..hand_size`,
  feasibility-masked), the observer included.
- Optional per-seat bid-hit logits (sigmoid), off in the current preset.

The shared output rows start at zero, so introducing them into an existing
checkpoint preserves every card logit exactly. They then learn with the core
policy parameters. Evaluation caches the 52 composed rows, retaining one
card-logit matrix multiplication per autoregressive step.

The policy reads only the final visible position. Beliefs are supervised at
every owned position of every branch leaf; value is supervised at the focal
decision positions where it is actually used as a control variate.
