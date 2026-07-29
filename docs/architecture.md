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

Every token contains twelve categorical slots: type, relative player, rank,
suit, exact card, bid, trick, position in trick, hand size, player count, next
actor, and next phase. Slot embeddings are fused into one lookup table and
summed with an absolute position embedding.

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

Branch placement and candidate selection are controlled by the training
configuration. The active preset branches every eligible decision through
seven cards and tapers geometrically to rate `0.5` at ten cards under a fixed
cache-row budget.

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
- Card logits over all 52 cards, masked to legal cards.
- Per-position relative value.
- Per-opponent suit-presence logits (sigmoid, four suits).
- Per-seat final trick-count logits (softmax over `0..hand_size`,
  feasibility-masked), the observer included.
- Optional per-seat bid-hit logits (sigmoid), off in the current preset.

The policy reads only the final visible position. Training can supervise value
and beliefs at every owned position of every branch leaf.
