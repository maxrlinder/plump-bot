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
configuration. The active preset branches bids to the top four candidates,
fully branches games of seven cards or fewer, and tapers the rate for longer
games under a fixed cache-row budget.

Play candidates come from `sample_k_plus_uniform`: three iid old-policy draws
plus one independent legal-uniform draw. Duplicate actions collapse into
empirical multiplicities. The policy draws therefore give an ordinary unbiased
Monte Carlo value backup, while the uniform-only arm has zero backup and
downstream reach weight. For legal action `a`, its exact inclusion probability
is:

```text
q(a) = 1 - (1 - pi_old(a))^k * (1 - 1 / |legal|)
```

This closed form is used by the control-variate NeuRD estimator. Drawing the
uniform arm independently means it can duplicate a policy draw; that small
efficiency cost is what keeps both the sampling design and its correction
exact.

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
