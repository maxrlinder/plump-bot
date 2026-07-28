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

## Model and rollout

`SeqPlumpModel` is a causal transformer with grouped-query attention. The
current preset uses a 384-wide, six-layer model with twelve query heads, two KV
heads, and a 1024-wide feed-forward block.

The rollout collector advances every live branch in event waves. KV prefixes
are copied when a focal decision branches, so siblings only decode their new
suffixes. Opponent turns and self/historical arms are batched by model.

Branch placement and candidate selection are controlled by the training
configuration. The active preset branches bids to the top four candidates,
uses distinct Gumbel top-k play candidates, fully branches short games, and
tapers the rate for longer games under a fixed cache-row budget.

## Heads

- Bid logits over `0..hand_size`.
- Card logits over all 52 cards, masked to legal cards.
- Per-position relative value.
- Per-seat suit-presence logits.
- Per-seat bid-hit logits.
- Optional per-seat final trick-count logits.

The policy reads only the final visible position. Training can supervise value
and beliefs at every owned position of every branch leaf.
