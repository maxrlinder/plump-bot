# Training

## Objective

### NeuRD

The current preset uses paper-backed NeuRD with a multi-action control-variate
estimator. For frozen pre-sampling value prediction `b`, action-value sample
`Q(a)`, and exact candidate-inclusion probability `q(a)`:

```text
Q_hat(a) = b + 1[a observed] * (Q(a) - b) / q(a)
A_hat(a) = Q_hat(a) - sum_x pi_old(x) Q_hat(x)
L_policy = -sum_a stop_gradient(A_hat(a)) * logit(a)
```

With exponent one and no clip/cap, every component has expectation
`Q_pi(a) - V_pi`; unobserved legal actions receive the centered control
variate rather than a fabricated zero. This is the sampled NeuRD update, not a
policy-gradient or self-normalized mirror approximation.

The collector uses iid old-policy action draws. Duplicate actions collapse
into empirical multiplicities, so the recursive value backup is the ordinary
unbiased Monte Carlo mean. One independent legal-uniform draw provides
exploration and a floor on `q(a)`; when it was not also policy-sampled it has
zero parent-backup and downstream reach weight.

Every branch child carries its empirical old-policy reach. Value, belief, and
descendant policy losses are weighted by that reach, so uniform-only
descendants can estimate their parent's counterfactual action value without
changing the on-policy state distribution. Cache blocking changes estimator
variance, not the expected objective.

After Adam proposes an update, KL is measured on every policy row. Both
weighted mean and weighted p99 caps must pass; p95, p99, and max are reported.
Failed proposals restore the exact model and optimizer state and retry at a
geometrically smaller learning rate.

### Sampled mirror target

`training.policy_objective="sampled_mirror"` remains an explicit
bias/variance alternative. It exponentiates the same stochastic full-action
advantage vector and fits the resulting target. The target is exact for that
realized vector, but expectation does not commute with exponentiation, so it
is intentionally described as sampled mirror improvement rather than an
unbiased full-information mirror-descent update.

For both objectives, per-tree weighting prevents large long-game trees from
silently dominating. A negative branch-depth exponent keeps bidding and early
play meaningful despite the multiplicity of deep branch nodes. The preset
uses four iid policy bid draws and three iid policy play draws, plus one
independent uniform draw at each expanded decision.

The optimizer also trains relative value plus suit-presence and final
trick-count auxiliaries. Both auxiliaries use coefficient `0.05`; value remains
`0.5`. Bid hit and entropy remain disabled.

The beliefs are chosen so each asks something the prefix does not already
answer. Own suit presence is excluded because the observer's hand and its plays
are both in its own token stream — supervising it would fit an identity. Own
trick count is included because the observer's final total depends on how the
rest of the round resolves. Bid hit is off by default: measured against a
per-position base rate it is the weakest of the three, and unlike the others it
needs a hidden layer, since `won == bid` is a bump rather than a threshold and
a linear readout cannot express two decision boundaries.

## Rollout packing and historical opponents

The preset deals two independent focal seats/hands per `(players, cards)`
shape. Through five cards they share one wave loop; from six cards upward each
deal runs in its own wave loop so a wide tree gets the full cache budget.
`rollout.deals_per_batch` and `rollout.parallel_deals_max_hand_size` select
these behaviors.

Historical opponents are currently disabled. When enabled, their games always
use a fresh independent deal and focal seat. `concurrent` joins those games to
the self-play wave; `sequential` uses separate waves to reduce peak memory.
Only checkpoints with iteration in `[ceil(current / 2), current]` are eligible
for sampling.

NeuRD follows [Neural Replicator
Dynamics](https://arxiv.org/abs/1906.00190). The candidate correction is a
Horvitz–Thompson/control-variate estimate with closed-form inclusion
probabilities. This is not labelled CFR, Deep CFR, or R-NaD: those methods
require cumulative/average-strategy state or reference-policy transformations
beyond this local optimizer.

## Configuration

`configs/train.toml` is the source of truth for a new run. Its defaults match
the optimized schema-v6 training entrypoint:

- balanced player/card/bidding-position schedule;
- schema-v6 model dimensions and token flags;
- branch rule and per-shape rate schedule;
- cache and microbatch limits;
- loss coefficients, KL cap, and warmup;
- policy objective, sampled-mirror target, and KL backtracking;
- evaluation, dashboard, checkpoint, and league cadence.

Use typed dotted overrides:

```bash
uv run plump train experiment \
  --set model.d_model=320 \
  --set training.learning_rate=0.0001 \
  --set rollout.kv_dtype=\"fp16\"
```

Unknown keys are rejected. The fully resolved configuration is stored in the
run and must match on resume.

## Benchmarks

Measurement-only tools live in `tools/benchmarks/`. They cover collect/update
throughput, KV-cache scaling, schedule calibration, per-shape cost, branch
rate grids, and rollout sweeps. They do not write into a training run unless
given an explicit output.
