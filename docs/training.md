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

The collector uses fixed-width stratified old-policy sampling. If the legal
set fits the branch budget it is fully enumerated. Otherwise legal actions are
partitioned into disjoint, policy-mass-balanced strata and one action is drawn
from each stratum under `pi_old(. | stratum)`. For stratum mass `M_g`:

```text
backup/reach weight = M_g
q(a) = pi_old(a) / M_g
V_hat = sum_g M_g Q(a_g)
```

The representatives are exactly distinct, their weights sum to one, and the
backup and represented descendant distribution are unbiased under the old
policy. Full enumeration has `q(a)=1` and weight `pi_old(a)`.

Every branch child carries its represented old-policy reach. Value, belief,
and descendant policy losses are weighted by that reach. Cache blocking
changes estimator variance, not the expected objective.

After Adam proposes an update, KL is measured on every policy row. Both
weighted mean and weighted p99 caps must pass; the current preset uses `0.01`
and `0.05`, respectively. Failed proposals restore the exact model and
optimizer state and retry at a geometrically smaller learning rate.

The first, nominal Adam proposal is always evaluated over the complete policy
row set before any reduction. Metrics retain its weighted mean, p95, p99, and
max KL plus booleans for which caps it exceeded. The existing `policy_kl*`
fields describe the final accepted proposal, not the rejected one. Intermediate
retries may still use the proof-only p99 early exit because they are immediately
discarded and are not used as the durable pre-backtrack diagnostic.

KL percentiles use the same objective weights as the loss. With equal
per-tree weighting and negative branch-depth weighting, a shallow decision can
carry at least one percent of total weight; weighted p99 can therefore equal
the maximum. This is intentional in the current implementation and should be
remembered when interpreting or changing the tail cap.

Backtracking scales the shared trunk and bid/card action heads because those
parameters can move the policy. It does not scale the value, suit-presence,
trick-count, or bid-hit readout heads: those heads only consume the shared
representation and cannot change policy KL. The current preset starts the core
at `2.5e-5` and retains `2e-4` for auxiliary readouts; the older
`learning_rate` field remains the fallback when a group-specific value is
omitted. Core and readout gradients are clipped independently, so an exact
NeuRD correction with a large norm cannot rescale the readout-head sample
before Adam. Older one-group checkpoints are split losslessly at load time.

### Sampled mirror target

`training.policy_objective="sampled_mirror"` remains an explicit
bias/variance alternative. It exponentiates the same stochastic full-action
advantage vector and fits the resulting target. The target is exact for that
realized vector, but expectation does not commute with exponentiation, so it
is intentionally described as sampled mirror improvement rather than an
unbiased full-information mirror-descent update.

For both objectives, per-tree weighting prevents large long-game trees from
silently dominating. A negative branch-depth exponent keeps bidding and early
play meaningful despite the multiplicity of deep branch nodes. The preset uses
`(1 + branch_depth)^-0.5`, a moderate early-decision preference, plus five
distinct bid strata and four distinct play strata. Smaller legal sets are fully
enumerated. The depth factors are normalized within each tree, so this changes
where its policy weight lands without changing that tree's total weight.

The optimizer also trains relative value plus suit-presence and final
trick-count auxiliaries. Both belief auxiliaries use coefficient `0.05`; value
remains `0.5`. Bid hit and entropy remain disabled.

Value is a control variate for the focal policy update, so the current
objective is normalized MSE at exactly the focal bid/play readout positions:

```text
L_value = 0.5 * ((V(s) - backed_return) / value_reward_scale)^2
```

MSE is intentional: it learns the conditional mean `E[R | s]` required by a
variance-reducing state value. Smooth-L1 becomes mostly absolute error for the
game's discrete ±5–20 point returns and therefore approaches a conditional
median, which can stay close to zero even when the expected return is
state-dependent. Decision rows use the same per-tree, reach, and branch-depth
weights as the policy objective. This both preserves the old-policy state
distribution and avoids spending most value supervision on event-token
positions where the baseline is never consumed. `value_positions="all"` and
`value_objective="smooth_l1"` remain explicit diagnostic alternatives.

Counterfactual branches do not turn these into off-policy labels. Descendant
positions carry their represented old-policy reach; the reaches sum to the
ordinary live-policy state distribution in expectation. Decision-state values
use the recursive unbiased branch backup, while outcome beliefs use the
originally sampled on-policy continuation. Consequently, residual value error
can reflect partial observability and Monte Carlo variance rather than an
unweighted exploratory policy.

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
  --set training.core_learning_rate=0.00002 \
  --set training.auxiliary_learning_rate=0.0002 \
  --set rollout.kv_dtype=\"fp16\"
```

Unknown keys are rejected. The fully resolved configuration is stored in the
run and must match on resume.

## Benchmarks

Measurement-only tools live in `tools/benchmarks/`. They cover collect/update
throughput, KV-cache scaling, schedule calibration, per-shape cost, branch
rate grids, rollout sweeps, and isolated solo-versus-paired shape grids.
Generated results are explicitly scoped to a selected run or output path.
