# Training

## Objective

Collection builds external-sampling counterfactual trees for one focal seat.
Every focal decision contributes one policy row, whether it branched or
followed the sampled spine. `training.policy_objective` selects the optimizer.

### Sampled policy mirror descent

The current preset uses sampled entropic policy mirror descent. For each
expanded action it forms a Horvitz–Thompson advantage estimate:

```text
A_hat(a) = 1[a expanded] * clip(Q(a) - V) / q(a)
```

Here `q(a)` is the candidate-inclusion probability recorded by the collector;
unexpanded legal actions receive zero in this sampled estimate. The exact
non-parametric improvement target for that estimate is:

```text
pi_target(a) ∝ anchor(a) * exp(step_size * A_hat(a))
```

The anchor is the behavior policy mixed with a configurable amount of
legal-uniform exploration. This gives a suppressed action finite recovery
velocity without pretending the uniform rollout arm was on-policy. The
exponentiated direction is reduced per row until
`KL(pi_old || pi_target) <= mirror_target_kl`, and the network minimizes
`KL(pi_target || pi_model)`. Its logit gradient is the bounded probability
difference `pi_model - pi_target`.

After Adam proposes the shared neural update, the full-policy KL guard is
measured over every policy row. An oversized step is retried from the exact
pre-step model and optimizer state at geometrically smaller learning rates.
Only when all configured retries fail is the update rolled back.

### NeuRD

The original objective remains available with
`training.policy_objective="neurd"`:

```text
L_policy = -sum_a weight(a) * stop_gradient(Q(a) - V) * logit(a)
```

This preserves the direct per-logit regret update and its original KL anchor.
It is useful as an exact baseline, but raw regret gradients can produce a
large shared-network step even after gradient clipping.

For both objectives, per-tree weighting prevents large long-game trees from
silently dominating. The current preset uses a negative branch-depth exponent
so bidding and early play retain meaningful policy weight despite the
multiplicity of deep branch nodes. It explicitly samples four distinct
policy-weighted bid actions plus one distinct uniform explorer. The historical
league is enabled once the first interval checkpoint exists.

The optimizer also trains relative value and two beliefs: suit presence over
opponents, and final trick count over every seat including the observer's own.
Their preset coefficients are deliberately smaller than the policy/value
terms, so they remain useful representation auxiliaries without dominating
the bounded policy gradient. Bid hit and the entropy bonus remain available
but disabled.

The beliefs are chosen so each asks something the prefix does not already
answer. Own suit presence is excluded because the observer's hand and its plays
are both in its own token stream — supervising it would fit an identity. Own
trick count is included because the observer's final total depends on how the
rest of the round resolves. Bid hit is off by default: measured against a
per-position base rate it is the weakest of the three, and unlike the others it
needs a hidden layer, since `won == bid` is a bump rather than a threshold and
a linear readout cannot express two decision boundaries.

The implementation follows the exponentiated update underlying
[policy mirror descent](https://arxiv.org/abs/2102.00135). NeuRD follows
[Neural Replicator Dynamics](https://arxiv.org/abs/1906.00190). This is not
labelled CFR, Deep CFR, or R-NaD: those methods require cumulative/average
strategy state or reference-policy reward transformations beyond a local
optimization objective.

## Configuration

`configs/train.toml` is the source of truth for a new run. Its defaults match
the optimized schema-v6 training entrypoint:

- balanced player/card/bidding-position schedule;
- schema-v6 model dimensions and token flags;
- branch rule and per-shape rate schedule;
- cache and microbatch limits;
- loss coefficients, KL cap, and warmup;
- policy objective, mirror target, and KL backtracking;
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
