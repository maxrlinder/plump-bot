# Training

## Objective

Collection builds external-sampling counterfactual trees for one focal seat.
Every focal decision contributes one NeuRD row, whether it branched or followed
the sampled spine:

```text
L_policy = -sum_a weight(a) * stop_gradient(Q(a) - V) * logit(a)
```

Candidate inclusion probabilities correct sampled action sets. Per-tree
weighting prevents large long-game trees from silently dominating the
objective. The optimizer also trains relative value and two beliefs:
suit presence over opponents, and final trick count over every seat including
the observer's own. Bid hit and the entropy bonus are available but disabled in
the versioned preset.

The beliefs are chosen so each asks something the prefix does not already
answer. Own suit presence is excluded because the observer's hand and its plays
are both in its own token stream — supervising it would fit an identity. Own
trick count is included because the observer's final total depends on how the
rest of the round resolves. Bid hit is off by default: measured against a
per-position base rate it is the weakest of the three, and unlike the others it
needs a hidden layer, since `won == bid` is a bump rather than a threshold and
a linear readout cannot express two decision boundaries.

The KL guard snapshots model and Adam state before an epoch. An update that
exceeds the configured full-policy KL cap is rolled back, and a kept-step
counter drives the learning-rate warmup.

## Configuration

`configs/train.toml` is the source of truth for a new run. Its defaults match
the optimized schema-v6 training entrypoint:

- balanced player/card/bidding-position schedule;
- schema-v6 model dimensions and token flags;
- branch rule and per-shape rate schedule;
- cache and microbatch limits;
- loss coefficients, KL cap, and warmup;
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
