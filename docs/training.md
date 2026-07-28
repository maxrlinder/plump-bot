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
objective. The optimizer also trains relative value, suit-presence, and
bid-hit heads; trick count and entropy are available but disabled in the
versioned preset.

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
