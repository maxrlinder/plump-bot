# Runs and Checkpoints

## Layout

```text
runs/<name>/
  config.toml
  metadata.json
  metrics.csv
  train.log
  dashboard.png
  checkpoints/
    iter_000200.pt
    iter_000400.pt
    latest.json
    best.pt
    best.json
  evaluations/<checkpoint>/
  analysis/<checkpoint>/
```

`metadata.json` records the command, Git revision and dirty state, Python and
Torch versions, device, seed, timestamps, and run status.

## Resume rules

`plump train <name>` creates a missing run or resumes an existing one. Resume
loads the checkpoint named by `latest.json` only after:

1. the requested config exactly matches the stored config;
2. the checkpoint size and SHA-256 match the manifest;
3. schema and game-rule fingerprints match.

A run lock prevents two trainers from writing the same directory. Stale locks
are removed only after confirming their process no longer exists.

To intentionally change training/rollout settings while preserving model,
optimizer, RNG, iteration, and collector state, fork into a new run:

```bash
uv run plump train NEW_RUN \
  --from-checkpoint runs/OLD_RUN/checkpoints/iter_000100.pt
```

The imported checkpoint is immediately rewritten under the new resolved
configuration and recorded as the new run's `latest` checkpoint. Existing runs
still reject configuration drift.

## Checkpoint contents

Every configured interval checkpoint is retained and is fully resumable. It
contains model and optimizer state, model/training configs, iteration and kept
optimizer steps, trainer/framework RNG states, collector adaptive batching and
seat cursors, rule fingerprint, and relocatable league references.

Writes go to a temporary sibling, are reloaded for validation, and atomically
replace the destination. Only then is `latest.json` advanced. Historical files
are never deleted when the active league rotates beyond its configured member
count.

`best.pt` is atomically replaced when held-out relative reward against the
heuristic improves. `latest`, `best`, and numeric iterations are accepted by
the evaluate, analyze, and play commands.
