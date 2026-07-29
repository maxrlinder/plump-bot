# Runs and Checkpoints

## Layout

```text
runs/<name>/
  config.toml
  metadata.json
  metrics.csv
  train.log
  dashboard.png
  dashboard-eval.png
  checkpoints/
    iter_000000.pt
    iter_000050.pt
    iter_000100.pt
    latest.json
    best.pt
    best.json
  evaluations/<checkpoint>/heuristic.json
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
When resuming an interval checkpoint, any metric rows written after that
checkpoint are atomically discarded before training continues, preventing
duplicate or stale dashboard history.

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
New runs also save iteration zero so the untrained baseline can be evaluated
and a run interrupted before its first interval can resume exactly.

Writes go to a temporary sibling, are reloaded for validation, and atomically
replace the destination. Only then is `latest.json` advanced. Historical files
are never deleted when the active league rotates beyond its configured member
count.

`best.pt` is atomically replaced when held-out relative reward against the
heuristic improves. `latest`, `best`, and numeric iterations are accepted by
the evaluate, analyze, and play commands.

## Sidecar evaluation

Evaluate every completed interval checkpoint on the same fixed deal bank:

```bash
uv run plump evaluate RUN --checkpoint all
```

The heuristic protocol covers every configured player-count/hand-size cell,
every focal hand, and every initial bidding position. Reports include
confidence intervals and are written atomically below
`evaluations/iter_NNNNNN/`. Existing reports with an identical protocol are
reused unless `--force` is supplied.

Evaluation can follow a live trainer without sharing its run lock or metrics
writer:

```bash
uv run plump evaluate RUN --checkpoint all --watch --batch-size 64
```

Only one checkpoint model is resident at a time. `--batch-size` trades a small
amount of inference memory for speed; it does not change the deal bank.
Sidecar evaluation defaults to at most 64 rows even if the inline evaluator is
configured higher. The watcher refreshes `dashboard-eval.png` on new training
rows as well as new checkpoint results, producing a stable live image that an
already running trainer with older dashboard code cannot overwrite.
