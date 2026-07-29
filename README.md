# Plump Bot

An autoregressive self-play agent for the Swedish trick-taking game Plump.
The current and only model pipeline is schema v6: a causal transformer,
KV-cached counterfactual rollout collector, and selectable NeuRD or sampled
mirror-target trainer.

## Quick start

```bash
uv sync --all-groups
uv run plump --help
uv run plump train first-run
```

Training uses the versioned defaults in `configs/train.toml`. Override a value
without editing the preset:

```bash
uv run plump train first-run \
  --set run.iterations=2000 \
  --set training.reference_rate=0.6
```

Re-running the same command safely resumes `runs/first-run`. A changed config
is rejected with a field-level diff.

## Daily commands

```bash
uv run plump dashboard first-run
uv run plump evaluate first-run --checkpoint latest
uv run plump analyze first-run --checkpoint best
uv run plump play first-run --checkpoint latest
```

Generated checkpoints, metrics, dashboards, evaluations, and analyses stay
under the ignored `runs/<name>/` directory. No generated images or model files
belong in the source tree.

## Source layout

```text
configs/train.toml       optimized, versioned training preset
plump/                   game engine and shared policies/evaluation
plump/seq/               schema-v6 tokens, model, KV cache, rollout, trainer
plump/gui/               local browser game
plump/analysis/          schema-v6 representation analysis
tests/                   engine, GUI, evaluation, and sequence tests
tools/benchmarks/        rollout and update measurement tools
```

The training hot path is intentionally isolated in `plump/seq`. Run
management, checkpoint manifests, dashboard rendering, and the CLI wrap that
pipeline without changing its rollout or update semantics.

## Documentation

- [Architecture](docs/architecture.md)
- [Training](docs/training.md)
- [Run artifacts and checkpoints](docs/runs.md)
- [Dashboard and analysis](docs/analysis.md)

## Development

```bash
uv run pytest
uv lock --check
```

`.venv` is the reproducible local environment. `.uv-cache`, `.pytest_cache`,
`__pycache__`, `runs`, and generated model artifacts are disposable and
ignored.
