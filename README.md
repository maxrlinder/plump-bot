# Plump Bot

An autoregressive self-play agent for the Swedish trick-taking game Plump.
The model pipeline is schema v6: a causal transformer with KV-cached rollout
and selectable NeuRD, sampled-mirror, or branch-free PPO training.

## Quick start

```bash
uv sync --all-groups
uv run plump --help
uv run plump train first-run
```

The PPO MPS preset adds an independent privileged critic, adaptive masked
entropy targets, BF16 autocast, and FP16 KV storage:

```bash
uv run plump train ppo-mps --config configs/ppo-mps.toml
```

`training.ppo_trainable_policies` selects one shared actor or multiple
independent actor weight sets for self-play.

Training uses the versioned defaults in `configs/train.toml`. Override a value
without editing the preset:

```bash
uv run plump train first-run \
  --set run.iterations=2000 \
  --set training.reference_rate=0.6
```

Re-running the same command safely resumes `runs/first-run`. A changed config
is rejected with a field-level diff.

For an intentional mid-run curriculum or optimizer change, use the explicit
reconfiguration path. It archives the old config and writes a compatible
resume checkpoint before proceeding:

```bash
uv run plump train first-run --config configs/train.toml \
  --reconfigure --reconfigure-reason "introduce heuristic anchor"
```

## Daily commands

```bash
uv run plump dashboard first-run
uv run plump evaluate first-run --checkpoint latest
uv run plump evaluate first-run --checkpoint all --action-mode both
uv run plump analyze first-run --checkpoint best
uv run plump play first-run --checkpoint latest
```

Evaluation is safe to run beside training: it reads atomically completed
checkpoints, loads one model at a time, and never edits the trainer's metrics
file. To evaluate each existing and future checkpoint until interrupted:

```bash
uv run plump evaluate first-run --checkpoint all --action-mode both --watch
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
infra/modal_training.py  optional persistent Modal L40S execution sidecar
```

The training hot path is intentionally isolated in `plump/seq`. Run
management, checkpoint manifests, dashboard rendering, and the CLI wrap that
pipeline without changing its rollout or update semantics.

## Documentation

- [Architecture](docs/architecture.md)
- [Algorithm and math](docs/algorithm.md)
- [Training](docs/training.md)
- [Run artifacts and checkpoints](docs/runs.md)
- [Dashboard and analysis](docs/analysis.md)
- [Modal L40S training](docs/modal.md)

## Development

```bash
uv run pytest
uv lock --check
```

`.venv` is the reproducible local environment. `.uv-cache`, `.pytest_cache`,
`__pycache__`, `runs`, and generated model artifacts are disposable and
ignored.
