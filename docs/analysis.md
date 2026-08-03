# Dashboard and Analysis

## Dashboard

Training atomically refreshes `dashboard.png` every five iterations by default.
It can also be regenerated without touching training:

```bash
uv run plump dashboard RUN
```

The top-left graph combines run-scoped checkpoint evaluations against the
heuristic with any older inline evaluation rows. Sidecar reports take
precedence and show separate deterministic-argmax and policy-sampled relative
reward/confidence bands plus bid accuracy. The remaining panels show rollout
outcomes, value and belief learning, entropy and KL, branch/data volume,
pre-clip gradient norms, wall-time breakdown, cache pressure, device memory,
and rollback state. Learning rate is hidden unless `--include-learning-rate`
is passed. Sparse evaluation results and resumed CSV files are supported.

Every continuous per-iteration training series uses a trailing 50-iteration
mean by default, with raw observations retained as faint lines underneath.
Checkpoint evaluations, confidence intervals, trust-region cap lines,
rollback markers, and cache-cap events remain at their exact iterations.
`--smooth N` changes the trailing window for an on-demand render.

The trust-region panel distinguishes the exact nominal Adam proposal from the
accepted post-backtracking update. The dashed `proposed mean KL` series is
measured before any reduction; accepted mean, weighted p99, and max retain
their usual meanings. The KL axis is always logarithmic, and the current run's
mean and p99 caps appear as dotted reference lines. Full rollbacks retain red
crosses; ordinary successful backtracks are already represented by the
accepted and proposed KL series and do not add markers. Runs created before
this telemetry simply begin the dashed series after their next resumed update.

The value-and-belief panel is scoped to focal decision states. Raw reward-point
value RMSE has its own axis; weighted prediction/target correlation,
suit-presence loss, and final trick-count loss share the dimensionless axis.
The gradient panel shows the separately clipped core/shared and
value/belief-readout parameter-group norms before clipping on a logarithmic
scale, with the configured `1.0` threshold as a dotted reference.
`value_zero_rmse`, `value_prediction_std`, normalized MSE, and value-row count
remain available in `metrics.csv`. These series start at the first update
produced by the newer reporting schema.

While an older trainer process is still refreshing `dashboard.png`, a sidecar
evaluator writes the new layout to `dashboard-eval.png`.

## Card representation analysis

Install the optional analysis group and analyze any saved checkpoint:

```bash
uv run --group analysis plump analyze RUN --checkpoint latest
```

For both the effective card input embedding and the effective exact + rank +
suit card-action output rows, the command writes PCA, cosine MDS, UMAP,
cosine-similarity heatmaps, and cross-validated suit/rank probe plots. A
`report.json` records checkpoint identity and all quantitative results.

Outputs are always scoped to `runs/<name>/analysis/<checkpoint>/`; source
directories never contain generated PNGs.
