# Dashboard and Analysis

## Dashboard

Training atomically refreshes `dashboard.png` every five iterations by default.
It can also be regenerated without touching training:

```bash
uv run plump dashboard RUN
```

The top-left graph combines run-scoped checkpoint evaluations against the
heuristic with any older inline evaluation rows. Sidecar reports take
precedence and show relative reward with its confidence band plus bid
accuracy. The remaining panels show rollout outcomes, objective losses,
entropy and KL, branch/data volume, throughput, wall-time breakdown, cache
pressure, device memory, and rollback state. Learning rate is hidden unless
`--include-learning-rate` is passed. Sparse evaluation results and resumed CSV
files are supported.

The trust-region panel distinguishes the exact nominal Adam proposal from the
accepted post-backtracking update. The dashed `proposed mean KL` series is
measured before any reduction; accepted mean, weighted p99, and max retain
their usual meanings. The KL axis is always logarithmic, and the current run's
mean and p99 caps appear as dotted reference lines. Orange triangles mark
iterations that backtracked and sit at the accepted mean KL—they are events,
not rejected-candidate measurements. Runs created before this telemetry simply
begin the dashed series after their next resumed update.

The auxiliary-loss panel includes a `value: predict zero` reference computed
with the same Smooth-L1 objective and branch-reach weights as the learned value
loss. A learned value below that line is adding predictive information; a high
raw loss with a similarly high reference mostly reflects the return scale and
irreducible outcome variance. This series starts at the first update produced
by the newer reporting schema.

While an older trainer process is still refreshing `dashboard.png`, a sidecar
evaluator writes the new layout to `dashboard-eval.png`.

## Card representation analysis

Install the optional analysis group and analyze any saved checkpoint:

```bash
uv run --group analysis plump analyze RUN --checkpoint latest
```

For both the effective card input embedding and card-action head, the command
writes PCA, cosine MDS, UMAP, cosine-similarity heatmaps, and cross-validated
suit/rank probe plots. A `report.json` records checkpoint identity and all
quantitative results.

Outputs are always scoped to `runs/<name>/analysis/<checkpoint>/`; source
directories never contain generated PNGs.
