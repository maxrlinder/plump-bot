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
