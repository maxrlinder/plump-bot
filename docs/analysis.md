# Dashboard and Analysis

## Dashboard

Training atomically refreshes `dashboard.png` every five iterations by default.
It can also be regenerated without touching training:

```bash
uv run plump dashboard RUN
```

The dashboard shows held-out strength and bid hit, rollout outcomes, objective
losses, entropy and KL, branch/data volume, throughput, wall-time breakdown,
cache pressure, device memory, learning rate, and rollback state. Sparse
evaluation columns and resumed CSV files are supported.

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
