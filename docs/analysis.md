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
reward plus bid-accuracy lines. These sparse series use a trailing five-
evaluation mean and no point markers or confidence shading; exact scores and
confidence intervals remain available in the sidecar JSON reports. The remaining panels show rollout
outcomes, value and belief learning, entropy and KL, pre-clip gradient norms,
wall-time breakdown, and rollback state. The former memory panel shows
normalized bid/play entropy against their current targets on its left axis and
the corresponding adaptive coefficients on a logarithmic right axis. The
footer reports iterations and ETA to the next evaluation using mean update
time since the latest checkpoint. Learning rate is hidden unless
`--include-learning-rate` is passed. Sparse evaluation results and resumed CSV
files are supported.
New automatic evaluations always contribute four exact observations: sampled and
argmax relative reward plus sampled and argmax bid accuracy.

Every continuous per-iteration training series uses a trailing 50-iteration
mean by default, with raw observations retained as faint lines underneath.
Checkpoint evaluation lines use their fixed five-evaluation window.
Trust-region cap lines, rollback markers, and cache-cap events remain at their exact iterations.
`--smooth N` changes the trailing window for an on-demand render.

The trust-region panel distinguishes the exact nominal Adam proposal from the
accepted post-backtracking update. The dashed `proposed mean KL` series is
measured before any reduction; accepted mean, weighted p99, and max retain
their usual meanings. The KL axis is always logarithmic. The current mean cap
appears as a dotted reference line, as does the p99 cap when that guard is
enabled. Full rollbacks retain red crosses; ordinary successful backtracks are
already represented by the
accepted and proposed KL series and do not add markers. Runs created before
this telemetry simply begin the dashed series after their next resumed update.

For counterfactual runs, the value-and-belief panel is scoped to focal decision
states. Raw reward-point value RMSE has its own axis; weighted
prediction/target correlation, suit-presence loss, and final trick-count loss
share the dimensionless axis.

When oracle-PPO telemetry is present, that panel automatically becomes
**Oracle critic dynamics**. It compares decision-only acting-seat RMSE, dense
all-active-seat RMSE over every genuine oracle token, and the decision-only
zero-baseline RMSE, then shows the corresponding return correlations and the
fractional critic-loss reduction from the first to last critic epoch in an
update. `metrics.csv` retains the raw `critic_loss_first_epoch` and
`critic_loss_last_epoch` values. The gradient panel adds the separately clipped
`critic_grad_norm` beside the core/shared and value/belief groups on its
logarithmic scale. Historical CSVs are upgraded with blank cells, so new series
begin only when the resumed trainer can measure them.

Two of that panel's series split the actor gradient by *objective* rather than
by parameter, and are present only when `ppo_auxiliary_grad_fraction > 0`.
`policy_loss_grad_norm` and `auxiliary_loss_grad_norm` are each the norm over
every actor parameter of what one half of the loss produced on its own, taken
before the rebalance and before the clip; `auxiliary_grad_scale` is the factor
applied to bring their ratio to the configured fraction. The other three norms
in the panel slice the *summed* gradient by parameter group, so they answer a
different question and are not comparable with these: measured at v4's
iteration 47,600 the actor's 6.3 pre-clip norm was 6.37 PPO against 0.41
auxiliary, while the `auxiliary_grad_norm` parameter slice read 0.0012.

Belief losses replace the old tree/search-volume panel. It includes the masked
rank-boundary BCE alongside suit, final-trick, and oracle card-order losses.
In the active PPO profile these losses are trajectory-normalized averages over
all genuine causal tokens, while the 0/4/8 diagnostics remain sampled at their
named actor decision states.
The card-order series is normalized Plackett--Luce NLL summed over critic
epochs; its companion top-1 series asks whether the oracle's highest-scored
remaining card is each player's next realized card. The bottom row adds PPO
auxiliary telemetry: actor accuracies at 0/4/8 cards personally played in
ten-card games. The former evaluation-gap panel now shows stage-specific
suit-presence and rank-boundary BCE at those same 0/4/8-card states.
Suit accuracy is per opponent/suit bit. Rank-boundary accuracy is class
balanced over lower/higher bits for each opponent and the any-opponent aggregate;
suits absent from the observer's current hand are excluded. Trick accuracy is exact final-count per
active seat. Oracle trick accuracy is shown separately over all real oracle
prefixes and must not be compared as hidden-information inference. The oracle
trick loss is summed across the configured critic epochs; actor belief losses
are summed across actor epochs (one in the current run).

When an automatic background evaluation completes, the trainer refreshes
`dashboard.png` directly, so all four evaluation observations appear without
waiting for the next five-update dashboard cadence. Their exact mode-gap
values remain available in `metrics.csv` even though the dashboard uses that
panel for stage-specific hidden-card losses.

`value_prediction_std`, normalized MSE, and value-row count remain available
in `metrics.csv` even when they are not separately plotted.

Manual or watch-mode sidecar evaluators write `dashboard-eval.png`; the current
training pipeline keeps `dashboard.png` authoritative for automatic jobs.

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

To track that geometry over every interval checkpoint, run:

```bash
uv run --group analysis plump analyze RUN --history
```

This writes `analysis/card_geometry_history.png`, a longitudinal input/output
comparison of MDS stress, PCA concentration, suit accuracy and nearest-suit
structure, rank MAE and nearest-rank gap, permutation p-values, and cosine
distances for all/same-suit/same-rank/nearest card pairs. The accompanying CSV
and JSON retain every scalar. The JSON is also an incremental cache: rerunning
the command analyzes only new or replaced checkpoints unless `--force` is
given.
