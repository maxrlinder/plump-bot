# Schema-v6 Benchmarks

These tools measure the current autoregressive pipeline without managing a
durable training run.

- `benchmark_seq_throughput.py`: full collect/update cycles.
- `benchmark_seq_rollout_scaling.py`: cache and no-cache scaling.
- `benchmark_stratified_shape_grid.py`: isolated solo-versus-paired collection
  for every configured player/card shape, with repeat-level timing and memory.
  `--matched-two-deal` feeds the same two deals and CRN seeds to two serial
  one-deal waves and one paired two-deal wave. It fingerprints inputs and
  realized work, then reports game and forward-row throughput scaling.
- `calibrate_seq_schedule.py`: whole-grid rollout timing.
- `report_seq_branch_shape.py`: branching depth and leaf shape.
- `report_seq_rate_grid.py`: rate cost by game shape.
- `report_seq_shape_cost.py`: time, memory, and positions per deal.
- `sweep_seq_rollout.py`: isolated model/shape subprocess sweeps.

Run them from the repository root with `uv run python`. The stratified grid
benchmark writes its incremental JSON and summary CSV under the selected run's
ignored `benchmarks/` directory.

```bash
uv run python tools/benchmarks/benchmark_stratified_shape_grid.py \
  --run mirror-8m --checkpoint 200 --matched-two-deal --repeats 1
```

Matched reports classify doubling the batch as sublinear below `1.9x`
forward-row throughput, linear from `1.9x` through `2.1x`, and superlinear
above `2.1x`.
