# Schema-v6 Benchmarks

These tools measure the current autoregressive pipeline without managing a
durable training run.

- `benchmark_seq_throughput.py`: full collect/update cycles.
- `benchmark_seq_rollout_scaling.py`: cache and no-cache scaling.
- `calibrate_seq_schedule.py`: whole-grid rollout timing.
- `report_seq_branch_shape.py`: branching depth and leaf shape.
- `report_seq_rate_grid.py`: rate cost by game shape.
- `report_seq_shape_cost.py`: time, memory, and positions per deal.
- `sweep_seq_rollout.py`: isolated model/shape subprocess sweeps.

Run them from the repository root with `.venv/bin/python`.
