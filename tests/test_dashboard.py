"""Static dashboard rendering tolerates sparse and resumed metrics."""

from __future__ import annotations

import csv

from plump.dashboard import render_dashboard


def test_dashboard_renders_sparse_partial_and_resumed_rows(tmp_path):
    metrics = tmp_path / "metrics.csv"
    with metrics.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "iteration",
                "elapsed_sec",
                "loss_policy",
                "eval_reward_vs_heuristic",
                "rolled_back",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "iteration": 1,
                "elapsed_sec": 2.0,
                "loss_policy": 0.5,
                "eval_reward_vs_heuristic": "",
                "rolled_back": 0,
            }
        )
        writer.writerow(
            {
                "iteration": 5,
                "elapsed_sec": 11.0,
                "loss_policy": "",
                "eval_reward_vs_heuristic": 0.2,
                "rolled_back": 1,
            }
        )

    output = tmp_path / "dashboard.png"
    assert render_dashboard(metrics, output, smooth=2, dpi=40) == 2
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not list(tmp_path.glob(".*.tmp.png"))
