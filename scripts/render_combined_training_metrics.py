#!/usr/bin/env python3
"""Render one continuous Modal-to-laptop training dashboard."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from examples.plot_training_metrics import render_metrics_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modal-dir", type=Path, required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--combined-csv", type=Path, required=True)
    parser.add_argument(
        "--smooth",
        type=int,
        default=50,
        help="Rolling window for metrics recorded every iteration.",
    )
    parser.add_argument(
        "--diagnostic-smooth",
        type=int,
        default=50,
        help="Rolling window for sparse diagnostic metric observations.",
    )
    return parser.parse_args()


def read_metrics(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        fields = list(reader.fieldnames or ())
        rows = [
            row
            for row in reader
            if row.get("iteration")
            and row.get("timestamp_utc")
            and row.get("iteration_sec")
            and row.get("samples")
        ]
    if not fields:
        raise ValueError(f"Metrics CSV has no header: {path}")
    return fields, rows


def merge_metric_rows(
    modal_rows: list[dict[str, str]],
    local_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Follow the active local resume branch and switch to it at its fork."""

    if not local_rows:
        return _latest_rows_by_iteration(modal_rows)
    active_local, _ = active_resume_branch(local_rows)
    local_by_iteration = {
        int(float(row["iteration"])): row
        for row in active_local
    }
    local_start = min(local_by_iteration)
    modal_before_fork = [
        row
        for row in _latest_rows_by_iteration(modal_rows)
        if int(float(row["iteration"])) < local_start
    ]
    return [
        *modal_before_fork,
        *(local_by_iteration[key] for key in sorted(local_by_iteration)),
    ]


def active_resume_branch(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int | None]:
    """Discard abandoned future rows after the most recent rollback.

    Metrics are append-only. A resumed older checkpoint therefore appears as
    a lower (or repeated) iteration after the abandoned branch's last row.
    Preserve the shared prefix, delete its future in memory, and then follow
    the newly appended branch.
    """

    selected: dict[int, dict[str, str]] = {}
    previous_iteration = -1
    resume_iteration: int | None = None
    for row in rows:
        iteration = int(float(row["iteration"]))
        if iteration <= previous_iteration:
            selected = {
                saved_iteration: saved_row
                for saved_iteration, saved_row in selected.items()
                if saved_iteration < iteration
            }
            resume_iteration = iteration
        selected[iteration] = row
        previous_iteration = iteration
    return (
        [selected[key] for key in sorted(selected)],
        resume_iteration,
    )


def _latest_rows_by_iteration(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    selected: dict[int, dict[str, str]] = {}
    for row in rows:
        iteration = int(float(row["iteration"]))
        previous = selected.get(iteration)
        if (
            previous is None
            or row["timestamp_utc"] >= previous["timestamp_utc"]
        ):
            selected[iteration] = row
    return [selected[key] for key in sorted(selected)]


def write_metrics(
    path: Path,
    fields: list[str],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    modal_fields, modal_rows = read_metrics(args.modal_dir / "metrics.csv")
    local_fields, local_rows = read_metrics(args.local_dir / "metrics.csv")
    fields = [*modal_fields]
    fields.extend(field for field in local_fields if field not in fields)
    combined = merge_metric_rows(modal_rows, local_rows)
    _, latest_resume_iteration = active_resume_branch(local_rows)
    if not combined:
        raise SystemExit("No combined metric rows to plot.")
    write_metrics(args.combined_csv, fields, combined)
    plotted = render_metrics_plot(
        metrics_path=args.combined_csv,
        output_path=args.output,
        smooth=args.smooth,
        diagnostic_smooth=args.diagnostic_smooth,
        title="Plump PPO Training: Modal L40S \u2192 local M5 Pro",
        since_restart=True,
    )
    if not plotted:
        raise SystemExit("Combined metrics have no plottable rows.")
    print(
        f"wrote {args.output} rows={len(combined)} "
        f"fork={local_rows[0]['iteration']} latest={combined[-1]['iteration']} "
        f"resume={latest_resume_iteration}"
    )


if __name__ == "__main__":
    main()
