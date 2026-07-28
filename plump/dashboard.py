"""Static training dashboard for schema-v6 runs."""

from __future__ import annotations

import csv
import math
import uuid
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def render_dashboard(
    metrics_path: str | Path,
    output_path: str | Path,
    *,
    title: str | None = None,
    smooth: int = 20,
    dpi: int = 150,
) -> int:
    metrics_path = Path(metrics_path)
    output_path = Path(output_path)
    rows = _read_rows(metrics_path)
    if not rows:
        raise ValueError(f"No metric rows in {metrics_path}")

    iteration = _series(rows, "iteration")
    fig, axes = plt.subplots(3, 3, figsize=(20, 13), constrained_layout=True)
    fig.suptitle(title or f"Plump schema-v6 · {metrics_path.parent.name}", fontsize=16)

    _lines(
        axes[0, 0],
        iteration,
        rows,
        (
            ("eval_reward_vs_heuristic", "relative reward"),
            ("eval_bid_hit", "bid hit"),
        ),
        smooth=1,
        title="Held-out evaluation",
    )
    _lines(
        axes[0, 1],
        iteration,
        rows,
        (
            ("reward_self", "self"),
            ("reward_historical", "historical"),
            ("bid_hit_rate", "rollout bid hit"),
        ),
        smooth=smooth,
        title="Rollout outcomes",
    )
    _lines(
        axes[0, 2],
        iteration,
        rows,
        (
            ("loss_policy", "policy"),
            ("loss_value", "value"),
            ("loss_suit", "suit"),
            ("loss_bid_hit", "bid hit"),
            ("loss_trick", "trick count"),
        ),
        smooth=smooth,
        title="Objectives",
    )
    _lines(
        axes[1, 0],
        iteration,
        rows,
        (
            ("spine_entropy", "spine entropy"),
            ("entropy", "update entropy"),
            ("policy_kl", "policy KL"),
            ("rolled_back", "rollback"),
        ),
        smooth=smooth,
        title="Policy health",
    )
    _lines(
        axes[1, 1],
        iteration,
        rows,
        (
            ("trees", "trees"),
            ("leaves", "leaves"),
            ("positions", "positions"),
            ("forward_rows", "forward rows"),
        ),
        smooth=smooth,
        title="Training volume",
    )
    _lines(
        axes[1, 2],
        iteration,
        rows,
        (
            ("branched_rows", "branched rows"),
            ("unbranched_rows", "unbranched rows"),
            ("branch_decisions", "branch decisions"),
            ("blocked_by_cache", "cache blocked"),
            ("skipped_by_placement", "placement skips"),
        ),
        smooth=smooth,
        title="Branching",
    )
    _lines(
        axes[2, 0],
        iteration,
        rows,
        (
            ("positions_per_sec", "positions/s"),
            ("forward_rows_per_sec", "forward rows/s"),
        ),
        smooth=smooth,
        title="Throughput",
    )
    _lines(
        axes[2, 1],
        iteration,
        rows,
        (
            ("collect_sec", "collect"),
            ("update_sec", "update"),
            ("forward_sec", "forward"),
            ("token_build_sec", "token build"),
        ),
        smooth=smooth,
        title="Wall time (seconds)",
    )
    _lines(
        axes[2, 2],
        iteration,
        rows,
        (
            ("peak_cache_rows", "peak cache rows"),
            ("cache_rows_allocated", "allocated rows"),
            ("cache_pressure", "cache pressure"),
            ("peak_device_gb", "peak device GB"),
            ("learning_rate", "learning rate"),
        ),
        smooth=smooth,
        title="Memory and optimizer",
    )

    for ax in axes.flat:
        ax.set_xlabel("Iteration")
        ax.grid(alpha=0.18)
        if ax.lines:
            ax.legend(fontsize=8, loc="best")

    last = rows[-1]
    fig.text(
        0.5,
        0.005,
        " · ".join(
            (
                f"iteration {int(float(last['iteration']))}",
                f"elapsed {float(last.get('elapsed_sec') or 0):.1f}s",
                f"rollback {int(float(last.get('rolled_back') or 0))}",
            )
        ),
        ha="center",
        fontsize=9,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.stem}.{uuid.uuid4().hex}.tmp{output_path.suffix}"
    )
    try:
        fig.savefig(temporary, dpi=dpi)
        plt.close(fig)
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return len(rows)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _series(rows: list[dict[str, str]], name: str) -> np.ndarray:
    values = []
    for row in rows:
        raw = row.get(name, "")
        try:
            values.append(float(raw) if raw not in ("", None) else math.nan)
        except ValueError:
            values.append(math.nan)
    return np.asarray(values, dtype=np.float64)


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    out = np.full_like(values, np.nan)
    for index in range(len(values)):
        chunk = values[max(0, index - window + 1) : index + 1]
        finite = chunk[np.isfinite(chunk)]
        if finite.size:
            out[index] = finite.mean()
    return out


def _lines(
    ax,
    iteration: np.ndarray,
    rows: list[dict[str, str]],
    fields: tuple[tuple[str, str], ...],
    *,
    smooth: int,
    title: str,
) -> None:
    plotted = False
    for field, label in fields:
        values = _series(rows, field)
        valid = np.isfinite(iteration) & np.isfinite(values)
        if not valid.any():
            continue
        rendered = _smooth(values, smooth)
        ax.plot(iteration[valid], rendered[valid], linewidth=1.6, label=label)
        plotted = True
    ax.set_title(title)
    if not plotted:
        ax.text(
            0.5,
            0.5,
            "No data yet",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#777777",
        )
