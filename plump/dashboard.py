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


Field = tuple[str, str]


def render_dashboard(
    metrics_path: str | Path,
    output_path: str | Path,
    *,
    title: str | None = None,
    smooth: int = 20,
    dpi: int = 150,
) -> int:
    """Render comparable quantities together and discrete events without smoothing."""

    metrics_path = Path(metrics_path)
    output_path = Path(output_path)
    rows = _read_rows(metrics_path)
    if not rows:
        raise ValueError(f"No metric rows in {metrics_path}")

    iteration = _series(rows, "iteration")
    fig, axes = plt.subplots(3, 3, figsize=(20, 13), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0.0, 0.035, 1.0, 0.96))
    fig.suptitle(
        title or f"Plump schema-v6 · {metrics_path.parent.name}",
        fontsize=16,
    )

    _dual_lines(
        axes[0, 0],
        iteration,
        rows,
        left=(("eval_reward_vs_heuristic", "relative reward"),),
        right=(("eval_bid_hit", "bid accuracy"),),
        smooth=1,
        title="Held-out evaluation",
        left_label="relative reward",
        right_label="bid accuracy",
    )
    _dual_lines(
        axes[0, 1],
        iteration,
        rows,
        left=(
            ("reward_self", "self-play reward"),
            ("reward_historical", "historical-opponent reward"),
        ),
        right=(("bid_hit_rate", "observed bid accuracy"),),
        smooth=smooth,
        title="Observed outcomes",
        left_label="relative reward",
        right_label="bid accuracy",
        omit_zero=True,
    )
    _dual_lines(
        axes[0, 2],
        iteration,
        rows,
        left=(
            ("loss_policy", "target-fit loss"),
            ("policy_kl", "realized KL"),
        ),
        right=(("learning_rate", "learning rate"),),
        smooth=min(smooth, 5),
        right_smooth=1,
        title="Policy update",
        left_label="KL / loss",
        right_label="learning rate",
        omit_zero=True,
    )
    _lines(
        axes[1, 0],
        iteration,
        rows,
        (
            ("loss_value", "value"),
            ("loss_suit", "suit presence"),
            ("loss_trick", "trick count"),
            ("loss_bid_hit", "bid-hit belief"),
        ),
        smooth=smooth,
        title="Active auxiliary losses",
        ylabel="loss",
        omit_zero=True,
    )
    _lines(
        axes[1, 1],
        iteration,
        rows,
        (
            ("leaves", "terminal leaves"),
            ("policy_rows", "policy rows"),
            ("branch_decisions", "branch decisions"),
            ("skipped_by_placement", "placement skips"),
        ),
        smooth=smooth,
        title="Tree and search volume",
        ylabel="count",
    )
    _lines(
        axes[1, 2],
        iteration,
        rows,
        (
            ("spine_entropy", "rollout policy"),
            ("entropy", "updated policy"),
        ),
        smooth=smooth,
        title="Policy entropy",
        ylabel="nats",
    )
    _dual_lines(
        axes[2, 0],
        iteration,
        rows,
        left=(("positions_per_sec", "positions/s"),),
        right=(("forward_rows_per_sec", "forward rows/s"),),
        smooth=smooth,
        title="Throughput",
        left_label="positions/s",
        right_label="forward rows/s",
    )
    _lines(
        axes[2, 1],
        iteration,
        rows,
        (
            ("collect_sec", "collect"),
            ("update_sec", "update"),
            ("forward_sec", "forward within collect"),
        ),
        smooth=smooth,
        title="Wall time",
        ylabel="seconds",
    )
    _dual_lines(
        axes[2, 2],
        iteration,
        rows,
        left=(
            ("peak_cache_rows", "rows used"),
            ("cache_rows_allocated", "rows reserved"),
        ),
        right=(("peak_device_gb", "device high-water"),),
        smooth=1,
        title="Memory (raw, unsmoothed)",
        left_label="KV-cache rows",
        right_label="device GB",
    )
    _mark_cache_cap_hits(axes[2, 2], iteration, rows)
    _mark_policy_events(axes[0, 2], iteration, rows)

    for ax in axes.flat:
        ax.set_xlabel("Iteration")
        ax.grid(alpha=0.18)

    last = rows[-1]
    backtracks = int(sum(_finite(_series(rows, "backtracks"))))
    rollbacks = int(sum(_finite(_series(rows, "rolled_back"))))
    fig.text(
        0.5,
        0.012,
        " · ".join(
            (
                f"iteration {int(_number(last, 'iteration'))}",
                f"optimizer steps {int(_number(last, 'optimizer_steps'))}",
                f"LR {_number(last, 'learning_rate'):.2e}",
                f"step scale {_number(last, 'step_scale'):.3f}",
                f"backtracks {backtracks}",
                f"rollbacks {rollbacks}",
                f"elapsed {_duration(_number(last, 'elapsed_sec'))}",
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


def _has_signal(values: np.ndarray, *, omit_zero: bool) -> bool:
    finite = _finite(values)
    if not finite.size:
        return False
    return not (omit_zero and np.allclose(finite, 0.0))


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _plot_fields(
    ax,
    iteration: np.ndarray,
    rows: list[dict[str, str]],
    fields: tuple[Field, ...],
    *,
    smooth: int,
    omit_zero: bool,
    color_offset: int = 0,
) -> list:
    handles = []
    for index, (field, label) in enumerate(fields):
        values = _series(rows, field)
        if not _has_signal(values, omit_zero=omit_zero):
            continue
        rendered = _smooth(values, smooth)
        valid = np.isfinite(iteration) & np.isfinite(rendered)
        if not valid.any():
            continue
        (line,) = ax.plot(
            iteration[valid],
            rendered[valid],
            linewidth=1.7,
            color=f"C{color_offset + index}",
            label=label,
        )
        handles.append(line)
    return handles


def _lines(
    ax,
    iteration: np.ndarray,
    rows: list[dict[str, str]],
    fields: tuple[Field, ...],
    *,
    smooth: int,
    title: str,
    ylabel: str | None = None,
    omit_zero: bool = False,
) -> None:
    handles = _plot_fields(
        ax,
        iteration,
        rows,
        fields,
        smooth=smooth,
        omit_zero=omit_zero,
    )
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    if handles:
        ax.legend(handles=handles, fontsize=8, loc="best")
    else:
        _no_data(ax)


def _dual_lines(
    ax,
    iteration: np.ndarray,
    rows: list[dict[str, str]],
    *,
    left: tuple[Field, ...],
    right: tuple[Field, ...],
    smooth: int,
    title: str,
    left_label: str,
    right_label: str,
    right_smooth: int | None = None,
    omit_zero: bool = False,
):
    right_ax = ax.twinx()
    left_handles = _plot_fields(
        ax,
        iteration,
        rows,
        left,
        smooth=smooth,
        omit_zero=omit_zero,
    )
    right_handles = _plot_fields(
        right_ax,
        iteration,
        rows,
        right,
        smooth=smooth if right_smooth is None else right_smooth,
        omit_zero=omit_zero,
        color_offset=len(left),
    )
    ax.set_title(title)
    if left_handles:
        ax.set_ylabel(left_label)
    if right_handles:
        right_ax.set_ylabel(right_label)
        right_ax.grid(False)
    else:
        right_ax.set_yticks([])
        right_ax.spines["right"].set_visible(False)
    handles = [*left_handles, *right_handles]
    if handles:
        ax.legend(handles=handles, fontsize=8, loc="best")
    else:
        _no_data(ax)
    return right_ax


def _no_data(ax) -> None:
    ax.text(
        0.5,
        0.5,
        "No data yet",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="#777777",
    )


def _mark_cache_cap_hits(
    ax,
    iteration: np.ndarray,
    rows: list[dict[str, str]],
) -> None:
    blocked = _series(rows, "blocked_by_cache")
    used = _series(rows, "peak_cache_rows")
    valid = (
        np.isfinite(iteration)
        & np.isfinite(blocked)
        & np.isfinite(used)
        & (blocked > 0)
    )
    if not valid.any():
        return
    ax.scatter(
        iteration[valid],
        used[valid],
        marker="x",
        s=42,
        linewidths=1.5,
        color="#d62728",
        zorder=5,
    )
    first = int(np.flatnonzero(valid)[0])
    ax.annotate(
        f"cap hit · {int(blocked[first]):,} branches blocked",
        (iteration[first], used[first]),
        xytext=(7, -15),
        textcoords="offset points",
        fontsize=7,
        color="#b22222",
    )


def _mark_policy_events(
    ax,
    iteration: np.ndarray,
    rows: list[dict[str, str]],
) -> None:
    kl = _series(rows, "policy_kl")
    backtracks = _series(rows, "backtracks")
    rollbacks = _series(rows, "rolled_back")
    finite = np.isfinite(iteration) & np.isfinite(kl)
    for values, marker, color in (
        (backtracks, "v", "#ff7f0e"),
        (rollbacks, "x", "#d62728"),
    ):
        active = finite & np.isfinite(values) & (values > 0)
        if active.any():
            ax.scatter(
                iteration[active],
                kl[active],
                marker=marker,
                s=38,
                color=color,
                zorder=5,
            )


def _number(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    try:
        return float(raw) if raw not in ("", None) else 0.0
    except ValueError:
        return 0.0


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"
