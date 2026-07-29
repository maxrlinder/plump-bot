"""Static training dashboard for schema-v6 runs."""

from __future__ import annotations

import csv
import json
import math
import re
import tomllib
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
    include_learning_rate: bool = False,
    evaluations_path: str | Path | None = None,
) -> int:
    """Render comparable quantities together and discrete events without smoothing."""

    metrics_path = Path(metrics_path)
    output_path = Path(output_path)
    rows = _read_rows(metrics_path)
    if not rows:
        raise ValueError(f"No metric rows in {metrics_path}")

    iteration = _series(rows, "iteration")
    evaluation_points = _evaluation_points(
        rows,
        Path(evaluations_path)
        if evaluations_path is not None
        else metrics_path.parent / "evaluations",
    )
    fig, axes = plt.subplots(3, 3, figsize=(20, 13), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0.0, 0.035, 1.0, 0.96))
    fig.suptitle(
        title or f"Plump schema-v6 · {metrics_path.parent.name}",
        fontsize=16,
    )

    _checkpoint_evaluation(
        axes[0, 0],
        evaluation_points,
    )
    _dual_lines(
        axes[0, 1],
        iteration,
        rows,
        left=(
            ("reward_focal", "focal reward"),
            ("reward_non_focal", "non-focal reward"),
        ),
        right=(
            ("bid_hit_focal", "focal bid accuracy"),
            ("bid_hit_non_focal", "non-focal bid accuracy"),
        ),
        smooth=smooth,
        title="Observed outcomes",
        left_label="relative reward",
        right_label="bid accuracy",
        omit_zero=True,
    )
    _trust_region(
        axes[0, 2],
        iteration,
        rows,
        metrics_path=metrics_path,
        smooth=min(smooth, 5),
        include_learning_rate=include_learning_rate,
    )
    _lines(
        axes[1, 0],
        iteration,
        rows,
        (
            ("loss_value", "value"),
            ("loss_value_zero", "value: predict zero"),
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
    footer = [
        f"iteration {int(_number(last, 'iteration'))}",
        f"optimizer steps {int(_number(last, 'optimizer_steps'))}",
    ]
    if include_learning_rate:
        footer.append(f"LR {_number(last, 'learning_rate'):.2e}")
    footer.extend(
        (
            f"step scale {_number(last, 'step_scale'):.3f}",
            f"backtracks {backtracks}",
            f"rollbacks {rollbacks}",
            f"elapsed {_duration(_number(last, 'elapsed_sec'))}",
        )
    )
    fig.text(
        0.5,
        0.012,
        " · ".join(footer),
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


def _evaluation_points(
    rows: list[dict[str, str]],
    evaluations_path: Path,
) -> list[dict[str, float]]:
    """Merge inline legacy scores with checkpoint-scoped sidecar reports."""

    points: dict[int, dict[str, float]] = {}
    iterations = _series(rows, "iteration")
    rewards = _series(rows, "eval_reward_vs_heuristic")
    bids = _series(rows, "eval_bid_hit")
    for iteration, reward, bid in zip(iterations, rewards, bids):
        if np.isfinite(iteration) and (np.isfinite(reward) or np.isfinite(bid)):
            points[int(iteration)] = {
                "iteration": float(iteration),
                "reward": float(reward),
                "bid_hit": float(bid),
                "ci_low": math.nan,
                "ci_high": math.nan,
            }

    if evaluations_path.is_dir():
        for path in sorted(evaluations_path.glob("iter_*/heuristic.json")):
            try:
                payload = json.loads(path.read_text())
                report = payload.get("report", payload)
                raw_iteration = payload.get("iteration")
                if raw_iteration is None:
                    match = re.fullmatch(r"iter_(\d+)", path.parent.name)
                    if match is None:
                        continue
                    raw_iteration = match.group(1)
                iteration = int(raw_iteration)
                points[iteration] = {
                    "iteration": float(iteration),
                    "reward": float(report["macro_relative_reward"]),
                    "bid_hit": float(report["macro_bid_hit_rate"]),
                    "ci_low": float(report.get("relative_reward_ci_low", math.nan)),
                    "ci_high": float(
                        report.get("relative_reward_ci_high", math.nan)
                    ),
                }
            except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
                continue
    return [points[key] for key in sorted(points)]


def _checkpoint_evaluation(ax, points: list[dict[str, float]]) -> None:
    right_ax = ax.twinx()
    ax.set_title("Checkpoint evaluation vs heuristic")
    if not points:
        right_ax.set_yticks([])
        right_ax.spines["right"].set_visible(False)
        _no_data(ax)
        return

    iteration = np.asarray(
        [point["iteration"] for point in points],
        dtype=np.float64,
    )
    reward = np.asarray([point["reward"] for point in points], dtype=np.float64)
    bid_hit = np.asarray([point["bid_hit"] for point in points], dtype=np.float64)
    ci_low = np.asarray([point["ci_low"] for point in points], dtype=np.float64)
    ci_high = np.asarray([point["ci_high"] for point in points], dtype=np.float64)

    handles = []
    reward_valid = np.isfinite(iteration) & np.isfinite(reward)
    if reward_valid.any():
        (reward_line,) = ax.plot(
            iteration[reward_valid],
            reward[reward_valid],
            color="C0",
            linewidth=1.8,
            marker="o",
            label="relative reward",
        )
        handles.append(reward_line)
        ci_valid = (
            reward_valid
            & np.isfinite(ci_low)
            & np.isfinite(ci_high)
        )
        if ci_valid.any():
            ax.fill_between(
                iteration[ci_valid],
                ci_low[ci_valid],
                ci_high[ci_valid],
                color="C0",
                alpha=0.12,
                linewidth=0,
            )
        ax.axhline(0.0, color="#777777", linewidth=0.8, alpha=0.45)
        ax.set_ylabel("relative reward")

    bid_valid = np.isfinite(iteration) & np.isfinite(bid_hit)
    if bid_valid.any():
        (bid_line,) = right_ax.plot(
            iteration[bid_valid],
            bid_hit[bid_valid],
            color="C1",
            linewidth=1.8,
            marker="s",
            label="bid accuracy",
        )
        handles.append(bid_line)
        right_ax.set_ylabel("bid accuracy")
        right_ax.grid(False)
    else:
        right_ax.set_yticks([])
        right_ax.spines["right"].set_visible(False)
    if handles:
        ax.legend(handles=handles, fontsize=8, loc="best")


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


def _trust_region(
    ax,
    iteration: np.ndarray,
    rows: list[dict[str, str]],
    *,
    metrics_path: Path,
    smooth: int,
    include_learning_rate: bool,
) -> None:
    """Show the nominal proposal separately from the accepted guarded step."""

    accepted = (
        ("policy_kl", "accepted mean KL"),
        ("policy_kl_p99", "accepted p99 KL"),
        ("policy_kl_max", "accepted max KL"),
    )
    handles = _plot_fields(
        ax,
        iteration,
        rows,
        accepted,
        smooth=smooth,
        omit_zero=True,
    )
    proposed = _series(rows, "proposed_policy_kl")
    proposed_signal = _has_signal(proposed, omit_zero=True)
    if proposed_signal:
        rendered = _smooth(proposed, smooth)
        valid = (
            np.isfinite(iteration)
            & np.isfinite(rendered)
            & (rendered > 0)
        )
        if valid.any():
            (line,) = ax.plot(
                iteration[valid],
                rendered[valid],
                linewidth=1.8,
                linestyle="--",
                color="C3",
                label="proposed mean KL (before backtracking)",
            )
            handles.append(line)

    if handles:
        caps = _current_kl_caps(metrics_path)
        if caps is not None:
            mean_cap, p99_cap = caps
            handles.extend(
                (
                    ax.axhline(
                        mean_cap,
                        linewidth=1.0,
                        linestyle=":",
                        color="C0",
                        alpha=0.8,
                        label=f"current mean cap ({mean_cap:g})",
                    ),
                    ax.axhline(
                        p99_cap,
                        linewidth=1.0,
                        linestyle=":",
                        color="C1",
                        alpha=0.8,
                        label=f"current p99 cap ({p99_cap:g})",
                    ),
                )
            )
        # Nominal proposals can be orders of magnitude above accepted KL, and
        # even accepted mean/p99/max differ materially in scale. Keep this
        # panel logarithmic with or without proposal telemetry.
        ax.set_yscale("log")

    if include_learning_rate:
        right_ax = ax.twinx()
        lr_handles = _plot_fields(
            right_ax,
            iteration,
            rows,
            (("learning_rate", "learning rate"),),
            smooth=1,
            omit_zero=True,
            color_offset=4,
        )
        handles.extend(lr_handles)
        if lr_handles:
            right_ax.set_ylabel("learning rate")
            right_ax.grid(False)
        else:
            right_ax.set_yticks([])
            right_ax.spines["right"].set_visible(False)

    ax.set_title("Policy trust region")
    if handles:
        ax.set_ylabel("old → new KL")
        ax.legend(handles=handles, fontsize=7, loc="best")
    else:
        _no_data(ax)


def _current_kl_caps(metrics_path: Path) -> tuple[float, float] | None:
    """Read current run caps for dashboard reference lines, if available."""

    config_path = metrics_path.parent / "config.toml"
    if not config_path.is_file():
        return None
    try:
        with config_path.open("rb") as handle:
            training = tomllib.load(handle)["training"]
        mean_cap = float(training["policy_kl_cap"])
        p99_cap = float(training["policy_kl_p99_cap"])
    except (KeyError, OSError, TypeError, ValueError, tomllib.TOMLDecodeError):
        return None
    if mean_cap <= 0 or p99_cap <= 0:
        return None
    return mean_cap, p99_cap


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
