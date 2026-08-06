"""Static training dashboard for schema-v6 runs."""

from __future__ import annotations

import csv
import math
import re
import tomllib
import uuid
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plump.run_evaluation import ensure_evaluation_summary

Field = tuple[str, str]
DEFAULT_SMOOTH_WINDOW = 50


def render_dashboard(
    metrics_path: str | Path,
    output_path: str | Path,
    *,
    title: str | None = None,
    smooth: int = DEFAULT_SMOOTH_WINDOW,
    dpi: int = 150,
    include_learning_rate: bool = False,
    evaluations_path: str | Path | None = None,
) -> int:
    """Render per-iteration trends and exact sparse/discrete observations."""

    if smooth < 1:
        raise ValueError("smooth must be at least 1")

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
    fig, axes = plt.subplots(4, 3, figsize=(20, 17), constrained_layout=True)
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
        smooth=smooth,
        include_learning_rate=include_learning_rate,
    )
    _value_and_critic_learning(
        axes[1, 0],
        iteration,
        rows,
        smooth=smooth,
    )
    _lines(
        axes[1, 1],
        iteration,
        rows,
        (
            ("loss_suit", "actor opponent-suit BCE"),
            ("loss_trick", "actor final-trick CE"),
            ("loss_oracle_trick", "oracle final-trick CE · epoch sum"),
        ),
        smooth=smooth,
        title="Belief auxiliary losses",
        ylabel="normalized loss",
        omit_zero=True,
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
    _gradient_norms(
        axes[2, 0],
        iteration,
        rows,
        smooth=smooth,
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
        smooth=smooth,
        title="Memory",
        left_label="KV-cache rows",
        right_label="device GB",
    )
    _mark_cache_cap_hits(axes[2, 2], iteration, rows)
    _mark_policy_rollbacks(axes[0, 2], iteration, rows)
    _lines(
        axes[3, 0],
        iteration,
        rows,
        (
            ("suit_accuracy_10c_0", "0 cards played"),
            ("suit_accuracy_10c_4", "4 cards played"),
            ("suit_accuracy_10c_8", "8 cards played"),
        ),
        smooth=smooth,
        title="Opponent-suit accuracy · 10-card games",
        ylabel="bit accuracy",
        omit_zero=True,
    )
    axes[3, 0].set_ylim(0.0, 1.0)
    _lines(
        axes[3, 1],
        iteration,
        rows,
        (
            ("trick_accuracy_10c_0", "actor · 0 played"),
            ("trick_accuracy_10c_4", "actor · 4 played"),
            ("trick_accuracy_10c_8", "actor · 8 played"),
            ("oracle_trick_accuracy", "oracle · all prefixes"),
        ),
        smooth=smooth,
        title="Final-trick exact accuracy",
        ylabel="seat accuracy",
        omit_zero=True,
    )
    axes[3, 1].set_ylim(0.0, 1.0)
    _evaluation_gap(axes[3, 2], evaluation_points)

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
            f"rolling mean {smooth} iters",
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
) -> list[dict[str, float | str]]:
    """Merge paired inline scores with checkpoint-scoped sidecar reports."""

    points: dict[tuple[str, int], dict[str, float | str]] = {}
    iterations = _series(rows, "iteration")
    explicit_fields = {
        "sample": (
            "eval_reward_vs_heuristic_sample",
            "eval_bid_hit_sample",
        ),
        "argmax": (
            "eval_reward_vs_heuristic_argmax",
            "eval_bid_hit_argmax",
        ),
    }
    for mode, (reward_field, bid_field) in explicit_fields.items():
        rewards = _series(rows, reward_field)
        bids = _series(rows, bid_field)
        for iteration, reward, bid in zip(iterations, rewards, bids):
            if np.isfinite(iteration) and (
                np.isfinite(reward) or np.isfinite(bid)
            ):
                points[(mode, int(iteration))] = {
                    "iteration": float(iteration),
                    "mode": mode,
                    "reward": float(reward),
                    "bid_hit": float(bid),
                    "ci_low": math.nan,
                    "ci_high": math.nan,
                }

    # Before paired evaluation existed, the inline columns represented the
    # configured action mode. Preserve those historical observations, but let
    # explicit paired columns (and then sidecars below) take precedence.
    legacy_mode = _legacy_evaluation_mode(evaluations_path.parent / "config.toml")
    rewards = _series(rows, "eval_reward_vs_heuristic")
    bids = _series(rows, "eval_bid_hit")
    for iteration, reward, bid in zip(iterations, rewards, bids):
        if np.isfinite(iteration) and (np.isfinite(reward) or np.isfinite(bid)):
            points.setdefault(
                (legacy_mode, int(iteration)),
                {
                    "iteration": float(iteration),
                    "mode": legacy_mode,
                    "reward": float(reward),
                    "bid_hit": float(bid),
                    "ci_low": math.nan,
                    "ci_high": math.nan,
                },
            )

    if evaluations_path.is_dir():
        paths = sorted(evaluations_path.glob("iter_*/heuristic.json"))
        paths.extend(
            sorted(evaluations_path.glob("iter_*/heuristic_sample.json"))
        )
        for path in paths:
            try:
                payload = ensure_evaluation_summary(path)
                report = payload.get("report", payload)
                protocol = payload.get("protocol", {})
                greedy = bool(
                    protocol.get("greedy", path.stem != "heuristic_sample")
                )
                mode = "argmax" if greedy else "sample"
                raw_iteration = payload.get("iteration")
                if raw_iteration is None:
                    match = re.fullmatch(r"iter_(\d+)", path.parent.name)
                    if match is None:
                        continue
                    raw_iteration = match.group(1)
                iteration = int(raw_iteration)
                points[(mode, iteration)] = {
                    "iteration": float(iteration),
                    "mode": mode,
                    "reward": float(report["macro_relative_reward"]),
                    "bid_hit": float(report["macro_bid_hit_rate"]),
                    "ci_low": float(report.get("relative_reward_ci_low", math.nan)),
                    "ci_high": float(
                        report.get("relative_reward_ci_high", math.nan)
                    ),
                }
            except (KeyError, TypeError, ValueError, OSError):
                continue
    return [points[key] for key in sorted(points)]


def _legacy_evaluation_mode(config_path: Path) -> str:
    """Infer the meaning of pre-paired inline evaluation columns."""

    try:
        with config_path.open("rb") as handle:
            mode = tomllib.load(handle)["evaluation"]["training_action_mode"]
        return "sample" if mode == "sample" else "argmax"
    except (KeyError, OSError, TypeError, ValueError, tomllib.TOMLDecodeError):
        return "argmax"


def _checkpoint_evaluation(
    ax,
    points: list[dict[str, float | str]],
) -> None:
    right_ax = ax.twinx()
    ax.set_title("Checkpoint evaluation vs heuristic")
    if not points:
        right_ax.set_yticks([])
        right_ax.spines["right"].set_visible(False)
        _no_data(ax)
        return

    handles = []
    any_bid = False
    styles = {
        "argmax": ("C0", "C1", "-", "argmax"),
        "sample": ("C2", "C3", "--", "sample"),
    }
    for mode in ("argmax", "sample"):
        selected = [point for point in points if point.get("mode") == mode]
        if not selected:
            continue
        reward_color, bid_color, line_style, label = styles[mode]
        iteration = np.asarray(
            [point["iteration"] for point in selected],
            dtype=np.float64,
        )
        reward = np.asarray(
            [point["reward"] for point in selected],
            dtype=np.float64,
        )
        bid_hit = np.asarray(
            [point["bid_hit"] for point in selected],
            dtype=np.float64,
        )
        ci_low = np.asarray(
            [point["ci_low"] for point in selected],
            dtype=np.float64,
        )
        ci_high = np.asarray(
            [point["ci_high"] for point in selected],
            dtype=np.float64,
        )

        reward_valid = np.isfinite(iteration) & np.isfinite(reward)
        if reward_valid.any():
            (reward_line,) = ax.plot(
                iteration[reward_valid],
                reward[reward_valid],
                color=reward_color,
                linestyle=line_style,
                linewidth=1.8,
                marker="o",
                label=f"reward · {label}",
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
                    color=reward_color,
                    alpha=0.10,
                    linewidth=0,
                )

        bid_valid = np.isfinite(iteration) & np.isfinite(bid_hit)
        if bid_valid.any():
            (bid_line,) = right_ax.plot(
                iteration[bid_valid],
                bid_hit[bid_valid],
                color=bid_color,
                linestyle=line_style,
                linewidth=1.8,
                marker="s",
                label=f"bid accuracy · {label}",
            )
            handles.append(bid_line)
            any_bid = True

    ax.axhline(0.0, color="#777777", linewidth=0.8, alpha=0.45)
    ax.set_ylabel("relative reward")
    if any_bid:
        right_ax.set_ylabel("bid accuracy")
        right_ax.grid(False)
    else:
        right_ax.set_yticks([])
        right_ax.spines["right"].set_visible(False)
    if handles:
        ax.legend(handles=handles, fontsize=8, loc="best")


def _evaluation_gap(
    ax,
    points: list[dict[str, float | str]],
) -> None:
    """Plot exact argmax-minus-sample reward and bid-accuracy differences."""

    by_key = {
        (str(point["mode"]), int(float(point["iteration"]))): point
        for point in points
    }
    iterations = sorted(
        iteration
        for mode, iteration in by_key
        if mode == "sample" and ("argmax", iteration) in by_key
    )
    right_ax = ax.twinx()
    ax.set_title("Evaluation mode gap · argmax − sample")
    if not iterations:
        right_ax.set_yticks([])
        right_ax.spines["right"].set_visible(False)
        _no_data(ax)
        return
    reward_gap = np.asarray(
        [
            float(by_key[("argmax", iteration)]["reward"])
            - float(by_key[("sample", iteration)]["reward"])
            for iteration in iterations
        ]
    )
    bid_gap = np.asarray(
        [
            float(by_key[("argmax", iteration)]["bid_hit"])
            - float(by_key[("sample", iteration)]["bid_hit"])
            for iteration in iterations
        ]
    )
    x = np.asarray(iterations, dtype=np.float64)
    handles = []
    reward_valid = np.isfinite(reward_gap)
    if reward_valid.any():
        (line,) = ax.plot(
            x[reward_valid],
            reward_gap[reward_valid],
            marker="o",
            color="C0",
            label="relative reward gap",
        )
        handles.append(line)
    bid_valid = np.isfinite(bid_gap)
    if bid_valid.any():
        (line,) = right_ax.plot(
            x[bid_valid],
            bid_gap[bid_valid],
            marker="s",
            color="C1",
            label="bid accuracy gap",
        )
        handles.append(line)
    ax.axhline(0.0, color="#777777", linewidth=0.8, alpha=0.45)
    ax.set_ylabel("reward difference")
    right_ax.set_ylabel("bid-accuracy difference")
    right_ax.grid(False)
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
        if omit_zero:
            # A zero-only prefix conventionally means that this diagnostic or
            # loss was disabled. Do not let those placeholders dilute the
            # rolling mean for its first real observations after reconfigure.
            signal = np.flatnonzero(np.isfinite(values) & ~np.isclose(values, 0.0))
            if signal.size:
                values = values.copy()
                values[: signal[0]] = np.nan
        rendered = _smooth(values, smooth)
        color = f"C{color_offset + index}"
        raw_valid = np.isfinite(iteration) & np.isfinite(values)
        if smooth > 1 and raw_valid.any():
            ax.plot(
                iteration[raw_valid],
                values[raw_valid],
                linewidth=0.65,
                alpha=0.16,
                color=color,
                zorder=1,
            )
        valid = np.isfinite(iteration) & np.isfinite(rendered)
        if not valid.any():
            continue
        (line,) = ax.plot(
            iteration[valid],
            rendered[valid],
            linewidth=1.7,
            color=color,
            label=label,
            zorder=2,
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
        raw_valid = (
            np.isfinite(iteration)
            & np.isfinite(proposed)
            & (proposed > 0)
        )
        if smooth > 1 and raw_valid.any():
            ax.plot(
                iteration[raw_valid],
                proposed[raw_valid],
                linewidth=0.65,
                linestyle="--",
                alpha=0.16,
                color="C3",
                zorder=1,
            )
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
                zorder=2,
            )
            handles.append(line)

    if handles:
        caps = _current_kl_caps(metrics_path)
        if caps is not None:
            mean_cap, p99_cap = caps
            handles.append(
                ax.axhline(
                    mean_cap,
                    linewidth=1.0,
                    linestyle=":",
                    color="C0",
                    alpha=0.8,
                    label=f"current mean cap ({mean_cap:g})",
                )
            )
            if p99_cap is not None:
                handles.append(
                    ax.axhline(
                        p99_cap,
                        linewidth=1.0,
                        linestyle=":",
                        color="C1",
                        alpha=0.8,
                        label=f"current p99 cap ({p99_cap:g})",
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
            smooth=smooth,
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


def _gradient_norms(
    ax,
    iteration: np.ndarray,
    rows: list[dict[str, str]],
    *,
    smooth: int,
) -> None:
    """Show the separately clipped parameter-group norms before clipping."""

    handles = _plot_fields(
        ax,
        iteration,
        rows,
        (
            ("core_grad_norm", "core/shared pre-clip"),
            ("auxiliary_grad_norm", "value/belief heads pre-clip"),
            ("critic_grad_norm", "PPO critic pre-clip"),
        ),
        smooth=smooth,
        omit_zero=True,
    )
    ax.set_title("Pre-clip gradient norms")
    if handles:
        ax.set_yscale("log")
        ax.set_ylabel("L2 norm (log scale)")
        ax.axhline(
            1.0,
            linewidth=1.0,
            linestyle=":",
            color="#666666",
            alpha=0.8,
            label="clip threshold (1.0)",
        )
        ax.legend(handles=[*handles, ax.lines[-1]], fontsize=8, loc="best")
    else:
        _no_data(ax)


def _value_and_critic_learning(
    ax,
    iteration: np.ndarray,
    rows: list[dict[str, str]],
    *,
    smooth: int,
) -> None:
    """Use the value panel for oracle dynamics when PPO telemetry exists."""

    oracle_signal = _has_signal(
        _series(rows, "critic_all_player_rmse"), omit_zero=True
    )
    if oracle_signal:
        _dual_lines(
            ax,
            iteration,
            rows,
            left=(
                ("value_rmse", "acting-seat RMSE"),
                ("critic_all_player_rmse", "all-seat RMSE"),
                ("value_zero_rmse", "zero-baseline RMSE"),
            ),
            right=(
                ("value_correlation", "acting-seat correlation"),
                (
                    "critic_all_player_correlation",
                    "all-seat correlation",
                ),
                ("critic_loss_reduction", "within-update loss reduction"),
            ),
            smooth=smooth,
            title="Oracle critic dynamics",
            left_label="RMSE (reward points)",
            right_label="correlation / fractional reduction",
            omit_zero=True,
        )
        return

    _dual_lines(
        ax,
        iteration,
        rows,
        left=(("value_rmse", "value RMSE"),),
        right=(
            ("value_correlation", "value correlation"),
            ("loss_suit", "suit-presence loss"),
            ("loss_trick", "trick-count loss"),
        ),
        smooth=smooth,
        title="Value and belief learning",
        left_label="value RMSE (reward points)",
        right_label="correlation / belief loss",
        omit_zero=True,
    )


def _current_kl_caps(metrics_path: Path) -> tuple[float, float | None] | None:
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
    if mean_cap <= 0 or p99_cap < 0:
        return None
    return mean_cap, p99_cap if p99_cap > 0 else None


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


def _mark_policy_rollbacks(
    ax,
    iteration: np.ndarray,
    rows: list[dict[str, str]],
) -> None:
    kl = _series(rows, "policy_kl")
    rollbacks = _series(rows, "rolled_back")
    finite = np.isfinite(iteration) & np.isfinite(kl)
    active = finite & np.isfinite(rollbacks) & (rollbacks > 0)
    if active.any():
        ax.scatter(
            iteration[active],
            kl[active],
            marker="x",
            s=38,
            color="#d62728",
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
