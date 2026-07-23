"""Plot key metrics for one Plump PPO training run.

Eight panels, two rows. Row 1 is the game: strength vs the frozen pool, bid
quality, and the belief heads, all measured on clean (non-explore) rounds.
Row 2 is the machine: sharpness, value fit, PPO trust-region health, and
throughput. The suit-belief and value-fit panels also show their corresponding
training losses on a right-hand axis so fit quality and optimization pressure
can be read together.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a metrics dashboard for a Plump training run.")
    parser.add_argument("run_dir", type=Path, help="Training run directory containing metrics.csv.")
    parser.add_argument("--output", type=Path, default=None, help="Output image path. Defaults to RUN_DIR/metrics.png.")
    parser.add_argument(
        "--smooth",
        type=int,
        default=50,
        help="Rolling window for non-evaluation metrics captured every iteration.",
    )
    parser.add_argument(
        "--diagnostic-smooth",
        type=int,
        default=50,
        help="Rolling window over sparse non-evaluation diagnostic observations.",
    )
    parser.add_argument(
        "--bid-smooth",
        type=int,
        default=150,
        help="Rolling iteration window for the noisier bid-hit-rate panel.",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--min-iteration",
        type=int,
        default=None,
        help="Drop rows below this iteration before plotting.",
    )
    parser.add_argument(
        "--since-restart",
        action="store_true",
        help=(
            "Plot only from the clean/explore soft restart, detected as the "
            "first row carrying the explore-arm columns."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = args.run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise SystemExit(f"No metrics.csv found at {metrics_path}")

    output_path = args.output or (args.run_dir / "metrics.png")
    rows = render_metrics_plot(
        metrics_path=metrics_path,
        output_path=output_path,
        smooth=args.smooth,
        diagnostic_smooth=args.diagnostic_smooth,
        bid_smooth=args.bid_smooth,
        dpi=args.dpi,
        title=f"Plump PPO Training: {args.run_dir.name}",
        min_iteration=args.min_iteration,
        since_restart=args.since_restart,
    )
    if rows is None:
        raise SystemExit(f"{metrics_path} has no plottable metric rows.")

    last_iteration = int(rows[-1].get("iteration", 0))
    print(f"wrote {output_path}")
    print(f"rows={len(rows)} last_iteration={last_iteration}")
    _print_latest_metrics(rows)


def render_metrics_plot(
    *,
    metrics_path: Path,
    output_path: Path,
    smooth: int = 50,
    diagnostic_smooth: int = 50,
    bid_smooth: int = 150,
    dpi: int = 150,
    title: str | None = None,
    min_iteration: int | None = None,
    since_restart: bool = False,
) -> list[dict[str, float | str]] | None:
    """Render the metrics dashboard; returns the parsed rows (None if empty)."""

    rows = _read_metrics(metrics_path)
    if since_restart:
        restart = _restart_iteration(rows)
        if restart is not None:
            min_iteration = max(min_iteration or 0, restart)
    if min_iteration is not None:
        rows = [
            row
            for row in rows
            if isinstance(row.get("iteration"), float)
            and not math.isnan(row["iteration"])
            and int(row["iteration"]) >= min_iteration
        ]
    if not rows:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), constrained_layout=True)
    resolved_title = title or f"Plump PPO Training: {metrics_path.parent.name}"
    if min_iteration is not None:
        visible_iterations = [
            int(row["iteration"])
            for row in rows
            if isinstance(row.get("iteration"), float)
            and not math.isnan(row["iteration"])
        ]
        first_iteration = min(visible_iterations, default=min_iteration)
        last_iteration = max(visible_iterations, default=min_iteration)
        resolved_title += (
            f" — iterations {first_iteration:,}–{last_iteration:,} (live)"
        )
    fig.suptitle(resolved_title, fontsize=16)

    x = _series(rows, "iteration")
    smooth = max(smooth, 1)
    diagnostic_smooth = max(diagnostic_smooth, 1)
    bid_smooth = max(bid_smooth, 1)

    # Row 1: the game — strength, bids, and beliefs, clean rounds only.
    _plot_reward_vs_pool(axes[0][0], rows, x, smooth)
    _plot_bid_quality(axes[0][1], rows, x, bid_smooth)
    _plot_trick_belief(axes[0][2], rows, diagnostic_smooth)
    _plot_suit_belief(axes[0][3], rows, diagnostic_smooth)

    # Row 2: the machine — sharpness, value fit, trust region, speed.
    _plot_policy_sharpness(axes[1][0], rows, diagnostic_smooth)
    _plot_explained_variance(axes[1][1], rows, diagnostic_smooth)
    _plot_ppo_stability(axes[1][2], rows, x, smooth)
    _plot_throughput(axes[1][3], rows, x, smooth)

    # One shared x-range everywhere: series that only exist on recent rows
    # (new columns, sparse diagnostics) must not zoom their panel in.
    finite_x = [value for value in x if not math.isnan(value)]
    x_min = min(finite_x, default=0.0)
    x_max = max(finite_x, default=1.0)
    margin = max(x_max - x_min, 1.0) * 0.01
    for ax in axes.flat:
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.grid(alpha=0.25)
        ax.set_xlabel("iteration")

    temporary_path = output_path.with_name(
        f".{output_path.stem}.tmp{output_path.suffix}"
    )
    fig.savefig(temporary_path, dpi=dpi)
    plt.close(fig)
    temporary_path.replace(output_path)
    return rows


def _read_metrics(metrics_path: Path) -> list[dict[str, float | str]]:
    with metrics_path.open(newline="") as file:
        reader = csv.DictReader(file)
        parsed = [{key: _parse_value(value) for key, value in row.items()} for row in reader]

    # A resumed run can append lower iteration numbers after unsaved rows from
    # an abandoned trajectory. Roll back those later rows before adding the
    # resumed branch.
    by_iteration: dict[int, dict[str, float | str]] = {}
    previous_iteration = -1
    for row in parsed:
        iteration = row.get("iteration", math.nan)
        if isinstance(iteration, float) and not math.isnan(iteration):
            current_iteration = int(iteration)
            if current_iteration <= previous_iteration:
                by_iteration = {
                    saved_iteration: saved_row
                    for saved_iteration, saved_row in by_iteration.items()
                    if saved_iteration < current_iteration
                }
            by_iteration[current_iteration] = row
            previous_iteration = current_iteration
    return [by_iteration[iteration] for iteration in sorted(by_iteration)]


def _restart_iteration(rows: list[dict[str, float | str]]) -> int | None:
    """First iteration of the clean/explore regime (explore columns present)."""

    for row in rows:
        marker = row.get("rollout_explore_self_rounds")
        iteration = row.get("iteration")
        if (
            isinstance(marker, float)
            and not math.isnan(marker)
            and isinstance(iteration, float)
            and not math.isnan(iteration)
        ):
            return int(iteration)
    return None


def _parse_value(value: str) -> float | str:
    if value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return value


def _series(rows: list[dict[str, float | str]], name: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(name, math.nan)
        values.append(float(value) if isinstance(value, float) else math.nan)
    return values


def _smooth(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    result: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        chunk = [value for value in values[start : index + 1] if not math.isnan(value)]
        result.append(sum(chunk) / len(chunk) if chunk else math.nan)
    return result


def _has_data(values: list[float]) -> bool:
    return any(not math.isnan(value) for value in values)


def _diagnostic_series(
    rows: list[dict[str, float | str]],
    name: str,
    smooth: int,
) -> tuple[list[float], list[float]]:
    """Return only rows where a sparse diagnostic was actually measured."""

    points = [
        (float(row["iteration"]), float(row[name]))
        for row in rows
        if isinstance(row.get("iteration"), float)
        and not math.isnan(float(row["iteration"]))
        and isinstance(row.get(name), float)
        and not math.isnan(float(row[name]))
    ]
    if not points:
        return [], []
    iterations, values = zip(*points)
    return list(iterations), _smooth(list(values), smooth)


def _plot_diagnostic_lines(
    ax,
    rows: list[dict[str, float | str]],
    smooth: int,
    series: tuple[tuple[str, str, str, str], ...],
) -> bool:
    """Shared renderer for sparse diagnostics; dots keep lone points visible."""

    plotted = False
    for name, label, color, style in series:
        diagnostic_x, values = _diagnostic_series(rows, name, smooth)
        if not values:
            continue
        ax.plot(
            diagnostic_x,
            values,
            label=label,
            color=color,
            linestyle=style,
            marker=".",
            markersize=3,
            alpha=0.85,
        )
        plotted = True
    return plotted


def _no_data(ax) -> None:
    ax.text(0.5, 0.5, "no data yet", ha="center", va="center", transform=ax.transAxes)


def _plot_reward_vs_pool(
    ax,
    rows: list[dict[str, float | str]],
    x: list[float],
    smooth: int,
) -> None:
    """Focal reward vs the frozen checkpoint pool — the live strength gauge.

    Only clean-focal rounds count, so the series starts at the clean/explore
    soft restart: earlier historical rounds played a deliberately noised
    focal seat and their depressed levels are not comparable. Zero is pool
    parity; the pool resamples from the whole run's checkpoints, so staying
    above zero means the newest weights keep beating their own history.
    """

    historical = _series(rows, "rollout_historical_relative_reward")
    historical_rounds = _series(rows, "rollout_historical_rounds")
    era = _series(rows, "rollout_explore_self_rounds")
    values = [
        value
        if not math.isnan(era_marker)
        and not math.isnan(count)
        and count > 0
        else math.nan
        for value, count, era_marker in zip(historical, historical_rounds, era)
    ]
    if _has_data(values):
        ax.plot(
            x,
            _smooth(values, smooth),
            color="tab:blue",
            linewidth=2,
            marker=".",
            markersize=3,
            label="vs frozen league pool",
        )
    else:
        _no_data(ax)
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    ax.set_title("Reward vs Frozen Pool (clean, soft restart)")
    ax.set_ylabel("focal relative score (pts/round)")
    ax.legend(fontsize=8, loc="lower right")


def _plot_bid_quality(
    ax,
    rows: list[dict[str, float | str]],
    x: list[float],
    smooth: int,
) -> None:
    """Focal bid-hit rate, overall and by hand size, clean rounds only.

    Hand size is the difficulty axis: big hands are the unsolved part of the
    game, and their line closing the gap on the small-hand line is the win.
    """

    plotted = False
    for name, label, color, width in (
        ("rollout_bid_hit_rate", "overall", "black", 2.0),
        ("rollout_bid_hit_h3_5", "3-5 cards", "tab:blue", 1.4),
        ("rollout_bid_hit_h6_8", "6-8 cards", "tab:orange", 1.4),
        ("rollout_bid_hit_h9_10", "9-10 cards", "tab:green", 1.4),
    ):
        values = _series(rows, name)
        if not _has_data(values):
            continue
        ax.plot(x, _smooth(values, smooth), label=label, color=color, linewidth=width)
        plotted = True
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Bid Hit Rate ({smooth}-iteration rolling, clean arms)")
    ax.set_ylabel("focal bid hit rate")
    if plotted:
        ax.legend(fontsize=8, loc="upper left")
    else:
        _no_data(ax)


def _plot_trick_belief(
    ax,
    rows: list[dict[str, float | str]],
    smooth: int,
) -> None:
    """Trick-count belief accuracy by decision stage (clean-play states).

    Bid-time is the pure-inference case; the play stages should stack
    strictly upward as cards reveal information.
    """

    accuracy_plotted = _plot_diagnostic_lines(
        ax,
        rows,
        smooth,
        (
            ("pred_trick_count_accuracy", "overall", "black", "-"),
            ("pred_trick_accuracy_bidtime", "bid-time", "tab:red", "-"),
            ("pred_trick_accuracy_early", "early play", "tab:orange", "-"),
            ("pred_trick_accuracy_mid", "mid play", "tab:blue", "-"),
            ("pred_trick_accuracy_late", "late play", "tab:green", "-"),
        ),
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Trick Belief by Stage")
    ax.set_ylabel("trick-count accuracy")

    loss_ax = ax.twinx()
    loss_x, loss = _diagnostic_series(rows, "loss_trick", smooth)
    loss_plotted = bool(loss)
    if loss_plotted:
        loss_ax.plot(
            loss_x,
            loss,
            label="training loss",
            color="tab:purple",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
        )
        loss_ax.set_ylim(bottom=0.0)
    loss_ax.set_ylabel("trick-belief loss", color="tab:purple")
    loss_ax.tick_params(axis="y", colors="tab:purple")

    if accuracy_plotted or loss_plotted:
        handles, labels = ax.get_legend_handles_labels()
        loss_handles, loss_labels = loss_ax.get_legend_handles_labels()
        ax.legend(
            [*handles, *loss_handles],
            [*labels, *loss_labels],
            fontsize=8,
            loc="upper left",
        )
    else:
        _no_data(ax)


def _plot_suit_belief(
    ax,
    rows: list[dict[str, float | str]],
    smooth: int,
) -> None:
    """Suit-presence (void-tracking) accuracy by round stage.

    The trained hidden-card signal: which suits each opponent still holds.
    Later stages should sit higher as follows/discards reveal voids. (The
    card-owner head is retired — it never beat chance on opponent cards.)"""

    accuracy_plotted = _plot_diagnostic_lines(
        ax,
        rows,
        smooth,
        (
            ("pred_suit_presence_accuracy", "overall", "black", "-"),
            ("pred_suit_presence_accuracy_early", "early", "tab:red", "-"),
            ("pred_suit_presence_accuracy_mid", "mid", "tab:orange", "-"),
            ("pred_suit_presence_accuracy_late", "late", "tab:green", "-"),
        ),
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Suit Belief by Stage")
    ax.set_ylabel("suit-presence accuracy")

    loss_ax = ax.twinx()
    loss_x, loss = _diagnostic_series(rows, "loss_suit_presence", smooth)
    loss_plotted = bool(loss)
    if loss_plotted:
        loss_ax.plot(
            loss_x,
            loss,
            label="training loss",
            color="tab:purple",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
        )
        loss_ax.set_ylim(bottom=0.0)
    loss_ax.set_ylabel("suit-presence loss", color="tab:purple")
    loss_ax.tick_params(axis="y", colors="tab:purple")

    if accuracy_plotted or loss_plotted:
        handles, labels = ax.get_legend_handles_labels()
        loss_handles, loss_labels = loss_ax.get_legend_handles_labels()
        ax.legend(
            [*handles, *loss_handles],
            [*labels, *loss_labels],
            fontsize=8,
            loc="lower right",
        )
    else:
        _no_data(ax)


def _plot_policy_sharpness(
    ax,
    rows: list[dict[str, float | str]],
    smooth: int,
) -> None:
    """Entropy of the raw policy on clean-play states.

    Decay alongside rising bid quality is healthy sharpening. The 0.15 hline
    is the standing tripwire: below it, reintroduce ~0.002 entropy bonus
    (behavior diversity is otherwise guaranteed externally by the explore
    arms, not by policy entropy)."""

    plotted = _plot_diagnostic_lines(
        ax,
        rows,
        smooth,
        (
            ("pred_bid_entropy", "bid entropy", "tab:blue", "-"),
            ("pred_play_entropy", "play entropy", "tab:orange", "-"),
        ),
    )
    ax.axhline(
        0.15,
        color="tab:red",
        linewidth=1,
        linestyle=":",
        alpha=0.8,
        label="entropy-bonus tripwire",
    )
    ax.set_ylim(bottom=0.0)
    ax.set_title("Policy Sharpness (clean states)")
    ax.set_ylabel("entropy (nats)")
    if plotted:
        ax.legend(fontsize=8, loc="upper left")
    else:
        _no_data(ax)


def _plot_explained_variance(
    ax,
    rows: list[dict[str, float | str]],
    smooth: int,
) -> None:
    """How much of the return the critics explain — climb, then hold."""

    variance_plotted = _plot_diagnostic_lines(
        ax,
        rows,
        smooth,
        (
            ("pred_value_explained_variance", "value head", "tab:green", "-"),
            (
                "pred_oracle_value_explained_variance",
                "oracle critic",
                "tab:red",
                "-",
            ),
        ),
    )
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.4)
    ax.set_ylim(-0.1, 1.0)
    ax.set_title("Value / Critic Fit")
    ax.set_ylabel("explained variance")

    loss_ax = ax.twinx()
    loss_plotted = False
    for name, label, color in (
        ("loss_value", "value-head loss", "tab:green"),
        ("loss_oracle_value", "oracle-critic loss", "tab:red"),
    ):
        loss_x, loss = _diagnostic_series(rows, name, smooth)
        if not loss:
            continue
        loss_ax.plot(
            loss_x,
            loss,
            label=label,
            color=color,
            linestyle="--",
            linewidth=1.4,
            alpha=0.65,
        )
        loss_plotted = True
    if loss_plotted:
        loss_ax.set_ylim(bottom=0.0)
    loss_ax.set_ylabel("training loss")

    if variance_plotted or loss_plotted:
        handles, labels = ax.get_legend_handles_labels()
        loss_handles, loss_labels = loss_ax.get_legend_handles_labels()
        ax.legend(
            [*handles, *loss_handles],
            [*labels, *loss_labels],
            fontsize=8,
            loc="lower right",
        )
    else:
        _no_data(ax)


def _plot_ppo_stability(
    ax,
    rows: list[dict[str, float | str]],
    x: list[float],
    smooth: int,
) -> None:
    """Trust-region health on one log axis: KL per update and the fraction of
    samples at the clip boundary. Level shifts matter, decades don't."""

    plotted = False
    for name, label, color in (
        ("approx_kl", "approx KL", "tab:blue"),
        ("clip_fraction", "clip fraction", "tab:orange"),
    ):
        values = [
            value if not math.isnan(value) and value > 0.0 else math.nan
            for value in _smooth(_series(rows, name), smooth)
        ]
        if not _has_data(values):
            continue
        ax.plot(x, values, label=label, color=color)
        plotted = True
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(
        LogLocator(base=10.0, subs=(1.0, 2.0, 5.0))
    )
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{value:.3g}")
    )
    ax.set_title("PPO Stability")
    ax.set_ylabel("per-update level (log)")
    if plotted:
        ax.legend(fontsize=8, loc="lower left")
    else:
        _no_data(ax)


def _plot_throughput(
    ax,
    rows: list[dict[str, float | str]],
    x: list[float],
    smooth: int,
) -> None:
    """Training samples per wall-clock second; drops mean thermal or perf
    regressions, steps mean batch-composition or code changes."""

    samples = _series(rows, "samples")
    seconds = _series(rows, "iteration_sec")
    rate = [
        sample_count / duration
        if not math.isnan(sample_count)
        and not math.isnan(duration)
        and duration > 0
        else math.nan
        for sample_count, duration in zip(samples, seconds)
    ]
    if _has_data(rate):
        ax.plot(x, _smooth(rate, smooth), color="tab:blue", linewidth=2)
        ax.set_ylim(bottom=0.0)
    else:
        _no_data(ax)
    ax.set_title("Throughput")
    ax.set_ylabel("training samples / second")





def _print_latest_metrics(rows: list[dict[str, float | str]]) -> None:
    latest = rows[-1]
    print(
        "latest "
        f"bid_hit={_format(latest, 'rollout_bid_hit_rate')} "
        f"league_reward={_format(latest, 'rollout_historical_relative_reward')} "
        f"explore_self_reward={_format(latest, 'rollout_explore_self_relative_reward')} "
        f"bid_hit_p3/p4/p5={_format(latest, 'rollout_bid_hit_p3')}/"
        f"{_format(latest, 'rollout_bid_hit_p4')}/"
        f"{_format(latest, 'rollout_bid_hit_p5')} "
        f"entropy_update={_format(latest, 'entropy_update')} "
        f"kl={_format(latest, 'approx_kl')} "
        f"clip={_format(latest, 'clip_fraction')}"
    )

    prediction = _latest_row_with(rows, "pred_samples")
    if prediction is not None:
        print(
            "prediction "
            f"iteration={int(float(prediction['iteration']))} "
            f"value_explained={_format(prediction, 'pred_value_explained_variance')} "
            f"trick_accuracy={_format(prediction, 'pred_trick_count_accuracy')} "
            f"trick_acc_bidtime={_format(prediction, 'pred_trick_accuracy_bidtime')} "
            f"suit_accuracy={_format(prediction, 'pred_suit_presence_accuracy')} "
            f"bid_entropy={_format(prediction, 'pred_bid_entropy')}"
        )


def _latest_row_with(
    rows: list[dict[str, float | str]],
    name: str,
) -> dict[str, float | str] | None:
    for row in reversed(rows):
        value = row.get(name, math.nan)
        if isinstance(value, float) and not math.isnan(value):
            return row
    return None


def _format(row: dict[str, float | str], name: str) -> str:
    value = row.get(name, math.nan)
    if not isinstance(value, float) or math.isnan(value):
        return "n/a"
    return f"{value:.4f}"


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
