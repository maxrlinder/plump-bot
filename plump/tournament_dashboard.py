"""Live, partial-safe dashboard rendering for checkpoint tournaments."""

from __future__ import annotations

import json
import math
import uuid
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dedddaff"

CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a")
DEPTH_RAMP = ("#86b6ef", "#3987e5", "#256abf", "#0d366b")
SERIES = "#2a78d6"
BAND = "#9ec5f4"
REFERENCE = "#8a8984"

DEPTH_GROUPS = (
    ("3-4 cards", (3, 4)),
    ("5-6 cards", (5, 6)),
    ("7-8 cards", (7, 8)),
    ("9-10 cards", (9, 10)),
)
SMOOTH = 5


def _smooth(values: np.ndarray, window: int = SMOOTH) -> np.ndarray:
    """Centred rolling mean that keeps endpoints and tolerates missing cells."""

    if values.size < window:
        return values
    result = np.empty_like(values, dtype=float)
    half = window // 2
    for index in range(values.size):
        start = max(0, index - half)
        stop = min(values.size, index + half + 1)
        window_values = values[start:stop]
        finite = window_values[np.isfinite(window_values)]
        result[index] = float(np.mean(finite)) if finite.size else np.nan
    return result


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, color=INK, pad=8)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_MUTED)
    ax.set_xlabel("iteration", fontsize=9, color=INK_MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, colors=INK_MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _mark_forks(ax, forks: list[tuple[int, str]]) -> None:
    for iteration, label in forks:
        ax.axvline(iteration, color=REFERENCE, linewidth=1.0, linestyle=":", zorder=1)
        ax.annotate(
            label,
            xy=(iteration, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(3, -10),
            textcoords="offset points",
            fontsize=7,
            color=INK_MUTED,
        )


def _reference(ax, value: float, label: str, *, corner: str = "lower right") -> None:
    x, y, ha, va = {
        "lower right": (0.985, 0.03, "right", "bottom"),
        "lower left": (0.015, 0.03, "left", "bottom"),
    }[corner]
    ax.text(
        x,
        y,
        f"{label}: {value:+.3f}",
        transform=ax.transAxes,
        fontsize=8,
        color=INK_MUTED,
        ha=ha,
        va=va,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor=SURFACE,
            edgecolor=GRID,
            linewidth=0.8,
        ),
    )


def _cell_mean(entry: dict, keep) -> float:
    values = []
    for name, value in entry.get("cells", {}).items():
        players, hand = name.split("p-")
        hand = hand.removesuffix("c")
        if keep(int(players), int(hand)):
            values.append(float(value))
    return float(np.mean(values)) if values else np.nan


def _progress_text(payload: dict) -> str:
    progress = payload.get("progress", {})
    status = payload.get("status", "complete")
    if status == "loading":
        loaded = int(progress.get("loaded_policies", 0))
        total = int(progress.get("total_policies", len(payload.get("roster", []))))
        return f"loading policies {loaded}/{total}"
    fraction = float(progress.get("fraction", 1.0 if status == "complete" else 0.0))
    completed = int(progress.get("completed_matches", 0))
    total = int(progress.get("total_matches", 0))
    shape = progress.get("current_shape")
    shape_text = ""
    if isinstance(shape, list) and len(shape) == 2:
        shape_text = f" · now {shape[0]}p-{shape[1]}c"
    return f"{status} · {100.0 * fraction:.1f}% · tables {completed}/{total}{shape_text}"


def _save_atomic(fig, output_path: Path, *, dpi: int) -> Path:
    """Publish a complete PNG in one rename so image viewers never see a tear."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.stem}.{uuid.uuid4().hex}.tmp{output_path.suffix}"
    )
    try:
        fig.savefig(temporary, dpi=dpi, facecolor=SURFACE)
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output_path


def render_tournament_dashboard(
    payload_path: str | Path,
    output_path: str | Path,
    *,
    dpi: int = 150,
) -> Path:
    """Render either a running partial aggregate or a completed tournament."""

    payload_path = Path(payload_path)
    output_path = Path(output_path)
    payload = json.loads(payload_path.read_text())
    roster = {entry["key"]: entry for entry in payload.get("roster", [])}
    entries = payload.get("report", {}).get("entries", [])
    heuristic = next((entry for entry in entries if entry["key"] == "heuristic"), None)
    models = sorted(
        (entry for entry in entries if entry["key"] != "heuristic"),
        key=lambda entry: entry["iteration"],
    )
    for entry in models:
        entry["run"] = roster.get(entry["key"], {}).get("run", entry.get("run", ""))

    protocol = payload.get("protocol", {})
    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
    fig.suptitle(
        f"Tournament · {payload.get('run', '')} · {payload_path.stem}\n"
        f"{len(models)}/{max(len(roster) - ('heuristic' in roster), 0)} checkpoints visible · "
        f"{payload.get('report', {}).get('rounds', 0):,} completed round-seats · "
        f"{'argmax' if protocol.get('greedy', True) else 'sampled'} actions · "
        f"{protocol.get('waves', '?')} waves × {protocol.get('deals_per_match', '?')} deals · "
        f"{_progress_text(payload)}",
        fontsize=13,
        color=INK,
    )

    if not models:
        for ax in axes.flat:
            ax.set_axis_off()
        axes[0, 1].text(
            0.5,
            0.5,
            _progress_text(payload),
            transform=axes[0, 1].transAxes,
            ha="center",
            va="center",
            fontsize=18,
            color=INK_MUTED,
        )
        path = _save_atomic(fig, output_path, dpi=dpi)
        plt.close(fig)
        return path

    iteration = np.array([entry["iteration"] for entry in models], dtype=float)
    reward = np.array([entry["relative_reward"] for entry in models], dtype=float)
    low = np.array([entry["relative_reward_ci_low"] for entry in models], dtype=float)
    high = np.array([entry["relative_reward_ci_high"] for entry in models], dtype=float)

    forks: list[tuple[int, str]] = []
    for previous, entry in zip(models, models[1:]):
        if entry["run"] != previous["run"]:
            short_a = previous["run"].split("-")[-1]
            short_b = entry["run"].split("-")[-1]
            forks.append((entry["iteration"], f"{short_a} → {short_b}"))

    ax = axes[0, 0]
    ax.fill_between(iteration, low, high, color=BAND, alpha=0.45, linewidth=0, zorder=2)
    ax.plot(
        iteration,
        reward,
        color=SERIES,
        linewidth=0,
        marker="o",
        markersize=3.5,
        alpha=0.55,
        zorder=3,
    )
    ax.plot(iteration, _smooth(reward), color=SERIES, linewidth=2.0, zorder=4)
    ax.axhline(0.0, color=GRID, linewidth=1.0, zorder=1)
    _mark_forks(ax, forks)
    _style(ax, "Relative reward per round", "reward (points)")
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                color=SERIES,
                linewidth=2.0,
                marker="o",
                markersize=4,
                label=f"checkpoint (line: {SMOOTH}-point mean)",
            ),
            Line2D([], [], color=BAND, linewidth=7, alpha=0.6, label="95% bootstrap CI"),
        ],
        fontsize=8,
        loc="upper left",
        frameon=False,
        labelcolor=INK_MUTED,
    )
    if heuristic is not None:
        _reference(ax, heuristic["relative_reward"], "heuristic anchor")

    ax = axes[0, 1]
    for slot, players in enumerate((3, 4, 5)):
        series = np.array(
            [_cell_mean(entry, lambda p, h, target=players: p == target) for entry in models]
        )
        ax.plot(
            iteration,
            _smooth(series),
            color=CATEGORICAL[slot],
            linewidth=2.0,
            label=f"{players} players",
            zorder=3,
        )
    ax.axhline(0.0, color=GRID, linewidth=1.0, zorder=1)
    _mark_forks(ax, forks)
    _style(ax, "Relative reward by table size", "reward (points)")
    ax.legend(fontsize=8, loc="upper left", frameon=False, labelcolor=INK_MUTED)

    ax = axes[0, 2]
    for slot, (label, (low_hand, high_hand)) in enumerate(DEPTH_GROUPS):
        series = np.array(
            [
                _cell_mean(
                    entry,
                    lambda p, h, low=low_hand, high=high_hand: low <= h <= high,
                )
                for entry in models
            ]
        )
        ax.plot(
            iteration,
            _smooth(series),
            color=DEPTH_RAMP[slot],
            linewidth=2.0,
            label=label,
            zorder=3,
        )
    ax.axhline(0.0, color=GRID, linewidth=1.0, zorder=1)
    _mark_forks(ax, forks)
    _style(ax, "Relative reward by hand depth", "reward (points)")
    ax.legend(fontsize=8, loc="upper left", frameon=False, labelcolor=INK_MUTED)

    ax = axes[1, 0]
    hits = np.array([entry["bid_hit_rate"] for entry in models], dtype=float)
    ax.plot(
        iteration,
        hits,
        color=SERIES,
        linewidth=0,
        marker="o",
        markersize=3.5,
        alpha=0.5,
        zorder=3,
    )
    ax.plot(iteration, _smooth(hits), color=SERIES, linewidth=2.0, zorder=4)
    _mark_forks(ax, forks)
    _style(ax, "Bid hit rate", "fraction of bids made exactly")
    if heuristic is not None:
        _reference(ax, heuristic["bid_hit_rate"], "heuristic anchor")

    ax = axes[1, 1]
    errors = np.array([entry["bid_abs_error"] for entry in models], dtype=float)
    ax.plot(
        iteration,
        errors,
        color=SERIES,
        linewidth=0,
        marker="o",
        markersize=3.5,
        alpha=0.5,
        zorder=3,
    )
    ax.plot(iteration, _smooth(errors), color=SERIES, linewidth=2.0, zorder=4)
    _mark_forks(ax, forks)
    _style(ax, "Bid absolute error", "mean |bid − tricks taken|")
    if heuristic is not None:
        _reference(ax, heuristic["bid_abs_error"], "heuristic anchor", corner="lower left")

    ax = axes[1, 2]
    player_counts = tuple(protocol.get("player_counts", (3, 4, 5)))
    hand_sizes = tuple(protocol.get("hand_sizes", range(3, 11)))
    shapes = [f"{players}p-{hand}c" for players in player_counts for hand in hand_sizes]
    first_bin = math.floor(float(iteration.min()) / 2000.0) * 2000
    last_bin = math.floor(float(iteration.max()) / 2000.0) * 2000
    starts = np.arange(first_bin, last_bin + 2000, 2000, dtype=float)
    columns: list[list[float]] = []
    labels: list[str] = []
    for start in starts:
        group = [entry for entry in models if start <= entry["iteration"] < start + 2000]
        if not group:
            continue
        column = []
        for shape in shapes:
            values = [entry.get("cells", {}).get(shape) for entry in group]
            present = [float(value) for value in values if value is not None]
            column.append(float(np.mean(present)) if present else np.nan)
        columns.append(column)
        labels.append(f"{int(start / 1000)}k")
    grid = np.asarray(columns, dtype=float).T
    finite = np.abs(grid[np.isfinite(grid)])
    span = max(float(finite.max()) if finite.size else 0.0, 1e-6)
    mesh = ax.pcolormesh(
        np.arange(len(labels) + 1),
        np.arange(len(shapes) + 1),
        grid,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span),
        edgecolors=SURFACE,
        linewidth=1.0,
    )
    ax.set_xticks(np.arange(len(labels)) + 0.5, labels, fontsize=7)
    ax.set_yticks(np.arange(len(shapes)) + 0.5, shapes, fontsize=7)
    ax.invert_yaxis()
    ax.set_title("Relative reward by shape and era", fontsize=11, color=INK, pad=8)
    ax.set_xlabel("iteration bin (2000 wide)", fontsize=9, color=INK_MUTED)
    ax.tick_params(colors=INK_MUTED, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    bar = fig.colorbar(mesh, ax=ax, fraction=0.045, pad=0.02)
    bar.set_label("reward (points)", fontsize=8, color=INK_MUTED)
    bar.ax.tick_params(labelsize=7, colors=INK_MUTED)
    bar.outline.set_visible(False)

    path = _save_atomic(fig, output_path, dpi=dpi)
    plt.close(fig)
    return path
