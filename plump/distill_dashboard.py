"""Dashboard for the on-policy distillation bootstrap.

The reinforcement dashboard is organized around reward, advantage and the
trust region. None of those exist here: there is no return to maximize, no
behaviour policy to stay near, and no dual to watch. The question this run
answers is narrower and sharper -- *how close is the student to the teacher,
and on which head is it still wrong* -- so the panels are organized around
divergence rather than reward.

Read it top-left to bottom-right: agreement says whether the student would
play the same card, KL says how far the whole distribution still is, the head
panel says which part of the model is lagging, and the throughput row says
whether you will reach the update budget.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plump.dashboard import (
    DEFAULT_SMOOTH_WINDOW,
    _duration,
    _lines,
    _no_data,
    _number,
    _read_rows,
    _series,
    _smooth,
)

# A student that agrees with the teacher on this share of decisions is
# behaving like it; below this the bootstrap is not yet usable as a warm start.
AGREEMENT_TARGET = 0.90
# Forward KL in nats. Under ~0.05 the distributions are close enough that
# continuing to distill buys less than switching to the reinforcement run.
KL_TARGET = 0.05


def render_distill_dashboard(
    metrics_path: str | Path,
    output_path: str | Path,
    *,
    title: str | None = None,
    smooth: int = DEFAULT_SMOOTH_WINDOW,
    dpi: int = 150,
    update_budget: int | None = None,
) -> int:
    """Render the distillation trends. Returns the number of rows plotted."""

    if smooth < 1:
        raise ValueError("smooth must be at least 1")

    metrics_path = Path(metrics_path)
    output_path = Path(output_path)
    rows = _read_rows(metrics_path)
    if not rows:
        raise ValueError(f"No metric rows in {metrics_path}")

    iteration = _series(rows, "iteration")
    fig, axes = plt.subplots(3, 3, figsize=(20, 13), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0.0, 0.045, 1.0, 0.95))
    fig.suptitle(
        title or f"Plump distillation · {metrics_path.parent.name}",
        fontsize=16,
    )

    _agreement(axes[0][0], iteration, rows, smooth=smooth)
    _kl(axes[0][1], iteration, rows, smooth=smooth)
    _top1_mass(axes[0][2], iteration, rows, smooth=smooth)

    _entropy_match(axes[1][0], iteration, rows, smooth=smooth)
    _value_match(axes[1][1], iteration, rows, smooth=smooth)
    _head_losses(axes[1][2], iteration, rows, smooth=smooth)

    _belief_transfer(axes[2][0], iteration, rows, smooth=smooth)
    _optimization(axes[2][1], iteration, rows, smooth=smooth)
    _throughput(axes[2][2], iteration, rows, smooth=smooth)

    fig.text(
        0.5,
        0.012,
        _footer(rows, update_budget=update_budget),
        ha="center",
        fontsize=10,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return len(rows)


def _relegend(ax, *extra_axes) -> None:
    """Rebuild the legend from every artist carrying a real label.

    Matplotlib names unlabelled artists ``_childN``; a threshold line added
    after ``_lines`` has already drawn its legend is otherwise invisible in it,
    and a bare ``axhline`` would otherwise appear *as* ``_childN``.
    """

    handles = []
    for axis in (ax, *extra_axes):
        handles.extend(
            artist
            for artist in axis.get_lines()
            if not artist.get_label().startswith("_")
        )
    if handles:
        ax.legend(
            handles=handles,
            labels=[artist.get_label() for artist in handles],
            fontsize=8,
            loc="best",
        )


def _agreement(ax, iteration, rows, *, smooth: int) -> None:
    """The headline: would the student make the same decision?"""

    _lines(
        ax,
        iteration,
        rows,
        (
            ("distill_agree", "all decisions"),
            ("distill_agree_bid", "bid"),
            ("distill_agree_play", "play"),
        ),
        smooth=smooth,
        title="Top-1 agreement with teacher",
        ylabel="share of decisions",
    )
    ax.axhline(
        AGREEMENT_TARGET,
        color="0.35",
        linestyle=":",
        linewidth=1.2,
        label=f"target {AGREEMENT_TARGET:.0%}",
    )
    ax.set_ylim(0.0, 1.0)
    _relegend(ax)


def _kl(ax, iteration, rows, *, smooth: int) -> None:
    """Forward KL, log scale: it should fall by orders of magnitude."""

    _lines(
        ax,
        iteration,
        rows,
        (
            ("distill_kl", "all decisions"),
            ("distill_kl_bid", "bid"),
            ("distill_kl_play", "play"),
        ),
        smooth=smooth,
        title="KL(teacher ‖ student)",
        ylabel="nats per decision",
    )
    values = _series(rows, "distill_kl")
    finite = values[np.isfinite(values) & (values > 0)]
    if finite.size:
        ax.set_yscale("log")
        ax.axhline(
            KL_TARGET,
            color="0.35",
            linestyle=":",
            linewidth=1.2,
            label=f"target {KL_TARGET}",
        )
        _relegend(ax)


def _top1_mass(ax, iteration, rows, *, smooth: int) -> None:
    """Probability the student puts on the teacher's preferred action.

    Distinct from agreement: the student can pick the same argmax while still
    being far less confident, which is exactly what a half-trained student
    looks like.
    """

    _lines(
        ax,
        iteration,
        rows,
        (
            ("distill_top1_mass_bid", "bid"),
            ("distill_top1_mass_play", "play"),
        ),
        smooth=smooth,
        title="Student mass on teacher's top action",
        ylabel="probability",
    )
    ax.set_ylim(0.0, 1.0)


def _entropy_match(ax, iteration, rows, *, smooth: int) -> None:
    """Teacher and student spread, together.

    A student whose entropy sits far below the teacher's has collapsed onto
    the mode and is copying the argmax rather than the distribution -- it will
    look fine on agreement and behave badly off-distribution.
    """

    _lines(
        ax,
        iteration,
        rows,
        (
            ("distill_teacher_entropy_bid", "teacher bid"),
            ("distill_student_entropy_bid", "student bid"),
            ("distill_teacher_entropy_play", "teacher play"),
            ("distill_student_entropy_play", "student play"),
        ),
        smooth=smooth,
        title="Policy entropy: teacher vs student",
        ylabel="nats",
    )


def _value_match(ax, iteration, rows, *, smooth: int) -> None:
    correlation = _series(rows, "distill_value_correlation")
    rmse = _series(rows, "distill_value_rmse")
    valid = np.isfinite(iteration)
    if not (valid.any() and (np.isfinite(correlation).any() or np.isfinite(rmse).any())):
        _no_data(ax)
        ax.set_title("Value head agreement")
        return
    ax.plot(iteration, _smooth(correlation, smooth), color="C0", label="correlation")
    ax.set_ylabel("correlation with teacher")
    ax.set_ylim(-1.0, 1.0)
    ax.axhline(0.0, color="0.8", linewidth=0.8)
    twin = ax.twinx()
    twin.plot(
        iteration, _smooth(rmse, smooth), color="C3", linestyle="--", label="RMSE"
    )
    twin.set_ylabel("RMSE vs teacher")
    _relegend(ax, twin)
    ax.set_title("Value head agreement")


def _head_losses(ax, iteration, rows, *, smooth: int) -> None:
    """Which objective is actually carrying the loss right now."""

    _lines(
        ax,
        iteration,
        rows,
        (
            ("distill_loss_policy", "policy KL"),
            ("distill_loss_value", "value"),
            ("distill_loss_aux", "auxiliary heads"),
            ("distill_aux_entropy_floor", "auxiliary floor (teacher entropy)"),
        ),
        smooth=smooth,
        title="Distillation loss by objective",
        ylabel="loss",
        omit_zero=True,
    )
    # The auxiliary cross-entropy sits on top of the teacher's own entropy,
    # which no training can remove and which dwarfs the learnable part -- it
    # reads as flat long before the heads stop improving. Drawing the floor
    # alongside it shows how much of the curve is actually in play; the
    # belief-head panel plots the difference, which is what does move.
    _relegend(ax)
    values = np.concatenate(
        [
            _series(rows, name)
            for name in (
                "distill_loss_policy",
                "distill_loss_value",
                "distill_loss_aux",
            )
        ]
    )
    finite = values[np.isfinite(values) & (values > 0)]
    if finite.size and finite.max() / max(finite.min(), 1e-12) > 50:
        ax.set_yscale("log")


def _belief_transfer(ax, iteration, rows, *, smooth: int) -> None:
    """Per-head divergence from the teacher's beliefs.

    These are the heads that took tens of thousands of reinforcement
    iterations to train, so transferring them is most of the value of the
    bootstrap. Plotted as KL rather than the cross-entropy the optimizer sees:
    cross-entropy carries the teacher's own entropy as an additive floor, which
    differs per head and never reaches zero, so the raw losses cannot be
    compared to each other or read as "how far is left". These all go to zero.
    """

    _lines(
        ax,
        iteration,
        rows,
        (
            ("distill_aux_kl_suit", "suit presence"),
            ("distill_aux_kl_rank_boundary", "rank boundary"),
            ("distill_aux_kl_trick", "trick count"),
            ("distill_aux_kl_bid_hit", "bid hit"),
            ("distill_aux_kl_next_winner", "next trick winner"),
        ),
        smooth=smooth,
        title="Belief head KL from teacher",
        ylabel="nats per position",
        omit_zero=True,
    )
    values = np.concatenate(
        [
            _series(rows, name)
            for name in (
                "distill_aux_kl_suit",
                "distill_aux_kl_rank_boundary",
                "distill_aux_kl_trick",
                "distill_aux_kl_bid_hit",
                "distill_aux_kl_next_winner",
            )
        ]
    )
    finite = values[np.isfinite(values) & (values > 0)]
    if finite.size and finite.max() / max(finite.min(), 1e-12) > 50:
        ax.set_yscale("log")


def _optimization(ax, iteration, rows, *, smooth: int) -> None:
    _lines(
        ax,
        iteration,
        rows,
        (
            ("core_grad_norm", "core"),
            ("embedding_grad_norm", "embedding"),
            ("auxiliary_grad_norm", "auxiliary"),
        ),
        smooth=smooth,
        title="Gradient norms",
        ylabel="norm",
        omit_zero=True,
    )


def _throughput(ax, iteration, rows, *, smooth: int) -> None:
    _lines(
        ax,
        iteration,
        rows,
        (
            ("collect_sec", "collect"),
            ("update_sec", "update"),
        ),
        smooth=smooth,
        title="Seconds per iteration",
        ylabel="seconds",
    )


def _footer(rows: list[dict[str, str]], *, update_budget: int | None) -> str:
    """One line stating where the run is against the thresholds that matter."""

    last = rows[-1]
    iteration = int(_number(last, "iteration"))
    parts = [f"iteration {iteration:,}"]

    agree = _number(last, "distill_agree")
    if math.isfinite(agree) and agree > 0:
        parts.append(f"agreement {agree:.1%} (target {AGREEMENT_TARGET:.0%})")
    kl = _number(last, "distill_kl")
    if math.isfinite(kl) and kl > 0:
        parts.append(f"KL {kl:.4f} nats (target {KL_TARGET})")
    correlation = _number(last, "distill_value_correlation")
    if math.isfinite(correlation) and correlation != 0:
        parts.append(f"value r {correlation:.3f}")

    if update_budget:
        remaining = max(update_budget - iteration, 0)
        per_iteration = _number(last, "collect_sec") + _number(last, "update_sec")
        if math.isfinite(per_iteration) and per_iteration > 0 and remaining:
            parts.append(
                f"{remaining:,} updates left (~{_duration(remaining * per_iteration)})"
            )
    total = _number(last, "total_sec")
    if math.isfinite(total) and total > 0:
        parts.append(f"elapsed {_duration(total)}")
    return "  ·  ".join(parts)
