"""The distillation dashboard renders from a real metrics header."""

from __future__ import annotations

import csv

import pytest

from plump.cli import METRIC_COLUMNS
from plump.distill_dashboard import (
    AGREEMENT_TARGET,
    KL_TARGET,
    _footer,
    render_distill_dashboard,
)


def write_metrics(path, count=40, *, converging=True):
    """A run whose student is closing on the teacher."""

    rows = []
    for index in range(count):
        progress = index / max(count - 1, 1)
        row = {column: "" for column in METRIC_COLUMNS}
        row.update(
            iteration=index * 10,
            optimizer_steps=index * 10,
            collect_sec=3.8,
            update_sec=0.6,
            total_sec=index * 4.4,
            core_grad_norm=1.5 - progress,
            embedding_grad_norm=0.4,
            auxiliary_grad_norm=0.2,
            distill_kl=(1.0 - 0.9 * progress) if converging else 1.0,
            distill_kl_bid=(2.0 - 1.8 * progress) if converging else 2.0,
            distill_kl_play=(0.7 - 0.6 * progress) if converging else 0.7,
            distill_agree=0.45 + 0.5 * progress,
            distill_agree_bid=0.40 + 0.5 * progress,
            distill_agree_play=0.47 + 0.5 * progress,
            distill_top1_mass_bid=0.5 + 0.4 * progress,
            distill_top1_mass_play=0.5 + 0.4 * progress,
            distill_teacher_entropy_bid=0.8,
            distill_teacher_entropy_play=1.1,
            distill_student_entropy_bid=1.6 - 0.8 * progress,
            distill_student_entropy_play=2.0 - 0.9 * progress,
            distill_loss_policy=1.0 - 0.9 * progress,
            distill_loss_value=0.01 - 0.008 * progress,
            distill_loss_aux=0.5 - 0.3 * progress,
            distill_value_rmse=0.10 - 0.06 * progress,
            distill_value_correlation=-0.1 + 0.9 * progress,
            distill_aux_kl_suit=0.40 - 0.35 * progress,
            distill_aux_kl_rank_boundary=0.30 - 0.25 * progress,
            distill_aux_kl_trick=0.25 - 0.20 * progress,
            distill_aux_kl_bid_hit=0.15 - 0.12 * progress,
            distill_aux_kl_next_winner=0.20 - 0.17 * progress,
        )
        rows.append(row)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def test_every_distill_column_exists_in_the_metrics_header():
    """The dashboard may only read columns the trainer actually writes."""

    required = {
        "distill_kl",
        "distill_kl_bid",
        "distill_kl_play",
        "distill_agree",
        "distill_agree_bid",
        "distill_agree_play",
        "distill_top1_mass_bid",
        "distill_top1_mass_play",
        "distill_teacher_entropy_bid",
        "distill_teacher_entropy_play",
        "distill_student_entropy_bid",
        "distill_student_entropy_play",
        "distill_loss_policy",
        "distill_loss_value",
        "distill_loss_aux",
        "distill_value_rmse",
        "distill_value_correlation",
    }
    assert required <= set(METRIC_COLUMNS)


def test_renders_a_png(tmp_path):
    metrics = tmp_path / "metrics.csv"
    write_metrics(metrics)
    out = tmp_path / "distill.png"
    count = render_distill_dashboard(metrics, out, smooth=5)
    assert count == 40
    assert out.exists() and out.stat().st_size > 10_000


def test_renders_from_a_single_row(tmp_path):
    """A dashboard drawn at iteration 0 must not blow up on empty history."""

    metrics = tmp_path / "metrics.csv"
    write_metrics(metrics, count=1)
    out = tmp_path / "distill.png"
    assert render_distill_dashboard(metrics, out, smooth=50) == 1
    assert out.exists()


def test_empty_metrics_is_an_error(tmp_path):
    metrics = tmp_path / "metrics.csv"
    with open(metrics, "w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=METRIC_COLUMNS).writeheader()
    with pytest.raises(ValueError, match="No metric rows"):
        render_distill_dashboard(metrics, tmp_path / "x.png")


def test_footer_states_position_against_the_targets(tmp_path):
    rows = write_metrics(tmp_path / "metrics.csv")
    text = _footer(
        [{k: str(v) for k, v in row.items()} for row in rows], update_budget=10_000
    )
    assert "agreement" in text and f"{AGREEMENT_TARGET:.0%}" in text
    assert "KL" in text and str(KL_TARGET) in text
    assert "value r" in text
    # Remaining budget and its wall-clock estimate.
    assert "updates left" in text


def test_footer_without_a_budget_omits_the_estimate(tmp_path):
    rows = write_metrics(tmp_path / "metrics.csv")
    text = _footer(
        [{k: str(v) for k, v in row.items()} for row in rows], update_budget=None
    )
    assert "updates left" not in text


class _FakeRun:
    """Duck-typed stand-in for RunDirectory: the dispatcher only reads these."""

    def __init__(self, metrics, name, objective):
        self.metrics = metrics
        self.name = name
        self._objective = objective

    def recorded_config(self):
        return {"training": {"policy_objective": self._objective}}


def test_cli_dispatches_distill_runs_to_the_distill_dashboard(tmp_path):
    from plump.cli import _render_run_dashboard

    metrics = tmp_path / "metrics.csv"
    write_metrics(metrics)
    out = tmp_path / "dashboard.png"
    # The reinforcement renderer is given options this one has no analogue
    # for; the dispatcher must drop them rather than forward them.
    rows = _render_run_dashboard(
        _FakeRun(metrics, "distill-run", "distill"),
        out,
        title="Plump schema-v6 · distill-run",
        smooth=5,
        dpi=80,
        include_learning_rate=True,
    )
    assert rows == 40
    assert out.exists()


def test_cli_leaves_reinforcement_runs_on_the_original_dashboard(tmp_path):
    from plump.cli import _render_run_dashboard

    metrics = tmp_path / "metrics.csv"
    write_metrics(metrics)
    out = tmp_path / "dashboard.png"
    rows = _render_run_dashboard(
        _FakeRun(metrics, "ppo-run", "ppo"), out, title="x", smooth=5
    )
    assert rows == 40 and out.exists()


def test_dispatch_survives_a_run_with_no_recorded_config(tmp_path):
    from plump.cli import _render_run_dashboard

    class Broken(_FakeRun):
        def recorded_config(self):
            raise FileNotFoundError("config.toml")

    metrics = tmp_path / "metrics.csv"
    write_metrics(metrics)
    out = tmp_path / "dashboard.png"
    assert _render_run_dashboard(Broken(metrics, "r", "distill"), out, title="x") == 40


def test_belief_panel_reads_per_head_kl_columns(tmp_path):
    """The panel must read columns the distillation update actually writes.

    It previously plotted the PPO stage-accuracy columns, which only the
    reinforcement path fills, so it rendered "No data yet" for every
    distillation run.
    """

    required = {
        "distill_aux_kl_suit",
        "distill_aux_kl_rank_boundary",
        "distill_aux_kl_trick",
        "distill_aux_kl_bid_hit",
        "distill_aux_kl_next_winner",
    }
    assert required <= set(METRIC_COLUMNS)

    import matplotlib.pyplot as plt

    from plump.dashboard import _read_rows, _series
    from plump.distill_dashboard import _belief_transfer

    metrics = tmp_path / "metrics.csv"
    write_metrics(metrics)
    rows = _read_rows(metrics)
    figure, axis = plt.subplots()
    _belief_transfer(axis, _series(rows, "iteration"), rows, smooth=5)
    # Five heads, all populated by write_metrics below. Each series draws a
    # faint raw line plus the smoothed one, and only the smoothed one is
    # labelled, so count by label rather than by artist.
    labelled = [
        line.get_label()
        for line in axis.lines
        if not line.get_label().startswith("_")
    ]
    assert sorted(labelled) == sorted(
        ["suit presence", "rank boundary", "trick count", "bid hit", "next trick winner"]
    )
    plt.close(figure)


def test_background_dashboard_respects_its_interval(monkeypatch):
    """Training must refresh the dashboard itself, on dashboard_every."""

    from plump.cli import _BackgroundDashboard

    launched = []

    class FakeProcess:
        def __init__(self, done=True):
            self._done = done

        def poll(self):
            return 0 if self._done else None

    def fake_popen(cmd, **kwargs):
        launched.append(cmd)
        return FakeProcess()

    monkeypatch.setattr("plump.cli.subprocess.Popen", fake_popen)
    dash = _BackgroundDashboard("some-run", every=5)
    for iteration in range(1, 13):
        dash.maybe_refresh(iteration)
    # Fires at 5 and 10, not on every iteration.
    assert len(launched) == 2
    assert launched[0][-2:] == ["dashboard", "some-run"]


def test_background_dashboard_disabled_when_every_is_zero(monkeypatch):
    from plump.cli import _BackgroundDashboard

    launched = []
    monkeypatch.setattr(
        "plump.cli.subprocess.Popen", lambda cmd, **k: launched.append(cmd)
    )
    dash = _BackgroundDashboard("r", every=0)
    for iteration in range(1, 50):
        dash.maybe_refresh(iteration)
    assert not launched


def test_background_dashboard_skips_while_a_render_is_in_flight(monkeypatch):
    """A slow render must never queue up behind itself and stall training."""

    from plump.cli import _BackgroundDashboard

    launched = []

    class Running:
        def poll(self):
            return None  # still going

    monkeypatch.setattr(
        "plump.cli.subprocess.Popen",
        lambda cmd, **k: (launched.append(cmd), Running())[1],
    )
    dash = _BackgroundDashboard("r", every=1)
    for iteration in range(1, 10):
        dash.maybe_refresh(iteration)
    assert len(launched) == 1


def test_background_dashboard_never_takes_training_down(monkeypatch):
    from plump.cli import _BackgroundDashboard

    def boom(cmd, **kwargs):
        raise OSError("no fork for you")

    monkeypatch.setattr("plump.cli.subprocess.Popen", boom)
    dash = _BackgroundDashboard("r", every=1)
    dash.maybe_refresh(1)  # must not raise
    dash.finish()
