"""Unified command-line interface for schema-v6 Plump workflows."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from plump.dashboard import render_dashboard
from plump.evaluation import DealBank, evaluate_policy
from plump.gui.app import run as run_gui
from plump.policies import HeuristicPolicy
from plump.run_config import (
    DEFAULT_CONFIG_PATH,
    config_diff,
    load_training_config,
    resolve_training_config,
)
from plump.runs import RunDirectory, atomic_write_json
from plump.seq.model import SeqPlumpModel
from plump.seq.policy import SeqModelPolicy, best_seq_device
from plump.seq.trainer import SeqTrainer

METRIC_COLUMNS = (
    "iteration",
    "optimizer_steps",
    "learning_rate",
    "elapsed_sec",
    "total_sec",
    "collect_sec",
    "update_sec",
    "trees",
    "leaves",
    "decisions",
    "policy_rows",
    "branched_rows",
    "unbranched_rows",
    "positions",
    "forward_rows",
    "branch_decisions",
    "bid_hit_rate",
    "reward_self",
    "reward_historical",
    "spine_entropy",
    "loss_policy",
    "loss_value",
    "loss_suit",
    "loss_trick",
    "loss_bid_hit",
    "entropy",
    "policy_kl",
    "policy_kl_p95",
    "policy_kl_p99",
    "policy_kl_max",
    "backtracks",
    "step_scale",
    "rolled_back",
    "eval_reward_vs_heuristic",
    "eval_bid_hit",
    "peak_cache_rows",
    "cache_rows_allocated",
    "cache_pressure",
    "peak_device_gb",
    "blocked_by_cache",
    "skipped_by_placement",
    "sample_sec",
    "step_sec",
    "compact_sec",
    "token_build_sec",
    "forward_sec",
    "positions_per_sec",
    "forward_rows_per_sec",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plump",
        description="Schema-v6 Plump training and inspection tools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="create or safely resume a run")
    train.add_argument("run")
    train.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    train.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
    )
    train.add_argument("--iterations", type=int)
    train.add_argument("--device")
    train.add_argument(
        "--from-checkpoint",
        type=Path,
        help="start a new run from a compatible schema-v6 checkpoint",
    )
    train.set_defaults(handler=train_command)

    dashboard = subparsers.add_parser(
        "dashboard",
        help="render a run's static metrics dashboard",
    )
    dashboard.add_argument("run")
    dashboard.add_argument("--smooth", type=int, default=20)
    dashboard.add_argument("--dpi", type=int, default=150)
    dashboard.set_defaults(handler=dashboard_command)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="evaluate a schema-v6 checkpoint against the heuristic",
    )
    evaluate.add_argument("run")
    evaluate.add_argument("--checkpoint", default="latest")
    evaluate.add_argument("--device")
    evaluate.set_defaults(handler=evaluate_command)

    analyze = subparsers.add_parser(
        "analyze",
        help="write run-scoped card-representation analyses",
    )
    analyze.add_argument("run")
    analyze.add_argument("--checkpoint", default="latest")
    analyze.add_argument("--seed", type=int, default=42)
    analyze.add_argument("--permutations", type=int, default=1000)
    analyze.add_argument("--dpi", type=int, default=180)
    analyze.set_defaults(handler=analyze_command)

    play = subparsers.add_parser("play", help="launch the local browser GUI")
    play.add_argument("run", nargs="?")
    play.add_argument("--checkpoint", default="latest")
    play.add_argument("--device")
    play.add_argument("--host", default="127.0.0.1")
    play.add_argument("--port", type=int, default=8765)
    play.set_defaults(handler=play_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.invocation = ["plump", *(argv if argv is not None else sys.argv[1:])]
    try:
        return int(args.handler(args) or 0)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def train_command(args: argparse.Namespace) -> int:
    overrides = list(args.overrides)
    if args.iterations is not None:
        overrides.append(f"run.iterations={args.iterations}")
    if args.device is not None:
        overrides.append(f"run.device={json.dumps(args.device)}")
    requested = load_training_config(args.config, overrides=overrides)
    run = RunDirectory(args.run)
    existed = run.exists
    if existed and args.from_checkpoint is not None:
        raise ValueError("--from-checkpoint requires a new run name.")

    with run.acquire_lock():
        if existed:
            recorded_raw = run.recorded_config()
            differences = config_diff(recorded_raw, requested.raw)
            if differences:
                rendered = "\n  ".join(differences)
                raise ValueError(
                    "Run configuration differs from the recorded config:\n  " + rendered
                )
            resolved = resolve_training_config(recorded_raw)
        else:
            run.create(requested.raw, args.invocation)
            resolved = requested

        device_value = str(resolved.run["device"])
        device = (
            str(best_seq_device()) if device_value in {"", "auto"} else device_value
        )
        seed = int(resolved.training.seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        trainer = SeqTrainer(
            SeqPlumpModel(resolved.model),
            resolved.training,
            device=device,
        )
        trainer.resolved_config = resolved.raw
        if existed:
            checkpoint = run.resolve_checkpoint("latest")
            trainer.load_checkpoint(checkpoint)
            _emit(
                run,
                f"Resumed {run.name} from {checkpoint.name} "
                f"at iteration {trainer.iteration}.",
            )
        elif args.from_checkpoint is not None:
            source = args.from_checkpoint.expanduser().resolve()
            trainer.load_checkpoint(
                source,
                allow_training_config_mismatch=True,
            )
            trainer.resolved_config = resolved.raw
            imported = run.interval_checkpoint(trainer.iteration)
            trainer.save_checkpoint(imported)
            run.record_latest(imported, trainer.iteration)
            run.update_metadata(
                parent_checkpoint=str(source),
                parent_iteration=trainer.iteration,
            )
            _emit(
                run,
                f"Forked {run.name} from {source.name} "
                f"at iteration {trainer.iteration}.",
            )
        else:
            _emit(run, f"Created run {run.name} on {device}.")

        _ensure_metrics_header(run.metrics)
        elapsed_before = _recorded_elapsed(run.metrics)
        run.update_metadata(
            status="running",
            device=device,
            seed=seed,
            target_iterations=int(resolved.run["iterations"]),
        )

        evaluation = resolved.evaluation
        eval_bank = DealBank.generate(
            player_counts=resolved.training.player_counts,
            hand_sizes=tuple(
                sorted({cell.hand_size for cell in resolved.training.schedule_cells})
            ),
            deals_per_configuration=int(evaluation["deals"]),
            seed=int(evaluation["seed"]),
        )
        heuristic = HeuristicPolicy()
        target = int(resolved.run["iterations"])
        checkpoint_every = int(resolved.run["checkpoint_every"])
        dashboard_every = int(resolved.run["dashboard_every"])

        for iteration in range(trainer.iteration + 1, target + 1):
            trainer.iteration = iteration
            started = time.perf_counter()
            trees, summary = trainer.collect()
            stats = trainer.update(trees)
            collector = trainer.collector.stats

            eval_reward: float | None = None
            eval_bid_hit: float | None = None
            if (
                int(evaluation["every"]) > 0
                and iteration % int(evaluation["every"]) == 0
            ):
                policy = SeqModelPolicy(
                    trainer.model,
                    device=device,
                    greedy=True,
                    name="candidate",
                )
                report = evaluate_policy(
                    policy,
                    heuristic,
                    eval_bank,
                    batch_size=int(evaluation["batch_size"]),
                )
                eval_reward = report.macro_relative_reward
                eval_bid_hit = report.macro_bid_hit_rate
                trainer.model.eval()
                best = run.best_metric()
                if best is None or eval_reward > best:
                    best_path = run.checkpoints / "best.pt"
                    trainer.save_checkpoint(best_path)
                    run.record_best(best_path, iteration, eval_reward)

            total = time.perf_counter() - started
            row = _metric_row(
                trainer,
                summary,
                stats,
                collector,
                total,
                elapsed_before + total,
                eval_reward,
                eval_bid_hit,
            )
            _append_metric(run.metrics, row)
            elapsed_before += total

            if (
                checkpoint_every > 0 and iteration % checkpoint_every == 0
            ) or iteration == target:
                checkpoint = run.interval_checkpoint(iteration)
                snapshot_id = f"iter_{iteration}"
                trainer.league.add(snapshot_id, str(checkpoint), iteration)
                try:
                    trainer.save_checkpoint(checkpoint)
                except Exception:
                    trainer.league.snapshots = [
                        snap
                        for snap in trainer.league.snapshots
                        if snap.snapshot_id != snapshot_id
                    ]
                    raise
                run.record_latest(checkpoint, iteration)

            if dashboard_every > 0 and (
                iteration % dashboard_every == 0 or iteration == target
            ):
                try:
                    render_dashboard(
                        run.metrics,
                        run.dashboard,
                        title=f"Plump schema-v6 · {run.name}",
                    )
                except Exception as error:
                    _emit(run, f"Dashboard refresh failed: {error}")

            message = (
                f"iter {iteration:5d} | {total:6.1f}s "
                f"(collect {summary.collect_sec:.1f} update {stats.update_sec:.1f}) "
                f"| leaves {summary.leaves:5d} positions {stats.positions:6d} "
                f"| bid_hit {summary.bid_hit_rate:.3f} "
                f"| kl {stats.policy_kl:.5f} "
                f"| step {stats.step_scale:.3f}"
            )
            if stats.backtracks:
                message += f" ({stats.backtracks} backtracks)"
            if stats.rolled_back:
                message += " ROLLBACK"
            if eval_reward is not None:
                message += f" | eval {eval_reward:.4f}"
            _emit(run, message)

        run.update_metadata(
            status="complete",
            completed_iteration=trainer.iteration,
        )
    return 0


def dashboard_command(args: argparse.Namespace) -> int:
    run = RunDirectory(args.run)
    rows = render_dashboard(
        run.metrics,
        run.dashboard,
        title=f"Plump schema-v6 · {run.name}",
        smooth=args.smooth,
        dpi=args.dpi,
    )
    print(f"Wrote {run.dashboard} from {rows} rows.")
    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    run = RunDirectory(args.run)
    checkpoint = run.resolve_checkpoint(args.checkpoint)
    resolved = resolve_training_config(run.recorded_config())
    evaluation = resolved.evaluation
    device = args.device or str(best_seq_device())
    policy = SeqModelPolicy.from_checkpoint(
        checkpoint,
        device=device,
        greedy=True,
        name=checkpoint.stem,
    )
    bank = DealBank.generate(
        player_counts=resolved.training.player_counts,
        hand_sizes=tuple(
            sorted({cell.hand_size for cell in resolved.training.schedule_cells})
        ),
        deals_per_configuration=int(evaluation["deals"]),
        seed=int(evaluation["seed"]),
    )
    report = evaluate_policy(
        policy,
        HeuristicPolicy(),
        bank,
        batch_size=int(evaluation["batch_size"]),
    )
    output = run.evaluations / _checkpoint_output_name(checkpoint) / "heuristic.json"
    atomic_write_json(output, dataclasses.asdict(report))
    print(
        f"{checkpoint.name}: reward={report.macro_relative_reward:.4f} "
        f"bid_hit={report.macro_bid_hit_rate:.4f} rounds={report.rounds}"
    )
    print(f"Wrote {output}.")
    return 0


def analyze_command(args: argparse.Namespace) -> int:
    from plump.analysis.card_geometry import analyze_checkpoint

    run = RunDirectory(args.run)
    checkpoint = run.resolve_checkpoint(args.checkpoint)
    output = run.analysis / _checkpoint_output_name(checkpoint)
    report = analyze_checkpoint(
        checkpoint,
        output,
        seed=args.seed,
        permutations=args.permutations,
        dpi=args.dpi,
    )
    print(f"Wrote {len(report['outputs'])} analyses and {output / 'report.json'}.")
    return 0


def play_command(args: argparse.Namespace) -> int:
    checkpoint = None
    if args.run is not None:
        checkpoint = RunDirectory(args.run).resolve_checkpoint(args.checkpoint)
    run_gui(
        host=args.host,
        port=args.port,
        checkpoint_path=checkpoint,
        device=args.device,
    )
    return 0


def _metric_row(
    trainer,
    summary,
    stats,
    collector,
    total: float,
    elapsed: float,
    eval_reward: float | None,
    eval_bid_hit: float | None,
) -> dict[str, Any]:
    return {
        "iteration": trainer.iteration,
        "optimizer_steps": trainer.optimizer_steps,
        "learning_rate": trainer.optimizer.param_groups[0]["lr"],
        "elapsed_sec": elapsed,
        "total_sec": total,
        "collect_sec": summary.collect_sec,
        "update_sec": stats.update_sec,
        "trees": summary.trees,
        "leaves": summary.leaves,
        "decisions": summary.decisions,
        "policy_rows": stats.policy_rows,
        "branched_rows": stats.branched_rows,
        "unbranched_rows": stats.unbranched_rows,
        "positions": stats.positions,
        "forward_rows": collector.forward_rows,
        "branch_decisions": collector.branch_decisions,
        "bid_hit_rate": summary.bid_hit_rate,
        "reward_self": summary.reward_self,
        "reward_historical": summary.reward_historical,
        "spine_entropy": summary.spine_entropy,
        "loss_policy": stats.loss_policy,
        "loss_value": stats.loss_value,
        "loss_suit": stats.loss_suit,
        "loss_trick": stats.loss_trick,
        "loss_bid_hit": stats.loss_bid_hit,
        "entropy": stats.entropy,
        "policy_kl": stats.policy_kl,
        "policy_kl_p95": stats.policy_kl_p95,
        "policy_kl_p99": stats.policy_kl_p99,
        "policy_kl_max": stats.policy_kl_max,
        "backtracks": stats.backtracks,
        "step_scale": stats.step_scale,
        "rolled_back": int(stats.rolled_back),
        "eval_reward_vs_heuristic": ("" if eval_reward is None else eval_reward),
        "eval_bid_hit": "" if eval_bid_hit is None else eval_bid_hit,
        "peak_cache_rows": collector.peak_cache_rows,
        "cache_rows_allocated": collector.cache_rows_allocated,
        "cache_pressure": collector.peak_cache_rows
        / max(collector.cache_rows_allocated, 1),
        "peak_device_gb": collector.peak_device_bytes / (1024**3),
        "blocked_by_cache": collector.blocked_by_cache,
        "skipped_by_placement": collector.skipped_by_placement,
        "sample_sec": collector.sample_sec,
        "step_sec": collector.step_sec,
        "compact_sec": collector.compact_sec,
        "token_build_sec": collector.token_build_sec,
        "forward_sec": collector.forward_sec,
        "positions_per_sec": stats.positions / max(total, 1e-9),
        "forward_rows_per_sec": collector.forward_rows / max(summary.collect_sec, 1e-9),
    }


def _ensure_metrics_header(path: Path) -> None:
    if path.exists():
        with path.open(newline="") as handle:
            header = tuple(next(csv.reader(handle)))
        if header != METRIC_COLUMNS:
            raise ValueError("Existing metrics.csv has an incompatible header.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=METRIC_COLUMNS).writeheader()


def _append_metric(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        writer.writerow(row)
        handle.flush()


def _recorded_elapsed(path: Path) -> float:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return 0.0
    return float(rows[-1].get("elapsed_sec") or 0.0)


def _checkpoint_output_name(checkpoint: Path) -> str:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    iteration = int(payload.get("iteration", 0))
    return f"iter_{iteration:06d}" if iteration > 0 else checkpoint.stem


def _emit(run: RunDirectory, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rendered = f"[{timestamp}] {message}"
    print(rendered, flush=True)
    with run.train_log.open("a") as handle:
        handle.write(rendered + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
